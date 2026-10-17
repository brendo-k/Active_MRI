"""Conv samplers: the mask is predicted from what has already been observed.

Unlike the LOUPE family these are conditioned on the current k-space and the
previous mask, so the pattern adapts per input and per sequential step.

    SampProb2D                    conv_2d - U-Net over k-space, per-location scores
    KspaceLineConstrainedSampler  conv_1d - MLP over k-space, per-column scores

Both zero out already-acquired locations so no budget is spent re-selecting.
"""
import torch
from torch import nn

from activemri.reconstructors.UNet import LOUPEUNet, upsample_block
from activemri.utils.conditioning import DEFAULT_EMBED_DIM, AnatomyEmbedding, FiLM, broadcast_channels
from activemri.utils.mask_binarization import floored_grad_sigmoid


LOGIT_LIMIT = 6.0
LOGIT_LEAK = 0.01
LOCATION_HEADS = ('none', 'bias', 'mlp')
LOCATION_HEAD_HIDDEN = 256
# Multiplies the per-location bias, like LOUPE's slope: Adam moves each parameter ~lr per step, so this
# lets the bias move as fast in logit space as a slope-5 LOUPE mask does.
LOCATION_BIAS_SCALE = 5.0


def scores_to_prob(logits, grad_floor=0.0, logit_limit=LOGIT_LIMIT, logit_leak=LOGIT_LEAK):
    """Squash raw network scores to a per-location probability in (0, 1).

    The logit is soft-clamped to +/-`logit_limit` (default LOGIT_LIMIT) before the sigmoid. Left
    unbounded, a location pushed far enough negative receives a vanishing
    gradient and can never be selected again, so the sampler ratchets whole
    regions of k-space to zero and never recovers them; outside the clamp the
    `logit_leak` slope (default LOGIT_LEAK) keeps that gradient finite.

    The result is already bounded, so `RescaleProbMap` alone sets the sparsity
    and no max-normalisation is needed. This matches the LOUPE samplers, which
    expose their weights through a plain sigmoid. `grad_floor` additionally
    floors the sigmoid's backward derivative, as in `ProbMask`.
    """
    clamped = logits.clamp(-logit_limit, logit_limit)
    return floored_grad_sigmoid(clamped + logit_leak * (logits - clamped), 1.0, grad_floor)


class LocationHead(nn.Module):
    """Location-specific correction added to the U-Net's score map.

    A convolutional U-Net can only make a score depend on position through the
    input and the zero-padded borders, so it cannot express a per-location
    pattern like LOUPE's. 'bias' adds a learned logit per location (LOUPE's
    parameter, shared by every input); 'mlp' also adds an input-dependent term,
    an MLP over the whole score map through LOCATION_HEAD_HIDDEN units. Both
    start at zero, so training starts from the plain U-Net. The MLP's input is
    layer-normalised, so its first layer sees unit-scale scores however far the
    U-Net's scores drift; without it the two layers' product blows the sampler's
    gradient up within a few hundred steps.
    """

    def __init__(self, shape, kind):
        super().__init__()
        if kind not in LOCATION_HEADS[1:]:
            raise ValueError(f"Unknown location head {kind!r}; expected one of {LOCATION_HEADS[1:]}")
        num_locations = shape[0] * shape[1]
        self.bias = nn.Parameter(torch.zeros(1, 1, *shape))
        self.mlp = None
        if kind == 'mlp':
            output = nn.Linear(LOCATION_HEAD_HIDDEN, num_locations, bias=False)
            nn.init.zeros_(output.weight)
            self.mlp = nn.Sequential(
                nn.Flatten(), nn.LayerNorm(num_locations), nn.Linear(num_locations, LOCATION_HEAD_HIDDEN),
                nn.ReLU(inplace=True), output,
            )

    def forward(self, scores):
        corrected = scores + LOCATION_BIAS_SCALE * self.bias
        if self.mlp is not None:
            corrected = corrected + self.mlp(scores).view_as(scores)
        return corrected


class SampProb2D(nn.Module):
    """2D sampling probability model based on a U-Net.

    Given current recon, k-space observations, and previous mask, generates
    a new sampling probability in 2D. Optionally outputs an uncertainty map.

    Args:
        fixed_input: If True, use a fixed learnable input tensor (debugging).
        with_recon: If True, expect reconstructed image (2ch complex) concatenated
            to the input, increasing in_chans by 2.
        with_gfactor: If True, expect a g-factor map (1ch) concatenated to the
            input, increasing in_chans by 1.
        uncertainty_mode: One of 'none', 'channel', or 'head'.
            - 'none': No uncertainty output.
            - 'channel': Extra output channel on the same UNet (out_chans=2).
            - 'head': Separate decoder head sharing the encoder.
        num_anatomy_classes: If > 0, add a classification head on the bottleneck.
        num_conditioning_classes: If > 0, condition the network on the anatomy by
            appending its learned embedding as constant input channels, so the
            predicted pattern can differ between anatomies.
        anatomy_embed_dim: Width of that embedding.
        chans: Output channels of the U-Net's first convolution; doubles at each depth.
        depth: Number of down-/up-sampling stages in the U-Net.
        grad_floor: Lower bound on the score sigmoid's backward derivative, as a
            fraction of its peak (0 = exact).
        head: Location-specific head on the probability scores, one of LOCATION_HEADS
            (see LocationHead).
        logit_limit, logit_leak: The score clamp and the slope beyond it, as in `scores_to_prob`.
    """

    def __init__(self, fixed_input=False, with_recon=False, with_gfactor=False,
                 uncertainty_mode='none', num_anatomy_classes=0, shape=(128, 128),
                 num_conditioning_classes=0, anatomy_embed_dim=DEFAULT_EMBED_DIM,
                 chans=64, depth=4, grad_floor=0.0, head='none',
                 logit_limit=LOGIT_LIMIT, logit_leak=LOGIT_LEAK):
        super().__init__()
        self.grad_floor = grad_floor
        self.logit_limit = logit_limit
        self.logit_leak = logit_leak
        self.location_head = LocationHead(shape, head) if head != 'none' else None

        self.bottleneck_chans = chans * 2 ** depth

        self.with_recon = with_recon
        self.with_gfactor = with_gfactor
        self.uncertainty_mode = uncertainty_mode
        self.num_anatomy_classes = num_anatomy_classes
        self.num_conditioning_classes = num_conditioning_classes
        self.shape = shape

        self.anatomy_embedding = (
            AnatomyEmbedding(num_conditioning_classes, anatomy_embed_dim)
            if num_conditioning_classes > 0 else None
        )
        conditioning_chans = anatomy_embed_dim if self.anatomy_embedding is not None else 0

        self.observation_chans = 5 + (2 if with_recon else 0) + (1 if with_gfactor else 0)
        in_chans = self.observation_chans + conditioning_chans
        out_chans = 2 if uncertainty_mode == 'channel' else 1

        self.unet = LOUPEUNet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            depth=depth,
            drop_prob=0,
            upsample_mode='bilinear',
        )

        if uncertainty_mode == 'head':
            self.uncertainty_decoder = nn.ModuleList()
            ch = self.bottleneck_chans
            for _ in range(depth):
                self.uncertainty_decoder.append(upsample_block(ch, ch // 2, drop_prob=0))
                ch //= 2
            self.uncertainty_conv = nn.Conv2d(ch, 1, kernel_size=1)

        if num_anatomy_classes > 0:
            self.anatomy_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.bottleneck_chans, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_anatomy_classes),
            )

        self.fixed_input = fixed_input
        if fixed_input:
            # Only the observation channels are replaced; the anatomy embedding is
            # still appended, so a fixed-input run stays conditioned.
            fixed_input_tensor = torch.randn(size=[1, self.observation_chans, shape[0], shape[1]])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, x, mask, recon=None, gfactor=None, anatomy_labels=None):
        """
        Args:
            x: Observed kspace data, shape NHWC (4 channels: full + masked, real+imag)
            mask: Previous sampling mask, shape N1HW
            recon: Optional reconstructed image, shape [N, 2, H, W] (real+imag)
            gfactor: Optional g-factor map, shape [N, 1, H, W]
            anatomy_labels: Optional long tensor [N] of anatomy indices

        Returns:
            prob_mask: Sampling probability [N, 1, H, W]
            uncertainty: Uncertainty map [N, 1, H, W] or None
            anatomy_logits: [N, num_classes] or None
        """
        x = x.permute(0, -1, 1, 2)  # [N, 4, H, W]

        channels = [x]
        if self.with_recon:
            if recon is None:
                raise ValueError("with_recon=True but no recon provided")
            channels.append(recon)
        if self.with_gfactor:
            if gfactor is None:
                gfactor = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            channels.append(gfactor)
        channels.append(mask)
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]

        observation = self.fixed_input_tensor.expand(batch_size, -1, -1, -1) if self.fixed_input \
            else torch.cat(channels, dim=1)

        if self.anatomy_embedding is not None:
            embedding = self.anatomy_embedding(anatomy_labels, batch_size)
            x = torch.cat([observation, broadcast_channels(embedding, height, width)], dim=1)
        else:
            x = observation

        use_encoder_path = self.uncertainty_mode == 'head' or self.num_anatomy_classes > 0

        if use_encoder_path:
            out, skips, bottleneck = self._forward_encoder(x)
            prob = scores_to_prob(self._located(out[:, :1]), self.grad_floor, self.logit_limit, self.logit_leak)

            if self.uncertainty_mode == 'head':
                unc_x = bottleneck
                for up, skip in zip(self.uncertainty_decoder, reversed(skips)):
                    unc_x = up(unc_x, skip)
                uncertainty = self.uncertainty_conv(unc_x)
            elif self.uncertainty_mode == 'channel':
                uncertainty = out[:, 1:2]
            else:
                uncertainty = None

            if self.num_anatomy_classes > 0:
                anatomy_logits = self.anatomy_head(bottleneck)
                if anatomy_logits.shape[0] != batch_size:
                    anatomy_logits = anatomy_logits.expand(batch_size, -1)
            else:
                anatomy_logits = None
        else:
            out = self.unet(x)

            if self.uncertainty_mode == 'channel':
                prob = scores_to_prob(self._located(out[:, :1]), self.grad_floor, self.logit_limit, self.logit_leak)
                uncertainty = out[:, 1:2]
            else:
                prob = scores_to_prob(self._located(out), self.grad_floor, self.logit_limit, self.logit_leak)
                uncertainty = None

            anatomy_logits = None

        prob = prob * (1 - mask)

        return prob, uncertainty, anatomy_logits

    def _located(self, scores):
        return scores if self.location_head is None else self.location_head(scores)

    def _forward_encoder(self, x):
        """Run the shared encoder and return prob output + intermediate states for the uncertainty head."""
        skips = []
        enc_x = x
        for down in self.unet.down_sample_layers:
            enc_x, skip = down(enc_x)
            skips.append(skip)

        bottleneck = self.unet.bottleneck(enc_x)

        dec_x = bottleneck
        for up, skip in zip(self.unet.up_sample_layers, reversed(skips)):
            dec_x = up(dec_x, skip)

        out = self.unet.final_conv(dec_x)
        out = out + x[:, :self.unet.out_chans]

        return out, skips, bottleneck


class KspaceLineConstrainedSampler(nn.Module):
    """1D sampling probability model: one score per k-space column.

    Flattens the observed k-space plus the previous mask and maps it through an
    MLP to a score per readout line. Architecture follows the evaluator in
    https://arxiv.org/pdf/1902.03051.pdf.

    Args:
        num_lines: Number of k-space columns, i.e. the image side length.
        fixed_input: If True, use a fixed random input tensor (debugging).
        num_conditioning_classes: If > 0, modulate the hidden state with the
            anatomy embedding so the predicted lines can differ between anatomies.
        anatomy_embed_dim: Width of that embedding.
        grad_floor: As in SampProb2D.
    """

    HIDDEN_DIM = 512

    def __init__(self, num_lines, fixed_input=False, num_anatomy_classes=0,
                 num_conditioning_classes=0, anatomy_embed_dim=DEFAULT_EMBED_DIM, grad_floor=0.0):
        super().__init__()
        self.grad_floor = grad_floor

        self.num_anatomy_classes = num_anatomy_classes
        self.num_conditioning_classes = num_conditioning_classes
        self.flatten_size = num_lines ** 2 * 5

        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_size, self.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Linear(self.HIDDEN_DIM, num_lines)

        if num_conditioning_classes > 0:
            self.anatomy_embedding = AnatomyEmbedding(num_conditioning_classes, anatomy_embed_dim)
            self.anatomy_film = FiLM(anatomy_embed_dim, self.HIDDEN_DIM)
        else:
            self.anatomy_embedding = None

        if num_anatomy_classes > 0:
            self.anatomy_head = nn.Sequential(
                nn.Linear(self.HIDDEN_DIM, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_anatomy_classes),
            )

        self.fixed_input = fixed_input
        if fixed_input:
            fixed_input_tensor = torch.randn(size=[1, 5, num_lines, num_lines])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, kspace, mask, anatomy_labels=None):
        """
        Args:
            kspace: Already sampled measurements, shape NHWC
            mask: Previous sampling trajectories, shape N1HW
            anatomy_labels: Optional long tensor [N] of anatomy indices

        Returns:
            prob_mask: [N, 1, 1, W]
            anatomy_logits: [N, num_classes] or None
        """
        N, _, H, _ = mask.shape

        feat_map = torch.cat([kspace.permute(0, 3, 1, 2), mask], dim=1)
        if self.fixed_input:
            feat_map = self.fixed_input_tensor.repeat(N, 1, 1, 1).to(kspace.device)

        hidden = self.encoder(feat_map.flatten(start_dim=1))
        if self.anatomy_embedding is not None:
            hidden = self.anatomy_film(hidden, self.anatomy_embedding(anatomy_labels, hidden.shape[0]))
        out = scores_to_prob(self.mask_head(hidden), self.grad_floor)

        already_sampled = (mask.sum(dim=-2) == H).float().reshape(N, -1)
        new_mask = out * (1 - already_sampled)

        anatomy_logits = self.anatomy_head(hidden) if self.num_anatomy_classes > 0 else None

        return new_mask.reshape(N, 1, 1, -1), anatomy_logits
