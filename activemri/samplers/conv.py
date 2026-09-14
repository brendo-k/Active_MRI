"""Conv samplers: the mask is predicted from what has already been observed.

Unlike the LOUPE family these are conditioned on the current k-space and the
previous mask, so the pattern adapts per input and per sequential step.

    SampProb2D                    conv_2d - U-Net over k-space, per-location scores
    KspaceLineConstrainedSampler  conv_1d - MLP over k-space, per-column scores

Both zero out already-acquired locations so no budget is spent re-selecting.
"""
import torch
from torch import nn
from torch.nn import functional as F

from activemri.reconstructors.UNet import LOUPEUNet, upsample_block


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
    """

    BOTTLENECK_CHANS = 64 * (2 ** 4)  # 1024

    def __init__(self, fixed_input=False, with_recon=False, with_gfactor=False,
                 uncertainty_mode='none', num_anatomy_classes=0):
        super().__init__()

        self.with_recon = with_recon
        self.with_gfactor = with_gfactor
        self.uncertainty_mode = uncertainty_mode
        self.num_anatomy_classes = num_anatomy_classes
        in_chans = 5 + (2 if with_recon else 0) + (1 if with_gfactor else 0)
        out_chans = 2 if uncertainty_mode == 'channel' else 1

        self.unet = LOUPEUNet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=64,
            depth=4,
            drop_prob=0,
            upsample_mode='bilinear',
        )

        if uncertainty_mode == 'head':
            self.uncertainty_decoder = nn.ModuleList()
            ch = self.BOTTLENECK_CHANS
            for _ in range(4):
                self.uncertainty_decoder.append(upsample_block(ch, ch // 2, drop_prob=0))
                ch //= 2
            self.uncertainty_conv = nn.Conv2d(ch, 1, kernel_size=1)

        if num_anatomy_classes > 0:
            self.anatomy_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.BOTTLENECK_CHANS, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_anatomy_classes),
            )

        self.fixed_input = fixed_input
        if fixed_input:
            fixed_input_tensor = torch.randn(size=[1, in_chans, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, x, mask, recon=None, gfactor=None):
        """
        Args:
            x: Observed kspace data, shape NHWC (4 channels: full + masked, real+imag)
            mask: Previous sampling mask, shape N1HW
            recon: Optional reconstructed image, shape [N, 2, H, W] (real+imag)
            gfactor: Optional g-factor map, shape [N, 1, H, W]

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
        x = torch.cat(channels, dim=1)

        if self.fixed_input:
            x = self.fixed_input_tensor

        use_encoder_path = self.uncertainty_mode == 'head' or self.num_anatomy_classes > 0

        if use_encoder_path:
            out, skips, bottleneck = self._forward_encoder(x)
            prob = F.softplus(out[:, :1])

            if self.uncertainty_mode == 'head':
                unc_x = bottleneck
                for up, skip in zip(self.uncertainty_decoder, reversed(skips)):
                    unc_x = up(unc_x, skip)
                uncertainty = self.uncertainty_conv(unc_x)
            elif self.uncertainty_mode == 'channel':
                uncertainty = out[:, 1:2]
            else:
                uncertainty = None

            anatomy_logits = self.anatomy_head(bottleneck) if self.num_anatomy_classes > 0 else None
        else:
            out = self.unet(x)

            if self.uncertainty_mode == 'channel':
                prob = F.softplus(out[:, :1])
                uncertainty = out[:, 1:2]
            else:
                prob = F.softplus(out)
                uncertainty = None

            anatomy_logits = None

        max_val = prob.flatten(1).max(dim=1)[0].reshape(-1, 1, 1, 1)
        prob = prob / max_val

        prob = prob * (1 - mask)

        return prob, uncertainty, anatomy_logits

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
    """

    HIDDEN_DIM = 512

    def __init__(self, num_lines, fixed_input=False, num_anatomy_classes=0):
        super().__init__()

        self.num_anatomy_classes = num_anatomy_classes
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

    def forward(self, kspace, mask):
        """
        Args:
            kspace: Already sampled measurements, shape NHWC
            mask: Previous sampling trajectories, shape N1HW

        Returns:
            prob_mask: [N, 1, 1, W]
            anatomy_logits: [N, num_classes] or None
        """
        N, _, H, _ = mask.shape

        feat_map = torch.cat([kspace.permute(0, 3, 1, 2), mask], dim=1)
        if self.fixed_input:
            feat_map = self.fixed_input_tensor.repeat(N, 1, 1, 1).to(kspace.device)

        hidden = self.encoder(feat_map.flatten(start_dim=1))
        out = self.mask_head(hidden)
        out = F.softplus(out)
        out = out / out.max(dim=1, keepdim=True)[0]

        already_sampled = (mask.sum(dim=-2) == H).float().reshape(N, -1)
        new_mask = out * (1 - already_sampled)

        anatomy_logits = self.anatomy_head(hidden) if self.num_anatomy_classes > 0 else None

        return new_mask.reshape(N, 1, 1, -1), anatomy_logits
