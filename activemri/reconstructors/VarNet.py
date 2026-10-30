"""End-to-end variational network reconstructor.

Adapted from the fastMRI VarNet (Sriram et al., 2020). Sensitivity maps are
supplied externally (estimated by the sampler's SensetivityModel_mc) rather than
learned inside this module, and all operations use the repository's complex
tensor + fft conventions to stay consistent with the other reconstructors.
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from fastmri.models import Unet

from activemri.reconstructors.csme import CascadeSensitivityModel
from activemri.utils.complex_conversion import complex_to_real, real_to_complex
from activemri.utils.conditioning import DEFAULT_EMBED_DIM, AnatomyEmbedding, AnatomyPrompt, FiLM, broadcast_channels
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img

BACKBONES = ("unet", "promptmr")
# The U-Net body's autocast dtype; the normalisation around it, the FFTs and data consistency stay float32.
UNET_PRECISIONS = {"fp32": None, "bf16": torch.bfloat16}

# Side of the stored prompt maps at the coarsest decoder level, doubling per level (8 ... 64 for 4 pools).
PROMPT_BASE_SIZE = 8


class FastMRIUnet(Unet):
    """fastMRI's U-Net (same modules, parameter names and outputs) with two speed fixes.

    Its LeakyReLUs are out of place: in place on InstanceNorm's reshaped output, autograd copies each activation back
    through the view in the backward pass (out of place keeps one more activation per conv). Its forward checks the
    decoder's odd-size padding with Python ints; fastMRI's goes through a tensor, a torch.compile graph break at every
    decoder level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.modules():
            if isinstance(layer, nn.LeakyReLU):
                layer.inplace = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        output = self.conv(output)
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            # fastMRI's reflect pad on the right / bottom for odd input sizes
            padding = [0, int(output.shape[-1] != downsample_layer.shape[-1]),
                       0, int(output.shape[-2] != downsample_layer.shape[-2])]
            if any(padding):
                output = F.pad(output, padding, "reflect")
            output = conv(torch.cat([output, downsample_layer], dim=1))
        return output


class ChannelAttentionBlock(nn.Module):
    """PromptMR+'s CAB: two 3x3 convs whose output is reweighted per channel (squeeze-excitation) and added back."""

    def __init__(self, chans: int, reduction: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(chans, chans, kernel_size=3, padding=1, bias=False),
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chans, chans // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(chans // reduction, chans, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        return x + residual * self.attention(residual)


class MomentumLayer(nn.Module):
    """PromptMR+'s momentum layer (eq. 10): fuse a decoder level's input with the same level's inputs from the
    previous `history` cascades, by a 1x1 conv and a CAB.

    The history is newest first. The first cascade has no past, so every slot holds its own features.
    """

    def __init__(self, chans: int, history: int):
        super().__init__()
        self.history = history
        self.fuse = nn.Sequential(
            nn.Conv2d(chans * (history + 1), chans, kernel_size=1, bias=False),
            ChannelAttentionBlock(chans),
        )

    def forward(self, features: torch.Tensor, history: torch.Tensor = None):
        """Returns the fused features and the history the next cascade receives."""
        if history is None:
            history = features.repeat(1, self.history, 1, 1)
        fused = self.fuse(torch.cat([features, history], dim=1))
        return fused, torch.cat([features, history[:, :-features.shape[1]]], dim=1)


class FiLMBottleneck(nn.Module):
    """U-Net bottleneck block followed by FiLM from `self.embedding`, set before each forward."""

    def __init__(self, block: nn.Module, num_features: int, embed_dim: int):
        super().__init__()
        self.block = block
        self.film = FiLM(embed_dim, num_features)
        self.embedding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.film(self.block(x), self.embedding)


class PromptUnet(nn.Module):
    """PromptMR+'s U-Net: CAB stacks instead of plain convs, strided-conv downsampling, a CAB on every skip,
    bilinear upsampling, and at every decoder level a learned prompt (selected from the pooled features, plus the
    anatomy embedding when there is one) fused with the level's input by CABs. It has three levels, as in PromptMR+.

    Widths scale with `chans` (the first level); `chans=48` gives PromptMR+'s [48, 72, 96, 120] with prompt widths
    [24, 48, 72]. The momentum layer, when `momentum_history > 0`, acts on each decoder level's input before the
    prompt is fused; the prompt and the stored history come from the input before the momentum layer.

    Args:
        film_embed_dim: If > 0, FiLM from the embedding on the bottleneck features.
        prompt_embed_dim: If > 0, the embedding also selects the prompt, in addition to the pooled features.
    """

    LEVELS = 3
    ENCODER_CABS = (2, 3, 3)
    # Per decoder level, coarsest first.
    DECODER_CABS = (3, 2, 2)
    BOTTLENECK_CABS = 3
    PROMPT_LENGTH = 5
    # Stored prompt side per decoder level, coarsest first.
    PROMPT_SIZES = (16, 32, 64)

    def __init__(self, in_chans: int, out_chans: int, chans: int, momentum_history: int = 0,
                 film_embed_dim: int = 0, prompt_embed_dim: int = 0):
        super().__init__()
        widths = [chans, chans * 3 // 2, chans * 2, chans * 5 // 2]
        prompt_widths = [chans // 2, chans, chans * 3 // 2]
        self.extract = nn.Conv2d(in_chans, widths[0], kernel_size=3, padding=1, bias=False)
        self.encoder = nn.ModuleList([
            nn.Sequential(*[ChannelAttentionBlock(widths[level]) for _ in range(self.ENCODER_CABS[level])])
            for level in range(self.LEVELS)
        ])
        self.down = nn.ModuleList([
            nn.Conv2d(widths[level], widths[level + 1], kernel_size=3, stride=2, padding=1)
            for level in range(self.LEVELS)
        ])
        self.skip = nn.ModuleList([ChannelAttentionBlock(widths[level]) for level in range(self.LEVELS)])
        self.bottleneck = nn.Sequential(*[ChannelAttentionBlock(widths[-1]) for _ in range(self.BOTTLENECK_CABS)])
        self.film = FiLM(film_embed_dim, widths[-1]) if film_embed_dim > 0 else None

        # Decoder levels are indexed coarsest first, so level k pairs with encoder level LEVELS - 1 - k.
        decoder_inputs = widths[::-1][:self.LEVELS]
        decoder_outputs = widths[::-1][1:]
        decoder_prompts = prompt_widths[::-1]
        self.momentum = (
            nn.ModuleList([MomentumLayer(c, momentum_history) for c in decoder_inputs]) if momentum_history > 0 else None
        )
        self.prompts = nn.ModuleList([
            AnatomyPrompt(c, prompt_embed_dim, p, prompt_len=self.PROMPT_LENGTH, prompt_size=size)
            for c, p, size in zip(decoder_inputs, decoder_prompts, self.PROMPT_SIZES)
        ])
        self.fuse = nn.ModuleList([
            nn.Sequential(*[ChannelAttentionBlock(c + p) for _ in range(self.DECODER_CABS[level])])
            for level, (c, p) in enumerate(zip(decoder_inputs, decoder_prompts))
        ])
        self.reduce = nn.ModuleList([
            nn.Conv2d(c + p, c, kernel_size=1, bias=False) for c, p in zip(decoder_inputs, decoder_prompts)
        ])
        self.up = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                          nn.Conv2d(c, o, kernel_size=1, bias=False))
            for c, o in zip(decoder_inputs, decoder_outputs)
        ])
        self.refine = nn.ModuleList([ChannelAttentionBlock(o) for o in decoder_outputs])
        self.last = nn.Conv2d(widths[0], out_chans, kernel_size=5, padding=2, bias=False)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor = None, history=None):
        """Returns the output image and the decoder history for the next cascade (empty without momentum)."""
        x = self.extract(x)
        skips = []
        for encoder, down in zip(self.encoder, self.down):
            features = encoder(x)
            skips.append(features)
            x = down(features)
        x = self.bottleneck(x)
        if self.film is not None:
            x = self.film(x, embedding)

        next_history = []
        for level in range(self.LEVELS):
            level_input = x
            if self.momentum is not None:
                x, level_history = self.momentum[level](level_input, history[level] if history else None)
                next_history.append(level_history)
            prompt = self.prompts[level](level_input, embedding)
            x = self.reduce[level](self.fuse[level](torch.cat([x, prompt], dim=1)))
            x = self.refine[level](self.up[level](x) + self.skip[self.LEVELS - 1 - level](skips[self.LEVELS - 1 - level]))
        return self.last(x), next_history


class NormUnet(nn.Module):
    """U-Net with input normalization operating on complex images.

    The complex input [B, C, H, W] is mapped to a 2C-channel real tensor,
    group-normalized (real and imaginary parts separately), padded to a
    multiple of 16, passed through the U-Net, and then mapped back to complex.

    When `film_embed_dim > 0` the embedding instead modulates the U-Net bottleneck
    features through FiLM.

    When `conditioning_chans > 0` an anatomy embedding is appended as constant
    extra channels after normalization, so the regularizer can learn priors that
    differ between anatomies.

    The input of every decoder level can also pass through PromptMR+'s momentum layer
    (`momentum_history > 0`, fused with the same level's inputs from earlier cascades) and
    take an :class:`AnatomyPrompt` (`prompt_embed_dim > 0`), fused by a 1x1 conv. PromptMR+
    fuses its prompts with CABs; the 1x1 conv keeps the bottleneck's 288 + 144 channels cheap.

    The `unet` backbone is fastMRI's U-Net (:class:`FastMRIUnet`). `precision='bf16'` autocasts the U-Net body.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        conditioning_chans: int = 0,
        film_embed_dim: int = 0,
        momentum_history: int = 0,
        prompt_embed_dim: int = 0,
        backbone: str = "unet",
        precision: str = "fp32",
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
        if precision not in UNET_PRECISIONS:
            raise ValueError(f"precision must be one of {tuple(UNET_PRECISIONS)}, got {precision!r}")
        self.autocast_dtype = UNET_PRECISIONS[precision]
        self.conditioning_chans = conditioning_chans
        self.momentum_history = momentum_history
        self.promptmr_backbone = backbone == "promptmr"
        if self.promptmr_backbone:
            if num_pools != PromptUnet.LEVELS:
                raise ValueError(f"the promptmr backbone has {PromptUnet.LEVELS} levels, got num_pools={num_pools}")
            self.unet = PromptUnet(in_chans + conditioning_chans, out_chans, chans, momentum_history,
                                   film_embed_dim, prompt_embed_dim)
            self.film = self.momentum = self.prompts = None
            return
        self.unet = FastMRIUnet(
            in_chans=in_chans + conditioning_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )
        self.film = None
        if film_embed_dim > 0:
            self.film = FiLMBottleneck(self.unet.conv, chans * 2 ** num_pools, film_embed_dim)
            self.unet.conv = self.film

        if (momentum_history > 0 or prompt_embed_dim > 0) and num_pools > 4:
            raise ValueError("the decoder-input forward skips fastMRI's odd-size padding, so it needs the input's "
                             f"multiple-of-16 padding to cover every pool; got num_pools={num_pools}")
        decoder_chans = [chans * 2 ** (num_pools - level) for level in range(num_pools)]
        self.momentum = (
            nn.ModuleList([MomentumLayer(c, momentum_history) for c in decoder_chans])
            if momentum_history > 0 else None
        )
        self.prompts = self.prompt_fuse = None
        if prompt_embed_dim > 0:
            self.prompts = nn.ModuleList([
                AnatomyPrompt(c, prompt_embed_dim, c // 2, prompt_size=PROMPT_BASE_SIZE * 2 ** level)
                for level, c in enumerate(decoder_chans)
            ])
            self.prompt_fuse = nn.ModuleList([nn.Conv2d(c + c // 2, c, kernel_size=1, bias=False) for c in decoder_chans])

    def norm(self, x):
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1)
        x = x.view(b, 2, c // 2, h, w)
        return ((x - mean) / (std + 1e-12)).view(b, c, h, w), mean, std

    def unnorm(self, x, mean, std):
        b, c, h, w = x.shape
        return (x.view(b, 2, c // 2, h, w) * std + mean).view(b, c, h, w)

    def pad(self, x):
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x, h_pad, w_pad, h_mult, w_mult):
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, anatomy_embedding: torch.Tensor = None, history=None):
        """The U-Net's complex output; with `momentum_history > 0` also the decoder history for the next cascade."""
        x = complex_to_real(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        if self.conditioning_chans > 0:
            if anatomy_embedding is None:
                raise ValueError("NormUnet is anatomy-conditioned but no embedding was provided")
            x = torch.cat(
                [x, broadcast_channels(anatomy_embedding, x.shape[-2], x.shape[-1])], dim=1
            )
        with torch.autocast(x.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
            if self.promptmr_backbone:
                if self.unet.film is not None and anatomy_embedding is None:
                    raise ValueError("NormUnet is FiLM-conditioned but no embedding was provided")
                x, history = self.unet(x, anatomy_embedding, history)
            else:
                if self.film is not None:
                    if anatomy_embedding is None:
                        raise ValueError("NormUnet is FiLM-conditioned but no embedding was provided")
                    self.film.embedding = anatomy_embedding
                if self.prompts is not None and anatomy_embedding is None:
                    raise ValueError("NormUnet is prompt-conditioned but no embedding was provided")
                if self.momentum is None and self.prompts is None:
                    x = self.unet(x)
                else:
                    x, history = self._unet_with_decoder_inputs(x, anatomy_embedding, history)
        x = self.unpad(x.float(), *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = real_to_complex(x)
        return (x, history) if self.momentum_history > 0 else x

    def _unet_with_decoder_inputs(self, x, anatomy_embedding, history):
        """fastMRI's U-Net forward with the momentum layer and the prompt applied to every decoder level's input.

        As in PromptMR+, the prompt and the stored history come from the level's input before the momentum layer.
        """
        stack = []
        for layer in self.unet.down_sample_layers:
            x = layer(x)
            stack.append(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.unet.conv(x)

        next_history = []
        for level, (transpose_conv, conv) in enumerate(zip(self.unet.up_transpose_conv, self.unet.up_conv)):
            level_input = x
            if self.momentum is not None:
                x, level_history = self.momentum[level](level_input, history[level] if history else None)
                next_history.append(level_history)
            if self.prompts is not None:
                prompt = self.prompts[level](level_input, anatomy_embedding)
                x = self.prompt_fuse[level](torch.cat([x, prompt], dim=1))
            x = conv(torch.cat([transpose_conv(x), stack.pop()], dim=1))
        return x, next_history


DC_MODES = ("where", "multiply")
# PromptMR+'s adaptive input: the current image plus A^H A x, a latent carried between cascades and
# A^H y. PromptMR+'s code also passes A^H A x - A^H y, which the first (linear) conv can form from the
# two, so it is left out. The U-Net returns its correction and the next latent.
ADAPTIVE_INPUT_IMAGES = 4
ADAPTIVE_OUTPUT_IMAGES = 2


class VarNetBlock(nn.Module):
    """Single cascade: soft data consistency plus a U-Net regularizer.

    ``dc_mode='where'`` gates the DC residual with ``torch.where``, so no gradient reaches
    the mask through the DC term. ``dc_mode='multiply'`` multiplies by the mask instead,
    which is identical in the forward pass for a binary mask but gives a learnable mask
    the gradient ``g * (current - ref)`` at unobserved locations (``ref`` must then be the
    full, unmasked k-space for that signal to be meaningful).

    ``dc_weight_shape`` replaces the scalar DC weight with a learned weight per k-space
    location (NV-Raw2Insights-MRI's weighted DC), so the cascade can trust measured samples
    differently across k-space. ``adaptive_input`` conditions the U-Net on A^H A x and A^H y
    as in PromptMR+, so it sees where its estimate disagrees with the data instead of having
    to infer the sampling pattern from the image's aliasing.

    A cascade with ``adaptive_input`` or a momentum U-Net carries state to the next one: it takes
    and returns the latent and the decoder history (None where unused).

    ``csme`` gives the cascade its own :class:`CascadeSensitivityModel`, which estimates the maps from the
    cascade's current per-coil image. The first cascade sees the zero-filled coil images. The maps passed in are
    then ignored, and the cascade returns the ones it used.
    """

    def __init__(self, model: nn.Module, dc_mode: str = "where", dc_weight_shape=None, adaptive_input: bool = False,
                 csme: bool = False):
        super().__init__()
        if dc_mode not in DC_MODES:
            raise ValueError(f"dc_mode must be one of {DC_MODES}, got {dc_mode!r}")
        self.model = model
        self.dc_mode = dc_mode
        self.adaptive_input = adaptive_input
        self.has_momentum = getattr(model, "momentum_history", 0) > 0
        self.sense_model = CascadeSensitivityModel() if csme else None
        self.carries_state = adaptive_input or self.has_momentum or csme
        self.dc_weight = nn.Parameter(torch.ones(1, 1, *dc_weight_shape) if dc_weight_shape else torch.ones(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before dc_weight became a raw weight hold a scalar logit.
        key = prefix + "dc_weight"
        if key in state_dict and state_dict[key].dim() == 0 and self.dc_weight.dim() == 1:
            state_dict[key] = torch.sigmoid(state_dict[key]).reshape(1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def estimate_maps(self, coil_images: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self.sense_model, coil_images, use_reentrant=False)
        return self.sense_model(coil_images)

    def sens_expand(self, image: torch.Tensor, sense_maps: torch.Tensor) -> torch.Tensor:
        return fft_2d_img(image * sense_maps)

    def sens_reduce(self, kspace: torch.Tensor, sense_maps: torch.Tensor) -> torch.Tensor:
        return (ifft_2d_img(kspace) * sense_maps.conj()).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sense_maps: torch.Tensor,
        anatomy_embedding: torch.Tensor = None,
        kspace_support: torch.Tensor = None,
        zero_filled: torch.Tensor = None,
        latent: torch.Tensor = None,
        history=None,
    ):
        """Returns the next k-space estimate; a cascade that carries state or estimates sense maps returns the
        tuple (k-space, next latent, next history, sense maps used), with None where unused.

        With CSME the maps are estimated from the current per-coil image, the inverse FFT of ``current_kspace``,
        unless ``sense_maps`` is given, which lets the caller reuse an estimate it has already made from the same image.
        """
        if self.sense_model is not None and sense_maps is None:
            sense_maps = self.estimate_maps(ifft_2d_img(current_kspace))
        if self.dc_mode == "multiply":
            soft_dc = mask * (current_kspace - ref_kspace) * self.dc_weight
        else:
            zero = torch.zeros(1, device=current_kspace.device, dtype=current_kspace.dtype)
            soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        image = self.sens_reduce(current_kspace, sense_maps)
        if self.adaptive_input:
            normal = self.sens_reduce(mask * current_kspace, sense_maps)
            image = torch.cat([image, normal, latent, zero_filled], dim=1)
        if self.has_momentum:
            output, history = self.model(image, anatomy_embedding, history)
        else:
            output = self.model(image, anatomy_embedding)
        if self.adaptive_input:
            correction, latent = output.chunk(ADAPTIVE_OUTPUT_IMAGES, dim=1)
        else:
            correction = output
        model_term = self.sens_expand(correction, sense_maps)
        if kspace_support is not None:
            model_term = model_term * kspace_support
        next_kspace = current_kspace - soft_dc - model_term
        return (next_kspace, latent, history, sense_maps) if self.carries_state else next_kspace


class VarNet(nn.Module):
    """Cascaded variational network reconstructor.

    Shares the k-space reconstructor interface of :class:`CGSENSEReconstructor`:
    takes masked multi-coil k-space, externally estimated sensitivity maps, and
    the sampling mask, and returns a coil-combined complex image.

    Args:
        num_conditioning_classes: If > 0, every cascade is conditioned on a
            learned per-class embedding shared across cascades.
        film: Inject the embedding by FiLM on each cascade's bottleneck instead
            of as constant input channels.
        prompt: Inject the embedding through an :class:`AnatomyPrompt` at every decoder level instead
            of as constant input channels.
        label_blind: Build the conditioning but feed every sample class 0: the anatomy-blind control of a
            conditioned model, with the same parameters and initialisation.
        dc_mode: ``'where'`` or ``'multiply'``; see :class:`VarNetBlock`.
            ``'multiply'`` requires ``full_kspace`` in :meth:`forward`.
        dc_weight_shape: k-space (H, W) of a learned per-location DC weight in every cascade;
            None keeps the scalar weight. See :class:`VarNetBlock`.
        adaptive_input: Condition every cascade's U-Net on A^H A x and A^H y (PromptMR+).
        momentum_history: If > 0, PromptMR+'s momentum layer at every decoder level, over this many
            earlier cascades (PromptMR+ uses num_cascades - 1).
        backbone: ``'unet'`` (fastMRI's U-Net) or ``'promptmr'`` (:class:`PromptUnet`, three levels; `chans=48` and
            `pools=3` give PromptMR+'s widths).
        csme: Every cascade estimates its sensitivity maps from its current per-coil image
            (:class:`CascadeSensitivityModel`). The first cascade sees the zero-filled coil images, and
            `sense_maps` is ignored. The returned image is combined with the last cascade's maps.
        checkpoint_cascades: Recompute each cascade in the backward pass instead of storing its
            activations, trading compute for memory while training.
        unet_precision: ``'fp32'`` or ``'bf16'``, the autocast dtype of every cascade's U-Net body.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        num_conditioning_classes: int = 0,
        anatomy_embed_dim: int = DEFAULT_EMBED_DIM,
        film: bool = False,
        prompt: bool = False,
        label_blind: bool = False,
        dc_mode: str = "where",
        dc_weight_shape=None,
        adaptive_input: bool = False,
        momentum_history: int = 0,
        checkpoint_cascades: bool = False,
        backbone: str = "unet",
        csme: bool = False,
        unet_precision: str = "fp32",
    ):
        super().__init__()
        if film and prompt:
            raise ValueError("choose one injection of the anatomy embedding: film or prompt")
        self.dc_mode = dc_mode
        self.adaptive_input = adaptive_input
        self.csme = csme
        self.checkpoint_cascades = checkpoint_cascades
        self.label_blind = label_blind
        self.anatomy_embedding = (
            AnatomyEmbedding(num_conditioning_classes, anatomy_embed_dim)
            if num_conditioning_classes > 0 else None
        )
        if label_blind and self.anatomy_embedding is None:
            raise ValueError("label_blind needs a conditioned model (num_conditioning_classes > 0)")
        has_embedding = self.anatomy_embedding is not None
        conditioning_chans = anatomy_embed_dim if has_embedding and not (film or prompt) else 0
        film_embed_dim = anatomy_embed_dim if has_embedding and film else 0
        prompt_embed_dim = anatomy_embed_dim if has_embedding and prompt else 0
        self.requires_labels = prompt_embed_dim > 0 and not label_blind
        unet_images = (ADAPTIVE_INPUT_IMAGES, ADAPTIVE_OUTPUT_IMAGES) if adaptive_input else (1, 1)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, in_chans=2 * unet_images[0], out_chans=2 * unet_images[1],
                                  conditioning_chans=conditioning_chans, film_embed_dim=film_embed_dim,
                                  momentum_history=momentum_history, prompt_embed_dim=prompt_embed_dim,
                                  backbone=backbone, precision=unet_precision),
                         dc_mode=dc_mode, dc_weight_shape=dc_weight_shape, adaptive_input=adaptive_input,
                         csme=csme)
             for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sense_maps: torch.Tensor,
        mask: torch.Tensor,
        anatomy_labels: torch.Tensor = None,
        full_kspace: torch.Tensor = None,
        kspace_support: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            masked_kspace: [B, C, H, W] complex.
            sense_maps: [B, C, H, W] complex.
            mask: [B, 1, H, W] binary float.
            anatomy_labels: Optional long tensor [B] of anatomy indices.
            full_kspace: [B, C, H, W] complex unmasked k-space; the DC reference
                in ``'multiply'`` mode, ignored in ``'where'`` mode.
            kspace_support: Optional [B, 1, H, W] binary k-space region the regularizer may write to.
                Without it the U-Net can put energy where nothing was acquired (the empty partial-Fourier
                band) and no loss on measured data will ever see it.

        Returns:
            Coil-combined reconstruction [B, 1, H, W] complex.
        """
        batch_size = masked_kspace.shape[0]
        if self.label_blind:
            anatomy_labels = torch.zeros(batch_size, dtype=torch.long, device=masked_kspace.device)
        elif self.requires_labels and anatomy_labels is None:
            raise ValueError("this VarNet is prompt-conditioned on the anatomy, so it needs anatomy_labels; "
                             "the mean-embedding fallback would silently score an unconditioned model")
        anatomy_embedding = (
            self.anatomy_embedding(anatomy_labels, batch_size)
            if self.anatomy_embedding is not None else None
        )

        if self.dc_mode == "multiply":
            if full_kspace is None:
                raise ValueError("dc_mode='multiply' needs full_kspace as the DC reference")
            ref_kspace = full_kspace
        else:
            ref_kspace = masked_kspace

        def run(cascade, *inputs):
            if self.checkpoint_cascades and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(cascade, *inputs, use_reentrant=False)
            return cascade(*inputs)

        kspace_pred = masked_kspace.clone()
        if self.csme and self.adaptive_input:
            # Cascade 0 estimates from the same zero-filled coil images as the adaptive input, so the estimate is shared.
            sense_maps = self.cascades[0].estimate_maps(ifft_2d_img(masked_kspace))
        elif self.csme:
            sense_maps = None
        zero_filled = latent = history = None
        if self.adaptive_input:
            zero_filled = (ifft_2d_img(masked_kspace) * sense_maps.conj()).sum(dim=1, keepdim=True)
            latent = zero_filled
        for cascade in self.cascades:
            output = run(cascade, kspace_pred, ref_kspace, mask, sense_maps, anatomy_embedding, kspace_support,
                         zero_filled, latent, history)
            if cascade.carries_state:
                kspace_pred, latent, history, used_maps = output
            else:
                kspace_pred, used_maps = output, sense_maps
            if self.csme:
                sense_maps = None
        return (ifft_2d_img(kspace_pred) * used_maps.conj()).sum(dim=1, keepdim=True)
