"""End-to-end variational network reconstructor.

Adapted from the fastMRI VarNet (Sriram et al., 2020). Sensitivity maps are
supplied externally (estimated by the sampler's SensetivityModel_mc) rather than
learned inside this module, and all operations use the repository's complex
tensor + fft conventions to stay consistent with the other reconstructors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.models import Unet

from activemri.utils.complex_conversion import complex_to_real, real_to_complex
from activemri.utils.conditioning import DEFAULT_EMBED_DIM, AnatomyEmbedding, FiLM, broadcast_channels
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img


class FiLMBottleneck(nn.Module):
    """U-Net bottleneck block followed by FiLM from `self.embedding`, set before each forward."""

    def __init__(self, block: nn.Module, num_features: int, embed_dim: int):
        super().__init__()
        self.block = block
        self.film = FiLM(embed_dim, num_features)
        self.embedding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.film(self.block(x), self.embedding)


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
    ):
        super().__init__()
        self.conditioning_chans = conditioning_chans
        self.unet = Unet(
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

    def norm(self, x):
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / (std + 1e-12), mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

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

    def forward(self, x: torch.Tensor, anatomy_embedding: torch.Tensor = None) -> torch.Tensor:
        x = complex_to_real(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        if self.conditioning_chans > 0:
            if anatomy_embedding is None:
                raise ValueError("NormUnet is anatomy-conditioned but no embedding was provided")
            x = torch.cat(
                [x, broadcast_channels(anatomy_embedding, x.shape[-2], x.shape[-1])], dim=1
            )
        if self.film is not None:
            if anatomy_embedding is None:
                raise ValueError("NormUnet is FiLM-conditioned but no embedding was provided")
            self.film.embedding = anatomy_embedding
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return real_to_complex(x)


DC_MODES = ("where", "multiply")


class VarNetBlock(nn.Module):
    """Single cascade: soft data consistency plus a U-Net regularizer.

    ``dc_mode='where'`` gates the DC residual with ``torch.where``, so no gradient reaches
    the mask through the DC term. ``dc_mode='multiply'`` multiplies by the mask instead,
    which is identical in the forward pass for a binary mask but gives a learnable mask
    the gradient ``g * (current - ref)`` at unobserved locations (``ref`` must then be the
    full, unmasked k-space for that signal to be meaningful).
    """

    def __init__(self, model: nn.Module, dc_mode: str = "where"):
        super().__init__()
        if dc_mode not in DC_MODES:
            raise ValueError(f"dc_mode must be one of {DC_MODES}, got {dc_mode!r}")
        self.model = model
        self.dc_mode = dc_mode
        self.dc_weight = nn.Parameter(torch.ones(1))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before dc_weight became a raw weight hold a scalar logit.
        key = prefix + "dc_weight"
        if key in state_dict and state_dict[key].dim() == 0:
            state_dict[key] = torch.sigmoid(state_dict[key]).reshape(1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
    ) -> torch.Tensor:
        if self.dc_mode == "multiply":
            soft_dc = mask * (current_kspace - ref_kspace) * self.dc_weight
        else:
            zero = torch.zeros(1, device=current_kspace.device, dtype=current_kspace.dtype)
            soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sense_maps), anatomy_embedding), sense_maps
        )
        return current_kspace - soft_dc - model_term


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
        dc_mode: ``'where'`` or ``'multiply'``; see :class:`VarNetBlock`.
            ``'multiply'`` requires ``full_kspace`` in :meth:`forward`.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        num_conditioning_classes: int = 0,
        anatomy_embed_dim: int = DEFAULT_EMBED_DIM,
        film: bool = False,
        dc_mode: str = "where",
    ):
        super().__init__()
        self.dc_mode = dc_mode
        self.anatomy_embedding = (
            AnatomyEmbedding(num_conditioning_classes, anatomy_embed_dim)
            if num_conditioning_classes > 0 else None
        )
        has_embedding = self.anatomy_embedding is not None
        conditioning_chans = anatomy_embed_dim if has_embedding and not film else 0
        film_embed_dim = anatomy_embed_dim if has_embedding and film else 0
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, conditioning_chans=conditioning_chans, film_embed_dim=film_embed_dim),
                         dc_mode=dc_mode)
             for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sense_maps: torch.Tensor,
        mask: torch.Tensor,
        anatomy_labels: torch.Tensor = None,
        full_kspace: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            masked_kspace: [B, C, H, W] complex.
            sense_maps: [B, C, H, W] complex.
            mask: [B, 1, H, W] binary float.
            anatomy_labels: Optional long tensor [B] of anatomy indices.
            full_kspace: [B, C, H, W] complex unmasked k-space; the DC reference
                in ``'multiply'`` mode, ignored in ``'where'`` mode.

        Returns:
            Coil-combined reconstruction [B, 1, H, W] complex.
        """
        anatomy_embedding = (
            self.anatomy_embedding(anatomy_labels, masked_kspace.shape[0])
            if self.anatomy_embedding is not None else None
        )

        if self.dc_mode == "multiply":
            if full_kspace is None:
                raise ValueError("dc_mode='multiply' needs full_kspace as the DC reference")
            ref_kspace = full_kspace
        else:
            ref_kspace = masked_kspace

        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, ref_kspace, mask, sense_maps, anatomy_embedding)
        return (ifft_2d_img(kspace_pred) * sense_maps.conj()).sum(dim=1, keepdim=True)
