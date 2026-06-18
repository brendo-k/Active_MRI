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
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img


class NormUnet(nn.Module):
    """U-Net with input normalization operating on complex images.

    The complex input [B, C, H, W] is mapped to a 2C-channel real tensor,
    group-normalized (real and imaginary parts separately), padded to a
    multiple of 16, passed through the U-Net, and then mapped back to complex.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = complex_to_real(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return real_to_complex(x)


class VarNetBlock(nn.Module):
    """Single cascade: soft data consistency plus a U-Net regularizer."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

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
    ) -> torch.Tensor:
        zero = torch.zeros(1, device=current_kspace.device, dtype=current_kspace.dtype)
        soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sense_maps)), sense_maps
        )
        return current_kspace - soft_dc - model_term


class VarNet(nn.Module):
    """Cascaded variational network reconstructor.

    Shares the k-space reconstructor interface of :class:`CGSENSEReconstructor`:
    takes masked multi-coil k-space, externally estimated sensitivity maps, and
    the sampling mask, and returns a coil-combined complex image.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sense_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            masked_kspace: [B, C, H, W] complex.
            sense_maps: [B, C, H, W] complex.
            mask: [B, 1, H, W] binary float.

        Returns:
            Coil-combined reconstruction [B, 1, H, W] complex.
        """
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sense_maps)
        return (ifft_2d_img(kspace_pred) * sense_maps.conj()).sum(dim=1, keepdim=True)
