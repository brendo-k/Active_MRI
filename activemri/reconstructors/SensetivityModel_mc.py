import einops
import torch
import math

import torch.nn as nn
import torch.nn.functional as F
from activemri.utils.image_processing import ifft_2d_img, root_sum_of_squares
from activemri.reconstructors.unet_backbone import Unet
from activemri.reconstructors.VarNet import UNET_PRECISIONS
from activemri.utils.complex_conversion import real_to_complex, complex_to_real

from typing import Optional, Tuple, Literal

COIL_BUCKET = 16


def upsample_centred(images: torch.Tensor, size) -> torch.Tensor:
    """Bilinear upsampling of fftshift-centred images [N, C, h, w] to `size`, keeping the centre pixel (h // 2, w // 2)
    on the output's centre pixel, as a centre crop of k-space places it (F.interpolate would shift it by a fraction
    of a pixel)."""
    (h, w), (out_h, out_w) = images.shape[-2:], size
    ys = (torch.arange(out_h, device=images.device) - out_h // 2) * (h / out_h) + h // 2
    xs = (torch.arange(out_w, device=images.device) - out_w // 2) * (w / out_w) + w // 2
    grid = torch.stack(torch.meshgrid(2 * xs / (w - 1) - 1, 2 * ys / (h - 1) - 1, indexing='xy'), dim=-1)
    grid = grid.to(images.dtype).expand(images.shape[0], -1, -1, -1)
    return F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=True)


class SensetivityModel_mc(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int,
            conv_after_upsample: bool = False,
            upsample_method: Literal['conv', 'max', 'bilinear'] = 'conv',
            resolution: Optional[int] = None,
            precision: str = 'fp32',
            ):

        """Module used to estimate sensetivity maps based on masked k-space center

        Args:
            in_chans (int): _description_
            out_chans (int): _description_
            chans (int): Number of convolutional channels for U-Net estimator
            conv_after_upsample (bool): boolean flag for addition of 3x3 convolution after upsampling
            upsample_method (str): upsampling method for U-Net ('bilinear', 'max', 'conv')
            resolution (int): run the U-Net on the resolution x resolution centre of k-space, which holds the whole
                masked centre, and upsample its output bilinearly to the full image; None runs it at full resolution
            precision (str): 'fp32' or 'bf16', the autocast dtype of the U-Net body; its input normalisation and
                everything after it stay float32
        """
        super().__init__()
        if precision not in UNET_PRECISIONS:
            raise ValueError(f"precision must be one of {tuple(UNET_PRECISIONS)}, got {precision!r}")
        self.autocast_dtype = UNET_PRECISIONS[precision]
        self.resolution = resolution
        self.model = Unet(
            in_chans,
            out_chans,
            chans=chans,
            conv_after_upsample=conv_after_upsample,
            upsample_method=upsample_method
            )

    def forward(
            self,
            images: torch.Tensor,
            mask_method: Optional[Literal['pre_network', 'post_network']] = None,
            mask_threshold: float = 0.1,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            images: coil k-spaces [B, C, H, W] complex
            mask_method:
                None           – return only sense maps
                'pre_network'  – mask from RSS of IFFT'd center k-space (before U-Net)
                'post_network' – mask from RSS of U-Net output (before RSS normalisation)
            mask_threshold: fraction of per-sample RSS max used as binary threshold

        Returns:
            sense_maps: [B, C, H, W] complex, RSS-normalised. When mask_method is
                set, maps are zeroed outside the signal region (sense_mask).
            sense_mask: [B, 1, H, W] float in {0, 1}, or None when mask_method is None
        """
        if torch.isnan(images).any():
            print(f"SensetivityModel input kspace has NaN")
        kspace = self.mask_center(images)
        images = ifft_2d_img(kspace, axes=[-1, -2])
        if torch.isnan(images).any():
            print(f"SensetivityModel after ifft has NaN")
        assert isinstance(images, torch.Tensor)
        number_of_coils = images.shape[1]
        sense_mask: Optional[torch.Tensor] = None

        if mask_method == 'pre_network':
            rss_zf = root_sum_of_squares(images, coil_dim=1)  # [B, H, W]
            max_val = rss_zf.amax(dim=(-2, -1), keepdim=True)
            sense_mask = (rss_zf > mask_threshold * max_val).float().unsqueeze(1)

        full_size = images.shape[-2:]
        if self.resolution is not None:
            images = ifft_2d_img(self.crop_center(kspace, self.resolution), axes=[-1, -2])

        images = einops.rearrange(images, 'b c h w -> (b c) 1 h w')
        coils = self.coils_to_run(images)
        coil_images = complex_to_real(images.index_select(0, coils))
        coil_images, mean, std = self.norm(coil_images)
        torch._dynamo.maybe_mark_dynamic(coil_images, 0)
        with torch.autocast(coil_images.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
            coil_images = self.model(coil_images)
        coil_images = self.unnorm(coil_images.float(), mean, std)
        if self.resolution is not None:
            coil_images = upsample_centred(coil_images, full_size)
        coil_images = real_to_complex(coil_images)
        images = coil_images.new_zeros(images.shape[0], *coil_images.shape[1:]).index_copy(0, coils, coil_images)
        images = einops.rearrange(images, '(b c) 1 h w -> b c h w', c=number_of_coils)


        rss_norm = root_sum_of_squares(images, coil_dim=1).unsqueeze(1)  # [B, 1, H, W]

        if mask_method == 'post_network':
            max_val = rss_norm.amax(dim=(-2, -1), keepdim=True)
            sense_mask = (rss_norm > mask_threshold * max_val).float()

        sense_maps = images / rss_norm.clamp(min=1e-8)

        if mask_method is None:
            return sense_maps, None

        sense_maps = sense_maps * sense_mask
        return sense_maps, sense_mask

    @staticmethod
    def coils_to_run(coil_images: torch.Tensor) -> torch.Tensor:
        """Indices into the [(B C), 1, h, w] coil images the U-Net has to see: every coil with a nonzero pixel, topped
        up with all-zero coils to a multiple of COIL_BUCKET (at most all of them). An all-zero coil (one the loader
        padded) has instance-norm mean and std 0, so unnorm maps whatever the U-Net makes of it to exactly 0, with zero
        gradient: skipping it is exact. The bucket bounds the distinct batch sizes, each of which costs a cuDNN
        autotune of the backward convolutions (~4 s on an A6000); torch.compile keeps one graph, as forward marks the
        batch dim dynamic."""
        is_live = coil_images.flatten(1).ne(0).any(dim=1)
        num_live = int(is_live.sum())
        num_run = min(-(-num_live // COIL_BUCKET) * COIL_BUCKET, len(is_live))
        return torch.argsort(~is_live, stable=True)[:num_run]

    def mask_center(self, coil_k_spaces):
        mask = torch.zeros_like(coil_k_spaces)
        center_h = coil_k_spaces.shape[-2] // 2
        center_w = coil_k_spaces.shape[-1] // 2
        mask[:, :, center_h-5:center_h+5, center_w-5:center_w+5] = 1

        return coil_k_spaces * mask

    def crop_center(self, coil_k_spaces, size):
        top = coil_k_spaces.shape[-2] // 2 - size // 2
        left = coil_k_spaces.shape[-1] // 2 - size // 2
        return coil_k_spaces[..., top:top + size, left:left + size]



    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / (std + 1e-20), mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean