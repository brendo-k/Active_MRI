import einops
import torch
import math

import torch.nn as nn
from activemri.utils.image_processing import ifft_2d_img, root_sum_of_squares
from activemri.baselines.sequential_sampling_codes.UNet import Unet
from activemri.utils.complex_conversion import real_to_complex, complex_to_real

from typing import Optional, Tuple, Literal

class SensetivityModel_mc(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int,
            conv_after_upsample: bool = False,
            upsample_method: Literal['conv', 'max', 'bilinear'] = 'conv'
            ):

        """Module used to estimate sensetivity maps based on masked k-space center

        Args:
            in_chans (int): _description_
            out_chans (int): _description_
            chans (int): Number of convolutional channels for U-Net estimator
            conv_after_upsample (bool): boolean flag for addition of 3x3 convolution after upsampling
            upsample_method (str): upsampling method for U-Net ('bilinear', 'max', 'conv')
        """
        super().__init__()
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
            sense_maps: [B, C, H, W] complex, RSS-normalised
            sense_mask: [B, 1, H, W] float in {0, 1}, or None when mask_method is None
        """
        if torch.isnan(images).any():
            print(f"SensetivityModel input kspace has NaN")
        images = self.mask_center(images)
        images = ifft_2d_img(images, axes=[-1, -2])
        if torch.isnan(images).any():
            print(f"SensetivityModel after ifft has NaN")
        assert isinstance(images, torch.Tensor)
        number_of_coils = images.shape[1]
        sense_mask: Optional[torch.Tensor] = None

        if mask_method == 'pre_network':
            rss_zf = root_sum_of_squares(images, coil_dim=1)  # [B, H, W]
            max_val = rss_zf.amax(dim=(-2, -1), keepdim=True)
            sense_mask = (rss_zf > mask_threshold * max_val).float().unsqueeze(1)

        images = einops.rearrange(images, 'b c h w -> (b c) 1 h w')
        images = complex_to_real(images)
        images, mean, std = self.norm(images)
        images = self.model(images)
        images = self.unnorm(images, mean, std)
        images = real_to_complex(images)
        images = einops.rearrange(images, '(b c) 1 h w -> b c h w', c=number_of_coils)

        rss_norm = root_sum_of_squares(images, coil_dim=1).unsqueeze(1)  # [B, 1, H, W]

        if mask_method == 'post_network':
            max_val = rss_norm.amax(dim=(-2, -1), keepdim=True)
            sense_mask = (rss_norm > mask_threshold * max_val).float()

        sense_maps = images / rss_norm.clamp(min=1e-8)

        if mask_method is None:
            return sense_maps, None
        return sense_maps, sense_mask

    def mask_center(self, coil_k_spaces):
        mask = torch.zeros_like(coil_k_spaces)
        center_h = coil_k_spaces.shape[-2] // 2
        center_w = coil_k_spaces.shape[-1] // 2
        mask[:, :, center_h-5:center_h+5, center_w-5:center_w+5] = 1

        return coil_k_spaces * mask

        

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