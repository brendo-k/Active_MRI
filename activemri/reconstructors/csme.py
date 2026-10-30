"""Cascade-wise coil sensitivity map estimation (CSME).

Each cascade estimates its own sensitivity maps from that cascade's current per-coil image,
with a U-Net shared across coils. The first cascade sees the zero-filled coil images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.basic_unet import BasicUNet

# SDUM's CoilSensitivityModel_DCAE widths; the last level returns to the first width, as BasicUNet expects.
SDUM_FEATURES = (12, 24, 48, 96, 192)
PAD_MULTIPLE = 2 ** (len(SDUM_FEATURES) - 1)


class CascadeSensitivityModel(nn.Module):
    """A U-Net on each coil's current image, with maps normalised by their RSS over coils."""

    def __init__(self, features=SDUM_FEATURES):
        super().__init__()
        self.unet = BasicUNet(
            spatial_dims=2, in_channels=2, out_channels=2, features=[*features, features[0]],
            act=("LeakyReLU", {"negative_slope": 0.01, "inplace": True}), norm=("instance", {"affine": True}),
            bias=True, dropout=0.0, upsample="deconv",
        )

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coil_images: [B, C, H, W] complex. The cascade's current per-coil image; the first cascade receives
                the zero-filled coil images.

        Returns:
            [B, C, H, W] complex sensitivity maps.
        """
        batch, coils, height, width = coil_images.shape
        images = coil_images.reshape(batch * coils, 1, height, width)
        x = torch.view_as_real(images).squeeze(1).permute(0, 3, 1, 2)
        pad_h, pad_w = -height % PAD_MULTIPLE, -width % PAD_MULTIPLE
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        x = self.unet(F.pad(x, padding))
        x = x[..., padding[2]:padding[2] + height, padding[0]:padding[0] + width]
        maps = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()).reshape(batch, coils, height, width)
        return maps / maps.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
