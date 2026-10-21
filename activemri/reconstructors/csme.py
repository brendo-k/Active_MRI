"""Cascade-wise coil sensitivity map estimation (CSME), as in SDUM (NV-Raw2Insights-MRI).

Every cascade estimates its own sensitivity maps from the fully-sampled centre of its current k-space estimate,
with a U-Net shared across coils, instead of all cascades using one set of externally estimated maps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.basic_unet import BasicUNet

from activemri.utils.image_processing import ifft_2d_img

# SDUM's CoilSensitivityModel_DCAE widths; the last level returns to the first width, as BasicUNet expects.
SDUM_FEATURES = (12, 24, 48, 96, 192)
PAD_MULTIPLE = 2 ** (len(SDUM_FEATURES) - 1)


def acs_region_mask(mask: torch.Tensor) -> torch.Tensor:
    """The largest rectangle of acquired samples around the k-space centre, grown as SDUM's ``get_acs_region``.

    Args:
        mask: [B, 1, H, W] binary sampling mask.

    Returns:
        [B, 1, H, W] float mask of the rectangle, per sample.
    """
    acquired = (mask.detach() > 0).squeeze(1).cpu()
    _, height, width = acquired.shape
    centre_row, centre_col = height // 2, width // 2
    region = torch.zeros_like(acquired)
    for sample, rows_by_cols in enumerate(acquired):
        if not rows_by_cols[centre_row, centre_col]:
            raise ValueError("the sampling mask does not acquire the k-space centre, so there is no ACS region to "
                             "estimate sensitivity maps from")
        top = bottom = centre_row
        left = right = centre_col
        grew = True
        while grew:
            grew = False
            if left > 0 and rows_by_cols[top:bottom + 1, left - 1].all():
                left, grew = left - 1, True
            if right < width - 1 and rows_by_cols[top:bottom + 1, right + 1].all():
                right, grew = right + 1, True
            if top > 0 and rows_by_cols[top - 1, left:right + 1].all():
                top, grew = top - 1, True
            if bottom < height - 1 and rows_by_cols[bottom + 1, left:right + 1].all():
                bottom, grew = bottom + 1, True
        region[sample, top:bottom + 1, left:right + 1] = True
    return region.unsqueeze(1).to(device=mask.device, dtype=torch.float32)


class CascadeSensitivityModel(nn.Module):
    """SDUM's CoilSensitivityModel_DCAE: a U-Net on each coil's ACS image, maps normalised by their RSS over coils."""

    def __init__(self, features=SDUM_FEATURES):
        super().__init__()
        self.unet = BasicUNet(
            spatial_dims=2, in_channels=2, out_channels=2, features=[*features, features[0]],
            act=("LeakyReLU", {"negative_slope": 0.01, "inplace": True}), norm=("instance", {"affine": True}),
            bias=True, dropout=0.0, upsample="deconv",
        )

    def forward(self, kspace: torch.Tensor, acs_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kspace: [B, C, H, W] complex current k-space estimate.
            acs_mask: [B, 1, H, W] region of ``kspace`` to estimate from, see :func:`acs_region_mask`.

        Returns:
            [B, C, H, W] complex sensitivity maps.
        """
        batch, coils, height, width = kspace.shape
        images = ifft_2d_img(kspace * acs_mask).reshape(batch * coils, 1, height, width)
        x = torch.view_as_real(images).squeeze(1).permute(0, 3, 1, 2)
        pad_h, pad_w = -height % PAD_MULTIPLE, -width % PAD_MULTIPLE
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        x = self.unet(F.pad(x, padding))
        x = x[..., padding[2]:padding[2] + height, padding[0]:padding[0] + width]
        maps = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()).reshape(batch, coils, height, width)
        return maps / maps.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
