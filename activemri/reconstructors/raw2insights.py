"""Frozen NV-Raw2Insights-MRI (SDUM) reconstructor with the `VarNet` call signature.

`MRIEnv` calls `reconstructor(masked_kspace, sense_maps, mask)` on static
multi-coil slices. The NVIDIA model is a dynamic (time-windowed) model: it
folds coils into a batch, collapses them to one virtual coil with its own
learned sensitivity model, and stacks `num_frames` time frames as channels.
For a static slice this wrapper repeats the slice across the window, ignores
the caller's `sense_maps`, and returns the RSS-combined centre-frame image.

The NVIDIA package uses top-level imports (`models`, `utils`, `transforms`),
so its `scripts/` directory is put on `sys.path` before importing.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from monai.data.fft_utils import ifftn_centered

_ACCELERATIONS = (8, 16, 24)


def _import_nvidia_modules(repo_path: Path):
    scripts_dir = str(repo_path / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from models.latent_recon import create_mri_recon_model
    from monai.networks.utils import copy_model_state
    from utils import complex_zscore, load_config, resolve_checkpoint_path

    return create_mri_recon_model, copy_model_state, complex_zscore, load_config, resolve_checkpoint_path


class Raw2InsightsRecon(nn.Module):
    """
    Args:
        repo_path: Clone of NVIDIA-Medtech/NV-Raw2insights-MRI.
        config_name: Config under `<repo>/configs` (base = 18 cascades).
        checkpoint: Local `.pt`; downloaded from Hugging Face when None.
        mask_type: Conditioning label; must contain one of
            `uniform` / `gaussian` / `radial`.
        acq_type: Conditioning label from the model's `acq_types`
            (BlackBlood, Cine, Flow2d, LGE, Mapping, Perfusion, T1rho, T1w, T2w).
        autocast_dtype: bf16 autocast for the network; data consistency
            inside the model is always fp32.
    """

    def __init__(
        self,
        repo_path: str,
        config_name: str = "nv_raw2insights_mri_base.json",
        checkpoint: str = None,
        mask_type: str = "uniform",
        acq_type: str = "T2w",
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        repo = Path(repo_path)
        create_model, copy_model_state, self._zscore, load_config, resolve_ckpt = _import_nvidia_modules(repo)

        config = load_config(repo / "configs" / config_name)
        checkpoint_path = resolve_ckpt(config.model_variant, checkpoint)
        self.network = create_model(config)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["net_state_dict"]
        _, updated_keys, unchanged_keys = copy_model_state(self.network, state)
        if unchanged_keys:
            raise RuntimeError(f"{len(unchanged_keys)} parameters not found in checkpoint, e.g. {unchanged_keys[:3]}")

        self.num_frames = int(config.num_frames)
        self.mask_type = mask_type
        self.acq_type = acq_type
        self.autocast_dtype = autocast_dtype
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _nearest_acceleration(mask: torch.Tensor) -> int:
        """Snap `width / sampled_columns` to the accelerations the model was conditioned on."""
        sampled_columns = mask[:, 0, 0, :].sum(dim=-1).float().mean().clamp(min=1)
        acceleration = mask.shape[-1] / sampled_columns.item()
        return min(_ACCELERATIONS, key=lambda a: abs(a - acceleration))

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sense_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            masked_kspace: complex `[B, C, H, W]`, DC-centred.
            sense_maps: unused; the network estimates its own from the ACS region.
            mask: `[B, 1, H, W]` column mask, DC-centred; the centre column must be sampled.

        Returns:
            Complex `[B, 1, H, W]` (RSS magnitude, zero imaginary part).
        """
        b, c, h, w = masked_kspace.shape
        acceleration = self._nearest_acceleration(mask)

        image = ifftn_centered(torch.view_as_real(masked_kspace), spatial_dims=2, is_complex=True)
        image, mean, std = self._zscore(image, dim=[1, 2, 3])  # mean [B,1,1,1,2], std [B,1,1,1,1]

        t = self.num_frames
        window = image.unsqueeze(1).expand(-1, t, -1, -1, -1, -1).contiguous()
        window_mask = mask.bool().unsqueeze(-1).unsqueeze(1).expand(b, t, c, h, w, 2).contiguous()

        with torch.autocast("cuda", self.autocast_dtype, enabled=masked_kspace.is_cuda):
            output = self.network(window, window_mask, self.mask_type, acceleration, self.acq_type)
        centre = output[:, t // 2].float()  # [B, C, H, W, 2]
        centre = centre * std + mean

        rss = centre.pow(2).sum(dim=-1).sum(dim=1).sqrt()  # [B, H, W]
        return torch.complex(rss, torch.zeros_like(rss)).unsqueeze(1)
