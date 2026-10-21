"""A saved binary mask used for every slice in place of the sampler's."""
import torch
from torch import nn


class FileMaskPolicy(nn.Module):
    """Returns one saved mask for every slice, so only the reconstructor trains.

    The file is a `torch.save` of `{'mask': [H, W]}` (or the bare tensor): the complete centred mask,
    pre-selected centre included, e.g. one written by scripts/diagnostics/make_spacing_masks_r8.py.

    Args:
        path: The saved mask.
        shape: The model's (H, W) resolution, which the mask must match.
        centre: Optional [H, W] pre-selected centre; the mask must sample all of it.
    """

    def __init__(self, path: str, shape, centre=None):
        super().__init__()
        saved = torch.load(path, map_location="cpu", weights_only=False)
        mask = torch.as_tensor(saved["mask"] if isinstance(saved, dict) else saved).float()
        if tuple(mask.shape) != tuple(shape):
            raise ValueError(f"{path} holds a {tuple(mask.shape)} mask; the model's resolution is {tuple(shape)}")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{path} is not a binary mask")
        if centre is not None and bool((centre * (1 - mask)).any()):
            raise ValueError(f"{path} leaves part of the pre-selected centre unsampled; the mask must include it")
        self.register_buffer("mask", mask.view(1, 1, *shape), persistent=False)

    def forward(self, full_kspace: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W, 2]` -> the saved mask `[B, 1, H, W]`."""
        return self.mask.to(full_kspace.dtype).expand(full_kspace.shape[0], -1, -1, -1)
