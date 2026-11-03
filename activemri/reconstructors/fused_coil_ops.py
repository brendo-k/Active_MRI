"""The VarNet cascade's multi-coil arithmetic in real form, compiled so each operation is one fused kernel (and so is
its backward), instead of eager complex kernels that each read and write the whole [B, C, H, W] coil tensor.

    coil_combine(x, S)    sum_c conj(S_c) x_c                                  (the SENSE reduce after the inverse FFT)
    coil_expand(image, S) S_c image                                            (the SENSE expand before the FFT)
    dc_update(k, k0, m, w, t)  k - where(m != 0, k - k0, 0) * w - t            (soft DC and the regulariser step)

The FFTs stay cuFFT calls outside. Each function takes and returns complex tensors and compiles a real-valued body
(complex tensors never enter the compiled graph: inductor miscomputed a compiled cascade on complex inputs). The
arithmetic is the eager expressions written out per component, so results match eager up to float rounding (the
coil sum's order).
"""
import torch

_COMPILED = {}


def _compiled(fn):
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=False, fullgraph=True)
    return _COMPILED[fn]


def _combine(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    xr, xi, sr, si = x[..., 0], x[..., 1], s[..., 0], s[..., 1]
    real = (xr * sr + xi * si).sum(dim=1, keepdim=True)
    imag = (xi * sr - xr * si).sum(dim=1, keepdim=True)
    return torch.stack((real, imag), dim=-1)


def _expand(image: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    ir, ii, sr, si = image[..., 0], image[..., 1], s[..., 0], s[..., 1]
    return torch.stack((ir * sr - ii * si, ir * si + ii * sr), dim=-1)


def _dc_update(current: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor,
               term: torch.Tensor) -> torch.Tensor:
    keep = (mask != 0).unsqueeze(-1)
    soft_dc = torch.where(keep, current - ref, torch.zeros_like(current)) * weight
    return current - soft_dc - term


def coil_combine(coil_images: torch.Tensor, sense_maps: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] complex coil images and maps -> [B, 1, H, W] complex sum_c conj(S_c) x_c."""
    return torch.view_as_complex(_compiled(_combine)(torch.view_as_real(coil_images), torch.view_as_real(sense_maps)))


def coil_expand(image: torch.Tensor, sense_maps: torch.Tensor) -> torch.Tensor:
    """[B, 1, H, W] complex image and [B, C, H, W] complex maps -> [B, C, H, W] complex S_c image."""
    return torch.view_as_complex(_compiled(_expand)(torch.view_as_real(image), torch.view_as_real(sense_maps)))


def dc_update(current: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor,
              term: torch.Tensor) -> torch.Tensor:
    """current - where(mask != 0, current - ref, 0) * weight - term, for complex [B, C, H, W] k-spaces, a real
    [B, 1, H, W] mask and a real scalar weight of shape [1]."""
    return torch.view_as_complex(_compiled(_dc_update)(
        torch.view_as_real(current), torch.view_as_real(ref), mask, weight, torch.view_as_real(term)))
