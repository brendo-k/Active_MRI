"""PISCO-style subspace sensitivity map estimation (Lobos, Chan, Haldar, TMI 2024) in torch.

Nullspace of the calibration Hankel matrix -> G(x) = H(x)^H H(x) on a low-resolution grid via one
zero-padded FFT per coil pair -> smallest eigenvector of G(x) -> phase alignment -> sinc interpolation.
The smallest-eigenvalue map doubles as the object support.
"""
from dataclasses import dataclass

import torch

from activemri.utils.image_processing import fft_2d_img, ifft_2d_img


@dataclass(frozen=True)
class PiscoConfig:
    calib: int = 24  # side of the centre k-space block the maps are estimated from
    kernel: int = 5  # diameter of the ellipsoidal FIR support
    singular_rtol: float = 0.02  # C singular values above this fraction of the largest span the signal subspace
    grid_extra: int = 24  # low-resolution grid side = calib + grid_extra
    support_threshold: float = 0.95  # 1 - lambda_min(G)/|Lambda| above this counts as object; 0 = ignore this criterion
    support_rss: float = 0.0  # calibration-image RSS above this fraction of its peak counts as object; 0 = ignore this criterion
    support_dilate: int = 0  # pixels the support grows by, so thresholded edges do not clip anatomy
    support_noise: float = 0.0  # calibration-image RSS above this multiple of its 10th percentile (the air level) counts as object; 0 = ignore this criterion


# Tuned on train volumes, confirmed on val (scripts/diagnostics/tune_pisco_support.py): brain wants a tight support that drops air,
# knee a loose one that never clips tissue.
PRESETS = {
    'brain': PiscoConfig(calib=24, kernel=5, singular_rtol=0.1, grid_extra=24, support_threshold=0.0, support_rss=0.1, support_dilate=3),
    'knee': PiscoConfig(calib=24, kernel=5, singular_rtol=0.03, grid_extra=24, support_threshold=0.9, support_dilate=3),
}


def ellipsoid_offsets(kernel: int, device) -> torch.Tensor:
    """[|Lambda|, 2] integer offsets within the disc of radius (kernel - 1) / 2."""
    radius = (kernel - 1) / 2
    span = torch.arange(-int(radius), int(radius) + 1, device=device)
    grid = torch.stack(torch.meshgrid(span, span, indexing='ij'), dim=-1).reshape(-1, 2)
    return grid[(grid.float() ** 2).sum(dim=1) <= radius ** 2 + 1e-6]


def calibration_matrix(block: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Hankel matrix C [P, Q*|Lambda|] of a [Q, n, n] calibration block; columns ordered (coil, offset)."""
    Q, n, _ = block.shape
    reach = int(offsets.abs().max())
    centres = torch.arange(reach, n - reach, device=block.device)
    rows = centres[:, None, None] + offsets[None, None, :, 0]
    cols = centres[None, :, None] + offsets[None, None, :, 1]
    rows = rows.expand(-1, len(centres), -1)
    cols = cols.expand(len(centres), -1, -1)
    patches = block[:, rows, cols]  # [Q, P1, P2, |Lambda|]
    return patches.permute(1, 2, 0, 3).reshape(-1, Q * len(offsets))


def nullspace_projector(C: torch.Tensor, singular_rtol: float) -> torch.Tensor:
    """W = I - V_signal V_signal^H, the projector onto the approximate nullspace of C."""
    gram = C.conj().T @ C
    evals, evecs = torch.linalg.eigh(gram)
    signal = evals.clamp(min=0).sqrt() > singular_rtol * evals[-1].clamp(min=0).sqrt()
    V = evecs[:, signal]
    return torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) - V @ V.conj().T


def gram_on_grid(W: torch.Tensor, offsets: torch.Tensor, Q: int, grid: int) -> torch.Tensor:
    """G(x) [grid, grid, Q, Q] centred on the image grid: G_pq(x) = sum_{s,t} e^{-i(n_s - n_t).x} W[(q,s),(p,t)]."""
    L = len(offsets)
    W = W.reshape(Q, L, Q, L)  # [q, s, p, t]
    diff = offsets[:, None, :] - offsets[None, :, :]  # [s, t, 2]
    reach = 2 * int(offsets.abs().max())
    side = 2 * reach + 1
    flat = (diff[..., 0] + reach) * side + (diff[..., 1] + reach)  # [s, t]
    coeff = torch.zeros(side * side, Q, Q, dtype=W.dtype, device=W.device)
    coeff.index_add_(0, flat.reshape(-1), W.permute(1, 3, 2, 0).reshape(L * L, Q, Q))  # [s*t, p, q]
    coeff = coeff.reshape(side, side, Q, Q)

    padded = torch.zeros(grid, grid, Q, Q, dtype=W.dtype, device=W.device)
    index = torch.arange(-reach, reach + 1, device=W.device) % grid
    padded[index[:, None], index[None, :]] = coeff
    G = torch.fft.fft2(padded, dim=(0, 1))  # sum_d coeff[d] e^{-i2pi d.x}: C holds correlation patches s[n+m], so the image-domain filter is sum_m h[m] e^{-i2pi m.x}
    return torch.fft.fftshift(G, dim=(0, 1))


def sinc_interpolate(images: torch.Tensor, shape) -> torch.Tensor:
    """Periodic sinc interpolation of [..., h, w] to [..., H, W] by zero-padding centred k-space."""
    h, w = images.shape[-2:]
    H, W = shape
    k = fft_2d_img(images)
    out = torch.zeros(*images.shape[:-2], H, W, dtype=k.dtype, device=k.device)
    top, left = H // 2 - h // 2, W // 2 - w // 2
    out[..., top:top + h, left:left + w] = k
    return ifft_2d_img(out) * ((H * W) / (h * w)) ** 0.5


def centre_block(kspace: torch.Tensor, size: int) -> torch.Tensor:
    H, W = kspace.shape[-2:]
    return kspace[..., H // 2 - size // 2:H // 2 + size // 2, W // 2 - size // 2:W // 2 + size // 2]


@dataclass
class PiscoMaps:
    """Unmasked estimate; `with_support` turns it into the maps a reconstruction uses."""
    maps: torch.Tensor  # [C, H, W], unit norm over the live coils everywhere, zero for padded coils
    subspace_residual: torch.Tensor  # [H, W], lambda_min(G)/|Lambda|
    calibration_rss: torch.Tensor  # [H, W], RSS of the sinc-interpolated calibration images

    def with_support(self, config: PiscoConfig):
        """(maps [C, H, W], support [H, W]) with sum_c |S_c|^2 = 1 on the support and 0 outside."""
        support = torch.ones_like(self.subspace_residual)
        if config.support_threshold > 0:
            support = support * ((1 - self.subspace_residual) > config.support_threshold).float()
        if config.support_rss > 0:
            support = support * (self.calibration_rss > config.support_rss * self.calibration_rss.max()).float()
        if config.support_noise > 0:
            air_level = torch.quantile(self.calibration_rss.flatten(), 0.1)
            support = support * (self.calibration_rss > config.support_noise * air_level).float()
        if config.support_dilate > 0:
            width = 2 * config.support_dilate + 1
            support = torch.nn.functional.max_pool2d(support[None, None], width, stride=1, padding=config.support_dilate)[0, 0]
        return self.maps * support, support


def pisco_estimate(kspace: torch.Tensor, config: PiscoConfig = PiscoConfig()) -> PiscoMaps:
    """Sensitivity maps from a fully sampled [C, H, W] complex k-space; all-zero (padded) coils are skipped."""
    C_total, H, W = kspace.shape
    live = kspace.abs().flatten(1).amax(dim=1) > 0
    ks = kspace[live]
    Q = ks.shape[0]

    offsets = ellipsoid_offsets(config.kernel, ks.device)
    block = centre_block(ks, config.calib)
    nullspace = nullspace_projector(calibration_matrix(block, offsets), config.singular_rtol)

    grid = config.calib + config.grid_extra
    G = gram_on_grid(nullspace, offsets, Q, grid).reshape(grid * grid, Q, Q)
    G = 0.5 * (G + G.conj().transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(G)
    low_maps = evecs[..., 0].T.reshape(Q, grid, grid)
    residual = (evals[:, 0] / len(offsets)).reshape(grid, grid)

    padded_block = torch.zeros(Q, grid, grid, dtype=ks.dtype, device=ks.device)
    start = grid // 2 - config.calib // 2
    padded_block[:, start:start + config.calib, start:start + config.calib] = block
    low_res = ifft_2d_img(padded_block)
    combined = (low_maps.conj() * low_res).sum(dim=0)
    low_maps = low_maps * torch.exp(1j * combined.angle())

    maps = sinc_interpolate(low_maps, (H, W))
    maps = maps / maps.abs().pow(2).sum(dim=0).sqrt().clamp(min=1e-8)
    residual = sinc_interpolate(residual.to(maps.dtype), (H, W)).real
    rss = sinc_interpolate(low_res, (H, W)).abs().pow(2).sum(dim=0).sqrt()

    full = torch.zeros(C_total, H, W, dtype=maps.dtype, device=maps.device)
    full[live] = maps
    return PiscoMaps(full, residual, rss)


def pisco_sense_maps(kspace: torch.Tensor, config: PiscoConfig = PiscoConfig()):
    """(maps [C, H, W], support [H, W], subspace_residual [H, W]) for a fully sampled [C, H, W] complex k-space."""
    estimate = pisco_estimate(kspace, config)
    maps, support = estimate.with_support(config)
    return maps, support, estimate.subspace_residual
