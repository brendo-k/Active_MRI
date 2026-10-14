"""Turn a Siemens 3D Cartesian (SPACE) twix file into the per-volume h5 that `SiemensSpaceDataset` reads.

The scans are accelerated by the scanner (2x2 CAIPIRINHA, partial Fourier, elliptical scanning), so there is
no fully-sampled k-space: the acquired set Omega is the ceiling on what any sampler can pick, and training is
self-supervised. The readout is fully sampled, so only (ky, kz) can be undersampled; the volume is
inverse-transformed along the readout and each readout position becomes one 2D example of shape [C, ky, kz].

Steps: read twix (Seg merged: every (Lin, Par) is acquired exactly once) -> SVD coil compression -> embed the
acquired lines on a k-space grid whose centre sits at index N // 2, the repo's fftshift convention (partial
Fourier is then the zero band it really is, not a phase ramp) -> readout IFFT -> BART ESPIRiT maps from the
separate 24x24 reference scan (the scan's own centre is only a 2x2 lattice, no ACS) -> CG-SENSE image of all acquired data,
used for monitoring only, never as a training target.
"""
import pathlib
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass

import h5py
import mapvbvd
import numpy as np
import torch

from activemri.reconstructors.CG_Sense import conjugate_gradient_sense

NUM_COILS = 8
SAMPLES_FOR_COMPRESSION = 400_000
ECALIB_ARGS = ['-m1', '-r', '24:24:24']
CG_ITERATIONS, CG_LAMBDA = 50, 0.01
REFERENCE_BATCH = 16
SIGNAL_SLICE_FRACTION = 0.1  # readout positions whose reference energy is below this fraction of the peak slice are not examples


@dataclass
class TwixVolume:
    kspace: np.ndarray  # [Col, Cha, Lin, Par] complex64, readout still in k-space
    lin: np.ndarray
    par: np.ndarray
    center: tuple[int, int]  # (Lin, Par) index of k = 0
    acs: np.ndarray  # [Col, Cha, 24, 24] reference scan
    acs_center: tuple[int, int]  # index of k = 0 inside the acs block
    attrs: dict


def _merged_segments(scan):
    scan.flagIgnoreSeg = True
    scan.squeeze = True
    return np.asarray(scan['']).astype(np.complex64)


def read_twix(dat_path: pathlib.Path) -> TwixVolume:
    twix = mapvbvd.mapVBVD(str(dat_path), quiet=True)
    twix = twix[-1] if isinstance(twix, list) else twix
    image = twix.image
    lin, par = np.asarray(image.Lin).astype(int), np.asarray(image.Par).astype(int)
    centre = (int(np.unique(image.centerLin)[0]), int(np.unique(image.centerPar)[0]))
    kspace = _merged_segments(image)
    counts = Counter(zip(lin, par))
    repeated = np.array([key for key, count in counts.items() if count > 1]).reshape(-1, 2)
    if len(repeated):
        # Merging segments gives a repeated line a meaningless amplitude (about half its neighbours' energy), so it is dropped.
        kspace[:, :, repeated[:, 0], repeated[:, 1]] = 0
        keep = np.array([counts[key] == 1 for key in zip(lin, par)])
        lin, par = lin[keep], par[keep]

    ref = twix.refscan
    acs = _merged_segments(ref)
    ref_lin, ref_par = np.asarray(ref.Lin).astype(int), np.asarray(ref.Par).astype(int)
    acs_centre = (int(np.unique(ref.centerLin)[0]) - ref_lin.min(), int(np.unique(ref.centerPar)[0]) - ref_par.min())

    yaps = twix.hdr['MeasYaps']
    attrs = {
        'protocol': str(yaps[('tProtocolName',)]) if ('tProtocolName',) in yaps else dat_path.stem,
        'fov_read_mm': float(yaps[('sSliceArray', 'asSlice', '0', 'dReadoutFOV')]),
        'fov_phase_mm': float(yaps[('sSliceArray', 'asSlice', '0', 'dPhaseFOV')]),
        'source': dat_path.name,
    }
    return TwixVolume(kspace, lin, par, centre, acs, acs_centre, attrs)


def coil_compression_matrix(kspace: np.ndarray, num_coils: int, seed: int = 0) -> np.ndarray:
    """[Cha, num_coils] right-singular vectors of the acquired samples, by energy."""
    samples = np.moveaxis(kspace, 1, -1).reshape(-1, kspace.shape[1])
    samples = samples[np.abs(samples).sum(axis=1) > 0]
    rng = np.random.default_rng(seed)
    samples = samples[rng.choice(len(samples), min(len(samples), SAMPLES_FOR_COMPRESSION), replace=False)]
    _, _, vh = np.linalg.svd(samples, full_matrices=False)
    return vh.conj().T[:, :num_coils].astype(np.complex64)


def compress(kspace: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply the coil compression to axis 1."""
    return np.einsum('xcyz,cd->xdyz', kspace, matrix, optimize=True).astype(np.complex64)


def embed_on_grid(lines: np.ndarray, rows: np.ndarray, cols: np.ndarray, centre, grid: tuple[int, int]) -> np.ndarray:
    """Place [Col, Cha, Lin, Par] on a [Col, Cha, H, W] grid with k = 0 at (H // 2, W // 2); `rows`/`cols` are the Lin/Par of every acquired line."""
    grid_centre = (grid[0] // 2, grid[1] // 2)
    row_offset, col_offset = grid_centre[0] - centre[0], grid_centre[1] - centre[1]
    if rows.min() + row_offset < 0 or rows.max() + row_offset >= grid[0] or cols.min() + col_offset < 0 or cols.max() + col_offset >= grid[1]:
        raise ValueError(f"acquired lines {rows.min()}..{rows.max()} x {cols.min()}..{cols.max()} around centre {centre} do not fit a {grid} grid")
    out = np.zeros((*lines.shape[:2], *grid), np.complex64)
    out[:, :, rows + row_offset, cols + col_offset] = lines[:, :, rows, cols]
    return out


def readout_ifft(kspace: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace, axes=0), axis=0, norm='ortho'), axes=0).astype(np.complex64)


def write_cfl(path: pathlib.Path, array: np.ndarray):
    dims = list(array.shape) + [1] * (16 - array.ndim)
    path.with_suffix('.hdr').write_text('# Dimensions\n' + ' '.join(map(str, dims)) + '\n')
    np.asarray(array, np.complex64).ravel(order='F').tofile(path.with_suffix('.cfl'))


def read_cfl(path: pathlib.Path) -> np.ndarray:
    dims = [int(d) for d in path.with_suffix('.hdr').read_text().splitlines()[1].split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return np.fromfile(path.with_suffix('.cfl'), np.complex64).reshape(dims, order='F')


def bart(*args):
    subprocess.run(['bart', *map(str, args)], check=True)


def estimate_sense_maps(acs: np.ndarray, acs_centre, readout: int, grid: tuple[int, int], work: pathlib.Path) -> np.ndarray:
    """ESPIRiT maps [Col, Cha, H, W] (image domain, x-first) from the reference block zero-padded onto the grid."""
    padded = np.zeros((readout, acs.shape[1], *grid), np.complex64)
    row0, col0 = grid[0] // 2 - acs_centre[0], grid[1] // 2 - acs_centre[1]
    x0 = readout // 2 - acs.shape[0] // 2
    padded[x0:x0 + acs.shape[0], :, row0:row0 + acs.shape[2], col0:col0 + acs.shape[3]] = acs
    write_cfl(work / 'acs', np.moveaxis(padded, 1, -1))
    bart('ecalib', *ECALIB_ARGS, work / 'acs', work / 'sens')
    return np.moveaxis(read_cfl(work / 'sens'), -1, 1)


def cg_reference(kspace: np.ndarray, sense: np.ndarray, omega: np.ndarray, device: str) -> np.ndarray:
    """Magnitude of a lightly regularized CG-SENSE image of everything acquired, [Col, H, W]. Monitoring only.

    Tikhonov CG is invariant to the scale of the data, so it needs no normalization guess, and it fits the
    acquired data by construction.
    """
    mask = torch.from_numpy(omega).float().to(device)[None, None]
    out = np.empty((kspace.shape[0], *omega.shape), np.float32)
    for start in range(0, kspace.shape[0], REFERENCE_BATCH):
        k = torch.from_numpy(kspace[start:start + REFERENCE_BATCH]).to(device)
        s = torch.from_numpy(sense[start:start + REFERENCE_BATCH]).to(device)
        image = conjugate_gradient_sense(k, s, mask.expand(len(k), -1, -1, -1), CG_ITERATIONS, CG_LAMBDA)
        out[start:start + len(k)] = image.abs().cpu().numpy()
    return out


def preprocess(dat_path: pathlib.Path, out_path: pathlib.Path, grid: tuple[int, int]):
    volume = read_twix(dat_path)
    matrix = coil_compression_matrix(volume.kspace, NUM_COILS)
    lines = compress(volume.kspace, matrix)
    acs = compress(volume.acs, matrix)

    kspace_hybrid_k = embed_on_grid(lines, volume.lin, volume.par, volume.center, grid)
    omega = (np.abs(kspace_hybrid_k).sum(axis=(0, 1)) > 0).astype(np.uint8)
    readout = kspace_hybrid_k.shape[0]

    with tempfile.TemporaryDirectory(dir=out_path.parent) as scratch:
        sense = estimate_sense_maps(acs, volume.acs_center, readout, grid, pathlib.Path(scratch))

    kspace = readout_ifft(kspace_hybrid_k)
    reference = cg_reference(kspace, sense, omega, 'cuda' if torch.cuda.is_available() else 'cpu')
    scale = float(np.percentile(reference, 99.9))
    kspace, reference = kspace / scale, reference / scale
    energy = (reference ** 2).sum(axis=(1, 2))
    signal_slices = np.flatnonzero(energy > SIGNAL_SLICE_FRACTION * energy.max())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('kspace', data=kspace, chunks=(1, *kspace.shape[1:]))
        f.create_dataset('sense', data=sense, chunks=(1, *sense.shape[1:]))
        f.create_dataset('reference', data=reference, chunks=(1, *reference.shape[1:]))
        f.create_dataset('omega', data=omega)
        f.create_dataset('signal_slices', data=signal_slices)
        f.attrs.update(volume.attrs)
        f.attrs.update(scale=scale, grid=grid, k_center=volume.center, acquired_fraction=float(omega.mean()),
                       coils_in=volume.kspace.shape[1], coils_out=NUM_COILS)
    return out_path
