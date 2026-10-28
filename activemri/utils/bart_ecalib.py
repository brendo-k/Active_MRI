"""BART `ecalib` (ESPIRiT) sensitivity maps for a [C, H, W] complex k-space, through the bart CLI."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


# Tuned on train volumes, confirmed on val (scripts/diagnostics/tune_ecalib.py); 24x24 calibration, soft-SENSE maps with an eigenvalue crop.
PRESETS = {
    'brain': dict(calib=24, kernel=5, threshold=0.001, crop=0.93, soft=True),
    'knee': dict(calib=24, kernel=6, threshold=0.001, crop=0.8, soft=True),
}


def write_cfl(name: Path, array: np.ndarray) -> None:
    dims = list(array.shape) + [1] * (16 - array.ndim)
    Path(f'{name}.hdr').write_text('# Dimensions\n' + ' '.join(map(str, dims)) + '\n')
    np.asfortranarray(array.astype(np.complex64)).T.copy().view(np.float32).tofile(f'{name}.cfl')


def read_cfl(name: Path) -> np.ndarray:
    dims = [int(d) for d in Path(f'{name}.hdr').read_text().splitlines()[1].split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    data = np.fromfile(f'{name}.cfl', dtype=np.float32)
    return (data[0::2] + 1j * data[1::2]).reshape(dims, order='F')


def ecalib_sense_maps(kspace: torch.Tensor, calib: int = 24, kernel: int = 6, threshold: float = 0.001, crop: float = 0.8,
                      soft: bool = False, extra=()):
    """(maps [C, H, W] complex with sum_c |S_c|^2 = 1 where kept, support [H, W]) from the centre `calib` x `calib` block.

    `crop`: eigenvalue below which a pixel is dropped (0 keeps every pixel). All-zero (padded) coils are skipped.
    """
    C, H, W = kspace.shape
    live = (kspace.abs().flatten(1).amax(dim=1) > 0).cpu().numpy()
    k = kspace.cpu().numpy()[live].transpose(1, 2, 0)[:, :, None, :]  # [H, W, 1, C_live]
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        tmp = Path(tmp)
        write_cfl(tmp / 'k', k)
        command = ['bart', 'ecalib', '-m', '1', '-r', f'{calib}:{calib}:1', '-k', f'{kernel}:{kernel}:1', '-t', str(threshold), '-c', str(crop), *extra]
        if soft:
            command.append('-S')
        subprocess.run([*command, str(tmp / 'k'), str(tmp / 's')], check=True, capture_output=True)
        maps = read_cfl(tmp / 's').reshape(H, W, -1)
    out = np.zeros((C, H, W), dtype=np.complex64)
    out[live] = maps.transpose(2, 0, 1)
    out = torch.from_numpy(out).to(kspace.device)
    support = (out.abs().pow(2).sum(0) > 0.5).to(out.real.dtype)
    return out, support
