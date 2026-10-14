"""Self-supervised (SSDU) examples from preprocessed Siemens SPACE volumes.

Each example is one readout position of a volume: a [C, ky, kz] multicoil k-space that was already
accelerated by the scanner, so only its acquired set Omega carries data. Every example hides a random
subset Lambda of Omega:

    kspace          y on Omega \\ Lambda  - everything the model (and the sampler) may see or pick from
    kspace_support  the acquired envelope (Omega dilated) - the only region the reconstructor may fill; the
                    loss never samples outside Omega, so energy placed elsewhere would go unpunished
    holdout_kspace  y on Lambda          - never an input; the loss is evaluated here
    holdout_mask    Lambda

Lambda is drawn independently of the sampler, so the sampler cannot lower the loss by moving points between
its input and the loss set. Train draws a fresh Lambda per access; val derives it from the example index.
"""
import pathlib
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ANATOMY = 'space'
VAL_BLOCKS = (2, 6)  # of 10 equal blocks of the signal-bearing readout positions
BLOCK_COUNT = 10
BLOCK_GUARD = 4  # positions dropped from each side of a val block's train neighbours: adjacent readout slices are near-duplicates
VAL_SEED = 1234
SUPPORT_DILATION = 5  # closes the gaps between lattice points (and those Lambda leaves) so the support is the acquired envelope


def split_slices(signal_slices: np.ndarray, split: str) -> np.ndarray:
    """Train or val readout positions: val is two contiguous blocks, train is the rest minus a guard band around them."""
    blocks = np.array_split(np.asarray(signal_slices), BLOCK_COUNT)
    val = np.concatenate([blocks[b] for b in VAL_BLOCKS])
    if split == 'val':
        return val
    guard = np.concatenate([np.arange(blocks[b][0] - BLOCK_GUARD, blocks[b][-1] + BLOCK_GUARD + 1) for b in VAL_BLOCKS])
    return np.setdiff1d(signal_slices, guard)


def draw_holdout(omega: torch.Tensor, fraction: float, generator: torch.Generator) -> torch.Tensor:
    """Lambda: a uniformly random `fraction` of the acquired points, [H, W] float."""
    acquired = omega.flatten().nonzero().squeeze(1)
    count = int(round(fraction * len(acquired)))
    chosen = acquired[torch.randperm(len(acquired), generator=generator)[:count]]
    holdout = torch.zeros(omega.numel())
    holdout[chosen] = 1
    return holdout.view_as(omega)


class SiemensSpaceDataset(torch.utils.data.Dataset):
    def __init__(self, h5_paths: Sequence[pathlib.Path], split: str, holdout_fraction: float = 0.2,
                 image_shape: Tuple[int, int] = (256, 64), val_volumes: Sequence[str] = ()):
        self.split = split
        self.holdout_fraction = holdout_fraction
        self.anatomy = ANATOMY
        self.examples: List[Tuple[pathlib.Path, int]] = []
        self.omegas = {}
        self.supports = {}
        for path in map(pathlib.Path, h5_paths):
            with h5py.File(path, 'r') as f:
                if tuple(f.attrs['grid']) != tuple(image_shape):
                    raise ValueError(f"{path.name} is on a {tuple(f.attrs['grid'])} grid, not {tuple(image_shape)}")
                self.omegas[path] = torch.from_numpy(f['omega'][:]).float()
                self.supports[path] = (F.max_pool2d(self.omegas[path][None, None], SUPPORT_DILATION, 1, SUPPORT_DILATION // 2)[0] > 0).float()
                signal = f['signal_slices'][:]
                if val_volumes:
                    positions = signal if (path.stem in val_volumes) == (split == 'val') else []
                else:
                    positions = split_slices(signal, split)
            self.examples += [(path, int(x)) for x in positions]

    def __len__(self):
        return len(self.examples)

    def _holdout(self, index: int, omega: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator()
        if self.split == 'val':
            generator.manual_seed(VAL_SEED + index)
        else:
            generator.seed()
        return draw_holdout(omega, self.holdout_fraction, generator)

    def __getitem__(self, index):
        path, position = self.examples[index]
        with h5py.File(path, 'r') as f:
            kspace = torch.from_numpy(f['kspace'][position])
            sense = torch.from_numpy(f['sense'][position])
            target = torch.from_numpy(f['reference'][position]).float()

        omega = self.omegas[path]
        holdout = self._holdout(index, omega)
        pool = omega - holdout
        return {
            'kspace': torch.view_as_real(kspace * pool),
            'holdout_kspace': torch.view_as_real(kspace * holdout),
            'holdout_mask': holdout.unsqueeze(0),
            'kspace_support': self.supports[path],
            'sense_maps': sense,
            'target': target,
            'anatomy': self.anatomy,
            'condition': self.anatomy,
            'attrs': {},
            'fname': path.name,
            'slice_id': position,
        }


def build_siemens_space_dataloaders(
    h5_paths: Sequence[pathlib.Path],
    image_shape: Tuple[int, int],
    batch_size: int,
    holdout_fraction: float = 0.2,
    num_workers: int = 4,
    val_volumes: Sequence[str] = (),
):
    """With `val_volumes` (file stems) those volumes are val entirely and the rest train entirely; otherwise each volume is split by readout block."""
    train = SiemensSpaceDataset(h5_paths, 'train', holdout_fraction, image_shape, val_volumes)
    val = SiemensSpaceDataset(h5_paths, 'val', holdout_fraction, image_shape, val_volumes)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
