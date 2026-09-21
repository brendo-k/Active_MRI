import math
import pathlib
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data

ifft_2d_img = lambda x, dim=(-1, -2): np.fft.fftshift(
    np.fft.ifft2(np.fft.ifftshift(x, axes=dim), axes=dim, norm='ortho'), axes=dim
)
rss = lambda x, dim=0: np.sqrt(np.sum(np.power(np.abs(x), 2), axis=dim))

ANATOMY_DIR_MAP = {
    'brain': 'multicoil_brain',
    'knee': 'multicoil_knee',
    'breast': 'multicoil_breast',
    'prostate': 'multicoil_prostate',
}

DEFAULT_BRAIN_CONTRAST = 'T2'


class FastMRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        anatomy: str,
        image_shape: List[int],
        transform: Optional[Callable] = None,
        num_coils: int = 8,
        num_volumes: Optional[int] = None,
        min_energy_fraction: float = 0.2,
        file_names: Optional[Sequence[str]] = None,
        slice_ids: Optional[Sequence[int]] = None,
    ):
        self.root = pathlib.Path(root)
        self.anatomy = anatomy
        self.image_shape = image_shape
        self.transform = transform
        self.num_coils = num_coils
        self.min_energy_fraction = min_energy_fraction

        self.examples: List[Tuple[pathlib.Path, int]] = []
        rng = np.random.RandomState(1234)

        anatomy_file_markers = {
            'brain': 'brain',
            'prostate': 'prostate',
            'breast': 'breast',
        }

        files = sorted(
            f for f in self.root.iterdir()
            if f.suffix in ('.h5', '.hdf5')
        )

        marker = anatomy_file_markers.get(self.anatomy)
        if marker is not None:
            files = [f for f in files if marker in f.name]
        else:
            exclude = set(anatomy_file_markers.values())
            files = [f for f in files if not any(m in f.name for m in exclude)]

        if file_names is not None:
            wanted = set(file_names)
            files = [f for f in files if f.name in wanted]
            missing = wanted - {f.name for f in files}
            if missing:
                raise FileNotFoundError(f"Requested files not found under {self.root}: {sorted(missing)}")

        if num_volumes is not None:
            rng.shuffle(files)
            files = sorted(files[:num_volumes])

        if slice_ids is not None:
            self.examples = [(f, sid) for f in files for sid in slice_ids]
            return

        for fname in files:
            try:
                with h5py.File(fname, 'r') as data:
                    if 'kspace' not in data:
                        continue
                    # Prostate k-space is (averages, slices, coils, readout, phase),
                    # so the slice dimension is axis 1 rather than axis 0.
                    slice_axis = 1 if self.anatomy == 'prostate' else 0
                    num_slices = data['kspace'].shape[slice_axis]

                    if self.min_energy_fraction > 0:
                        energies = []
                        for sid in range(num_slices):
                            if self.anatomy == 'prostate':
                                kslice = data['kspace'][:, sid].sum(axis=0)
                            else:
                                kslice = data['kspace'][sid]
                            energies.append(self._center_energy(kslice))
                        energies = np.array(energies)
                        threshold = self.min_energy_fraction * energies.max()
                        valid_slices = [s for s in range(num_slices) if energies[s] >= threshold]
                    else:
                        valid_slices = list(range(num_slices))

                    self.examples += [(fname, sid) for sid in valid_slices]
            except OSError as err:
                warnings.warn(f"Skipping unreadable file {fname}: {err}")

    @staticmethod
    def _center_energy(kspace_slice, center_size=10):
        h, w = kspace_slice.shape[-2], kspace_slice.shape[-1]
        ch, cw = h // 2, w // 2
        n = center_size // 2
        center = kspace_slice[..., ch - n:ch + n, cw - n:cw + n]
        return np.sqrt(np.sum(np.abs(center) ** 2))

    def __len__(self):
        return len(self.examples)

    def _pad_or_crop_coils(self, kspace: torch.Tensor) -> torch.Tensor:
        num_coils = kspace.shape[0]
        if num_coils > self.num_coils:
            kspace = kspace[:self.num_coils]
        elif num_coils < self.num_coils:
            pad = torch.zeros(
                self.num_coils - num_coils, *kspace.shape[1:], dtype=kspace.dtype
            )
            kspace = torch.cat([kspace, pad], dim=0)
        return kspace

    def _center_crop(self, data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        return data[..., w_from:w_from + shape[0], h_from:h_from + shape[1]]

    def _merge_prostate_averages(self, averages: np.ndarray) -> np.ndarray:
        """Merge the interleaved averages of a prostate slice into full k-space.

        Each average is R=2 undersampled along the phase-encode axis with the
        averages interleaved (e.g. avgs 0/2 cover odd lines, avg 1 covers even),
        so their union is effectively fully sampled. Sum the averages and divide
        by the number that sampled each phase-encode line to avoid double-counting
        lines acquired in more than one average.

        Args:
            averages: complex k-space of shape (averages, coils, readout, phase).

        Returns:
            Merged k-space of shape (coils, readout, phase).
        """
        sampled = np.abs(averages).sum(axis=(1, 2)) > 0  # (averages, phase)
        counts = np.maximum(sampled.sum(axis=0), 1)      # (phase,)
        merged = averages.sum(axis=0)                    # (coils, readout, phase)
        return merged / counts[None, None, :]

    def _get_center_kspace(
        self, kspace: torch.Tensor, center_size: Tuple[int, int] = (10, 10)
    ) -> torch.Tensor:
        mask = torch.zeros_like(kspace)
        h, w = self.image_shape
        ch, cw = center_size
        mask[
            ...,
            h // 2 - ch // 2 : h // 2 + math.ceil(ch / 2),
            w // 2 - cw // 2 : w // 2 + math.ceil(cw / 2),
        ] = 1
        return kspace * mask

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]

        with h5py.File(fname, 'r') as data:
            if self.anatomy == 'prostate':
                averages = data['kspace'][:, slice_id]
                kspace = torch.from_numpy(self._merge_prostate_averages(averages))
            else:
                kspace = torch.from_numpy(data['kspace'][slice_id])
            attrs = dict(data.attrs)

            # Load precomputed ESPIRiT sense maps if available. Brain files also carry a
            # 'sense_maps' key, but it is degenerate (support over a third to a half as many
            # rows as 'sense' in the same file) -- 'sense' is the only key actually used.
            espirit_maps = None
            if 'sense' in data:
                raw = data['sense']
                if self.anatomy == 'prostate':
                    # (averages, slices, coils, H, W) — average across the averages axis
                    espirit_maps = torch.from_numpy(raw[:, slice_id].mean(axis=0))
                else:
                    espirit_maps = torch.from_numpy(raw[slice_id])

        # Downsample the largest spatial dimension by 2 before cropping
        if self.anatomy not in ('knee', 'prostate'):
            if kspace.shape[-2] > kspace.shape[-1]:
                kspace = kspace[..., ::2, :]
                if espirit_maps is not None:
                    espirit_maps = espirit_maps[..., ::2, :]
            elif kspace.shape[-1] > kspace.shape[-2]:
                kspace = kspace[..., ::2]
                if espirit_maps is not None:
                    espirit_maps = espirit_maps[..., ::2]

        kspace = self._center_crop(kspace, self.image_shape)
        kspace = self._pad_or_crop_coils(kspace)
        if espirit_maps is not None:
            espirit_maps = self._center_crop(espirit_maps, self.image_shape)
            espirit_maps = self._pad_or_crop_coils(espirit_maps)

        if self.anatomy == 'brain':
            kspace = kspace.flip(-2)
            if espirit_maps is not None:
                espirit_maps = espirit_maps.flip(-2)

        center_k = self._get_center_kspace(kspace)
        norm_factor = np.max(rss(ifft_2d_img(center_k.numpy())))
        kspace = kspace / norm_factor

        target = torch.from_numpy(rss(ifft_2d_img(kspace.numpy()))).float()
        kspace = torch.view_as_real(kspace)

        sample = {
            'kspace': kspace,
            'target': target,
            'anatomy': self.anatomy,
            'attrs': attrs,
            'fname': fname.name,
            'slice_id': slice_id,
            'espirit_maps': espirit_maps,  # [C, H, W] complex or None
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def _resolve_anatomy_dirs(
    data_root: pathlib.Path,
    anatomy: str,
    contrast: Optional[str] = None,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Resolve train/val directories for an anatomy under data_root.

    Handles multiple directory structures:
      - Has train/val subdirs directly (e.g. multicoil_brain/T2/train/)
      - Has contrast subdirs with train/val inside (e.g. multicoil_brain/T2/train/)
      - Flat directory with only h5 files (splits by 80/20 ratio inside Dataset)
    """
    dir_name = ANATOMY_DIR_MAP.get(anatomy)
    if dir_name is None:
        raise ValueError(
            f"Unknown anatomy '{anatomy}'. Available: {list(ANATOMY_DIR_MAP.keys())}"
        )

    anatomy_dir = data_root / dir_name
    if not anatomy_dir.exists():
        raise FileNotFoundError(f"Anatomy directory not found: {anatomy_dir}")

    if anatomy == 'brain':
        contrast_name = contrast or DEFAULT_BRAIN_CONTRAST
        contrast_dir = anatomy_dir / contrast_name
        if not contrast_dir.exists():
            raise FileNotFoundError(
                f"Brain contrast directory not found: {contrast_dir}. "
                f"Available: {[d.name for d in anatomy_dir.iterdir() if d.is_dir()]}"
            )
        anatomy_dir = contrast_dir

    train_dir = anatomy_dir / 'train'
    val_dir = anatomy_dir / 'val'

    if train_dir.exists() and val_dir.exists():
        return train_dir, val_dir

    return anatomy_dir, anatomy_dir


def _collate_fastmri_batch(samples: List[Dict]) -> Dict:
    """Like default_collate, but keeps 'espirit_maps' as a plain list.

    Not every file ships precomputed ESPIRiT maps, so this field is None for
    some samples and a tensor for others — default_collate can't stack that.
    """
    espirit_maps = [s.pop('espirit_maps') for s in samples]
    batch = torch.utils.data.default_collate(samples)
    batch['espirit_maps'] = espirit_maps
    return batch


def build_fastmri_dataloaders(
    data_root: pathlib.Path,
    anatomies: List[str],
    image_shape: List[int],
    batch_size: int,
    transform: Optional[Callable] = None,
    num_coils: int = 8,
    num_workers: int = 4,
    val_fraction: float = 0.2,
    contrast: Optional[str] = None,
    data_dir: Optional[str] = None,
    min_energy_fraction: float = 0.2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and val dataloaders from one or more anatomies.

    Args:
        data_root: Root directory containing anatomy subdirs (e.g. ~/data/fastmri/).
        anatomies: List of anatomy names (brain, knee, breast, prostate).
        image_shape: Target image dimensions [H, W].
        batch_size: Batch size for dataloaders.
        transform: Optional transform applied to each sample.
        num_coils: Number of coils to pad/crop to.
        num_workers: DataLoader workers.
        val_fraction: Fraction of data for validation when no train/val split exists.
        contrast: Brain contrast to use (T2, Flair, T1POST). Defaults to T2.
        data_dir: If provided, use this directory directly (expects train/val subdirs).
            Bypasses anatomy-based directory resolution.
    """
    train_datasets = []
    val_datasets = []

    if data_dir is not None:
        data_path = pathlib.Path(data_dir)
        train_dir = data_path / 'train'
        val_dir = data_path / 'val'
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        for anatomy in anatomies:
            train_datasets.append(
                FastMRIDataset(
                    root=train_dir,
                    anatomy=anatomy,
                    image_shape=image_shape,
                    transform=transform,
                    num_coils=num_coils,
                    min_energy_fraction=min_energy_fraction,
                )
            )
            if val_dir.exists():
                val_datasets.append(
                    FastMRIDataset(
                        root=val_dir,
                        anatomy=anatomy,
                        image_shape=image_shape,
                        transform=transform,
                        num_coils=num_coils,
                        min_energy_fraction=min_energy_fraction,
                    )
                )
            else:
                val_datasets.append(train_datasets[-1])
    else:
        for anatomy in anatomies:
            train_dir, val_dir = _resolve_anatomy_dirs(data_root, anatomy, contrast)

            if train_dir == val_dir:
                dataset = FastMRIDataset(
                    root=train_dir,
                    anatomy=anatomy,
                    image_shape=image_shape,
                    transform=transform,
                    num_coils=num_coils,
                    min_energy_fraction=min_energy_fraction,
                )
                n_val = int(len(dataset) * val_fraction)
                n_train = len(dataset) - n_val
                train_ds, val_ds = torch.utils.data.random_split(
                    dataset, [n_train, n_val],
                    generator=torch.Generator().manual_seed(1234),
                )
                train_datasets.append(train_ds)
                val_datasets.append(val_ds)
            else:
                train_datasets.append(
                    FastMRIDataset(
                        root=train_dir,
                        anatomy=anatomy,
                        image_shape=image_shape,
                        transform=transform,
                        num_coils=num_coils,
                        min_energy_fraction=min_energy_fraction,
                    )
                )
                val_datasets.append(
                    FastMRIDataset(
                        root=val_dir,
                        anatomy=anatomy,
                        image_shape=image_shape,
                        transform=transform,
                        num_coils=num_coils,
                        min_energy_fraction=min_energy_fraction,
                    )
                )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_collate_fastmri_batch,
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_datasets),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_datasets),
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader
