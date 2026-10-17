import json
import math
import os
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
SLICE_ENERGY_CACHE_DIR = '.slice_energy'


def condition_name(anatomy: str, attrs: Dict) -> str:
    """Conditioning label: the anatomy, split by knee fat saturation (PD vs PDFS)."""
    if anatomy != 'knee':
        return anatomy
    acquisition = attrs.get('acquisition', '')
    return 'knee-PDFS' if 'PDFS' in str(acquisition) else 'knee-PD'


ANATOMY_FILE_MARKERS = {
    'brain': 'brain',
    'prostate': 'prostate',
    'breast': 'breast',
}


def list_anatomy_files(root: pathlib.Path, anatomy: str) -> List[pathlib.Path]:
    """HDF5 files under `root` belonging to `anatomy`, sorted by name.

    Knee files carry no marker, so knee is whatever is left after excluding the
    other anatomies.
    """
    root = pathlib.Path(root)
    entries = [*root.iterdir(), *(f for d in root.iterdir() if d.is_dir() and d.name.endswith('_h5') for f in d.iterdir())]
    files = sorted(f for f in entries if f.suffix in ('.h5', '.hdf5'))
    marker = ANATOMY_FILE_MARKERS.get(anatomy)
    if marker is not None:
        return [f for f in files if marker in f.name]
    return [f for f in files if not any(m in f.name for m in ANATOMY_FILE_MARKERS.values())]


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

        files = list_anatomy_files(self.root, self.anatomy)

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
                        energies = self._cached_slice_energies(fname, data, num_slices)
                        threshold = self.min_energy_fraction * energies.max()
                        valid_slices = [s for s in range(num_slices) if energies[s] >= threshold]
                    else:
                        valid_slices = list(range(num_slices))

                    self.examples += [(fname, sid) for sid in valid_slices]
            except OSError as err:
                warnings.warn(f"Skipping unreadable file {fname}: {err}")

    def _cached_slice_energies(self, fname: pathlib.Path, data: h5py.File, num_slices: int) -> np.ndarray:
        """Per-slice centre energies, cached beside the volume so restarts skip the full k-space read."""
        cache_path = fname.parent / SLICE_ENERGY_CACHE_DIR / f'{fname.stem}.json'
        stat = fname.stat()
        signature = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'anatomy': self.anatomy}
        try:
            cached = json.loads(cache_path.read_text())
            if cached['signature'] == signature and len(cached['energies']) == num_slices:
                return np.array(cached['energies'])
        except (OSError, ValueError, KeyError):
            pass

        energies = []
        for sid in range(num_slices):
            if self.anatomy == 'prostate':
                kslice = data['kspace'][:, sid].sum(axis=0)
            else:
                kslice = data['kspace'][sid]
            energies.append(float(self._center_energy(kslice)))

        try:
            cache_path.parent.mkdir(exist_ok=True)
            tmp_path = cache_path.with_suffix(f'.{os.getpid()}.tmp')
            tmp_path.write_text(json.dumps({'signature': signature, 'energies': energies}))
            tmp_path.replace(cache_path)
        except OSError as err:
            warnings.warn(f"Could not write slice energy cache {cache_path}: {err}")
        return np.array(energies)

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
            'condition': condition_name(self.anatomy, attrs),
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


def _slice_count(fname: pathlib.Path, anatomy: str) -> int:
    with h5py.File(fname, 'r') as data:
        return data['kspace'].shape[1 if anatomy == 'prostate' else 0]


def _split_volumes(files: List[pathlib.Path], val_fraction: float, seed: int):
    """Split whole volumes (not slices) so val slices never share a scan with train slices."""
    shuffled = list(files)
    np.random.RandomState(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _pick_volumes(files: List[pathlib.Path], anatomy: str, target_slices: int, seed: int) -> List[str]:
    """Random volumes of `files` until their raw slice count first reaches `target_slices`."""
    shuffled = list(files)
    np.random.RandomState(seed).shuffle(shuffled)
    picked, total = [], 0
    for fname in shuffled:
        if total >= target_slices:
            break
        picked.append(fname.name)
        total += _slice_count(fname, anatomy)
    return picked


def _trim_examples(dataset: FastMRIDataset, num_examples: int, seed: int) -> None:
    """Keep `num_examples` slices, dropping whole randomly-chosen volumes first."""
    by_volume: Dict[pathlib.Path, list] = {}
    for example in dataset.examples:
        by_volume.setdefault(example[0], []).append(example)
    volumes = list(by_volume)
    np.random.RandomState(seed).shuffle(volumes)
    dataset.examples = [ex for v in volumes for ex in by_volume[v]][:num_examples]


def _build_balanced_datasets(
    data_root: pathlib.Path,
    anatomies: List[str],
    contrast: Optional[str],
    val_fraction: float,
    seed: int = 1234,
    train_slices: Optional[Dict[str, int]] = None,
    **dataset_kwargs,
) -> Tuple[List[FastMRIDataset], List[FastMRIDataset]]:
    """Per-anatomy train/val datasets with an equal number of slices per anatomy.

    `train_slices` overrides the train slice count of the named anatomies, so a
    single-anatomy run can use as many slices as the full multi-anatomy train set.
    Val sets stay balanced.

    Volumes are pre-selected from HDF5 shape metadata so the (slow) energy filter
    only scans volumes that are kept, then the larger anatomies are trimmed to the
    exact slice count of the smallest. Anatomies without a train/val split are
    split by volume.
    """
    files = {'train': {}, 'val': {}}
    for anatomy in anatomies:
        train_dir, val_dir = _resolve_anatomy_dirs(data_root, anatomy, contrast)
        if train_dir == val_dir:
            train_files, val_files = _split_volumes(list_anatomy_files(train_dir, anatomy), val_fraction, seed)
            roots = {'train': train_dir, 'val': train_dir}
        else:
            train_files, val_files = list_anatomy_files(train_dir, anatomy), list_anatomy_files(val_dir, anatomy)
            roots = {'train': train_dir, 'val': val_dir}
        files['train'][anatomy] = (roots['train'], train_files)
        files['val'][anatomy] = (roots['val'], val_files)

    train_slices = train_slices or {}
    datasets = {}
    for split in ('train', 'val'):
        raw_totals = {
            a: sum(_slice_count(f, a) for f in split_files)
            for a, (_, split_files) in files[split].items()
        }
        target = min(raw_totals.values())
        built = {}
        overridden = {a: n for a, n in train_slices.items() if split == 'train' and a in files[split]}
        for anatomy, (root, split_files) in files[split].items():
            # The energy filter drops slices, so an overridden anatomy starts from all its volumes and is trimmed after.
            names = _pick_volumes(split_files, anatomy, raw_totals[anatomy] if anatomy in overridden else target, seed)
            built[anatomy] = FastMRIDataset(
                root=root, anatomy=anatomy, file_names=names, **dataset_kwargs,
            )
        for anatomy, ds in built.items():
            if anatomy in overridden and len(ds) < overridden[anatomy]:
                raise ValueError(
                    f"{anatomy} train has {len(ds)} slices after filtering, fewer than the {overridden[anatomy]} requested"
                )
        balanced = [ds for a, ds in built.items() if a not in overridden]
        num_examples = min(len(ds) for ds in balanced)
        for anatomy, ds in built.items():
            _trim_examples(ds, overridden.get(anatomy, num_examples), seed)
        counts = {a: (len({e[0] for e in ds.examples}), len(ds)) for a, ds in built.items()}
        print(f"Balanced {split} set (volumes, slices): {counts}")
        datasets[split] = list(built.values())

    return datasets['train'], datasets['val']


def _collate_fastmri_batch(samples: List[Dict]) -> Dict:
    """Like default_collate, but keeps 'espirit_maps' as a plain list.

    Not every file ships precomputed ESPIRiT maps, so this field is None for
    some samples and a tensor for others — default_collate can't stack that.
    """
    espirit_maps = [s.pop('espirit_maps') for s in samples]
    batch = torch.utils.data.default_collate(samples)
    batch['espirit_maps'] = espirit_maps
    return batch


MAX_VAL_WORKERS = 4


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
    balance: bool = False,
    train_anatomies: Optional[List[str]] = None,
    train_slices: Optional[int] = None,
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
        balance: If True, give every anatomy the same number of train slices (and
            of val slices) by dropping volumes from the larger ones. Anatomies with
            no train/val split are split by volume rather than by slice.
        train_anatomies: If given, train only on these anatomies while still
            validating on all of `anatomies`. Splits are built over all of
            `anatomies` first, so the kept train slices match a run that trains
            on every anatomy.
        train_slices: With balance and train_anatomies, the number of train slices per
            trained anatomy instead of the balanced count. Val sets are unchanged.
    """
    train_datasets = []
    val_datasets = []

    if balance:
        if data_dir is not None:
            raise ValueError("balance=True is only supported when data_dir is None")
        train_datasets, val_datasets = _build_balanced_datasets(
            data_root, anatomies, contrast, val_fraction,
            train_slices={a: train_slices for a in train_anatomies or []} if train_slices else None,
            image_shape=image_shape,
            transform=transform,
            num_coils=num_coils,
            min_energy_fraction=min_energy_fraction,
        )
    elif data_dir is not None:
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

    if train_anatomies:
        unknown = set(train_anatomies) - set(anatomies)
        if unknown:
            raise ValueError(f"train_anatomies {sorted(unknown)} are not in anatomies {anatomies}")
        train_datasets = [
            ds for ds, anatomy in zip(train_datasets, anatomies) if anatomy in train_anatomies
        ]

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_collate_fastmri_batch,
    )
    val_loader_kwargs = {**loader_kwargs, 'num_workers': min(num_workers, MAX_VAL_WORKERS)}
    if val_loader_kwargs['num_workers'] == 0:
        val_loader_kwargs.update(persistent_workers=False, prefetch_factor=None)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_datasets),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_datasets),
        shuffle=False,
        **val_loader_kwargs,
    )

    return train_loader, val_loader
