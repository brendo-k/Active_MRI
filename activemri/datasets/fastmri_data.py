import json
import math
import os
import pathlib
import warnings
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data

from activemri.datasets.sense_cache import SenseCache, cache_signature

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


def flip_rows_about_centre(tensor: torch.Tensor) -> torch.Tensor:
    """Reflects row i to (H - i) mod H, keeping row H // 2 (the DC row of a centred FFT) in place. In k-space this flips
    the image upside down with no phase change; a plain flip on an even grid moves DC to row H // 2 - 1, which puts a
    one-cycle-per-FOV phase ramp on the image."""
    return torch.roll(tensor.flip(-2), 1, -2)


def crop_readout_oversampling(kspace: torch.Tensor) -> torch.Tensor:
    """Drop the 2x readout oversampling (axis -2, DC at the centre) by keeping the central half of the image along the readout.

    A 1D transform only, so phase-encode columns the scanner never acquired stay exactly zero.
    """
    readout = kspace.shape[-2]
    image = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(kspace, dim=-2), dim=-2, norm='ortho'), dim=-2)
    image = image[..., readout // 4: readout // 4 + readout // 2, :]
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(image, dim=-2), dim=-2, norm='ortho'), dim=-2)


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
        sense_cache=None,
        readout_crop: bool = False,
    ):
        """`readout_crop` removes the readout oversampling by an image crop (`crop_readout_oversampling`) for every
        anatomy, instead of decimating the longer axis of non-knee k-space by 2; precomputed 'sense' maps are then not loaded."""
        self.root = pathlib.Path(root)
        self.readout_crop = readout_crop
        self.sense_cache = sense_cache
        self.sense_signature = cache_signature(anatomy, image_shape, num_coils) if sense_cache is not None else None
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
        # Zero-pad a dimension that is smaller than the target (e.g. brain files with a 264-wide phase axis);
        # a negative start index would otherwise slice from the far end and return a narrow, ragged batch element.
        out = data.new_zeros(*data.shape[:-2], *shape)
        src, dst = [], []
        for n, target in zip(data.shape[-2:], shape):
            offset = (n - target) // 2
            src.append(slice(max(offset, 0), max(offset, 0) + min(n, target)))
            dst.append(slice(max(-offset, 0), max(-offset, 0) + min(n, target)))
        out[..., dst[0], dst[1]] = data[..., src[0], src[1]]
        return out

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
            if 'sense' in data and not self.readout_crop:
                raw = data['sense']
                if self.anatomy == 'prostate':
                    # (averages, slices, coils, H, W) — average across the averages axis
                    espirit_maps = torch.from_numpy(raw[:, slice_id].mean(axis=0))
                else:
                    espirit_maps = torch.from_numpy(raw[slice_id])

        if self.readout_crop:
            kspace = crop_readout_oversampling(kspace)
        # Downsample the largest spatial dimension by 2 before cropping
        elif self.anatomy not in ('knee', 'prostate'):
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
            kspace = flip_rows_about_centre(kspace)
            if espirit_maps is not None:
                espirit_maps = flip_rows_about_centre(espirit_maps)

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

        if self.sense_cache is not None:
            sample['sense_maps'] = self.sense_cache.load(self.anatomy, fname, slice_id, self.sense_signature)

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


def _flat_split(files: List[pathlib.Path], val_fraction: float, test_fraction: float, seed: int):
    """(train, val, test) volumes of a flat directory. The test volumes are drawn first, with their own seed, so
    they are the same whatever `val_fraction` is; val is then a fraction of what remains."""
    rest, test = _split_volumes(files, test_fraction, seed + 1) if test_fraction else (list(files), [])
    train, val = _split_volumes(rest, val_fraction, seed)
    return train, val, test


def held_out_test_volumes(
    data_root: pathlib.Path,
    anatomies: List[str],
    contrast: Optional[str],
    test_fraction: float,
    seed: int = 1234,
) -> Dict[str, List[str]]:
    """File names of the held-out test volumes per condition class (brain, knee-PD, knee-PDFS), exactly the
    volumes `build_fastmri_dataloaders(..., test_fraction=...)` keeps out of train and val."""
    out = {}
    for anatomy in anatomies:
        train_dir, val_dir = _resolve_anatomy_dirs(data_root, anatomy, contrast)
        if train_dir != val_dir:
            raise ValueError(f"{anatomy} has a train/val split on disk; a test hold-out needs a flat directory")
        _, _, test = _flat_split(list_anatomy_files(train_dir, anatomy), 0.0, test_fraction, seed)
        for group, group_files in _group_files(anatomy, test, 'condition').items():
            out[group] = sorted(f.name for f in group_files)
    return out


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


def _group_files(anatomy: str, files: List[pathlib.Path], balance_by: str) -> Dict[str, List[pathlib.Path]]:
    """The balancing groups of one anatomy's files: the anatomy itself, or for knee under `balance_by='condition'` its protocols (PD / PDFS)."""
    if balance_by == 'anatomy' or anatomy != 'knee':
        return {anatomy: files}
    if balance_by != 'condition':
        raise ValueError(f"balance_by must be 'anatomy' or 'condition', got '{balance_by}'")
    groups: Dict[str, List[pathlib.Path]] = {}
    for f in files:
        with h5py.File(f, 'r') as data:
            groups.setdefault(condition_name(anatomy, dict(data.attrs)), []).append(f)
    return groups


def _build_balanced_datasets(
    data_root: pathlib.Path,
    anatomies: List[str],
    contrast: Optional[str],
    val_fraction: float,
    seed: int = 1234,
    train_slices: Optional[Dict[str, int]] = None,
    balance_by: str = 'anatomy',
    test_fraction: float = 0.0,
    trim_to_smallest: bool = True,
    **dataset_kwargs,
) -> Tuple[List[FastMRIDataset], List[FastMRIDataset]]:
    """Per-anatomy train/val datasets with an equal number of slices per anatomy.

    With `balance_by='condition'` the groups are the conditions instead (brain, knee-PD, knee-PDFS):
    each knee protocol gets as many slices as brain. Volumes are split into train/val exactly as for
    `balance_by='anatomy'` and only then divided by protocol, so a volume never changes side.

    `train_slices` overrides the train slice count of the named anatomies, so a
    single-anatomy run can use as many slices as the full multi-anatomy train set.
    Val sets stay balanced.

    Volumes are pre-selected from HDF5 shape metadata so the (slow) energy filter
    only scans volumes that are kept, then the larger anatomies are trimmed to the
    exact slice count of the smallest. Anatomies without a train/val split are
    split by volume.
    """
    files = {'train': {}, 'val': {}}
    group_anatomy = {}
    for anatomy in anatomies:
        train_dir, val_dir = _resolve_anatomy_dirs(data_root, anatomy, contrast)
        if train_dir == val_dir:
            train_files, val_files, _ = _flat_split(list_anatomy_files(train_dir, anatomy), val_fraction, test_fraction, seed)
            roots = {'train': train_dir, 'val': train_dir}
        else:
            if test_fraction:
                raise ValueError(f"{anatomy} has a train/val split on disk; a test hold-out needs a flat directory")
            train_files, val_files = list_anatomy_files(train_dir, anatomy), list_anatomy_files(val_dir, anatomy)
            roots = {'train': train_dir, 'val': val_dir}
        for split, split_files in (('train', train_files), ('val', val_files)):
            for group, group_files in _group_files(anatomy, split_files, balance_by).items():
                files[split][group] = (roots[split], group_files)
                group_anatomy[group] = anatomy

    if balance_by == 'condition' and train_slices:
        raise ValueError("train_slices is not supported with balance_by='condition'")
    train_slices = train_slices or {}
    datasets = {}
    for split in ('train', 'val'):
        raw_totals = {
            g: sum(_slice_count(f, group_anatomy[g]) for f in split_files)
            for g, (_, split_files) in files[split].items()
        }
        target = min(raw_totals.values())
        built = {}
        overridden = {a: n for a, n in train_slices.items() if split == 'train' and a in files[split]}
        for group, (root, split_files) in files[split].items():
            # The energy filter drops slices, so an overridden anatomy starts from all its volumes and is trimmed after.
            names = [f.name for f in split_files] if not trim_to_smallest else _pick_volumes(split_files, group_anatomy[group], raw_totals[group] if group in overridden else target, seed)
            built[group] = FastMRIDataset(
                root=root, anatomy=group_anatomy[group], file_names=names, **dataset_kwargs,
            )
        for anatomy, ds in built.items():
            if anatomy in overridden and len(ds) < overridden[anatomy]:
                raise ValueError(
                    f"{anatomy} train has {len(ds)} slices after filtering, fewer than the {overridden[anatomy]} requested"
                )
        balanced = [ds for a, ds in built.items() if a not in overridden]
        num_examples = min(len(ds) for ds in balanced)
        for anatomy, ds in built.items() if trim_to_smallest else ():
            _trim_examples(ds, overridden.get(anatomy, num_examples), seed)
        counts = {a: (len({e[0] for e in ds.examples}), len(ds)) for a, ds in built.items()}
        print(f"{'Balanced' if trim_to_smallest else 'Unbalanced'} {split} set (volumes, slices): {counts}")
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
    sense_cache_dir: Optional[str] = None,
    balance_by: str = 'anatomy',
    test_fraction: float = 0.0,
    trim_to_smallest: bool = True,
    shuffle_seed: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and val dataloaders from one or more anatomies.

    Args:
        shuffle_seed: Seeds the train loader's own shuffle generator, so the batch order depends on this seed alone and
            not on how much of the global RNG model construction used. None shuffles from the global RNG.
        data_root: Root directory containing anatomy subdirs (e.g. ~/data/fastmri/).
        anatomies: List of anatomy names (brain, knee, breast, prostate).
        image_shape: Target image dimensions [H, W].
        batch_size: Batch size for dataloaders.
        transform: Optional transform applied to each sample.
        num_coils: Number of coils to pad/crop to.
        num_workers: DataLoader workers.
        val_fraction: Fraction of data for validation when no train/val split exists.
        trim_to_smallest: With balance, False keeps every volume of every group instead of trimming the larger
            groups to the smallest one's slice count (the splits and condition classes are unchanged).
        test_fraction: With balance, fraction of each anatomy's volumes held out of train and val (never loaded
            here; `held_out_test_volumes` lists them). Drawn before the val split, so val is a fraction of the rest.
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
            sense_cache=SenseCache(sense_cache_dir) if sense_cache_dir else None,
            balance_by=balance_by,
            test_fraction=test_fraction,
            trim_to_smallest=trim_to_smallest,
        )
    elif data_dir is not None:
        if sense_cache_dir:
            raise ValueError("sense_cache_dir is only supported with balance=True")
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
        if sense_cache_dir:
            raise ValueError("sense_cache_dir is only supported with balance=True")
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

    return _dataloaders(train_datasets, val_datasets, batch_size, num_workers, shuffle_seed)


def _dataloaders(train_datasets, val_datasets, batch_size, num_workers, shuffle_seed):
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
        generator=None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed),
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_datasets),
        shuffle=False,
        **val_loader_kwargs,
    )

    return train_loader, val_loader


MANIFEST_TRAIN_TIERS = {'S': ('S',), 'L': ('S', 'L-only')}
LIMIT_VOLUMES_SEED = 1234  # fixed; a training seed must not change which volumes a cap keeps
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _volumes_to_keep(n: int, fraction: float) -> int:
    """Volumes to keep out of `n`. 1 keeps them all. A smaller fraction keeps at least one, rounded to the nearest count."""
    if fraction == 1:
        return n
    return min(n, max(1, int(round(fraction * n))))


def _limit_volume_rows(rows: List[Dict], fraction: Optional[float], seed: int = LIMIT_VOLUMES_SEED) -> List[Dict]:
    """Keep `fraction` of the volumes of each condition. `None` and 1 keep every row.

    Volumes of a condition are sorted by filename, then drawn from one seeded shuffle. The same call returns the same
    files, and a smaller fraction is a prefix of a larger one. Each condition is shuffled on its own, so the files kept
    for one condition do not move when another condition is added or removed.
    """
    if fraction is None:
        return rows
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(f"limit_volumes must be in (0, 1], got {fraction}")
    if fraction == 1:
        return rows
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(condition_name(row['anatomy'], row), []).append(row)
    kept: List[Dict] = []
    for condition in sorted(groups):
        group = sorted(groups[condition], key=lambda row: row['file'])
        rng = np.random.RandomState(zlib.crc32(f"{seed}:{condition}".encode()))
        order = rng.permutation(len(group))
        kept.extend(group[i] for i in order[:_volumes_to_keep(len(group), fraction)])
    return kept


VAL_HALVES = ('all', 'dev', 'holdout')
VAL_HALF_SEED = 20261111


def _val_half_rows(rows: List[Dict], half: str, seed: int = VAL_HALF_SEED) -> List[Dict]:
    """Split the val volumes of each condition into two fixed halves: `dev` to iterate on, `holdout` to confirm on.

    Volumes of a condition are sorted by filename and shuffled once with a seed of their own; `dev` takes the first
    ceil(n / 2), `holdout` the rest. The split never depends on the training seed or on the other conditions.
    """
    if half not in VAL_HALVES:
        raise ValueError(f"val_half must be one of {VAL_HALVES}, got {half!r}")
    if half == 'all':
        return rows
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(condition_name(row['anatomy'], row), []).append(row)
    kept: List[Dict] = []
    for condition in sorted(groups):
        group = sorted(groups[condition], key=lambda row: row['file'])
        order = np.random.RandomState(zlib.crc32(f"{seed}:{condition}".encode())).permutation(len(group))
        cut = (len(group) + 1) // 2
        kept.extend(group[i] for i in (order[:cut] if half == 'dev' else order[cut:]))
    return kept


def _manifest_datasets(rows: List[Dict], roots: Dict[str, str], split: str, **dataset_kwargs) -> List[FastMRIDataset]:
    """One FastMRIDataset per (anatomy, directory) of the manifest rows; prints the volumes and slices per condition."""
    groups: Dict[Tuple[str, pathlib.Path], List[str]] = {}
    for row in rows:
        path = pathlib.Path(roots[row['source']].format(anatomy=row['anatomy'])).expanduser() / row['hpc_relpath']
        groups.setdefault((row['anatomy'], path.parent), []).append(path.name)
    datasets = [FastMRIDataset(root=root, anatomy=anatomy, file_names=names, **dataset_kwargs)
                for (anatomy, root), names in groups.items()]
    condition_of = {row['file']: condition_name(row['anatomy'], row) for row in rows}
    volumes: Dict[str, set] = {}
    slices: Dict[str, int] = {}
    for ds in datasets:
        for fname, _ in ds.examples:
            condition = condition_of[fname.name]
            volumes.setdefault(condition, set()).add(fname.name)
            slices[condition] = slices.get(condition, 0) + 1
    print(f"Manifest {split} set (volumes, slices): { {c: (len(volumes[c]), slices[c]) for c in sorted(slices)} }")
    return datasets


def build_manifest_dataloaders(
    manifest: str,
    roots: Dict[str, str],
    classes: Sequence[str],
    image_shape: List[int],
    batch_size: int,
    num_coils: int,
    train_tier: str = 'S',
    num_workers: int = 4,
    min_energy_fraction: float = 0.2,
    shuffle_seed: Optional[int] = None,
    transform: Optional[Callable] = None,
    train_anatomies: Sequence[str] = (),
    limit_volumes: Optional[float] = None,
    val_half: str = 'all',
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train and val loaders over the raw h5 volumes a subset manifest lists (scripts/data/select_fastmri_subset.py).

    Train is `train_tier` (S, or L = S plus the L-only rows) of each class in `classes` (the manifest's `cls`: knee, T2,
    FLAIR); val is its val rows. Test rows are never loaded. `roots` maps the manifest's `source` to the directory its
    `hpc_relpath` is relative to, with `{anatomy}` filled in. Every slice keeps all its coils (zero-padded to `num_coils`)
    and has its readout oversampling cropped away. A relative `manifest` is relative to the repository root.
    `train_anatomies`, when given, trains on those anatomies' rows alone while val keeps every class.
    `limit_volumes` is the fraction of each condition kept in train and again in val (`_limit_volume_rows`). 1 and
    `None` keep the whole tier. The choice does not depend on `shuffle_seed`. `val_half` keeps one fixed half of each
    condition's val volumes (`_val_half_rows`), after `limit_volumes`.
    """
    import csv
    with open(REPO_ROOT / pathlib.Path(manifest).expanduser()) as f:
        rows = [row for row in csv.DictReader(f) if row['cls'] in classes]
    unknown = set(classes) - {row['cls'] for row in rows}
    if unknown:
        raise ValueError(f"classes {sorted(unknown)} are not in {manifest}")
    most_coils = max(int(row['coils']) for row in rows)
    if most_coils > num_coils:
        raise ValueError(f"{manifest} has volumes with {most_coils} coils; num_coils={num_coils} would drop some")
    unknown = set(train_anatomies) - {row['anatomy'] for row in rows}
    if unknown:
        raise ValueError(f"train_anatomies {sorted(unknown)} are not among the anatomies of classes {list(classes)}")
    train_rows = [row for row in rows if row['split'] == 'train' and row['tier'] in MANIFEST_TRAIN_TIERS[train_tier]
                  and (not train_anatomies or row['anatomy'] in train_anatomies)]
    val_rows = [row for row in rows if row['split'] == 'val']
    if limit_volumes is not None and float(limit_volumes) != 1:
        train_rows = _limit_volume_rows(train_rows, limit_volumes)
        val_rows = _limit_volume_rows(val_rows, limit_volumes)
        print(f"limit_volumes={float(limit_volumes):g} of each condition (seed {LIMIT_VOLUMES_SEED})")
    if val_half != 'all':
        val_rows = _val_half_rows(val_rows, val_half)
        print(f"val_half={val_half} (seed {VAL_HALF_SEED})")
    dataset_kwargs = dict(image_shape=image_shape, transform=transform, num_coils=num_coils,
                          min_energy_fraction=min_energy_fraction, readout_crop=True)
    return _dataloaders(_manifest_datasets(train_rows, roots, 'train', **dataset_kwargs),
                        _manifest_datasets(val_rows, roots, 'val', **dataset_kwargs),
                        batch_size, num_workers, shuffle_seed)
