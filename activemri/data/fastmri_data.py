import math
import pathlib
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
    ):
        self.root = pathlib.Path(root)
        self.anatomy = anatomy
        self.image_shape = image_shape
        self.transform = transform
        self.num_coils = num_coils

        self.examples: List[Tuple[pathlib.Path, int]] = []
        rng = np.random.RandomState(1234)

        files = sorted(
            f for f in self.root.iterdir()
            if f.suffix in ('.h5', '.hdf5')
        )

        if num_volumes is not None:
            rng.shuffle(files)
            files = sorted(files[:num_volumes])

        for fname in files:
            with h5py.File(fname, 'r') as data:
                if 'kspace' not in data:
                    continue
                num_slices = data['kspace'].shape[0]
                self.examples += [(fname, sid) for sid in range(num_slices)]

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
            kspace = torch.from_numpy(data['kspace'][slice_id])
            attrs = dict(data.attrs)

        # Downsample the largest spatial dimension by 2 before cropping
        if kspace.shape[-2] > kspace.shape[-1]:
            kspace = kspace[..., ::2, :]
        elif kspace.shape[-1] > kspace.shape[-2]:
            kspace = kspace[..., ::2]

        kspace = self._center_crop(kspace, self.image_shape)
        kspace = self._pad_or_crop_coils(kspace)

        if self.anatomy == 'brain':
            kspace = kspace.flip(-2)

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
    """
    train_datasets = []
    val_datasets = []

    for anatomy in anatomies:
        train_dir, val_dir = _resolve_anatomy_dirs(data_root, anatomy, contrast)

        if train_dir == val_dir:
            dataset = FastMRIDataset(
                root=train_dir,
                anatomy=anatomy,
                image_shape=image_shape,
                transform=transform,
                num_coils=num_coils,
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
                )
            )
            val_datasets.append(
                FastMRIDataset(
                    root=val_dir,
                    anatomy=anatomy,
                    image_shape=image_shape,
                    transform=transform,
                    num_coils=num_coils,
                )
            )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
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
