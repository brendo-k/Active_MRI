"""On-disk cache of one ESPIRiT coil-combined complex image per slice.

Built from the k-space a ``FastMRIDataset`` actually serves (readout crop, shape, coil pad, flip,
normalisation). The sensitivity maps are not stored. A signature mismatch on load raises.
"""
import json
import os
import pathlib
from typing import Dict, Sequence

import h5py
import numpy as np
import torch

from activemri.utils.bart_ecalib import PRESETS

CACHE_VERSION = 2  # 2: brain rows flipped about the centre (DC stays on row H // 2)


def cache_signature(anatomy: str, image_shape: Sequence[int], num_coils: int, min_energy_fraction: float) -> str:
    return json.dumps({
        'version': CACHE_VERSION,
        'kind': 'espirit-complex-image',
        'preset': PRESETS[anatomy],
        'image_shape': list(image_shape),
        'num_coils': num_coils,
        'readout_crop': True,
        'min_energy_fraction': min_energy_fraction,
        'norm': 'loader-center-rss',
    }, sort_keys=True)


class ComplexImageCache:
    def __init__(self, root):
        self.root = pathlib.Path(root)

    def volume_path(self, anatomy: str, fname) -> pathlib.Path:
        return self.root / anatomy / f'{pathlib.Path(fname).stem}.h5'

    def cached_slices(self, anatomy: str, fname, signature: str) -> set:
        path = self.volume_path(anatomy, fname)
        if not path.exists():
            return set()
        with h5py.File(path, 'r', locking=False) as f:
            if f.attrs['signature'] != signature:
                raise ValueError(f'{path} was built with a different pipeline: {f.attrs["signature"]} != {signature}')
            return {int(name.split('_')[1]) for name in f}

    def load(self, anatomy: str, fname, slice_id: int, signature: str) -> torch.Tensor:
        """Complex image [H, W] complex64."""
        path = self.volume_path(anatomy, fname)
        if not path.exists():
            raise FileNotFoundError(f'no cached complex image at {path}; build it with scripts/data/build_complex_image_cache.py')
        with h5py.File(path, 'r', locking=False) as f:
            if f.attrs['signature'] != signature:
                raise ValueError(f'{path} was built with a different pipeline: {f.attrs["signature"]} != {signature}')
            key = f'slice_{slice_id}'
            if key not in f:
                raise KeyError(f'slice {slice_id} missing from {path}')
            packed = torch.from_numpy(f[key][()].astype(np.float32))
        return torch.complex(packed[0], packed[1])

    def write_volume(self, anatomy: str, fname, signature: str, images: Dict[int, np.ndarray]) -> None:
        """Adds `images` ({slice_id: [H, W] complex}) to the volume's file, atomically."""
        path = self.volume_path(anatomy, fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        existing = {}
        if path.exists():
            with h5py.File(path, 'r', locking=False) as f:
                if f.attrs['signature'] != signature:
                    raise ValueError(f'{path} was built with a different pipeline: {f.attrs["signature"]} != {signature}')
                existing = {name: f[name][()] for name in f}
        with h5py.File(tmp, 'w') as f:
            f.attrs['signature'] = signature
            for name, packed in existing.items():
                f.create_dataset(name, data=packed, compression='gzip', compression_opts=4, shuffle=True)
            for slice_id, image in images.items():
                packed = np.stack([np.asarray(image).real, np.asarray(image).imag]).astype(np.float32)
                f.create_dataset(f'slice_{slice_id}', data=packed, compression='gzip', compression_opts=4, shuffle=True)
        tmp.replace(path)
