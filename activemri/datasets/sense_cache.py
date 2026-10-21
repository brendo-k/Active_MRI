"""On-disk cache of BART ecalib sensitivity maps, one HDF5 per volume, estimated from the k-space a dataset actually serves.

Maps depend on the dataset's preprocessing (decimation, crop, coil crop/pad, flip, normalisation), so the cache is built by
iterating the dataset itself and stamped with a signature (estimator preset, image shape, coil count); a mismatch on load raises
instead of silently serving maps for a different pipeline.
"""
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch

from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps

CACHE_VERSION = 1


def cache_signature(anatomy: str, image_shape: Sequence[int], num_coils: int) -> str:
    return json.dumps({'version': CACHE_VERSION, 'estimator': 'bart-ecalib', 'preset': PRESETS[anatomy],
                       'image_shape': list(image_shape), 'num_coils': num_coils}, sort_keys=True)


class SenseCache:
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
        """Maps [C, H, W] complex64 exactly as estimated (norm 1 inside the object, tapering to 0 at soft-SENSE edges, 0 outside and for padded coils)."""
        path = self.volume_path(anatomy, fname)
        if not path.exists():
            raise FileNotFoundError(f'no cached sense maps at {path}; build them with scripts/data/build_sense_cache.py')
        with h5py.File(path, 'r', locking=False) as f:
            if f.attrs['signature'] != signature:
                raise ValueError(f'{path} was built with a different pipeline: {f.attrs["signature"]} != {signature}')
            key = f'slice_{slice_id}'
            if key not in f:
                raise KeyError(f'slice {slice_id} missing from {path}')
            packed = torch.from_numpy(f[key][()].astype(np.float32))
        return torch.complex(packed[0], packed[1])

    def write_volume(self, anatomy: str, fname, signature: str, maps: Dict[int, np.ndarray]) -> None:
        """Adds `maps` ({slice_id: [C, H, W] complex}) to the volume's file, atomically."""
        path = self.volume_path(anatomy, fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        existing = {}
        if path.exists():
            with h5py.File(path, 'r', locking=False) as f:
                existing = {name: f[name][()] for name in f if f.attrs['signature'] == signature}
        with h5py.File(tmp, 'w') as f:
            f.attrs['signature'] = signature
            for name, packed in existing.items():
                f.create_dataset(name, data=packed, compression='gzip', compression_opts=4, shuffle=True)
            for slice_id, m in maps.items():
                packed = np.stack([m.real, m.imag]).astype(np.float16)
                f.create_dataset(f'slice_{slice_id}', data=packed, compression='gzip', compression_opts=4, shuffle=True)
        tmp.replace(path)


def build_cache(datasets: List, cache: SenseCache, workers: int = 24, log=print) -> None:
    """Estimates and stores maps for every example of every dataset (a `FastMRIDataset`, without a cache attached)."""
    for dataset in datasets:
        anatomy = dataset.anatomy
        signature = cache_signature(anatomy, dataset.image_shape, dataset.num_coils)
        by_volume: Dict[pathlib.Path, List] = {}
        for index, (fname, slice_id) in enumerate(dataset.examples):
            by_volume.setdefault(fname, []).append((index, slice_id))
        for number, (fname, items) in enumerate(by_volume.items(), 1):
            done = cache.cached_slices(anatomy, fname, signature)
            todo = [(i, s) for i, s in items if s not in done]
            if not todo:
                continue

            def estimate(item):
                index, slice_id = item
                kspace = torch.view_as_complex(dataset[index]['kspace'].contiguous())
                maps, _ = ecalib_sense_maps(kspace, **PRESETS[anatomy])
                return slice_id, maps.numpy()

            with ThreadPoolExecutor(workers) as pool:
                results = dict(pool.map(estimate, todo))
            cache.write_volume(anatomy, fname, signature, results)
            log(f'[{anatomy}] volume {number}/{len(by_volume)} {pathlib.Path(fname).name}: {len(todo)} slices')
