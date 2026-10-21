"""Build the BART ecalib sensitivity-map cache for the train and val slices a config trains on (resumable).

    python scripts/data/build_sense_cache.py --cache_dir /data/yanglab2/labmembers2/bkadota/data/fastmri/sense_cache_ecalib
    python scripts/data/build_sense_cache.py --cache_dir ... --verify 8     # recompute 8 random cached slices and compare
"""
import argparse
import pathlib

import numpy as np
import torch

import _path  # noqa: F401
from activemri.datasets.fastmri_data import _build_balanced_datasets
from activemri.datasets.sense_cache import SenseCache, build_cache, cache_signature
from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--data_root', default='/home/bkadota/data/fastmri')
    parser.add_argument('--anatomies', nargs='+', default=['brain', 'knee'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--num_coils', type=int, default=16)
    parser.add_argument('--min_energy_fraction', type=float, default=0.2)
    parser.add_argument('--balance_by', default='anatomy', choices=['anatomy', 'condition'], help='must match datasets.balance_by of the config that will use the cache')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--verify', type=int, default=0, help='instead of building, recompute this many random cached slices and report the difference')
    args = parser.parse_args()

    train, val = _build_balanced_datasets(pathlib.Path(args.data_root), args.anatomies, args.contrast, 0.2, image_shape=args.resolution,
                                          transform=None, num_coils=args.num_coils, min_energy_fraction=args.min_energy_fraction,
                                          balance_by=args.balance_by)
    datasets, cache = train + val, SenseCache(args.cache_dir)
    if not args.verify:
        build_cache(datasets, cache, args.workers)
        return
    rng = np.random.RandomState(0)
    for _ in range(args.verify):
        dataset = datasets[rng.randint(len(datasets))]
        index = rng.randint(len(dataset))
        fname, slice_id = dataset.examples[index]
        cached = cache.load(dataset.anatomy, fname, slice_id, cache_signature(dataset.anatomy, dataset.image_shape, dataset.num_coils))
        fresh, _ = ecalib_sense_maps(torch.view_as_complex(dataset[index]['kspace'].contiguous()), **PRESETS[dataset.anatomy])
        kept = (fresh.abs().pow(2).sum(0) > 0.5) & (cached.abs().pow(2).sum(0) > 0.5)
        direction = (cached.conj() * fresh).sum(0).abs() / (cached.abs().pow(2).sum(0).sqrt() * fresh.abs().pow(2).sum(0).sqrt()).clamp(min=1e-8)
        support_flips = ((cached.abs().pow(2).sum(0) > 0.5) != (fresh.abs().pow(2).sum(0) > 0.5)).float().mean()
        print(f'{dataset.anatomy} {fname.name} slice {slice_id}: min per-pixel direction agreement {float(direction[kept].min()):.5f}, '
              f'max |sumsq difference| {float((cached.abs().pow(2).sum(0) - fresh.abs().pow(2).sum(0)).abs().max()):.5f}, support pixels flipped {float(support_flips):.5f}')


if __name__ == '__main__':
    main()
