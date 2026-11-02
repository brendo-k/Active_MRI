"""Cache one ESPIRiT complex image per slice of the saved fastMRI volumes (resumable).

Walks every .h5 under the raw trees, serves k-space through FastMRIDataset (readout crop, 320 grid,
20 coils, the loader's center-RSS norm), runs the anatomy preset in bart_ecalib.PRESETS, coil-combines,
and stores the image. Maps are discarded.

    python scripts/data/build_complex_image_cache.py
    python scripts/data/build_complex_image_cache.py --verify 8
"""
import argparse
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

import _path  # noqa: F401
from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature
from activemri.datasets.fastmri_data import FastMRIDataset
from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps
from activemri.utils.image_processing import ifft_2d_img

RAW_ROOTS = (
    pathlib.Path('/export/scratch/bkadota/fastmri_raw'),
    pathlib.Path('/export/scratch/bkadota/fastmri_knee'),
)
DEFAULT_CACHE = pathlib.Path('/export/scratch/bkadota/fastmri_complex_espirit_v2')


def discover(roots):
    """(anatomy, directory) -> file names, for every saved volume."""
    groups = {}
    for root in roots:
        for path in sorted(root.rglob('*.h5')):
            anatomy = 'brain' if 'brain' in path.parts else 'knee'
            groups.setdefault((anatomy, path.parent), []).append(path.name)
    return groups


def datasets_for(roots, image_shape, num_coils, min_energy_fraction):
    found = []
    for (anatomy, directory), names in discover(roots).items():
        found.append(FastMRIDataset(
            directory, anatomy, image_shape, num_coils=num_coils,
            min_energy_fraction=min_energy_fraction, file_names=names, readout_crop=True,
        ))
    return found


def combine(kspace, anatomy):
    maps, _ = ecalib_sense_maps(kspace, **PRESETS[anatomy])
    return (ifft_2d_img(kspace) * maps.conj()).sum(0)


def estimate(job):
    dataset, anatomy, fname, index, slice_id = job
    kspace = torch.view_as_complex(dataset[index]['kspace'].contiguous())
    try:
        image = combine(kspace, anatomy).numpy()
    except Exception as err:
        return anatomy, fname, slice_id, None, f'{err}'
    return anatomy, fname, slice_id, image, None


def build(datasets, cache, workers, log=print):
    failures = []
    jobs = []
    signatures = {}
    for dataset in datasets:
        anatomy = dataset.anatomy
        signature = cache_signature(anatomy, dataset.image_shape, dataset.num_coils, dataset.min_energy_fraction)
        by_volume = {}
        for index, (fname, slice_id) in enumerate(dataset.examples):
            by_volume.setdefault(fname, []).append((index, slice_id))
        pending = 0
        for fname, items in by_volume.items():
            signatures[(anatomy, fname)] = signature
            done = cache.cached_slices(anatomy, fname, signature)
            for index, slice_id in items:
                if slice_id not in done:
                    jobs.append((dataset, anatomy, fname, index, slice_id))
                    pending += 1
        log(f'[{anatomy}] {dataset.root}: {len(by_volume)} volumes, {len(dataset.examples)} slices, {pending} to estimate')

    log(f'{len(jobs)} slices to estimate with {workers} workers')
    pending_images = {}
    done_count = 0

    def flush():
        for (anatomy, fname), images in pending_images.items():
            cache.write_volume(anatomy, fname, signatures[(anatomy, fname)], images)
        pending_images.clear()

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(estimate, job) for job in jobs]
        for fut in as_completed(futures):
            anatomy, fname, slice_id, image, err = fut.result()
            done_count += 1
            if err is not None:
                failures.append((anatomy, pathlib.Path(fname).name, slice_id, err))
            else:
                pending_images.setdefault((anatomy, fname), {})[slice_id] = image
            if done_count % 64 == 0:
                flush()
                log(f'{done_count}/{len(jobs)} slices, {len(failures)} failed')
    flush()
    log(f'{done_count}/{len(jobs)} slices, {len(failures)} failed')
    return failures


def verify(datasets, cache, count):
    import numpy as np
    rng = np.random.RandomState(0)
    for _ in range(count):
        dataset = datasets[rng.randint(len(datasets))]
        index = rng.randint(len(dataset))
        fname, slice_id = dataset.examples[index]
        signature = cache_signature(dataset.anatomy, dataset.image_shape, dataset.num_coils, dataset.min_energy_fraction)
        cached = cache.load(dataset.anatomy, fname, slice_id, signature)
        kspace = torch.view_as_complex(dataset[index]['kspace'].contiguous())
        fresh = combine(kspace, dataset.anatomy)
        gap = (cached - fresh).abs().max()
        print(f'{dataset.anatomy} {fname.name} slice {slice_id}: max |cached - fresh| {float(gap):.3e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache_dir', type=pathlib.Path, default=DEFAULT_CACHE)
    parser.add_argument('--roots', type=pathlib.Path, nargs='+', default=list(RAW_ROOTS))
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 320])
    parser.add_argument('--num_coils', type=int, default=20)
    parser.add_argument('--min_energy_fraction', type=float, default=0.2)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--verify', type=int, default=0, help='recompute this many random cached slices and report the difference')
    args = parser.parse_args()

    datasets = datasets_for(args.roots, args.resolution, args.num_coils, args.min_energy_fraction)
    cache = ComplexImageCache(args.cache_dir)
    if args.verify:
        verify(datasets, cache, args.verify)
        return
    failures = build(datasets, cache, args.workers)
    if failures:
        print(f'{len(failures)} slices failed')
        for row in failures[:20]:
            print(' ', row)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
