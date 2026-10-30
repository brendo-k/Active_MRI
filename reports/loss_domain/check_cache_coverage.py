"""Every train and val slice of the local v1 FLAIR + knee manifest has a cached ESPIRiT image with the loader's
signature (a missing slice would raise KeyError mid-training).

    python reports/loss_domain/check_cache_coverage.py [--cache DIR] [Hydra-style root overrides as KEY=VALUE]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from activemri.datasets.complex_image_cache import ComplexImageCache  # noqa: E402
from activemri.datasets.fastmri_data import build_manifest_dataloaders  # noqa: E402

ROOTS = {'local_batch2': '/export/scratch/bkadota/fastmri_knee/batch2', 'hpc': '/export/scratch/bkadota/fastmri_raw/{anatomy}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='/export/scratch/bkadota/fastmri_complex_espirit')
    parser.add_argument('roots', nargs='*', help='source=path overrides of the manifest roots')
    args = parser.parse_args()
    roots = dict(ROOTS, **dict(item.split('=', 1) for item in args.roots))
    loaders = build_manifest_dataloaders(
        manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=roots, classes=['FLAIR', 'knee'],
        image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=0, complex_cache_dir=args.cache)
    cache = ComplexImageCache(Path(args.cache).expanduser())
    missing, total = [], 0
    for split, loader in zip(('train', 'val'), loaders):
        for dataset in loader.dataset.datasets:
            by_volume = {}
            for fname, slice_id in dataset.examples:
                by_volume.setdefault(fname, []).append(slice_id)
            for fname, slice_ids in by_volume.items():
                cached = cache.cached_slices(dataset.anatomy, fname, dataset.complex_signature)
                missing += [(split, fname.name, s) for s in slice_ids if s not in cached]
                total += len(slice_ids)
    print(f'{total} slices checked, {len(missing)} missing')
    for row in missing[:20]:
        print('  missing', row)
    raise SystemExit(1 if missing else 0)


if __name__ == '__main__':
    main()
