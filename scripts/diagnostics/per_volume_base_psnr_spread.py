"""Spread of per-volume base PSNR at fixed R=8: how much reconstruction quality varies between volumes under one
mask and one VarNet (the LOUPE base of subject_coadapt_ceiling_r8.py, scored slices of each volume). Motivates
choosing the budget R per volume rather than the mask shape.

    python scripts/diagnostics/per_volume_base_psnr_spread.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('reports/subject_coadaptation_r8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dirs', default='brain_block_full,knee_block_full', help='folders under reports/subject_coadaptation_r8')
    parser.add_argument('--out', type=Path, default=ROOT / 'per_volume_base_psnr_spread.json')
    return parser.parse_args()


def per_volume_base(folder):
    volumes = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
    return [(v['anatomy'], v['volume'], float(np.mean(v['psnr']['base']))) for v in volumes]


def describe(values):
    values = np.sort(np.asarray(values))
    return {
        'volumes': int(len(values)),
        'min': float(values[0]),
        'p10': float(np.percentile(values, 10)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'max': float(values[-1]),
    }


def main():
    args = parse_args()
    rows = [row for name in args.dirs.split(',') for row in per_volume_base(ROOT / name)]
    summary = {'all': describe([psnr for _, _, psnr in rows])}
    for anatomy in sorted({anatomy for anatomy, _, _ in rows}):
        summary[anatomy] = describe([psnr for a, _, psnr in rows if a == anatomy])
    result = {
        'summary': summary,
        'volumes': [{'anatomy': a, 'volume': v, 'base_psnr': psnr} for a, v, psnr in sorted(rows, key=lambda r: r[2])],
        'config': {'dirs': args.dirs},
    }
    args.out.write_text(json.dumps(result, indent=1))
    for group, stats in summary.items():
        print(f"{group:6s} n={stats['volumes']:2d}  min {stats['min']:.1f}  p10 {stats['p10']:.1f}  "
              f"median {stats['median']:.1f}  p90 {stats['p90']:.1f}  max {stats['max']:.1f} dB")
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
