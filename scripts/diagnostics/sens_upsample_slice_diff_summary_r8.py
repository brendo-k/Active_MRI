"""Volume clustering and tail summary of the per-slice conv vs bilinear PSNR differences.

Reads reports/sens_upsample_r8/slice_diff_psnr.npz (written by sens_upsample_slice_diff_r8.py).

    python scripts/diagnostics/sens_upsample_slice_diff_summary_r8.py
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path('reports/sens_upsample_r8')
THRESHOLDS = (0.1, 0.25, 0.5)


def main():
    data = np.load(OUT_DIR / 'slice_diff_psnr.npz')
    delta = data['bilinear'] - data['conv']
    anatomy, volume, condition = data['anatomy'], data['volume'], data['condition']
    result = {}
    for name in np.unique(anatomy):
        rows = anatomy == name
        d, v = delta[rows], volume[rows]
        volumes = np.unique(v)
        volume_mean = np.array([d[v == u].mean() for u in volumes])
        within = np.concatenate([d[v == u] - d[v == u].mean() for u in volumes])
        order = np.argsort(volume_mean)
        result[name] = {
            'slices': int(rows.sum()), 'volumes': int(len(volumes)),
            'frac_abs_gt': {str(t): float((np.abs(d) > t).mean()) for t in THRESHOLDS},
            'frac_positive': float((d > 0).mean()),
            'total_var': float(d.var()), 'between_volume_var_share': float(volume_mean[np.searchsorted(volumes, v)].var() / d.var()),
            'volume_mean_sd': float(volume_mean.std()), 'within_volume_sd': float(within.std()),
            'volumes_worst': [(Path(volumes[i]).name, float(volume_mean[i])) for i in order[:3]],
            'volumes_best': [(Path(volumes[i]).name, float(volume_mean[i])) for i in order[::-1][:3]],
        }
    knee_cond = {c: float((delta[condition == c]).mean()) for c in np.unique(condition)}
    result['condition_mean'] = knee_cond
    (OUT_DIR / 'slice_diff_summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
