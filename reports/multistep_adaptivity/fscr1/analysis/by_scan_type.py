"""fscr1's paired contrasts split by scan type (post hoc, not pre-registered): dev volumes whose k-space is narrower
than the 320-column grid against full-width ones, and 1.5 T against 3 T, each with the scorer's volume-cluster bootstrap.

    python reports/multistep_adaptivity/fscr1/analysis/by_scan_type.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from activemri.utils.multistep import cluster_bootstrap  # noqa: E402

FSCR1 = Path(__file__).resolve().parents[1]
SCORES = FSCR1 / 'scores'
MANIFEST = REPO / 'configs/datasets/fastmri_local_v1/manifest.csv'
GRID_COLUMNS = 320
ARMS = {f'{arm}_{tier}': f'{arm}_{steps}_s42_fscr1_{tier}'
        for arm, steps in (('F', 4), ('Fb', 4), ('L', 1)) for tier in ('tF', 'tS5')}
CONTRASTS = {
    'F-L tF': [(1, 'F_tF'), (-1, 'L_tF')],
    'F-Fb tF': [(1, 'F_tF'), (-1, 'Fb_tF')],
    'Fb-L tF': [(1, 'Fb_tF'), (-1, 'L_tF')],
    'F-L tS5': [(1, 'F_tS5'), (-1, 'L_tS5')],
    'F-Fb tS5': [(1, 'F_tS5'), (-1, 'Fb_tS5')],
    '(F-L) tF-tS5': [(1, 'F_tF'), (-1, 'L_tF'), (-1, 'F_tS5'), (1, 'L_tS5')],
}


def per_slice_mean(values):
    """Mean over draw seeds, NaN where every seed is NaN (a slice with no donor), as the scorer takes it."""
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    return np.where(count > 0, np.where(finite, values, 0.0).sum(axis=-1) / np.maximum(count, 1), np.nan)


def main():
    scans = {row['file']: row for row in csv.DictReader(MANIFEST.open())}
    scores = {arm: np.load(SCORES / f'{run}.npz', allow_pickle=False) for arm, run in ARMS.items()}
    volumes = scores['F_tF']['volume']
    if any((s['volume'] != volumes).any() for s in scores.values()):
        raise SystemExit('the arms were scored on different slices')
    columns = np.array([int(scans[v]['kspace_shape'].strip('[]').split(',')[-1]) for v in volumes])
    field = np.array([scans[v]['field'] for v in volumes])
    groups = {
        'narrower than the grid': columns < GRID_COLUMNS,
        'full width': columns == GRID_COLUMNS,
        '1.5T': field == '1.5T',
        '3T': field == '3T',
        'all': np.ones(len(volumes), dtype=bool),
    }
    psnr = {arm: per_slice_mean(s['psnr']) for arm, s in scores.items()}
    swap_cost = per_slice_mean(scores['F_tF']['psnr']) - per_slice_mean(scores['F_tF']['swap_psnr'])
    strata = np.zeros(len(volumes), dtype=int)
    results = {}
    for group, inside in groups.items():
        row = {'n_volumes': int(len(np.unique(volumes[inside]))), 'n_slices': int(inside.sum()),
               'L_tF_psnr': float(psnr['L_tF'][inside].mean())}
        for label, terms in CONTRASTS.items():
            delta = sum(sign * psnr[arm] for sign, arm in terms)
            row[label] = cluster_bootstrap(delta[inside], volumes[inside], strata[inside])
        row['F_tF swap cost'] = cluster_bootstrap(swap_cost[inside], volumes[inside], strata[inside])
        results[group] = row
    labels = [*CONTRASTS, 'F_tF swap cost']
    print(f"{'group':24s} {'vols':>4s} {'L_tF':>6s} " + ' '.join(f'{label:>23s}' for label in labels))
    for group, row in results.items():
        cells = ' '.join(f"{row[label]['mean']:+.3f} [{row[label]['ci95'][0]:+.3f},{row[label]['ci95'][1]:+.3f}]"
                         for label in labels)
        print(f"{group:24s} {row['n_volumes']:4d} {row['L_tF_psnr']:6.2f} {cells}")
    (FSCR1 / 'by_scan_type.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
