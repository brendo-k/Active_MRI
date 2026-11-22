"""Tables and figures for the multistep_adaptivity__seqmri_lr_bias_r8 report, from the committed score files, the
scorer's contrasts and the training logs in this folder. No GPU.

    tables   ../summary.{json,txt}: each run's pooled metrics, guards, data-swap movement and swap cost;
             ../by_scan_type.{json,txt}: post hoc splits (narrower than the grid / full width, 1.5 T / 3 T) of the
             main contrasts with the scorer's volume-cluster bootstrap.
    curves   ../curves.json and ../figures/curves.png: the trainer's per-epoch validation PSNR and training loss of
             every run and its fscr1 control.
    plots    ../figures/contrasts.png (decision rows and secondary contrasts against the +0.10 dB bar) and
             ../figures/data_swap.png (the share of each step's draws that a data swap moves).

Run from the repository root: `python reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/summarise.py all`.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from activemri.utils.multistep import cluster_bootstrap  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
SCORES = REPORT / 'scores'
FIGURES = REPORT / 'figures'
FSCR1 = REPO / 'reports/multistep_adaptivity/fscr1'
MANIFEST = REPO / 'configs/datasets/fastmri_local_v1/manifest.csv'
GRID_COLUMNS = 320
PASS_DB = 0.10

RUNS = {
    'F_tF': 'F_4_s42_fscr1_tF', 'Fb_tF': 'Fb_4_s42_fscr1_tF', 'L_tF': 'L_1_s42_fscr1_tF',
    'F_r150': 'F_4_s42_fscr1_tF_r150', 'Fb_r150': 'Fb_4_s42_fscr1_tF_r150', 'L_r150': 'L_1_s42_fscr1_tF_r150',
    'F_lrs': 'F_4_s42_lrs_tF', 'Fb_lrs': 'Fb_4_s42_lrs_tF',
    'F_route': 'F_4_s42_route_tF', 'Fb_route': 'Fb_4_s42_route_tF',
    'F_routebias': 'F_4_s42_routebias_tF', 'Fb_routebias': 'Fb_4_s42_routebias_tF',
}
LOGS = {
    'F_tF': FSCR1 / 'multistep-3090320.log', 'Fb_tF': FSCR1 / 'multistep-3090321.log',
    'L_tF': FSCR1 / 'multistep-3090322.log',
    'F_r150': REPORT / 'logs/seqmri_lrb-3161833.log', 'Fb_r150': REPORT / 'logs/seqmri_lrb-3161834.log',
    'L_r150': REPORT / 'logs/seqmri_lrb-3161835.log', 'Fb_lrs': REPORT / 'logs/seqmri_lrb-3161836.log',
    'F_lrs': REPORT / 'logs/seqmri_lrb-3161837.log', 'F_route': REPORT / 'logs/seqmri_lrb-3161839.log',
    'F_routebias': REPORT / 'logs/seqmri_lrb-3161840.log', 'Fb_route': REPORT / 'logs/seqmri_lrb-3161841.log',
    'Fb_routebias': REPORT / 'logs/seqmri_lrb-3161842.log',
}
SPLIT_CONTRASTS = {
    'F_lrs-L': [(1, 'F_lrs'), (-1, 'L_tF')],
    'F_lrs-Fb_lrs': [(1, 'F_lrs'), (-1, 'Fb_lrs')],
    'Fb_lrs-L': [(1, 'Fb_lrs'), (-1, 'L_tF')],
    'F_r150-L_r150': [(1, 'F_r150'), (-1, 'L_r150')],
    'F_routebias-Fb_routebias': [(1, 'F_routebias'), (-1, 'Fb_routebias')],
    'F_tF-L (fscr1)': [(1, 'F_tF'), (-1, 'L_tF')],
}
COLOURS = {'F': '#2a78d6', 'Fb': '#eb6834', 'L': '#1baf7a'}
PANELS = {
    'R: resumed to epoch 150': ['F_tF', 'Fb_tF', 'L_tF', 'F_r150', 'Fb_r150', 'L_r150'],
    'S: sampler at lr 1e-4': ['F_tF', 'Fb_tF', 'L_tF', 'F_lrs', 'Fb_lrs'],
    'B: route and per-step maps': ['L_tF', 'F_route', 'Fb_route', 'F_routebias', 'Fb_routebias'],
}
STYLES = {'tF': ':', 'r150': '-', 'lrs': '-', 'route': '--', 'routebias': '-'}
CONTRAST_ROWS = [
    ('contrasts.json', 'R1', 'R1  (F-L) at 150 minus at 100'),
    ('contrasts.json', 'S1', 'S1  F_lrs - F'),
    ('contrasts.json', 'B1', 'B1  F_routebias - F_route'),
    ('contrasts.json', 'B2', 'B2  F_routebias - Fb_routebias'),
    ('contrasts.json', 'B3', 'B3  Fb_routebias - L'),
    ('contrasts_secondary.json', 'F150_minus_L150', 'F_r150 - L_r150'),
    ('contrasts_secondary.json', 'Flrs_minus_L', 'F_lrs - L'),
    ('contrasts_secondary.json', 'Flrs_minus_Fblrs', 'F_lrs - Fb_lrs'),
    ('contrasts_secondary.json', 'Fblrs_minus_L', 'Fb_lrs - L'),
    ('contrasts_secondary.json', 'Fblrs_minus_Fb', 'Fb_lrs - Fb'),
    ('contrasts_secondary.json', 'route_F', 'F_route - F (route alone)'),
    ('contrasts_secondary.json', 'route_Fb', 'Fb_route - Fb (route alone)'),
    ('contrasts_secondary.json', 'Fb150_minus_Fb100', 'Fb_r150 - Fb'),
]


def per_slice_mean(values):
    """Mean over draw seeds, NaN where every seed is NaN (a slice with no donor), as the scorer takes it."""
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    return np.where(count > 0, np.where(finite, values, 0.0).sum(axis=-1) / np.maximum(count, 1), np.nan)


def load_scores():
    return {arm: (np.load(SCORES / f'{run}.npz', allow_pickle=False), json.loads((SCORES / f'{run}.json').read_text()))
            for arm, run in RUNS.items()}


def tables():
    scores = load_scores()
    summary = {}
    for arm, (_, scored) in scores.items():
        summary[arm] = {
            'run': RUNS[arm], 'epoch': scored['epoch'] + 1, 'eval_commit': scored['eval_code']['commit'],
            **{metric: scored['means'][metric]['all'] for metric in ('psnr', 'ssim', 'nmse', 'realised_fraction')},
            'ssim_pooled': scored['guards']['ssim_pooled'],
            'most_draws_outside_the_acquired_region': scored['guards']['most_draws_outside_the_acquired_region'],
            'data_swap_moved_by_step': scored['data_swap']['moved_by_step']['all'] if 'data_swap' in scored else None,
            'swap_cost_psnr': scored['swap_cost']['psnr']['all'] if 'swap_cost' in scored else None,
        }
    lines = [f"{'run':14s} {'epoch':>5s} {'PSNR':>7s} {'SSIM':>6s} {'NMSE':>7s} {'realised':>8s} "
             f"{'data swap moved, steps 0-3':>28s} {'swap cost (dB)':>24s}"]
    for arm, row in summary.items():
        moved = ' '.join(f'{value:.3f}' for value in row['data_swap_moved_by_step'] or []) or 'LOUPE: no data'
        cost = row['swap_cost_psnr'] or {'mean': float('nan'), 'ci95': [float('nan')] * 2}
        lines.append(f"{arm:14s} {row['epoch']:5d} {row['psnr']:7.3f} {row['ssim_pooled']:6.4f} {row['nmse']:7.4f} "
                     f"{row['realised_fraction']:8.4f} {moved:>28s} "
                     f"{cost['mean']:+.3f} [{cost['ci95'][0]:+.3f},{cost['ci95'][1]:+.3f}]")
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (REPORT / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    scans = {row['file']: row for row in csv.DictReader(MANIFEST.open())}
    volumes = scores['F_tF'][0]['volume']
    if any((npz['volume'] != volumes).any() for npz, _ in scores.values()):
        raise SystemExit('the runs were scored on different slices')
    columns = np.array([int(scans[v]['kspace_shape'].strip('[]').split(',')[-1]) for v in volumes])
    field = np.array([scans[v]['field'] for v in volumes])
    groups = {'narrower than the grid': columns < GRID_COLUMNS, 'full width': columns == GRID_COLUMNS,
              '1.5T': field == '1.5T', '3T': field == '3T', 'all': np.ones(len(volumes), dtype=bool)}
    psnr = {arm: per_slice_mean(npz['psnr']) for arm, (npz, _) in scores.items()}
    strata = np.zeros(len(volumes), dtype=int)
    split = {}
    for group, inside in groups.items():
        row = {'n_volumes': int(len(np.unique(volumes[inside]))), 'n_slices': int(inside.sum())}
        for label, terms in SPLIT_CONTRASTS.items():
            delta = sum(sign * psnr[arm] for sign, arm in terms)
            row[label] = cluster_bootstrap(delta[inside], volumes[inside], strata[inside])
        split[group] = row
    lines = [f"{'group':24s} {'vols':>4s} " + ' '.join(f'{label:>25s}' for label in SPLIT_CONTRASTS)]
    for group, row in split.items():
        cells = ' '.join(f"{row[label]['mean']:+.3f} [{row[label]['ci95'][0]:+.3f},{row[label]['ci95'][1]:+.3f}]"
                         .rjust(25) for label in SPLIT_CONTRASTS)
        lines.append(f"{group:24s} {row['n_volumes']:4d} {cells}")
    (REPORT / 'by_scan_type.json').write_text(json.dumps(split, indent=2) + '\n')
    (REPORT / 'by_scan_type.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def parse_log(path):
    """Per epoch: the trainer's validation PSNR and SSIM (the [brain] line before each epoch line) and both losses."""
    epochs, metrics = [], None
    for line in path.read_text().splitlines():
        found = re.search(r'\[brain\] .*PSNR=([\d.]+) SSIM=([\d.]+)', line)
        if found:
            metrics = (float(found.group(1)), float(found.group(2)))
        found = re.search(r'Epoch = \[\s*(\d+)/\s*\d+\] TrainLoss = ([\d.eE+-]+) ValLoss = ([\d.eE+-]+)', line)
        if found and metrics:
            epochs.append({'epoch': int(found.group(1)), 'val_psnr': metrics[0], 'val_ssim': metrics[1],
                           'train_loss': float(found.group(2)), 'val_loss': float(found.group(3))})
            metrics = None
    return epochs


def style(arm):
    kind, _, variant = arm.partition('_')
    return COLOURS[kind], STYLES[variant]


def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {arm: parse_log(path) for arm, path in LOGS.items()}
    (REPORT / 'curves.json').write_text(json.dumps(data) + '\n')
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex='col')
    for column, (title, arms) in enumerate(PANELS.items()):
        for arm in arms:
            colour, dash = style(arm)
            epochs = [row['epoch'] for row in data[arm]]
            for row_index, key in enumerate(('val_psnr', 'train_loss')):
                axes[row_index, column].plot(epochs, [row[key] for row in data[arm]], dash, color=colour, lw=1.6,
                                             label=arm)
        axes[0, column].set_title(title)
        axes[0, column].set_ylim(34.0, 35.0)
        axes[1, column].set_ylim(0.0225, 0.0255)
        axes[1, column].set_xlabel('epoch')
        axes[0, column].legend(fontsize=8, loc='lower right', ncol=2)
        for axis in axes[:, column]:
            axis.axvline(90, color='#999999', lw=0.8)
            axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel('trainer val PSNR (dB, 307 dev slices)')
    axes[1, 0].set_ylabel('training loss (L1)')
    figure.suptitle('Training curves (dotted: fscr1 controls; grey line: lr step at epoch 90)')
    figure.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / 'curves.png', dpi=110)
    for arm, rows in data.items():
        print(f"{arm:14s} epochs {rows[0]['epoch']}-{rows[-1]['epoch']}, last val PSNR {rows[-1]['val_psnr']:.2f}, "
              f"mean of the last 10 {np.mean([row['val_psnr'] for row in rows[-10:]]):.3f}")


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    loaded = {name: json.loads((REPORT / name).read_text())['contrasts']
              for name in ('contrasts.json', 'contrasts_secondary.json')}
    figure, axis = plt.subplots(figsize=(9, 6.5))
    for position, (name, key, label) in enumerate(CONTRAST_ROWS):
        value = loaded[name][key]['psnr']['all']
        colour = '#2a78d6' if name == 'contrasts.json' else '#777777'
        axis.errorbar(value['mean'], position, xerr=[[value['mean'] - value['ci95'][0]], [value['ci95'][1] - value['mean']]],
                      fmt='o', color=colour, capsize=3, ms=6)
        axis.text(0.62, position, f"{value['mean']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}]",
                  va='center', fontsize=8)
    axis.set_yticks(range(len(CONTRAST_ROWS)), [label for _, _, label in CONTRAST_ROWS], fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0, color='black', lw=0.8)
    axis.axvline(PASS_DB, color='#1baf7a', lw=1, ls='--')
    axis.axhline(4.5, color='#cccccc', lw=0.8)
    axis.set_xlim(-0.45, 1.0)
    axis.set_xlabel('paired pooled PSNR difference, dB (95% volume bootstrap; green: the +0.10 dB pass bar)')
    axis.set_title('Decision rows (blue) and secondary contrasts (grey), seed 42')
    figure.tight_layout()
    figure.savefig(FIGURES / 'contrasts.png', dpi=110)

    summary = json.loads((REPORT / 'summary.json').read_text())
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for arm in ('F_tF', 'F_r150', 'F_lrs', 'F_route', 'F_routebias'):
        colour, dash = style(arm)
        axis.plot(range(4), summary[arm]['data_swap_moved_by_step'], dash, marker='o', color=colour, label=arm,
                  alpha=0.5 if arm.endswith('tF') else 1.0)
    axis.set_xticks(range(4))
    axis.set_xlabel('step')
    axis.set_ylabel("share of the step's draws a data swap moves")
    axis.set_title('Data-swap movement of the F runs (twins: 0 at every step)')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / 'data_swap.png', dpi=110)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('tables', 'curves', 'plots', 'all'))
    stage = parser.parse_args().stage
    for name, function in (('tables', tables), ('curves', curves), ('plots', plots)):
        if stage in (name, 'all'):
            function()
