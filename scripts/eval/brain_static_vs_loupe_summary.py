"""Decide the frozen criteria of reports/brain_static_vs_loupe/proposal.md from eval.json, and plot the mean brain maps.

Paired contrasts come from brain_static_vs_loupe_eval.py (per slice, full brain val, 95% volume-bootstrap CI). Here:
  seed spread      max - min of an arm's mean PSNR over seeds
  seed swap        arm_a(seed i) - arm_b(seed j), i != j: the gap an unpaired comparison would see
  criteria         thresholds 3-6 of the proposal, applied to each contrast; 1 (SSIM >= 0.80) and 2 (budget >= 10%)

    python scripts/eval/brain_static_vs_loupe_summary.py --report reports/brain_static_vs_loupe
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ARMS = ('static', 'loupe_shared', 'loupe_uniform')
CONTRASTS = (('loupe_shared', 'static'), ('loupe_uniform', 'static'), ('loupe_shared', 'loupe_uniform'))
WIN, TIE = 0.10, 0.05
MIN_SSIM, MIN_RELOCATED = 0.80, 0.10


def verdict(deltas, spread):
    mean = float(np.mean(deltas))
    all_positive, all_negative = all(d > 0 for d in deltas), all(d < 0 for d in deltas)
    if mean >= WIN and all_positive and mean > spread:
        return 'first arm wins'
    if mean <= -WIN and all_negative:
        return 'second arm wins'
    if abs(mean) < TIE:
        return 'no effect'
    return 'inconclusive'


def plot_maps(maps, seeds, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(seeds), 4, figsize=(13, 3.2 * len(seeds)), squeeze=False)
    for row, seed in enumerate(seeds):
        static = maps[f'static_s{seed}'].float()
        panels = [('static (shared S, frozen)', static, 'viridis', (0, 1))]
        for arm in ('loupe_shared', 'loupe_uniform'):
            panels.append((arm, maps[f'{arm}_s{seed}'].float(), 'viridis', (0, 1)))
        panels.append(('loupe_shared - static', panels[1][1] - static, 'RdBu_r', (-0.5, 0.5)))
        for ax, (title, image, cmap, (lo, hi)) in zip(axes[row], panels):
            shown = ax.imshow(image.numpy(), cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f'{title}, s{seed}', fontsize=9)
            ax.set_xlabel('kx (phase encode)')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(shown, ax=ax, fraction=0.046)
    fig.suptitle('Mean brain sampling probability (after budget rescaling), full brain val')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    evaluated = json.loads((args.report / 'eval.json').read_text())
    runs, contrasts, relocated = evaluated['runs'], evaluated['contrasts'], evaluated['budget_relocated_vs_static']
    seeds = sorted({int(name.rsplit('_s', 1)[1]) for name in runs})

    psnr = {arm: {s: runs[f'{arm}_s{s}']['psnr'] for s in seeds} for arm in ARMS}
    summary = {'seeds': seeds, 'n_slices': evaluated['n_slices'], 'n_volumes': evaluated['n_volumes'],
               'arms': {arm: {m: {'mean': float(np.mean([runs[f'{arm}_s{s}'][m] for s in seeds])),
                                  'min': float(min(runs[f'{arm}_s{s}'][m] for s in seeds)),
                                  'max': float(max(runs[f'{arm}_s{s}'][m] for s in seeds))}
                              for m in ('psnr', 'ssim', 'nmse', 'fraction')} for arm in ARMS},
               'seed_spread_psnr': {arm: float(max(v.values()) - min(v.values())) for arm, v in psnr.items()},
               'contrasts': {}}
    static_spread = summary['seed_spread_psnr']['static']
    for a, b in CONTRASTS:
        paired = [contrasts[f'{a}-{b}_s{s}']['psnr']['mean'] for s in seeds]
        swapped = [psnr[a][i] - psnr[b][j] for i, j in itertools.permutations(seeds, 2)]
        summary['contrasts'][f'{a}-{b}'] = {
            'paired_psnr_by_seed': dict(zip(map(str, seeds), paired)),
            'paired_ci95_by_seed': {str(s): contrasts[f'{a}-{b}_s{s}']['psnr']['ci95'] for s in seeds},
            'paired_ssim_by_seed': {str(s): contrasts[f'{a}-{b}_s{s}']['ssim']['mean'] for s in seeds},
            'paired_nmse_by_seed': {str(s): contrasts[f'{a}-{b}_s{s}']['nmse']['mean'] for s in seeds},
            'mean_paired_psnr': float(np.mean(paired)),
            'seed_swap_psnr_range': [float(min(swapped)), float(max(swapped))],
            'verdict': verdict(paired, static_spread),
        }
    min_ssim = min(r['ssim'] for r in runs.values())
    shared_relocated = [relocated[f'loupe_shared_s{s}'] for s in seeds]
    summary['criteria'] = {
        '1_validity_min_ssim': {'value': min_ssim, 'met': min_ssim >= MIN_SSIM and len(runs) == 3 * len(seeds)},
        '2_loupe_shared_relocated': {'value': shared_relocated, 'met': min(shared_relocated) >= MIN_RELOCATED},
        'budget_relocated_vs_static': relocated,
        'primary_loupe_shared-static': summary['contrasts']['loupe_shared-static']['verdict'],
        'secondary_loupe_uniform-static': summary['contrasts']['loupe_uniform-static']['verdict'],
        'start_effect_loupe_shared-loupe_uniform': summary['contrasts']['loupe_shared-loupe_uniform']['verdict'],
    }
    (args.report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    lines = ['| contrast | ' + ' | '.join(f's{s}' for s in seeds) + ' | mean | seed-swap range | verdict |',
             '|---|' + '---|' * (len(seeds) + 3)]
    for name, c in summary['contrasts'].items():
        cells = [f"{c['paired_psnr_by_seed'][str(s)]:+.3f} [{c['paired_ci95_by_seed'][str(s)][0]:+.2f}, "
                 f"{c['paired_ci95_by_seed'][str(s)][1]:+.2f}]" for s in seeds]
        swap = c['seed_swap_psnr_range']
        lines.append(f"| {name} | " + ' | '.join(cells) + f" | {c['mean_paired_psnr']:+.3f} | "
                     f"{swap[0]:+.2f} to {swap[1]:+.2f} | {c['verdict']} |")
    print('\n'.join(lines))
    print('seed spread (PSNR max - min):', {k: round(v, 3) for k, v in summary['seed_spread_psnr'].items()})
    print('criteria:', json.dumps(summary['criteria'], indent=1))

    per_slice = torch.load(args.report / 'eval_per_slice.pt', weights_only=False)
    plot_maps(per_slice['maps'], seeds, args.report / 'figures' / 'mean_brain_maps.png')


if __name__ == '__main__':
    main()
