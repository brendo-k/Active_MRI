"""How the learned LOUPE masks of varnet_loupe_lr_schedule_r8 differ between recon learning rates, seeds and depths.

Reads `sampler.mask_net.mask` (the 320x320 logits) from each run's last_model.pt and best_model.pt and works on the raw
sampling probability p = sigmoid(slope * logits), before the per-slice budget rescale. CPU only.

Per run: mean p, binarization mean|2p - 1|, support (fraction of the 320x320 grid with p > 0.05), the share of total p
inside radius bands, and a radial profile.
Per pair of masks: Pearson correlation of p, correlation of the residual after removing each map's own radial profile
(the fine, pixel-scale structure), and the Jaccard overlap of the top 12.5% locations. Pairs are grouped as
  lr     same seed and depth, arm vs the 1e-3 step control
  seed   same arm and depth, s42 vs s43
  depth  same arm and seed, 6 vs 12 cascades
  time   same run, best_model vs last_model
The question the grouping answers: does changing the recon lr move the mask more than changing the training seed does?

    python scripts/plots/varnet_loupe_lr_schedule_r8_masks.py \
        --runs /export/scratch/bkadota/varnet_loupe_lr_schedule_r8 --report reports/varnet_loupe_lr_schedule_r8
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import _path  # noqa: F401

SLOPE = 10
TOP_FRACTION = 0.125
SUPPORT_P = 0.05
BANDS = ((0, 16), (16, 48), (48, 96), (96, 1000))
RADIAL_STEP = 4
GROUPS = {
    'full_12c': ('runs_full', ('lr1e3_step', 'lr2e3_step', 'lr4e3_step', 'lr2e3_wcos', 'lr4e3_wcos'), (42, 43)),
    'full_6c': ('runs_full_c6', ('lr1e3_step', 'lr2e3_step', 'lr4e3_step'), (42, 43)),
    'data20_12c': ('runs', ('lr1e3_step', 'lr2e3_step', 'shared_2e3_step', 'lr1e3_const'), (42, 43, 44)),
}
CONTROL = 'lr1e3_step'


def probability(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    logits = checkpoint['model']['sampler.mask_net.mask'].float().squeeze()
    return torch.sigmoid(SLOPE * logits).numpy(), int(checkpoint['epoch']) + 1


def radius_map(shape):
    ky, kx = (np.arange(n) - n // 2 for n in shape)
    return np.hypot(ky[:, None], kx[None, :])


def radial_profile(p, radius):
    bins = (radius // RADIAL_STEP).astype(int)
    totals = np.bincount(bins.ravel(), p.ravel())
    counts = np.bincount(bins.ravel())
    return totals / np.maximum(counts, 1), bins


def run_stats(p, radius):
    profile, _ = radial_profile(p, radius)
    total = p.sum()
    return {'mean_p': float(p.mean()), 'binarization': float(np.abs(2 * p - 1).mean()), 'support': float((p > SUPPORT_P).mean()),
            'band_share': {f'{lo}-{hi}': float(p[(radius >= lo) & (radius < hi)].sum() / total) for lo, hi in BANDS},
            'radial_profile': profile.tolist()}


def residual(p, radius):
    profile, bins = radial_profile(p, radius)
    return p - profile[bins]


def compare(a, b, radius):
    top = int(TOP_FRACTION * a.size)
    sa, sb = set(np.argsort(a.ravel())[-top:]), set(np.argsort(b.ravel())[-top:])
    ra, rb = residual(a, radius).ravel(), residual(b, radius).ravel()
    return {'corr': float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 'residual_corr': float(np.corrcoef(ra, rb)[0, 1]),
            'top_jaccard': len(sa & sb) / len(sa | sb), 'mean_abs_diff': float(np.abs(a - b).mean())}


def pairs(maps):
    out = []
    for group, (_, arms, seeds) in GROUPS.items():
        for arm in arms:
            if arm != CONTROL:
                out += [('lr', group, arm, f'{arm}_s{s}', f'{CONTROL}_s{s}', 'last', 'last') for s in seeds]
            out += [('seed', group, arm, f'{arm}_s{a}', f'{arm}_s{b}', 'last', 'last') for a, b in itertools.combinations(seeds, 2)]
            out += [('time', group, arm, f'{arm}_s{s}', f'{arm}_s{s}', 'best', 'last') for s in seeds]
    for arm in GROUPS['full_6c'][1]:
        out += [('depth', 'full_6c_vs_12c', arm, f'{arm}_s{s}', f'{arm}_s{s}', 'last', 'last') for s in GROUPS['full_6c'][2]]
    return out


def figure(maps, stats, path):
    shown = [('full_12c', 'lr1e3_step'), ('full_12c', 'lr4e3_step'), ('full_6c', 'lr1e3_step'), ('full_6c', 'lr4e3_step')]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11.5))
    for row, seed in enumerate((42, 43)):
        for col, (group, arm) in enumerate(shown):
            p = maps[group][f'{arm}_s{seed}']['last']
            image = axes[row, col].imshow(p, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            axes[row, col].set_title(f"{arm}, {group.split('_')[1]}, s{seed}\nsupport {(p > SUPPORT_P).mean():.2f}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.colorbar(image, ax=axes[:2, :], fraction=0.02, pad=0.01, label='p = sigmoid(10 logits), full 320x320 grid')
    for ax, group in zip(axes[2, :2], ('full_12c', 'full_6c')):
        for arm, style in (('lr1e3_step', 'k-'), ('lr2e3_step', 'C0--'), ('lr4e3_step', 'C0-'), ('lr4e3_wcos', 'C1-')):
            for s in GROUPS[group][2]:
                key = f'{arm}_s{s}'
                if key in stats[group]:
                    profile = stats[group][key]['last']['radial_profile']
                    ax.plot(np.arange(len(profile)) * RADIAL_STEP, profile, style, lw=1.2, alpha=0.8,
                            label=arm if s == GROUPS[group][2][0] else None)
        ax.set_xlim(0, 226)
        ax.set_xlabel('k-space radius (pixels)')
        ax.set_ylabel('mean p')
        ax.set_title(f'radial profile, {group}, both seeds')
        ax.legend(fontsize=8)
    seed = 42
    for ax, group in zip(axes[2, 2:], ('full_12c', 'full_6c')):
        diff = maps[group][f'lr4e3_step_s{seed}']['last'] - maps[group][f'{CONTROL}_s{seed}']['last']
        image = ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f"lr4e3_step - lr1e3_step, s{seed}, {group.split('_')[1]}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes[2, 2:], fraction=0.04, pad=0.01, label='difference in p')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True, help='folder holding runs/, runs_full/ and runs_full_c6/')
    parser.add_argument('--report', type=Path, default=Path('reports/varnet_loupe_lr_schedule_r8'))
    args = parser.parse_args()

    maps, stats, epochs = {}, {}, {}
    radius = None
    for group, (folder, arms, seeds) in GROUPS.items():
        maps[group], stats[group] = {}, {}
        for arm, s in itertools.product(arms, seeds):
            name = f'{arm}_s{s}'
            maps[group][name], stats[group][name] = {}, {}
            for which in ('last', 'best'):
                p, epoch = probability(args.runs / folder / name / f'{which}_model.pt')
                radius = radius_map(p.shape) if radius is None else radius
                maps[group][name][which] = p
                stats[group][name][which] = run_stats(p, radius) | {'epoch': epoch}
    maps['full_6c_vs_12c'] = None

    comparisons = []
    for kind, group, arm, a, b, wa, wb in pairs(maps):
        ga, gb = ('full_6c', 'full_12c') if group == 'full_6c_vs_12c' else (group, group)
        comparisons.append({'kind': kind, 'group': group, 'arm': arm, 'a': f'{a}:{wa}', 'b': f'{b}:{wb}'}
                           | compare(maps[ga][a][wa], maps[gb][b][wb], radius))

    table = {}
    for c in comparisons:
        table.setdefault(f"{c['group']}|{c['kind']}|{c['arm']}", []).append(c)
    summary = {key: {m: float(np.mean([c[m] for c in rows])) for m in ('corr', 'residual_corr', 'top_jaccard', 'mean_abs_diff')}
               | {'n': len(rows)} for key, rows in table.items()}

    (args.report / 'masks.json').write_text(json.dumps({'slope': SLOPE, 'top_fraction': TOP_FRACTION, 'runs': stats,
                                                         'comparisons': comparisons, 'summary': summary}, indent=1))
    figure(maps, stats, args.report / 'figures' / 'masks.png')

    print(f"{'group|kind|arm':40s} {'corr':>6s} {'resid':>6s} {'top-J':>6s} {'|dp|':>6s}  n")
    for key, row in summary.items():
        print(f"{key:40s} {row['corr']:6.3f} {row['residual_corr']:6.3f} {row['top_jaccard']:6.3f} {row['mean_abs_diff']:6.3f}  {row['n']}")
    print(f"\n{'group run':34s} {'ep':>4s} {'mean p':>7s} {'binar':>6s} {'supp':>6s} " + ' '.join(f'{lo}-{hi:>4}' for lo, hi in BANDS))
    for group, runs in stats.items():
        for name, entry in runs.items():
            for which in ('best', 'last'):
                e = entry[which]
                print(f"{group + ' ' + name + ':' + which:34s} {e['epoch']:4d} {e['mean_p']:7.4f} {e['binarization']:6.3f} {e['support']:6.3f} "
                      + ' '.join(f'{v:7.3f}' for v in e['band_share'].values()))


if __name__ == '__main__':
    main()
