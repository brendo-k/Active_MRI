"""Tables, frozen criteria and figures of reports/brain_varnet_scaling from the files of brain_varnet_scaling_eval.py.

Per run: full-val PSNR / SSIM / NMSE at best_model.pt (the brain-val peak) and last_model.pt, and the train - val
gap at the peak. Per data size: the ranking, the seed-43 picks (the two highest seed-42 peaks, plus 12 cascades at
20 volumes when it is not among them) and, per seed, Q1: the best smaller size minus 12 cascades at 20 volumes, paired
per slice, with a 95% bootstrap CI over volumes. Q2 compares the best cascade count at 80 and 20 volumes per seed.

Writes <eval>/summary.json and <figures>/full_val_vs_cascades.png, full_val_vs_volumes.png.

    python scripts/eval/brain_varnet_scaling_summary.py
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

EVAL = Path('reports/brain_varnet_scaling/eval')
FIGURES = Path('reports/brain_varnet_scaling/figures')
CASCADES = (2, 4, 6, 8, 12)
VOLUMES = (20, 40, 80)
CASCADE_COLOURS = dict(zip(CASCADES, ('#7fb6dc', '#4f97cc', '#2b78b8', '#145a9a', '#08306b')))
VOLUME_COLOURS = {20: '#e08214', 40: '#5e3c99', 80: '#1b7837'}
FILE = re.compile(r'^(val|train)_(best_model|last_model)_c(\d+)_v(\d+)_s(\d+)\.pt$')


def load_scores(folder):
    scores = {}
    for path in folder.glob('*.pt'):
        m = FILE.match(path.name)
        if m:
            saved = torch.load(path, weights_only=False)
            scores[m[1], m[2], int(m[3]), int(m[4]), int(m[5])] = (
                {k: v.numpy() for k, v in saved['scores'].items()}, np.array(saved['volume']))
    return scores


def volume_ci(diff, volumes, n=10000):
    rng = np.random.default_rng(0)
    names = np.unique(volumes)
    means = np.array([diff[volumes == v].mean() for v in names])
    counts = np.array([(volumes == v).sum() for v in names])
    draws = rng.integers(0, len(names), (n, len(names)))
    boot = (means[draws] * counts[draws]).sum(1) / counts[draws].sum(1)
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def q1_verdict(per_seed):
    """per_seed: {seed: (best smaller - c12 mean, CI, smaller - c12 means for every smaller size)}."""
    if len(per_seed) < 2:
        return 'needs both seeds'
    diffs = [d for d, _, _ in per_seed.values()]
    if all(d >= 0.15 and ci[0] > 0 for d, ci, _ in per_seed.values()):
        return 'smaller wins'
    if all(abs(d) < 0.05 for d in diffs):
        return 'capacity not the limit'
    if all(max(others.values()) <= -0.15 for _, _, others in per_seed.values()):
        return 'bigger wins'
    return 'inconclusive'


def plot(table, seed, x_key, line_key, lines, colours, xlabel, label, path, logx=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for value in lines:
        points = sorted((r[x_key], r) for r in table if r[line_key] == value and r['seed'] == seed)
        if not points:
            continue
        ax.plot([p[0] for p in points], [p[1]['best_psnr'] for p in points], '-o', color=colours[value],
                linewidth=2, markersize=8, label=label(value))
    if logx:
        ax.set_xscale('log')
        ax.set_xticks(VOLUMES, [str(v) for v in VOLUMES])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_xticks(CASCADES)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('full brain val PSNR at peak checkpoint (dB)')
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f'seed {seed}; best_model.pt, 40 volumes / 605 slices, mask S', fontsize=9, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, default=EVAL)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    args = parser.parse_args()
    scores = load_scores(args.eval)
    runs = sorted({(c, v, s) for (split, ckpt, c, v, s) in scores if split == 'val' and ckpt == 'best_model'})
    table = []
    for c, v, s in runs:
        best, _ = scores['val', 'best_model', c, v, s]
        last, _ = scores['val', 'last_model', c, v, s]
        row = {'cascades': c, 'volumes': v, 'seed': s,
               **{f'best_{m}': float(best[m].mean()) for m in ('psnr', 'ssim', 'nmse')},
               **{f'last_{m}': float(last[m].mean()) for m in ('psnr', 'ssim', 'nmse')}}
        if ('train', 'best_model', c, v, s) in scores:
            row['gap_best'] = float(scores['train', 'best_model', c, v, s][0]['psnr'].mean() - row['best_psnr'])
            row['gap_last'] = float(scores['train', 'last_model', c, v, s][0]['psnr'].mean() - row['last_psnr'])
        table.append(row)

    seeds = sorted({r['seed'] for r in table})
    summary = {'runs': table, 'ranking': {}, 'seed43_picks': {}, 'q1': {}, 'q2': {}}
    for s in seeds:
        for v in VOLUMES:
            rows = sorted((r for r in table if r['seed'] == s and r['volumes'] == v), key=lambda r: -r['best_psnr'])
            summary['ranking'][f's{s}_v{v}'] = [(r['cascades'], round(r['best_psnr'], 3)) for r in rows]
            if s == 42 and len(rows) == len(CASCADES):
                picks = [r['cascades'] for r in rows[:2]]
                if v == 20 and 12 not in picks:
                    picks.append(12)
                summary['seed43_picks'][f'v{v}'] = picks

    q1 = {}
    for s in seeds:
        have = {c for (split, ckpt, c, v, ss) in scores if split == 'val' and ckpt == 'best_model' and v == 20 and ss == s}
        if 12 not in have or len(have) < 2:
            continue
        ref, volumes = scores['val', 'best_model', 12, 20, s]
        diffs = {c: scores['val', 'best_model', c, 20, s][0]['psnr'] - ref['psnr'] for c in sorted(have - {12})}
        means = {c: float(d.mean()) for c, d in diffs.items()}
        best_c = max(means, key=means.get)
        q1[s] = (means[best_c], volume_ci(diffs[best_c], volumes), means)
        summary['q1'][f's{s}'] = {'best_smaller': best_c, 'diff': means[best_c], 'ci': q1[s][1],
                                  'every_smaller_minus_c12': means}
    summary['q1']['verdict'] = q1_verdict(q1)
    for s in seeds:
        best = {v: summary['ranking'][f's{s}_v{v}'][0][0] for v in (20, 80) if summary['ranking'].get(f's{s}_v{v}')}
        if len(best) == 2:
            summary['q2'][f's{s}'] = {'best_c_v20': best[20], 'best_c_v80': best[80], 'shift': best[80] > best[20]}

    for r in sorted(table, key=lambda r: (r['seed'], r['volumes'], r['cascades'])):
        gap = f"  gap {r['gap_best']:.2f} / {r['gap_last']:.2f}" if 'gap_best' in r else ''
        print(f"s{r['seed']} v{r['volumes']:<2d} c{r['cascades']:<2d}  best {r['best_psnr']:.3f} "
              f"(SSIM {r['best_ssim']:.4f})  last {r['last_psnr']:.3f}{gap}")
    print(json.dumps({k: summary[k] for k in ('ranking', 'seed43_picks', 'q1', 'q2')}, indent=1))
    (args.eval / 'summary.json').write_text(json.dumps(summary, indent=1))

    args.figures.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        suffix = '' if s == 42 else f'_s{s}'
        plot(table, s, 'cascades', 'volumes', VOLUMES, VOLUME_COLOURS, 'VarNet cascades', lambda v: f'{v} volumes',
             args.figures / f'full_val_vs_cascades{suffix}.png')
        plot(table, s, 'volumes', 'cascades', CASCADES, CASCADE_COLOURS, 'training volumes',
             lambda c: f'{c} cascades', args.figures / f'full_val_vs_volumes{suffix}.png', logx=True)


if __name__ == '__main__':
    main()
