"""Compare zero-step LOUPE 2D masks learned separately on brain and knee.

Reads <root>/<anatomy>/seed_<n>/best_model.pt written by scripts/train_loupe_anatomy.sh.
The mask is p = sigmoid(slope * logits), the LOUPE free parameter before the
sampler's sparsity rescaling.

Two questions:
  stability  - are within-anatomy seed pairs similar to each other?
  difference - are brain-vs-knee pairs less similar than seed pairs?

Raw Pearson r is dominated by the shared dense-centre falloff, so each comparison
is also made on a smoothed mask (init noise is iid per pixel) and on the residual
left after subtracting each mask's radial average.

    python scripts/analyze_loupe_anatomy.py --root runs/loupe_anatomy
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
from scipy.ndimage import gaussian_filter

import _path  # noqa: F401

ANATOMIES = ('brain', 'knee')
COLORS = {'brain': '#0072B2', 'knee': '#D55E00'}
RADIAL_BINS = 32


def load_mask(run_dir):
    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    key = next(k for k in ckpt['model'] if k.endswith('mask_net.mask'))
    logits = ckpt['model'][key].float().squeeze().numpy()
    slope = ckpt['options'].slope
    return 1.0 / (1.0 + np.exp(-slope * logits)), ckpt['epoch'] + 1, ckpt['options'].num_epochs


def normalized_radius(shape):
    ky = (np.arange(shape[0]) - shape[0] // 2) / (shape[0] / 2)
    kx = (np.arange(shape[1]) - shape[1] // 2) / (shape[1] / 2)
    return np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def radial_bins(shape):
    return np.minimum((normalized_radius(shape) * RADIAL_BINS).astype(int), RADIAL_BINS)


def radial_profile(mask, bins):
    return np.bincount(bins.ravel(), mask.ravel()) / np.bincount(bins.ravel())


def radial_residual(mask, bins):
    return mask - radial_profile(mask, bins)[bins]


def pearson(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def pair_stats(masks, transform):
    t = {key: transform(m) for key, m in masks.items()}
    within = {a: [pearson(t[x], t[y]) for x, y in itertools.combinations([k for k in t if k[0] == a], 2)]
              for a in ANATOMIES}
    cross = [pearson(t[b], t[k]) for b in t if b[0] == 'brain' for k in t if k[0] == 'knee']
    return within, cross, t


def span(values):
    return f'{np.mean(values):.3f} [{np.min(values):.3f}, {np.max(values):.3f}] n={len(values)}'


def plot_masks(masks, out):
    seeds = sorted({s for _, s in masks})
    fig, axes = plt.subplots(2, len(seeds), figsize=(3.4 * len(seeds), 7), squeeze=False)
    for row, anat in enumerate(ANATOMIES):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            if (anat, seed) in masks:
                im = ax.imshow(masks[anat, seed], cmap='magma', vmin=0, vmax=1)
            ax.set_title(f'{anat} seed {seed}')
            ax.axis('off')
    fig.colorbar(im, ax=axes, shrink=0.6, label='learned probability')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mean_difference(masks, out):
    mean = {a: np.mean([m for (x, _), m in masks.items() if x == a], axis=0) for a in ANATOMIES}
    diff = mean['brain'] - mean['knee']
    lim = np.abs(diff).max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, anat in zip(axes[:2], ANATOMIES):
        im = ax.imshow(mean[anat], cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'{anat} mean over seeds')
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label='learned probability')
    im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[2].set_title('brain - knee')
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.axis('off')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_profiles(masks, bins, out):
    centers = (np.arange(RADIAL_BINS + 1) + 0.5) / RADIAL_BINS
    panels = [
        ('radial profile', lambda m: radial_profile(m, bins), centers[:RADIAL_BINS], 'normalized k-space radius'),
        ('mean over kx (profile along ky)', lambda m: m.mean(axis=1), np.linspace(-1, 1, 256), 'ky'),
        ('mean over ky (profile along kx)', lambda m: m.mean(axis=0), np.linspace(-1, 1, 256), 'kx'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, fn, x, xlabel) in zip(axes, panels):
        for anat in ANATOMIES:
            curves = [fn(m)[:len(x)] for (a, _), m in masks.items() if a == anat]
            for c in curves:
                ax.plot(x, c, color=COLORS[anat], lw=1, alpha=0.35)
            ax.plot(x, np.mean(curves, axis=0), color=COLORS[anat], lw=2, label=anat)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('learned probability')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_correlation(transformed, out):
    keys = sorted(transformed, key=lambda k: (ANATOMIES.index(k[0]), k[1]))
    matrix = np.array([[pearson(transformed[a], transformed[b]) for b in keys] for a in keys])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(matrix, cmap='Blues', vmin=min(0, matrix.min()), vmax=1)
    ax.set_xticks(range(len(keys)), [f'{a} s{s}' for a, s in keys], rotation=45, ha='right')
    ax.set_yticks(range(len(keys)), [f'{a} s{s}' for a, s in keys])
    for i, j in itertools.product(range(len(keys)), repeat=2):
        ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                color='white' if matrix[i, j] > 0.6 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, label='Pearson r (smoothed masks)')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/loupe_anatomy'))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--smooth-sigma', type=float, default=2.0)
    args = parser.parse_args()
    out = args.out or args.root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)

    masks, epochs = {}, {}
    for anat in ANATOMIES:
        for run_dir in sorted((args.root / anat).glob('seed_*')):
            if not (run_dir / 'best_model.pt').exists():
                continue
            seed = int(run_dir.name.split('_')[1])
            masks[anat, seed], done, total = load_mask(run_dir)
            epochs[f'{anat}_s{seed}'] = f'{done}/{total}'
    print('checkpoint epoch (best val loss) / planned epochs:', epochs)
    if any(done.split('/')[0] != done.split('/')[1] for done in epochs.values()):
        print('WARNING: some runs have not reached their final epoch; masks are not final.')

    shape = next(iter(masks.values())).shape
    bins = radial_bins(shape)
    transforms = {
        'raw': lambda m: m,
        f'smoothed (sigma={args.smooth_sigma:g})': lambda m: gaussian_filter(m, args.smooth_sigma),
        'radial residual (smoothed)': lambda m: gaussian_filter(radial_residual(m, bins), args.smooth_sigma),
    }

    summary = {'checkpoints': epochs, 'mean_probability': {
        a: float(np.mean([m.mean() for (x, _), m in masks.items() if x == a])) for a in ANATOMIES}}
    transformed_for_plot = None
    for name, transform in transforms.items():
        within, cross, transformed = pair_stats(masks, transform)
        if name.startswith('smoothed'):
            transformed_for_plot = transformed
        all_within = within['brain'] + within['knee']
        print(f'\n[{name}] Pearson r')
        for anat in ANATOMIES:
            print(f'  within {anat:5s}: {span(within[anat])}')
        print(f'  brain vs knee: {span(cross)}')
        gap = float(np.min(all_within) - np.max(cross))
        print(f'  min(within) - max(cross) = {gap:+.3f}  '
              f'({"every seed pair beats every brain-knee pair" if gap > 0 else "distributions overlap"})')
        summary[name] = {'within_brain': within['brain'], 'within_knee': within['knee'],
                         'cross': cross, 'min_within_minus_max_cross': gap}

    plot_masks(masks, out / 'masks.png')
    plot_mean_difference(masks, out / 'mean_and_difference.png')
    plot_profiles(masks, bins, out / 'profiles.png')
    plot_correlation(transformed_for_plot, out / 'correlation.png')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
