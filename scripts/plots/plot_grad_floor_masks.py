"""Ending LOUPE-2D masks with vs without the gradient floor (R=8, brain+knee VarNet).

For each run's checkpoint: p(k) = sigmoid(slope * logits), the learned free parameter before budget
rescaling, and the effective per-location sampling probability after the sampler's RescaleProbMap
(target mean 1/R over the whole grid, with the 10x10 preselected centre fixed at 1). k-space is
DC-centred, 256 x 256. Also: floor - nofloor difference, radial profile of the effective probability,
how binary each mask is, and the saturated-logit fraction (|slope * w| > log 99).

    python scripts/plots/plot_grad_floor_masks.py --checkpoint last_model.pt
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.utils.mask_binarization import RescaleProbMap

RUN_DIRS = {
    'floor 0.05': Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05'),
    'no floor': Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_nofloor'),
}
SATURATED_LOGIT = math.log(99)
RADIAL_BINS = 64


def load_mask(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    options = ckpt['options']
    logits = options.slope * ckpt['model']['sampler.mask_net.mask'].float().squeeze()
    probability = torch.sigmoid(logits)

    height, width = probability.shape
    half = options.preselect_num // 2
    preselected = torch.zeros(height, width)
    preselected[height // 2 - half:height // 2 + half, width // 2 - half:width // 2 + half] = 1
    sparsity = 1.0 / options.accelerations[0] - preselected.sum() / preselected.numel()
    rescaled = RescaleProbMap((probability * (1 - preselected))[None, None], sparsity,
                              (1 - preselected)[None, None])[0, 0]
    effective = torch.clamp(rescaled + preselected, max=1)
    return {'epoch': int(ckpt['epoch']), 'logits': logits.numpy(), 'p': probability.numpy(),
            'effective': effective.numpy()}


def radial_profile(image):
    height, width = image.shape
    y, x = np.indices(image.shape)
    radius = np.hypot(y - height // 2, x - width // 2) / (height // 2)
    edges = np.linspace(0, np.sqrt(2), RADIAL_BINS + 1)
    which = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, RADIAL_BINS - 1)
    sums = np.bincount(which, image.ravel(), RADIAL_BINS)
    counts = np.bincount(which, minlength=RADIAL_BINS)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres[counts > 0], (sums / np.maximum(counts, 1))[counts > 0]


def mask_stats(mask):
    p, effective, logits = mask['p'], mask['effective'], mask['logits']
    return {
        'epoch': mask['epoch'],
        'mean_p': float(p.mean()),
        'mean_effective': float(effective.mean()),
        'p_below_0.01_frac': float((p < 0.01).mean()),
        'p_above_0.99_frac': float((p > 0.99).mean()),
        'effective_below_0.01_frac': float((effective < 0.01).mean()),
        'effective_above_0.99_frac': float((effective > 0.99).mean()),
        'saturated_logit_frac': float((np.abs(logits) > SATURATED_LOGIT).mean()),
        'logit_abs_max': float(np.abs(logits).max()),
        'logit_abs_p99': float(np.quantile(np.abs(logits), 0.99)),
        'effective_entropy_bits_per_loc': float(np.mean(
            -effective * np.log2(np.clip(effective, 1e-12, 1)) - (1 - effective) * np.log2(np.clip(1 - effective, 1e-12, 1)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, default=Path('reports/grad_floor_r8/masks_last.png'))
    args = parser.parse_args()

    masks = {label: load_mask(run_dir / args.checkpoint) for label, run_dir in RUN_DIRS.items()}
    labels = list(masks)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for row, label in enumerate(labels):
        m = masks[label]
        im = axes[row, 0].imshow(m['p'], cmap='viridis', vmin=0, vmax=1)
        axes[row, 0].set_title(f'{label} (epoch {m["epoch"]})\np(k) = sigmoid(slope * w)')
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046)
        im = axes[row, 1].imshow(m['effective'], cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'{label}\neffective sampling probability (mean {m["effective"].mean():.3f})')
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)

    difference = masks[labels[0]]['effective'] - masks[labels[1]]['effective']
    limit = np.abs(difference).max()
    im = axes[0, 2].imshow(difference, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[0, 2].set_title(f'effective: {labels[0]} - {labels[1]}')
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    for label in labels:
        r, profile = radial_profile(masks[label]['effective'])
        axes[0, 3].plot(r, profile, label=label)
        axes[1, 2].hist(masks[label]['logits'].ravel(), bins=200, histtype='step', label=label, log=True)
        axes[1, 3].hist(masks[label]['effective'].ravel(), bins=100, histtype='step', label=label, log=True)
    axes[0, 3].set(title='radial profile of effective probability', xlabel='|k| / k_max', ylabel='mean probability')
    axes[1, 2].axvline(SATURATED_LOGIT, color='k', lw=0.5)
    axes[1, 2].axvline(-SATURATED_LOGIT, color='k', lw=0.5)
    axes[1, 2].set(title='slope * w (lines: saturation at +/- log 99)', xlabel='logit', ylabel='count')
    axes[1, 3].set(title='effective probability histogram', xlabel='probability', ylabel='count')
    for ax in (axes[0, 3], axes[1, 2], axes[1, 3]):
        ax.legend()
    for ax in axes[:, :2].ravel().tolist() + [axes[0, 2]]:
        ax.set(xlabel='kx', ylabel='ky')
    fig.suptitle(f'LOUPE-2D R=8 brain+knee, {args.checkpoint}: gradient floor 0.05 vs none')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches='tight')

    stats = {label: mask_stats(m) for label, m in masks.items()}
    a, b = masks[labels[0]]['effective'], masks[labels[1]]['effective']
    stats['comparison'] = {
        'effective_correlation': float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        'effective_l1_over_l1': float(np.abs(a - b).sum() / (a.sum() + b.sum())),
        'top_budget_overlap_frac': float(_top_overlap(a, b)),
    }
    args.out.with_suffix('.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def _top_overlap(a, b):
    """Share of the top-(budget) locations that the two masks have in common."""
    k = int(round(a.sum()))
    top_a = set(np.argsort(a.ravel())[-k:])
    top_b = set(np.argsort(b.ravel())[-k:])
    return len(top_a & top_b) / k


if __name__ == '__main__':
    main()
