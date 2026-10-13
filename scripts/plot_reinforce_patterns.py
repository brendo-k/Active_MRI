"""Plot the sampling patterns learned by the R=8 brain+knee REINFORCE runs, with LOUPE as reference.

Per run (one row):
  inclusion   how often each k-space location is picked: the policy's marginal, from Monte Carlo
              Plackett-Luce draws (REINFORCE) or the rescaled Bernoulli probability (LOUPE)
  one draw    a single sampled mask, what validation now deploys (reinforce_eval='sample')
  top-k       the deterministic top-k mask, what validation deployed before 2026-10-07
  residual    inclusion minus its own ring average, Gaussian-smoothed: structure beyond the
              radial density profile (anisotropy, left/right asymmetry)
Plus one panel comparing every run's radial density profile.

Reads only the mask parameters from each best_model.pt (no data, CPU only).

    python scripts/plot_reinforce_patterns.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import _path  # noqa: F401
from activemri.utils.mask_binarization import RescaleProbMap
from compare_loupe_reinforce_r8 import pl_draws, preselect_region, radius_grid, topk_mask

RUNS = [
    # (label, checkpoint dir, kind)
    ('LOUPE, 50 ep', 'checkpoints/loupe2d_varnet_r8_brain_knee', 'loupe'),
    ('REINFORCE uniform init, 100 ep', 'checkpoints/reinforce2d_varnet_r8_brain_knee_v2_100ep', 'reinforce'),
    ('REINFORCE vd2.5 init, 100 ep', 'checkpoints/reinforce2d_varnet_r8_brain_knee_vd2.5_100ep', 'reinforce'),
    ('REINFORCE vd2.5, policy lr 0.002', 'checkpoints/reinforce2d_varnet_r8_brain_knee_vd2.5_plr0.002_100ep', 'reinforce'),
]
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # categorical slots 1-4, fixed order
MC_DRAWS = 64
RESIDUAL_SMOOTH = 3.0
PROFILE_BINS = np.linspace(0, 1.42, 72)
OUT_DIR = Path('reports/reinforce_patterns_r8')


def run_pattern(path, kind):
    ckpt = torch.load(Path(path) / 'best_model.pt', map_location='cpu', weights_only=False)
    opts = ckpt['options']
    logits = ckpt['model']['sampler.mask_net.mask'].float().squeeze()
    h, w = logits.shape
    excluded = preselect_region(h, w, opts.preselect_num)
    budget = int(round((1 / opts.accelerations[0]) * h * w - excluded.sum().item()))
    torch.manual_seed(0)
    if kind == 'loupe':
        prob = torch.sigmoid(opts.slope * logits)
        inclusion = RescaleProbMap(prob[None, None], budget / (h * w))[0, 0].clamp(max=1)
        draw = (torch.rand_like(inclusion) < inclusion).float()
    else:
        draws = pl_draws(logits, budget, excluded, MC_DRAWS)
        inclusion, draw = draws.mean(0), draws[0]
    inclusion = inclusion.masked_fill(excluded, 1.0)
    draw = draw.masked_fill(excluded, 1.0)
    top = topk_mask(logits, budget, excluded).masked_fill(excluded, 1.0)
    return {'inclusion': inclusion.numpy(), 'draw': draw.numpy(), 'topk': top.numpy(),
            'epoch': ckpt.get('epoch', 0) + 1, 'budget': budget}


def ring_profile(field, radius):
    index = np.clip(np.digitize(radius, PROFILE_BINS) - 1, 0, len(PROFILE_BINS) - 2)
    totals = np.bincount(index.ravel(), weights=field.ravel(), minlength=len(PROFILE_BINS) - 1)
    counts = np.bincount(index.ravel(), minlength=len(PROFILE_BINS) - 1)
    profile = totals / np.maximum(counts, 1)
    return profile, profile[index]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns = [run_pattern(path, kind) for _, path, kind in RUNS]
    h, w = patterns[0]['inclusion'].shape
    radius = radius_grid(h, w).numpy()

    residuals, profiles, summary = [], [], {}
    for (label, path, _), pattern in zip(RUNS, patterns):
        profile, ring_mean = ring_profile(pattern['inclusion'], radius)
        residuals.append(gaussian_filter(pattern['inclusion'] - ring_mean, RESIDUAL_SMOOTH))
        profiles.append(profile)
        inc = pattern['inclusion']
        summary[label] = {
            'checkpoint': path, 'epoch': pattern['epoch'], 'budget': pattern['budget'],
            'kx_negative_share': float(inc[:, :w // 2].sum() / inc.sum()),
            'ky_vs_kx_extent': float((inc * np.abs(np.arange(h) - h // 2)[:, None]).sum()
                                     / (inc * np.abs(np.arange(w) - w // 2)[None, :]).sum()),
            'corner_share_r_gt_1': float(inc[radius > 1].sum() / inc.sum()),
            'residual_smoothed_abs_max': float(np.abs(residuals[-1]).max()),
        }
    residual_limit = max(np.abs(r).max() for r in residuals)

    columns = ['inclusion probability', 'one sampled mask (val, new)', 'top-k mask (val, old)',
               f'residual vs ring mean (σ={RESIDUAL_SMOOTH:g} px)']
    fig = plt.figure(figsize=(17, 4.0 * len(RUNS) + 4.2), facecolor='white')
    grid = fig.add_gridspec(len(RUNS) + 1, 4, height_ratios=[1] * len(RUNS) + [1.05], hspace=0.32, wspace=0.12,
                            top=0.95, bottom=0.04)
    for row, ((label, _, _), pattern, residual) in enumerate(zip(RUNS, patterns, residuals)):
        fields = [pattern['inclusion'], pattern['draw'], pattern['topk'], residual]
        for col, field in enumerate(fields):
            ax = fig.add_subplot(grid[row, col])
            if col == 3:
                image = ax.imshow(field, cmap='RdBu_r', vmin=-residual_limit, vmax=residual_limit)
            elif col == 0:
                image = ax.imshow(field, cmap='Blues', vmin=0, vmax=1)
            else:
                image = ax.imshow(field, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=11, color='#0b0b0b')
            if col == 0:
                ax.set_ylabel(f"{label}\n(epoch {pattern['epoch']})", fontsize=10.5, color='#0b0b0b')
            if row == len(RUNS) - 1:
                ax.set_xlabel('kx (phase-encode columns) →', fontsize=9, color='#52514e')
            if col in (0, 3):
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=8, colors='#52514e')

    ax = fig.add_subplot(grid[len(RUNS), 1:3])
    centers = 0.5 * (PROFILE_BINS[1:] + PROFILE_BINS[:-1])
    for (label, _, _), profile, color in zip(RUNS, profiles, SERIES_COLORS):
        ax.plot(centers, profile, color=color, lw=2, label=label)
    ax.axvline(1.0, color='#c3c2b7', lw=1, ls='--')
    ax.text(1.01, 0.92, 'edge of k-space along axes;\nbeyond = corners', fontsize=8, color='#52514e',
            transform=ax.get_xaxis_transform(), va='top')
    ax.set(xlabel='normalized k-space radius', ylabel='sampled fraction of ring', ylim=(0, 1.02),
           title='radial density profile (inclusion probability, ring average)')
    ax.grid(color='#ecebe8', lw=0.8); ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle('Learned sampling patterns, R=8 brain+knee, 12-cascade VarNet (best checkpoints)',
                 fontsize=14, color='#0b0b0b', y=0.995)
    fig.savefig(OUT_DIR / 'patterns.png', dpi=110, bbox_inches='tight')
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
