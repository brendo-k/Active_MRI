"""Plot the per-column loss gradient from scripts/diagnostics/loupe_column_gradient.py against the mask's streak pattern.

    python scripts/plots/plot_column_gradient.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

import _path  # noqa: F401
from plot_mean_masks import load_probability

ROOT = Path('runs/loupe_anatomy')
COLUMNS = slice(8, 120)
SMOOTH_WIDTH = 13
PANELS = [
    ('column_gradient_brain_seed0_destreaked.npy', 'destreaked mask, original validation data'),
    ('column_gradient_brain_seed0_destreaked_scramble1.npy', 'destreaked mask, validation k-space phase-scrambled (1 rad, 6-16 cycles/FOV)'),
]


def detrend(v):
    return v - uniform_filter1d(v, SMOOTH_WIDTH, mode='nearest')


def zscore(v):
    return (v - v.mean()) / v.std()


ripple = detrend(load_probability(ROOT / 'brain/seed_0').mean(axis=0))
columns = np.arange(256)[COLUMNS]
fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 3.3 * len(PANELS)), sharex=True)
for ax, (name, title) in zip(axes, PANELS):
    gradient = detrend(np.load(ROOT / 'analysis' / name).mean(axis=0))
    r = np.corrcoef(gradient[COLUMNS], ripple[COLUMNS])[0, 1]
    ax.plot(columns, zscore(ripple[COLUMNS]), color='C3', lw=1.2, label='learned mask ripple (seed 0)')
    ax.plot(columns, zscore(-gradient[COLUMNS]), color='C0', lw=1.2, label='- dLoss/dp per column (up = loss wants more probability)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set(title=f'{title}\nr(gradient, ripple) = {r:+.2f}', ylabel='z-score')
    ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('kx column (phase-encode axis)')
fig.tight_layout()
out = ROOT / 'analysis' / 'column_gradient.png'
fig.savefig(out, dpi=130)
print(out)
