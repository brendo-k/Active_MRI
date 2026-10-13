"""Per-seed detrended column profile of the learned LOUPE mask: are the vertical streaks reproducible?

Column profile = mean of p(k) over rows (the readout axis), minus its 13-column moving average.
Columns are the phase-encode axis of the fastMRI k-space.

    python scripts/plot_column_ripple.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

import _path  # noqa: F401
from plot_mean_masks import load_probability

GROUPS = [
    ('brain, 12 casc', 'runs/loupe_anatomy/brain/seed_*'),
    ('brain, 1 casc', 'runs/loupe_casc1/brain/seed_*'),
    ('brain, 1 casc, phase scramble 1.5 rad', 'runs/loupe_casc1_phasescramble_strong/brain/seed_*'),
    ('knee, 12 casc', 'runs/loupe_anatomy/knee/seed_*'),
]
SMOOTH_WIDTH = 13
COLUMNS = slice(8, 120)


def detrended_column_profile(mask):
    profile = mask.mean(axis=0)
    return profile - uniform_filter1d(profile, SMOOTH_WIDTH, mode='nearest')


fig, axes = plt.subplots(len(GROUPS), 1, figsize=(12, 2.4 * len(GROUPS)), sharex=True, sharey=True)
for ax, (label, pattern) in zip(axes, GROUPS):
    for run in sorted(Path('.').glob(pattern)):
        ax.plot(np.arange(256)[COLUMNS], detrended_column_profile(load_probability(run))[COLUMNS],
                lw=1, label=f'seed {run.name.split("_")[-1]}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(label, fontsize=10, loc='left')
    ax.set_ylabel('detrended\nmean p(k)')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
axes[-1].set_xlabel('kx column (phase-encode axis)')
fig.tight_layout()
out = Path('runs/loupe_anatomy/analysis/column_ripple.png')
fig.savefig(out, dpi=130)
print(out)
