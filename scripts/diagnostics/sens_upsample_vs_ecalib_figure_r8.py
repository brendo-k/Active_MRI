"""Figure for sens_upsample_vs_ecalib_r8.py: the four final masks, their difference from conv, and radial density profiles."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from anatcond_cgsense_cross_eval_r8 import blurred, radius_map

OUT_DIR = Path('reports/sens_upsample_r8')
CROP = 96


def centre(p):
    c = p.shape[0] // 2
    return p[c - CROP:c + CROP, c - CROP:c + CROP]


def main():
    masks = torch.load(OUT_DIR / 'ecalib_comparison_masks.pt')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, p) in zip(axes[0], masks.items()):
        ax.imshow(centre(p), vmin=0, vmax=1, cmap='viridis'); ax.set_title(f'{name} p(k), mean {p.mean():.3f}'); ax.axis('off')
    for ax, (name, p) in zip(axes[1][:3], list(masks.items())[1:]):
        diff = centre(blurred(p, 2) - blurred(masks['conv'], 2))
        ax.imshow(diff, vmin=-0.3, vmax=0.3, cmap='RdBu_r'); ax.set_title(f'{name} - conv (2 px blur)'); ax.axis('off')
    radius = radius_map(masks['conv'].shape)
    bins = np.arange(0, 129, 4)
    for name, p in masks.items():
        profile = [float(p[(radius >= lo) & (radius < lo + 4)].mean()) for lo in bins[:-1]]
        axes[1][3].plot(bins[:-1] + 2, profile, label=name)
    axes[1][3].set_xlabel('k-space radius (px)'); axes[1][3].set_ylabel('mean p(k)'); axes[1][3].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / 'ecalib_comparison_masks.png', dpi=110)


if __name__ == '__main__':
    main()
