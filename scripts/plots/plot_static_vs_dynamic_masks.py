"""Sampling-probability maps of static (PI / CAIPI, Poisson) vs dynamic (LOUPE, anatomy-conditioned LOUPE) masks, brain and knee, for CG-SENSE (R=4)
and PromptMR (R=8) in one figure.

Reads the outputs of compute_mask_probabilities.py (mean probability over validation slices of each anatomy). Every map gets the same
small box filter so a binary lattice and a learned probability map are drawn on the same footing; the colour scale tops out at twice the
nominal sampling rate of each reconstructor, so the always-sampled centre saturates.

    python scripts/plots/plot_static_vs_dynamic_masks.py
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

COLUMNS = ('pi', 'poisson', 'loupe', 'loupe_anat')
TITLES = {'pi': 'PI (CG, 2x2) / CAIPI (PromptMR, 4x2)', 'poisson': 'Poisson-disc', 'loupe': 'LOUPE', 'loupe_anat': 'LOUPE anatomy'}
ANATOMIES = (('brain', 'Brain (FLAIR)'), ('knee-PD', 'Knee (PD)'))
SMOOTH = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cg', type=Path, default=Path('reports/mask_probabilities/cg.npz'))
    parser.add_argument('--promptmr', type=Path, default=Path('reports/mask_probabilities/promptmr.npz'))
    parser.add_argument('--out', type=Path, default=Path('reports/mask_probabilities/static_vs_dynamic_probabilities'))
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    sources = (('CG-SENSE, R=4', np.load(args.cg), 4), ('PromptMR, R=8', np.load(args.promptmr), 8))
    fig, axes = plt.subplots(4, 4, figsize=(13, 13), gridspec_kw={'hspace': 0.06, 'wspace': 0.03})
    for block, (recon, saved, acceleration) in enumerate(sources):
        vmax = 2.0 / acceleration
        for row, (condition, label) in enumerate(ANATOMIES):
            for j, name in enumerate(COLUMNS):
                ax = axes[2 * block + row, j]
                shown = ax.imshow(uniform_filter(saved[f'{condition}__{name}'], size=SMOOTH, mode='nearest'), cmap='magma', vmin=0, vmax=vmax,
                                  interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if 2 * block + row == 0:
                    ax.set_title(TITLES[name], fontsize=11)
            axes[2 * block + row, 0].set_ylabel(f'{recon}\n{label}  (n={int(saved[f"{condition}__n"])})', fontsize=11)
        colorbar = fig.colorbar(shown, ax=axes[2 * block:2 * block + 2, :], fraction=0.02, pad=0.015)
        colorbar.set_label(f'sampling probability ({SMOOTH}x{SMOOTH} box-smoothed)', fontsize=10)
    for x, text in ((0.30, 'Static'), (0.68, 'Dynamic (learned)')):
        fig.text(x, 0.92, text, ha='center', fontsize=14, fontweight='bold')
    fig.savefig(args.out.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(args.out.with_suffix('.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
