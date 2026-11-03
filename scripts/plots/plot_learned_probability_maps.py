"""Learned sampling-probability maps (LOUPE and anatomy-conditioned LOUPE) for CG-SENSE (R=4) and PromptMR (R=8), brain and knee, in one figure.

Reads the outputs of compute_mask_probabilities.py (mean probability over validation slices of each anatomy). Every map gets the same
small box filter as plot_static_vs_dynamic_masks.py, so the binary CG-SENSE lattice and the learned PromptMR map read on the same footing.
Each reconstructor has its own colour scale, topping out at 2x its nominal sampling rate so the always-sampled centre saturates.

    python scripts/plots/plot_learned_probability_maps.py
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

SMOOTH = 5
COLUMNS = (('loupe', 'LOUPE'), ('loupe_anat', 'LOUPE (anatomy-conditioned)'))
ANATOMIES = (('brain', 'Brain (FLAIR)'), ('knee-PD', 'Knee (PD)'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cg', type=Path, default=Path('reports/mask_probabilities/cg.npz'))
    parser.add_argument('--promptmr', type=Path, default=Path('reports/mask_probabilities/promptmr.npz'))
    parser.add_argument('--out', type=Path, default=Path('reports/mask_probabilities/learned_probability_maps'))
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--cmap', default='magma', help='matplotlib colormap for the probability maps')
    return parser.parse_args()


def main():
    args = parse_args()
    sources = (('CG-SENSE, R=4', np.load(args.cg), 4), ('PromptMR, R=8', np.load(args.promptmr), 8))
    fig, axes = plt.subplots(4, 2, figsize=(7, 13), gridspec_kw={'hspace': 0.08, 'wspace': 0.05})
    for block, (recon, saved, acceleration) in enumerate(sources):
        vmax = 2.0 / acceleration
        for row, (condition, label) in enumerate(ANATOMIES):
            ax_row = 2 * block + row
            for j, (name, title) in enumerate(COLUMNS):
                ax = axes[ax_row, j]
                smoothed = uniform_filter(saved[f'{condition}__{name}'], size=SMOOTH, mode='nearest')
                shown = ax.imshow(smoothed, cmap=args.cmap, vmin=0, vmax=vmax, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if ax_row == 0:
                    ax.set_title(title, fontsize=11)
            axes[ax_row, 0].set_ylabel(f'{recon}\n{label}  (n={int(saved[f"{condition}__n"])})', fontsize=11)
        colorbar = fig.colorbar(shown, ax=axes[2 * block:2 * block + 2, :], fraction=0.05, pad=0.03)
        colorbar.set_label('sampling probability', fontsize=10)
    fig.savefig(args.out.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(args.out.with_suffix('.pdf'), bbox_inches='tight')
    print('wrote', args.out.with_suffix('.png'))


if __name__ == '__main__':
    main()
