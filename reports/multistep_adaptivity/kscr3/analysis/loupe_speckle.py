"""Raw, unsmoothed LOUPE probability maps, sigmoid(slope * logits) per pixel before the budget rescale: kscr2's one-step
LOUPE (its PD map) against kscr3's L, full grid, a zoomed crop and the histogram of probabilities.
Usage: python loupe_speckle.py <out.png>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DATA = Path.home() / 'data/seqmri_checkpoints'
RUNS = [('kscr2 one-step LOUPE (PD map), ~17,000 steps', DATA / 'multistep_kscr2/T_1_s42_kscr2/last_model.pt'),
        ('kscr3 one-step LOUPE (L), ~3,400 steps', DATA / 'multistep_kscr3/L_1_s42_kscr3/last_model.pt')]
SLOPE = 10.0
CROP = (slice(110, 190), slice(60, 140))  # rows, columns of the zoom


def probabilities(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    logits = SLOPE * checkpoint['model']['sampler.mask_net.mask'].float()[0, ..., 0].numpy()
    return 1 / (1 + np.exp(-logits)), int(checkpoint['epoch']) + 1


def main(out):
    fig, axes = plt.subplots(len(RUNS), 3, figsize=(15, 5 * len(RUNS)))
    for r, (title, path) in enumerate(RUNS):
        p, epoch = probabilities(path)
        axes[r, 0].imshow(p, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        axes[r, 0].add_patch(plt.Rectangle((CROP[1].start, CROP[0].start), CROP[1].stop - CROP[1].start,
                                           CROP[0].stop - CROP[0].start, fill=False, edgecolor='cyan', linewidth=1.5))
        axes[r, 0].set_ylabel(f'{title}\nepoch {epoch}', fontsize=12)
        axes[r, 1].imshow(p[CROP], cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        for ax in axes[r, :2]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 2].hist(p.ravel(), bins=50, range=(0, 1), color='#c2410c', log=True)
        axes[r, 2].set_xlabel('per-pixel probability before the budget rescale')
        axes[r, 2].set_ylabel('pixels (log scale)')
        axes[r, 2].text(0.5, 0.92, f'{np.mean(p < 0.01):.0%} of pixels below 0.01, {np.mean(p > 0.99):.1%} above 0.99',
                        transform=axes[r, 2].transAxes, ha='center', fontsize=11)
    axes[0, 0].set_title('Whole grid (no smoothing)', fontsize=13)
    axes[0, 1].set_title('Zoom (cyan box), one cell per pixel', fontsize=13)
    axes[0, 2].set_title('Histogram of probabilities', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
