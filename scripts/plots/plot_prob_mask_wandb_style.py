"""Learned LOUPE sampling probability of the CG-SENSE and VarNet runs, drawn the way the trainer's wandb `prob_mask` panel
draws it, with the pre-selected centre forced to 1.

The panel shows the sampler's unscaled probability, sigmoid(slope * w), straight from the trained mask parameter: no budget rescaling, no
slice averaging, no smoothing, inferno, and a colour scale fitted to each image. The pre-selected centre (the preselect_num x preselect_num
block at k-space DC, which the sampler always acquires) is set to 1 so it reads as always sampled.

Layout: 2 x 4. Each reconstructor takes a 2 x 2 block: rows are brain and knee, columns are non-adaptive (one LOUPE mask for every anatomy)
and adaptive (anatomy-conditioned LOUPE, one mask per anatomy).

Runs on a CPU node (loads four ~750 MB checkpoints):
    srun -p cpu -A hpc_core -q cpu -n1 -c 4 --mem=16G --time=10:00 python scripts/plots/plot_prob_mask_wandb_style.py
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import _path  # noqa: F401
from activemri.utils.mask_binarization import floored_grad_sigmoid

LOG_ROOT = Path('/home/remote/bkadota/scratch/logs/active_mri')
NON_ADAPTIVE = 'Non-adaptive (LOUPE)'
ADAPTIVE = 'Adaptive (anatomy-conditioned LOUPE)'
RECONS = (
    ('CG-SENSE, R=4', ((NON_ADAPTIVE, 'cg_masks_single-2413163'), (ADAPTIVE, 'cg_masks_routed-2413164'))),
    ('VarNet, R=8',((NON_ADAPTIVE, 'promptmr_loupe_trim'), (ADAPTIVE, 'promptmr_loupe_anat_trim'))),
)
ANATOMIES = (('brain', 'Brain (FLAIR)'), ('knee', 'Knee (PD)'))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, default=Path('reports/mask_probabilities/prob_mask_cg_promptmr'))
    parser.add_argument('--dpi', type=int, default=600)
    return parser.parse_args()


def centre_block(shape, preselect_num):
    """The pre-selected centre: the sampler's n x n corners, fftshifted, i.e. a 2n x 2n block around DC (n = preselect_num // 2)."""
    n = preselect_num // 2
    block = np.zeros(shape, dtype=bool)
    h, w = shape
    block[h // 2 - n:h // 2 + n, w // 2 - n:w // 2 + n] = True
    return block


def unscaled_probability(path, anatomy, checkpoint):
    """The trained sampler's probability map for one anatomy, as the wandb panel draws it (centre set to 1), and the checkpoint's epoch."""
    ck = torch.load(path / checkpoint, map_location='cpu', weights_only=False)
    weights = ck['model']['sampler.mask_net.mask'].float()
    options = ck['options']
    if weights.shape[0] > 1:
        index = list(options.anatomy_classes).index(anatomy)
        weights = weights[index:index + 1]
    prob = floored_grad_sigmoid(weights, options.slope, 0.0).squeeze().numpy()
    prob[centre_block(prob.shape, options.preselect_num)] = 1.0
    return prob, ck.get('epoch')


def main():
    args = parse_args()
    fig, axes = plt.subplots(len(ANATOMIES), 2 * len(RECONS), figsize=(3.6 * 2 * len(RECONS), 4.0 * len(ANATOMIES)), layout='constrained')
    for block, (recon, runs) in enumerate(RECONS):
        for method, (title, run) in enumerate(runs):
            col = 2 * block + method
            for row, (anatomy, label) in enumerate(ANATOMIES):
                prob, epoch = unscaled_probability(LOG_ROOT / run, anatomy, args.checkpoint)
                ax = axes[row, col]
                shown = ax.imshow(prob, cmap='inferno', aspect='equal', vmin=0, vmax=1)
                ax.axis('off')
                if row == 0:
                    ax.set_title(title, fontsize=11)
                if col == 0:
                    ax.text(-0.04, 0.5, label, transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
                fig.colorbar(shown, ax=ax, fraction=0.046)
                print(run, anatomy, 'epoch', epoch, 'min %.4f max %.4f mean %.4f' % (prob.min(), prob.max(), prob.mean()), flush=True)
    fig.canvas.draw()
    for block, (recon, _) in enumerate(RECONS):
        left, right = axes[0, 2 * block].get_position(), axes[0, 2 * block + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, 0.99, recon, ha='center', va='top', fontsize=13, fontweight='bold')
    fig.savefig(args.out.with_suffix('.png'), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(args.out.with_suffix('.pdf'), bbox_inches='tight')
    print('wrote', args.out.with_suffix('.png'))


if __name__ == '__main__':
    main()
