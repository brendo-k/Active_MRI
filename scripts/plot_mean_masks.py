"""Mean learned LOUPE mask p(k) and its Hermitian asymmetry p(k) - p(-k), one column per run group.

Each group is LABEL=GLOB, a glob over run directories holding best_model.pt; the mean is over
the matched seeds. p(k) = sigmoid(slope * logits), the LOUPE free parameter before the sampler's
budget rescaling. k-space is DC-centred, so p(-k) maps index i to -i mod N on both axes.

    python scripts/plot_mean_masks.py \
        'brain, 12 casc=runs/loupe_anatomy/brain/seed_*' \
        'knee, 12 casc=runs/loupe_anatomy/knee/seed_*' \
        'brain, 1 casc=runs/loupe_casc1/brain/seed_*'
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401

DEFAULT_GROUPS = [
    'brain, 12 casc=runs/loupe_anatomy/brain/seed_*',
    'knee, 12 casc=runs/loupe_anatomy/knee/seed_*',
    'brain, 1 casc=runs/loupe_casc1/brain/seed_*',
]


def load_probability(run_dir):
    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    key = next(k for k in ckpt['model'] if k.endswith('mask_net.mask'))
    logits = ckpt['model'][key].float().squeeze().numpy()
    return 1.0 / (1.0 + np.exp(-ckpt['options'].slope * logits))


def hermitian_partner(mask):
    return np.roll(mask[::-1, ::-1], shift=(1, 1), axis=(0, 1))


def asymmetry_stats(mean_mask):
    centre = mean_mask.shape[1] // 2
    left, right = mean_mask[:, :centre].sum(), mean_mask[:, centre:].sum()
    partner = hermitian_partner(mean_mask)
    return {
        'mean_probability': float(mean_mask.mean()),
        'right_half_mass_fraction': float(right / (left + right)),
        'asymmetry': float(np.abs(mean_mask - partner).sum() / (mean_mask + partner).sum()),
    }


def parse_group(spec):
    label, pattern = spec.rsplit('=', 1)
    runs = sorted(Path('.').glob(pattern))
    if not runs:
        raise SystemExit(f'no runs match {pattern}')
    return label, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('groups', nargs='*', default=DEFAULT_GROUPS)
    parser.add_argument('--title', default='Learned LOUPE masks (R=4)')
    parser.add_argument('--out', type=Path, default=Path('runs/loupe_anatomy/analysis/mean_masks_p_k.png'))
    args = parser.parse_args()

    groups = [parse_group(spec) for spec in args.groups]
    fig, axes = plt.subplots(2, len(groups), figsize=(5.6 * len(groups), 7.8), squeeze=False)
    stats = {}
    for col, (label, runs) in enumerate(groups):
        mean_mask = np.mean([load_probability(r) for r in runs], axis=0)
        difference = mean_mask - hermitian_partner(mean_mask)
        stats[label] = {'runs': [str(r) for r in runs], **asymmetry_stats(mean_mask)}

        axes[0, col].imshow(mean_mask, cmap='viridis')
        axes[0, col].set_title(f'{label} ({len(runs)} seed{"s" * (len(runs) > 1)})\nmean p(k)', fontsize=10)
        limit = np.abs(difference).max()
        axes[1, col].imshow(difference, cmap='RdBu', vmin=-limit, vmax=limit)
        axes[1, col].set_title('mean p(k) - p(-k)', fontsize=10)
        for ax in axes[:, col]:
            ax.set(xlabel='kx (columns)', ylabel='ky')
    fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    args.out.with_suffix('.json').write_text(json.dumps(stats, indent=2))
    for label, s in stats.items():
        print(f'{label:16s} mean p {s["mean_probability"]:.3f}  right-half mass {s["right_half_mass_fraction"]:.3f}  '
              f'asymmetry {s["asymmetry"]:.3f}')


if __name__ == '__main__':
    main()
