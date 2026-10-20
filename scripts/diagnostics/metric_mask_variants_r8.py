"""Compare target-only metric masks (identical across runs): current threshold vs closed/filled vs convex hull.

    python scripts/diagnostics/metric_mask_variants_r8.py
"""
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage
from skimage import morphology
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.utils.metric_mask import MASK_PARAMS

OUT_PATH = Path('reports/sens_upsample_r8/metric_mask_variants.png')
CONFIG = 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10/.hydra/config.yaml'
SLICES = {'brain': 152, 'knee': 473, 'knee 2': 20, 'knee 3': 120}


def threshold(image, fraction=0.1):
    return image > fraction * image.max()


def closed_filled(image, fraction=0.1, radius=6):
    padded = np.pad(threshold(image, fraction), radius)
    mask = morphology.closing(padded, morphology.disk(radius))[radius:-radius, radius:-radius]
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    largest = np.argmax(ndimage.sum(mask, labels, range(1, count + 1))) + 1
    return labels == largest


def smoothed(image, fraction=0.05, sigma=4):
    return ndimage.gaussian_filter(image, sigma) > fraction * image.max()


def apodized_mask(apodized_image, anatomy):
    return closed_filled(apodized_image, **MASK_PARAMS[anatomy])


def convex_hull(image, fraction=0.1):
    return morphology.convex_hull_image(closed_filled(image, fraction))


VARIANTS = {'current (>0.1 max)': threshold, 'closing r6 + fill + largest CC': closed_filled,
            'gaussian σ4 > 0.05 max': smoothed, 'convex hull': convex_hull}


def main():
    cfg = OmegaConf.load(CONFIG)
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    fig, axes = plt.subplots(len(VARIANTS) + 1, len(SLICES), figsize=(3.6 * len(SLICES), 3.6 * (len(VARIANTS) + 1)))
    for column, (label, index) in enumerate(SLICES.items()):
        batch = next(iter(DataLoader(Subset(val_loader.dataset, [index]), batch_size=1, collate_fn=val_loader.collate_fn)))
        image = batch['target'][0].squeeze().cpu().numpy()
        axes[0, column].imshow(image, cmap='gray'); axes[0, column].set_title(f'{label}: RSS')
        for row, (name, build) in enumerate(VARIANTS.items(), start=1):
            mask = build(image)
            axes[row, column].imshow(mask, cmap='gray')
            axes[row, column].set_title(f'{name}\n{mask.mean():.0%} px', fontsize=8)
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=80)


if __name__ == '__main__':
    main()
