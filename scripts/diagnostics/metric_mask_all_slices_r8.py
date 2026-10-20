"""Montage of the closing-based metric mask contour over every validation slice, plus per-slice mask statistics.

    python scripts/diagnostics/metric_mask_all_slices_r8.py
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
from torch.utils.data import DataLoader

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.utils.metric_mask import apodized_rss
from metric_mask_variants_r8 import CONFIG, apodized_mask

OUT_DIR = Path('reports/sens_upsample_r8/metric_mask_all_slices_padded')
COLUMNS, ROWS = 16, 12


def main():
    cfg = OmegaConf.load(CONFIG)
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=32, num_workers=4, collate_fn=val_loader.collate_fn)
    images, apodized, anatomies = [], [], []
    for batch in loader:
        images.append(batch['target'].squeeze(1).numpy())
        anatomies.extend(batch['anatomy'])
        apodized.append(apodized_rss(torch.view_as_complex(batch['kspace'].contiguous())).numpy())
    images, apodized = np.concatenate(images), np.concatenate(apodized)
    masks = np.stack([apodized_mask(image, anatomy) for image, anatomy in zip(apodized, anatomies)])
    fraction = masks.mean((1, 2))
    print(f'{len(images)} slices; >70%: {(fraction > 0.7).sum()} (legit for knee posterior slices), <5%: {(fraction < 0.05).sum()}; mask fraction min/median/max {fraction.min():.3f}/{np.median(fraction):.3f}/{fraction.max():.3f}')
    print('smallest-mask slices:', np.argsort(fraction)[:10].tolist(), 'largest:', np.argsort(fraction)[-10:].tolist())
    np.savez(OUT_DIR.parent / 'metric_mask_all_slices_stats.npz', fraction=fraction)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_page = COLUMNS * ROWS
    for page, start in enumerate(range(0, len(images), per_page)):
        fig, axes = plt.subplots(ROWS, COLUMNS, figsize=(COLUMNS * 1.6, ROWS * 1.6))
        for ax in axes.flat:
            ax.axis('off')
        for ax, index in zip(axes.flat, range(start, min(start + per_page, len(images)))):
            ax.imshow(images[index] ** 0.5, cmap='gray')
            ax.contour(masks[index], levels=[0.5], colors='red', linewidths=0.6)
            ax.set_title(str(index), fontsize=5, pad=1)
        fig.subplots_adjust(0, 0, 1, 0.98, 0.02, 0.1)
        fig.savefig(OUT_DIR / f'page{page}.png', dpi=90)
        plt.close(fig)


if __name__ == '__main__':
    main()
