"""Figures: (1) conv / bilinear / ecalib_c10 masks, probability and one Bernoulli draw; (2) sensitivity-map magnitudes on a
median-PSNR brain and knee validation slice for the conv net, bilinear net and cached ecalib maps.

    python scripts/diagnostics/sens_upsample_vs_ecalib_maps_figure_r8.py
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
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load

OUT_DIR = Path('reports/sens_upsample_r8')
RUNS = {'conv': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
        'bilinear': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep',
        'ecalib_c10': 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10'}
CROP = 96
COILS = 4


def median_slice(psnr, anatomy, position, name):
    rows = np.where((anatomy == name) & (position > 0.3) & (position < 0.7))[0]
    return int(rows[np.argmin(np.abs(psnr[rows] - np.median(psnr[rows])))])


def masks_figure(models, batch):
    kspace = batch['kspace'].cuda()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for column, (name, model) in enumerate(models.items()):
        probability = unscaled(model, kspace, None)
        old_mask, valid_mask = model._initial_masks(kspace)
        torch.manual_seed(0)
        draw = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
        c = probability.shape[-1] // 2
        window = (slice(c - CROP, c + CROP),) * 2
        axes[0, column].imshow(probability[0, 0][window].cpu(), vmin=0, vmax=1)
        axes[0, column].set_title(f'{name}: p(k), mean {float(probability.mean()):.3f}')
        axes[1, column].imshow(draw[0].reshape(probability.shape[-2:])[window].cpu(), cmap='gray')
        axes[1, column].set_title(f'{name}: one Bernoulli draw (R=8)')
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ecalib_comparison_masks_draws.png', dpi=110)


@torch.no_grad()
def maps_figure(models, dataset, collate, indices):
    fig, axes = plt.subplots(2 * 2, 4, figsize=(15, 15))
    for row, (label, index) in enumerate(indices.items()):
        batch = next(iter(DataLoader(Subset(dataset, [index]), batch_size=1, collate_fn=collate)))
        kspace = batch['kspace'].cuda()
        cache = batch['sense_maps'].cuda()
        complex_kspace = torch.view_as_complex(kspace.contiguous())
        maps = {name: models[name].sensetivity_net(complex_kspace, mask_method=models[name].sense_mask_method)[0] for name in ('conv', 'bilinear')}
        maps['ecalib'] = cache
        target = batch['target'][0].squeeze().cpu()
        axes[2 * row, 0].imshow(target, cmap='gray')
        axes[2 * row, 0].set_title(f'{label} target')
        axes[2 * row + 1, 0].axis('off')
        for column, (name, m) in enumerate(maps.items(), start=1):
            magnitude = m[0].abs().cpu()
            axes[2 * row, column].imshow(magnitude.pow(2).sum(0).sqrt(), cmap='viridis')
            axes[2 * row, column].set_title(f'{label} {name}: sqrt(sum |S_c|^2)')
            axes[2 * row + 1, column].imshow(torch.cat(list(magnitude[:COILS]), dim=1), cmap='magma')
            axes[2 * row + 1, column].set_title(f'{name}: |S_c|, coils 0-{COILS - 1}')
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ecalib_comparison_sense_maps.png', dpi=100)


def main():
    models = {name: load(Path(path), 'last_model.pt', 'cuda')[0] for name, path in RUNS.items()}
    cfg = OmegaConf.load(Path(RUNS['ecalib_c10']) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    reference = np.load(OUT_DIR / 'ecalib_comparison_psnr.npz', allow_pickle=True)
    indices = {a: median_slice(reference['own__conv'], reference['anatomy'], reference['position'], a) for a in ('brain', 'knee')}
    print(indices, {a: (str(reference['volume'][i]).split('/')[-1], int(reference['slice_id'][i])) for a, i in indices.items()})
    probe = next(iter(DataLoader(Subset(val_loader.dataset, [indices['brain']]), batch_size=1, collate_fn=val_loader.collate_fn)))
    masks_figure(models, probe)
    maps_figure(models, val_loader.dataset, val_loader.collate_fn, indices)


if __name__ == '__main__':
    main()
