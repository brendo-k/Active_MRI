"""2xN montages (sampled R=8 mask on top, reconstruction below), one figure per anatomy, one column per run.

Every run uses its own mask (one common seeded Bernoulli draw) and its own maps: the sense U-Net for conv / bilinear, the cached
BART ecalib maps for ecalib_c10 / ecalib_c24. Shown on the median-PSNR mid-volume validation slice of each anatomy. Images are
the full coil-combined output (not signal-masked) on a common window; titles give the trainer-path PSNR / SSIM of the slice.

    python scripts/diagnostics/mask_recon_montage_r8.py
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
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load

OUT_DIR = Path('reports/sens_upsample_r8')
RUNS = {'conv': ('checkpoints/loupe2d_cgsense_r8_brain_knee_shared', False),
        'bilinear': ('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep', False),
        'ecalib_c10': ('checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10', True),
        'ecalib_c24': ('checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24', True)}
SLICES = {'brain': 152, 'knee': 473}
DPI = 250


@torch.no_grad()
def main():
    models = {name: load(Path(path), 'last_model.pt', 'cuda')[0] for name, (path, _) in RUNS.items()}
    cfg = OmegaConf.load('checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10/.hydra/config.yaml')
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    for anatomy, index in SLICES.items():
        batch = next(iter(DataLoader(Subset(val_loader.dataset, [index]), batch_size=1, collate_fn=val_loader.collate_fn)))
        kspace, target = batch['kspace'].cuda(), batch['target'].cuda()
        cache = batch['sense_maps'].cuda()
        truth = target[0].squeeze().cpu()
        window = float(truth.max())
        fig, axes = plt.subplots(2, len(RUNS), figsize=(5 * len(RUNS), 11.2))
        for column, (name, (_, cached)) in enumerate(RUNS.items()):
            model = models[name]
            probability = unscaled(model, kspace, None)
            old_mask, valid_mask = model._initial_masks(kspace)
            torch.manual_seed(0)
            mask = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
            output = model(target, kspace, fixed_mask=mask, sense_maps=cache if cached else None)['output']
            target4 = target if target.dim() == 4 else target.unsqueeze(1)
            masked_output, masked_target = _apply_signal_mask(output, target4, metric_mask_from_real_kspace(kspace, batch['anatomy']))
            psnr = float(compute_psnr_torch(masked_output, masked_target)[0])
            ssim = float(compute_ssim_torch(masked_output, masked_target)[0])
            axes[0, column].imshow(mask[0].reshape(mask.shape[-2:]).cpu(), cmap='gray', interpolation='nearest')
            axes[0, column].set_title(f'{name}: sampled mask ({float(mask.mean()):.3f} of k-space)', fontsize=11)
            axes[1, column].imshow(output[0].squeeze().cpu(), cmap='gray', vmin=0, vmax=window, interpolation='nearest')
            axes[1, column].set_title(f'{name}: recon, PSNR {psnr:.2f} dB, SSIM {ssim:.4f}', fontsize=11)
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle(f'{anatomy}: validation slice {index}, R=8, seed-0 Bernoulli draw, common display window', fontsize=13)
        fig.tight_layout(h_pad=3.0, rect=(0, 0, 1, 0.96))
        fig.savefig(OUT_DIR / f'montage_mask_recon_{anatomy}.png', dpi=DPI)
        plt.close(fig)
        print(anatomy, 'saved')


if __name__ == '__main__':
    main()
