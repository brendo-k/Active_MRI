"""Show the trainer's metric support (apodized k-space mask) against the sense net's body mask and the ecalib map support.

    python scripts/diagnostics/metric_support_figure_r8.py
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
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from mask_gradient_anatomy_r8 import load

OUT_DIR = Path('reports/sens_upsample_r8')
RUN = 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep'
CONFIG = 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10/.hydra/config.yaml'
SLICES = {'brain': 152, 'knee': 473}


@torch.no_grad()
def main():
    model = load(Path(RUN), 'last_model.pt', 'cuda')[0]
    cfg = OmegaConf.load(CONFIG)
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for row, (label, index) in enumerate(SLICES.items()):
        batch = next(iter(DataLoader(Subset(val_loader.dataset, [index]), batch_size=1, collate_fn=val_loader.collate_fn)))
        target = batch['target'].cuda()
        kspace = batch['kspace'].cuda()
        signal = metric_mask_from_real_kspace(kspace, [label])[0].squeeze().cpu().numpy()
        body = model.sensetivity_net(torch.view_as_complex(kspace.contiguous()), mask_method='pre_network')[1][0, 0].cpu().numpy()
        ecalib = (batch['sense_maps'][0].abs().pow(2).sum(0) > 0.5).numpy()
        image = target[0].squeeze().cpu().numpy()
        panels = [('target', image, None), ('metric support: apodized k-space mask', signal, None),
                  ('net body mask (10x10 centre RSS > 0.1 max)', body, None), ('ecalib support (sum|S|^2 > 0.5)', ecalib.astype(float), None)]
        for ax, (title, data, _) in zip(axes[row], panels):
            ax.imshow(data, cmap='gray'); ax.axis('off'); ax.set_title(f'{label}: {title}', fontsize=9)
        for ax, region, colour in ((axes[row][2], signal, 'red'), (axes[row][3], signal, 'red')):
            ax.contour(region, levels=[0.5], colors=colour, linewidths=0.8)
        print(label, 'signal fraction', round(float(signal.mean()), 3), 'net body', round(float(body.mean()), 3), 'ecalib', round(float(ecalib.mean()), 3),
              'signal px outside net body', round(float(((signal > 0) & (body == 0)).sum() / signal.sum()), 4))
    fig.tight_layout(); fig.savefig(OUT_DIR / 'metric_support.png', dpi=100)


if __name__ == '__main__':
    main()
