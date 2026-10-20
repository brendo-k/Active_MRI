"""Check the torch metric mask against the numpy reference on every validation slice, and time it.

    python scripts/diagnostics/metric_mask_torch_check_r8.py
"""
import os
import time

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.utils.metric_mask import apodized_rss, metric_support_mask
from metric_mask_variants_r8 import CONFIG, apodized_mask

BATCH_SIZE = 16


def main():
    cfg = OmegaConf.load(CONFIG)
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=val_loader.collate_fn)
    device = 'cuda'
    intersection_over_union, differing_pixels, seconds = [], [], []
    for batch in loader:
        kspace = torch.view_as_complex(batch['kspace'].contiguous()).to(device)
        anatomies = list(batch['anatomy'])
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch_mask = metric_support_mask(kspace, anatomies)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        reference_images = apodized_rss(kspace).cpu().numpy()
        for index, anatomy in enumerate(anatomies):
            reference = apodized_mask(reference_images[index], anatomy)
            candidate = torch_mask[index, 0].cpu().numpy() > 0
            intersection_over_union.append((reference & candidate).sum() / max((reference | candidate).sum(), 1))
            differing_pixels.append((reference != candidate).sum())
    iou, differing = np.array(intersection_over_union), np.array(differing_pixels)
    print(f'{len(iou)} slices; IoU min {iou.min():.4f} mean {iou.mean():.5f}; exact match {(differing == 0).sum()}; max differing pixels {differing.max()}')
    print(f'worst slices: {np.argsort(iou)[:5].tolist()}; batch of {BATCH_SIZE}: median {np.median(seconds) * 1e3:.0f} ms, max {max(seconds) * 1e3:.0f} ms')


if __name__ == '__main__':
    main()
