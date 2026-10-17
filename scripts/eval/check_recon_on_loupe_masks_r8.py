"""PSNR of several reconstructors on the LOUPE run's own masks, whole val set, common seeded draws (the masks and
draws of eval_adaptive_samplers_r8.py). Separates "the reconstructor is worse" from "the sampler trained against it
moved the mask": a frozen-reconstructor run's recon must score exactly like its source here.

    python scripts/eval/check_recon_on_loupe_masks_r8.py
"""
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from eval_adaptive_samplers_r8 import CONV_RUN, LOUPE_RUN, OUT, probability_maps, psnr_with_donors
from mask_gradient_anatomy_r8 import load

RECONS = {
    'loupe_recon': (LOUPE_RUN, 'last_model.pt'),
    'generalist': (Path('/tmp/bkadota_runs/generalist_vd_varnet_r8'), 'last_model.pt'),
    'static_frozen-generalist_recon': (Path('/tmp/bkadota_runs/static_frozen-generalist_R8'), 'last_model.pt'),
}


def main():
    cfg = OmegaConf.load(CONV_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val = build_dataloaders(cfg)
    loader = torch.utils.data.DataLoader(val.dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=val.collate_fn)
    anatomy = np.array(sum((list(b['anatomy']) for b in loader), []))
    loupe, checkpoint = load(LOUPE_RUN, 'last_model.pt', 'cuda')
    loupe.label_classes = list(checkpoint['options'].label_classes)
    maps = probability_maps(loupe, loader, 'cuda')
    own = {'own': torch.arange(len(anatomy))}
    results = {}
    for name, path in RECONS.items():
        model, checkpoint = load(*path, 'cuda')
        model.label_classes = list(checkpoint['options'].label_classes)
        psnr = psnr_with_donors(model, loader, maps, own, 'cuda')['own'].numpy()
        results[name] = {a: float(psnr[anatomy == a].mean()) for a in ('brain', 'knee')}
        print(name, json.dumps(results[name]), flush=True)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'recon_on_loupe_masks.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
