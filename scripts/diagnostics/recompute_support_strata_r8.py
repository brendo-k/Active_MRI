"""Recompute the per-slice support (fraction of pixels in the trainer's signal mask) for the 688 validation slices and redo the
support strata and correlations of sens_upsample_slice_diff_r8.py / sens_upsample_vs_ecalib_r8.py from their saved PSNR arrays.
(Both originally computed support from a single [H, W] target, which normalises each row by its own max.)

    python scripts/diagnostics/recompute_support_strata_r8.py
"""
import json
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.utils.metric_mask import metric_mask_from_real_kspace

OUT_DIR = Path('reports/sens_upsample_r8')


def tercile_means(delta, support, anatomy=None):
    edges = np.percentile(support, [33.3, 66.7])
    third = np.digitize(support, edges)
    return {n: {'n': int((third == i).sum()), 'mean': float(delta[third == i].mean()), 'sd': float(delta[third == i].std())}
            for i, n in enumerate(('low', 'mid', 'high'))}


def main():
    cfg = OmegaConf.load('checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10/.hydra/config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    support = []
    for batch in DataLoader(val_loader.dataset, batch_size=16, collate_fn=val_loader.collate_fn):
        support += metric_mask_from_real_kspace(batch['kspace'].cuda(), batch['anatomy']).flatten(1).mean(1).tolist()
    support = np.array(support)
    slice_diff = np.load(OUT_DIR / 'slice_diff_psnr.npz', allow_pickle=True)
    ecalib = np.load(OUT_DIR / 'ecalib_comparison_psnr.npz', allow_pickle=True)
    anatomy = slice_diff['anatomy']
    result = {'support_mean': {a: float(support[anatomy == a].mean()) for a in np.unique(anatomy)},
              'old_support_mean': {a: float(slice_diff['support'][anatomy == a].mean()) for a in np.unique(anatomy)}}
    contrasts = {'bilinear - conv (full)': slice_diff['bilinear'] - slice_diff['conv'],
                 'bilinear - conv (mask only)': slice_diff['conv_sense+bilinear_mask'] - slice_diff['conv'],
                 'bilinear - conv (sense net only)': slice_diff['bilinear_sense+conv_mask'] - slice_diff['conv'],
                 'ecalib_c10 - conv (own)': ecalib['own__ecalib_c10'] - ecalib['own__conv'],
                 'ecalib maps, c10 mask - conv mask': ecalib['ecalib_maps__ecalib_c10'] - ecalib['ecalib_maps__conv'],
                 'maps only (conv mask): ecalib - conv net': ecalib['ecalib_maps__conv'] - ecalib['conv_net__conv']}
    result['strata'] = {k: tercile_means(d, support) for k, d in contrasts.items()}
    result['corr_with_support'] = {k: float(np.corrcoef(d, support)[0, 1]) for k, d in contrasts.items()}
    result['corr_with_support_by_anatomy'] = {a: {k: float(np.corrcoef(d[anatomy == a], support[anatomy == a])[0, 1]) for k, d in contrasts.items()}
                                              for a in np.unique(anatomy)}
    (OUT_DIR / 'support_recomputed.json').write_text(json.dumps(result, indent=1) + '\n')
    np.save(OUT_DIR / 'support_correct.npy', support)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
