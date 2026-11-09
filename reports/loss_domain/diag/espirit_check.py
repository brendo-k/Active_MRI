"""Is the cached ESPIRiT image x_E = sum_c conj(S_c) F^-1(k_c) (scripts/data/build_complex_image_cache.py) computed right?
Every 48th val slice, BART maps recomputed with the anatomy preset (soft-SENSE, no -P).

Per slice, in tissue (metric mask and |x_E| > 0.3 max):
- the cache against the fresh recompute;
- sum_c |S_c|^2, which is A^H A for A = F S with full sampling (M = I), so the least-squares SENSE image
  (A^H A)^-1 A^H y = x_E / sum_c |S_c|^2 differs from x_E in magnitude only;
- the coil fit |S x - coils| / |coils| with the conj (x_E) and without it (x = sum_c S_c F^-1(k_c));
- the phase of x_E against a virtual coil, sum_c v_c F^-1(k_c), with v the first principal component of the 24x24
  calibration k-space (both conjugations of v), which is what BART's default phase rotation should reproduce.

usage: python reports/loss_domain/diag/espirit_check.py
"""
import sys
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0, 'scripts/eval'); sys.path.insert(0, '.')
import eval_loss_domain as e
from activemri.datasets.fastmri_data import build_manifest_dataloaders
from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.metric_mask import metric_support_mask

_, val_loader = build_manifest_dataloaders(manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=e.ROOTS,
    classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8, complex_cache_dir=e.CACHE)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_loader.dataset, list(range(0, len(val_loader.dataset), 48))),
                                     batch_size=8, num_workers=8, collate_fn=val_loader.collate_fn)


def relative(a, b, w):
    return float(((a - b).abs().pow(2) * w).sum() / (b.abs().pow(2) * w).sum()) ** 0.5


def median_abs_phase(a, b, mask):
    return float(torch.angle(a * b.conj())[mask].abs().median())


st = defaultdict(lambda: defaultdict(list))
for batch in loader:
    k = torch.view_as_complex(batch['kspace'].contiguous())
    xe_cache = torch.view_as_complex(batch['complex_target'].contiguous())
    metric = metric_support_mask(k, list(batch['anatomy']))[:, 0].bool()
    for i, (condition, anatomy) in enumerate(zip(batch['condition'], batch['anatomy'])):
        live = k[i].abs().flatten(1).amax(1) > 0
        ki = k[i][live]
        coils = ifft_2d_img(ki)
        S, _ = ecalib_sense_maps(ki, **PRESETS[anatomy])
        xe = (coils * S.conj()).sum(0)
        x_no_conj = (coils * S).sum(0)
        power = S.abs().pow(2).sum(0)
        tissue = metric[i] & (xe.abs() > 0.3 * xe.abs().max())
        w = tissue.float()
        H, W = ki.shape[-2:]
        calib = ki[:, H // 2 - 12:H // 2 + 12, W // 2 - 12:W // 2 + 12].reshape(len(ki), -1).T   # [576, C]
        v = torch.linalg.svd(calib, full_matrices=False).Vh[0]                                  # v^H, [C]
        virtual = {'sum_c v_c^* coil_c': (coils * v[:, None, None]).sum(0),
                   'sum_c v_c coil_c': (coils * v.conj()[:, None, None]).sum(0)}
        for d in (st[condition], st['all']):
            d['cache vs fresh BART (rel err)'].append(relative(xe_cache[i], xe, metric[i].float()))
            d['sum|S|^2 in tissue: median'].append(float(power[tissue].median()))
            d['sum|S|^2 in tissue: 5th percentile'].append(float(power[tissue].quantile(0.05)))
            d['phase of (A^HA)^-1 A^H y vs x_E (rad)'].append(
                median_abs_phase(xe / power.clamp(min=1e-6), xe, tissue))
            d['coil fit with conj (x_E)'].append(relative(S * xe, coils, w))
            d['coil fit without conj'].append(relative(S * x_no_conj, coils, w))
            d['|x without conj| vs |x_E| (rel err)'].append(relative(x_no_conj.abs(), xe.abs(), w))
            for name, image in virtual.items():
                d[f'|phase(x_E) - phase({name})|'].append(median_abs_phase(xe, image, tissue))
            d['|phase(x_E)| median'].append(float(torch.angle(xe)[tissue].abs().median()))
for condition in ('all', 'brain', 'knee-PD', 'knee-PDFS'):
    d = st[condition]
    print(f'== {condition} (n={len(next(iter(d.values())))})')
    for name, values in d.items():
        print(f'   {name:52s} median {np.median(values):.4f}   max {np.max(values):.4f}')
