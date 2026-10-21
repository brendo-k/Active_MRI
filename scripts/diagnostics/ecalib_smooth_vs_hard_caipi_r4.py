"""Smooth (soft-SENSE ecalib) vs hard-edged sensitivity maps with a fixed CAIPI 2x2 (R=4) mask, CG-SENSE, 3 volumes.

CG-SENSE has no weights, so only the maps differ between arms (same mask, same solver). The mask is the 2x2 CAIPI lattice
plus the 10x10 preselected centre. Arms:
  net_conv / net_bilinear  sense U-Net maps of the matched 50-epoch LOUPE runs: unit norm, times a binary body mask
  ecalib_soft              cached BART ecalib soft-SENSE maps (norm tapers to 0 at the edge)
  ecalib_hard              the same maps made net-style: support = sum|S|^2 > 0.5, unit norm inside, zero outside
Volumes: the median-PSNR validation volume of brain, knee-PD and knee-PDFS (by the conv run's mean PSNR), all slices.
Metrics are the trainer's (inside the target > 0.1 * max signal mask).

    python scripts/diagnostics/ecalib_smooth_vs_hard_caipi_r4.py
"""
import json
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
from activemri.samplers.caipi import caipi_mask
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from mask_gradient_anatomy_r8 import load
from sens_upsample_slice_diff_r8 import slice_metadata

RUNS = {'conv': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
        'bilinear': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep'}
CONFIG = 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10/.hydra/config.yaml'
REFERENCE = Path('reports/sens_upsample_r8/ecalib_comparison_psnr.npz')
OUT_DIR = Path('reports/ecalib_smooth_vs_hard_caipi_r4')
ARMS = ('net_conv', 'net_bilinear', 'ecalib_soft', 'ecalib_hard')
SUPPORT_NORM_SQ = 0.5


def hard_version(maps):
    norm = maps.abs().pow(2).sum(dim=1, keepdim=True)
    return maps / norm.sqrt().clamp(min=1e-8) * (norm > SUPPORT_NORM_SQ)


def median_volumes(reference):
    chosen = {}
    for label in ('brain', 'knee-PD', 'knee-PDFS'):
        rows = reference['condition'] == label
        volumes = np.unique(reference['volume'][rows])
        means = np.array([reference['own__conv'][rows & (reference['volume'] == v)].mean() for v in volumes])
        chosen[label] = str(volumes[np.argsort(means)[len(means) // 2]])
    return chosen


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {n: load(Path(p), 'last_model.pt', 'cuda')[0] for n, p in RUNS.items()}
    engine = models['conv']
    cfg = OmegaConf.load(CONFIG)
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    names, slice_ids, _ = slice_metadata(val_loader.dataset)
    volumes = median_volumes(np.load(REFERENCE, allow_pickle=True))
    print(volumes)

    results, examples = {}, {}
    for label, volume in volumes.items():
        indices = np.where(names == volume)[0]
        loader = DataLoader(Subset(val_loader.dataset, indices.tolist()), batch_size=8, collate_fn=val_loader.collate_fn)
        rows = {arm: {'psnr': [], 'ssim': [], 'missed': []} for arm in ARMS}
        middle, offset = len(indices) // 2, 0
        examples[label] = {'output': {}}
        for batch in loader:
            kspace, target = batch['kspace'].cuda(), batch['target'].cuda()
            metric_mask = metric_mask_from_real_kspace(kspace, batch['anatomy'])
            cache = batch['sense_maps'].cuda()
            old_mask, _ = engine._initial_masks(kspace)
            lattice = caipi_mask(kspace.shape[-3:-1] if kspace.shape[-1] == 2 else kspace.shape[-2:]).to(kspace.device)
            mask = torch.clamp(old_mask + lattice[None, None], max=1)
            complex_kspace = torch.view_as_complex(kspace.contiguous())
            maps = {'net_conv': models['conv'].sensetivity_net(complex_kspace, mask_method='pre_network')[0],
                    'net_bilinear': models['bilinear'].sensetivity_net(complex_kspace, mask_method='pre_network')[0],
                    'ecalib_soft': cache, 'ecalib_hard': hard_version(cache)}
            signal = metric_mask.float().reshape(len(target), -1)
            for arm, m in maps.items():
                output = engine(target, kspace, fixed_mask=mask, sense_maps=m)['output']
                masked_output, masked_target = _apply_signal_mask(output, target if target.dim() == 4 else target.unsqueeze(1), metric_mask)
                rows[arm]['psnr'] += compute_psnr_torch(masked_output, masked_target).flatten().tolist()
                rows[arm]['ssim'] += compute_ssim_torch(masked_output, masked_target).flatten().tolist()
                covered = (m.abs().pow(2).sum(dim=1) > SUPPORT_NORM_SQ).float().reshape(len(target), -1)
                rows[arm]['missed'] += (signal * (1 - covered)).sum(1).div(signal.sum(1)).tolist()
                if offset <= middle < offset + len(kspace):
                    examples[label]['target'] = target[middle - offset].squeeze().cpu()
                    examples[label]['output'][arm] = output[middle - offset].squeeze().cpu()
                    examples[label]['mask_fraction'] = float(mask.mean())
            offset += len(kspace)
        results[label] = {'volume': Path(volume).name, 'slices': len(indices),
                          'sampling_fraction': examples[label]['mask_fraction'],
                          'arms': {a: {k: float(np.mean(v)) for k, v in r.items()} for a, r in rows.items()},
                          'per_slice_psnr': {a: r['psnr'] for a, r in rows.items()}}
    (OUT_DIR / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    for label, r in results.items():
        print(label, r['volume'], r['slices'], 'slices, sampled fraction', round(r['sampling_fraction'], 3))
        for a, s in r['arms'].items():
            print(f"  {a:<13} PSNR {s['psnr']:.2f}  SSIM {s['ssim']:.4f}  signal pixels outside map support {100 * s['missed']:.2f}%")

    for label, r in results.items():
        p = r['per_slice_psnr']
        for a, b in (('ecalib_soft', 'ecalib_hard'), ('ecalib_hard', 'net_bilinear'), ('net_bilinear', 'net_conv')):
            d = np.array(p[a]) - np.array(p[b])
            print(f'{label}: {a} - {b} PSNR {d.mean():+.3f} dB (se {d.std(ddof=1) / np.sqrt(len(d)):.3f}, {int((d > 0).sum())}/{len(d)} slices positive)')

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    for row, (label, e) in enumerate(examples.items()):
        axes[row, 0].imshow(e['target'], cmap='gray'); axes[row, 0].set_title(f"{label} target")
        for col, arm in enumerate(ARMS, start=1):
            error = (e['output'][arm] - e['target']).abs()
            axes[row, col].imshow(error, cmap='magma', vmin=0, vmax=0.3 * float(e['target'].max()))
            axes[row, col].set_title(f'{arm} |error|')
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT_DIR / 'caipi_r4_errors.png', dpi=90)


if __name__ == '__main__':
    main()
