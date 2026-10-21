"""Are the bilinear sensitivity maps worse? Per-slice SSIM/NMSE/PSNR plus reference-free sense-map diagnostics.

Same matched conv / bilinear CG-SENSE LOUPE R=8 runs as sens_upsample_slice_diff_r8.py. Per validation slice:
  metrics   trainer-path (signal-masked) PSNR, SSIM and NMSE of each run's own mask + sense net, averaged over
            SCORING_SEEDS Bernoulli draws
  body mask the sensitivity net's 'pre_network' body mask (RSS of the 10x10 centre of k-space > 10% of max) is
            weight-free; checked to be identical across the two nets, and scored for the share of signal pixels it drops
  fit       coil-model residual of the fully sampled data, ||y - S x|| / ||y|| inside the body mask, with
            x = sum_c conj(S_c) y_c (0 = maps explain the coil images exactly)
  oracle    NMSE of that fully sampled SENSE image vs the RSS target inside the signal mask (no undersampling)
  maps      relative difference of |S| between the nets inside the body mask; high-frequency energy of |S|
            (share of 2D-FFT energy outside a radius of 12 cycles per FOV)
Optionally saves a figure of chosen slices.

    python scripts/diagnostics/sens_upsample_sense_maps_r8.py
"""
import argparse
import json
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load

RUNS = {'conv': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared'),
        'bilinear': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep')}
OUT_DIR = Path('reports/sens_upsample_r8')
SCORING_SEEDS = (0, 1, 2, 3)
HIGH_FREQUENCY_RADIUS = 12


def compute_nmse_torch(output, target):
    return ((target - output) ** 2).flatten(1).sum(1) / (target ** 2).flatten(1).sum(1)


@torch.no_grad()
def own_metrics(model, part, probability, seed_key):
    """[3, batch] seed-averaged PSNR, SSIM, NMSE on the signal-masked trainer path."""
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    total = torch.zeros(3, len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        out_m, tgt_m = _apply_signal_mask(output, target, part['metric_mask'])
        total += torch.stack([compute_psnr_torch(out_m, tgt_m).flatten(), compute_ssim_torch(out_m, tgt_m).flatten(),
                              compute_nmse_torch(out_m, tgt_m).flatten()]).cpu() / len(SCORING_SEEDS)
    return total


@torch.no_grad()
def sense_diagnostics(model, kspace_complex, target, signal, device):
    maps, body = model.sensetivity_net(kspace_complex, mask_method=model.sense_mask_method)
    coil_images = ifft_2d_img(kspace_complex, axes=[-1, -2])
    image = (maps.conj() * coil_images).sum(1, keepdim=True)
    body_pixels = body.sum((1, 2, 3)).clamp(min=1)
    residual = ((coil_images - maps * image).abs() ** 2).sum(1, keepdim=True) * body
    fit = (residual.sum((1, 2, 3)) / ((coil_images.abs() ** 2).sum(1, keepdim=True) * body).sum((1, 2, 3)).clamp(min=1e-12)).sqrt()
    oracle_out, oracle_tgt = image.abs() * signal, target.reshape_as(signal) * signal
    oracle_nmse = compute_nmse_torch(oracle_out, oracle_tgt)
    dropped = (signal * (1 - body)).sum((1, 2, 3)) / signal.sum((1, 2, 3))
    spectrum = torch.fft.fftshift(torch.fft.fft2(maps.abs()), dim=(-2, -1)).abs() ** 2
    size = maps.shape[-1]
    radius = torch.hypot(*torch.meshgrid((torch.arange(size, device=device) - size // 2).float(), (torch.arange(size, device=device) - size // 2).float(), indexing='ij'))
    high_frequency = spectrum[..., radius > HIGH_FREQUENCY_RADIUS].sum(-1) / spectrum.sum((-2, -1)).clamp(min=1e-12)
    return {'maps': maps, 'body': body, 'fit': fit.cpu(), 'oracle_nmse': oracle_nmse.flatten().cpu(), 'dropped': dropped.flatten().cpu(),
            'high_frequency': high_frequency.mean(1).cpu(), 'body_pixels': body_pixels.cpu()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    models = {name: load(path, 'last_model.pt', args.device)[0] for name, path in RUNS.items()}
    cfg = OmegaConf.load(RUNS['conv'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)

    rows = {k: [] for k in ('psnr', 'ssim', 'nmse')}
    per = {name: {k: [] for k in ('fit', 'oracle_nmse', 'dropped', 'high_frequency')} for name in RUNS}
    map_diff, body_identical, anatomy, volume, slice_id, offset = [], [], [], [], [], 0
    metrics = {name: [] for name in RUNS}
    for batch in loader:
        kspace = batch['kspace'].to(args.device)
        part = {'kspace': kspace, 'target': batch['target'].to(args.device), 'metric_mask': metric_mask_from_real_kspace(kspace, batch['anatomy'])}
        kc = torch.view_as_complex(kspace.contiguous())
        diag = {name: sense_diagnostics(model, kc, part['target'], part['metric_mask'], args.device) for name, model in models.items()}
        for name, model in models.items():
            metrics[name].append(own_metrics(model, part, unscaled(model, kspace, None), offset))
            for key in per[name]:
                per[name][key].append(diag[name][key])
        body = diag['conv']['body']
        body_identical.append(((diag['conv']['body'] - diag['bilinear']['body']).abs().sum((1, 2, 3))).cpu())
        a, b = diag['conv']['maps'].abs(), diag['bilinear']['maps'].abs()
        map_diff.append((((a - b) ** 2 * body).sum((1, 2, 3)) / ((a ** 2 * body).sum((1, 2, 3)).clamp(min=1e-12))).sqrt().cpu())
        anatomy += list(batch['anatomy'])
        volume += [str(v) for v in batch['fname']]
        slice_id += [int(s) for s in batch['slice_id']]
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    out = {'anatomy': np.array(anatomy), 'volume': np.array(volume), 'slice_id': np.array(slice_id),
           'body_pixels_differing': torch.cat(body_identical).numpy(), 'map_rel_diff': torch.cat(map_diff).numpy()}
    for name in RUNS:
        m = torch.cat(metrics[name], dim=1).numpy()
        out.update({f'{name}_psnr': m[0], f'{name}_ssim': m[1], f'{name}_nmse': m[2]})
        out.update({f'{name}_{k}': torch.cat(v).numpy() for k, v in per[name].items()})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / 'sense_maps_per_slice.npz', **out)

    summary = {'body_mask_identical_slices': float((out['body_pixels_differing'] == 0).mean())}
    anatomy = out['anatomy']
    for group in ('all', 'brain', 'knee'):
        rows = np.ones(len(anatomy), bool) if group == 'all' else anatomy == group
        summary[group] = {'map_rel_diff_mean': float(out['map_rel_diff'][rows].mean())}
        for key in ('psnr', 'ssim', 'nmse', 'fit', 'oracle_nmse', 'dropped', 'high_frequency'):
            c, b = out[f'conv_{key}'][rows], out[f'bilinear_{key}'][rows]
            summary[group][key] = {'conv': float(c.mean()), 'bilinear': float(b.mean()), 'delta_mean': float((b - c).mean()), 'delta_sd': float((b - c).std())}
    (OUT_DIR / 'sense_maps_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
