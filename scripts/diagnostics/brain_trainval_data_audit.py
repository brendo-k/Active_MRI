"""Is the brain train - val loss gap of reports/brain_varnet_scaling partly in the data? Model-free baselines per set.

Sets (brain FLAIR, local v1 manifest, the loader of the c6_v80_s42 run):
  train         the 80 tier-S training volumes (1,211 slices)
  trainer_val   the 10 volumes / 150 slices the trainer validated on (val_limit_volumes=0.25)
  full_val      all 40 val volumes / 605 slices

Per slice:
  - data: target max, full-image mean, mean inside the trainer's apodized metric mask, metric-mask area, the
    centre 10x10 k-space image peak (1 by the loader's normalization), and the noise sigma of the normalized k-space:
    std of the outer 16 readout lines of every live coil, taken inside the acquired extent because short-readout
    volumes are zero-padded (orthonormal FFT, so also the image noise per coil); the same from the outer
    phase-encode lines; SNR = metric-mask mean / readout-edge sigma. Rows carry the volume's acquisition group,
    field strength, scanner and coil count from the manifest.
  - recons, all under mask S with the common Bernoulli draw of seed 1234:
      zero_filled   RSS of the masked k-space
      cg_sense      10 CG iterations, no regularization, BART ecalib maps (brain preset) from the centre 24x24
      sense_full    SENSE coil combination of the fully sampled k-space (the ecalib model's floor against RSS)
      c6_v80        best_model.pt of c6_v80_s42 (peak; 80 volumes)
      c12_v20       best_model.pt of c12_v20_s42 (peak; 20 volumes)
  - per recon: L1 over the full image (the training loss), L1 inside the metric mask, NMSE and PSNR inside the
    metric mask (the scoring of brain_varnet_scaling_eval.py).

Writes <out>/slices.csv (one row per set, slice and recon) and <out>/summary.json (means and medians per set, and
the trainer_val / train and full_val / train ratios).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts/diagnostics python scripts/diagnostics/brain_trainval_data_audit.py
"""
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.bart_ecalib import PRESETS, ecalib_sense_maps
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.util import compute_psnr_torch
from eval_adaptive_samplers_r8 import thresholds
from mask_gradient_anatomy_r8 import load, to_device

QUEUE = Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_scaling')
OUT = Path('reports/brain_varnet_scaling/data_audit')
MASK_SEED = 1234
MODELS = {'c6_v80': 'c6_v80_s42', 'c12_v20': 'c12_v20_s42'}
MANIFEST = {r['file']: r for r in csv.DictReader(open('configs/datasets/fastmri_local_v1/manifest.csv'))}
NOISE_ROWS = 16
CENTRE = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None, help='first N slices of each set (smoke test)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def brain_sets(run, chunk, limit):
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, trainer_val_loader = build_dataloaders(cfg)
    cfg.datasets.val_limit_volumes = None
    _, full_val_loader = build_dataloaders(cfg)
    sets = {}
    for name, loader in (('train', train_loader), ('trainer_val', trainer_val_loader), ('full_val', full_val_loader)):
        dataset = torch.utils.data.ConcatDataset([p for p in loader.dataset.datasets if p[0]['anatomy'] == 'brain'])
        if limit is not None:
            dataset = torch.utils.data.Subset(dataset, list(range(min(limit, len(dataset)))))
        sets[name] = torch.utils.data.DataLoader(dataset, batch_size=chunk, shuffle=False, num_workers=4,
                                                 collate_fn=trainer_val_loader.collate_fn)
    return sets


def rss(coil_images):
    return coil_images.abs().pow(2).sum(1).sqrt()


def live_coils(kspace):
    return kspace.abs().flatten(2).amax(2) > 0


def edge_sigma(kspace, dim):
    """Noise std per slice: real and imaginary parts of the outer NOISE_ROWS lines along `dim` (2 readout, 3 phase
    encode) inside the slice's nonzero k-space extent, over live coils. Volumes with a short readout are zero-padded
    to the grid, so the outer lines of the grid itself can be empty; the outer lines of the acquired extent are not."""
    sigmas = []
    for b in range(len(kspace)):
        k = kspace[b][live_coils(kspace)[b]]
        filled = torch.nonzero(k.abs().amax(dim=(0, 2) if dim == 2 else (0, 1)) > 0).flatten()
        if filled.numel() < 4 * NOISE_ROWS:
            sigmas.append(kspace.real.new_tensor(float('nan')))
            continue
        lines = torch.cat([filled[:NOISE_ROWS], filled[-NOISE_ROWS:]])
        values = k.index_select(dim - 1, lines)
        values = values[values.abs() > 0]
        sigmas.append(torch.cat([values.real, values.imag]).std())
    return torch.stack(sigmas)


def centre_scale(kspace):
    """Undo nothing; the loader's divisor recomputed on normalized k-space is 1, so report the centre-image peak of
    the normalized data relative to the target peak instead (a sharpness / dynamic-range proxy)."""
    h, w = kspace.shape[-2:]
    centre = torch.zeros_like(kspace)
    centre[..., h // 2 - CENTRE // 2:h // 2 + CENTRE // 2, w // 2 - CENTRE // 2:w // 2 + CENTRE // 2] = 1
    return rss(ifft_2d_img(kspace * centre)).flatten(1).amax(1)


def scores(image, target, metric_mask):
    out_m, tgt_m = _apply_signal_mask(image, target, metric_mask)
    weight = metric_mask if metric_mask.dim() == image.dim() else metric_mask.unsqueeze(1)
    weight = weight.expand_as(image).float()
    l1_mask = ((image - target).abs() * weight).flatten(1).sum(1) / weight.flatten(1).sum(1)
    return {'l1_full': (image - target).abs().flatten(1).mean(1),
            'l1_mask': l1_mask,
            'nmse': ((out_m - tgt_m) ** 2).flatten(1).sum(1) / (tgt_m ** 2).flatten(1).sum(1),
            'psnr': compute_psnr_torch(out_m, tgt_m).flatten()}


@torch.no_grad()
def audit(name, loader, models, pool, device):
    rows, start = [], 0
    for batch in loader:
        part = to_device(batch, device)
        kspace = torch.view_as_complex(part['kspace'].contiguous())
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        n = len(kspace)
        index = torch.arange(start, start + n)
        maps_future = [pool.submit(ecalib_sense_maps, kspace[b].cpu(), **PRESETS['brain']) for b in range(n)]

        model, classes = next(iter(models.values()))
        labels = torch.full((n,), classes.index('brain'), device=device)
        proposal = model.propose_mask(part['kspace'], anatomy_labels=labels)
        rescaled = proposal['rescaled'][:, 0].float()
        old, valid = proposal['old_mask'][:, 0].bool(), proposal['valid_mask'][:, 0].bool()
        threshold = torch.stack([thresholds(MASK_SEED, int(i), old.shape[1:]) for i in index]).to(device)
        mask = (old | ((rescaled > threshold) & valid)).float().unsqueeze(1)

        maps = torch.stack([f.result()[0] for f in maps_future]).to(device)
        recons = {
            'zero_filled': rss(ifft_2d_img(kspace * mask)).unsqueeze(1),
            'cg_sense': conjugate_gradient_sense(kspace * mask, maps, mask, 10, 0.0).abs().unsqueeze(1),
            'sense_full': (ifft_2d_img(kspace) * maps.conj()).sum(1).abs().unsqueeze(1),
        }
        for key, (net, net_classes) in models.items():
            net_labels = torch.full((n,), net_classes.index('brain'), device=device)
            output = net(part['target'], part['kspace'], anatomy_labels=net_labels, fixed_mask=mask)['output']
            recons[key] = output[-1] if isinstance(output, list) else output

        metric_mask = part['metric_mask']
        weight = (metric_mask if metric_mask.dim() == 4 else metric_mask.unsqueeze(1)).float()
        sigma = edge_sigma(kspace, 2)
        sigma_pe = edge_sigma(kspace, 3)
        signal = (target * weight).flatten(1).sum(1) / weight.flatten(1).sum(1)
        data = {'target_max': target.flatten(1).amax(1), 'target_mean': target.flatten(1).mean(1),
                'signal_mean': signal, 'mask_area': weight.flatten(1).mean(1),
                'centre_peak': centre_scale(kspace), 'noise_sigma': sigma.to(device), 'noise_sigma_pe': sigma_pe.to(device),
                'snr': signal / sigma.to(device),
                'acquired_fraction': mask.flatten(1).mean(1)}
        per_recon = {key: scores(image.float(), target, metric_mask) for key, image in recons.items()}
        for b in range(n):
            volume = str(batch['fname'][b])
            info = MANIFEST.get(volume, {})
            base = {'set': name, 'volume': volume, 'slice': int(batch['slice_id'][b]), 'group': volume.split('_')[3],
                    'field': info.get('field'), 'scanner': info.get('model'), 'coils': info.get('coils'),
                    **{k: float(v[b]) for k, v in data.items()}}
            for key, s in per_recon.items():
                rows.append({**base, 'recon': key, **{m: float(v[b]) for m, v in s.items()}})
        start += n
        print(f'{name}: {start} slices', flush=True)
    return rows


def summarize(rows):
    sets = ('train', 'trainer_val', 'full_val')
    recons = sorted({r['recon'] for r in rows})
    data_fields = ('target_max', 'target_mean', 'signal_mean', 'mask_area', 'centre_peak', 'noise_sigma',
                   'noise_sigma_pe', 'snr', 'acquired_fraction')
    metrics = ('l1_full', 'l1_mask', 'nmse', 'psnr')
    out = {'data': {}, 'recons': {}, 'ratios': {}}
    for s in sets:
        first = [r for r in rows if r['set'] == s and r['recon'] == recons[0]]
        out['data'][s] = {'slices': len(first), 'volumes': len({r['volume'] for r in first}),
                          **{f: {'mean': float(np.nanmean([r[f] for r in first])),
                                 'median': float(np.nanmedian([r[f] for r in first])), 'nan': int(np.isnan([r[f] for r in first]).sum())} for f in data_fields}}
        out['recons'][s] = {k: {m: {'mean': float(np.mean([r[m] for r in rows if r['set'] == s and r['recon'] == k])),
                                    'median': float(np.median([r[m] for r in rows if r['set'] == s and r['recon'] == k]))}
                                for m in metrics} for k in recons}
    for s in ('trainer_val', 'full_val'):
        out['ratios'][f'{s}_over_train'] = {
            'data': {f: out['data'][s][f]['mean'] / out['data']['train'][f]['mean'] for f in data_fields},
            'recons': {k: {m: (out['recons'][s][k][m]['mean'] - out['recons']['train'][k][m]['mean']) if m == 'psnr'
                           else out['recons'][s][k][m]['mean'] / out['recons']['train'][k][m]['mean']
                           for m in metrics} for k in recons}}
    out['by_field'] = {}
    for s in sets:
        for field in sorted({r['field'] for r in rows if r['set'] == s}):
            sub = [r for r in rows if r['set'] == s and r['field'] == field]
            first = [r for r in sub if r['recon'] == recons[0]]
            out['by_field'][f'{s}_{field}'] = {
                'volumes': len({r['volume'] for r in first}), 'slices': len(first),
                'snr': float(np.nanmean([r['snr'] for r in first])),
                'noise_sigma': float(np.nanmean([r['noise_sigma'] for r in first])),
                **{f'{k}_psnr': float(np.mean([r['psnr'] for r in sub if r['recon'] == k])) for k in recons},
                **{f'{k}_l1_full': float(np.mean([r['l1_full'] for r in sub if r['recon'] == k])) for k in recons}}
    return out


def main():
    args = parse_args()
    models = {}
    for key, run in MODELS.items():
        model, checkpoint = load(args.queue / run, 'best_model.pt', args.device)
        models[key] = (model, list(checkpoint['options'].label_classes))
    sets = brain_sets(args.queue / MODELS['c6_v80'], args.chunk, args.limit)
    rows = []
    with ThreadPoolExecutor(args.threads) as pool:
        for name, loader in sets.items():
            rows += audit(name, loader, models, pool, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.limit is None:
        with open(args.out / 'slices.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    if args.limit is None:
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['ratios'], indent=1))
    for s, by_recon in summary['recons'].items():
        for k, m in by_recon.items():
            print(f"{s:12s} {k:12s} L1 full {m['l1_full']['mean']:.5f}  L1 mask {m['l1_mask']['mean']:.5f}  "
                  f"NMSE {m['nmse']['mean']:.5f}  PSNR {m['psnr']['mean']:.2f}")


if __name__ == '__main__':
    main()
