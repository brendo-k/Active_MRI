"""Cross-mask evaluation of the 1-step conv_2d and LOUPE-2D models (R=8, brain+knee VarNet, floor 0.05).

Separates how much of the conv - LOUPE gap comes from the mask and how much from the reconstructor.
Each reconstructor is evaluated on the validation set with masks drawn from:

  conv         the conv sampler's mask for that slice
  loupe        LOUPE's mask
  loupe_blur   LOUPE's probability map Gaussian-blurred (sigma 2 px), then rescaled to the same budget:
               LOUPE's coarse density without its per-pixel structure
  *_reseed     the reconstructor's own mask with a different random draw: the draw-to-draw noise floor

Every condition draws its Bernoulli mask from the same seeded uniform noise, so conditions differ only
in the probability map. Masks go through each sampler's `select_locations` (same budget, same valid
region) and into the model as `fixed_mask`.

Metrics follow the trainer: output and target are multiplied by the body mask. `trainer_*` reproduces
the logged per-anatomy numbers (batch-max data range). Paired statistics use per-slice PSNR/SSIM and
95% bootstrap intervals over per-volume mean differences.

    CUDA_VISIBLE_DEVICES=1 python scripts/eval/eval_conv_loupe_cross_mask_r8.py
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_conv_loupe_cross_mask_r8.py \
        --conv-run checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput_headmlp_ln_clipsplit --tag imginput_headmlp
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _AnatomyStats, _apply_signal_mask, build_model
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch

RUNS = {
    'loupe': Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05'),
    'conv': Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05'),
}
OUT_DIR = Path('reports/conv_vs_loupe_r8')
MASK_SEED = 1234
RESEED_OFFSET = 10_000
BLUR_SIGMA = 2.0
BOOTSTRAP_SAMPLES = 10_000
ANATOMIES = ('brain', 'knee')
COMPARISONS = {
    'loupe': [('conv', 'loupe'), ('loupe_blur', 'loupe'), ('loupe_reseed', 'loupe')],
    'conv': [('loupe', 'conv'), ('loupe_blur', 'conv'), ('conv_reseed', 'conv')],
}


def load_model(run_dir, checkpoint_name, device):
    checkpoint = torch.load(run_dir / checkpoint_name, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), int(checkpoint['epoch']) + 1


def gaussian_blur(image, sigma):
    radius = int(round(4 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    blurred = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode='reflect'), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode='reflect'), kernel.view(1, 1, -1, 1))


def blurred_outside_centre(prob, centre):
    """Blur `prob` ignoring the pre-selected centre, where it is zero, so the centre does not darken its surroundings."""
    outside = 1 - centre
    return gaussian_blur(prob * outside, BLUR_SIGMA) / gaussian_blur(outside, BLUR_SIGMA).clamp_min(1e-6) * outside


def draw(model, unscaled, proposal, seed):
    torch.manual_seed(seed)
    _, mask = model.sampler.select_locations(unscaled, proposal['old_mask'], model.budget, proposal['valid_mask'])
    return mask


@torch.no_grad()
def masks_for_batch(models, kspace, seed):
    proposals = {name: model.propose_mask(kspace) for name, model in models.items()}
    loupe, conv = proposals['loupe'], proposals['conv']
    return {
        'conv': draw(models['conv'], conv['unscaled'], conv, seed),
        'conv_reseed': draw(models['conv'], conv['unscaled'], conv, seed + RESEED_OFFSET),
        'loupe': draw(models['loupe'], loupe['unscaled'], loupe, seed),
        'loupe_reseed': draw(models['loupe'], loupe['unscaled'], loupe, seed + RESEED_OFFSET),
        'loupe_blur': draw(models['loupe'], blurred_outside_centre(loupe['unscaled'], loupe['old_mask']), loupe, seed),
    }


@torch.no_grad()
def evaluate(models, val_loader, device, max_batches=None):
    conditions = {recon: ['conv', 'loupe', 'loupe_blur', f'{recon}_reseed'] for recon in models}
    stats = {(recon, mask): _AnatomyStats(device) for recon, masks in conditions.items() for mask in masks}
    per_slice = {key: {'psnr': [], 'ssim': []} for key in stats}
    slices = {'anatomy': [], 'volume': []}
    sparsity = {mask: [] for mask in ('conv', 'loupe', 'loupe_blur')}

    for batch_index, batch in enumerate(val_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        anatomies = list(batch['anatomy'])
        slices['anatomy'] += anatomies
        slices['volume'] += list(batch['fname'])
        masks = masks_for_batch(models, kspace, MASK_SEED + batch_index)
        for name in sparsity:
            sparsity[name].append(masks[name].mean().item())
        target_4d = target if target.dim() == 4 else target.unsqueeze(1)
        for (recon, mask_name), anatomy_stats in stats.items():
            output = models[recon](target, kspace, fixed_mask=masks[mask_name])['output']
            masked_output, masked_target = _apply_signal_mask(output, target_4d)
            anatomy_stats.update(anatomies, masked_output, masked_target)
            per_slice[(recon, mask_name)]['psnr'].append(compute_psnr_torch(masked_output, masked_target).flatten().cpu())
            per_slice[(recon, mask_name)]['ssim'].append(compute_ssim_torch(masked_output, masked_target).flatten().cpu())
        print(f'batch {batch_index + 1}/{len(val_loader)}', flush=True)

    trainer = {key: {a: {k: float(v) for k, v in s.metrics[a].compute().items()} for a in s.metrics}
               for key, s in stats.items()}
    per_slice = {key: {m: torch.cat(v).numpy() for m, v in d.items()} for key, d in per_slice.items()}
    return trainer, per_slice, {k: np.array(v) for k, v in slices.items()}, {k: float(np.mean(v)) for k, v in sparsity.items()}


def paired_difference(a, b, volumes, rng):
    """Mean per-slice a - b, with a bootstrap 95% interval over volumes and the fraction of volumes where a > b."""
    names = np.unique(volumes)
    volume_means = np.array([(a - b)[volumes == v].mean() for v in names])
    draws = rng.choice(volume_means, (BOOTSTRAP_SAMPLES, len(volume_means))).mean(axis=1)
    return {'mean': float((a - b).mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            'frac_volumes_positive': float((volume_means > 0).mean()), 'volumes': len(names)}


def summarize(trainer, per_slice, slices):
    rng = np.random.RandomState(0)
    summary = {'absolute': {}, 'paired': {}}
    for (recon, mask), scores in per_slice.items():
        for anatomy in ANATOMIES:
            pick = slices['anatomy'] == anatomy
            summary['absolute'][f'{recon} recon | {mask} mask | {anatomy}'] = {
                'slice_mean_psnr': float(scores['psnr'][pick].mean()), 'slice_mean_ssim': float(scores['ssim'][pick].mean()),
                'trainer_psnr': trainer[(recon, mask)][anatomy]['PSNR'], 'trainer_ssim': trainer[(recon, mask)][anatomy]['SSIM'],
            }
    for recon, pairs in COMPARISONS.items():
        for mask, reference in pairs:
            for anatomy in ANATOMIES:
                pick = slices['anatomy'] == anatomy
                summary['paired'][f'{recon} recon: {mask} - {reference} | {anatomy}'] = {
                    metric: paired_difference(per_slice[(recon, mask)][metric][pick],
                                              per_slice[(recon, reference)][metric][pick], slices['volume'][pick], rng)
                    for metric in ('psnr', 'ssim')
                }
    for anatomy in ANATOMIES:
        pick = slices['anatomy'] == anatomy
        summary['paired'][f'own vs own: conv - loupe | {anatomy}'] = {
            metric: paired_difference(per_slice[('conv', 'conv')][metric][pick],
                                      per_slice[('loupe', 'loupe')][metric][pick], slices['volume'][pick], rng)
            for metric in ('psnr', 'ssim')
        }
    return summary


def markdown(summary, sparsity, epochs):
    lines = [f"Checkpoints: last_model.pt (conv epoch {epochs['conv']}, LOUPE epoch {epochs['loupe']}). "
             f"Mean mask density: " + ', '.join(f'{k} {v:.4f}' for k, v in sparsity.items()) + '.', '',
             '| Reconstructor | Mask | Brain PSNR / SSIM (trainer) | Knee PSNR / SSIM (trainer) |', '|---|---|---|---|']
    for recon in ('loupe', 'conv'):
        for mask in ('loupe', 'conv', 'loupe_blur', f'{recon}_reseed'):
            cells = [summary['absolute'][f'{recon} recon | {mask} mask | {a}'] for a in ANATOMIES]
            lines.append(f'| {recon} | {mask} | ' + ' | '.join(f"{c['trainer_psnr']:.2f} / {c['trainer_ssim']:.4f}" for c in cells) + ' |')
    lines += ['', '| Paired difference | Brain PSNR (dB), 95% CI | Brain SSIM | Knee PSNR (dB), 95% CI | Knee SSIM |', '|---|---|---|---|---|']
    labels = sorted({key.rsplit(' | ', 1)[0] for key in summary['paired']})
    for label in labels:
        cells = []
        for anatomy in ANATOMIES:
            d = summary['paired'][f'{label} | {anatomy}']
            cells += [f"{d['psnr']['mean']:+.3f} [{d['psnr']['ci95'][0]:+.3f}, {d['psnr']['ci95'][1]:+.3f}]",
                      f"{d['ssim']['mean']:+.4f}"]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--conv-run', type=Path, default=RUNS['conv'])
    parser.add_argument('--tag', default='', help='suffix for the output files, e.g. the conv variant')
    parser.add_argument('--max-batches', type=int, default=None, help='stop early, for a smoke test')
    args = parser.parse_args()

    runs = {**RUNS, 'conv': args.conv_run}
    loaded = {name: load_model(run_dir, args.checkpoint, args.device) for name, run_dir in runs.items()}
    models = {name: model for name, (model, _) in loaded.items()}
    epochs = {name: epoch for name, (_, epoch) in loaded.items()}

    cfg = OmegaConf.load(runs['conv'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    trainer, per_slice, slices, sparsity = evaluate(models, val_loader, args.device, args.max_batches)
    summary = summarize(trainer, per_slice, slices)
    summary.update(sparsity=sparsity, epochs=epochs, mask_seed=MASK_SEED, blur_sigma=BLUR_SIGMA,
                   runs={name: str(path) for name, path in runs.items()})
    stem = f'cross_mask_{args.tag}' if args.tag else 'cross_mask'

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{stem}.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = markdown(summary, sparsity, epochs)
    (args.out / f'{stem}.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
