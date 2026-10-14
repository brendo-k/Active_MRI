"""How much of a trained LOUPE 2D model's quality comes from its own learned mask?

Re-evaluates each model from scripts/launch/train_loupe_anatomy.sh on its own validation
set with the mask logits replaced (reconstructor fixed, same 25% budget, same RNG
stream for every condition):

  own           the mask the model was trained with
  own_reseed    own mask, different RNG stream: the draw-to-draw noise floor
  seed_swap_s   the same anatomy's mask from seed s: different init, same anatomy
  cross_s       the other anatomy's mask from seed s
  kx_symmetric  own mask averaged with its kx mirror (same budget, no asymmetry)
  radial        own mask averaged over rings: density falloff only, no fine structure
  uniform       constant probability everywhere: an unstructured random mask

The reconstructor was co-trained with its own mask, so every non-own condition also
pays a train/test mismatch. seed_swap pays that mismatch without changing anatomy, so
the anatomy effect is cross - seed_swap, paired per slice.

Metrics follow the trainer: output and target are multiplied by the body mask
(`_apply_signal_mask`). `trainer_*` aggregates use the trainer's batch-max data range
and reproduce its logged numbers at the trainer's val batch size; paired statistics use
per-slice PSNR/SSIM with a per-slice data range, so batch makeup does not leak in, and\nconfidence intervals over per-volume means.

    CUDA_VISIBLE_DEVICES=1 python scripts/eval/eval_cross_mask.py --anatomies brain
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from activemri.training.trainer import _apply_signal_mask, build_model
from activemri.utils.evaluate import build_metrics
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from eval_flipped_mask import CONFIG_NUM_COILS, RNG_SEED_A, RNG_SEED_B, mirror

ANATOMIES = ('brain', 'knee')
RADIAL_BINS = 32
EPS = 1e-4


def radial_average(logits, slope):
    h, w = logits.shape[1:3]
    ky = (torch.arange(h) - h // 2) / (h / 2)
    kx = (torch.arange(w) - w // 2) / (w / 2)
    bins = (torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) * RADIAL_BINS).long().clamp(max=RADIAL_BINS)
    prob = torch.sigmoid(slope * logits.cpu()).squeeze()
    counts = torch.bincount(bins.flatten())
    ring_mean = torch.bincount(bins.flatten(), weights=prob.flatten()) / counts
    smoothed = ring_mean[bins].clamp(EPS, 1 - EPS)
    return (torch.log(smoothed / (1 - smoothed)) / slope).reshape(logits.shape).to(logits)


def load_val_batches(anatomy, args):
    """Validation batches as (kspace, target, volume names); same data as eval_flipped_mask."""
    _, val_loader = build_fastmri_dataloaders(
        data_root=args.data_root, anatomies=[anatomy], image_shape=[256, 256],
        batch_size=args.batch_size, num_coils=CONFIG_NUM_COILS[anatomy],
        num_workers=args.num_workers, contrast='T2',
    )
    batches = []
    for i, batch in enumerate(val_loader):
        if args.max_batches and i >= args.max_batches:
            break
        batches.append((batch['kspace'], batch['target'], list(batch['fname'])))
    return batches


def load_logits(root, anatomy, seed):
    ckpt = torch.load(root / anatomy / f'seed_{seed}' / 'best_model.pt', map_location='cpu', weights_only=False)
    key = next(k for k in ckpt['model'] if k.endswith('mask_net.mask'))
    return ckpt['model'][key].float()


@torch.no_grad()
def evaluate(model, batches, device):
    """Trainer-style aggregates plus per-slice PSNR/SSIM, all over the body mask."""
    metrics = build_metrics(device=device)
    psnr, ssim = [], []
    for kspace, target, _ in batches:
        kspace, target = kspace.to(device), target.to(device)
        output = model(target, kspace)['output']
        if isinstance(output, list):
            output = output[-1]
        target_4d = target if target.dim() == output.dim() else target.unsqueeze(1)
        masked_output, masked_target = _apply_signal_mask(output, target_4d)
        data_range = target_4d.max()
        metrics['PSNR'].data_range = data_range
        metrics['SSIM'].data_range = data_range
        metrics.update(masked_output, masked_target)
        psnr.append(compute_psnr_torch(masked_output, masked_target).flatten().cpu())
        ssim.append(compute_ssim_torch(masked_output, masked_target).flatten().cpu())
    trainer = {f'trainer_{k.lower()}': float(v) for k, v in metrics.compute().items()}
    return {**trainer, 'psnr': torch.cat(psnr), 'ssim': torch.cat(ssim)}


def evaluate_run(root, anatomy, seed, seeds, batches, device):
    ckpt = torch.load(root / anatomy / f'seed_{seed}' / 'best_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    mask_param = model.sampler.mask_net.mask
    slope = ckpt['options'].slope
    own = mask_param.detach().clone()
    other = ANATOMIES[1 - ANATOMIES.index(anatomy)]

    conditions = {
        'own': (own, RNG_SEED_A),
        'own_reseed': (own, RNG_SEED_B),
        **{f'seed_swap_{s}': (load_logits(root, anatomy, s), RNG_SEED_A) for s in seeds if s != seed},
        **{f'cross_{s}': (load_logits(root, other, s), RNG_SEED_A) for s in seeds},
        'kx_symmetric': (0.5 * (own + mirror(own, [2])), RNG_SEED_A),
        'radial': (radial_average(own, slope), RNG_SEED_A),
        'uniform': (torch.zeros_like(own), RNG_SEED_A),
    }
    results = {}
    for name, (logits, rng_seed) in conditions.items():
        with torch.no_grad():
            mask_param.copy_(logits.to(own))
        torch.manual_seed(rng_seed)
        results[name] = evaluate(model, batches, device)
        r = results[name]
        print(f'  {anatomy}/seed_{seed} {name:13s} trainer PSNR={r["trainer_psnr"]:.3f} SSIM={r["trainer_ssim"]:.4f}'
              f' | per-slice PSNR={r["psnr"].mean():.3f} SSIM={r["ssim"].mean():.4f}', flush=True)
    return results


def group_delta(per_seed, prefix, metric):
    """Per-slice delta vs own, averaged over the conditions named prefix* in a run, then over runs."""
    runs = []
    for r in per_seed.values():
        names = [n for n in r if n == prefix or n.startswith(prefix + '_')]
        runs.append(torch.stack([r[n][metric] - r['own'][metric] for n in names]).mean(0))
    return torch.stack(runs).mean(0)


def paired_ci(delta, volumes):
    """Mean over slices, a 95% CI over per-volume means (slices of a volume are correlated),
    and the share of volumes where delta < 0."""
    names = sorted(set(volumes))
    index = torch.tensor([names.index(v) for v in volumes])
    per_volume = torch.zeros(len(names)).index_add_(0, index, delta) / torch.bincount(index).float()
    mean = delta.mean().item()
    half = 1.96 * per_volume.std().item() / np.sqrt(len(names))
    return {'mean': mean, 'ci': [mean - half, mean + half], 'frac_volumes_worse': (per_volume < 0).float().mean().item()}


def summarize(anatomy, per_seed, volumes):
    print(f'\n[{anatomy}] per-slice delta vs own, averaged over seeds (n={len(volumes)} slices, {len(set(volumes))} volumes)')
    summary = {'trainer_own': {m: float(np.mean([r['own'][f'trainer_{m}'] for r in per_seed.values()]))
                               for m in ('psnr', 'ssim')}}
    print(f'  own (trainer metric)  PSNR={summary["trainer_own"]["psnr"]:.3f} SSIM={summary["trainer_own"]["ssim"]:.4f}')
    for metric in ('psnr', 'ssim'):
        deltas = {g: group_delta(per_seed, g, metric)
                  for g in ('own_reseed', 'seed_swap', 'cross', 'kx_symmetric', 'radial', 'uniform')}
        deltas['anatomy_effect'] = deltas['cross'] - deltas['seed_swap']
        summary[metric] = {g: paired_ci(d, volumes) for g, d in deltas.items()}
        for g, s in summary[metric].items():
            print(f'  {metric} {g:15s} {s["mean"]:+.4f}  95% CI [{s["ci"][0]:+.4f}, {s["ci"][1]:+.4f}]'
                  f'  worse on {100 * s["frac_volumes_worse"]:.0f}% of volumes')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/loupe_anatomy'))
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomies', nargs='+', default=list(ANATOMIES))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch-size', type=int, default=4, help='trainer val batch size, for trainer_* metrics')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=0, help='limit val batches (0 = all)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    for anatomy in args.anatomies:
        batches = load_val_batches(anatomy, args)
        print(f'{anatomy}: {sum(b[0].shape[0] for b in batches)} val slices', flush=True)
        per_seed = {s: evaluate_run(args.root, anatomy, s, args.seeds, batches, device) for s in args.seeds}
        volumes = [v for _, _, names in batches for v in names]
        summary = summarize(anatomy, per_seed, volumes)
        out_dir = args.root / 'analysis'
        torch.save(per_seed, out_dir / f'cross_mask_eval_{anatomy}_per_slice.pt')
        (out_dir / f'cross_mask_eval_{anatomy}.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
