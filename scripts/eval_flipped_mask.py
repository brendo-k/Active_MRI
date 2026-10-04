"""Does a trained LOUPE 2D reconstruction depend on the orientation of its learned mask?

Re-evaluates each model from scripts/train_loupe_anatomy.sh on its own validation
set after editing the mask logits (k-space is DC-centred, so a flip about DC maps
index i to -i mod N):

  orig          unchanged (RNG seed A)
  orig_reseed   unchanged, different RNG seed: the draw-to-draw noise floor
  kx_flip       mirror along kx
  ky_flip       mirror along ky (control: learned ky profile is already symmetric)
  point_flip    mirror along both, i.e. the conjugate-symmetric partner -k
  kx_symmetric  average of the mask and its kx mirror (same budget, no asymmetry)

The reconstructor stays fixed, so a metric drop means it relies on the learned layout.
The binary mask is a random draw from the rescaled probabilities, so every condition
reuses the same RNG stream.

    CUDA_VISIBLE_DEVICES=1 python scripts/eval_flipped_mask.py --anatomies brain
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch

CONFIG_NUM_COILS = {'brain': 16, 'knee': 15}
LOSS_META = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
RNG_SEED_A, RNG_SEED_B = 1234, 4321


def mirror(logits, dims):
    return torch.roll(torch.flip(logits, dims=dims), shifts=(1,) * len(dims), dims=dims)


CONDITIONS = {
    'orig': (lambda m: m, RNG_SEED_A),
    'orig_reseed': (lambda m: m, RNG_SEED_B),
    'kx_flip': (lambda m: mirror(m, [2]), RNG_SEED_A),
    'ky_flip': (lambda m: mirror(m, [1]), RNG_SEED_A),
    'point_flip': (lambda m: mirror(m, [1, 2]), RNG_SEED_A),
    'kx_symmetric': (lambda m: 0.5 * (m + mirror(m, [2])), RNG_SEED_A),
}


def load_val_batches(anatomy, args):
    _, val_loader = build_fastmri_dataloaders(
        data_root=args.data_root, anatomies=[anatomy], image_shape=[256, 256],
        batch_size=args.batch_size, num_coils=CONFIG_NUM_COILS[anatomy],
        num_workers=args.num_workers, contrast='T2',
    )
    batches = []
    for i, batch in enumerate(val_loader):
        if args.max_batches and i >= args.max_batches:
            break
        batches.append((batch['kspace'], batch['target']))
    return batches


@torch.no_grad()
def evaluate(model, batches, device, loss_type='l1'):
    totals = {'l1': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    count = 0
    for kspace, target in batches:
        kspace, target = kspace.to(device), target.to(device)
        pred = model(target, kspace)
        loss, _ = model.loss(pred, {'target': target, 'kspace': kspace}, LOSS_META, loss_type)
        output = pred['output']
        target_4d = target if target.dim() == 4 else target.unsqueeze(1)
        n = target.shape[0]
        totals['l1'] += loss.item() * n
        totals['psnr'] += compute_psnr_torch(output, target_4d).mean().item() * n
        totals['ssim'] += compute_ssim_torch(output, target_4d).mean().item() * n
        count += n
    return {k: v / count for k, v in totals.items()}


def evaluate_run(run_dir, batches, device):
    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    mask_param = model.sampler.mask_net.mask
    original = mask_param.detach().clone()
    results = {}
    for name, (edit, rng_seed) in CONDITIONS.items():
        with torch.no_grad():
            mask_param.copy_(edit(original))
        torch.manual_seed(rng_seed)
        results[name] = evaluate(model, batches, device)
        print(f'  {run_dir.parent.name}/{run_dir.name} {name:13s} '
              f'L1={results[name]["l1"]:.5f} PSNR={results[name]["psnr"]:.3f} SSIM={results[name]["ssim"]:.4f}',
              flush=True)
    return results, ckpt['epoch'] + 1


def summarize(anatomy, per_seed):
    print(f'\n[{anatomy}] change vs orig, mean over seeds [min, max]  (negative PSNR/SSIM delta = worse)')
    summary = {}
    for name in CONDITIONS:
        if name == 'orig':
            continue
        row = {}
        for metric in ('l1', 'psnr', 'ssim'):
            deltas = [r[name][metric] - r['orig'][metric] for r in per_seed.values()]
            row[metric] = {'mean': float(np.mean(deltas)), 'min': float(np.min(deltas)), 'max': float(np.max(deltas))}
        summary[name] = row
        print(f'  {name:13s} dPSNR {row["psnr"]["mean"]:+.3f} [{row["psnr"]["min"]:+.3f}, {row["psnr"]["max"]:+.3f}]'
              f'   dSSIM {row["ssim"]["mean"]:+.4f}   dL1 {row["l1"]["mean"]:+.5f}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/loupe_anatomy'))
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomies', nargs='+', default=['brain', 'knee'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=0, help='limit val batches (0 = all)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    for anatomy in args.anatomies:
        print(f'loading {anatomy} validation set...', flush=True)
        batches = load_val_batches(anatomy, args)
        print(f'{anatomy}: {sum(b[0].shape[0] for b in batches)} val slices', flush=True)
        per_seed, epochs = {}, {}
        for seed in args.seeds:
            per_seed[seed], epochs[seed] = evaluate_run(args.root / anatomy / f'seed_{seed}', batches, device)
        summary = summarize(anatomy, per_seed)
        out = args.root / 'analysis' / f'flip_eval_{anatomy}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'checkpoint_epochs': epochs, 'per_seed': per_seed, 'delta_vs_orig': summary}, indent=2))


if __name__ == '__main__':
    main()
