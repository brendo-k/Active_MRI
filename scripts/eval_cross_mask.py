"""How much of a trained LOUPE 2D model's quality comes from its own learned mask?

Re-evaluates each model from scripts/train_loupe_anatomy.sh on its own validation
set with the mask logits replaced (reconstructor fixed, same 25% budget, same RNG
stream for every condition):

  own          the mask the model was trained with
  cross        the other anatomy's learned mask (mean over its 3 seeds)
  radial       own mask averaged over rings: density falloff only, no fine structure
  uniform      constant probability everywhere: an unstructured random mask

The reconstructor was co-trained with its own mask, so every non-own condition also
pays a train/test mismatch. A fair random baseline needs a reconstructor trained with
mask_train_mode=random; read `uniform` here as a lower bound on that baseline.

    CUDA_VISIBLE_DEVICES=1 python scripts/eval_cross_mask.py --anatomies brain
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from activemri.training.trainer import build_model
from eval_flipped_mask import RNG_SEED_A, evaluate, load_val_batches

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


def load_logits(root, anatomy, seed):
    ckpt = torch.load(root / anatomy / f'seed_{seed}' / 'best_model.pt', map_location='cpu', weights_only=False)
    key = next(k for k in ckpt['model'] if k.endswith('mask_net.mask'))
    return ckpt['model'][key].float()


def evaluate_run(root, anatomy, seed, seeds, batches, device):
    ckpt = torch.load(root / anatomy / f'seed_{seed}' / 'best_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    mask_param = model.sampler.mask_net.mask
    slope = ckpt['options'].slope
    own = mask_param.detach().clone()
    other = ANATOMIES[1 - ANATOMIES.index(anatomy)]

    def run(logits):
        with torch.no_grad():
            mask_param.copy_(logits.to(own))
        torch.manual_seed(RNG_SEED_A)
        return evaluate(model, batches, device)

    results = {'own': run(own), 'radial': run(radial_average(own, slope)), 'uniform': run(torch.zeros_like(own))}
    cross = [run(load_logits(root, other, s)) for s in seeds]
    results['cross'] = {m: float(np.mean([c[m] for c in cross])) for m in cross[0]}
    for name, r in results.items():
        print(f'  {anatomy}/seed_{seed} {name:8s} L1={r["l1"]:.5f} PSNR={r["psnr"]:.3f} SSIM={r["ssim"]:.4f}', flush=True)
    return results


def summarize(anatomy, per_seed):
    print(f'\n[{anatomy}] mean over seeds; delta is vs own mask (negative PSNR/SSIM = worse)')
    for name in ('own', 'cross', 'radial', 'uniform'):
        mean = {m: np.mean([r[name][m] for r in per_seed.values()]) for m in ('l1', 'psnr', 'ssim')}
        delta = {m: np.mean([r[name][m] - r['own'][m] for r in per_seed.values()]) for m in mean}
        print(f'  {name:8s} L1={mean["l1"]:.5f} PSNR={mean["psnr"]:.3f} SSIM={mean["ssim"]:.4f}'
              f'   dPSNR {delta["psnr"]:+.3f}  dSSIM {delta["ssim"]:+.4f}  dL1 {delta["l1"]:+.5f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/loupe_anatomy'))
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomies', nargs='+', default=list(ANATOMIES))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=0, help='limit val batches (0 = all)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    for anatomy in args.anatomies:
        batches = load_val_batches(anatomy, args)
        print(f'{anatomy}: {sum(b[0].shape[0] for b in batches)} val slices', flush=True)
        per_seed = {s: evaluate_run(args.root, anatomy, s, args.seeds, batches, device) for s in args.seeds}
        summarize(anatomy, per_seed)
        out = args.root / 'analysis' / f'cross_mask_eval_{anatomy}.json'
        out.write_text(json.dumps(per_seed, indent=2))


if __name__ == '__main__':
    main()
