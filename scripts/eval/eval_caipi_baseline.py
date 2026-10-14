"""Evaluate fixed-CAIPI VarNet checkpoints on the full brain T2 validation set.

Uses the same loader and metric code as scripts/eval/eval_cross_mask.py, so L1/PSNR/SSIM are
directly comparable to its LOUPE rows. Only brain is evaluated: brain has real train/val
directories in every setup, whereas knee is split by slice in the per-anatomy LOUPE runs
and by volume in the balanced mixed-anatomy runs, so a knee val set is contaminated for
whichever model trained on the other split.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval/eval_caipi_baseline.py checkpoints/varnet_caipi2d_r4_brain_knee
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
from dataclasses import MISSING, fields
from pathlib import Path

import torch

import _path  # noqa: F401
from activemri.config import TrainConfig
from activemri.training.trainer import build_model
from eval_flipped_mask import RNG_SEED_A, evaluate, load_val_batches


def with_current_defaults(options):
    """Older checkpoints pickled a TrainConfig from before some fields existed."""
    for f in fields(TrainConfig):
        if not hasattr(options, f.name):
            setattr(options, f.name, f.default_factory() if f.default is MISSING else f.default)
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dirs', type=Path, nargs='+')
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    batches = load_val_batches('brain', args)
    print(f'brain: {sum(b[0].shape[0] for b in batches)} val slices', flush=True)
    for run_dir in args.checkpoint_dirs:
        ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
        model = build_model(with_current_defaults(ckpt['options'])).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        torch.manual_seed(RNG_SEED_A)
        r = evaluate(model, batches, device)
        print(f'  {run_dir.name} (epoch {ckpt["epoch"] + 1}) '
              f'L1={r["l1"]:.5f} PSNR={r["psnr"]:.3f} SSIM={r["ssim"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
