"""Re-run the trainer's own validation on finished REINFORCE checkpoints, deploying the
policy both ways: its top-k logits (the old validation) and a seeded draw from it
(`reinforce_eval='sample'`, the new default and what the reconstructor trained on).

Builds a `Sequential_Trainer` from the run's hydra config with wandb disabled, loads
`best_model.pt` and calls `val_epoch()`, so the numbers come from exactly the
validation loop that training uses (background-masked metrics, val seeds). Outputs go
to reports/reinforce_sampled_val/<run>/, never the checkpoint directory.

    CUDA_VISIBLE_DEVICES=1 python scripts/revalidate_reinforce.py checkpoints/run_a checkpoints/run_b
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['WANDB_MODE'] = 'disabled'

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
import activemri.training.trainer as trainer_module
from activemri.training.trainer import Sequential_Trainer
from train_sequential import build_dataloaders, build_train_config

OUT_DIR = Path('reports/reinforce_sampled_val')
EVAL_MODES = ('topk', 'sample')


def revalidate(run, num_workers):
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = num_workers
    options = build_train_config(cfg)
    options.out_dir = [OUT_DIR / run.name]
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = Sequential_Trainer(options, train_loader, torch.device(options.device), val_loader=val_loader)
    ckpt = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=False)
    trainer.model.load_state_dict(ckpt['model'])
    trainer.epoch = ckpt.get('epoch', 0)

    captured = {}
    trainer_module.wandb.log = lambda log, **_: captured.update(log)
    results = {'run': str(run), 'epoch': trainer.epoch + 1}
    for mode in EVAL_MODES:
        trainer.model.sampler.reinforce_eval = mode
        val_loss, _, _, _ = trainer.val_epoch()
        results[mode] = {'val_loss': float(val_loss), **{
            key.removeprefix('val/'): float(value) for key, value in captured.items()
            if key.startswith('val/') and isinstance(value, (int, float)) and ('PSNR' in key or 'SSIM' in key or 'NMSE' in key)}}
        print(f"{run.name} epoch {trainer.epoch + 1} [{mode}] val_loss={val_loss:.5f} " + ' '.join(
            f"{k}={v:.4f}" for k, v in results[mode].items() if k.startswith(('brain', 'knee')) and ('PSNR' in k or 'SSIM' in k)),
            flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--num-workers', type=int, default=6)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        result = revalidate(run, args.num_workers)
        (OUT_DIR / f'{run.name}.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
