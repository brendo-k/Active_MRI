"""How far the exact step-time changes move a real training, against how far the same code moves when run twice.

Trains two epochs on 5% of the data four times, through train.py with bf16 and compiled U-Nets as in a run: the code from
before the changes twice, and the code after them with and without the options the arms use. Those options change no math,
only Adam's kernels and the sensitivity net's layout, so that last pair shows how far a pure rounding change moves a training.
The cuDNN autotuner is off. It reports the relative distance between final weights, whole-model and per parameter, and each
run's last losses. A run whose checkpoint is already in the work dir is not trained again.

    python scripts/diagnostics/muon_layer_speed_equivalence.py --before <repo> --after <repo> --work-dir <dir> --out equivalence.json
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch

COMMON = ['recon=varnet', 'datasets=fastmri_local_v1_flair_knee', 'misc.resolution=[320,320]', 'sampler.accelerations=[8.0]',
          'sampler.budget_over=acquired', 'training.num_step=1', 'sampler.sampler_type=loupe_2d', 'sampler.loupe_vd_init_power=4',
          'sampler.anatomy_conditioning=false', 'datasets.limit_volumes=0.05', 'training.batch_size=16', 'training.num_epochs=2',
          'training.lr_step_size=2', 'training.seed=42', 'training.optimizer=muon', 'recon.norm=layer', 'sampler.sens_norm=layer',
          'training.cudnn_benchmark=false', 'training.ema_decay=0.999']
FAST = ['training.fused_adam=true', 'training.channels_last_unets=sens']
RUNS = {'before': ('before', []), 'before_again': ('before', []), 'after': ('after', FAST), 'after_plain': ('after', [])}
PAIRS = [('before', 'before_again'), ('before', 'after'), ('before', 'after_plain'), ('after_plain', 'after')]
LAST_EPOCH = re.compile(r'Epoch = \[ *2/ *2\] TrainLoss = (\S+) ValLoss = (\S+)')
MASK_PARAMETER = 'mask_net.mask'


def train(name, repo, overrides, work_dir):
    environment = {**os.environ, 'PYTHONPATH': repo, 'WANDB_MODE': 'disabled', 'CC': 'gcc', 'CXX': 'g++'}
    log = work_dir / f'{name}.log'
    if not (work_dir / name / 'last_model.pt').exists():
        with log.open('w') as handle:
            subprocess.run([sys.executable, 'train.py', '--model', 'sequential', *COMMON, *overrides, f'output.out_dir={work_dir / name}'],
                           cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    train_loss, val_loss = LAST_EPOCH.findall(log.read_text(errors='replace'))[-1]
    return {'train_loss': float(train_loss), 'val_loss': float(val_loss)}


def weights(work_dir, name):
    return torch.load(work_dir / name / 'last_model.pt', map_location='cpu', weights_only=False)['model']


def distance(first, second):
    keys = [key for key in first if first[key].is_floating_point()]
    per_parameter = {key: ((first[key] - second[key]).norm() / first[key].norm().clamp(min=1e-30)).item() for key in keys}
    squared = sum((first[key] - second[key]).pow(2).sum() for key in keys) / sum(first[key].pow(2).sum() for key in keys)
    mask = next(value for key, value in per_parameter.items() if key.endswith(MASK_PARAMETER))
    worst = max(per_parameter, key=per_parameter.get)
    return {'whole_model': squared.sqrt().item(), 'median_parameter': statistics.median(per_parameter.values()),
            'worst_parameter': per_parameter[worst], 'worst_parameter_name': worst, 'mask': mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.after)  # the checkpoints pickle activemri's config class

    losses = {name: train(name, args.before if side == 'before' else args.after, overrides, args.work_dir)
              for name, (side, overrides) in RUNS.items()}
    distances = {f'{first} vs {second}': distance(weights(args.work_dir, first), weights(args.work_dir, second)) for first, second in PAIRS}
    result = {'losses': losses, 'distances': distances, 'before': args.before, 'after': args.after, 'after_options': FAST}
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
