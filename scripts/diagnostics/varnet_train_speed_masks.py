"""Learned LOUPE masks of a queue's runs: mean probability, correlation with each seed's control mask, and the radial
profile of p(k) = sigmoid(slope * logits) (pixels from the k-space centre), from last_model.pt on the CPU.

    python scripts/diagnostics/varnet_train_speed_masks.py reports/varnet_train_speed/masks.json
    python scripts/diagnostics/varnet_train_speed_masks.py reports/muon_layer_speed/masks.json \\
        --queue-dir ~/data/seqmri_checkpoints/muon_layer_speed --control ref --runs ref_s42 e30_step_s42
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401  (activemri, for unpickling the checkpoints' options)

QUEUE_DIR = Path('~/data/seqmri_checkpoints/varnet_train_speed').expanduser()
RUN_NAME = re.compile(r'(?P<arm>.+)_s(?P<seed>\d+)$')
RINGS = ((0, 20), (20, 40), (40, 80), (80, 120), (120, 160), (160, 230))
EPOCH_LINE = re.compile(r'Epoch = \[ *(\d+)/ *(\d+)\]')


def finished(run_dir):
    log = run_dir.with_suffix('.log')
    epochs = EPOCH_LINE.findall(log.read_text()) if log.exists() else []
    return bool(epochs) and epochs[-1][0] == epochs[-1][1]


def probabilities(run_dir):
    checkpoint = torch.load(run_dir / 'last_model.pt', map_location='cpu', weights_only=False)
    logits = checkpoint['model']['sampler.mask_net.mask'].float().squeeze()
    return torch.sigmoid(checkpoint['options'].slope * logits).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path)
    parser.add_argument('--queue-dir', type=Path, default=QUEUE_DIR)
    parser.add_argument('--control', default='control', help='arm each run is correlated with, per seed')
    parser.add_argument('--runs', nargs='*', help='run names (default: every finished run)')
    args = parser.parse_args()
    queue_dir = args.queue_dir.expanduser()
    runs = [queue_dir / name for name in args.runs] if args.runs else sorted(
        p for p in queue_dir.iterdir() if p.is_dir() and RUN_NAME.match(p.name) and '.attempt' not in p.name
        and not p.is_symlink() and (p / 'last_model.pt').exists() and finished(p))
    masks = {run.name: probabilities(run) for run in runs}
    height, width = next(iter(masks.values())).shape
    yy, xx = np.mgrid[:height, :width]
    radius = np.hypot(yy - height // 2, xx - width // 2)
    result = {}
    for name, p in masks.items():
        seed = RUN_NAME.match(name)['seed']
        control = masks.get(f'{args.control}_s{seed}')
        result[name] = {
            'mean_p': float(p.mean()),
            'corr_with_control_same_seed': float(np.corrcoef(p.ravel(), control.ravel())[0, 1]) if control is not None else None,
            'radial_mean_p': {f'{lo}-{hi}': float(p[(radius >= lo) & (radius < hi)].mean()) for lo, hi in RINGS},
        }
    args.out.write_text(json.dumps(result, indent=1))
    for name, row in result.items():
        print(f"{name:24s} mean {row['mean_p']:.4f} corr {row['corr_with_control_same_seed'] or float('nan'):.3f} "
              + ' '.join(f'{v:.3f}' for v in row['radial_mean_p'].values()))


if __name__ == '__main__':
    main()
