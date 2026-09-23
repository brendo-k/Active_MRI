"""Example script showing improved REINFORCE 2D training with stability fixes.

Compare this with reinforce2d_vs_greedy_loupe.py to see the impact of:
  - Higher episode count (32 vs 6) for lower-variance gradient estimates
  - Lower learning rate (1e-4 vs 0.2) to prevent policy collapse
  - Higher baseline momentum (0.99 vs 0.9) to prevent baseline lag
  - Entropy regularization (0.01) to encourage exploration
  - Gradient clipping (1.0) to prevent gradient spikes
  - Advantage normalization for round-independent scaling

Usage:
    python scripts/reinforce2d_stable.py --anatomy brain \\
        --resolution 48 --acceleration 4 --out-dir runs/reinforce2d_stable
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.datasets.fastmri_data import FastMRIDataset, _resolve_anatomy_dirs
from activemri.samplers.oracle import (
    _cg_solve_batch, estimate_sense_maps, greedy_2d, reinforce_2d,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', default='brain', choices=['brain', 'knee', 'breast', 'prostate'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--num-slices', type=int, default=1, help='search slices (2D reward is expensive)')
    parser.add_argument('--resolution', type=int, default=48)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--acceleration', type=float, default=4.0)
    parser.add_argument('--preselect', type=int, default=6, help='side length of the fixed centre block')
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--acs', type=int, default=16)
    parser.add_argument('--cg-iters', type=int, default=10)
    parser.add_argument('--reinforce-batch', type=int, default=None,
                        help='defaults to budget // 6 (~6 rounds), matching greedy')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def load_slices(args):
    train_dir, _ = _resolve_anatomy_dirs(args.data_root, args.anatomy, args.contrast)
    res = [args.resolution, args.resolution]

    probe = FastMRIDataset(root=train_dir, anatomy=args.anatomy, image_shape=res,
                           num_coils=args.num_coils, min_energy_fraction=0.0, slice_ids=[0])
    file_name = probe.examples[0][0].name

    dataset = FastMRIDataset(
        root=train_dir, anatomy=args.anatomy, image_shape=res, num_coils=args.num_coils,
        min_energy_fraction=0.0, file_names=[file_name], slice_ids=list(range(args.num_slices)),
    )

    device = torch.device(args.device)
    slices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        kspace = torch.view_as_complex(sample['kspace']).to(device).to(torch.complex64)
        maps, support = estimate_sense_maps(kspace, acs=args.acs)
        slices.append({'kspace': kspace, 'maps': maps, 'support': support})
    return slices, file_name


def nmse_on(kspace, maps, support, mask):
    C, H, W = kspace.shape
    device = kspace.device
    with torch.no_grad():
        reference = _cg_solve_batch(kspace, maps, torch.ones(1, H, W, device=device), 20, 1e-2)[0]
        reference = reference * support
        ref_energy = reference.abs().pow(2).sum().clamp(min=1e-12)
        recon = _cg_solve_batch(kspace, maps, mask.unsqueeze(0), 20, 1e-2)[0]
        return (((recon - reference).abs().pow(2) * support).sum() / ref_energy).item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    slices, file_name = load_slices(args)
    device = torch.device(args.device)
    H = W = args.resolution
    budget = int(round(H * W / args.acceleration))
    reinforce_batch = args.reinforce_batch or max(1, budget // 6)
    print(f'{args.anatomy} {file_name}: H=W={H}, budget={budget} points (R={args.acceleration:g})',
          flush=True)

    search = slices[0]

    print('\n=== greedy_2d (baseline) ===', flush=True)
    greedy = greedy_2d(search['kspace'], search['maps'], budget=budget, batch=reinforce_batch,
                       preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
                       support=search['support'], verbose=verbose)

    print('\n=== reinforce_2d: default params (baseline) ===', flush=True)
    reinforce_default = reinforce_2d(
        search['kspace'], search['maps'], budget=budget, batch=reinforce_batch,
        preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
        episodes_per_update=6,
        num_updates=25,
        lr=0.2,
        baseline_momentum=0.9,
        support=search['support'], verbose=verbose, log_prefix='dflt ')

    print('\n=== reinforce_2d: stable params (improved) ===', flush=True)
    reinforce_stable = reinforce_2d(
        search['kspace'], search['maps'], budget=budget, batch=reinforce_batch,
        preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
        episodes_per_update=32,
        num_updates=100,
        lr=0.0001,
        baseline_momentum=0.99,
        entropy_weight=0.01,
        clip_grad_norm=1.0,
        normalize_advantages=True,
        support=search['support'], verbose=verbose, log_prefix='stbl ')

    results = {
        'greedy_2d': {'nmse': nmse_on(search['kspace'], search['maps'], search['support'],
                                      greedy['mask'])},
        'reinforce_2d_default': {'nmse': nmse_on(search['kspace'], search['maps'], search['support'],
                                                  reinforce_default['mask'])},
        'reinforce_2d_stable': {'nmse': nmse_on(search['kspace'], search['maps'], search['support'],
                                                 reinforce_stable['mask'])},
    }

    print(f'\n{"method":<30s} {"final nmse":>12s}', flush=True)
    for name, entry in results.items():
        print(f'{name:<30s} {entry["nmse"]:12.6g}', flush=True)

    # Plot learning curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(reinforce_default['history'], marker='o', label='REINFORCE (default)', alpha=0.7)
    ax.plot(reinforce_stable['history'], marker='s', label='REINFORCE (stable)', alpha=0.7)
    ax.axhline(results['greedy_2d']['nmse'], ls='--', color='k', lw=1, label='greedy_2d')
    ax.set_xlabel('policy-gradient update')
    ax.set_ylabel('nmse')
    ax.set_title(f'REINFORCE convergence: default vs stable hyperparameters ({args.anatomy})')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'reinforce_comparison.png', dpi=150)
    plt.close(fig)

    payload = {
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'file_name': file_name,
        'budget_points': budget,
        'reinforce_batch': reinforce_batch,
        'greedy_history': greedy['history'],
        'reinforce_default_history': reinforce_default['history'],
        'reinforce_stable_history': reinforce_stable['history'],
        'results': results,
    }
    (args.out_dir / 'results.json').write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
