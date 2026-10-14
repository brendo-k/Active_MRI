"""Search for the best undersampling pattern using CG-SENSE as the oracle.

CG-SENSE has no learnable parameters, so the reconstruction is a fixed linear
function of the mask. That makes it possible to score a candidate mask directly
instead of training a sampler against it.

For 1D line masks the normal operator is exactly block diagonal in the readout
index, so each score is a batch of small exact solves rather than a CG run --
see :mod:`activemri.samplers.oracle`. Greedy line selection is then cheap enough
to run to convergence, and the result is compared against equispaced, random and
low-pass baselines scored with the same oracle.

The 2D arm drops the separability and falls back to matrix-free batched greedy
on the k-space residual, to see what an unconstrained pattern buys.

    python scripts/oracle/oracle_sampling_search.py --anatomy brain --num-slices 8 \\
        --acceleration 4 --out-dir runs/oracle/brain_R4
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
    LineOracle, MeanLineOracle, center_lines, estimate_sense_maps, greedy_2d, greedy_lines,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', default='brain', choices=['brain', 'knee', 'breast', 'prostate'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--num-slices', type=int, default=8,
                        help='slices averaged into the greedy objective (the population oracle)')
    parser.add_argument('--held-out-slices', type=int, default=8,
                        help='further slices, never seen by the search, used to test generalisation')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--acceleration', type=float, default=4.0)
    parser.add_argument('--preselect', type=int, default=8, help='fixed centre (ACS) lines')
    parser.add_argument('--objective', default='mse_noise',
                        choices=['mse_noise', 'mse', 'trace', 'logdet'],
                        help="'mse' is noiseless and degenerates to a low-pass block, since a "
                             'linear reconstructor cannot recover an unsampled frequency at all. '
                             "'mse_noise' adds the estimator variance, which is what makes the "
                             'search trade energy capture against conditioning.')
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--noise-level', type=float, default=0.02,
                        help='NMSE that noise alone costs at full sampling; puts the noise term '
                             'on the same scale as the aliasing error')
    parser.add_argument('--maps', default='estimated', choices=['estimated', 'stored'],
                        help="'estimated' re-derives sensitivities from the preprocessed k-space. "
                             "The stored ESPIRiT maps live on the raw grid and no longer align "
                             'after the loader decimates and crops.')
    parser.add_argument('--acs', type=int, default=24, help='centre block used to estimate maps')
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--skip-2d', action='store_true')
    parser.add_argument('--greedy-2d-batch', type=int, default=64)
    parser.add_argument('--cg-iters', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def load_slices(args):
    """Load slices with their precomputed ESPIRiT maps, as [C, H, W] complex tensors."""
    train_dir, _ = _resolve_anatomy_dirs(args.data_root, args.anatomy, args.contrast)
    total = args.num_slices + args.held_out_slices
    res = [args.resolution, args.resolution]

    probe = FastMRIDataset(root=train_dir, anatomy=args.anatomy, image_shape=res,
                           num_coils=args.num_coils, min_energy_fraction=0.0, slice_ids=[0])
    file_name = probe.examples[0][0].name

    dataset = FastMRIDataset(
        root=train_dir, anatomy=args.anatomy, image_shape=res, num_coils=args.num_coils,
        min_energy_fraction=0.0, file_names=[file_name], slice_ids=list(range(total)),
    )

    device = torch.device(args.device)
    slices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        kspace = torch.view_as_complex(sample['kspace']).to(device).to(torch.complex64)
        if args.maps == 'estimated':
            maps, support = estimate_sense_maps(kspace, acs=args.acs)
        else:
            if sample['espirit_maps'] is None:
                raise RuntimeError(f"{file_name} has no precomputed sense maps")
            maps = sample['espirit_maps'].to(device).to(torch.complex64)
            support = (maps.abs().pow(2).sum(dim=0) > 1e-6).float()
        slices.append({'kspace': kspace, 'maps': maps, 'support': support,
                       'target': sample['target'], 'slice_id': sample['slice_id']})
    return slices, file_name


def baseline_masks(W, budget, preselect, device, seed):
    """Reference 1D patterns, all at the same line budget."""
    masks = {}

    center = center_lines(W, preselect, device)

    equispaced = center.clone()
    remaining = budget - int(equispaced.sum().item())
    free = (equispaced == 0).nonzero(as_tuple=False).flatten()
    if remaining > 0:
        pick = torch.linspace(0, free.numel() - 1, remaining).round().long()
        equispaced[free[pick.unique()]] = 1.0
    masks['equispaced+acs'] = equispaced

    generator = torch.Generator(device='cpu').manual_seed(seed)
    rnd = center.clone()
    free = (rnd == 0).nonzero(as_tuple=False).flatten()
    remaining = budget - int(rnd.sum().item())
    rnd[free[torch.randperm(free.numel(), generator=generator)[:remaining]]] = 1.0
    masks['random+acs'] = rnd

    masks['low_pass'] = center_lines(W, budget, device)
    return masks


def plot_masks(out_dir, results, budget, W, key):
    order = ['greedy(oracle)'] + [k for k in results if k != 'greedy(oracle)']
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 1.05 * len(order)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), order):
        ax.imshow(results[name]['mask'][None, :], cmap='gray', aspect='auto',
                  extent=[0, W, 0, 1], vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_ylabel(f"{name}\n{results[name]['score']:.4f}",
                      rotation=0, ha='right', va='center', fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('phase-encode line')
    fig.suptitle(f'1D masks at {budget} lines (label = {key} on search slices)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'masks_1d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_curves(out_dir, greedy, results, history_2d, budget):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    lines = np.arange(budget - len(greedy['history']) + 1, budget + 1)
    ax[0].semilogy(lines, greedy['history'], label='greedy (oracle)')
    for name, res in results.items():
        if name == 'greedy(oracle)':
            continue
        ax[0].axhline(res['score'], ls='--', lw=1, label=name)
    ax[0].set_xlabel('sampled lines')
    ax[0].set_ylabel('greedy objective (normalised)')
    ax[0].set_title('1D greedy line selection')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    if history_2d is not None:
        ax[1].semilogy(history_2d['budgets'], history_2d['history'], label='greedy 2D (residual)')
        ax[1].axhline(results['greedy(oracle)']['score'], ls='--', color='k', lw=1,
                      label='greedy 1D at full budget')
        ax[1].set_xlabel('sampled k-space points')
        ax[1].set_ylabel('NMSE')
        ax[1].set_title('2D greedy (single slice)')
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
    else:
        ax[1].axis('off')

    fig.tight_layout()
    fig.savefig(out_dir / 'curves.png', dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slices, file_name = load_slices(args)
    device = torch.device(args.device)
    W = args.resolution
    budget = int(round(W / args.acceleration))
    print(f'{args.anatomy} {file_name}: {len(slices)} slices, W={W}, '
          f'budget={budget} lines (R={args.acceleration:g})', flush=True)

    make = lambda s: LineOracle(s['kspace'], s['maps'], lam=args.lam,
                                noise_level=args.noise_level, support=s['support'])
    search = [make(s) for s in slices[:args.num_slices]]
    held_out = [make(s) for s in slices[args.num_slices:]]
    train_oracle = MeanLineOracle(search)
    test_oracle = MeanLineOracle(held_out) if held_out else None

    print(f'greedy line search, objective={args.objective}', flush=True)
    greedy = greedy_lines(train_oracle, budget, objective=args.objective,
                          preselect=args.preselect, chunk=args.chunk)

    masks = {'greedy(oracle)': greedy['mask']}
    masks.update(baseline_masks(W, budget, args.preselect, device, args.seed))

    results = {}
    for name, mask in masks.items():
        entry = {
            'mask': mask.cpu().numpy().astype(int).tolist(),
            'num_lines': int(mask.sum().item()),
            'train_mse': float(train_oracle.score(mask.unsqueeze(0), 'mse').item()),
            'train_mse_noise': float(train_oracle.score(mask.unsqueeze(0), 'mse_noise').item()),
            'train_trace': float(train_oracle.score(mask.unsqueeze(0), 'trace').item()),
        }
        if test_oracle is not None:
            entry['heldout_mse'] = float(test_oracle.score(mask.unsqueeze(0), 'mse').item())
            entry['heldout_mse_noise'] = float(test_oracle.score(mask.unsqueeze(0), 'mse_noise').item())
        results[name] = entry
        print(f"  {name:16s} lines={entry['num_lines']:3d} "
              f"noiseless={entry['train_mse']:.5g} "
              f"noisy={entry['train_mse_noise']:.5g} "
              f"heldout_noisy={entry.get('heldout_mse_noise', float('nan')):.5g}", flush=True)

    history_2d = None
    if not args.skip_2d:
        print('2D batched greedy on the first search slice', flush=True)
        point_budget = budget * args.resolution
        res2d = greedy_2d(slices[0]['kspace'], slices[0]['maps'], budget=point_budget,
                          batch=args.greedy_2d_batch * args.resolution // 64,
                          preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
                          support=slices[0]['support'])
        step = point_budget // max(1, len(res2d['history']) - 1)
        history_2d = {
            'history': res2d['history'],
            'budgets': [min(point_budget, args.preselect ** 2 + i * step)
                        for i in range(len(res2d['history']))],
        }
        np.save(args.out_dir / 'mask_2d.npy', res2d['mask'].cpu().numpy())
        single = make(slices[0])
        results['greedy_2d(slice0)'] = {
            'num_points': int(res2d['mask'].sum().item()),
            'slice0_mse': res2d['history'][-1],
            'slice0_mse_greedy_1d': float(single.score(greedy['mask'].unsqueeze(0), 'mse').item()),
        }
        print(f"  2D nmse={res2d['history'][-1]:.6g} vs 1D greedy "
              f"{results['greedy_2d(slice0)']['slice0_mse_greedy_1d']:.6g}", flush=True)

    key = 'train_mse_noise' if args.objective == 'mse_noise' else 'train_mse'
    plot_masks(args.out_dir, {k: {'mask': np.array(v['mask']), 'score': v[key]}
                              for k, v in results.items() if 'mask' in v}, budget, W, key)
    plot_curves(args.out_dir, greedy,
                {k: {**v, 'score': v[key]} for k, v in results.items() if 'mask' in v},
                history_2d, budget)

    payload = {
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'file_name': file_name,
        'budget_lines': budget,
        'greedy_order': greedy['order'],
        'greedy_history': greedy['history'],
        'results': results,
        'history_2d': history_2d,
    }
    (args.out_dir / 'results.json').write_text(json.dumps(payload, indent=2))
    np.save(args.out_dir / 'mask_1d.npy', greedy['mask'].cpu().numpy())
    print(f'wrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
