"""Quantify the gap between LOUPE-style learning, RL, and greedy line search.

``oracle_sampling_search.py`` already shows how far greedy line selection gets
above equispaced/random baselines. What it doesn't show is how close a *learned*
sampler gets to greedy -- which is the question that actually matters, since
greedy needs the full population of slices and a differentiable oracle up
front, while a trained sampler only needs a forward pass at test time.

This script controls for every confound between "LOUPE can't find a good mask"
and "the neural reconstructor's gradient is too weak to point at one" by
routing every method through the exact same :class:`LineOracle` that
``oracle_sampling_search.py`` uses -- no UNet/VarNet in the loop:

  - greedy(oracle):    the incumbent, from :func:`greedy_lines`.
  - loupe(oracle):     :func:`loupe_style_design` -- the sigmoid-relaxed,
                        gradient-descent mask LOUPE converges to in the limit
                        of a perfect reconstructor and infinite slope.
  - loupe+greedy-init: the same local search, warm-started from the greedy
                        mask, to see whether gradient descent can at least
                        *hold* the greedy optimum once handed it.
  - reinforce(oracle): :func:`reinforce_lines` -- a policy-gradient learner
                        over the same hard combinatorial action space DDQN-style
                        RL baselines use, but trained directly against oracle
                        reward instead of a step-by-step CG solve.
  - equispaced / random / low_pass: the usual non-learned references.

Every method is scored on both the slices it was fit to (or, for greedy,
searched over) and a held-out set, so the comparison also shows how much each
method overfits its search population.

    python scripts/loupe_vs_greedy_oracle.py --anatomy brain --num-slices 8 \\
        --acceleration 4 --out-dir runs/oracle/loupe_vs_greedy_brain_R4
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
    LineOracle, MeanLineOracle, center_lines, estimate_sense_maps, greedy_lines,
    loupe_style_design, reinforce_lines,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', default='brain', choices=['brain', 'knee', 'breast', 'prostate'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--num-slices', type=int, default=8)
    parser.add_argument('--held-out-slices', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--acceleration', type=float, default=4.0)
    parser.add_argument('--preselect', type=int, default=8)
    parser.add_argument('--objective', default='mse_noise', choices=['mse_noise', 'mse'])
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--noise-level', type=float, default=0.02)
    parser.add_argument('--acs', type=int, default=24)
    parser.add_argument('--chunk', type=int, default=32)

    parser.add_argument('--loupe-steps', type=int, default=400)
    parser.add_argument('--loupe-lr', type=float, default=0.5)
    parser.add_argument('--reinforce-episodes', type=int, default=400)
    parser.add_argument('--reinforce-batch', type=int, default=64)
    parser.add_argument('--reinforce-lr', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def load_slices(args):
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
        maps, support = estimate_sense_maps(kspace, acs=args.acs)
        slices.append({'kspace': kspace, 'maps': maps, 'support': support})
    return slices, file_name


def baseline_masks(W, budget, preselect, device, seed):
    masks = {}
    center = center_lines(W, preselect, device)

    equispaced = center.clone()
    free = (equispaced == 0).nonzero(as_tuple=False).flatten()
    remaining = budget - int(equispaced.sum().item())
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


def score_all(oracle, mask, objective):
    return {
        'mse': float(oracle.score(mask.unsqueeze(0), 'mse').item()),
        'mse_noise': float(oracle.score(mask.unsqueeze(0), 'mse_noise').item()),
        f'{objective}': float(oracle.score(mask.unsqueeze(0), objective).item()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    slices, file_name = load_slices(args)
    device = torch.device(args.device)
    W = args.resolution
    budget = int(round(W / args.acceleration))
    print(f'{args.anatomy} {file_name}: W={W}, budget={budget} lines (R={args.acceleration:g}), '
          f'objective={args.objective}', flush=True)

    make = lambda s: LineOracle(s['kspace'], s['maps'], lam=args.lam,
                                noise_level=args.noise_level, support=s['support'])
    search = [make(s) for s in slices[:args.num_slices]]
    held_out = [make(s) for s in slices[args.num_slices:]]
    train_oracle = MeanLineOracle(search)
    test_oracle = MeanLineOracle(held_out) if held_out else None

    print('\n=== greedy(oracle) ===', flush=True)
    greedy = greedy_lines(train_oracle, budget, objective=args.objective,
                          preselect=args.preselect, chunk=args.chunk, verbose=verbose)

    print('\n=== loupe(oracle): sigmoid-relaxed gradient descent from scratch ===', flush=True)
    loupe = loupe_style_design(train_oracle, budget, objective=args.objective,
                               num_steps=args.loupe_steps, lr=args.loupe_lr, verbose=verbose)

    print('\n=== loupe+greedy-init: same local search, warm-started from greedy ===', flush=True)
    warm_p = greedy['mask'].clone().to(device)
    loupe_warm = loupe_style_design(train_oracle, budget, objective=args.objective,
                                    num_steps=args.loupe_steps, lr=args.loupe_lr, init=warm_p,
                                    verbose=verbose, log_prefix='warm ')

    print('\n=== reinforce(oracle): Plackett-Luce policy gradient ===', flush=True)
    reinforce = reinforce_lines(train_oracle, budget, objective=args.objective,
                                preselect=args.preselect, num_episodes=args.reinforce_episodes,
                                batch_size=args.reinforce_batch, lr=args.reinforce_lr,
                                chunk=args.chunk, verbose=verbose)

    masks = {
        'greedy(oracle)': greedy['mask'],
        'loupe(oracle)': loupe['mask'],
        'loupe+greedy-init': loupe_warm['mask'],
        'reinforce(oracle)': reinforce['mask'],
    }
    masks.update(baseline_masks(W, budget, args.preselect, device, args.seed))

    results = {}
    for name, mask in masks.items():
        entry = {'mask': mask.cpu().numpy().astype(int).tolist(),
                 'num_lines': int(mask.sum().item()),
                 'train': score_all(train_oracle, mask, args.objective)}
        if test_oracle is not None:
            entry['heldout'] = score_all(test_oracle, mask, args.objective)
        results[name] = entry

    greedy_train = results['greedy(oracle)']['train'][args.objective]
    greedy_heldout = results['greedy(oracle)'].get('heldout', {}).get(args.objective)

    print(f'\n{"method":<20s} {"lines":>5s} {"train":>12s} {"gap-to-greedy":>14s} {"heldout":>12s} {"gap-to-greedy":>14s}',
          flush=True)
    for name, entry in results.items():
        train_score = entry['train'][args.objective]
        train_gap = (train_score - greedy_train) / abs(greedy_train)
        line = f'{name:<20s} {entry["num_lines"]:5d} {train_score:12.6g} {train_gap:+13.2%} '
        if test_oracle is not None:
            heldout_score = entry['heldout'][args.objective]
            heldout_gap = (heldout_score - greedy_heldout) / abs(greedy_heldout)
            line += f'{heldout_score:12.6g} {heldout_gap:+13.2%}'
        print(line, flush=True)

    # --- plots ---
    order = ['greedy(oracle)', 'loupe(oracle)', 'loupe+greedy-init', 'reinforce(oracle)',
             'equispaced+acs', 'random+acs', 'low_pass']
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 1.05 * len(order)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), order):
        mask = np.array(results[name]['mask'])
        ax.imshow(mask[None, :], cmap='gray', aspect='auto', extent=[0, W, 0, 1], vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_ylabel(f"{name}\n{results[name]['train'][args.objective]:.4f}",
                      rotation=0, ha='right', va='center', fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('phase-encode line')
    fig.suptitle(f'1D masks at {budget} lines ({args.objective} on search slices)', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'masks.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    lines = np.arange(budget - len(greedy['history']) + 1, budget + 1)
    ax[0].semilogy(lines, greedy['history'], label='greedy (oracle)')
    for name in ['loupe(oracle)', 'loupe+greedy-init', 'reinforce(oracle)', 'equispaced+acs', 'random+acs', 'low_pass']:
        ax[0].axhline(results[name]['train'][args.objective], ls='--', lw=1, label=name)
    ax[0].set_xlabel('sampled lines')
    ax[0].set_ylabel(f'{args.objective} (normalised)')
    ax[0].set_title('greedy trajectory vs. learned/baseline masks')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(reinforce['history'], label='reinforce (batch mean score)')
    ax[1].axhline(greedy_train, ls='--', color='k', lw=1, label='greedy(oracle)')
    ax[1].set_xlabel('episode')
    ax[1].set_ylabel(f'{args.objective}')
    ax[1].set_title('reinforce learning curve')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'curves.png', dpi=150)
    plt.close(fig)

    payload = {
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'file_name': file_name,
        'budget_lines': budget,
        'greedy_history': greedy['history'],
        'reinforce_history': reinforce['history'],
        'results': results,
    }
    (args.out_dir / 'results.json').write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
