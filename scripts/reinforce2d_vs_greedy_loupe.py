"""Compare batched macro-round REINFORCE, LOUPE-style, and greedy on 2D k-space masks.

Extends ``loupe_vs_greedy_oracle.py`` (1D lines) to arbitrary 2D k-space
points, where there is no closed-form oracle: every mask's score is an actual
CG-SENSE solve (:func:`activemri.samplers.oracle._cg_solve_batch`), the same
reconstructor :func:`greedy_2d` already uses. Three search strategies are
compared, all routed through that same solver so the only thing that differs
is how each one decides where to sample:

  - greedy_2d:            deterministic top-batch residual selection.
  - loupe_style_design_2d: differentiable gradient descent on a soft 2D
                           sampling density (LOUPE's mask in the limit of a
                           perfect reconstructor and infinite sigmoid slope).
  - reinforce_2d:          batched macro-round REINFORCE -- the same round
                           structure as greedy_2d, but each round's batch is
                           a learned stochastic Plackett-Luce draw instead of
                           a deterministic argmax.
  - reinforce_2d+residual-init: the same policy, with its logits initialised
                           from the k-space residual against the preselected
                           mask alone (the same quantity greedy_2d's own
                           first round ranks by) instead of zero, so training
                           starts biased toward greedy's own choice function
                           rather than uniform.

Because a single 2D reward already costs a CG solve, this script works on one
search slice and one held-out slice by default (num-slices/held-out can raise
that, at proportionally higher cost) -- the 2D analogue of what
``oracle_sampling_search.py``'s "2D batched greedy on the first search slice"
already does.

    python scripts/reinforce2d_vs_greedy_loupe.py --anatomy brain \\
        --resolution 48 --acceleration 4 --out-dir runs/oracle/reinforce2d_brain_R4
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

from activemri.datasets.fastmri_data import FastMRIDataset, _resolve_anatomy_dirs
from activemri.samplers.oracle import (
    _cg_solve_batch, estimate_sense_maps, greedy_2d, kspace_residual_scores, loupe_style_design_2d,
    reinforce_2d,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', default='brain', choices=['brain', 'knee', 'breast', 'prostate'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--num-slices', type=int, default=1, help='search slices (2D reward is expensive)')
    parser.add_argument('--held-out-slices', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=48)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--acceleration', type=float, default=4.0)
    parser.add_argument('--preselect', type=int, default=6, help='side length of the fixed centre block')
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--acs', type=int, default=16)
    parser.add_argument('--cg-iters', type=int, default=10)

    parser.add_argument('--greedy-batch', type=int, default=None,
                        help='defaults to budget // 6 (~6 rounds)')

    parser.add_argument('--loupe-steps', type=int, default=100)
    parser.add_argument('--loupe-lr', type=float, default=0.5)

    parser.add_argument('--reinforce-batch', type=int, default=None,
                        help='defaults to budget // 6 (~6 rounds), matching greedy')
    parser.add_argument('--reinforce-episodes-per-update', type=int, default=6)
    parser.add_argument('--reinforce-updates', type=int, default=25)
    parser.add_argument('--reinforce-lr', type=float, default=0.2)
    parser.add_argument('--reinforce-init-temperature', type=float, default=2.0,
                        help='scale of the residual-score prior fed to reinforce_2d+residual-init')

    parser.add_argument('--skip-loupe', action='store_true',
                        help='skip both loupe_style_design_2d runs (already validated); '
                             'saves that compute for more reinforce_2d training')
    parser.add_argument('--skip-residual-init', action='store_true',
                        help='skip the reinforce_2d+residual-init run (already found not to help)')

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


def baseline_masks(H, W, budget, preselect, device, seed):
    masks = {}
    n = preselect // 2
    center = torch.zeros(H, W, device=device)
    center[H // 2 - n: H // 2 + n, W // 2 - n: W // 2 + n] = 1.0

    masks['low_pass'] = torch.zeros(H, W, device=device)
    n2 = int(round((budget ** 0.5) / 2))
    masks['low_pass'][H // 2 - n2: H // 2 + n2, W // 2 - n2: W // 2 + n2] = 1.0

    generator = torch.Generator(device='cpu').manual_seed(seed)
    rnd = center.clone()
    free = (rnd.view(-1) == 0).nonzero(as_tuple=False).flatten()
    remaining = budget - int(rnd.sum().item())
    rnd.view(-1)[free[torch.randperm(free.numel(), generator=generator)[:remaining]]] = 1.0
    masks['random+acs'] = rnd
    return masks


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
    greedy_batch = args.greedy_batch or max(1, budget // 6)
    reinforce_batch = args.reinforce_batch or max(1, budget // 6)
    print(f'{args.anatomy} {file_name}: H=W={H}, budget={budget} points (R={args.acceleration:g})',
          flush=True)

    search = slices[0]
    held_out = slices[args.num_slices] if args.held_out_slices > 0 else None

    print('\n=== greedy_2d ===', flush=True)
    greedy = greedy_2d(search['kspace'], search['maps'], budget=budget, batch=greedy_batch,
                       preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
                       support=search['support'], verbose=verbose)

    loupe = loupe_warm = None
    if not args.skip_loupe:
        print('\n=== loupe_style_design_2d: from scratch ===', flush=True)
        loupe = loupe_style_design_2d(search['kspace'], search['maps'], budget=budget,
                                      cg_iters=args.cg_iters, lam=args.lam, num_steps=args.loupe_steps,
                                      lr=args.loupe_lr, support=search['support'], verbose=verbose)

        print('\n=== loupe_style_design_2d: warm-started from greedy_2d ===', flush=True)
        loupe_warm = loupe_style_design_2d(search['kspace'], search['maps'], budget=budget,
                                           cg_iters=args.cg_iters, lam=args.lam, num_steps=args.loupe_steps,
                                           lr=args.loupe_lr, init=greedy['mask'].clone(),
                                           support=search['support'], verbose=verbose, log_prefix='warm ')
    else:
        print('\n=== skipping loupe_style_design_2d (--skip-loupe) ===', flush=True)

    print('\n=== reinforce_2d: batched macro-round policy gradient (zero init) ===', flush=True)
    reinforce = reinforce_2d(search['kspace'], search['maps'], budget=budget, batch=reinforce_batch,
                             preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
                             episodes_per_update=args.reinforce_episodes_per_update,
                             num_updates=args.reinforce_updates, lr=args.reinforce_lr,
                             support=search['support'], verbose=verbose)

    reinforce_init = None
    if not args.skip_residual_init:
        print('\n=== residual score field (prior for reinforce_2d+residual-init) ===', flush=True)
        n = args.preselect // 2
        preselect_mask = torch.zeros(H, W, device=device)
        preselect_mask[H // 2 - n: H // 2 + n, W // 2 - n: W // 2 + n] = 1.0
        with torch.no_grad():
            preselect_recon = _cg_solve_batch(search['kspace'], search['maps'],
                                              preselect_mask.unsqueeze(0), args.cg_iters, args.lam)[0]
            residual_prior = kspace_residual_scores(search['kspace'], search['maps'], preselect_mask,
                                                    preselect_recon)
        print(f'  residual_prior: min={residual_prior.min().item():.4g} max={residual_prior.max().item():.4g} '
              f'std={residual_prior.std().item():.4g}', flush=True)

        print('\n=== reinforce_2d: batched macro-round policy gradient (residual-init) ===', flush=True)
        reinforce_init = reinforce_2d(search['kspace'], search['maps'], budget=budget, batch=reinforce_batch,
                                      preselect=args.preselect, cg_iters=args.cg_iters, lam=args.lam,
                                      episodes_per_update=args.reinforce_episodes_per_update,
                                      num_updates=args.reinforce_updates, lr=args.reinforce_lr,
                                      init_logits=residual_prior, init_temperature=args.reinforce_init_temperature,
                                      support=search['support'], verbose=verbose, log_prefix='rinit ')
    else:
        print('\n=== skipping reinforce_2d+residual-init (--skip-residual-init) ===', flush=True)

    masks = {'greedy_2d': greedy['mask'], 'reinforce_2d': reinforce['mask']}
    if loupe is not None:
        masks['loupe_2d(oracle)'] = loupe['mask']
        masks['loupe_2d+greedy-init'] = loupe_warm['mask']
    if reinforce_init is not None:
        masks['reinforce_2d+residual-init'] = reinforce_init['mask']
    masks.update(baseline_masks(H, W, budget, args.preselect, device, args.seed))

    results = {}
    for name, mask in masks.items():
        entry = {'num_points': int(mask.sum().item()),
                 'search_nmse': nmse_on(search['kspace'], search['maps'], search['support'], mask)}
        if held_out is not None:
            entry['heldout_nmse'] = nmse_on(held_out['kspace'], held_out['maps'], held_out['support'], mask)
        results[name] = entry

    greedy_search = results['greedy_2d']['search_nmse']
    greedy_heldout = results['greedy_2d'].get('heldout_nmse')

    print(f'\n{"method":<22s} {"points":>7s} {"search":>12s} {"gap-to-greedy":>14s} {"heldout":>12s} {"gap-to-greedy":>14s}',
          flush=True)
    for name, entry in results.items():
        gap = (entry['search_nmse'] - greedy_search) / abs(greedy_search)
        line = f'{name:<22s} {entry["num_points"]:7d} {entry["search_nmse"]:12.6g} {gap:+13.2%} '
        if held_out is not None:
            hgap = (entry['heldout_nmse'] - greedy_heldout) / abs(greedy_heldout)
            line += f'{entry["heldout_nmse"]:12.6g} {hgap:+13.2%}'
        print(line, flush=True)

    # --- plots ---
    order = [name for name in
             ['greedy_2d', 'loupe_2d(oracle)', 'loupe_2d+greedy-init', 'reinforce_2d',
              'reinforce_2d+residual-init', 'low_pass', 'random+acs']
             if name in masks]
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.6))
    for ax, name in zip(axes, order):
        ax.imshow(masks[name].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{name}\nnmse={results[name]['search_nmse']:.4f}", fontsize=8)
        ax.axis('off')
    fig.suptitle(f'2D masks at {budget} points ({args.anatomy} R={args.acceleration:g})', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'masks_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].semilogy(greedy['history'], marker='o', label='greedy_2d (per round)')
    for name in order:
        if name == 'greedy_2d':
            continue
        ax[0].axhline(results[name]['search_nmse'], ls='--', lw=1, label=name)
    ax[0].set_xlabel('round')
    ax[0].set_ylabel('nmse')
    ax[0].set_title('greedy_2d trajectory vs. learned/baseline masks')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(reinforce['history'], label='reinforce_2d, zero init')
    if reinforce_init is not None:
        ax[1].plot(reinforce_init['history'], label='reinforce_2d, residual init')
    ax[1].axhline(greedy_search, ls='--', color='k', lw=1, label='greedy_2d')
    ax[1].set_xlabel('policy-gradient update')
    ax[1].set_ylabel('nmse')
    ax[1].set_title('reinforce_2d learning curves')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'curves_2d.png', dpi=150)
    plt.close(fig)

    payload = {
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'file_name': file_name,
        'budget_points': budget,
        'greedy_batch': greedy_batch,
        'reinforce_batch': reinforce_batch,
        'greedy_history': greedy['history'],
        'reinforce_history': reinforce['history'],
        'reinforce_residual_init_history': reinforce_init['history'] if reinforce_init is not None else None,
        'results': results,
    }
    (args.out_dir / 'results.json').write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
