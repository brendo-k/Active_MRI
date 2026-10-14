"""Check how far greedy line selection is from the true optimum.

Exhaustive search over line masks is combinatorial: C(W_free, k_free) grows
far too fast to run at the resolutions ``oracle_sampling_search.py`` actually
uses (e.g. C(120, 24) at W=128 is ~1e25). But at a small W it's cheap, since
:class:`~activemri.samplers.oracle.LineOracle` scores every candidate with one
batched Cholesky solve rather than an iterative reconstruction.

This script:

1. Shrinks the problem to a small W (e.g. 20) and brute-forces every mask,
   giving the *true* optimum for whatever objective ``LineOracle`` supports
   (including the non-convex ``mse`` / ``mse_noise``).
2. Runs ``greedy_lines`` on the same small problem and reports the gap to
   that true optimum.
3. Runs ``relaxed_design`` (convex relaxation, exact for ``trace``/``logdet``
   only) on the small problem too, so its bound can itself be checked against
   the true optimum -- a sanity check on the certificate, not just on greedy.
4. Runs ``relaxed_design`` at full resolution (W=128 by default), where
   exhaustive search is impossible, and reports the certified
   (relaxed - greedy) / relaxed gap as the only feasible optimality check.

    python scripts/oracle/greedy_vs_exhaustive.py --anatomy brain \\
        --small-resolution 20 --small-budget 6 --small-preselect 0 \\
        --full-resolution 128 --full-budget 32 --full-preselect 8
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import _path  # noqa: F401
import torch

from activemri.datasets.fastmri_data import FastMRIDataset, _resolve_anatomy_dirs
from activemri.samplers.oracle import (
    LineOracle, MeanLineOracle, center_lines, estimate_sense_maps, greedy_lines, relaxed_design, swap_refine, icd_refine,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', default='brain', choices=['brain', 'knee', 'breast', 'prostate'])
    parser.add_argument('--contrast', default='T2')
    parser.add_argument('--num-slices', type=int, default=1,
                        help='slices averaged into the objective (the population oracle)')

    parser.add_argument('--small-resolution', type=int, default=20)
    parser.add_argument('--small-budget', type=int, default=6)
    parser.add_argument('--small-preselect', type=int, default=0)

    parser.add_argument('--full-resolution', type=int, default=128)
    parser.add_argument('--full-budget', type=int, default=32)
    parser.add_argument('--full-preselect', type=int, default=8)

    parser.add_argument('--objective', default='mse_noise',
                        choices=['mse_noise', 'mse', 'trace', 'logdet'],
                        help='objective compared exhaustively vs. greedily at small resolution')
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--noise-level', type=float, default=0.02)
    parser.add_argument('--acs', type=int, default=8, help='centre block used to estimate maps')
    parser.add_argument('--num-coils', type=int, default=16)

    parser.add_argument('--eval-batch', type=int, default=4096,
                        help='candidate masks scored per batched solve during exhaustive search')
    parser.add_argument('--max-combos', type=int, default=2_000_000,
                        help='safety cap on C(W_free, k_free); pass --force to override')
    parser.add_argument('--force', action='store_true')

    parser.add_argument('--relaxed-steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def load_oracle(args, resolution):
    """A MeanLineOracle over the first args.num_slices slices at this resolution."""
    train_dir, _ = _resolve_anatomy_dirs(args.data_root, args.anatomy, args.contrast)
    res = [resolution, resolution]

    probe = FastMRIDataset(root=train_dir, anatomy=args.anatomy, image_shape=res,
                           num_coils=args.num_coils, min_energy_fraction=0.0, slice_ids=[0])
    file_name = probe.examples[0][0].name

    dataset = FastMRIDataset(
        root=train_dir, anatomy=args.anatomy, image_shape=res, num_coils=args.num_coils,
        min_energy_fraction=0.0, file_names=[file_name], slice_ids=list(range(args.num_slices)),
    )

    device = torch.device(args.device)
    oracles = []
    for i in range(len(dataset)):
        sample = dataset[i]
        kspace = torch.view_as_complex(sample['kspace']).to(device).to(torch.complex64)
        maps, support = estimate_sense_maps(kspace, acs=args.acs)
        oracles.append(LineOracle(kspace, maps, lam=args.lam, noise_level=args.noise_level,
                                  support=support))
    return MeanLineOracle(oracles) if len(oracles) > 1 else oracles[0], file_name


def exhaustive_lines(oracle, budget, preselect, objective, eval_batch, max_combos, force,
                     device):
    """True optimum over every line mask at this budget, by brute force.

    Feasible only because W is small here: every candidate is scored with one
    batched Cholesky solve (see :class:`LineOracle`), not an iterative solve,
    so millions of masks can be checked per second.

    Returns:
        ``mask``, ``score``, and ``num_combos`` actually enumerated.
    """
    W = oracle.W
    base = center_lines(W, preselect, device)
    free = (base == 0).nonzero(as_tuple=False).flatten().tolist()
    k_free = budget - preselect
    if k_free < 0:
        raise ValueError('budget must be >= preselect')
    if k_free == 0:
        score = oracle.score(base.unsqueeze(0), objective).item()
        return base, score, 1

    num_combos = math.comb(len(free), k_free)
    if num_combos > max_combos and not force:
        raise ValueError(
            f'C({len(free)}, {k_free}) = {num_combos:,} exceeds --max-combos={max_combos:,}; '
            'shrink --small-resolution/--small-budget or pass --force')

    best_score, best_mask = float('inf'), None
    combos = itertools.combinations(free, k_free)
    buffer = []
    seen = 0

    def flush(buffer):
        nonlocal best_score, best_mask
        if not buffer:
            return
        masks = base.unsqueeze(0).repeat(len(buffer), 1)
        for row, combo in enumerate(buffer):
            masks[row, list(combo)] = 1.0
        scores = oracle.score_chunked(masks, objective, eval_batch)
        idx = int(torch.argmin(scores).item())
        if scores[idx].item() < best_score:
            best_score = scores[idx].item()
            best_mask = masks[idx].clone()

    for combo in combos:
        buffer.append(combo)
        seen += 1
        if len(buffer) == eval_batch:
            flush(buffer)
            buffer = []
    flush(buffer)

    return best_mask, best_score, seen


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- small W: exhaustive vs. greedy vs. relaxed-design bound ---
    print(f'=== small problem: W={args.small_resolution}, budget={args.small_budget}, '
          f'preselect={args.small_preselect} ===', flush=True)
    small_oracle, small_file = load_oracle(args, args.small_resolution)

    free = args.small_resolution - args.small_preselect
    k_free = args.small_budget - args.small_preselect
    num_combos = math.comb(free, k_free)
    print(f'{small_file}: C({free}, {k_free}) = {num_combos:,} candidate masks', flush=True)

    exact_mask, exact_score, num_scored = exhaustive_lines(
        small_oracle, args.small_budget, args.small_preselect, args.objective,
        args.eval_batch, args.max_combos, args.force, device)
    print(f'exhaustive: {args.objective}={exact_score:.6g}  ({num_scored:,} masks scored)',
          flush=True)

    greedy_small = greedy_lines(small_oracle, args.small_budget, objective=args.objective,
                               preselect=args.small_preselect, chunk=args.eval_batch,
                               verbose=False)
    greedy_score = greedy_small['history'][-1]
    greedy_gap = (greedy_score - exact_score) / abs(exact_score) if exact_score else float('nan')
    print(f'greedy:     {args.objective}={greedy_score:.6g}  '
          f'(gap to true optimum: {greedy_gap:+.4%})', flush=True)

    refined_small = swap_refine(small_oracle, greedy_small['mask'], objective=args.objective,
                                preselect=args.small_preselect, chunk=args.eval_batch, verbose=False)
    refined_score = refined_small['history'][-1]
    refined_gap = (refined_score - exact_score) / abs(exact_score) if exact_score else float('nan')
    print(f'greedy+swap: {args.objective}={refined_score:.6g}  '
          f'(gap to true optimum: {refined_gap:+.4%}, {refined_small["num_swaps"]} swaps)', flush=True)

    icd_small = icd_refine(small_oracle, greedy_small['mask'], objective=args.objective,
                           preselect=args.small_preselect, chunk=args.eval_batch, verbose=False)
    icd_score = icd_small['history'][-1]
    icd_gap = (icd_score - exact_score) / abs(exact_score) if exact_score else float('nan')
    print(f'greedy+icd:  {args.objective}={icd_score:.6g}  '
          f'(gap to true optimum: {icd_gap:+.4%}, {icd_small["num_moves"]} moves)', flush=True)

    small_relaxed = {}
    if args.objective in ('trace', 'logdet'):
        relaxed = relaxed_design(small_oracle, args.small_budget, objective=args.objective,
                                 num_steps=args.relaxed_steps, verbose=False)
        relaxed_gap = (exact_score - relaxed['value']) / abs(relaxed['value'])
        print(f'relaxed:    {args.objective}={relaxed["value"]:.6g}  '
              f'(bound vs. true optimum: {relaxed_gap:+.4%}, should be >= 0)', flush=True)
        small_relaxed = {'value': relaxed['value'], 'gap_to_true_optimum': relaxed_gap}
    else:
        print(f"(relaxed_design not applicable to '{args.objective}'; only certifies "
              "'trace'/'logdet' -- see relaxed_design's docstring)", flush=True)

    # --- full W: relaxed-design bound only (exhaustive is infeasible here) ---
    print(f'\n=== full problem: W={args.full_resolution}, budget={args.full_budget}, '
          f'preselect={args.full_preselect} ===', flush=True)
    full_oracle, full_file = load_oracle(args, args.full_resolution)

    full_free = args.full_resolution - args.full_preselect
    full_k_free = args.full_budget - args.full_preselect
    full_combos = math.comb(full_free, full_k_free)
    print(f'{full_file}: C({full_free}, {full_k_free}) = {full_combos:.3e} candidate masks '
          '-- exhaustive search is infeasible here', flush=True)

    full_objective = args.objective if args.objective in ('trace', 'logdet') else 'logdet'
    greedy_full = greedy_lines(full_oracle, args.full_budget, objective=full_objective,
                              preselect=args.full_preselect, chunk=args.eval_batch,
                              verbose=False)
    greedy_full_score = greedy_full['history'][-1]
    relaxed_full = relaxed_design(full_oracle, args.full_budget, objective=full_objective,
                                  num_steps=args.relaxed_steps, verbose=False)
    full_gap = (greedy_full_score - relaxed_full['value']) / abs(relaxed_full['value'])
    print(f'greedy:  {full_objective}={greedy_full_score:.6g}', flush=True)
    print(f'relaxed: {full_objective}={relaxed_full["value"]:.6g}  '
          f'(certified gap: greedy is within {full_gap:+.4%} of the relaxed upper bound)',
          flush=True)

    payload = {
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'small': {
            'file_name': small_file,
            'num_combos': num_combos,
            'exact_score': exact_score,
            'greedy_score': greedy_score,
            'greedy_gap_to_true_optimum': greedy_gap,
            'swap_refined_score': refined_score,
            'swap_refined_gap_to_true_optimum': refined_gap,
            'num_swaps': refined_small['num_swaps'],
            'icd_refined_score': icd_score,
            'icd_refined_gap_to_true_optimum': icd_gap,
            'num_icd_moves': icd_small['num_moves'],
            'relaxed': small_relaxed,
        },
        'full': {
            'file_name': full_file,
            'num_combos': full_combos,
            'objective': full_objective,
            'greedy_score': greedy_full_score,
            'relaxed_value': relaxed_full['value'],
            'certified_gap_greedy_vs_relaxed': full_gap,
        },
    }
    (args.out_dir / 'greedy_vs_exhaustive.json').write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {args.out_dir / "greedy_vs_exhaustive.json"}', flush=True)


if __name__ == '__main__':
    main()
