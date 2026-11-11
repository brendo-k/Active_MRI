"""Tables for the mask-swap scores of brain_static_vs_loupe_swap.py: is a gap the mask, the VarNet, or their pairing?

P[r, d] is recipient VarNet r's per-slice PSNR with donor run d's brain mask (mean of the SEEDS draws). Any two runs
r and d with all four of P[r,r], P[r,d], P[d,r], P[d,d] split the paired gap P[r,r] - P[d,d] exactly into
  mask effect     1/2 [(P[r,r] - P[r,d]) + (P[d,r] - P[d,d])]   r's mask minus d's, averaged over both VarNets
  VarNet effect   1/2 [(P[r,r] - P[d,r]) + (P[r,d] - P[d,d])]   r's VarNet minus d's, averaged over both masks
and leave
  co-adaptation   1/2 [(P[r,r] - P[r,d]) + (P[d,d] - P[d,r])]   how much each VarNet prefers its own mask
Each comes with a 95% volume-bootstrap CI over slices. Pairs: the three arms of each seed, and the three
loupe_uniform runs with each other (same smooth density, different leftover start noise).

With --split train as well, the own-mask PSNR on the quarter of the train set each run trained on, and its gap to val.

    python scripts/eval/brain_static_vs_loupe_swap_summary.py --swap reports/brain_static_vs_loupe/swap
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from brain_static_vs_loupe_eval import ARMS
from eval_adaptive_samplers_r8 import bootstrap

SEEDS = (42, 43, 44)
ARM_PAIRS = (('loupe_shared', 'static'), ('loupe_uniform', 'static'), ('loupe_shared', 'loupe_uniform'))


def load_split(folder, split):
    psnr, volumes = {}, None
    for path in sorted(folder.glob(f'swap_{split}_*.pt')):
        saved = torch.load(path, weights_only=False)
        volumes = np.array(saved['volume']) if volumes is None else volumes
        assert list(volumes) == list(saved['volume']), f'{path} scored different slices'
        psnr[saved['recipient']] = {d: s['psnr'].double().numpy() for d, s in saved['scores'].items()}
    return psnr, volumes


def decompose(psnr, r, d, volumes, rng):
    p = lambda a, b: psnr[a][b]  # noqa: E731
    parts = {'paired gap': p(r, r) - p(d, d),
             'mask effect': 0.5 * ((p(r, r) - p(r, d)) + (p(d, r) - p(d, d))),
             'VarNet effect': 0.5 * ((p(r, r) - p(d, r)) + (p(r, d) - p(d, d))),
             'co-adaptation': 0.5 * ((p(r, r) - p(r, d)) + (p(d, d) - p(d, r)))}
    return {name: bootstrap(delta, volumes, rng) for name, delta in parts.items()}


def over_seeds(values):
    values = np.array(values)
    sd = values.std(ddof=1)
    return {'by_seed': values.tolist(), 'mean': float(values.mean()),
            't': float(values.mean() / (sd / np.sqrt(len(values)))) if sd > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--swap', type=Path, required=True, help='folder of swap_<split>_<run>.pt files')
    args = parser.parse_args()
    rng = np.random.RandomState(0)
    summary = {}
    for split in ('val', 'train'):
        psnr, volumes = load_split(args.swap, split)
        if not psnr:
            continue
        matrix = {r: {d: float(v.mean()) for d, v in by_donor.items()} for r, by_donor in psnr.items()}
        result = {'n_slices': int(len(volumes)), 'n_volumes': int(len(np.unique(volumes))), 'mean_psnr': matrix,
                  'arm_pairs': {}, 'uniform_pairs': {}}
        print(f'\n=== {split}: {len(volumes)} slices, {len(np.unique(volumes))} volumes ===')
        for seed in SEEDS:
            names = [f'{arm}_s{seed}' for arm in ARMS]
            if not all(n in psnr for n in names):
                continue
            print(f'seed {seed}: rows = VarNet, columns = mask ({", ".join(ARMS)})')
            for r in names:
                print(f'  {r:20s}' + ''.join(f'{matrix[r][d]:9.3f}' for d in names))
        for a, b in ARM_PAIRS:
            per_seed = {s: decompose(psnr, f'{a}_s{s}', f'{b}_s{s}', volumes, rng) for s in SEEDS
                        if f'{a}_s{s}' in psnr and f'{b}_s{s}' in psnr}
            result['arm_pairs'][f'{a}-{b}'] = {
                'by_seed': {str(s): parts for s, parts in per_seed.items()},
                **{name: over_seeds([parts[name]['mean'] for parts in per_seed.values()])
                   for name in next(iter(per_seed.values()))}}
        if all(f'loupe_uniform_s{s}' in psnr and f'loupe_uniform_s{t}' in psnr[f'loupe_uniform_s{s}']
               for s in SEEDS for t in SEEDS):
            for s, t in itertools.combinations(SEEDS, 2):
                result['uniform_pairs'][f's{s}-s{t}'] = decompose(
                    psnr, f'loupe_uniform_s{s}', f'loupe_uniform_s{t}', volumes, rng)
            uniform = [f'loupe_uniform_s{s}' for s in SEEDS]
            result['other_uniform_mask_minus_own_seed_uniform_mask'] = {
                r: float(np.mean([matrix[r][u] for u in uniform if u.rsplit('_s', 1)[1] != r.rsplit('_s', 1)[1]])
                         - matrix[r][f'loupe_uniform_s{r.rsplit("_s", 1)[1]}']) for r in psnr}
        print('\npaired gap = mask effect + VarNet effect; co-adaptation = own-mask preference (dB, mean [95% CI])')
        for name, entry in list(result['arm_pairs'].items()) + list(result['uniform_pairs'].items()):
            rows = entry['by_seed'].items() if 'by_seed' in entry else [('-', entry)]
            for seed, parts in rows:
                cells = '  '.join(f'{k} {v["mean"]:+.3f} [{v["ci95"][0]:+.2f},{v["ci95"][1]:+.2f}]' for k, v in parts.items())
                print(f'  {name:28s} s{seed}: {cells}')
            if 'by_seed' in entry:
                print(f'  {name:28s} mean: ' + '  '.join(
                    f'{k} {entry[k]["mean"]:+.3f} (t {entry[k]["t"]:+.1f})' for k in parts))
        summary[split] = result
    if 'train' in summary and 'val' in summary:
        own = {s: {r: summary[s]['mean_psnr'][r][r] for r in summary[s]['mean_psnr']} for s in ('train', 'val')}
        summary['train_minus_val_own_mask'] = {r: own['train'][r] - own['val'][r] for r in own['train'] if r in own['val']}
        print('\nown-mask PSNR, train quarter minus full val:',
              {r: round(v, 3) for r, v in summary['train_minus_val_own_mask'].items()})
    (args.swap / 'swap_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
