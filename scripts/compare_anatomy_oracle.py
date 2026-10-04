"""Does the CG-SENSE line oracle prefer different masks for brain and knee?

Greedy line selection is run per volume and over a population of volumes for
each anatomy, then judged three ways:

  1. Between- vs within-anatomy distance of the per-volume oracle masks. If
     brain-vs-knee is no larger than brain-vs-brain, there is no anatomy effect.
  2. A transfer matrix on held-out *volumes*: score each anatomy's population
     mask on both anatomies. The regret of the wrong anatomy's mask is the cost
     of ignoring anatomy.
  3. Robustness of the population masks to the oracle's free noise level.

    python scripts/compare_anatomy_oracle.py --device cuda:1 --out-dir runs/oracle/brain_vs_knee
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
    LineOracle, MeanLineOracle, center_lines, estimate_sense_maps, greedy_lines, icd_refine, swap_refine,
)

COLORS = {'brain': '#0072B2', 'knee': '#D55E00'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomies', nargs=2, default=['brain', 'knee'])
    parser.add_argument('--brain-contrast', default='T2')
    parser.add_argument('--volumes', type=int, default=12, help='volumes per anatomy')
    parser.add_argument('--search-volumes', type=int, default=8,
                        help='of those, volumes the population greedy sees; the rest are held out')
    parser.add_argument('--slices-per-volume', type=int, default=3)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--num-coils', type=int, default=16)
    parser.add_argument('--acceleration', type=float, default=4.0)
    parser.add_argument('--preselect', type=int, default=8)
    parser.add_argument('--objective', default='mse_noise', choices=['mse_noise', 'mse', 'trace'])
    parser.add_argument('--noise-level', type=float, default=0.02)
    parser.add_argument('--noise-sweep', type=float, nargs='*', default=[0.005, 0.08])
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--acs', type=int, default=24)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--smooth', type=int, default=5, help='box width for the line-density profile')
    parser.add_argument('--refine', default='none', choices=['none', 'swap', 'icd'],
                        help='local refinement after every greedy search: best single swap per pass, '
                             'or SUNO-style coordinate descent')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def load_volumes(args, anatomy):
    """One list of per-slice records for each of ``args.volumes`` volumes."""
    train_dir, _ = _resolve_anatomy_dirs(args.data_root, anatomy,
                                         args.brain_contrast if anatomy == 'brain' else None)
    res = [args.resolution, args.resolution]
    listing = FastMRIDataset(root=train_dir, anatomy=anatomy, image_shape=res, num_coils=args.num_coils,
                             min_energy_fraction=0.2, num_volumes=args.volumes)
    by_file = {}
    for path, sid in listing.examples:
        by_file.setdefault(path.name, []).append(sid)

    device = torch.device(args.device)
    volumes = []
    for name, valid in by_file.items():
        picks = np.linspace(0, len(valid) - 1, args.slices_per_volume + 2)[1:-1].round().astype(int)
        slice_ids = [valid[i] for i in picks]
        dataset = FastMRIDataset(root=train_dir, anatomy=anatomy, image_shape=res, num_coils=args.num_coils,
                                 min_energy_fraction=0.0, file_names=[name], slice_ids=slice_ids)
        records = []
        for i in range(len(dataset)):
            sample = dataset[i]
            kspace = torch.view_as_complex(sample['kspace']).to(device).to(torch.complex64)
            maps, support = estimate_sense_maps(kspace, acs=args.acs)
            records.append({'kspace': kspace, 'maps': maps, 'support': support})
        volumes.append({'name': name, 'slices': records})
        print(f'  {anatomy} {name}: slices {slice_ids}', flush=True)
    return volumes


def make_oracle(args, record, noise_level):
    return LineOracle(record['kspace'], record['maps'], lam=args.lam,
                      noise_level=noise_level, support=record['support'])


def population_oracle(args, volumes, noise_level):
    return MeanLineOracle([make_oracle(args, r, noise_level) for v in volumes for r in v['slices']])


def greedy_mask(args, oracle, budget):
    mask = greedy_lines(oracle, budget, objective=args.objective, preselect=args.preselect,
                        chunk=args.chunk, verbose=False)['mask']
    if args.refine != 'none':
        refine = {'swap': swap_refine, 'icd': icd_refine}[args.refine]
        mask = refine(oracle, mask, objective=args.objective, preselect=args.preselect,
                      chunk=args.chunk, verbose=False)['mask']
    return mask.cpu().numpy()


def density(mask, width):
    return np.convolve(mask, np.ones(width) / width, mode='same')


def mask_distance(a, b, width, free):
    return float(np.abs(density(a, width) - density(b, width))[free].mean())


def pairwise(masks_a, masks_b, width, free, same):
    dists = [mask_distance(a, b, width, free)
             for i, a in enumerate(masks_a) for j, b in enumerate(masks_b) if not same or i < j]
    return float(np.mean(dists)), float(np.std(dists))


def baselines(W, budget, preselect, device):
    center = center_lines(W, preselect, device)
    equi = center.clone()
    free = (equi == 0).nonzero(as_tuple=False).flatten()
    pick = torch.linspace(0, free.numel() - 1, budget - preselect).round().long()
    equi[free[pick.unique()]] = 1.0
    return {'equispaced+acs': equi.cpu().numpy(), 'low_pass': center_lines(W, budget, device).cpu().numpy()}


def plot_main(args, W, budget, pop, per_volume, transfer, distances, sweep, out_dir):
    a1, a2 = args.anatomies
    x = np.arange(W)
    free = center_lines(W, args.preselect, 'cpu').numpy() == 0

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.2, 1.1, 1.1])

    ax = fig.add_subplot(grid[0, :])
    for name in (a1, a2):
        ax.plot(x, density(pop[name], args.smooth), color=COLORS[name], lw=2, label=f'{name} population oracle')
        freq = np.mean(per_volume[name], axis=0)
        ax.fill_between(x, density(freq, args.smooth), alpha=0.18, color=COLORS[name],
                        label=f'{name} per-volume selection frequency')
    ax.set(xlabel='phase-encode line (0 = edge, W/2 = DC)', ylabel='local line density',
           title=f'Oracle line density, R={args.acceleration:g} ({budget} of {W} lines, '
                 f'{args.preselect} fixed ACS lines)')
    ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(grid[1, 0])
    diff = density(pop[a1], args.smooth) - density(pop[a2], args.smooth)
    ax.bar(x, diff, color=np.where(diff >= 0, COLORS[a1], COLORS[a2]), width=1.0)
    ax.axhline(0, color='k', lw=0.6)
    ax.set(xlabel='phase-encode line', ylabel='density difference',
           title=f'Population masks: {a1} minus {a2} (blue = {a1} wants more)')

    ax = fig.add_subplot(grid[1, 1])
    for name in (a1, a2):
        ax.step(x, np.cumsum(pop[name]), color=COLORS[name], lw=2, label=name)
    ax.set(xlabel='phase-encode line', ylabel='cumulative lines acquired',
           title='Cumulative sampling (a shared curve means a shared mask)')
    ax.legend(fontsize=8)

    for col, name in enumerate((a1, a2)):
        ax = fig.add_subplot(grid[2, col])
        ax.imshow(np.array(per_volume[name]), cmap='gray_r', aspect='auto', interpolation='nearest')
        ax.set(xlabel='phase-encode line', ylabel='volume',
               title=f'{name}: oracle mask of each volume (dark = acquired)')

    ax = fig.add_subplot(grid[3, 0])
    labels = ['within ' + a1, 'within ' + a2, 'between']
    means = [distances['within'][a1][0], distances['within'][a2][0], distances['between'][0]]
    stds = [distances['within'][a1][1], distances['within'][a2][1], distances['between'][1]]
    ax.bar(labels, means, yerr=stds, color=[COLORS[a1], COLORS[a2], '#666666'], capsize=4)
    ax.set(ylabel='mean |density difference|',
           title='Per-volume mask distance: between anatomies vs within (bars = std over pairs)')

    ax = fig.add_subplot(grid[3, 1])
    mask_names = list(transfer['masks'])
    width = 0.8 / len(mask_names)
    for k, mask_name in enumerate(mask_names):
        values = [transfer['scores'][mask_name][ev] for ev in (a1, a2)]
        ax.bar(np.arange(2) + k * width, values, width, label=mask_name)
    ax.set_xticks(np.arange(2) + 0.4 - width / 2, [f'eval on {a1}', f'eval on {a2}'])
    ax.set(ylabel=f'held-out {args.objective} (lower is better)', title='Transfer on held-out volumes')
    ax.legend(fontsize=7)
    fig.savefig(out_dir / 'brain_vs_knee_oracle.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(sweep), 1, figsize=(11, 1.7 * len(sweep) + 1), sharex=True, constrained_layout=True)
    for ax, (level, masks) in zip(np.atleast_1d(axes), sweep.items()):
        for name in (a1, a2):
            ax.plot(x, density(masks[name], args.smooth), color=COLORS[name], lw=2, label=name)
        ax.set(ylabel=f'noise {level:g}')
    np.atleast_1d(axes)[0].legend(fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('phase-encode line')
    fig.suptitle('Population oracle line density vs the oracle noise-level knob')
    fig.savefig(out_dir / 'noise_sweep.png', dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = args.resolution
    budget = int(round(W / args.acceleration))
    free = center_lines(W, args.preselect, 'cpu').numpy() == 0

    volumes = {}
    for anatomy in args.anatomies:
        print(f'loading {anatomy}', flush=True)
        volumes[anatomy] = load_volumes(args, anatomy)

    pop, per_volume, held_out = {}, {}, {}
    for anatomy in args.anatomies:
        vols = volumes[anatomy]
        search, held_out[anatomy] = vols[:args.search_volumes], vols[args.search_volumes:]
        print(f'{anatomy}: per-volume greedy over {len(vols)} volumes', flush=True)
        per_volume[anatomy] = [greedy_mask(args, population_oracle(args, [v], args.noise_level), budget)
                               for v in vols]
        print(f'{anatomy}: population greedy over {len(search)} volumes', flush=True)
        pop[anatomy] = greedy_mask(args, population_oracle(args, search, args.noise_level), budget)

    a1, a2 = args.anatomies
    distances = {
        'within': {n: pairwise(per_volume[n], per_volume[n], args.smooth, free, True) for n in (a1, a2)},
        'between': pairwise(per_volume[a1], per_volume[a2], args.smooth, free, False),
    }
    distances['population'] = mask_distance(pop[a1], pop[a2], args.smooth, free)
    both_free = free & ((pop[a1] > 0) | (pop[a2] > 0))
    distances['population_jaccard_of_free_lines'] = float(
        ((pop[a1] > 0) & (pop[a2] > 0))[free].sum() / max(1, both_free.sum()))

    print('held-out transfer matrix', flush=True)
    transfer_masks = {f'{a1} oracle': pop[a1], f'{a2} oracle': pop[a2], **baselines(W, budget, args.preselect, device)}
    scores = {}
    for mask_name, mask in transfer_masks.items():
        tensor = torch.from_numpy(mask).float().to(device).unsqueeze(0)
        scores[mask_name] = {
            ev: float(population_oracle(args, held_out[ev], args.noise_level).score(tensor, args.objective).item())
            for ev in (a1, a2)
        }
    transfer = {'masks': list(transfer_masks), 'scores': scores}
    regret = {ev: {'wrong_mask_cost_pct': 100 * (scores[f'{other} oracle'][ev] / scores[f'{ev} oracle'][ev] - 1)}
              for ev, other in ((a1, a2), (a2, a1))}

    sweep = {}
    for level in args.noise_sweep:
        print(f'noise sweep {level}', flush=True)
        sweep[level] = {n: greedy_mask(args, population_oracle(args, volumes[n][:args.search_volumes], level), budget)
                        for n in (a1, a2)}
    sweep = {args.noise_level: pop, **sweep}
    sweep = dict(sorted(sweep.items()))
    sweep_distance = {str(level): mask_distance(m[a1], m[a2], args.smooth, free) for level, m in sweep.items()}

    plot_main(args, W, budget, pop, per_volume, transfer, distances, sweep, args.out_dir)

    payload = {
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'volumes': {n: [v['name'] for v in volumes[n]] for n in args.anatomies},
        'population_masks': {n: pop[n].astype(int).tolist() for n in args.anatomies},
        'per_volume_masks': {n: [m.astype(int).tolist() for m in per_volume[n]] for n in args.anatomies},
        'distances': distances,
        'transfer_scores': scores,
        'wrong_mask_regret': regret,
        'noise_sweep_population_distance': sweep_distance,
    }
    (args.out_dir / 'results.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ('distances', 'transfer_scores', 'wrong_mask_regret',
                                              'noise_sweep_population_distance')}, indent=2))
    print(f'wrote {args.out_dir}')


if __name__ == '__main__':
    main()
