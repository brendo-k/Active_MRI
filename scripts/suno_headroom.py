"""How much does scan-adaptive line selection buy over a population mask? (SUNO-style test)

Reuses the per-volume and population oracle masks that ``compare_anatomy_oracle.py``
saved, scores every mask on every volume with the same oracle, and reads the
adaptivity headroom straight off that score matrix:

  own      the volume's own oracle mask. In-sample, so it is an upper bound.
  nn       the mask of the nearest *other* volume, found from the zero-filled
           reconstruction of the central lines only -- what a scanner would have
           before choosing a mask (Gautam et al., SUNO).
  random   the mean score of the other volumes' masks, the control for "any mask
           from a similar scan".
  pop      the population oracle mask, scored on held-out volumes only.

If ``nn`` does not beat ``random``, the neighbour lookup carries no scan-specific
information. If ``own`` does not beat ``pop``, there is no headroom to adapt into.

    python scripts/suno_headroom.py --results runs/oracle/brain_vs_knee/results.json \\
        --device cuda:1 --out-dir runs/oracle/suno_headroom
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.samplers.oracle import center_lines
from activemri.utils.image_processing import ifft_2d_img
from compare_anatomy_oracle import COLORS, load_volumes, population_oracle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--lf-lines', type=int, default=24,
                        help='central lines used for the neighbour search')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    return parser.parse_args()


def low_frequency_signature(volume, lf_lines, device):
    """Unit-norm zero-filled image of the central lines, slices concatenated."""
    parts = []
    for record in volume['slices']:
        kspace = record['kspace']
        keep = center_lines(kspace.shape[-1], lf_lines, device)
        image = ifft_2d_img(kspace * keep).abs().pow(2).sum(dim=0).sqrt().flatten()
        parts.append(image / image.norm().clamp(min=1e-12))
    return torch.cat(parts)


def main():
    cli = parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    saved = json.loads(cli.results.read_text())
    args = Namespace(**{**saved['config'], 'device': cli.device, 'data_root': Path(saved['config']['data_root'])})
    device = torch.device(cli.device)
    anatomies = args.anatomies

    volumes, masks, labels, in_search = [], [], [], []
    for anatomy in anatomies:
        loaded = {v['name']: v for v in load_volumes(args, anatomy)}
        names = saved['volumes'][anatomy]
        for idx, name in enumerate(names):
            volumes.append(loaded[name])
            masks.append(torch.tensor(saved['per_volume_masks'][anatomy][idx], dtype=torch.float32))
            labels.append(anatomy)
            in_search.append(idx < args.search_volumes)
    labels = np.array(labels)
    in_search = np.array(in_search)
    num = len(volumes)

    pop_masks = [torch.tensor(saved['population_masks'][a], dtype=torch.float32) for a in anatomies]
    all_masks = torch.stack(masks + pop_masks).to(device)

    print('scoring every mask on every volume', flush=True)
    scores = np.zeros((num, all_masks.shape[0]))
    for v, volume in enumerate(volumes):
        oracle = population_oracle(args, [volume], args.noise_level)
        scores[v] = oracle.score(all_masks, args.objective).cpu().numpy()

    signatures = torch.stack([low_frequency_signature(v, cli.lf_lines, device) for v in volumes])
    distance = torch.cdist(signatures, signatures).cpu().numpy()
    np.fill_diagonal(distance, np.inf)
    nearest = distance.argmin(axis=1)

    own = scores[np.arange(num), np.arange(num)]
    nn = scores[np.arange(num), nearest]
    other_mean = np.array([np.delete(scores[v, :num], v).mean() for v in range(num)])
    same_anatomy_mean = np.array([
        scores[v, :num][(labels == labels[v]) & (np.arange(num) != v)].mean() for v in range(num)])
    pct = lambda a, b: 100 * (a / b - 1)

    held_out = np.where(~in_search)[0]
    pop = np.array([scores[v, num + anatomies.index(labels[v])] for v in held_out])

    summary = {
        'num_volumes': num,
        'nn_matches_own_anatomy': float((labels[nearest] == labels).mean()),
        'leave_one_out': {
            'nn_vs_random_pct': pct(nn, other_mean).tolist(),
            'nn_vs_same_anatomy_random_pct': pct(nn, same_anatomy_mean).tolist(),
            'own_vs_nn_pct': pct(own, nn).tolist(),
        },
        'held_out': {
            'volumes': [saved['volumes'][labels[v]][int(np.sum(labels[:v] == labels[v]))] for v in held_out],
            'own_vs_pop_pct': pct(own[held_out], pop).tolist(),
            'nn_vs_pop_pct': pct(nn[held_out], pop).tolist(),
            'random_vs_pop_pct': pct(other_mean[held_out], pop).tolist(),
        },
    }
    mean = lambda x: float(np.mean(x))
    sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
    summary['means'] = {
        'nn_vs_random_pct': [mean(summary['leave_one_out']['nn_vs_random_pct']),
                             sem(summary['leave_one_out']['nn_vs_random_pct'])],
        'nn_vs_same_anatomy_random_pct': [mean(summary['leave_one_out']['nn_vs_same_anatomy_random_pct']),
                                          sem(summary['leave_one_out']['nn_vs_same_anatomy_random_pct'])],
        'own_vs_nn_pct': [mean(summary['leave_one_out']['own_vs_nn_pct']),
                          sem(summary['leave_one_out']['own_vs_nn_pct'])],
        'own_vs_pop_pct': [mean(summary['held_out']['own_vs_pop_pct']), sem(summary['held_out']['own_vs_pop_pct'])],
        'nn_vs_pop_pct': [mean(summary['held_out']['nn_vs_pop_pct']), sem(summary['held_out']['nn_vs_pop_pct'])],
        'random_vs_pop_pct': [mean(summary['held_out']['random_vs_pop_pct']),
                              sem(summary['held_out']['random_vs_pop_pct'])],
    }
    (cli.out_dir / 'results.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('nn_matches_own_anatomy', 'means')}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    relative = 100 * (scores[:, :num] / own[:, None] - 1)
    image = axes[0].imshow(relative, cmap='viridis', vmax=np.percentile(relative, 95))
    axes[0].set(xlabel='mask from volume', ylabel='scored on volume',
                title='Cost of using another volume\'s mask (% above own mask)')
    boundary = np.sum(labels == anatomies[0]) - 0.5
    axes[0].axhline(boundary, color='w', lw=1)
    axes[0].axvline(boundary, color='w', lw=1)
    fig.colorbar(image, ax=axes[0])

    ax = axes[1]
    groups = {
        'own mask (in-sample)': pct(own, other_mean),
        'nearest neighbour': pct(nn, other_mean),
        'same-anatomy average': pct(same_anatomy_mean, other_mean),
    }
    positions = np.arange(len(groups))
    for k, anatomy in enumerate(anatomies):
        sel = labels == anatomy
        ax.boxplot([v[sel] for v in groups.values()], positions=positions + (k - 0.5) * 0.35, widths=0.3,
                   patch_artist=True, boxprops={'facecolor': COLORS[anatomy], 'alpha': 0.6},
                   medianprops={'color': 'k'})
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(positions, list(groups), rotation=12)
    ax.set(ylabel="% vs average of other volumes' masks",
           title='Nearest-neighbour mask vs controls (leave-one-out)')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[a], alpha=0.6) for a in anatomies],
              labels=list(anatomies), fontsize=8)

    ax = axes[2]
    groups = {'own (in-sample)': pct(own[held_out], pop), 'nearest neighbour': pct(nn[held_out], pop),
              'other-volume average': pct(other_mean[held_out], pop)}
    positions = np.arange(len(groups))
    for k, anatomy in enumerate(anatomies):
        sel = labels[held_out] == anatomy
        ax.boxplot([v[sel] for v in groups.values()], positions=positions + (k - 0.5) * 0.35, widths=0.3,
                   patch_artist=True, boxprops={'facecolor': COLORS[anatomy], 'alpha': 0.6},
                   medianprops={'color': 'k'})
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(positions, list(groups), rotation=12)
    ax.set(ylabel='% vs population mask (negative = better)', title='Held-out volumes vs the population oracle')
    fig.savefig(cli.out_dir / 'suno_headroom.png', dpi=150)
    plt.close(fig)
    print(f'wrote {cli.out_dir}')


if __name__ == '__main__':
    main()
