"""Is the per-slice oracle mask a unique target, or one of many equally good masks?

oracle_labels_r8.py fits each slice's mask from one shared start (prior.pt). Its 0.97 retest only changed the
Bernoulli seed, so it shows the optimisation path is reproducible, not that the optimum is. Here the 32 retest
slices are refitted (same frozen VarNet, steps, draws, lr, seed 1) from other starts, whose label directories
hold their own prior.pt:

  init_loupe   the LOUPE run's logits (slope x w), a different population mask (logit corr 0.62 with prior.pt)
  init_noise   prior.pt plus i.i.d. N(0, 1) logit noise per location (logit corr 0.93)

Per pair of labels of the same slice: the correlation of their deviations over the available locations, raw and
after a Gaussian blur of that deviation (sigma px). The headline reference is each set's own leave-one-out anatomy
mean: a refit keeps its start's population map, which the shared anatomy-mean reference (training labels, also
reported) would count as per-slice disagreement. Same gain with a low raw correlation means the optimum is not
identifiable at pixel scale, so a predictor has no single target. Also the
correlation of each refit's deviation with its own start's deviation (does the fit keep its start?). Every arm is
scored with the trainer's masked PSNR over seeded draws shared by all arms, against the anatomy-mean label, with
volume-bootstrap 95% intervals; 'mean of A and B' averages two labels' probabilities.

    python scripts/diagnostics/oracle_init_identifiability_r8.py
"""
import argparse
import itertools
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.training.builders import build_dataloaders
from label_curriculum_r8 import gaussian_blur
from label_predictor_gonogo_r8 import (ANATOMIES, anatomy_reference, available_of, bootstrap_ci, label_probability,
                                       load_shards, logit_of)
from oracle_labels_r8 import OUT_DIR, RUN, load_frozen, loader_over, masks_from_logits, seeded_psnr, to_device

OUT = Path('reports/label_gonogo_r8/oracle_init_identifiability.json')
REFITS = {'retest': Path(f'{OUT_DIR}_retest'), 'init_loupe': Path(f'{OUT_DIR}_init_loupe'),
          'init_noise': Path(f'{OUT_DIR}_init_noise')}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0, 1, 2, 4, 8])
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def deviation_correlations(first, second, reference, available, sigma):
    """Per slice, corr over available locations of the sigma-blurred deviations of two probability maps."""
    def blurred(probability):
        return gaussian_blur((probability - reference) * available, sigma)
    x, y = blurred(first), blurred(second)
    correlations = []
    for a, b, keep in zip(x, y, available.bool()):
        correlations.append(float(torch.corrcoef(torch.stack([a[keep].double(), b[keep].double()]))[0, 1]))
    return np.array(correlations)


def leave_one_out_reference(probability, anatomies):
    """Each slice's anatomy mean over the other slices of the same set, which removes that set's own population map."""
    reference = torch.empty_like(probability)
    for anatomy in ANATOMIES:
        rows = np.where(anatomies == anatomy)[0]
        total = probability[rows].sum(0)
        for row in rows:
            reference[row] = (total - probability[row]) / (len(rows) - 1)
    return reference


def own_reference_correlations(first, second, anatomies, available, sigma):
    """deviation_correlations with each map's deviation taken from its own set's leave-one-out anatomy mean."""
    def blurred(probability):
        return gaussian_blur((probability - leave_one_out_reference(probability, anatomies)) * available, sigma)
    x, y = blurred(first), blurred(second)
    return np.array([float(torch.corrcoef(torch.stack([a[keep].double(), b[keep].double()]))[0, 1])
                     for a, b, keep in zip(x, y, available.bool())])


def matched_rows(labels, indices):
    row_of = {int(index): row for row, index in enumerate(labels['indices'])}
    rows = [row_of[int(index)] for index in indices]
    return {key: (value[rows] if len(value) == len(labels['indices']) else value) for key, value in labels.items()}


@torch.no_grad()
def main():
    args = parse_args()
    train = load_shards(args.label_dir, 'train')
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}

    refits = {name: load_shards(path, 'val') for name, path in REFITS.items()}
    indices = refits['retest']['indices']
    labels = {'original': matched_rows(load_shards(args.label_dir, 'val'), indices)}
    labels.update({name: matched_rows(shard, indices) for name, shard in refits.items()})
    anatomies = labels['original']['anatomy']
    reference = anatomy_reference(anatomy_mean, anatomies)
    available = available_of(labels['original'])
    probability = {name: label_probability(shard) for name, shard in labels.items()}
    starts = {name: scores_to_prob(torch.load(REFITS.get(name, args.label_dir) / 'prior.pt').float()).expand_as(reference)
              for name in ('original', 'init_loupe', 'init_noise')}

    correlation, own_reference_correlation = {}, {}
    for first, second in itertools.combinations(probability, 2):
        correlation[f'{first}_vs_{second}'] = {
            f'sigma{s:g}': float(deviation_correlations(probability[first], probability[second], reference, available,
                                                        s).mean()) for s in args.sigmas}
        own_reference_correlation[f'{first}_vs_{second}'] = {
            f'sigma{s:g}': float(own_reference_correlations(probability[first], probability[second], anatomies,
                                                            available, s).mean()) for s in args.sigmas}
    population_map_correlation = {}
    for name in probability:
        population_map_correlation[name] = {}
        for anatomy in ANATOMIES:
            rows, keep = torch.as_tensor(anatomies == anatomy), available[0, 0].bool()
            means = torch.stack([probability['original'][rows].mean(0)[0][keep], probability[name][rows].mean(0)[0][keep]])
            population_map_correlation[name][anatomy] = float(torch.corrcoef(means)[0, 1])
    keeps_start = {}
    for name, start in starts.items():
        keeps_start[name] = {f'sigma{s:g}': float(deviation_correlations(probability[name], start, reference, available,
                                                                          s).mean()) for s in args.sigmas}
    keeps_start['original_vs_start_loupe'] = {
        f'sigma{s:g}': float(deviation_correlations(probability['original'], starts['init_loupe'], reference, available,
                                                    s).mean()) for s in args.sigmas}
    keeps_start['original_vs_start_noise'] = {
        f'sigma{s:g}': float(deviation_correlations(probability['original'], starts['init_noise'], reference, available,
                                                    s).mean()) for s in args.sigmas}

    arms = {'anatomy_mean_label': logit_of(reference),
            **{f'start_{name}': logit_of(start) for name, start in starts.items()},
            **{name: shard['label_logits'].float() for name, shard in labels.items()},
            'mean_of_original_and_init_loupe': logit_of((probability['original'] + probability['init_loupe']) / 2),
            'mean_of_original_and_init_noise': logit_of((probability['original'] + probability['init_noise']) / 2)}

    model, _ = load_frozen(args.device)
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    psnr = {name: [] for name in arms}
    volumes = []
    for chunk_index, batch in enumerate(loader_over(val_loader, indices.numpy(), args.chunk)):
        part = to_device(batch, args.device)
        rows = slice(chunk_index * args.chunk, chunk_index * args.chunk + len(part['kspace']))
        for name, logits in arms.items():
            chunk_logits = logits[rows].to(args.device)
            psnr[name].append(seeded_psnr(model, part, lambda: masks_from_logits(model, chunk_logits, part['kspace']),
                                          chunk_index))
        volumes += list(batch['fname'])
        print(f"scored {len(volumes)}/{len(indices)}", flush=True)
    psnr = {name: torch.cat(values).numpy() for name, values in psnr.items()}
    volumes = np.array(volumes)

    rng = np.random.RandomState(args.seed)
    control = psnr['anatomy_mean_label']
    gains = {name: {'vs_anatomy_mean': bootstrap_ci(psnr[name] - control, volumes, rng),
                    'by_anatomy': {a: float((psnr[name] - control)[anatomies == a].mean()) for a in ANATOMIES}}
             for name in arms}
    summary = {'slices': len(indices), 'volumes': int(len(np.unique(volumes))),
               'deviation_correlation_own_reference': own_reference_correlation,
               'deviation_correlation_shared_reference': correlation,
               'population_map_correlation_with_original': population_map_correlation,
               'refit_vs_start_deviation': keeps_start, 'gain_db': gains}
    print(json.dumps(summary, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps({'summary': summary, 'config': config}, indent=2) + '\n')


if __name__ == '__main__':
    main()
