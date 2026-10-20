"""Upper bound for scan-adaptive mask *selection* (SUNO-style): how much does the best of K candidate masks gain?

Candidates are K medoids, per anatomy, of the training oracle labels of oracle_labels_r8.py (k-medoids on the
label probability over available k-space locations). Medoids are real, sharp label masks: averaging labels
blurs them, and oracle_blur_psnr_r8.py showed a blurred label loses the gain.

Every candidate is scored on every labelled validation slice through the frozen baseline VarNet with the
trainer's masked PSNR. The pick uses SELECTION_SEEDS draws; the reported PSNR uses the disjoint SCORING_SEEDS
draws shared with every other diagnostic, so the pick is not credited with its own draw luck. Arms, against the
anatomy-mean label with volume-bootstrap 95% intervals:

  oracle              the slice's own label (ceiling)
  best_of_k<K>        the candidate a perfect selector would pick from K medoids
  random_of_k<K>      the average candidate (what a selector with no information gets)

If best_of_k does not beat the anatomy mean, no selector from a candidate set can, through this reconstructor.
The reconstructor never trained on these candidates, so the bound is pessimistic for a jointly trained set.

    python scripts/diagnostics/best_of_k_oracle_r8.py --ks 4 16 64
"""
import argparse
import json
import time
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from label_predictor_gonogo_r8 import (ANATOMIES, anatomy_reference, available_of, bootstrap_ci, label_probability,
                                       load_shards, logit_of)
from oracle_labels_r8 import OUT_DIR, RUN, SCORING_SEEDS, load_frozen, loader_over, masks_from_logits, to_device

OUT = Path('analysis/label_gonogo/best_of_k_oracle.json')
SELECTION_SEEDS = (10, 11)
MEDOID_ITERATIONS = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--ks', type=int, nargs='+', default=[4, 16, 64])
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def k_medoids(distance, k, rng):
    """Alternating k-medoids on a precomputed distance matrix, k-means++ initialised. Returns medoid row indices."""
    medoids = [rng.randint(len(distance))]
    for _ in range(k - 1):
        nearest = distance[:, medoids].min(axis=1) ** 2
        medoids.append(rng.choice(len(distance), p=nearest / nearest.sum()))
    medoids = np.array(medoids)
    for _ in range(MEDOID_ITERATIONS):
        assignment = distance[:, medoids].argmin(axis=1)
        updated = np.array([np.flatnonzero(assignment == c)[distance[np.ix_(assignment == c, assignment == c)].sum(1).argmin()]
                            if (assignment == c).any() else medoids[c] for c in range(k)])
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return medoids


def candidate_sets(train, ks, rng, device):
    """Per anatomy: the union of medoid rows over all K, and for each K the positions of its medoids in that union."""
    probability = label_probability(train) * available_of(train)
    union, positions = {}, {}
    for anatomy in ANATOMIES:
        rows = np.flatnonzero(train['anatomy'] == anatomy)
        features = probability[torch.as_tensor(rows)].flatten(1).to(device)
        distance = torch.cdist(features, features).cpu().numpy()
        by_k = {k: rows[k_medoids(distance, k, rng)] for k in ks}
        union[anatomy] = np.unique(np.concatenate(list(by_k.values())))
        positions[anatomy] = {k: np.searchsorted(union[anatomy], by_k[k]) for k in ks}
    return union, positions


@torch.no_grad()
def psnr_over_seeds(model, part, logits, chunk_index, seeds):
    psnr = torch.zeros(len(part['kspace']))
    for seed in seeds:
        torch.manual_seed(seed * 100003 + chunk_index)
        mask = masks_from_logits(model, logits, part['kspace'])
        output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        psnr += compute_psnr_torch(*_apply_signal_mask(output, target, part['metric_mask'])).flatten().cpu() / len(seeds)
    return psnr


@torch.no_grad()
def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    model, _ = load_frozen(args.device)
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    train_logits = train['label_logits'].float()
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    union, positions = candidate_sets(train, args.ks, rng, args.device)
    candidates = max(len(rows) for rows in union.values())
    print({a: len(rows) for a, rows in union.items()}, 'candidates per anatomy', flush=True)

    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    indices, anatomies = val['indices'].numpy(), val['anatomy']
    reference = logit_of(anatomy_reference(anatomy_mean, anatomies))
    own_label = val['label_logits'].float()

    fixed = {'anatomy_mean_label': [], 'oracle': []}
    selection, scoring = torch.full((len(indices), candidates), torch.nan), torch.full((len(indices), candidates), torch.nan)
    volumes, started = [], time.time()
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, args.chunk)):
        part = to_device(batch, args.device)
        rows = np.arange(chunk_index * args.chunk, chunk_index * args.chunk + len(part['kspace']))
        for name, logits in (('anatomy_mean_label', reference), ('oracle', own_label)):
            fixed[name].append(psnr_over_seeds(model, part, logits[rows].to(args.device), chunk_index, SCORING_SEEDS))
        for j in range(candidates):
            source = [union[anatomies[r]][min(j, len(union[anatomies[r]]) - 1)] for r in rows]
            logits = train_logits[torch.as_tensor(source)].to(args.device)
            selection[rows, j] = psnr_over_seeds(model, part, logits, chunk_index, SELECTION_SEEDS)
            scoring[rows, j] = psnr_over_seeds(model, part, logits, chunk_index, SCORING_SEEDS)
        volumes += list(batch['fname'])
        print(f"scored {len(volumes)}/{len(indices)}, {(time.time() - started) / len(volumes):.1f} s/slice", flush=True)

    psnr = {name: torch.cat(values).numpy() for name, values in fixed.items()}
    selection, scoring = selection.numpy(), scoring.numpy()
    for k in args.ks:
        own = [positions[anatomies[r]][k] for r in range(len(indices))]
        picked = np.array([own[r][selection[r, own[r]].argmax()] for r in range(len(indices))])
        psnr[f'best_of_k{k}'] = scoring[np.arange(len(indices)), picked]
        psnr[f'best_of_k{k}_same_seeds'] = np.array([scoring[r, own[r]].max() for r in range(len(indices))])
        psnr[f'random_of_k{k}'] = np.array([scoring[r, own[r]].mean() for r in range(len(indices))])

    volumes, control = np.array(volumes), psnr['anatomy_mean_label']
    oracle_gain = float((psnr['oracle'] - control).mean())
    summary = {}
    for name, values in psnr.items():
        gain = values - control
        summary[name] = {'mean_psnr': float(values.mean()),
                         'vs_anatomy_mean': bootstrap_ci(gain, volumes, rng),
                         'by_anatomy': {a: float(gain[anatomies == a].mean()) for a in ANATOMIES},
                         'fraction_of_oracle_gain': float(gain.mean()) / oracle_gain}
        print(f"{name:28s} {summary[name]['vs_anatomy_mean']['mean']:+.3f} dB "
              f"[{summary[name]['vs_anatomy_mean']['ci95'][0]:+.3f}, {summary[name]['vs_anatomy_mean']['ci95'][1]:+.3f}]",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps({'summary': summary, 'config': config,
                                    'medoid_rows': {a: rows.tolist() for a, rows in union.items()}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
