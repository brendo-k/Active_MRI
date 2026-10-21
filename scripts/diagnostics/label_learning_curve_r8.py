"""Does predicting a slice's oracle-mask deviation get better with more training volumes?

The ridge and kNN predictors of smooth_label_predictor_r8.py are refitted on random subsets of the labelled training
volumes of oracle_labels_r8.py (fractions `--fractions`, `--repeats` subsets each, per anatomy) and scored on the
fixed 128 validation labels. One more arm at `--matched-fraction` keeps every volume but only that fraction of each
volume's labelled slices, with the label count of the paired volume subset: more subjects against more slices.

Targets, all against the full training set's anatomy-mean label (fixed for every subset):
  smooth<sigma>   the smoothed log density ratio of smooth_label_predictor_r8.py
  unsplit         the probability deviation label - anatomy mean over the available locations
  pixel4          its pixel band: the deviation minus its normalised 4 px blur (acs_vs_label_similarity_r8.py)

Predictability is the per-slice correlation of (prediction - the subset's own mean target) with (target - the full
training mean), so a small subset is not charged for estimating the population map, only for the per-slice part.
At fraction 1 the smooth targets reproduce analysis/label_gonogo/smooth_predictor.json. Steps between fractions
and the slope per doubling are paired over the same validation slices, with volume-bootstrap 95% intervals.

    python scripts/diagnostics/label_learning_curve_r8.py --device cuda:3
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from acs_vs_label_similarity_r8 import band_of
from label_predictor_gonogo_r8 import (ANATOMIES, BOOTSTRAPS, anatomy_reference, available_of, full_sampler_input,
                                       label_probability, load_shards)
from oracle_labels_r8 import OUT_DIR, RUN, load_frozen, loader_over, sampler_input, to_device
from smooth_label_predictor_r8 import PENALTIES, features, log_ratio

OUT = Path('analysis/label_gonogo/learning_curve.json')
REFERENCE = Path('analysis/label_gonogo/smooth_predictor.json')
PIXEL_SIGMA = 4
MAX_FOLDS = 5
SPOT_CHECKS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.125, 0.25, 0.5, 1.0])
    parser.add_argument('--matched-fraction', type=float, default=0.5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[4, 8])
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--knn', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def checked_sources(loader, labels):
    """Volume (file) of each label, asserting the loader still has the anatomy the label was made with."""
    examples, anatomies = [], []
    for dataset in loader.dataset.datasets:
        examples += dataset.examples
        anatomies += [dataset.anatomy] * len(dataset.examples)
    indices = labels['indices'].numpy()
    loader_anatomy = np.array([anatomies[int(i)] for i in indices])
    mismatched = int((loader_anatomy != labels['anatomy']).sum())
    assert mismatched == 0, f'{mismatched} labels have a different anatomy in the rebuilt loader'
    return np.array([str(examples[int(i)][0]) for i in indices])


@torch.no_grad()
def spot_check_inputs(model, loader, labels, device, rng):
    """Max relative error between the stored sampler input and the rebuilt loader's, on a few random labels."""
    rows = np.sort(rng.choice(len(labels['indices']), SPOT_CHECKS, replace=False))
    batch = next(iter(loader_over(loader, labels['indices'].numpy()[rows], SPOT_CHECKS)))
    rebuilt = sampler_input(model, to_device(batch, device)['kspace'])[0].cpu()
    stored = labels['kspace_in'][rows].float()
    return float(((rebuilt - stored).flatten(1).norm(dim=1) / stored.flatten(1).norm(dim=1)).max())


def targets_of(train, val, sigmas, eps):
    """{name: (train rows, val rows)} over the locations available in every label, float32."""
    train_probability, val_probability = label_probability(train), label_probability(val)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    train_reference = anatomy_reference(anatomy_mean, train['anatomy'])
    val_reference = anatomy_reference(anatomy_mean, val['anatomy'])
    train_available, val_available = available_of(train), available_of(val)
    keep = (train_available.bool().all(0) & val_available.bool().all(0))[0].flatten()
    assert bool((train_available.bool().flatten(1)[:, keep]).all()) and bool((val_available.bool().flatten(1)[:, keep]).all())
    out = {}
    for sigma in sigmas:
        out[f'smooth{sigma:g}'] = tuple(log_ratio(p, r, a, sigma, eps)[:, 0].flatten(1)[:, keep].float() for p, r, a in (
            (train_probability, train_reference, train_available), (val_probability, val_reference, val_available)))
    for band in ('unsplit', 'pixel'):
        name = 'unsplit' if band == 'unsplit' else f'pixel{PIXEL_SIGMA}'
        out[name] = tuple(band_of((p - r) * a, a, band, PIXEL_SIGMA)[:, 0].flatten(1)[:, keep].float() for p, r, a in (
            (train_probability, train_reference, train_available), (val_probability, val_reference, val_available)))
    return out


def kernel_ridge(x, y, query, penalties):
    """Ridge predictions of `query` for each penalty from one eigendecomposition of the centred Gram matrix
    (x, query double; y float32 on the same device)."""
    mean_x, mean_y = x.mean(0), y.mean(0)
    centred, query = x - mean_x, query - mean_x
    gram = centred @ centred.T
    scale = float(gram.trace()) / len(centred)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    projected = eigenvectors.T.float() @ (y - mean_y)
    cross = (query @ centred.T) @ eigenvectors
    for penalty in penalties:
        yield (cross / (eigenvalues + penalty * scale)).float() @ projected + mean_y


def ridge(x, y, volumes, query, rng):
    """One anatomy: penalty by volume-grouped CV, then the fit on every row. Returns the query predictions."""
    mean_x, std_x = x.mean(0), x.std(0).clamp(min=1e-8)
    x, query = (x - mean_x) / std_x, (query - mean_x) / std_x
    unique = np.unique(volumes)
    folds_count = min(MAX_FOLDS, len(unique))
    fold_of = dict(zip(rng.permutation(unique), np.arange(len(unique)) % folds_count))
    folds = np.array([fold_of[v] for v in volumes])
    errors = np.zeros(len(PENALTIES))
    for fold in range(folds_count):
        held, kept = torch.as_tensor(folds == fold, device=x.device), torch.as_tensor(folds != fold, device=x.device)
        for i, predicted in enumerate(kernel_ridge(x[kept], y[kept], x[held], PENALTIES)):
            errors[i] += float(((predicted - y[held]) ** 2).sum())
    return next(kernel_ridge(x, y, query, [PENALTIES[int(np.argmin(errors))]]))


def knn(x, y, query, k):
    mean, std = x.mean(0), x.std(0).clamp(min=1e-8)
    a, b = F.normalize((query - mean) / std, dim=1), F.normalize((x - mean) / std, dim=1)
    return y[(a @ b.T).topk(min(k, len(x)), dim=1).indices].mean(1)


def slice_correlations(prediction, prediction_mean, target, target_mean):
    p, t = prediction - prediction_mean, target - target_mean
    p, t = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    return ((p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-12)).cpu().numpy()


def subset_rows(anatomy, volume, fraction, matched, rng):
    """Training rows of one random subset: `fraction` of each anatomy's volumes with all their labels, or with
    `matched` every volume, each label ranked randomly within its volume and the lowest relative ranks kept up to the
    label count of an equally drawn volume subset."""
    rows = []
    for name in ANATOMIES:
        members = np.flatnonzero(anatomy == name)
        unique = np.unique(volume[members])
        chosen = rng.choice(unique, max(2, int(round(fraction * len(unique)))), replace=False)
        picked = members[np.isin(volume[members], chosen)]
        if matched:
            relative_rank = np.empty(len(members))
            for v in unique:
                inside = np.flatnonzero(volume[members] == v)
                relative_rank[inside] = (rng.permutation(len(inside)) + rng.rand()) / len(inside)
            picked = members[np.argsort(relative_rank, kind='stable')[:len(picked)]]
        rows.append(picked)
    return np.concatenate(rows)


def predict_subset(rows, inputs, targets, train_anatomy, train_volume, val_anatomy, args, fraction, rng, device):
    """{(predictor, input, target): per-val-slice correlations} for one training subset. `targets` holds
    (train rows on `device`, val rows, {anatomy: full training mean on `device`})."""
    knn_scaled = max(2, int(round(args.knn * fraction)))
    out = {}
    for input_name, (train_x, val_x) in inputs.items():
        for target_name, (train_y, val_y, full_means) in targets.items():
            for predictor in ('ridge', f'knn{args.knn}', 'knn_scaled'):
                out[(predictor, input_name, target_name)] = np.zeros(len(val_anatomy))
            for name in ANATOMIES:
                fit = rows[train_anatomy[rows] == name]
                queries = np.flatnonzero(val_anatomy == name)
                x, y = train_x[fit].to(device), train_y[torch.as_tensor(fit, device=device)]
                query, target = val_x[queries].to(device), val_y[queries].to(device)
                subset_mean, full_mean = y.mean(0), full_means[name]
                predictions = {'ridge': ridge(x, y, train_volume[fit], query, rng),
                               f'knn{args.knn}': knn(x, y, query, args.knn),
                               'knn_scaled': knn(x, y, query, knn_scaled)}
                for predictor, prediction in predictions.items():
                    out[(predictor, input_name, target_name)][queries] = slice_correlations(
                        prediction, subset_mean, target, full_mean)
    return out


def bootstrap(per_slice, volumes, rng):
    """Mean of a per-slice statistic and its 95% interval, resampling whole validation volumes."""
    unique = np.unique(volumes)
    by_volume = [per_slice[volumes == v] for v in unique]
    means = [np.concatenate([by_volume[i] for i in rng.randint(len(unique), size=len(unique))]).mean()
             for _ in range(BOOTSTRAPS)]
    return {'mean': float(per_slice.mean()), 'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def summarise(curves, fractions, matched, matched_fraction, val_anatomy, val_volume, rng):
    """Per arm: the curve, its spread over subsets, the top step, the slope per doubling and the matched arm."""
    log2 = np.log2(fractions)
    design = (log2 - log2.mean()) / ((log2 - log2.mean()) ** 2).sum()
    summary = {}
    for arm in curves[fractions[0]][0]:
        per_slice = {f: np.mean([repeat[arm] for repeat in curves[f]], axis=0) for f in fractions}
        repeat_means = {f: [float(repeat[arm].mean()) for repeat in curves[f]] for f in fractions}
        top, below = fractions[-1], fractions[-2]
        entry = {
            'curve': {f'{f:g}': {'mean': float(per_slice[f].mean()), 'subset_std': float(np.std(repeat_means[f])),
                                 **{a: float(per_slice[f][val_anatomy == a].mean()) for a in ANATOMIES}}
                      for f in fractions},
            f'step_{below:g}_to_{top:g}': bootstrap(per_slice[top] - per_slice[below], val_volume, rng),
            'slope_per_doubling': bootstrap(sum(w * per_slice[f] for w, f in zip(design, fractions)), val_volume, rng),
        }
        slices = np.mean([repeat[arm] for repeat in matched], axis=0)
        entry['matched_slices'] = float(slices.mean())
        entry['matched_slices_minus_volumes'] = bootstrap(slices - per_slice[matched_fraction], val_volume, rng)
        summary['/'.join(arm)] = entry
    return summary


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    model, _ = load_frozen(args.device)
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    train_volume, val_volume = checked_sources(train_loader, train), checked_sources(val_loader, val)
    spot = {'train': spot_check_inputs(model, train_loader, train, device, rng),
            'val': spot_check_inputs(model, val_loader, val, device, rng)}
    print('stored vs rebuilt sampler input, max relative error', spot, flush=True)
    assert max(spot.values()) < 0.02, 'labels no longer line up with the rebuilt loader'
    train_anatomy, val_anatomy = train['anatomy'], val['anatomy']
    coverage = {a: {'labels': int((train_anatomy == a).sum()), 'volumes': int(len(np.unique(train_volume[train_anatomy == a]))),
                    'val_labels': int((val_anatomy == a).sum()), 'val_volumes': int(len(np.unique(val_volume[val_anatomy == a])))}
                for a in ANATOMIES}
    print('coverage', coverage, flush=True)

    full = {split: full_sampler_input(model, loader, labels['indices'].numpy(), args.device, args.chunk)
            for split, loader, labels in (('train', train_loader, train), ('val', val_loader, val))}
    inputs = {
        'centre10': (features(train['kspace_in'], size=10), features(val['kspace_in'], size=10)),
        'centre24': (features(full['train'], size=24), features(full['val'], size=24)),
        'full': (features(full['train'], pool=8), features(full['val'], pool=8)),
    }
    targets = targets_of(train, val, args.sigmas, args.eps)
    targets = {name: (train_y.to(device), val_y,
                      {a: train_y[torch.as_tensor(train_anatomy == a)].mean(0).to(device) for a in ANATOMIES})
               for name, (train_y, val_y) in targets.items()}
    print('targets', {k: tuple(v[0].shape) for k, v in targets.items()}, flush=True)

    fractions = sorted(args.fractions)
    curves = {}
    for fraction in fractions:
        repeats = 1 if fraction >= 1 else args.repeats
        curves[fraction] = []
        for repeat in range(repeats):
            rows = subset_rows(train_anatomy, train_volume, fraction, False, rng)
            curves[fraction].append(predict_subset(rows, inputs, targets, train_anatomy, train_volume, val_anatomy,
                                                   args, fraction, rng, device))
            print(f'fraction {fraction:g} repeat {repeat}: {len(rows)} labels', flush=True)
    matched = []
    for repeat in range(args.repeats):
        rows = subset_rows(train_anatomy, train_volume, args.matched_fraction, True, rng)
        matched.append(predict_subset(rows, inputs, targets, train_anatomy, train_volume, val_anatomy, args,
                                      args.matched_fraction, rng, device))
        print(f'matched {args.matched_fraction:g} repeat {repeat}: {len(rows)} labels', flush=True)

    summary = summarise(curves, np.array(fractions), matched, args.matched_fraction, val_anatomy, val_volume, rng)
    reproduction = {}
    if REFERENCE.exists():
        old = json.loads(REFERENCE.read_text())['by_sigma']
        for sigma in args.sigmas:
            for old_name, arm in (('ridge_centre10', 'ridge/centre10'), ('ridge_centre24', 'ridge/centre24'),
                                  ('ridge_full', 'ridge/full'), ('knn_centre10', f'knn{args.knn}/centre10')):
                new = summary[f'{arm}/smooth{sigma:g}']['curve']['1']['mean']
                reproduction[f'{old_name}_sigma{sigma:g}'] = {'old': old[f'{sigma:g}']['predictability'][old_name], 'new': new}
        print('reproduction', json.dumps(reproduction), flush=True)

    report = {'coverage': coverage, 'spot_check_relative_error': spot, 'reproduction': reproduction,
              'summary': summary, 'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
