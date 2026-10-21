"""Can the smooth part of a slice's oracle mask be predicted from what the scanner has before sampling?

oracle_blur_psnr_r8.py --space density shows that each labelled slice's oracle, kept only at scales above sigma
(the anatomy-mean probability times the label's Gaussian-smoothed density ratio), still beats the anatomy mean by
+0.05-0.07 dB at sigma 4-8. The target here is that smoothed log density ratio
r = log((G_sigma * (label * available) + eps) / (G_sigma * (anatomy mean * available) + eps)), and every arm's mask is
anatomy mean * exp(r_hat), scored like oracle_blur_psnr_r8.py: the labels' frozen VarNet, the trainer's masked PSNR
over seeded Bernoulli draws shared by every arm, gains against the anatomy-mean label with volume-bootstrap 95%
intervals.

  smooth_oracle        the slice's own r (oracle_blur_psnr_r8.py --space density)
  anatomy_mean_ratio   the training labels' mean r per anatomy: a static smooth refit, not adaptive
  protocol_mean_ratio  the same per protocol (brain / knee-PD / knee-PDFS), known from the header
  other_slice          r of a labelled slice of the same anatomy from another volume
  volume_oracle        mean r of the other labelled slices of the same volume: one mask per volume, as SPACE needs
  ridge_<input>        ridge regression from calibration features to r, per anatomy, penalty by volume-grouped CV
  knn_centre10         mean r of the 16 nearest same-anatomy training slices by centre10 features

Inputs: centre10 is the sampler's own observation (the 10 x 10 centre, SENSE-combined); centre24 a 24 x 24 centre;
full the whole k-space pooled to 32 x 32 (privileged, a ceiling). Features are log-magnitudes normalised by the
slice's norm, plus the log norm. Predictability is the per-slice correlation of r_hat - mean and r - mean over the
available locations (mean: the anatomy's training mean of r); cv_r2 is ridge's out-of-fold explained variance.

    python scripts/diagnostics/smooth_label_predictor_r8.py
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.datasets.fastmri_data import condition_name
from activemri.training.builders import build_dataloaders
from label_curriculum_r8 import gaussian_blur
from label_predictor_gonogo_r8 import (ANATOMIES, anatomy_reference, available_of, bootstrap_ci, full_sampler_input,
                                       label_probability, load_shards, logit_of)
from oracle_labels_r8 import OUT_DIR, RUN, load_frozen, loader_over, masks_from_logits, seeded_psnr, to_device

OUT = Path('reports/label_gonogo_r8/smooth_predictor.json')
PROTOCOLS = ('brain', 'knee-PD', 'knee-PDFS')
CV_FOLDS = 5
PENALTIES = np.logspace(-3, 4, 15)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[4, 8])
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--knn', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def sources(loader, indices):
    """(file, protocol) of each dataset index of `loader`'s ConcatDataset."""
    examples, anatomies = [], []
    for dataset in loader.dataset.datasets:
        examples += dataset.examples
        anatomies += [dataset.anatomy] * len(dataset.examples)
    protocol_of = {}
    for path, anatomy in {(e[0], a) for e, a in zip(examples, anatomies)}:
        with h5py.File(path, 'r') as handle:
            protocol_of[path] = condition_name(anatomy, dict(handle.attrs))
    rows = [examples[int(i)][0] for i in indices]
    return np.array([str(path) for path in rows]), np.array([protocol_of[path] for path in rows])


def log_ratio(label, reference, available, sigma, eps):
    return torch.log((gaussian_blur(label * available, sigma) + eps) / (gaussian_blur(reference * available, sigma) + eps))


def features(kspace, size=None, pool=None):
    """Log-magnitude of the centred `size` block (or of the whole k-space average-pooled by `pool`), normalised by the
    slice's norm, with the log norm appended: [N, d]."""
    magnitude = kspace.float().norm(dim=-1)
    if size is not None:
        height, width = magnitude.shape[1:]
        top, left = height // 2 - size // 2, width // 2 - size // 2
        magnitude = magnitude[:, top:top + size, left:left + size]
    else:
        magnitude = F.avg_pool2d(magnitude.unsqueeze(1), pool)[:, 0]
    flat = magnitude.flatten(1)
    norm = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.cat([torch.log(flat / norm + 1e-6), norm.log()], dim=1).double()


def kernel_ridge(train_x, train_y, query_x, penalties):
    """Ridge predictions of query rows for every penalty (scaled by the mean Gram diagonal), from one eigendecomposition
    of the centred training Gram matrix; the training means are added back."""
    mean_x, mean_y = train_x.mean(0), train_y.mean(0)
    x, query = train_x - mean_x, query_x - mean_x
    gram = x @ x.T
    scale = float(gram.trace()) / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    projected_y = eigenvectors.T @ (train_y - mean_y)
    cross = (query @ x.T) @ eigenvectors
    for penalty in penalties:
        yield (cross / (eigenvalues + penalty * scale)) @ projected_y + mean_y


def ridge_by_anatomy(train_x, train_y, train_anatomy, train_volume, query_x, query_anatomy, rng):
    """Per anatomy: penalty by volume-grouped CV (out-of-fold MSE), then the fit on all of that anatomy's labels.
    Returns predictions for the queries (zero for other anatomies) and, per anatomy, the chosen penalty and the
    out-of-fold R^2 against predicting the fold's training mean."""
    prediction = torch.zeros(len(query_x), train_y.shape[1], dtype=train_y.dtype)
    report = {}
    for anatomy in ANATOMIES:
        rows = np.flatnonzero(train_anatomy == anatomy)
        x, y = train_x[rows], train_y[rows]
        mean_x, std_x = x.mean(0), x.std(0).clamp(min=1e-8)
        x = (x - mean_x) / std_x
        volumes = np.unique(train_volume[rows])
        fold_of = dict(zip(rng.permutation(volumes), np.arange(len(volumes)) % CV_FOLDS))
        folds = np.array([fold_of[v] for v in train_volume[rows]])
        errors, baseline = np.zeros(len(PENALTIES)), 0.0
        for fold in range(CV_FOLDS):
            held, kept = folds == fold, folds != fold
            baseline += float(((y[held] - y[kept].mean(0)) ** 2).sum())
            for i, predicted in enumerate(kernel_ridge(x[kept], y[kept], x[held], PENALTIES)):
                errors[i] += float(((predicted - y[held]) ** 2).sum())
        best = int(np.argmin(errors))
        queries = np.flatnonzero(query_anatomy == anatomy)
        prediction[queries] = next(kernel_ridge(x, y, (query_x[queries] - mean_x) / std_x, [PENALTIES[best]]))
        report[anatomy] = {'penalty': float(PENALTIES[best]), 'cv_r2': 1 - float(errors[best]) / baseline,
                           'penalty_at_grid_edge': best in (0, len(PENALTIES) - 1)}
    return prediction, report


def knn_ratio(train_x, train_r, train_anatomy, query_x, query_anatomy, k):
    out = torch.zeros(len(query_x), train_r.shape[1], dtype=train_r.dtype)
    for anatomy in ANATOMIES:
        pool, queries = np.flatnonzero(train_anatomy == anatomy), np.flatnonzero(query_anatomy == anatomy)
        reference = train_x[pool]
        mean, std = reference.mean(0), reference.std(0).clamp(min=1e-8)
        a = F.normalize((query_x[queries] - mean) / std, dim=1)
        b = F.normalize((reference - mean) / std, dim=1)
        out[queries] = train_r[pool][(a @ b.T).topk(k, dim=1).indices].mean(1)
    return out


def deviation_correlation(prediction, target, mean):
    p, t = prediction - mean, target - mean
    p, t = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    return float(((p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-12)).mean())


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    model, _ = load_frozen(args.device)
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    train_volume, train_protocol = sources(train_loader, train['indices'].numpy())
    val_volume, val_protocol = sources(val_loader, val['indices'].numpy())
    train_anatomy, val_anatomy = train['anatomy'], val['anatomy']

    full = {split: full_sampler_input(model, loader, labels['indices'].numpy(), args.device, args.chunk)
            for split, loader, labels in (('train', train_loader, train), ('val', val_loader, val))}
    inputs = {
        'centre10': (features(train['kspace_in'], size=10), features(val['kspace_in'], size=10)),
        'centre24': (features(full['train'], size=24), features(full['val'], size=24)),
        'full': (features(full['train'], pool=8), features(full['val'], pool=8)),
    }
    print('features', {k: tuple(v[0].shape) for k, v in inputs.items()}, flush=True)

    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train_anatomy == a)].mean(0) for a in ANATOMIES}
    train_reference, val_reference = anatomy_reference(anatomy_mean, train_anatomy), anatomy_reference(anatomy_mean, val_anatomy)
    val_label, val_available = label_probability(val), available_of(val)
    train_available = available_of(train)
    common = (train_available.bool().all(0) & val_available.bool().all(0))[0]

    other_rng = np.random.RandomState(args.seed + 1)
    other = np.array([other_rng.choice(np.flatnonzero((val_anatomy == val_anatomy[i]) & (val_volume != val_volume[i])))
                      for i in range(len(val_anatomy))])

    arms = {'anatomy_mean_label': logit_of(val_reference)}
    report = {'counts': {'train': len(train_anatomy), 'val': len(val_anatomy)}, 'by_sigma': {}}
    for sigma in args.sigmas:
        train_r = log_ratio(train_probability, train_reference, train_available, sigma, args.eps)[:, 0].double()
        val_r = log_ratio(val_label, val_reference, val_available, sigma, args.eps)[:, 0].double()
        flat_train, flat_val = train_r.flatten(1), val_r.flatten(1)
        mean_r = {a: flat_train[torch.as_tensor(train_anatomy == a)].mean(0) for a in ANATOMIES}
        protocol_r = {p: flat_train[torch.as_tensor(train_protocol == p)].mean(0) for p in PROTOCOLS}
        val_mean = torch.stack([mean_r[a] for a in val_anatomy])
        volume_r = torch.stack([flat_val[np.flatnonzero((val_volume == val_volume[i]) & (np.arange(len(val_volume)) != i))].mean(0)
                                if ((val_volume == val_volume[i]).sum() > 1) else val_mean[i] for i in range(len(val_volume))])
        report.setdefault('volume_oracle_slices_with_partners', int(sum((val_volume == v).sum() > 1 for v in val_volume)))
        predictions = {
            'smooth_oracle': flat_val,
            'anatomy_mean_ratio': val_mean,
            'protocol_mean_ratio': torch.stack([protocol_r[p] for p in val_protocol]),
            'other_slice': flat_val[other],
            'volume_oracle': volume_r,
        }
        ridge_report = {}
        for name, (train_x, val_x) in inputs.items():
            predicted, ridge_report[name] = ridge_by_anatomy(train_x, flat_train[:, common.flatten()], train_anatomy,
                                                             train_volume, val_x, val_anatomy, rng)
            prediction = val_mean.clone()
            prediction[:, common.flatten()] = predicted
            predictions[f'ridge_{name}'] = prediction
        predictions['knn_centre10'] = knn_ratio(inputs['centre10'][0], flat_train, train_anatomy, inputs['centre10'][1],
                                                val_anatomy, args.knn)
        keep = common.flatten()
        predictability = {name: deviation_correlation(p[:, keep], flat_val[:, keep], val_mean[:, keep])
                          for name, p in predictions.items()}
        report['by_sigma'][f'{sigma:g}'] = {'predictability': predictability, 'ridge': ridge_report}
        print(f'sigma {sigma:g}', json.dumps(report['by_sigma'][f'{sigma:g}']), flush=True)
        for name, prediction in predictions.items():
            probability = (val_reference[:, 0].double() * prediction.view_as(val_r).exp()).clamp(0, 1).float()
            arms[f'{name}_sigma{sigma:g}'] = logit_of(probability.unsqueeze(1))

    psnr = {name: [] for name in arms}
    volumes = []
    indices = val['indices'].numpy()
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, args.chunk)):
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
    control = psnr['anatomy_mean_label']
    summary = {}
    for name in arms:
        gain = psnr[name] - control
        summary[name] = {'mean_psnr': float(psnr[name].mean()), 'vs_anatomy_mean': bootstrap_ci(gain, volumes, rng),
                         'by_protocol': {p: float(gain[val_protocol == p].mean()) for p in PROTOCOLS}}
    report['psnr'] = summary
    print(json.dumps(summary, indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report['config'] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
