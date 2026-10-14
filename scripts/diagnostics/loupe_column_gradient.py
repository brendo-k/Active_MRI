"""Does the validation loss itself favour the learned mask's column streaks?

For a trained per-anatomy LOUPE run, accumulates dLoss/dp(k) over the validation set (straight-through
gradient through binarisation and budget rescaling, divided by slope * p * (1 - p) to move from logits to
probabilities) and sums it over rows into a per-column gradient. A negative value means raising that column's
probability lowers the loss.

Two mask states are probed:
  trained     the learned mask as is (near a stationary point, gradients are small)
  destreaked  each column rescaled to remove its ripple (13-column moving average), reconstructor unchanged

The gradient is split by volume into two halves, so reproducibility between halves is the noise floor.

    CUDA_VISIBLE_DEVICES=0 python scripts/diagnostics/loupe_column_gradient.py --anatomy brain --seed 0
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.utils.phase_scramble import scramble_phase_real
from plot_loupe_anatomy_recon import CONFIG_NUM_COILS, load_model

LOSS_META = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
SMOOTH_WIDTH = 13
COLUMNS = slice(8, 120)
RNG_SEED = 1234


def detrend(profile):
    return profile - uniform_filter1d(profile, SMOOTH_WIDTH, mode='nearest')


def set_probability(param, probability, slope):
    q = np.clip(probability, 1e-6, 1 - 1e-6)
    with torch.no_grad():
        param.copy_(torch.from_numpy(np.log(q / (1 - q)) / slope).to(param.dtype).reshape(param.shape).to(param.device))


def destreaked(probability):
    column = probability.mean(axis=0)
    factor = np.ones_like(column)
    active = column > 1e-3
    factor[active] = uniform_filter1d(column, 21, mode='nearest')[active] / column[active]
    return probability * factor[None, :]


def probability_gradient(model, loader, device, scramble):
    """Per-half sum of dL/dlogit over slices and per-half slice counts, split by batch parity."""
    param = model.sampler.mask_net.mask
    sums = [torch.zeros_like(param), torch.zeros_like(param)]
    counts = [0, 0]
    torch.manual_seed(RNG_SEED)
    generator = torch.Generator(device=device).manual_seed(RNG_SEED)
    for i, batch in enumerate(loader):
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        if scramble[0] > 0:
            kspace = scramble_phase_real(kspace, scramble[0], scramble[1], generator)
        model.zero_grad()
        pred = model(target, kspace)
        loss, _ = model.loss(pred, {'target': target, 'kspace': kspace}, LOSS_META, 'l1')
        loss.backward()
        sums[i % 2] += param.grad.detach() * target.shape[0]
        counts[i % 2] += target.shape[0]
    return [(s / c).squeeze().cpu().numpy() for s, c in zip(sums, counts)]


def column_gradient(logit_gradient, probability, slope):
    d_prob = logit_gradient / (slope * probability * (1 - probability) + 1e-12)
    d_prob = np.where(probability > 1e-3, d_prob, 0.0)
    return d_prob.sum(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/loupe_anatomy'))
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomy', default='brain')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--scramble-std', type=float, default=0.0, help='phase-scramble the validation k-space (radians)')
    parser.add_argument('--scramble-band', type=float, nargs=2, default=(6.0, 16.0))
    parser.add_argument('--states', nargs='+', default=['trained', 'destreaked'])
    args = parser.parse_args()
    device = torch.device(args.device)
    scramble = (args.scramble_std, tuple(args.scramble_band))
    tag = f'_scramble{args.scramble_std:g}' if args.scramble_std > 0 else ''

    _, loader = build_fastmri_dataloaders(
        data_root=args.data_root, anatomies=[args.anatomy], image_shape=[256, 256], batch_size=args.batch_size,
        num_coils=CONFIG_NUM_COILS[args.anatomy], num_workers=args.num_workers, contrast='T2')
    model, _ = load_model(args.root / args.anatomy / f'seed_{args.seed}', device)
    slope = 10.0
    param = model.sampler.mask_net.mask
    trained_logits = param.detach().clone()
    trained_probability = torch.sigmoid(slope * trained_logits).squeeze().cpu().numpy()
    ripple = detrend(trained_probability.mean(axis=0))

    results = {}
    for state in args.states:
        if state == 'trained':
            probability = trained_probability
            with torch.no_grad():
                param.copy_(trained_logits)
        else:
            probability = destreaked(trained_probability)
            set_probability(param, probability, slope)
        halves = [column_gradient(g, probability, slope) for g in probability_gradient(model, loader, device, scramble)]
        detrended = [detrend(h) for h in halves]
        mean_gradient = detrend(np.mean(halves, axis=0))
        corr = lambda a, b: float(np.corrcoef(a[COLUMNS], b[COLUMNS])[0, 1])
        results[state] = {
            'split_half_reproducibility_r': corr(*detrended),
            'r_gradient_vs_trained_ripple': corr(mean_gradient, ripple),
            'r_gradient_vs_trained_ripple_negated': -corr(mean_gradient, ripple),
        }
        np.save(args.root / 'analysis' / f'column_gradient_{args.anatomy}_seed{args.seed}_{state}{tag}.npy', np.array(halves))
        print(f'{state:11s} split-half r {results[state]["split_half_reproducibility_r"]:+.2f}   '
              f'r(gradient, mask ripple) {results[state]["r_gradient_vs_trained_ripple"]:+.2f}  '
              f'(negative = loss wants more probability where the mask has streak peaks)', flush=True)
    (args.root / 'analysis' / f'column_gradient_{args.anatomy}_seed{args.seed}{tag}.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
