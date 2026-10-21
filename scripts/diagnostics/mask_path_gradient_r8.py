"""Why is the CG-SENSE R8 knee-PDFS LOUPE mask worse than the brain mask on its own training slices?

Every mask row of the 100-epoch runs ends at the Adam noise floor (exp_avg^2 / exp_avg_sq ~ 0.05), so each row sits
where the trainer's expected gradient balances out. This asks whether that balance point is a real minimum of the
training loss:

  path   z(t) = z_own + t D, z = slope * logits, scored with the hard Bernoulli masks the trainer draws (common random
         numbers across t), per-slice training L1 and trainer-path PSNR. `--direction full`: D = z_target - z_own,
         which also blends the two pixel layouts. `--direction density`: D is a per-ring shift (`RING_PX`-wide rings
         outside the calibration centre) solved so that z(1) keeps z_own's pixel layout at z_target's radial density
         (ring-mean logits would not do it: the scores are saturated)
  slope  d/dt L at t = 0 two ways: the trainer's straight-through estimator <dL/dz, z_target - z_own>, and a
         least-squares fit of the hard-mask loss over the small-t points
  snr    |E g|^2 (split-half cross product, free of noise bias) against the variance of a training-batch gradient
         with `--per-batch` slices of this class

A monotone fall of L(t) with an estimator slope near zero or positive means the estimator is biased at this point; a
rise before the fall means a separate basin. The brain row along brain -> shared is the control.

    python scripts/diagnostics/mask_path_gradient_r8.py --cls knee-PDFS --target brain --direction density --per-batch 4
    python scripts/diagnostics/mask_path_gradient_r8.py --cls brain --target shared --direction density --per-batch 8 --stride 2
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.datasets.fastmri_data import condition_name
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import ScoreModel, load, per_sample_l1

RUN = Path('checkpoints/loupe2d_cgsense_r8_brain_knee_prototable_bilinear_const_100ep')
SHARED_RUN = Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep')
OUT_DIR = Path('analysis/anatcond_cgsense/metric_apodized')
TS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
SLOPE_TS = (0.0, 0.05, 0.1, 0.25)
RING_PX = 2
BOOTSTRAPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--cls', required=True, help='class whose row and training slices are probed')
    parser.add_argument('--target', required=True, help="another class of --run, or 'shared' for the shared run's mask")
    parser.add_argument('--direction', choices=('full', 'density'), required=True)
    parser.add_argument('--per-batch', type=float, required=True, help='slices of --cls in a 16-slice training batch')
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    return parser.parse_args()


def class_indices(dataset, cls, stride):
    """Training-set indices of the slices of `cls`, every stride-th, without loading k-space."""
    chosen, offset, cache = [], 0, {}
    for part in dataset.datasets:
        for position, (fname, _) in enumerate(part.examples):
            if fname not in cache:
                with h5py.File(fname, 'r') as data:
                    cache[fname] = condition_name(part.anatomy, dict(data.attrs))
            if cache[fname] == cls:
                chosen.append(offset + position)
        offset += len(part)
    return chosen[::stride]


def row_scores(model, index, slope):
    return model.sampler.mask_net.raw_logits(torch.tensor([index], device=next(model.parameters()).device)) * slope


def ring_sums(x, rings, weight):
    return torch.zeros(int(rings.max()) + 1, device=x.device, dtype=x.dtype).index_add_(0, rings, x.flatten() * weight)


def density_shift(z_own, z_target, outside_centre, iterations=60):
    """Per-ring score shift c(r) with sum sigmoid(z_own + c) = sum sigmoid(z_target) over every RING_PX-wide ring
    outside the calibration centre (bisection), broadcast to the map: z_own's layout at z_target's radial density."""
    height, width = z_own.shape
    ky = torch.arange(height, device=z_own.device) - height // 2
    kx = torch.arange(width, device=z_own.device) - width // 2
    rings = (torch.hypot(ky[:, None].float(), kx[None, :].float()) / RING_PX).long().flatten()
    weight = outside_centre.flatten().to(z_own.dtype)
    wanted = ring_sums(torch.sigmoid(z_target), rings, weight)
    low, high = torch.full_like(wanted, -40.0), torch.full_like(wanted, 40.0)
    for _ in range(iterations):
        middle = (low + high) / 2
        too_dense = ring_sums(torch.sigmoid(z_own + middle[rings].view(height, width)), rings, weight) > wanted
        high, low = torch.where(too_dense, middle, high), torch.where(too_dense, low, middle)
    return ((low + high) / 2)[rings].view(height, width) * outside_centre


def draw_mask(model, score_model, z, old_mask, valid_mask, seed):
    torch.manual_seed(seed)
    return model.sampler.select_locations(score_model.probability(z, old_mask), old_mask, model.budget, valid_mask)[1]


def volume_bootstrap(values, volumes, rng):
    unique = np.unique(volumes)
    per_volume = np.array([values[volumes == v].mean() for v in unique])
    draws = [per_volume[rng.randint(len(unique), size=len(unique))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(values.mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def main():
    args = parse_args()
    model, checkpoint = load(args.run, 'last_model.pt', args.device)
    score_model = ScoreModel(model)
    classes = list(checkpoint['options'].label_classes)
    own = classes.index(args.cls)
    z_own = row_scores(model, own, score_model.slope)
    if args.target == 'shared':
        shared, _ = load(SHARED_RUN, 'last_model.pt', args.device)
        z_target = row_scores(shared, 0, score_model.slope)
    else:
        z_target = row_scores(model, classes.index(args.target), score_model.slope)
    z_own, z_target = (z.view(1, 1, *z.shape[-2:]) for z in (z_own, z_target))

    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, _ = build_dataloaders(cfg)
    indices = class_indices(train_loader.dataset, args.cls, args.stride)
    loader = DataLoader(Subset(train_loader.dataset, indices), batch_size=args.batch, shuffle=False, num_workers=4,
                        collate_fn=train_loader.collate_fn)
    print(f'{len(indices)} {args.cls} training slices', flush=True)
    probe = next(iter(DataLoader(Subset(train_loader.dataset, indices[:1]), batch_size=1, collate_fn=train_loader.collate_fn)))
    outside_centre = 1 - model._initial_masks(probe['kspace'].to(args.device))[0][0, 0]
    direction = (z_target - z_own) * outside_centre
    if args.direction == 'density':
        direction = density_shift(z_own[0, 0], z_target[0, 0], outside_centre).view_as(direction)

    half = args.draws // 2
    loss = np.zeros((len(TS), len(indices), args.draws))
    psnr = np.zeros((len(TS), len(indices), args.draws))
    estimator_slope = np.zeros((len(indices), args.draws))
    grad_sum = {'a': torch.zeros_like(z_own[0, 0], dtype=torch.float64), 'b': torch.zeros_like(z_own[0, 0], dtype=torch.float64)}
    grad_sq_sum, volumes, position = 0.0, [], 0
    for batch in loader:
        kspace = batch['kspace'].to(args.device)
        part = {'kspace': kspace, 'target': batch['target'].to(args.device),
                'label': torch.full((len(kspace),), own, device=args.device)}
        metric_mask = metric_mask_from_real_kspace(kspace, batch['anatomy'])
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        if position == 0:
            score_model.check(part)
        old_mask, valid_mask = model._initial_masks(kspace)
        rows = slice(position, position + len(kspace))
        for d in range(args.draws):
            seed = indices[position] * 1009 + d
            with torch.no_grad():
                for k, t in enumerate(TS):
                    z = (z_own + t * direction).expand(len(kspace), -1, -1, -1)
                    mask = draw_mask(model, score_model, z, old_mask, valid_mask, seed)
                    output = model(part['target'], kspace, anatomy_labels=part['label'], fixed_mask=mask)['output']
                    loss[k, rows, d] = (output - target).abs().flatten(1).mean(1).cpu().numpy()
                    psnr[k, rows, d] = compute_psnr_torch(*_apply_signal_mask(output, target, metric_mask)).flatten().cpu().numpy()
            leaf = z_own.expand(len(kspace), -1, -1, -1).clone().requires_grad_(True)
            mask = draw_mask(model, score_model, leaf, old_mask, valid_mask, seed)
            per_sample_l1(model, part, mask).sum().backward()
            grad = leaf.grad[:, 0].double()
            estimator_slope[rows, d] = (grad * direction[0, 0].double()).flatten(1).sum(1).cpu().numpy()
            grad_sum['a' if d < half else 'b'] += grad.sum(0)
            grad_sq_sum += float((grad ** 2).sum())
        volumes += [Path(f).name for f in batch['fname']]
        position += len(kspace)
        print(f'{position}/{len(indices)}', flush=True)

    volumes = np.array(volumes)
    rng = np.random.RandomState(0)
    n = len(indices)
    mean_a, mean_b = grad_sum['a'] / (n * half), grad_sum['b'] / (n * (args.draws - half))
    signal = float((mean_a * mean_b).sum())
    single_variance = grad_sq_sum / (n * args.draws) - float(((mean_a + mean_b) / 2).pow(2).sum())
    batch_variance = single_variance / args.per_batch
    loss_mean, psnr_mean = loss.mean(2), psnr.mean(2)
    small = [TS.index(t) for t in SLOPE_TS]
    ts = np.array(SLOPE_TS)
    weights = (ts - ts.mean()) / ((ts - ts.mean()) ** 2).sum()
    fitted_slope = (loss_mean[small] * weights[:, None]).sum(0)
    result = {
        'run': str(args.run), 'class': args.cls, 'target': args.target, 'direction': args.direction, 'slices': n,
        'draws': args.draws,
        'path': {f'{t:g}': {'l1': volume_bootstrap(loss_mean[k], volumes, rng),
                            'psnr_vs_t0': volume_bootstrap(psnr_mean[k] - psnr_mean[0], volumes, rng)}
                 for k, t in enumerate(TS)},
        'slope_at_t0': {'hard_loss_fit': volume_bootstrap(fitted_slope, volumes, rng),
                        'straight_through': volume_bootstrap(estimator_slope.mean(1), volumes, rng)},
        'gradient': {'mean_energy': signal, 'single_slice_variance': single_variance,
                     'per_batch': args.per_batch, 'batch_snr': signal / batch_variance,
                     'cosine_mean_gradient_direction': float(((mean_a + mean_b) / 2 * direction[0, 0].double()).sum()
                                                            / (((mean_a + mean_b) / 2).norm() * direction.double().norm()))},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'mask_path_{args.cls}_to_{args.target}_{args.direction}'
    (args.out_dir / f'{stem}.json').write_text(json.dumps(result, indent=1) + '\n')
    np.savez(args.out_dir / f'{stem}.npz', loss=loss, psnr=psnr, estimator_slope=estimator_slope, volume=volumes, ts=np.array(TS))
    for t, row in result['path'].items():
        print(f"t={t:>4s}  L1 {row['l1']['mean']:.5f}  PSNR vs t0 {row['psnr_vs_t0']['mean']:+.3f} "
              f"[{row['psnr_vs_t0']['ci95'][0]:+.3f}, {row['psnr_vs_t0']['ci95'][1]:+.3f}]")
    for name, row in result['slope_at_t0'].items():
        print(f"dL/dt at t=0, {name}: {row['mean']:+.3e} [{row['ci95'][0]:+.3e}, {row['ci95'][1]:+.3e}]")
    print(json.dumps(result['gradient'], indent=1))


if __name__ == '__main__':
    main()
