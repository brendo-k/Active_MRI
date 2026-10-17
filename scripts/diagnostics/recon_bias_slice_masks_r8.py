"""Do per-slice masks fitted through a frozen reconstructor carry information that transfers, or exploit that reconstructor?

Same adjacent-slice pairs as mask_gradient_anatomy_r8.py (training volumes, same seed). Through a frozen
reconstructor R, every map starts from one prior (the conv run's population map, prior.pt of oracle_labels_r8.py)
and is fitted with the oracle-label estimator (the conv sampler's budget rescale, Bernoulli draw and floored
straight-through backward; Adam at `--lr` for `--steps`, `--draws` draws per step):

  shared   one free score map per anatomy, fitted on all its slices (the population refit through R)
  slice    one free score map per slice (the per-slice oracle through R)

Each slice map is then scored through R on its own slice, on its neighbour (the adjacent slice of the same
volume) and on `--others` slices of other volumes of the same anatomy, as a gain over the shared map on the
target slice. PSNR is the trainer's masked metric over common seeded draws. Confidence intervals bootstrap pairs
(one pair per volume).

    python scripts/diagnostics/recon_bias_slice_masks_r8.py --tag conv_last
    python scripts/diagnostics/recon_bias_slice_masks_r8.py --tag random --recon-run checkpoints/random_mask_varnet_r8_brain_knee
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import ANATOMIES, CONV_RUN, OUT_DIR, REPORT_DIR, adjacent_pairs, load, to_device

PRIOR = Path('/tmp/bkadota_oracle_labels_r8/prior.pt')
SCORING_SEEDS = (0, 1, 2, 3)
BOOTSTRAPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--recon-run', type=Path, default=CONV_RUN)
    parser.add_argument('--recon-ckpt', default='last_model.pt')
    parser.add_argument('--pairs-per-anatomy', type=int, default=12)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--others', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def stack(loader, device):
    parts = [to_device(b, device) for b in loader]
    anatomy = sum((list(b['anatomy']) for b in loader), [])
    fnames = sum((list(b['fname']) for b in loader), [])
    out = {k: torch.cat([p[k] for p in parts]) if parts[0][k] is not None else None for k in parts[0]}
    return out, np.array(anatomy), np.array(fnames)


def rows_of(data, rows):
    index = torch.as_tensor(rows, device=data['kspace'].device)
    return {k: (v[index] if v is not None else None) for k, v in data.items()}


def masks(estimator, logits, kspace):
    old_mask, valid_mask = estimator._initial_masks(kspace)
    return estimator.sampler.select_locations(scores_to_prob(logits), old_mask, estimator.budget, valid_mask)


def l1(recon, part, mask):
    output = recon(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    return (output - target).abs().mean()


def fit(recon, estimator, data, init, assignment, args):
    """Free score maps (one per row of `init`); slice r uses map assignment[r]. Loss averaged over all slices."""
    logits = init.clone().to(args.device).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    count = len(assignment)
    for _ in range(args.steps):
        optimizer.zero_grad()
        for rows in torch.arange(count).split(args.chunk):
            part = rows_of(data, rows.tolist())
            many = {k: (v.repeat(args.draws, *([1] * (v.dim() - 1))) if v is not None else None) for k, v in part.items()}
            chosen = logits[torch.as_tensor(assignment)[rows].to(args.device)].repeat(args.draws, 1, 1, 1)
            loss = l1(recon, many, masks(estimator, chosen, many['kspace'])[1])
            (loss * len(rows) / count).backward()
        optimizer.step()
    return logits.detach()


@torch.no_grad()
def score(recon, estimator, data, pairs, maps):
    """PSNR of (map index, target slice) pairs over common seeded draws: a draw depends only on (seed, target, pair position)."""
    psnr = torch.zeros(len(pairs))
    for seed in SCORING_SEEDS:
        for start in range(0, len(pairs), 8):
            chunk = pairs[start:start + 8]
            part = rows_of(data, [t for _, t in chunk])
            logits = maps[torch.as_tensor([m for m, _ in chunk])].to(part['kspace'].device)
            old_mask, valid_mask = estimator._initial_masks(part['kspace'])
            probability = estimator.sampler.rescale(scores_to_prob(logits), estimator.budget / logits[0, 0].numel(),
                                                    (1 - old_mask) * valid_mask)
            generator = torch.Generator(device='cpu')
            noise = []
            for _, target in chunk:
                generator.manual_seed(seed * 100003 + target)
                noise.append(torch.rand(probability.shape[1:], generator=generator))
            mask = torch.clamp(old_mask + (probability > torch.stack(noise).to(probability.device)).float() * valid_mask, max=1)
            output = recon(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
            target_image = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
            masked_output, masked_target = _apply_signal_mask(output, target_image)
            psnr[start:start + len(chunk)] += compute_psnr_torch(masked_output, masked_target).flatten().cpu() / len(SCORING_SEEDS)
    return psnr


def effective(estimator, maps, data):
    rescaled, _ = masks(estimator, maps.to(data['kspace'].device), data['kspace'])
    old_mask, _ = estimator._initial_masks(data['kspace'])
    return torch.clamp(rescaled + old_mask, max=1)[:, 0].cpu()


def bootstrap(values, groups, rng):
    unique = np.unique(groups)
    per_group = np.array([values[groups == g].mean() for g in unique])
    draws = [per_group[rng.randint(len(per_group), size=len(per_group))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(per_group.mean()), 'ci95': np.percentile(draws, [2.5, 97.5]).tolist()}


def correlation(x, y):
    x, y = x.flatten(1) - x.flatten(1).mean(1, keepdim=True), y.flatten(1) - y.flatten(1).mean(1, keepdim=True)
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-30)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    estimator, _ = load(CONV_RUN, 'last_model.pt', args.device)
    same = args.recon_run == CONV_RUN and args.recon_ckpt == 'last_model.pt'
    recon = estimator if same else load(args.recon_run, args.recon_ckpt, args.device)[0]

    cfg = OmegaConf.load(CONV_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, _ = build_dataloaders(cfg)
    indices = adjacent_pairs(train_loader.dataset, args.pairs_per_anatomy, rng)
    loader = DataLoader(Subset(train_loader.dataset, indices), batch_size=8, shuffle=False, num_workers=4,
                        collate_fn=train_loader.collate_fn)
    data, anatomy, fnames = stack(loader, args.device)
    count = len(anatomy)
    neighbour = np.array([i + 1 if i % 2 == 0 else i - 1 for i in range(count)])
    pair_id = np.arange(count) // 2

    prior = torch.load(PRIOR).to(args.device)
    anatomy_index = np.array([ANATOMIES.index(a) for a in anatomy])
    shared = fit(recon, estimator, data, prior.repeat(len(ANATOMIES), 1, 1, 1), anatomy_index, args)
    print('shared fitted', flush=True)
    slice_maps = fit(recon, estimator, data, prior.repeat(count, 1, 1, 1), np.arange(count), args)
    print('slice maps fitted', flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'shared': shared.cpu(), 'slice_maps': slice_maps.half().cpu(), 'anatomy': anatomy, 'fnames': fnames,
                'indices': indices}, OUT_DIR / f'slice_masks_{args.tag}_maps.pt')

    others = np.array([rng.choice(np.flatnonzero((anatomy == anatomy[i]) & (fnames != fnames[i])), args.others, replace=False)
                       for i in range(count)])
    targets = {'own': np.arange(count)[:, None], 'neighbour': neighbour[:, None], 'other': others}
    all_maps = torch.cat([prior, shared, slice_maps])
    prior_row, shared_row, slice_row = 0, 1, 1 + len(ANATOMIES)
    results = {'config': {'recon': f'{args.recon_run}/{args.recon_ckpt}', 'slices': count, 'steps': args.steps,
                          'draws': args.draws, 'lr': args.lr, 'prior': str(PRIOR)}}
    per_slice = {}
    for name, target_rows in targets.items():
        pairs_slice = [(slice_row + i, int(t)) for i in range(count) for t in target_rows[i]]
        pairs_shared = [(shared_row + anatomy_index[t], int(t)) for i in range(count) for t in target_rows[i]]
        pairs_prior = [(prior_row, int(t)) for i in range(count) for t in target_rows[i]]
        width = target_rows.shape[1]
        psnr_slice = score(recon, estimator, data, pairs_slice, all_maps).view(count, width).mean(1).numpy()
        psnr_shared = score(recon, estimator, data, pairs_shared, all_maps).view(count, width).mean(1).numpy()
        psnr_prior = score(recon, estimator, data, pairs_prior, all_maps).view(count, width).mean(1).numpy()
        per_slice[name] = {'slice': psnr_slice, 'shared': psnr_shared, 'prior': psnr_prior}
        results[name] = {'slice_minus_shared_db': bootstrap(psnr_slice - psnr_shared, pair_id, rng),
                         'shared_minus_prior_db': bootstrap(psnr_shared - psnr_prior, pair_id, rng),
                         'slice_minus_prior_db': bootstrap(psnr_slice - psnr_prior, pair_id, rng),
                         'by_anatomy_slice_minus_shared_db': {a: float((psnr_slice - psnr_shared)[anatomy == a].mean())
                                                              for a in ANATOMIES},
                         'psnr_shared': float(psnr_shared.mean())}
        print(name, json.dumps(results[name]), flush=True)

    p_slice, p_shared = effective(estimator, slice_maps, data), effective(estimator, shared[anatomy_index], data)
    p_prior = effective(estimator, prior.expand(count, -1, -1, -1), data)
    deviation = p_slice - p_shared
    results['masks'] = {
        'slice_vs_shared_corr': float(correlation(p_slice, p_shared).mean()),
        'shared_vs_prior_corr': float(correlation(p_shared, p_prior).mean()),
        'neighbour_deviation_corr': float(correlation(deviation, deviation[neighbour]).mean()),
        'other_deviation_corr': float(np.mean([correlation(deviation[[i]], deviation[[o]]).item()
                                               for i in range(count) for o in others[i]])),
        'mean_abs_deviation': float(deviation.abs().mean())}
    print('masks', json.dumps(results['masks']), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'shared': shared.cpu(), 'slice_maps': slice_maps.half().cpu(), 'anatomy': anatomy, 'fnames': fnames,
                'per_slice': per_slice, 'indices': indices}, OUT_DIR / f'slice_masks_{args.tag}.pt')
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f'slice_masks_{args.tag}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
