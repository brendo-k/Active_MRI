"""Can the 1-step conv_2d sampler fit per-slice masks, or is the model what keeps it from adapting?

Fixed training slices (both anatomies, spread over support size), VarNet and sensitivity net frozen. Every
arm optimises its masks with the trainer's estimator (`select_locations`: budget rescale, Bernoulli draw,
straight-through gradient with the checkpoint's floors) against the checkpoint's reconstruction loss:

  net      the conv sampler's own weights, fine-tuned from the checkpoint (the model under test)
  shared   one free score map for every slice, started at the mean of the sampler's scores
  slice    a free score map per slice, started at the sampler's own scores for it (the oracle)

Reading: net close to slice -> the model can fit per-slice masks, so generalisation or the input limits it;
net close to shared while slice is clearly better -> the model or its optimisation limits it; slice close to
shared -> adaptivity is worth little on these slices under this VarNet.

PSNR is the trainer's masked path (`_apply_signal_mask`), averaged over fixed seeded mask draws that every arm
shares. Held-out slices come from the validation set. Adaptivity of an arm is the mean correlation of a slice's
effective probability map with its anatomy's mean map (1 = one mask for all).

    python scripts/diagnostics/sampler_fit_capacity_r8.py
"""
import argparse
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
from activemri.training.trainer import _apply_signal_mask, _signal_mask, build_model
from activemri.utils.util import compute_psnr_torch

RUN = Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05')
OUT = Path('reports/conv_vs_loupe_r8/sampler_fit_capacity.json')
ANATOMIES = ('brain', 'knee')
LOSS_WEIGHTS = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
POOL_BATCHES = 12
SCORING_SEEDS = (0, 1, 2, 3)
PROBABILITY_EPS = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--per-anatomy', type=int, default=16)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--chunk', type=int, default=4, help='slices per forward/backward pass')
    parser.add_argument('--net-lrs', type=float, nargs='+', default=[1e-3, 5e-3])
    parser.add_argument('--free-lr', type=float, default=0.05)
    parser.add_argument('--trace-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_frozen(device):
    checkpoint = torch.load(RUN / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint['options'].loss_type


def support_fraction(target):
    return _signal_mask(target if target.dim() == 4 else target.unsqueeze(1)).flatten(1).float().mean(1)


def pick_slices(loader, per_anatomy, rng):
    """`per_anatomy` slices of each anatomy at evenly spaced support quantiles, from a random subset of the loader's batches."""
    chosen_batches = set(rng.choice(len(loader), min(POOL_BATCHES, len(loader)), replace=False).tolist())
    pool = {a: [] for a in ANATOMIES}
    for index, batch in enumerate(loader):
        if index not in chosen_batches:
            continue
        support = support_fraction(batch['target'])
        labels = batch.get('anatomy_labels')
        for row, anatomy in enumerate(batch['anatomy']):
            pool[anatomy].append({'kspace': batch['kspace'][row], 'target': batch['target'][row],
                                  'label': labels[row] if torch.is_tensor(labels) else None,
                                  'support': float(support[row]), 'anatomy': anatomy})
    selected = []
    for anatomy in ANATOMIES:
        ordered = sorted(pool[anatomy], key=lambda s: s['support'])
        positions = np.linspace(0, len(ordered) - 1, per_anatomy).round().astype(int)
        selected += [ordered[p] for p in positions]
    return stack(selected)


def stack(slices):
    labels = [s['label'] for s in slices]
    return {'kspace': torch.stack([s['kspace'] for s in slices]), 'target': torch.stack([s['target'] for s in slices]),
            'label': torch.stack(labels) if all(l is not None for l in labels) else None,
            'anatomy': np.array([s['anatomy'] for s in slices]), 'support': [s['support'] for s in slices]}


def chunk_of(data, positions, device):
    return {key: (value[positions].to(device) if torch.is_tensor(value) else None) for key, value in data.items()
            if key in ('kspace', 'target', 'label')}


def positions_in_chunks(count, chunk):
    return list(torch.arange(count).split(chunk))


def loss_of(model, loss_type, part, mask):
    output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)
    target_dict = {'target': part['target'], 'kspace': part['kspace']}
    if part['label'] is not None:
        target_dict['anatomy_labels'] = part['label']
    return model.loss(output, target_dict, LOSS_WEIGHTS, loss_type)[0]


def masks_from_logits(model, logits, part):
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    _, mask = model.sampler.select_locations(scores_to_prob(logits), old_mask, model.budget, valid_mask)
    return mask


def proposal(model, part):
    return model.propose_mask(part['kspace'], anatomy_labels=part['label'])


@torch.no_grad()
def initial_scores(model, data, device, chunk):
    return torch.cat([proposal(model, chunk_of(data, p, device))['unscaled'].cpu()
                      for p in positions_in_chunks(len(data['kspace']), chunk)])


def logit_of(probability):
    return torch.logit(probability.clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS))


@torch.no_grad()
def score(model, data, mask_fn, device, chunk):
    """Per-slice PSNR averaged over the shared seeded draws, and the effective probability map of each slice."""
    psnr = torch.zeros(len(data['kspace']))
    effective = []
    for seed in SCORING_SEEDS:
        for index, positions in enumerate(positions_in_chunks(len(data['kspace']), chunk)):
            part = chunk_of(data, positions, device)
            torch.manual_seed(seed * 1000 + index)
            mask, probability = mask_fn(part, positions)
            output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
            masked_output, masked_target = _apply_signal_mask(output, part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1))
            psnr[positions] += compute_psnr_torch(masked_output, masked_target).flatten().cpu() / len(SCORING_SEEDS)
            if seed == SCORING_SEEDS[0]:
                effective.append(probability.cpu())
    return psnr, torch.cat(effective)


def adaptivity(effective, anatomies):
    correlations = []
    for anatomy in ANATOMIES:
        maps = effective[torch.as_tensor(anatomies == anatomy)].numpy()
        centre = maps.mean(0)
        correlations += [np.corrcoef(m.ravel(), centre.ravel())[0, 1] for m in maps]
    return float(np.mean(correlations))


def summarise(psnr, effective, data):
    by_anatomy = {a: float(psnr[torch.as_tensor(data['anatomy'] == a)].mean()) for a in ANATOMIES}
    return {'psnr': float(psnr.mean()), 'psnr_by_anatomy': by_anatomy, 'slice_vs_mean_corr': adaptivity(effective, data['anatomy'])}


def native_mask_fn(model):
    def mask_fn(part, positions):  # noqa: ARG001
        result = proposal(model, part)
        return result['mask'], torch.clamp(result['rescaled'] + result['old_mask'], max=1)[:, 0]
    return mask_fn


def logits_mask_fn(model, logits_for):
    def mask_fn(part, positions):
        old_mask, valid_mask = model._initial_masks(part['kspace'])
        rescaled, mask = model.sampler.select_locations(scores_to_prob(logits_for(positions)), old_mask, model.budget, valid_mask)
        return mask, torch.clamp(rescaled + old_mask, max=1)[:, 0]
    return mask_fn


def free_logit_arm(model, loss_type, train, init_logits, per_slice, args):
    """Optimise free score maps: one per training slice, or a single shared one. Returns the PSNR trace and the maps."""
    logits = init_logits.clone().to(args.device).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=args.free_lr)
    count = len(train['kspace'])

    def logits_for(positions):
        return logits[positions.to(args.device)] if per_slice else logits.expand(len(positions), -1, -1, -1)

    trace = {}
    for step in range(args.steps + 1):
        if step % args.trace_every == 0:
            with torch.no_grad():
                trace[step] = summarise(*score(model, train, logits_mask_fn(model, logits_for), args.device, args.chunk), train)
        if step == args.steps:
            break
        optimizer.zero_grad()
        for positions in positions_in_chunks(count, args.chunk):
            part = chunk_of(train, positions, args.device)
            loss = loss_of(model, loss_type, part, masks_from_logits(model, logits_for(positions), part))
            (loss * len(positions) / count).backward()
        optimizer.step()
    return trace, logits_for


def net_arm(model, loss_type, train, lr, args):
    """Fine-tune the conv sampler's own weights through the full pipeline; returns the PSNR trace."""
    parameters = list(model.sampler.mask_net.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    count = len(train['kspace'])
    trace = {}
    for step in range(args.steps + 1):
        if step % args.trace_every == 0:
            with torch.no_grad():
                trace[step] = summarise(*score(model, train, native_mask_fn(model), args.device, args.chunk), train)
        if step == args.steps:
            break
        optimizer.zero_grad()
        for positions in positions_in_chunks(count, args.chunk):
            part = chunk_of(train, positions, args.device)
            (loss_of(model, loss_type, part, None) * len(positions) / count).backward()
        optimizer.step()
    for parameter in parameters:
        parameter.requires_grad_(False)
    return trace


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model, loss_type = load_frozen(args.device)

    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    train = pick_slices(train_loader, args.per_anatomy, rng)
    held_out = pick_slices(val_loader, args.per_anatomy, rng)
    print(f"train support {np.round(train['support'], 2).tolist()}", flush=True)

    def native_summary(data):
        return summarise(*score(model, data, native_mask_fn(model), args.device, args.chunk), data)

    results = {'reference_native': {'train': native_summary(train), 'held_out': native_summary(held_out)}}
    print('native', json.dumps(results['reference_native']), flush=True)

    starting_weights = {k: v.clone() for k, v in model.sampler.mask_net.state_dict().items()}
    start_scores = initial_scores(model, train, args.device, args.chunk)
    arms = {'shared': (False, logit_of(start_scores.mean(0, keepdim=True))), 'slice': (True, logit_of(start_scores))}
    for name, (per_slice, init) in arms.items():
        trace, logits_for = free_logit_arm(model, loss_type, train, init, per_slice, args)
        results[name] = {'train_trace': trace}
        if not per_slice:
            with torch.no_grad():
                results[name]['held_out'] = summarise(
                    *score(model, held_out, logits_mask_fn(model, logits_for), args.device, args.chunk), held_out)
        print(name, json.dumps(results[name]), flush=True)

    for lr in args.net_lrs:
        model.sampler.mask_net.load_state_dict(starting_weights)
        trace = net_arm(model, loss_type, train, lr, args)
        results[f'net_lr{lr:g}'] = {'train_trace': trace, 'held_out': native_summary(held_out)}
        print(f'net_lr{lr:g}', json.dumps(results[f'net_lr{lr:g}']), flush=True)

    results['config'] = {'steps': args.steps, 'per_anatomy': args.per_anatomy, 'free_lr': args.free_lr,
                         'net_lrs': args.net_lrs, 'loss_type': loss_type,
                         'train_support': train['support'], 'held_out_support': held_out['support']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
