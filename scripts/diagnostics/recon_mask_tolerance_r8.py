"""How much does a frozen reconstructor lose when masks deviate from the distribution it was trained with, and how
much of that is mismatch (recovered by training on the new masks) rather than information? (R=8, brain+knee)

Base masks: each slice's own score map z from a sampler run, through the conv sampler's estimator (sigmoid, budget
rescale, Bernoulli draw). Two perturbations, both keeping the budget:

  reshape   z + sigma * f_i, f_i a smooth Gaussian random field per slice (std 1, length `--field-length` px): a
            per-slice change of the density, the kind of change an adaptive sampler makes. Its size is reported as
            the mean correlation of a slice's perturbed probability map with its unperturbed one.
  ring      a fraction of the drawn non-centre samples moved to unsampled locations of the same radius ring
            (radial density kept exactly; angular arrangement randomised)

`tolerance`: every reconstructor scores the same perturbed masks (common random numbers). `finetune`: a copy of
one reconstructor (VarNet and sensitivity net) is trained for `--ft-steps` on training slices with masks at one
reshape level, and a control copy on unperturbed masks; both are scored on unperturbed and perturbed val masks.

    python scripts/diagnostics/recon_mask_tolerance_r8.py tolerance
    python scripts/diagnostics/recon_mask_tolerance_r8.py finetune --sigma 1.5
"""
import argparse
import copy
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import CONV_RUN, REPORT_DIR, ScoreModel, load, radius_map, to_device

LOUPE_RUN = Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05')
RANDOM_RUN = Path('checkpoints/random_mask_varnet_r8_brain_knee')
RECONS = {  # name: (reconstructor run, checkpoint, sampler run whose masks are its base, checkpoint)
    'conv_last': (CONV_RUN, 'last_model.pt', CONV_RUN, 'last_model.pt'),
    'conv_ep4': (CONV_RUN, 'resume_from_epoch4.pt', CONV_RUN, 'resume_from_epoch4.pt'),
    'loupe_last': (LOUPE_RUN, 'last_model.pt', LOUPE_RUN, 'last_model.pt'),
    'random': (RANDOM_RUN, 'last_model.pt', CONV_RUN, 'last_model.pt'),
}
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.5)
RING_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
SCORING_SEEDS = (0, 1)
BOOTSTRAPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('tolerance', 'finetune'))
    parser.add_argument('--val-stride', type=int, default=8)
    parser.add_argument('--field-length', type=float, default=8.0)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--ft-recon', default='conv_last')
    parser.add_argument('--ft-steps', type=int, default=1000)
    parser.add_argument('--ft-batch', type=int, default=8)
    parser.add_argument('--ft-lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def smooth_field(shape, length, generator, device):
    """[B, 1, H, W] Gaussian random field with unit std and Gaussian correlation of std `length` px."""
    noise = torch.randn(shape, generator=generator).to(device)
    radius = int(3 * length)
    grid = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-grid ** 2 / (2 * length ** 2))
    kernel = kernel / kernel.sum()
    field = F.conv2d(noise, kernel.view(1, 1, 1, -1), padding=(0, radius))
    field = F.conv2d(field, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return field / field.flatten(1).std(1).view(-1, 1, 1, 1)


def ring_permute(mask, centre, valid, fraction, generator):
    """Move `fraction` of each ring's drawn non-centre samples to unsampled valid locations of the same ring."""
    device = mask.device
    mask, centre, valid = mask.cpu(), centre.cpu(), valid.cpu()
    radius = radius_map(mask.shape[-2:]).round().long()
    out = mask.clone()
    for b in range(len(mask)):
        free = (valid[b, 0] > 0) & (centre[b, 0] == 0)
        for r in torch.unique(radius[free]):
            ring = free & (radius == r)
            taken = torch.nonzero(ring & (mask[b, 0] > 0))
            empty = torch.nonzero(ring & (mask[b, 0] == 0))
            moves = min(int(round(fraction * len(taken))), len(empty))
            if moves == 0:
                continue
            leave = taken[torch.randperm(len(taken), generator=generator)[:moves]]
            enter = empty[torch.randperm(len(empty), generator=generator)[:moves]]
            out[b, 0, leave[:, 0], leave[:, 1]] = 0
            out[b, 0, enter[:, 0], enter[:, 1]] = 1
    return out.to(device)


class MaskMaker:
    """Masks from one sampler run's per-slice scores, through the conv run's estimator, with seeded perturbations."""

    def __init__(self, sampler_model, estimator):
        self.scores = ScoreModel(sampler_model)
        self.estimator = estimator

    def probabilities(self, part, sigma, length, generator):
        z = self.scores.scores(part)
        if sigma > 0:
            z = z + sigma * smooth_field(z.shape, length, generator, z.device)
        old_mask, valid_mask = self.estimator._initial_masks(part['kspace'])
        available = (1 - old_mask) * valid_mask
        rescaled = self.estimator.sampler.rescale(self.scores.probability(z, old_mask), self.estimator.budget / z[0, 0].numel(), available)
        return rescaled, old_mask, valid_mask

    def draw(self, part, sigma, ring_fraction, length, seed):
        generator = torch.Generator().manual_seed(seed)
        probability, old_mask, valid_mask = self.probabilities(part, sigma, length, generator)
        threshold = torch.rand(probability.shape, generator=generator).to(probability.device)
        new = (probability > threshold).float() * valid_mask
        if ring_fraction > 0:
            new = ring_permute(new, old_mask, valid_mask, ring_fraction, generator)
        return torch.clamp(old_mask + new, max=1), probability


@torch.no_grad()
def psnr_of(recon, part, mask):
    output = recon(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    return compute_psnr_torch(*_apply_signal_mask(output, target)).flatten().cpu()


def val_subset(stride):
    cfg = OmegaConf.load(CONV_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    indices = list(range(0, len(val_loader.dataset), stride))
    return train_loader, val_loader, indices


def bootstrap(delta, groups, rng):
    unique = np.unique(groups)
    per_group = np.array([delta[groups == g].mean() for g in unique])
    draws = [per_group[rng.randint(len(per_group), size=len(per_group))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(delta.mean()), 'ci95': np.percentile(draws, [2.5, 97.5]).tolist()}


def settings():
    return [('reshape', s, 0.0) for s in SIGMAS] + [('ring', 0.0, f) for f in RING_FRACTIONS]


def score_settings(recon, maker, loader, args):
    """{setting name: per-slice PSNR}, and the reshape levels' map correlation with the unperturbed map."""
    out = {f'{kind}_{sigma:g}_{fraction:g}': [] for kind, sigma, fraction in settings()}
    correlation = {f'{sigma:g}': [] for sigma in SIGMAS}
    for index, batch in enumerate(loader):
        part = to_device(batch, args.device)
        base, _, _ = maker.probabilities(part, 0.0, args.field_length, torch.Generator().manual_seed(0))
        for kind, sigma, fraction in settings():
            total = torch.zeros(len(part['kspace']))
            for seed in SCORING_SEEDS:
                mask, probability = maker.draw(part, sigma, fraction, args.field_length, seed * 100003 + index)
                total += psnr_of(recon, part, mask) / len(SCORING_SEEDS)
                if kind == 'reshape' and seed == SCORING_SEEDS[0]:
                    a, b = base.flatten(1), probability.flatten(1)
                    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
                    correlation[f'{sigma:g}'] += ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).cpu().tolist()
            out[f'{kind}_{sigma:g}_{fraction:g}'].append(total)
    return {k: torch.cat(v).numpy() for k, v in out.items()}, {k: float(np.mean(v)) for k, v in correlation.items()}


def tolerance(args):
    rng = np.random.RandomState(args.seed)
    _, val_loader, indices = val_subset(args.val_stride)
    loader = DataLoader(Subset(val_loader.dataset, indices), batch_size=args.chunk, shuffle=False, num_workers=4,
                        collate_fn=val_loader.collate_fn)
    meta = [(b['fname'], b['anatomy']) for b in loader]
    volumes = np.array(sum((list(f) for f, _ in meta), []))
    anatomy = np.array(sum((list(a) for _, a in meta), []))
    estimator, _ = load(CONV_RUN, 'last_model.pt', args.device)
    results = {'config': {'val_slices': len(indices), 'field_length': args.field_length, 'sigmas': SIGMAS,
                          'ring_fractions': RING_FRACTIONS, 'scoring_seeds': SCORING_SEEDS}}
    for name, (recon_run, recon_ckpt, sampler_run, sampler_ckpt) in RECONS.items():
        recon, _ = load(recon_run, recon_ckpt, args.device)
        sampler_model = recon if (sampler_run, sampler_ckpt) == (recon_run, recon_ckpt) else load(sampler_run, sampler_ckpt, args.device)[0]
        psnr, correlation = score_settings(recon, MaskMaker(sampler_model, estimator), loader, args)
        base = psnr['reshape_0_0']
        rows = {'base_psnr': float(base.mean()),
                'base_psnr_by_anatomy': {a: float(base[anatomy == a].mean()) for a in ('brain', 'knee')},
                'reshape_map_corr': correlation}
        for key, values in psnr.items():
            if key != 'reshape_0_0':
                rows[f'{key}_minus_base_db'] = bootstrap(values - base, volumes, rng)
        results[name] = rows
        print(name, json.dumps(rows), flush=True)
        del recon, sampler_model
        torch.cuda.empty_cache()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / 'tolerance.json').write_text(json.dumps(results, indent=2) + '\n')


def finetune(args):
    rng = np.random.RandomState(args.seed)
    train_loader, val_loader, indices = val_subset(args.val_stride)
    val = DataLoader(Subset(val_loader.dataset, indices), batch_size=args.chunk, shuffle=False, num_workers=4,
                     collate_fn=val_loader.collate_fn)
    volumes = np.array(sum((list(b['fname']) for b in val), []))
    estimator, _ = load(CONV_RUN, 'last_model.pt', args.device)
    recon_run, recon_ckpt, sampler_run, sampler_ckpt = RECONS[args.ft_recon]
    frozen, _ = load(recon_run, recon_ckpt, args.device)
    sampler_model = frozen if (sampler_run, sampler_ckpt) == (recon_run, recon_ckpt) else load(sampler_run, sampler_ckpt, args.device)[0]
    maker = MaskMaker(sampler_model, estimator)
    train = DataLoader(train_loader.dataset, batch_size=args.ft_batch, shuffle=True, num_workers=4,
                       collate_fn=train_loader.collate_fn, drop_last=True,
                       generator=torch.Generator().manual_seed(args.seed))

    def scored(recon):
        out = {}
        for sigma in (0.0, args.sigma):
            values = []
            for index, batch in enumerate(val):
                part = to_device(batch, args.device)
                total = torch.zeros(len(part['kspace']))
                for seed in SCORING_SEEDS:
                    mask, _ = maker.draw(part, sigma, 0.0, args.field_length, seed * 100003 + index)
                    total += psnr_of(recon, part, mask) / len(SCORING_SEEDS)
                values.append(total)
            out[f'{sigma:g}'] = torch.cat(values).numpy()
        return out

    def trained(sigma):
        recon = copy.deepcopy(frozen)
        recon.sampler.requires_grad_(False)
        parameters = [p for n, p in recon.named_parameters() if not n.startswith('sampler.')]
        for p in parameters:
            p.requires_grad_(True)
        recon.train()
        optimizer = torch.optim.Adam(parameters, lr=args.ft_lr)
        step, generator = 0, torch.Generator().manual_seed(args.seed + 1)
        while step < args.ft_steps:
            for batch in train:
                part = to_device(batch, args.device)
                with torch.no_grad():
                    mask, _ = maker.draw(part, sigma, 0.0, args.field_length, int(torch.randint(1 << 30, (1,), generator=generator)))
                output = recon(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
                target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
                loss = (output - target).abs().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                step += 1
                if step % 100 == 0:
                    print(f'sigma {sigma:g} step {step} loss {loss.item():.4f}', flush=True)
                if step >= args.ft_steps:
                    break
        recon.eval()
        recon.requires_grad_(False)
        return recon

    results = {'config': {'recon': args.ft_recon, 'sigma': args.sigma, 'steps': args.ft_steps, 'batch': args.ft_batch,
                          'lr': args.ft_lr, 'val_slices': len(indices), 'field_length': args.field_length}}
    arms = {'frozen': scored(frozen)}
    print('frozen', {k: float(v.mean()) for k, v in arms['frozen'].items()}, flush=True)
    for name, sigma in (('ft_unperturbed', 0.0), ('ft_perturbed', args.sigma)):
        recon = trained(sigma)
        arms[name] = scored(recon)
        print(name, {k: float(v.mean()) for k, v in arms[name].items()}, flush=True)
        del recon
        torch.cuda.empty_cache()
    s = f'{args.sigma:g}'
    results['psnr'] = {arm: {k: float(v.mean()) for k, v in values.items()} for arm, values in arms.items()}
    results['penalty_frozen_db'] = bootstrap(arms['frozen']['0'] - arms['frozen'][s], volumes, rng)
    results['penalty_after_adapting_db'] = bootstrap(arms['ft_unperturbed']['0'] - arms['ft_perturbed'][s], volumes, rng)
    results['mismatch_db'] = bootstrap((arms['frozen']['0'] - arms['frozen'][s]) - (arms['ft_unperturbed']['0'] - arms['ft_perturbed'][s]), volumes, rng)
    results['ft_perturbed_cost_on_unperturbed_db'] = bootstrap(arms['ft_unperturbed']['0'] - arms['ft_perturbed']['0'], volumes, rng)
    print(json.dumps({k: v for k, v in results.items() if k != 'psnr'}), flush=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f'finetune_{args.ft_recon}_sigma{args.sigma:g}.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    args = parse_args()
    torch.set_num_threads(4)  # the ring permutation is many small CPU ops; all-core threading thrashes on a busy box
    torch.manual_seed(args.seed)
    (tolerance if args.mode == 'tolerance' else finetune)(args)


if __name__ == '__main__':
    main()
