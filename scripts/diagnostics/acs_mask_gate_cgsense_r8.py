"""Gate for an ACS-conditioned mask on CG-SENSE (R=8): is there volume-specific mask gain for a 24 x 24 ACS to predict?

Reconstructor: CG-SENSE (10 iterations, lambda 0) with BART ecalib maps estimated from the 24 x 24 ACS (the dataset's
sense cache), so the maps are what the scanner has and the only learned part is the mask. Population model: `--run`, a
LOUPE shared mask with a preselected 24 x 24 centre, trained with those maps.

One mask per volume is the granularity a 3D (SPACE) scan allows. Each validation volume's slices, in order, are split
into blocks: F = the first half, H = the second half after `--gap` slices. Every arm is scored on H with seeded
Bernoulli draws shared by every arm (trainer-path background-masked PSNR):

  base           the population mask
  mask           a free per-volume logit map fitted on F through the frozen reconstructor
  mask_ctrl      the same fit on as many training slices of the same anatomy from other volumes: a fit's gain that
                 is not specific to the volume (refit gain, budget redistribution)
  mask_in        the same fit on F and H: the in-sample oracle of one mask for the whole scan
  mask*_s{sigma} a free map kept at scales above sigma px: the population probability times the map's Gaussian-
                 smoothed density ratio (density space, so no budget moves between rings)
  grid, grid_ctrl, grid_in
                 the population density times exp(an 8 x 8 bicubic log-density grid): a smooth per-volume density
                 change, the kind of output an ACS -> density predictor would produce

Decision rule, written before any run:
  Stage 1 (this script) stops the ACS line unless, in at least one anatomy, a smooth held-out arm (grid, mask_s4 or
  mask_s8) minus its control is >= +0.05 dB with a volume-bootstrap 95% interval above 0.
  Stage 2 (a 24 x 24 ACS predictor of that smooth change) runs only if Stage 1 passes. An ACS-conditioned head is
  built only if the predictor beats a static per-anatomy refit by >= +0.1 dB with its interval above 0, and coil-wise
  ACS features beat coil-combined ones (the case for CG-SENSE is that its g-factor depends on the coil maps).

    python scripts/diagnostics/acs_mask_gate_cgsense_r8.py --anatomy brain
    python scripts/diagnostics/acs_mask_gate_cgsense_r8.py --summarize
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
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import load
from oracle_blur_psnr_r8 import density_blurred
from subject_coadapt_ceiling_r8 import DensityFamily, control_slices, evenly, volume_groups

RUN = Path('checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24')
OUT_DIR = Path('reports/acs_gate_cgsense_r8')
LOGITS_DIR = Path('/tmp/bkadota_acs_gate')
LOSS_WEIGHTS = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
SCORING_SEEDS = (0, 1, 2, 3)
PROBABILITY_EPS = 1e-4
ANATOMY_INDEX = {'brain': 0, 'knee': 1}
FIT_ROLES = {'': 'fit', '_ctrl': 'ctrl', '_in': 'fit+score'}
SMOOTH_ARMS = ('grid', 'mask_s4', 'mask_s8')
BOOTSTRAPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anatomy', choices=tuple(ANATOMY_INDEX))
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--volumes', default=':', help="start:stop over the anatomy's volumes")
    parser.add_argument('--reverse', action='store_true', help='last volume first (a helper shard; finished volumes are skipped)')
    parser.add_argument('--gap', type=int, default=2)
    parser.add_argument('--max-fit', type=int, default=16)
    parser.add_argument('--max-score', type=int, default=16)
    parser.add_argument('--batch', type=int, default=4, help='slices per fitting step')
    parser.add_argument('--draws', type=int, default=2, help='Bernoulli draws per slice per fitting step')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--mask-lr', type=float, default=0.05)
    parser.add_argument('--grid-lr', type=float, default=0.02)
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--objective', choices=('train', 'psnr'), default='train',
                        help="fitting loss: the population run's training loss, or minus the scored masked PSNR")
    parser.add_argument('--sigmas', type=float, nargs='+', default=[4, 8])
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='')
    parser.add_argument('--exclude', nargs='*', default=['file_brain_AXT2_201_2010571.h5'],
                        help='summary sensitivity: volumes left out (default: the known ~25 dB brain val outlier)')
    parser.add_argument('--min-score', type=int, default=2, help='summary sensitivity: fewest scored slices kept')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads for torch (data preprocessing)')
    return parser.parse_args()


class Slices:
    """One anatomy's dataset items with their cached sense maps, kept on the device once loaded (`forget` drops them)."""

    def __init__(self, dataset, offset, collate, device):
        self.dataset, self.offset, self.collate, self.device = dataset, offset, collate, device
        self.items = {}

    def item(self, local):
        if local not in self.items:
            batch = self.collate([self.dataset[local]])
            self.items[local] = {k: batch[k].to(self.device) for k in ('kspace', 'target', 'sense_maps')}
            self.items[local]['metric_mask'] = metric_mask_from_real_kspace(self.items[local]['kspace'], batch['anatomy'])
        return self.items[local]

    def batch(self, locals_, copies=1):
        part = {k: torch.cat([self.item(i)[k] for i in locals_]) for k in ('kspace', 'target', 'sense_maps', 'metric_mask')}
        return {k: v.repeat(copies, *([1] * (v.dim() - 1))) for k, v in part.items()} if copies > 1 else part

    def forget(self):
        self.items.clear()


def bernoulli_masks(model, logits, kspace):
    old_mask, valid_mask = model._initial_masks(kspace)
    probability = torch.sigmoid(logits).expand(len(kspace), -1, -1, -1)
    return model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]


@torch.no_grad()
def rescaled(model, logits, kspace):
    """(budget-rescaled sampling probability over the available locations, the available locations), [1, 1, H, W]."""
    old_mask, valid_mask = model._initial_masks(kspace[:1])
    available = (1 - old_mask) * valid_mask
    return model.sampler.select_locations(torch.sigmoid(logits), old_mask, model.budget, valid_mask)[0], available


def logit_of(probability):
    return torch.logit(probability.clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS))


def objective(model, loss_type, part, mask):
    """The training loss (`loss_type`, full image), or 'psnr': minus the scored metric, the background-masked PSNR."""
    output = model(part['target'], part['kspace'], sense_maps=part['sense_maps'], fixed_mask=mask)
    if loss_type == 'psnr':
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        return -compute_psnr_torch(*_apply_signal_mask(output['output'], target, part['metric_mask'])).mean()
    return model.loss(output, {'target': part['target'], 'kspace': part['kspace']}, LOSS_WEIGHTS, loss_type)[0]


def fit(model, loss_type, start, to_logits, slices, locals_, args, lr, on_step=None):
    """Adam on `start` (logits, or a family's parameters mapped by `to_logits`) over random batches of `locals_`.
    Returns the fitted parameters; `on_step(step, logits)` is called after every step."""
    parameters = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameters], lr=lr)
    generator = np.random.RandomState(args.seed)
    for step in range(1, args.steps + 1):
        chosen = generator.choice(locals_, size=min(args.batch, len(locals_)), replace=False).tolist()
        part = slices.batch(chosen, copies=args.draws)
        optimizer.zero_grad()
        objective(model, loss_type, part, bernoulli_masks(model, to_logits(parameters), part['kspace'])).backward()
        optimizer.step()
        if on_step:
            on_step(step, to_logits(parameters.detach()))
    return parameters.detach()


@torch.no_grad()
def score(model, logits, slices, locals_):
    """Per-slice PSNR on `locals_`, averaged over SCORING_SEEDS draws keyed to the slice's global index."""
    part = slices.batch(locals_)
    total = torch.zeros(len(locals_))
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    for seed in SCORING_SEEDS:
        masks = []
        for row, local in enumerate(locals_):
            torch.manual_seed(seed * 100003 + slices.offset + local)
            masks.append(bernoulli_masks(model, logits, part['kspace'][row:row + 1]))
        output = model(part['target'], part['kspace'], sense_maps=part['sense_maps'], fixed_mask=torch.cat(masks))
        total += compute_psnr_torch(*_apply_signal_mask(output['output'], target, part['metric_mask'])).flatten().cpu() / len(SCORING_SEEDS)
    return total.tolist()


def run_anatomy(args):
    base, checkpoint = load(args.run, args.checkpoint, args.device)
    objective_type = checkpoint['options'].loss_type if args.objective == 'train' else args.objective
    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    assert cfg.datasets.get('sense_cache_dir'), f'{args.run} was not trained with cached ecalib maps'
    train_loader, val_loader = build_dataloaders(cfg)
    a = ANATOMY_INDEX[args.anatomy]
    val_part, train_part = val_loader.dataset.datasets[a], train_loader.dataset.datasets[a]
    val = Slices(val_part, sum(len(d) for d in val_loader.dataset.datasets[:a]), val_loader.collate_fn, args.device)
    train = Slices(train_part, 0, train_loader.collate_fn, args.device)

    reference_kspace = val.batch([0])['kspace']
    with torch.no_grad():
        base_logits = logit_of(base.propose_mask(reference_kspace)['unscaled'])
    base_probability, available = rescaled(base, base_logits, reference_kspace)
    grid = DensityFamily('grid', base_logits, 8.0, args.grid_size)

    def smoothed(logits, sigma):
        probability = rescaled(base, logits, reference_kspace)[0]
        return logit_of(density_blurred(probability, base_probability, available, sigma, args.eps))

    groups = volume_groups(val_part)
    start, _, stop = args.volumes.partition(':')
    names = list(groups)[int(start or 0):int(stop) if stop else None]
    names = names[::-1] if args.reverse else names
    out_dir = OUT_DIR / (args.anatomy + (f'_{args.tag}' if args.tag else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = out_dir / f'{name}.json'
        if path.exists():
            continue
        started = time.time()
        members = groups[name]
        fit_set = evenly(members[:len(members) // 2], args.max_fit)
        score_set = evenly(members[len(members) // 2 + args.gap:], args.max_score)
        rng = np.random.RandomState(args.seed + sum(map(ord, name)))
        sets = {'fit': (val, fit_set), 'fit+score': (val, fit_set + score_set),
                'ctrl': (train, control_slices(train_part, args.anatomy, name, len(fit_set), rng))}
        result = {'volume': name, 'anatomy': args.anatomy, 'fit': fit_set, 'score': score_set, 'ctrl': sets['ctrl'][1],
                  'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  'psnr': {'base': score(base, base_logits, val, score_set)}}
        free_logits = {}
        for suffix, role in FIT_ROLES.items():
            source, locals_ = sets[role]
            free = fit(base, objective_type, base_logits, lambda z: z, source, locals_, args, args.mask_lr)
            free_logits[f'mask{suffix}'] = free.half().cpu()
            result['psnr'][f'mask{suffix}'] = score(base, free, val, score_set)
            for sigma in args.sigmas:
                result['psnr'][f'mask_s{sigma:g}{suffix}'] = score(base, smoothed(free, sigma), val, score_set)
            theta = fit(base, objective_type, grid.init(), grid, source, locals_, args, args.grid_lr)
            result.setdefault('grid_theta', {})[f'grid{suffix}'] = theta.flatten().tolist()
            result['psnr'][f'grid{suffix}'] = score(base, grid(theta), val, score_set)
        LOGITS_DIR.joinpath(out_dir.name).mkdir(parents=True, exist_ok=True)
        torch.save(free_logits, LOGITS_DIR / out_dir.name / f'{name}.pt')
        path.write_text(json.dumps(result) + '\n')
        val.forget(), train.forget()
        gains = {arm: np.mean(v) - np.mean(result['psnr']['base']) for arm, v in result['psnr'].items() if arm != 'base'}
        print(f'{name} ({time.time() - started:.0f} s): ' + ', '.join(f'{k} {v:+.3f}' for k, v in gains.items()), flush=True)


def bootstrap(per_volume, rng):
    draws = [per_volume[rng.randint(len(per_volume), size=len(per_volume))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(per_volume.mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            'median': float(np.median(per_volume)), 'positive': int((per_volume > 0).sum()), 'volumes': int(len(per_volume))}


def summarize(args):
    rng = np.random.RandomState(args.seed)
    summary, lines, kept = {}, [], {}
    for anatomy in ANATOMY_INDEX:
        results = [json.loads(p.read_text()) for p in sorted((OUT_DIR / (anatomy + (f'_{args.tag}' if args.tag else ''))).glob('*.json'))]
        if not results:
            continue
        arms = [arm for arm in results[0]['psnr'] if arm != 'base']
        per_volume = {arm: np.array([np.mean(r['psnr'][arm]) for r in results]) for arm in results[0]['psnr']}
        summary[anatomy] = {'base_psnr': float(per_volume['base'].mean()), 'vs_base': {}, 'vs_ctrl': {}}
        kept[anatomy] = (results, per_volume)
        for arm in arms:
            summary[anatomy]['vs_base'][arm] = bootstrap(per_volume[arm] - per_volume['base'], rng)
            if not arm.endswith(('_ctrl', '_in')):
                summary[anatomy]['vs_ctrl'][arm] = bootstrap(per_volume[arm] - per_volume[f'{arm}_ctrl'], rng)
        lines += [f'\n[{anatomy}] {len(results)} volumes, base {summary[anatomy]["base_psnr"]:.3f} dB',
                  '| arm | vs base | vs its control |', '|---|---|---|']
        for arm in arms:
            cells = [summary[anatomy]['vs_base'][arm]]
            cells.append(summary[anatomy]['vs_ctrl'].get(arm))
            lines.append(f'| {arm} | ' + ' | '.join(f"{c['mean']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if c else ''
                                                   for c in cells) + ' |')
    for anatomy, (results, per_volume) in kept.items():
        robust = (np.array([r['volume'] not in args.exclude and len(r['score']) >= args.min_score for r in results]))
        summary[anatomy]['robustness'] = {'excluded': [r['volume'] for r, k in zip(results, robust) if not k]}
        lines += [f'\n[{anatomy}] smooth arms minus control: concentration, and without {int((~robust).sum())} volumes '
                  f'(--exclude, or scored on < {args.min_score} slices)',
                  '| arm | median | positive | top-5 share of sum | corr(base PSNR, gain) | mean without excluded |', '|---|---|---|---|---|---|']
        for arm in SMOOTH_ARMS:
            gain = per_volume[arm] - per_volume[f'{arm}_ctrl']
            row = {'median': float(np.median(gain)), 'positive': int((gain > 0).sum()), 'volumes': int(len(gain)),
                   'top5_share_of_sum': float(np.sort(gain)[::-1][:5].sum() / gain.sum()) if gain.sum() > 0 else None,
                   'corr_base_psnr': float(np.corrcoef(per_volume['base'], gain)[0, 1]),
                   'without_excluded': bootstrap(gain[robust], rng)}
            summary[anatomy]['robustness'][arm] = row
            share = f"{row['top5_share_of_sum']:.2f}" if row['top5_share_of_sum'] is not None else ''
            w = row['without_excluded']
            lines.append(f"| {arm} | {row['median']:+.3f} | {row['positive']}/{row['volumes']} | {share} | {row['corr_base_psnr']:+.2f} | "
                         f"{w['mean']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}] |")
    passed = {anatomy: [arm for arm in SMOOTH_ARMS if arm in s.get('vs_ctrl', {}) and s['vs_ctrl'][arm]['mean'] >= 0.05
                        and s['vs_ctrl'][arm]['ci95'][0] > 0] for anatomy, s in summary.items()}
    summary['stage1_pass'] = {anatomy: arms for anatomy, arms in passed.items()}
    lines.append(f'\nStage 1 (smooth held-out arm - control >= +0.05 dB, interval above 0): '
                 + ('PASS ' + json.dumps(passed) if any(passed.values()) else 'STOP'))
    out = OUT_DIR / f'summary{"_" + args.tag if args.tag else ""}'
    out.with_suffix('.json').write_text(json.dumps(summary, indent=1) + '\n')
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.summarize:
        summarize(args)
    else:
        run_anatomy(args)


if __name__ == '__main__':
    main()
