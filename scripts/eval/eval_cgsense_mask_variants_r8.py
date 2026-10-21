"""Shared vs per-anatomy vs per-protocol masks, Bernoulli (LOUPE) vs deterministic DCM selection, CG-SENSE R=8.

Every run is scored on the full fastMRI validation set with its own sense net and its own mask(s), using seeded
draws shared across runs (DCM masks are deterministic). Labels follow each checkpoint's classes: the anatomy, or the
protocol (brain / knee-PD / knee-PDFS) when the run was trained with `datasets.condition_classes`. Reported per
group, as paired per-slice differences against the reference run with a volume bootstrap:
  - each run vs the shared Bernoulli run (`--reference`);
  - for conditioned runs, a class swap: each slice scored with the mask of another class (brain <-> knee for anatomy
    runs, knee-PD <-> knee-PDFS for protocol runs; brain slices of protocol runs use the knee-PD mask).

Runs trained with cached ecalib maps are scored with those maps. The ecalib pair compares a 24 x 24 against a 10 x 10
preselected centre with the same maps (both from the 24 x 24 calibration):

    python scripts/eval/eval_cgsense_mask_variants_r8.py
    python scripts/eval/eval_cgsense_mask_variants_r8.py --runs ecalib_c10 ecalib_c24 --reference ecalib_c10 \\
        --out analysis/acs_gate_cgsense/population_c24_vs_c10.json

The 100-epoch line (bilinear sense upsampling, constant lr, all from a random start):

    python scripts/eval/eval_cgsense_mask_variants_r8.py --runs shared_100ep anatomy_100ep protocol_100ep \\
        --reference shared_100ep --out analysis/cgsense_mask_variants/eval_100ep.json
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
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask, build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch

RUNS = {
    'shared': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
    'anatomy': 'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond',
    'protocol': 'checkpoints/loupe2d_cgsense_r8_brain_knee_prototable',
    'dcm_shared': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_shared',
    'dcm_anatomy': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_anatcond',
    'dcm_protocol': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_prototable',
    'ecalib_c24': 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24',
    'ecalib_c10': 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10',
    'shared_100ep': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep',
    'anatomy_100ep': 'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep',
    'protocol_100ep': 'checkpoints/loupe2d_cgsense_r8_brain_knee_prototable_bilinear_const_100ep',
}
SWAP = {'brain': 'knee', 'knee': 'brain', 'knee-PD': 'knee-PDFS', 'knee-PDFS': 'knee-PD'}
GROUPS = ('brain', 'knee', 'knee-PD', 'knee-PDFS')
SCORING_SEEDS = (0, 1)
BOOTSTRAPS = 2000
OUT = Path('analysis/cgsense_mask_variants/eval.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', default=list(RUNS)[:6])
    parser.add_argument('--reference', default='shared')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=OUT)
    return parser.parse_args()


def load(run, checkpoint_name, device):
    checkpoint = torch.load(run / checkpoint_name, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


def class_labels(names, classes, device):
    index = {name: i for i, name in enumerate(classes)}
    return torch.tensor([index[n] for n in names], device=device)


@torch.no_grad()
def scored(model, part, labels, seed_key):
    """Per-slice trainer-path PSNR of `model` with its own mask for `labels`, averaged over SCORING_SEEDS draws."""
    unscaled = model.propose_mask(part['kspace'], anatomy_labels=labels)['unscaled']
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    total = torch.zeros(len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = model.sampler.select_locations(unscaled, old_mask, model.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], sense_maps=part.get('sense_maps'), fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        total += compute_psnr_torch(*_apply_signal_mask(output, target, part['metric_mask'])).flatten().cpu() / len(SCORING_SEEDS)
    return total


def bootstrap(delta, groups, rng):
    unique = np.unique(groups)
    per_group = np.array([delta[groups == g].mean() for g in unique])
    draws = [per_group[rng.randint(len(per_group), size=len(per_group))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(delta.mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])], 'slices': int(len(delta))}


def main():
    args = parse_args()
    runs = {name: Path(RUNS[name]) for name in args.runs if (Path(RUNS[name]) / args.checkpoint).exists()}
    cfg = OmegaConf.load(runs[args.reference] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    indices = list(range(0, len(val_loader.dataset), args.stride))
    loader = DataLoader(Subset(val_loader.dataset, indices), batch_size=args.batch, shuffle=False, num_workers=4,
                        collate_fn=val_loader.collate_fn)

    psnr, meta = {}, {'anatomy': [], 'condition': [], 'volume': []}
    for name, run in runs.items():
        model, checkpoint = load(run, args.checkpoint, args.device)
        options = checkpoint['options']
        classes = list(options.label_classes) if options.anatomy_conditioning else []
        key = options.label_key
        own, swapped, position = [], [], 0
        for batch in loader:
            part = {k: batch[k].to(args.device) for k in ('kspace', 'target', 'sense_maps') if k in batch}
            part['metric_mask'] = metric_mask_from_real_kspace(part['kspace'], batch['anatomy'])
            names = list(batch[key])
            labels = class_labels(names, classes, args.device) if classes else None
            seed_key = indices[position]
            own.append(scored(model, part, labels, seed_key))
            if classes:
                other = [SWAP[n] if SWAP.get(n) in classes else next(c for c in classes if c != n) for n in names]
                swapped.append(scored(model, part, class_labels(other, classes, args.device), seed_key))
            if name == next(iter(runs)):
                meta['anatomy'] += list(batch['anatomy'])
                meta['condition'] += list(batch['condition'])
                meta['volume'] += list(batch['fname'])
            position += len(names)
        psnr[name] = torch.cat(own).numpy()
        if swapped:
            psnr[f'{name}_swap'] = torch.cat(swapped).numpy()
        print(f'{name}: mean PSNR {psnr[name].mean():.3f} ({len(classes)} classes, {options.selection if hasattr(options, "selection") else "bernoulli"})', flush=True)
        del model
        torch.cuda.empty_cache()

    anatomy, condition, volume = (np.array(meta[k]) for k in ('anatomy', 'condition', 'volume'))
    membership = {g: (anatomy == g) if g in ('brain', 'knee') else (condition == g) for g in GROUPS}
    rng = np.random.RandomState(0)
    result = {'runs': {k: str(v) for k, v in runs.items()}, 'reference': args.reference, 'mean_psnr': {}, 'vs_reference': {}, 'swap_cost': {}}
    for group, keep in membership.items():
        result['mean_psnr'][group] = {name: float(psnr[name][keep].mean()) for name in runs}
        result['vs_reference'][group] = {name: bootstrap(psnr[name][keep] - psnr[args.reference][keep], volume[keep], rng)
                                         for name in runs if name != args.reference}
        result['swap_cost'][group] = {name: bootstrap(psnr[name][keep] - psnr[f'{name}_swap'][keep], volume[keep], rng)
                                      for name in runs if f'{name}_swap' in psnr}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    np.savez(args.out.with_suffix('.npz'), anatomy=anatomy, condition=condition, volume=volume,
             **{f'psnr__{k}': v for k, v in psnr.items()})

    lines = ['| run | ' + ' | '.join(f'{g} PSNR' for g in GROUPS) + ' | ' + ' | '.join(f'{g} vs {args.reference}' for g in GROUPS) + ' |',
             '|---|' + '---|' * (2 * len(GROUPS))]
    for name in runs:
        cells = [f"{result['mean_psnr'][g][name]:.3f}" for g in GROUPS]
        for g in GROUPS:
            r = result['vs_reference'][g].get(name)
            cells.append(f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" if r else '')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| run (class swap: own mask - another class\'s mask) | ' + ' | '.join(GROUPS) + ' |', '|---|' + '---|' * len(GROUPS)]
    for name in runs:
        if any(name in result['swap_cost'][g] for g in GROUPS):
            cells = []
            for g in GROUPS:
                r = result['swap_cost'][g].get(name)
                cells.append(f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" if r else '')
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
