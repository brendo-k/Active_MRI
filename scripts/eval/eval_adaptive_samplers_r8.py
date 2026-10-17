"""Readout for the adaptive-sampler next steps (R=8, brain + knee), fixed before the runs finished.

For every run, on the whole validation set with common seeded Bernoulli draws (a slice's thresholds depend only on
the seed and the slice), through the run's own reconstructor:

  own                PSNR (trainer's masked metric) with the slice's own mask
  anatomy swap cost  own minus PSNR with a mask the sampler made for a random slice of the other anatomy
  slice swap cost    own minus PSNR with a mask made for a random slice of another volume of the same anatomy
  adaptivity         slice-vs-anatomy-mean correlation of the effective probability, brain-vs-knee correlation of the
                     anatomy means; for smooth_adapt_2d, the offset's mean |value| and its across-slice std

Paired comparisons (per-slice differences, 95% CI over volumes) are against each arm's matched static control. A donor's
mask keeps the donor's rescaled probabilities and is applied inside the target slice's own valid region.

    python scripts/eval/eval_adaptive_samplers_r8.py --runs adapt16_joint_R8 static_joint_R8 ...
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
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import CONV_RUN, load, to_device

RUNS_DIR = Path('/tmp/bkadota_runs')
LOUPE_RUN = Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05')
OUT = Path('reports/adaptivity_next_steps_r8')
SEEDS = (1234, 4321)
ANATOMIES = ('brain', 'knee')
BOOTSTRAPS = 2000
CONTROLS = {
    'adapt16_joint_R8': 'static_joint_R8',
    'anattable_joint_R8': 'static_joint_R8',
    'adapt16_frozen-loupe-recon_R8': 'loupe_original',
    'adapt16_frozen-generalist_R8': 'static_frozen-generalist_R8',
    'static_joint_R8': 'loupe_original',
    'adapt16_joint_flags_R8': 'static_joint_flags_R8',
    'anattable_joint_flags_R8': 'static_joint_flags_R8',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='run names under --runs-dir, or loupe_original')
    parser.add_argument('--runs-dir', type=Path, default=RUNS_DIR)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--tag', default='plain')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def run_path(name, args):
    return (LOUPE_RUN, 'last_model.pt') if name == 'loupe_original' else (args.runs_dir / name, args.checkpoint)


@torch.no_grad()
def sampler_observation(model, kspace):
    """The NHWC observation the sampler sees in propose_mask (SENSE-combined centre k-space, twice)."""
    from activemri.models.sequential_sampling import torch_fft2, torch_ifft2
    sense_maps, _ = model.sensetivity_net(torch.view_as_complex(kspace.contiguous()), mask_method=model.sense_mask_method)
    old_mask, _ = model._initial_masks(kspace)
    observed = torch.view_as_complex((kspace * old_mask.unsqueeze(-1)).contiguous())
    combined = torch.view_as_real(torch_fft2((torch_ifft2(observed) * sense_maps).sum(dim=1)))
    return torch.cat([combined, combined * old_mask.permute(0, 2, 3, 1)], dim=-1)


def with_labels(model, batch, device):
    """The batch on `device`, with anatomy labels built from the anatomy names as the trainer does."""
    part = to_device(batch, device)
    classes = list(getattr(model, 'label_classes', None) or ANATOMIES)
    part['label'] = torch.as_tensor([classes.index(a) for a in batch['anatomy']], device=device)
    return part


@torch.no_grad()
def probability_maps(model, loader, device):
    """Per val slice: rescaled probabilities (float16), old and valid masks, offset stats when the sampler has one."""
    rescaled, old, valid, offset_abs, offsets = [], [], [], [], []
    has_offset = getattr(model.sampler.mask_net, 'grid', 0) > 0
    for batch in loader:
        part = with_labels(model, batch, device)
        proposal = model.propose_mask(part['kspace'], anatomy_labels=part['label'])
        rescaled.append(proposal['rescaled'][:, 0].half().cpu())
        old.append(proposal['old_mask'][:, 0].bool().cpu())
        valid.append(proposal['valid_mask'][:, 0].bool().cpu())
        if has_offset:
            offset = model.sampler.mask_net.offset_logits(sampler_observation(model, part['kspace']))
            offset_abs.append(offset.abs().flatten(1).mean(1).cpu())
            offsets.append(offset[:, ::4, ::4].half().cpu())
    out = {'rescaled': torch.cat(rescaled), 'old': torch.cat(old), 'valid': torch.cat(valid)}
    if has_offset:
        stacked = torch.cat(offsets).float()
        out['offset_mean_abs'] = float(torch.cat(offset_abs).mean())
        out['offset_across_slice_std'] = float(stacked.std(0).mean())
    return out


def thresholds(seed, index, shape):
    generator = torch.Generator().manual_seed(seed * 1_000_003 + index)
    return torch.rand(shape, generator=generator)


@torch.no_grad()
def psnr_with_donors(model, loader, maps, donors, device):
    """PSNR per slice for each donor assignment (dict name -> [N] donor index), averaged over SEEDS."""
    results = {name: torch.zeros(len(maps['rescaled'])) for name in donors}
    start = 0
    for batch in loader:
        part = with_labels(model, batch, device)
        count = len(part['kspace'])
        rows = torch.arange(start, start + count)
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        for seed in SEEDS:
            threshold = torch.stack([thresholds(seed, int(i), maps['rescaled'].shape[1:]) for i in rows])
            for name, donor in donors.items():
                probability = maps['rescaled'][donor[rows]].float()
                new = (probability > threshold) & maps['valid'][rows]
                mask = (maps['old'][rows] | new).float().unsqueeze(1).to(device)
                output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
                results[name][rows] += compute_psnr_torch(*_apply_signal_mask(output, target)).flatten().cpu() / len(SEEDS)
        start += count
    return results


def correlation(a, b):
    a, b = a.flatten() - a.mean(), b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def adaptivity(maps, anatomy):
    effective = (maps['rescaled'].float() + maps['old'].float()).clamp(max=1)
    means = {a: effective[torch.as_tensor(anatomy == a)].mean(0) for a in ANATOMIES}
    per_slice = [correlation(effective[i], means[anatomy[i]]) for i in range(len(effective))]
    return {'slice_vs_mean_corr': {a: float(np.mean([c for c, x in zip(per_slice, anatomy) if x == a])) for a in ANATOMIES},
            'brain_vs_knee_corr': correlation(means['brain'], means['knee'])}


def bootstrap(delta, volumes, rng):
    unique = np.unique(volumes)
    per_volume = np.array([delta[volumes == v].mean() for v in unique])
    draws = [per_volume[rng.randint(len(per_volume), size=len(per_volume))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(delta.mean()), 'ci95': np.percentile(draws, [2.5, 97.5]).tolist()}


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    cfg = OmegaConf.load(CONV_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset if args.limit is None else torch.utils.data.Subset(
        val_loader.dataset, np.linspace(0, len(val_loader.dataset) - 1, args.limit).astype(int).tolist())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.chunk, shuffle=False, num_workers=4,
                                         collate_fn=val_loader.collate_fn)
    anatomy = np.array(sum((list(b['anatomy']) for b in loader), []))
    volumes = np.array(sum((list(b['fname']) for b in loader), []))
    count = len(anatomy)
    donor_rng = np.random.RandomState(7)
    donors = {'own': np.arange(count),
              'other_anatomy': np.array([donor_rng.choice(np.flatnonzero(anatomy != anatomy[i])) for i in range(count)]),
              'same_anatomy': np.array([donor_rng.choice(np.flatnonzero((anatomy == anatomy[i]) & (volumes != volumes[i])))
                                        for i in range(count)])}
    donors = {k: torch.as_tensor(v) for k, v in donors.items()}

    per_slice, results = {}, {}
    for name in args.runs:
        run_dir, checkpoint = run_path(name, args)
        model, checkpoint_data = load(run_dir, checkpoint, args.device)
        model.label_classes = list(checkpoint_data['options'].label_classes)
        maps = probability_maps(model, loader, args.device)
        psnr = psnr_with_donors(model, loader, maps, donors, args.device)
        per_slice[name] = {k: v.numpy() for k, v in psnr.items()}
        own = per_slice[name]['own']
        results[name] = {
            'psnr': {a: float(own[anatomy == a].mean()) for a in ANATOMIES},
            'anatomy_swap_cost_db': bootstrap(own - per_slice[name]['other_anatomy'], volumes, rng),
            'slice_swap_cost_db': bootstrap(own - per_slice[name]['same_anatomy'], volumes, rng),
            **adaptivity(maps, anatomy),
            **{k: v for k, v in maps.items() if k.startswith('offset')}}
        print(name, json.dumps(results[name]), flush=True)
        del model
        torch.cuda.empty_cache()

    paired = {}
    for arm, control in CONTROLS.items():
        if arm in per_slice and control in per_slice:
            delta = per_slice[arm]['own'] - per_slice[control]['own']
            paired[f'{arm} - {control}'] = {'all': bootstrap(delta, volumes, rng),
                                            **{a: bootstrap(delta[anatomy == a], volumes[anatomy == a], rng) for a in ANATOMIES}}
    results['paired'] = paired
    print(json.dumps(paired, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'eval_{args.tag}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
