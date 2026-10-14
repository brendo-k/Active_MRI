"""Paired comparison: one shared LOUPE mask vs one mask per anatomy (R=8, brain+knee VarNet).

  shared        checkpoints/loupe2d_varnet_r8_brain_knee          (wandb azd6r64k)
  shared_100ep  checkpoints/loupe2d_varnet_r8_brain_knee_100ep    (wandb 1k9d0xdj), twice the
                epochs and lr step at 90 instead of 40, so not like-for-like with anatcond
  anatcond      checkpoints/loupe2d_varnet_r8_brain_knee_anatcond (wandb pjysura8)
  brainonly,    single-anatomy runs (datasets.train_anatomies), 100 epochs, lr step 80: the same
  kneeonly      gradient steps as the 50-epoch shared run, since a one-anatomy epoch has half the slices
  brainonly2700 brain-only on 2700 brain slices (as many train slices as the shared run), 100 epochs, lr step 90;
  (_last)       `_last` scores its epoch-100 checkpoint instead of the best-val-loss one
  fixedmasks    joint brain+knee reconstructor trained from scratch on the frozen brainonly and kneeonly
                masks, one per anatomy via the conditioned sampler; fixedmasks_swap gives it the other label
  brainonly1350_200ep brain-only on the 1350 balanced brain slices for 200 epochs (lr step 180): the steps of
  (_last)       brainonly2700 and shared_100ep with half the unique slices
  fixed2700mask like fixedmasks, but the brain slot holds the brainonly2700 mask instead of the brainonly mask
  anatcond_swap the anatcond model given the other anatomy's label, so each slice is
                sampled with the other anatomy's mask; same reconstructor

Same config otherwise; anatcond only adds sampler.anatomy_conditioning=true. Every model
sees the same val slices and the same RNG stream for the Bernoulli mask draws. Metrics
follow the trainer (body-masked output and target): the aggregate reproduces the logged
per-anatomy numbers, and paired per-slice PSNR/SSIM use a per-slice data range with
confidence intervals over per-volume means.

    CUDA_VISIBLE_DEVICES=1 python scripts/eval/compare_anatcond_r8.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.trainer import _AnatomyStats, _apply_signal_mask, build_model
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from activemri.training.builders import build_dataloaders

RUNS = {
    'shared': Path('checkpoints/loupe2d_varnet_r8_brain_knee'),
    'shared_100ep': Path('checkpoints/loupe2d_varnet_r8_brain_knee_100ep'),
    'anatcond': Path('checkpoints/loupe2d_varnet_r8_brain_knee_anatcond'),
    'brainonly': Path('checkpoints/loupe2d_varnet_r8_brainonly_stepmatched'),
    'kneeonly': Path('checkpoints/loupe2d_varnet_r8_kneeonly_stepmatched'),
    'brainonly2700': Path('checkpoints/loupe2d_varnet_r8_brainonly_2700slices_100ep'),
    'brainonly2700_last': Path('checkpoints/loupe2d_varnet_r8_brainonly_2700slices_100ep'),
    'fixedmasks': Path('checkpoints/loupe2d_varnet_r8_brain_knee_fixedsinglemasks_100ep'),
    'brainonly1350_200ep': Path('checkpoints/loupe2d_varnet_r8_brainonly_1350slices_200ep'),
    'brainonly1350_200ep_last': Path('checkpoints/loupe2d_varnet_r8_brainonly_1350slices_200ep'),
    'fixed2700mask': Path('checkpoints/loupe2d_varnet_r8_brain_knee_fixed2700brainmask_100ep'),
}
CHECKPOINT_FILES = {'brainonly2700_last': 'last_model.pt', 'brainonly1350_200ep_last': 'last_model.pt'}  # every other run is scored at its best-val-loss epoch
CONDITIONS = [('shared', False), ('shared_100ep', False), ('anatcond', False), ('anatcond_swap', True),
              ('brainonly', False), ('kneeonly', False),
              ('brainonly2700', False), ('brainonly2700_last', False),
              ('fixedmasks', False), ('fixedmasks_swap', True),
              ('brainonly1350_200ep', False), ('brainonly1350_200ep_last', False),
              ('fixed2700mask', False), ('fixed2700mask_swap', True)]
COMPARISONS = [('anatcond', 'shared'), ('anatcond', 'shared_100ep'), ('shared_100ep', 'shared'),
               ('anatcond', 'anatcond_swap'),
               ('brainonly', 'shared'), ('kneeonly', 'shared'),
               ('brainonly', 'shared_100ep'), ('kneeonly', 'shared_100ep'),
               ('brainonly', 'anatcond'), ('kneeonly', 'anatcond'),
               ('brainonly2700', 'shared_100ep'), ('brainonly2700_last', 'shared_100ep'),
               ('brainonly2700', 'brainonly'), ('brainonly2700_last', 'brainonly2700'),
               ('fixedmasks', 'shared_100ep'), ('fixedmasks', 'anatcond'), ('fixedmasks', 'shared'),
               ('fixedmasks', 'brainonly'), ('fixedmasks', 'kneeonly'), ('fixedmasks', 'fixedmasks_swap'),
               ('brainonly1350_200ep', 'brainonly2700'), ('brainonly1350_200ep_last', 'brainonly2700_last'),
               ('brainonly1350_200ep', 'shared_100ep'), ('brainonly1350_200ep_last', 'shared_100ep'),
               ('brainonly1350_200ep', 'brainonly'), ('brainonly1350_200ep_last', 'brainonly1350_200ep'),
               ('fixed2700mask', 'fixedmasks'), ('fixed2700mask', 'shared_100ep'),
               ('fixed2700mask', 'kneeonly'), ('fixed2700mask', 'fixed2700mask_swap')]
METRICS = ('psnr', 'ssim')


def load_model(run, device):
    ckpt = torch.load(RUNS[run] / CHECKPOINT_FILES.get(run, 'best_model.pt'), map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['options']


@torch.no_grad()
def evaluate(model, options, val_loader, device, seed, swap_labels):
    classes = list(options.label_classes)
    conditioned = options.anatomy_conditioning
    torch.manual_seed(seed)
    stats = _AnatomyStats(device)
    rows = {'anatomy': [], 'volume': [], 'psnr': [], 'ssim': []}
    for batch in val_loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        anatomies = list(batch['anatomy'])
        labels = None
        if conditioned:
            labels = torch.tensor([classes.index(a) for a in anatomies], device=device)
            if swap_labels:
                labels = 1 - labels
        output = model(target, kspace, anatomy_labels=labels)['output']
        if isinstance(output, list):
            output = output[-1]
        target4 = target if output.dim() == target.dim() else target.unsqueeze(1)
        masked_output, masked_target = _apply_signal_mask(output, target4)
        stats.update(anatomies, masked_output, masked_target)
        rows['anatomy'] += anatomies
        rows['volume'] += [f'{a}/{f}' for a, f in zip(anatomies, batch['fname'])]
        rows['psnr'].append(compute_psnr_torch(masked_output, masked_target).flatten().cpu())
        rows['ssim'].append(compute_ssim_torch(masked_output, masked_target).flatten().cpu())
    rows['psnr'] = torch.cat(rows['psnr'])
    rows['ssim'] = torch.cat(rows['ssim'])
    aggregate = {a: {k: float(v) for k, v in m.compute().items()} for a, m in stats.metrics.items()}
    return aggregate, rows


def paired_stats(delta, volumes):
    """Mean over slices, 95% CI over per-volume means, share of volumes with delta > 0."""
    names = sorted(set(volumes))
    index = torch.tensor([names.index(v) for v in volumes])
    per_volume = torch.zeros(len(names)).index_add_(0, index, delta) / torch.bincount(index).float()
    mean = delta.mean().item()
    half = 1.96 * per_volume.std().item() / np.sqrt(len(names))
    return {'mean': mean, 'ci': [mean - half, mean + half], 'n_volumes': len(names),
            'frac_volumes_better': (per_volume > 0).float().mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321], help='RNG streams for mask draws')
    parser.add_argument('--out', type=Path, default=Path('reports/anatcond_r8'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(RUNS['shared'] / '.hydra' / 'config.yaml')
    _, val_loader = build_dataloaders(cfg)
    models = {run: load_model(run, device) for run in RUNS}

    results = {}
    for name, swap in CONDITIONS:
        model, options = models[name.removesuffix('_swap')]
        draws = [evaluate(model, options, val_loader, device, seed, swap) for seed in args.seeds]
        rows = draws[0][1]
        results[name] = {
            'aggregate': [d[0] for d in draws],
            'rows': rows,
            **{m: torch.stack([d[1][m] for d in draws]).mean(0) for m in METRICS},
        }
        for seed, (aggregate, _) in zip(args.seeds, draws):
            print(f'{name:14s} seed {seed}  ' + '  '.join(
                f'[{a}] PSNR={v["PSNR"]:.3f} SSIM={v["SSIM"]:.4f}' for a, v in sorted(aggregate.items())), flush=True)

    reference = results['shared']['rows']
    for name in results:
        assert results[name]['rows']['volume'] == reference['volume'], f'{name} saw slices in a different order'

    summary = {}
    for a_name, b_name in COMPARISONS:
        key = f'{a_name} - {b_name}'
        summary[key] = {}
        print(f'\n{key}  (per-slice, mean over {len(args.seeds)} mask draws; CI over volumes)')
        for anatomy in sorted(set(reference['anatomy'])):
            keep = torch.tensor([a == anatomy for a in reference['anatomy']])
            volumes = [v for v, k in zip(reference['volume'], keep) if k]
            summary[key][anatomy] = {}
            for metric in METRICS:
                delta = results[a_name][metric][keep] - results[b_name][metric][keep]
                s = paired_stats(delta, volumes)
                summary[key][anatomy][metric] = s
                print(f'  {anatomy:6s} {metric}  {s["mean"]:+.4f}  95% CI [{s["ci"][0]:+.4f}, {s["ci"][1]:+.4f}]'
                      f'  better on {100 * s["frac_volumes_better"]:.0f}% of {s["n_volumes"]} volumes')

    summary['aggregate'] = {name: r['aggregate'] for name, r in results.items()}
    (args.out / 'paired.json').write_text(json.dumps(summary, indent=2))
    torch.save({name: {m: r[m] for m in METRICS} | {'rows': r['rows']} for name, r in results.items()},
               args.out / 'per_slice.pt')


if __name__ == '__main__':
    main()
