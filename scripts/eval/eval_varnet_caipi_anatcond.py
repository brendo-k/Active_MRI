"""Anatomy conditioning on the PromptMR+ CAIPI VarNet: conditioned arm vs label-blind arm, per class.

Runs come from scripts/launch/train_varnet_caipi_promptmr_anatcond.sh. Within a seed the two arms share code,
parameters, initialisation and data order, and only the conditioned arm sees the class. Every val slice is scored
inside the apodized per-anatomy metric mask, two ways:

  per slice       PSNR / SSIM of each slice (data range = the slice's max), averaged over slices; paired statistics
                  use these
  trainer         the trainer's validation metric (_AnatomyStats: pooled MSE over the group, data range = a batch
                  max), per class and per anatomy; it weights high-error slices more

The script reports:

  cond - blind    per class (brain, knee-PD, knee-PDFS, and knee pooled), per seed and averaged over seeds,
                  with a 95% CI over per-volume means (normal and Student-t; the knee classes have 4-6 volumes)
  label swap      the conditioned model given each class for every slice, minus its own class: if this is ~0 the
                  model ignores the label, whatever the arm gap is

    CUDA_VISIBLE_DEVICES=2 python scripts/eval/eval_varnet_caipi_anatcond.py --seeds 42 43
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from scipy.stats import t as student_t

import _path  # noqa: F401
from compare_anatcond_r8 import METRICS, paired_stats
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _AnatomyStats, _apply_signal_mask, build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch

RUN_PATTERN = 'checkpoints/varnet_caipi2d_r4_promptmr_{arm}_seed{seed}'
KNEE = ('knee-PD', 'knee-PDFS')


def load(run, checkpoint, device):
    ckpt = torch.load(run / checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['options'], ckpt['epoch']


@torch.no_grad()
def score(model, options, val_loader, device, given_class=None):
    """Per-slice rows, plus the trainer's metrics per class and per anatomy under rows['trainer'].

    Each slice is labelled with its own class, or with `given_class` for all slices.
    """
    classes = list(options.label_classes)
    rows = {'class': [], 'volume': [], 'psnr': [], 'ssim': []}
    by_class, by_anatomy = _AnatomyStats(device), _AnatomyStats(device)
    for batch in val_loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        own = list(batch[options.label_key])
        labels = torch.tensor([classes.index(c) if given_class is None else classes.index(given_class) for c in own],
                              device=device)
        output = model(target, kspace, anatomy_labels=labels)['output']
        target4 = target if output.dim() == target.dim() else target.unsqueeze(1)
        masked_output, masked_target = _apply_signal_mask(output, target4, metric_mask_from_real_kspace(kspace, batch['anatomy']))
        by_class.update(own, masked_output, masked_target)
        by_anatomy.update(list(batch['anatomy']), masked_output, masked_target)
        rows['class'] += own
        rows['volume'] += [f'{c}/{f}' for c, f in zip(own, batch['fname'])]
        rows['psnr'].append(compute_psnr_torch(masked_output, masked_target).flatten().cpu())
        rows['ssim'].append(compute_ssim_torch(masked_output, masked_target).flatten().cpu())
    rows['psnr'], rows['ssim'] = torch.cat(rows['psnr']), torch.cat(rows['ssim'])
    rows['trainer'] = {}
    for key, value in {**by_class.compute('val'), **by_anatomy.compute('val')}.items():
        _, group, metric = key.split('/')
        rows['trainer'].setdefault(group, {})[metric.lower()] = value
    return rows


def groups(rows):
    """Class name -> boolean selector over the rows, with knee-PD and knee-PDFS also pooled as knee."""
    names = sorted(set(rows['class']))
    selectors = {name: torch.tensor([c == name for c in rows['class']]) for name in names}
    if all(k in names for k in KNEE):
        selectors['knee'] = torch.tensor([c in KNEE for c in rows['class']])
    return selectors


def compare(a, b, reference):
    """Paired a - b per group, over the reference rows' slice order."""
    out = {}
    for name, keep in groups(reference).items():
        volumes = [v for v, k in zip(reference['volume'], keep) if k]
        out[name] = {}
        for metric in METRICS:
            stats = paired_stats(a[metric][keep] - b[metric][keep], volumes)
            stats['ci_t'] = student_t_ci(stats)
            stats['a_mean'], stats['b_mean'] = a[metric][keep].mean().item(), b[metric][keep].mean().item()
            out[name][metric] = stats
    return out


def student_t_ci(stats):
    """paired_stats' normal 95% CI over volume means, widened to the Student-t quantile for its number of volumes."""
    half = (stats['ci'][1] - stats['ci'][0]) / 2 * student_t.ppf(0.975, stats['n_volumes'] - 1) / 1.96
    return [stats['mean'] - half, stats['mean'] + half]


def trainer_gap(a, b):
    """a - b in the trainer's per-group metrics."""
    return {group: {metric: {'a': a[group][metric], 'b': b[group][metric], 'diff': a[group][metric] - b[group][metric]}
                    for metric in a[group]}
            for group in a}


def print_table(title, table, a_name, b_name):
    print(f'\n{title}')
    for name, metrics in table.items():
        for metric, s in metrics.items():
            print(f'  {name:10s} {metric}  {a_name} {s["a_mean"]:.4f}  {b_name} {s["b_mean"]:.4f}  '
                  f'diff {s["mean"]:+.4f}  95% CI [{s["ci"][0]:+.4f}, {s["ci"][1]:+.4f}]  '
                  f't-CI [{s["ci_t"][0]:+.4f}, {s["ci_t"][1]:+.4f}]  '
                  f'{a_name} better on {100 * s["frac_volumes_better"]:.0f}% of {s["n_volumes"]} volumes')


def print_trainer_table(title, gap, a_name, b_name):
    print(f'\n{title} (trainer metric: pooled MSE per group)')
    for group, metrics in gap.items():
        cells = '  '.join(f'{metric} {a_name} {s["a"]:.4f} {b_name} {s["b"]:.4f} diff {s["diff"]:+.4f}'
                          for metric, s in metrics.items() if metric in ('psnr', 'ssim', 'nmse'))
        print(f'  {group:10s} {cells}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, default=Path('analysis/varnet_caipi_anatcond/eval.json'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    runs = {(arm, seed): Path(RUN_PATTERN.format(arm=arm, seed=seed)) for seed in args.seeds for arm in ('cond', 'blind')}
    _, val_loader = build_dataloaders(OmegaConf.load(runs['cond', args.seeds[0]] / '.hydra' / 'config.yaml'))

    summary = {'checkpoint': args.checkpoint, 'seeds': args.seeds, 'per_seed': {}}
    deltas = []
    for seed in args.seeds:
        scored = {}
        for arm in ('cond', 'blind'):
            model, options, epoch = load(runs[arm, seed], args.checkpoint, device)
            print(f'{runs[arm, seed]} {args.checkpoint}: epoch {epoch}, classes {list(options.label_classes)}', flush=True)
            scored[arm] = score(model, options, val_loader, device)
            if arm == 'cond':
                swaps = {given: score(model, options, val_loader, device, given) for given in options.label_classes}
            del model
        reference = scored['cond']
        assert scored['blind']['volume'] == reference['volume'], 'arms saw slices in a different order'
        if deltas:
            assert reference['volume'] == first_reference['volume'], 'seeds saw slices in a different order'
        else:
            first_reference = reference
        arm_gap = compare(scored['cond'], scored['blind'], reference)
        arm_gap_trainer = trainer_gap(scored['cond']['trainer'], scored['blind']['trainer'])
        print_table(f'seed {seed}: cond - blind', arm_gap, 'cond', 'blind')
        print_trainer_table(f'seed {seed}: cond - blind', arm_gap_trainer, 'cond', 'blind')
        label_swap, label_swap_trainer = {}, {}
        for given, rows in swaps.items():
            label_swap[given] = compare(rows, reference, reference)
            label_swap_trainer[given] = trainer_gap(rows['trainer'], reference['trainer'])
            print_table(f'seed {seed}: cond model given class {given} for every slice - own class',
                        label_swap[given], f'as-{given}', 'own')
            print_trainer_table(f'seed {seed}: cond model given class {given} for every slice - own class',
                                label_swap_trainer[given], f'as-{given}', 'own')
        summary['per_seed'][seed] = {'cond_minus_blind': arm_gap, 'cond_minus_blind_trainer': arm_gap_trainer,
                                     'label_swap_minus_own': label_swap, 'label_swap_minus_own_trainer': label_swap_trainer}
        deltas.append({m: scored['cond'][m] - scored['blind'][m] for m in METRICS})

    if len(args.seeds) > 1:
        mean_delta = {m: torch.stack([d[m] for d in deltas]).mean(0) for m in METRICS}
        zeros = {m: torch.zeros_like(v) for m, v in mean_delta.items()}
        pooled = compare(mean_delta, zeros, reference)
        print_table(f'cond - blind, per-slice difference averaged over seeds {args.seeds}', pooled, 'diff', '0')
        summary['seed_mean_cond_minus_blind'] = pooled

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
