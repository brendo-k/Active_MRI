"""Does an input-adaptive sampler predict a patient-specific mask? Mask-swap test against a static LOUPE control.

For each run (the adaptive sampler and its static control), on the whole validation set, each slice is reconstructed by
the run's own reconstructor from masks made with these donors' score maps:
  own              the slice's own map
  other_volume     the map of a random slice of another volume of the same condition (brain / knee-PD / knee-PDFS)
  condition_mean   the mean map of the slice's condition (a static per-condition mask built from the sampler itself)
A donor's *unscaled* scores are re-rescaled to the target slice's own available k-space and budget
(`SequentialSampling.mask_budget`), so a swap moves only the shape of the mask, never its extent or sample count.
Every mask is thresholded with the same per-slice seeded draws (SEEDS), and scored with the trainer's metric mask.

Headline: the adaptive run's other_volume swap cost (own minus other_volume, PSNR), per condition, with 95% volume
bootstrap CIs. A patient-specific sampler has a positive cost. The static control's maps do not depend on the slice,
so its swap costs are exactly 0 (null calibration). Secondary: adaptive own minus control own (different samplers and
recipes, one seed). Also saved: score-map adaptivity, the correlation of each slice's map with its condition mean and
with another volume's, inside the two slices' common acquired k-space.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_patient_specific_r8.py --adaptive <conv run> --control <loupe run> \
        --out reports/patient_specific_r8_local_v1
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
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from eval_adaptive_samplers_r8 import SEEDS, bootstrap, thresholds
from mask_gradient_anatomy_r8 import load, to_device

DONORS = ('own', 'other_volume', 'condition_mean')
METRICS = ('psnr', 'ssim')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--adaptive', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, required=True, help='folder for patient_specific.json / .pt')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def labelled(batch, classes, label_key, device):
    part = to_device(batch, device)
    part['label'] = torch.as_tensor([classes.index(n) for n in batch[label_key]], device=device)
    return part


@torch.no_grad()
def score_maps(model, classes, label_key, loader, device):
    """Per val slice: the sampler's unscaled scores (float16), pre-selected centre and acquired-k-space masks."""
    unscaled, old, valid = [], [], []
    for batch in loader:
        part = labelled(batch, classes, label_key, device)
        proposal = model.propose_mask(part['kspace'], anatomy_labels=part['label'])
        unscaled.append(proposal['unscaled'][:, 0].half().cpu())
        old.append(proposal['old_mask'][:, 0].bool().cpu())
        valid.append(proposal['valid_mask'][:, 0].bool().cpu())
    return torch.cat(unscaled), torch.cat(old), torch.cat(valid)


def donor_scores(unscaled, condition, donors):
    """[N, H, W] unscaled scores each slice receives under each donor assignment."""
    means = {c: unscaled[torch.as_tensor(condition == c)].float().mean(0) for c in np.unique(condition)}
    return {'own': unscaled, 'other_volume': unscaled[donors],
            'condition_mean': torch.stack([means[c] for c in condition]).half()}


@torch.no_grad()
def score_donors(model, classes, label_key, loader, maps, old, valid, device):
    """Per-slice PSNR / SSIM (mean over SEEDS) and acquired fraction for each donor's scores, re-rescaled per slice."""
    count = len(old)
    scores = {name: {m: torch.zeros(count) for m in (*METRICS, 'acquired_fraction')} for name in maps}
    start = 0
    for batch in loader:
        part = labelled(batch, classes, label_key, device)
        rows = torch.arange(start, start + len(part['kspace']))
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        old_rows = old[rows].unsqueeze(1).float().to(device)
        valid_rows = valid[rows].unsqueeze(1).float().to(device)
        budget = model.mask_budget(old_rows, valid_rows)
        for name, unscaled in maps.items():
            scores_rows = unscaled[rows].float().unsqueeze(1).to(device) * (1 - old_rows)
            rescaled = model.sampler.select_locations(scores_rows, old_rows, budget, valid_rows)[0][:, 0].cpu()
            for seed in SEEDS:
                threshold = torch.stack([thresholds(seed, int(i), old.shape[1:]) for i in rows])
                mask = (old[rows] | ((rescaled > threshold) & valid[rows])).float().unsqueeze(1).to(device)
                output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
                masked = _apply_signal_mask(output, target, part['metric_mask'])
                scores[name]['psnr'][rows] += compute_psnr_torch(*masked).flatten().cpu() / len(SEEDS)
                scores[name]['ssim'][rows] += compute_ssim_torch(*masked).flatten().cpu() / len(SEEDS)
                scores[name]['acquired_fraction'][rows] += (
                    (mask * valid_rows).flatten(1).sum(1) / valid_rows.flatten(1).sum(1)).cpu() / len(SEEDS)
        start += len(rows)
    return scores


def correlation_inside(a, b, support):
    a, b = a[support].float(), b[support].float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def adaptivity(unscaled, old, valid, condition, donors):
    """Mean per-slice correlation of the scores with the condition mean and with the other-volume donor's scores,
    inside the common acquired k-space minus the centre."""
    means = {c: unscaled[torch.as_tensor(condition == c)].float().mean(0) for c in np.unique(condition)}
    with_mean, with_donor = [], []
    for i in range(len(unscaled)):
        support = valid[i] & valid[donors[i]] & ~old[i]
        with_mean.append(correlation_inside(unscaled[i], means[condition[i]], support))
        with_donor.append(correlation_inside(unscaled[i], unscaled[donors[i]], support))
    with_mean, with_donor = np.array(with_mean), np.array(with_donor)
    return {c: {'slice_vs_condition_mean': float(with_mean[condition == c].mean()),
                'slice_vs_other_volume': float(with_donor[condition == c].mean())} for c in np.unique(condition)}


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    cfg = OmegaConf.load(args.adaptive / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset if args.limit is None else torch.utils.data.Subset(
        val_loader.dataset, np.linspace(0, len(val_loader.dataset) - 1, args.limit).astype(int).tolist())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.chunk, shuffle=False, num_workers=4,
                                         collate_fn=val_loader.collate_fn)
    condition = np.array(sum((list(b['condition']) for b in loader), []))
    volumes = np.array(sum((list(b['fname']) for b in loader), []))
    donor_rng = np.random.RandomState(7)
    donors = torch.as_tensor([donor_rng.choice(np.flatnonzero((condition == condition[i]) & (volumes != volumes[i])))
                              for i in range(len(condition))])

    scores, adaptivity_stats, runs = {}, {}, {}
    for name, run in (('adaptive', args.adaptive), ('control', args.control)):
        model, checkpoint = load(run, args.checkpoint, args.device)
        options = checkpoint['options']
        if getattr(options, 'budget_over', 'grid') != 'acquired':
            raise ValueError(f"{run} counts R over the grid; this eval re-rescales donors to each slice's acquired budget")
        classes, label_key = list(options.label_classes), options.label_key
        runs[name] = {'run': str(run), 'checkpoint': args.checkpoint, 'epoch': checkpoint['epoch'],
                      'sampler': options.sampler_type}
        print(f"{name}: {run} {args.checkpoint} epoch {checkpoint['epoch']} ({options.sampler_type})", flush=True)
        unscaled, old, valid = score_maps(model, classes, label_key, loader, args.device)
        adaptivity_stats[name] = adaptivity(unscaled, old, valid, condition, donors)
        scores[name] = score_donors(model, classes, label_key, loader,
                                    donor_scores(unscaled, condition, donors), old, valid, args.device)
        for donor in DONORS:
            print(f"  {donor:15s} PSNR " + '  '.join(f"{c} {scores[name][donor]['psnr'][condition == c].mean():.3f}"
                                                       for c in np.unique(condition)), flush=True)

    groups = {**{c: condition == c for c in np.unique(condition)}, 'all': np.ones(len(condition), bool)}
    contrasts = {
        'adaptive_swap_cost_other_volume': (('adaptive', 'own'), ('adaptive', 'other_volume')),
        'adaptive_gain_over_condition_mean': (('adaptive', 'own'), ('adaptive', 'condition_mean')),
        'control_swap_cost_other_volume': (('control', 'own'), ('control', 'other_volume')),
        'adaptive_minus_control_own': (('adaptive', 'own'), ('control', 'own')),
    }
    summary = {'runs': runs, 'seeds': list(SEEDS), 'n_slices': {g: int(k.sum()) for g, k in groups.items()},
               'n_volumes': {g: int(len(np.unique(volumes[k]))) for g, k in groups.items()},
               'means': {f'{r}/{d}': {m: {g: float(scores[r][d][m][k].mean()) for g, k in groups.items()}
                                      for m in (*METRICS, 'acquired_fraction')} for r in scores for d in DONORS},
               'adaptivity': adaptivity_stats, 'contrasts': {}}
    for contrast, ((run_a, donor_a), (run_b, donor_b)) in contrasts.items():
        summary['contrasts'][contrast] = {
            m: {g: bootstrap((scores[run_a][donor_a][m] - scores[run_b][donor_b][m]).numpy()[k], volumes[k], rng)
                for g, k in groups.items()} for m in METRICS}
        psnr = summary['contrasts'][contrast]['psnr']
        print(f'{contrast:36s} ' + '  '.join(f'{g} {psnr[g]["mean"]:+.3f} [{psnr[g]["ci95"][0]:+.3f}, {psnr[g]["ci95"][1]:+.3f}]'
                                              for g in groups), flush=True)
    if args.limit is None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'patient_specific.json').write_text(json.dumps(summary, indent=2) + '\n')
        torch.save({'condition': condition.tolist(), 'volume': volumes.tolist(), 'donors': donors,
                    'scores': {r: {d: {m: v.half() for m, v in s.items()} for d, s in per.items()} for r, per in scores.items()}},
                   args.out / 'patient_specific.pt')


if __name__ == '__main__':
    main()
