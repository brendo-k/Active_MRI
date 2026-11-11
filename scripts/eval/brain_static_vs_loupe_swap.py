"""Mask-swap scores of the brain_static_vs_loupe runs: every recipient VarNet with every donor run's brain mask.

The donor's budget-rescaled brain probability map for each slice is thresholded with the common per-slice
Bernoulli draws of brain_static_vs_loupe_eval.py (SEEDS 1234 and 4321) and passed to the recipient as
`fixed_mask`, so recipient == donor reproduces eval.json. Donors must agree on the always-acquired centre and
the valid region of every slice (asserted), so swapping changes only the probability map.

  --split val     full brain val (limit_volumes=1), as in eval.json
  --split train   the quarter of the brain train set the runs trained on (the run config's limit_volumes)
  --arms          the donor arms; the follow-up (followup_proposal.md) adds frozen_learned and static_repeat

Writes one file per recipient, <out>/swap_<split>_<recipient>.pt, holding per-slice PSNR/SSIM/NMSE for each
donor and the slice volumes. brain_static_vs_loupe_swap_summary.py turns them into tables.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_static_vs_loupe_swap.py \
        --queue ~/data/seqmri_checkpoints/brain_static_vs_loupe --recipients static_s42 static_s43
"""
import argparse
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
from brain_static_vs_loupe_eval import ARMS, with_brain_label
from eval_adaptive_samplers_r8 import SEEDS, thresholds
from mask_gradient_anatomy_r8 import load

METRICS = ('psnr', 'ssim', 'nmse')
OUT = Path('reports/brain_static_vs_loupe/swap')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, required=True, help='folder holding <arm>_s<seed>/ run folders')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--recipients', nargs='+', required=True, help='run names, e.g. static_s42')
    parser.add_argument('--arms', nargs='+', default=list(ARMS),
                        help='arms whose runs are donors; a seed without a run of an arm is skipped')
    parser.add_argument('--donors', choices=('all', 'same_seed', 'same_seed_and_uniform'), default='all',
                        help='every run; the recipient seed\'s arms; or those plus every loupe_uniform run')
    parser.add_argument('--split', choices=('val', 'train'), default='val')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def brain_loader(run, split, chunk, limit):
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    if split == 'val':
        cfg.datasets.limit_volumes = 1
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    dataset = (train_loader if split == 'train' else val_loader).dataset
    dataset = torch.utils.data.ConcatDataset([part for part in dataset.datasets if part[0]['anatomy'] == 'brain'])
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, np.linspace(0, len(dataset) - 1, limit).astype(int).tolist())
    return torch.utils.data.DataLoader(dataset, batch_size=chunk, shuffle=False, num_workers=4,
                                       collate_fn=val_loader.collate_fn)


def seed_of(name):
    return int(name.rsplit('_s', 1)[1])


@torch.no_grad()
def score_swaps(models, classes, recipients, donors_of, loader, device):
    """Per-slice metrics of each recipient with each of its donors' masks, averaged over the SEEDS draws."""
    scores = {r: {d: {m: [] for m in METRICS} for d in donors_of[r]} for r in recipients}
    donors = sorted({d for r in recipients for d in donors_of[r]})
    start, volumes = 0, []
    for batch in loader:
        volumes += [str(v) for v in batch['fname']]
        part = with_brain_label(batch, classes, device)
        proposals = {d: models[d].propose_mask(part['kspace'], anatomy_labels=part['label']) for d in donors}
        reference = proposals[donors[0]]
        old, valid = reference['old_mask'][:, 0].bool(), reference['valid_mask'][:, 0].bool()
        for d in donors:
            assert torch.equal(proposals[d]['old_mask'][:, 0].bool(), old), f'{d} acquires a different centre'
            assert torch.equal(proposals[d]['valid_mask'][:, 0].bool(), valid), f'{d} has a different valid region'
        rows = torch.arange(start, start + len(old))
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        batch_scores = {r: {d: {m: torch.zeros(len(rows)) for m in METRICS} for d in donors_of[r]} for r in recipients}
        for seed in SEEDS:
            threshold = torch.stack([thresholds(seed, int(i), old.shape[1:]) for i in rows]).to(device)
            masks = {d: (old | ((proposals[d]['rescaled'][:, 0].float() > threshold) & valid)).float().unsqueeze(1)
                     for d in donors}
            for r in recipients:
                for d in donors_of[r]:
                    output = models[r](part['target'], part['kspace'], anatomy_labels=part['label'],
                                       fixed_mask=masks[d])['output']
                    output = output[-1] if isinstance(output, list) else output
                    out_m, tgt_m = _apply_signal_mask(output, target, part['metric_mask'])
                    nmse = ((out_m - tgt_m) ** 2).flatten(1).sum(1) / (tgt_m ** 2).flatten(1).sum(1)
                    batch_scores[r][d]['psnr'] += compute_psnr_torch(out_m, tgt_m).flatten().cpu() / len(SEEDS)
                    batch_scores[r][d]['ssim'] += compute_ssim_torch(out_m, tgt_m).flatten().cpu() / len(SEEDS)
                    batch_scores[r][d]['nmse'] += nmse.cpu() / len(SEEDS)
        for r in recipients:
            for d in donors_of[r]:
                for m in METRICS:
                    scores[r][d][m].append(batch_scores[r][d][m])
        start += len(rows)
        print(f'{start} slices', flush=True)
    return {r: {d: {m: torch.cat(v) for m, v in by_metric.items()} for d, by_metric in by_donor.items()}
            for r, by_donor in scores.items()}, volumes


def main():
    args = parse_args()
    names = [f'{arm}_s{seed}' for seed in args.seeds for arm in args.arms if (args.queue / f'{arm}_s{seed}').is_dir()]
    donors_of = {r: [d for d in names if args.donors == 'all' or seed_of(d) == seed_of(r)
                     or (args.donors == 'same_seed_and_uniform' and d.startswith('loupe_uniform'))]
                 for r in args.recipients}
    needed = sorted(set(args.recipients) | {d for ds in donors_of.values() for d in ds})
    models, classes = {}, None
    for name in needed:
        models[name], checkpoint = load(args.queue / name, args.checkpoint, args.device)
        classes = list(checkpoint['options'].label_classes)
    loader = brain_loader(args.queue / args.recipients[0], args.split, args.chunk, args.limit)
    scores, volumes = score_swaps(models, classes, args.recipients, donors_of, loader, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    for r in args.recipients:
        for d in donors_of[r]:
            print(f'{r:18s} with mask of {d:18s} PSNR {scores[r][d]["psnr"].mean():.4f}', flush=True)
        if args.limit is None:
            torch.save({'recipient': r, 'split': args.split, 'checkpoint': args.checkpoint, 'thresholds': list(SEEDS),
                        'volume': volumes, 'scores': scores[r]},
                       args.out / f'swap_{args.split}_{r}.pt')


if __name__ == '__main__':
    main()
