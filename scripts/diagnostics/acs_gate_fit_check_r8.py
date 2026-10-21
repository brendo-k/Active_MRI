"""Validity check of the ACS gate's per-volume fits: does an in-sample fit (`grid_in`, `mask_in`) improve what it is fitted on?

In the first gate run (`acs_mask_gate_cgsense_r8.py`, L1 objective, grid lr 0.02) `grid_in` lost to the population
mask on all 10 knee volumes although it is fitted on the slices it is scored on. This refits `grid_in` on chosen
volumes (all F and H slices, as in the gate) under several fitting configurations and every `--every` steps records,
on those same slices with the gate's seeded scoring draws:
  - the full-image L1 (the population run's training loss);
  - the background-masked PSNR (the gate's metric).
L1 falling while PSNR falls means the objective disagrees with the metric; L1 not falling means the fit itself fails.

    python scripts/diagnostics/acs_gate_fit_check_r8.py --anatomy knee --volumes file1000662 file1000401 file1001437
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from acs_mask_gate_cgsense_r8 import (ANATOMY_INDEX, OUT_DIR, RUN, SCORING_SEEDS, Slices, bernoulli_masks, evenly, fit,
                                      logit_of, volume_groups)
from mask_gradient_anatomy_r8 import load
from subject_coadapt_ceiling_r8 import DensityFamily

CONFIGS = {'l1_lr0.02': ('grid', 'l1', 0.02), 'l1_lr0.002': ('grid', 'l1', 0.002), 'psnr_lr0.02': ('grid', 'psnr', 0.02),
           'grid_lr0.002': ('grid', 'l1', 0.002), 'grid_lr0.005': ('grid', 'l1', 0.005),
           'free_lr0.05': ('free', 'l1', 0.05), 'free_lr0.005': ('free', 'l1', 0.005)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anatomy', choices=tuple(ANATOMY_INDEX), required=True)
    parser.add_argument('--volumes', nargs='+', required=True, help='volume name prefixes')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS)[:3])
    parser.add_argument('--tag', default='', help='suffix of the output json')
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--every', type=int, default=25)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--gap', type=int, default=2)
    parser.add_argument('--max-fit', type=int, default=16)
    parser.add_argument('--max-score', type=int, default=16)
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, logits, slices, locals_):
    """(mean full-image L1, mean background-masked PSNR) on `locals_` over the gate's seeded scoring draws."""
    part = slices.batch(locals_)
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    l1, psnr = 0.0, 0.0
    for seed in SCORING_SEEDS:
        masks = []
        for row, local in enumerate(locals_):
            torch.manual_seed(seed * 100003 + slices.offset + local)
            masks.append(bernoulli_masks(model, logits, part['kspace'][row:row + 1]))
        output = model(part['target'], part['kspace'], sense_maps=part['sense_maps'], fixed_mask=torch.cat(masks))['output']
        l1 += float(F.l1_loss(output, target)) / len(SCORING_SEEDS)
        psnr += float(compute_psnr_torch(*_apply_signal_mask(output, target, part['metric_mask'])).mean()) / len(SCORING_SEEDS)
    return l1, psnr


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    base, _ = load(args.run, 'last_model.pt', args.device)
    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    a = ANATOMY_INDEX[args.anatomy]
    val_part = val_loader.dataset.datasets[a]
    val = Slices(val_part, sum(len(d) for d in val_loader.dataset.datasets[:a]), val_loader.collate_fn, args.device)
    with torch.no_grad():
        base_logits = logit_of(base.propose_mask(val.batch([0])['kspace'])['unscaled'])
    grid = DensityFamily('grid', base_logits, 8.0, args.grid_size)
    groups = volume_groups(val_part)

    out = OUT_DIR / f'fit_check_{args.anatomy}{"_" + args.tag if args.tag else ""}.json'
    result = json.loads(out.read_text()) if out.exists() else {}
    for prefix in args.volumes:
        name = next(n for n in groups if n.startswith(prefix))
        members = groups[name]
        locals_ = evenly(members[:len(members) // 2], args.max_fit) + evenly(members[len(members) // 2 + args.gap:], args.max_score)
        base_l1, base_psnr = evaluate(base, base_logits, val, locals_)
        result.setdefault(name, {})['base'] = {'l1': base_l1, 'psnr': base_psnr}
        for config in args.configs:
            family, loss_type, lr = CONFIGS[config]
            trace = []

            def on_step(step, logits):
                if step % args.every == 0:
                    trace.append([step, *evaluate(base, logits, val, locals_)])

            start, to_logits = (grid.init(), grid) if family == 'grid' else (base_logits, lambda z: z)
            fit(base, loss_type, start, to_logits, val, locals_, args, lr, on_step=on_step)
            result[name][config] = trace
            final = trace[-1]
            best = max(trace, key=lambda t: t[2])
            print(f'{name} {config}: L1 {base_l1:.5f} -> {final[1]:.5f} ({100 * (final[1] / base_l1 - 1):+.2f}%), '
                  f'PSNR {final[2] - base_psnr:+.3f} dB at step {final[0]} (best {best[2] - base_psnr:+.3f} at {best[0]})', flush=True)
        val.forget()
        out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
