"""CG iterations and Tikhonov weight needed by CG-SENSE with BART ecalib maps (R=8, brain and knee).

Reconstructs validation slices through the ecalib_c24 run's own reconstructor path (cached ecalib maps, seeded Bernoulli draws of
that run's learned mask, trainer-path background-masked PSNR) while sweeping `num_iters` x `lamda`. Reports per anatomy the mean
PSNR / NMSE of each cell, plus the fewest iterations within `--tol` dB of that lambda's 200-iteration solution.

    python scripts/diagnostics/cgsense_iters_lambda_r8.py --stride 4
"""
import argparse
import json
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load

RUN = Path('checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24')
OUT_DIR = Path('reports/cgsense_iters_lambda_r8')
ITERS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200)
LAMBDAS = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
SCORING_SEEDS = (0, 1)


def nmse(output, target):
    return ((target - output) ** 2).flatten(1).sum(1) / (target ** 2).flatten(1).sum(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stride', type=int, default=4, help='keep every stride-th validation slice')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--tol', type=float, default=0.02, help='dB below the 200-iteration solution that counts as converged')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    model = load(RUN, 'last_model.pt', args.device)[0]
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    subset = Subset(val_loader.dataset, range(0, len(val_loader.dataset), args.stride))
    loader = DataLoader(subset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)

    cells = [(n, l) for l in LAMBDAS for n in ITERS]
    psnr = {c: [] for c in cells}
    err = {c: [] for c in cells}
    anatomy, offset = [], 0
    for batch in loader:
        kspace = batch['kspace'].to(args.device)
        target = batch['target'].to(args.device)
        maps = batch['sense_maps'].to(args.device)
        metric_mask = metric_mask_from_real_kspace(kspace, batch['anatomy'])
        probability = unscaled(model, kspace, None)
        old_mask, valid_mask = model._initial_masks(kspace)
        masks = []
        for seed in SCORING_SEEDS:
            torch.manual_seed(seed * 100003 + offset)
            masks.append(model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1])
        tgt = target if target.dim() == 4 else target.unsqueeze(1)
        for (n, l) in cells:
            model.reconstructor.num_iters, model.reconstructor.lamda = n, l
            p = torch.zeros(len(kspace), device=args.device)
            e = torch.zeros_like(p)
            for mask in masks:
                with torch.no_grad():
                    out = model(target, kspace, fixed_mask=mask, sense_maps=maps)['output']
                o, t = _apply_signal_mask(out, tgt, metric_mask)
                p += compute_psnr_torch(o, t).flatten() / len(masks)
                e += nmse(o, t) / len(masks)
            psnr[(n, l)].append(p.cpu())
            err[(n, l)].append(e.cpu())
        anatomy += list(batch['anatomy'])
        offset += len(kspace)
        print(f'{offset}/{len(subset)}', flush=True)

    anatomy = np.array(anatomy)
    results = {}
    for name in sorted(set(anatomy)):
        sel = anatomy == name
        table = {f'{n}/{l:g}': {'psnr': float(torch.cat(psnr[(n, l)]).numpy()[sel].mean()),
                                'nmse': float(torch.cat(err[(n, l)]).numpy()[sel].mean())} for n, l in cells}
        results[name] = {'n_slices': int(sel.sum()), 'cells': table}
        print(f'\n== {name} ({sel.sum()} slices): mean PSNR, rows = iterations, cols = lambda')
        print('iters ' + ' '.join(f'{l:>8g}' for l in LAMBDAS))
        for n in ITERS:
            print(f'{n:>5} ' + ' '.join(f'{table[f"{n}/{l:g}"]["psnr"]:8.3f}' for l in LAMBDAS))
        best = max(table, key=lambda k: table[k]['psnr'])
        print(f'best cell {best}: {table[best]["psnr"]:.3f} dB')
        for l in LAMBDAS:
            top = table[f'{ITERS[-1]}/{l:g}']['psnr']
            needed = next(n for n in ITERS if table[f'{n}/{l:g}']['psnr'] >= top - args.tol)
            print(f'lambda {l:g}: 200-it {top:.3f} dB, within {args.tol} dB from {needed} iterations')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
