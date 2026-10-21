"""Are the CG-SENSE iteration count and Tikhonov weight tuned? Evaluation-only sweep with the trained run fixed.

Keeps a trained CG-SENSE LOUPE R=8 checkpoint (mask + sensitivity net) and re-scores the full validation set with
different `num_iters` / `lamda` in the reconstructor, on common seeded Bernoulli draws (same draws for every
setting). Trainer-path (signal-masked) PSNR / SSIM / NMSE, per-slice means by anatomy. The masks were learned
under the training setting (10 iterations, lamda 0), so a better setting here is a lower bound on what retraining
would give; an equal-or-worse setting does not exclude retraining gains.

    python scripts/diagnostics/cgsense_hyperparam_sweep_r8.py --run checkpoints/loupe2d_cgsense_r8_brain_knee_shared
"""
import argparse
import json
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load

ITERATIONS = (3, 5, 10, 20, 40, 80)
LAMDAS = (1e-4, 1e-3, 1e-2, 1e-1)
LAMDA_ITERATIONS = (10, 40)
SCORING_SEEDS = (0, 1)
OUT_DIR = Path('reports/cgsense_hyperparams_r8')


def settings():
    return [(n, 0.0) for n in ITERATIONS] + [(n, lam) for n in LAMDA_ITERATIONS for lam in LAMDAS]


def nmse(output, target):
    return ((target - output) ** 2).flatten(1).sum(1) / (target ** 2).flatten(1).sum(1)


@torch.no_grad()
def score(model, part, probability, seed_key):
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    total = torch.zeros(3, len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        out_m, tgt_m = _apply_signal_mask(output, target, part['metric_mask'])
        total += torch.stack([compute_psnr_torch(out_m, tgt_m).flatten(), compute_ssim_torch(out_m, tgt_m).flatten(),
                              nmse(out_m, tgt_m).flatten()]).cpu() / len(SCORING_SEEDS)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared'))
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    model, _ = load(args.run, 'last_model.pt', args.device)
    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)

    grid, scores, anatomy, offset = settings(), {s: [] for s in settings()}, [], 0
    for batch in loader:
        part = {'kspace': batch['kspace'].to(args.device), 'target': batch['target'].to(args.device)}
        part['metric_mask'] = metric_mask_from_real_kspace(part['kspace'], batch['anatomy'])
        probability = unscaled(model, part['kspace'], None)
        for iters, lamda in grid:
            model.reconstructor.num_iters, model.reconstructor.lamda = iters, lamda
            scores[(iters, lamda)].append(score(model, part, probability, offset))
        anatomy += list(batch['anatomy'])
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    anatomy = np.array(anatomy)
    result = {'run': str(args.run), 'baseline_setting': [10, 0.0], 'settings': {}}
    reference = torch.cat(scores[(10, 0.0)], dim=1).numpy()
    for (iters, lamda), chunks in scores.items():
        values = torch.cat(chunks, dim=1).numpy()
        entry = {}
        for group in ('all', 'brain', 'knee'):
            rows = np.ones(len(anatomy), bool) if group == 'all' else anatomy == group
            entry[group] = {'psnr': float(values[0][rows].mean()), 'ssim': float(values[1][rows].mean()), 'nmse': float(values[2][rows].mean()),
                            'psnr_delta_vs_10_0': float((values[0] - reference[0])[rows].mean()), 'psnr_delta_sd': float((values[0] - reference[0])[rows].std())}
        result['settings'][f'iters={iters},lamda={lamda:g}'] = entry
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{args.run.name}.json'
    out.write_text(json.dumps(result, indent=1) + '\n')
    for name, entry in result['settings'].items():
        print(f"{name:<22}", ' | '.join(f"{g} {e['psnr']:.3f} ({e['psnr_delta_vs_10_0']:+.3f}) ssim {e['ssim']:.4f}" for g, e in entry.items()))


if __name__ == '__main__':
    main()
