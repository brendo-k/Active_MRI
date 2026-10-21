"""Does the per-slice conv vs bilinear difference exceed seed-to-seed variation? Four matched CG-SENSE LOUPE R=8 models.

Scores conv / bilinear x seeds 42 / 43 on the full validation set, each model with its own mask and sensitivity
net, on common seeded Bernoulli draws (4 per slice). Contrasts per slice (dB):
  upsampling effect   bilinear - conv, within each seed
  seed effect         seed 43 - seed 42, within each arm (a control: same recipe, different seed)
Reports mean and sd of each, the correlation of the upsampling effect across seeds (does the same slice gain
both times?), per-volume means, and the share of slices where the upsampling effect has the same sign in both seeds.

    python scripts/diagnostics/sens_upsample_seeds_r8.py
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
from activemri.utils.util import compute_psnr_torch
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load

MODELS = {
    'conv42': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
    'bilinear42': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep',
    'conv43': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_seed43',
    'bilinear43': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43',
}
CONTRASTS = {
    'upsampling seed42 (bilinear-conv)': ('bilinear42', 'conv42'),
    'upsampling seed43 (bilinear-conv)': ('bilinear43', 'conv43'),
    'seed effect conv (43-42)': ('conv43', 'conv42'),
    'seed effect bilinear (43-42)': ('bilinear43', 'bilinear42'),
}
SCORING_SEEDS = (0, 1, 2, 3)
OUT_DIR = Path('reports/sens_upsample_r8')


@torch.no_grad()
def psnr(model, part, probability, seed_key):
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    total = torch.zeros(len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        total += compute_psnr_torch(*_apply_signal_mask(output, target, part['metric_mask'])).flatten().cpu() / len(SCORING_SEEDS)
    return total


def stats(x):
    return {'mean': float(x.mean()), 'sd': float(x.std()), 'frac_abs_gt_0.1dB': float((np.abs(x) > 0.1).mean()),
            'frac_abs_gt_0.25dB': float((np.abs(x) > 0.25).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    models = {name: load(Path(path), 'last_model.pt', args.device)[0] for name, path in MODELS.items()}
    cfg = OmegaConf.load(Path(MODELS['conv42']) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)

    scores = {name: [] for name in MODELS}
    anatomy, volume, slice_id, offset = [], [], [], 0
    for batch in loader:
        part = {'kspace': batch['kspace'].to(args.device), 'target': batch['target'].to(args.device)}
        part['metric_mask'] = metric_mask_from_real_kspace(part['kspace'], batch['anatomy'])
        for name, model in models.items():
            scores[name].append(psnr(model, part, unscaled(model, part['kspace'], None), offset))
        anatomy += list(batch['anatomy'])
        volume += [str(v) for v in batch['fname']]
        slice_id += [int(s) for s in batch['slice_id']]
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    scores = {k: torch.cat(v).numpy() for k, v in scores.items()}
    anatomy, volume = np.array(anatomy), np.array(volume)
    deltas = {name: scores[a] - scores[b] for name, (a, b) in CONTRASTS.items()}
    result = {'mean_psnr': {k: {g: float(v[(anatomy == g) if g != 'all' else slice(None)].mean()) for g in ('all', 'brain', 'knee')} for k, v in scores.items()},
              'contrasts': {}, 'correlation': {}, 'sign_agreement': {}, 'volume_means': {}}
    for group in ('all', 'brain', 'knee'):
        rows = slice(None) if group == 'all' else anatomy == group
        result['contrasts'][group] = {name: stats(d[rows]) for name, d in deltas.items()}
        up42, up43 = deltas['upsampling seed42 (bilinear-conv)'][rows], deltas['upsampling seed43 (bilinear-conv)'][rows]
        result['correlation'][group] = {'upsampling effect seed42 vs seed43': float(np.corrcoef(up42, up43)[0, 1]),
                                        'seed effect conv vs seed effect bilinear': float(np.corrcoef(deltas['seed effect conv (43-42)'][rows], deltas['seed effect bilinear (43-42)'][rows])[0, 1])}
        big = (np.abs(up42) > 0.25) & (np.abs(up43) > 0.25)
        result['sign_agreement'][group] = {'slices_abs_gt_0.25_both_seeds': int(big.sum()),
                                           'same_sign_share_of_those': float((np.sign(up42) == np.sign(up43))[big].mean()) if big.any() else None}
    for vol in np.unique(volume):
        rows = volume == vol
        result['volume_means'][Path(vol).name] = {n: float(d[rows].mean()) for n, d in deltas.items() if 'upsampling' in n}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'seeds_comparison.json').write_text(json.dumps(result, indent=1) + '\n')
    np.savez(OUT_DIR / 'seeds_comparison_psnr.npz', anatomy=anatomy, volume=volume, slice_id=np.array(slice_id), **scores)
    print(json.dumps({k: result[k] for k in ('mean_psnr', 'contrasts', 'correlation', 'sign_agreement')}, indent=1))
    top = sorted(result['volume_means'].items(), key=lambda kv: -abs(kv[1]['upsampling seed42 (bilinear-conv)']))[:6]
    for name, v in top:
        print(name, {k: round(x, 3) for k, x in v.items()})


if __name__ == '__main__':
    main()
