"""Per-slice comparison of conv vs bilinear sensitivity-U-Net upsampling (matched 50-epoch CG-SENSE LOUPE R=8 runs).

Final-epoch masks and sense nets can differ even when mean PSNR matches. This scores the full validation set with
every (reconstructor, mask) pair of the two runs on common seeded Bernoulli draws, then characterises the per-slice
PSNR difference bilinear - conv:
  full         each run's own mask + sense net
  mask only    bilinear mask vs conv mask, conv sense net
  sense only   bilinear sense net vs conv sense net, conv mask
and stratifies it by anatomy, contrast, slice position within the volume, support fraction and PSNR level.
Noise floor: the same conv model scored with two different sets of Bernoulli draws.

    python scripts/diagnostics/sens_upsample_slice_diff_r8.py
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
from anatcond_cgsense_cross_eval_r8 import mask_stats, unscaled
from mask_gradient_anatomy_r8 import load

RUNS = {'conv': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared'),
        'bilinear': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep')}
OUT_DIR = Path('reports/sens_upsample_r8')
SCORING_SEEDS = (0, 1, 2, 3)
BORDER_PX = 24


@torch.no_grad()
def psnr_per_seed(model, part, probability, seed_key):
    """[seeds, batch] trainer-path PSNR, one Bernoulli draw per seed."""
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    rows = []
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        rows.append(compute_psnr_torch(*_apply_signal_mask(output, target, part['metric_mask'])).flatten().cpu())
    return torch.stack(rows)


def slice_metadata(dataset):
    """Per global index: volume name, slice id, slice position (0-1) within its volume."""
    names, ids = [], []
    for part in dataset.datasets:
        for fname, slice_id in part.examples:
            names.append(str(fname))
            ids.append(int(slice_id))
    ids = np.array(ids)
    names = np.array(names)
    position = np.zeros(len(ids))
    for name in np.unique(names):
        rows = names == name
        position[rows] = (ids[rows] - ids[rows].min()) / max(ids[rows].max() - ids[rows].min(), 1)
    return names, ids, position


def summarise(delta, noise):
    return {'n': int(len(delta)), 'mean': float(delta.mean()), 'sd': float(delta.std()),
            'p5': float(np.percentile(delta, 5)), 'p95': float(np.percentile(delta, 95)),
            'frac_abs_gt_0.1dB': float((np.abs(delta) > 0.1).mean()), 'noise_sd': float(noise.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    models = {name: load(path, 'last_model.pt', args.device)[0] for name, path in RUNS.items()}
    cfg = OmegaConf.load(RUNS['conv'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)
    names, slice_ids, position = slice_metadata(val_loader.dataset)

    combos = {'conv': ('conv', 'conv'), 'bilinear': ('bilinear', 'bilinear'),
              'conv_sense+bilinear_mask': ('conv', 'bilinear'), 'bilinear_sense+conv_mask': ('bilinear', 'conv')}
    psnr = {k: [] for k in combos}
    anatomy, condition, support, offset = [], [], [], 0
    for batch in loader:
        part = {'kspace': batch['kspace'].to(args.device), 'target': batch['target'].to(args.device)}
        part['metric_mask'] = metric_mask_from_real_kspace(part['kspace'], batch['anatomy'])
        maps = {name: unscaled(model, part['kspace'], None) for name, model in models.items()}
        for key, (recon, mask) in combos.items():
            psnr[key].append(psnr_per_seed(models[recon], part, maps[mask], offset))
        anatomy += list(batch['anatomy'])
        condition += list(batch.get('condition', batch['anatomy']))
        support += part['metric_mask'].flatten(1).mean(1).tolist()
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    psnr = {k: torch.cat(v, dim=1).numpy() for k, v in psnr.items()}
    mean_psnr = {k: v.mean(0) for k, v in psnr.items()}
    anatomy, condition, support = np.array(anatomy), np.array(condition), np.array(support)
    noise = psnr['conv'][0] - psnr['conv'][1]
    contrasts = {'full (bilinear - conv)': mean_psnr['bilinear'] - mean_psnr['conv'],
                 'mask only': mean_psnr['conv_sense+bilinear_mask'] - mean_psnr['conv'],
                 'sense only': mean_psnr['bilinear_sense+conv_mask'] - mean_psnr['conv']}

    result = {'overall': {}, 'by_anatomy': {}, 'by_condition': {}, 'by_position_tercile': {}, 'by_support_tercile': {},
              'by_psnr_tercile': {}, 'worst_best_slices': {}}
    full = contrasts['full (bilinear - conv)']
    groups = {'overall': np.ones(len(full), bool)}
    groups.update({f'anatomy={a}': anatomy == a for a in np.unique(anatomy)})
    groups.update({f'condition={c}': condition == c for c in np.unique(condition) if c not in np.unique(anatomy)})
    for label, values in (('position', position), ('support', support), ('psnr', mean_psnr['conv'])):
        edges = np.percentile(values, [33.3, 66.7])
        for i, name in enumerate(('low', 'mid', 'high')):
            groups[f'{label}={name}'] = np.digitize(values, edges) == i
    result['strata'] = {g: {c: summarise(d[rows], noise[rows]) for c, d in contrasts.items()} for g, rows in groups.items()}
    result['noise_floor_sd_same_model_two_draws'] = float(noise.std())
    result['corr_full_delta_with'] = {'support': float(np.corrcoef(full, support)[0, 1]), 'position': float(np.corrcoef(full, position)[0, 1]),
                                      'conv_psnr': float(np.corrcoef(full, mean_psnr['conv'])[0, 1])}
    order = np.argsort(full)
    describe = lambda i: {'volume': names[i], 'slice': int(slice_ids[i]), 'anatomy': str(anatomy[i]), 'delta_dB': float(full[i]),
                          'conv_psnr': float(mean_psnr['conv'][i]), 'support': float(support[i])}
    result['worst_best_slices'] = {'bilinear_worse': [describe(i) for i in order[:8]], 'bilinear_better': [describe(i) for i in order[::-1][:8]]}

    probe = next(iter(DataLoader(val_loader.dataset, batch_size=1, collate_fn=val_loader.collate_fn)))
    kspace = probe['kspace'].to(args.device)
    probabilities = {n: unscaled(m, kspace, None)[0, 0].float().cpu() for n, m in models.items()}
    result['mask_stats'] = mask_stats(probabilities)
    size = probabilities['conv'].shape[0]
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    border = (torch.minimum(torch.minimum(yy, size - 1 - yy), torch.minimum(xx, size - 1 - xx)) < BORDER_PX)
    result['mask_stats']['border_mean_probability'] = {n: float(p[border].mean()) for n, p in probabilities.items()}
    result['mask_stats']['abs_diff_mean'] = float((probabilities['conv'] - probabilities['bilinear']).abs().mean())
    result['mask_stats']['abs_diff_max'] = float((probabilities['conv'] - probabilities['bilinear']).abs().max())
    result['mask_stats']['rel_abs_diff'] = result['mask_stats']['abs_diff_mean'] / float(probabilities['conv'].mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'slice_diff.json').write_text(json.dumps(result, indent=1) + '\n')
    np.savez(OUT_DIR / 'slice_diff_psnr.npz', anatomy=anatomy, condition=condition, volume=names, slice_id=slice_ids,
             position=position, support=support, **mean_psnr)
    torch.save(probabilities, OUT_DIR / 'final_masks.pt')
    for group, rows in result['strata'].items():
        print(f"{group:<22}", ' | '.join(f"{c.split(' ')[0]} {s['mean']:+.3f}±{s['sd']:.3f}" for c, s in rows.items()),
              f"| noise sd {rows['mask only']['noise_sd']:.3f} n={rows['mask only']['n']}")
    print(json.dumps({k: result[k] for k in ('noise_floor_sd_same_model_two_draws', 'corr_full_delta_with', 'mask_stats')}, indent=1))


if __name__ == '__main__':
    main()
