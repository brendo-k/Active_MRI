"""Per-slice comparison of the conv / bilinear sense-net LOUPE masks against the ESPIRiT (BART ecalib) LOUPE masks.

Four 50-epoch CG-SENSE R=8 LOUPE runs on the same 688 validation slices: `conv` and `bilinear` learn their own
sensitivity maps (U-Net), `ecalib_c10` and `ecalib_c24` use cached BART ecalib maps (10x10 / 24x24 preselected centre).
Each mask is drawn with its owner's sampler (so its own centre is kept) on common seeded Bernoulli draws and scored with
three reconstructors, giving per-slice PSNR for:
  own          each run's own mask + own maps
  ecalib maps  every mask reconstructed with the cached ecalib maps (isolates mask quality)
  conv net     every mask reconstructed with the conv run's sense net
Note: the ecalib maps come from a 24x24 calibration, so masks with only a 10x10 centre leak unsampled k-space into them.

    python scripts/diagnostics/sens_upsample_vs_ecalib_r8.py
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
from sens_upsample_slice_diff_r8 import slice_metadata

RUNS = {'conv': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
        'bilinear': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep',
        'ecalib_c10': 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c10',
        'ecalib_c24': 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24'}
OWN_RECON_MAPS = {'conv': None, 'bilinear': None, 'ecalib_c10': 'cache', 'ecalib_c24': 'cache'}
OUT_DIR = Path('reports/sens_upsample_r8')
SCORING_SEEDS = (0, 1, 2, 3)


@torch.no_grad()
def psnr(recon, owner, part, probability, sense_maps, seed_key):
    """Per-slice trainer-path PSNR over SCORING_SEEDS draws; the mask is drawn by `owner`, reconstructed by `recon`."""
    old_mask, valid_mask = owner._initial_masks(part['kspace'])
    total = torch.zeros(len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = owner.sampler.select_locations(probability, old_mask, owner.budget, valid_mask)[1]
        output = recon(part['target'], part['kspace'], fixed_mask=mask, sense_maps=sense_maps)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        total += compute_psnr_torch(*_apply_signal_mask(output, target, part['metric_mask'])).flatten().cpu() / len(SCORING_SEEDS)
    return total


def stratum(delta, volume, rows):
    d, v = delta[rows], volume[rows]
    volume_means = np.array([d[v == u].mean() for u in np.unique(v)])
    return {'n': int(rows.sum()), 'mean': float(d.mean()), 'sd': float(d.std()), 'volume_mean_sd': float(volume_means.std()),
            'frac_positive': float((d > 0).mean()), 'frac_abs_gt_0.25dB': float((np.abs(d) > 0.25).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    models = {name: load(Path(path), 'last_model.pt', args.device)[0] for name, path in RUNS.items()}
    cfg = OmegaConf.load(Path(RUNS['ecalib_c10']) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)
    names, slice_ids, position = slice_metadata(val_loader.dataset)
    reference = np.load(OUT_DIR / 'slice_diff_psnr.npz')
    assert (reference['volume'] == names).all() and (reference['slice_id'] == slice_ids).all(), 'validation slice order changed'

    scores = {}
    anatomy, condition, support, offset = [], [], [], 0
    for batch in loader:
        part = {'kspace': batch['kspace'].to(args.device), 'target': batch['target'].to(args.device)}
        part['metric_mask'] = metric_mask_from_real_kspace(part['kspace'], batch['anatomy'])
        cache = batch['sense_maps'].to(args.device)
        masks = {name: unscaled(model, part['kspace'], None) for name, model in models.items()}
        for owner_name, owner in models.items():
            own_maps = cache if OWN_RECON_MAPS[owner_name] else None
            scores.setdefault(f'own/{owner_name}', []).append(
                psnr(owner, owner, part, masks[owner_name], own_maps, offset))
            scores.setdefault(f'ecalib_maps/{owner_name}', []).append(
                psnr(models['ecalib_c10'], owner, part, masks[owner_name], cache, offset))
            scores.setdefault(f'conv_net/{owner_name}', []).append(
                psnr(models['conv'], owner, part, masks[owner_name], None, offset))
        anatomy += list(batch['anatomy'])
        condition += list(batch.get('condition', batch['anatomy']))
        support += part['metric_mask'].flatten(1).mean(1).tolist()
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    scores = {k: torch.cat(v).numpy() for k, v in scores.items()}
    anatomy, condition, support = np.array(anatomy), np.array(condition), np.array(support)

    groups = {'all': np.ones(len(anatomy), bool)}
    groups.update({f'anatomy={a}': anatomy == a for a in np.unique(anatomy)})
    groups.update({f'condition={c}': condition == c for c in np.unique(condition) if c not in np.unique(anatomy)})
    edges = np.percentile(support, [33.3, 66.7])
    groups.update({f'support={n}': np.digitize(support, edges) == i for i, n in enumerate(('low', 'mid', 'high'))})
    contrasts = {
        'own: bilinear - conv': ('own/bilinear', 'own/conv'),
        'own: ecalib_c10 - conv': ('own/ecalib_c10', 'own/conv'),
        'own: ecalib_c10 - bilinear': ('own/ecalib_c10', 'own/bilinear'),
        'own: ecalib_c24 - ecalib_c10': ('own/ecalib_c24', 'own/ecalib_c10'),
        'ecalib maps, mask bilinear - conv': ('ecalib_maps/bilinear', 'ecalib_maps/conv'),
        'ecalib maps, mask ecalib_c10 - conv': ('ecalib_maps/ecalib_c10', 'ecalib_maps/conv'),
        'ecalib maps, mask ecalib_c10 - bilinear': ('ecalib_maps/ecalib_c10', 'ecalib_maps/bilinear'),
        'ecalib maps, mask ecalib_c24 - ecalib_c10': ('ecalib_maps/ecalib_c24', 'ecalib_maps/ecalib_c10'),
        'conv net, mask ecalib_c10 - conv': ('conv_net/ecalib_c10', 'conv_net/conv'),
        'conv net, mask bilinear - conv': ('conv_net/bilinear', 'conv_net/conv'),
        'maps only (conv mask): ecalib - conv net': ('ecalib_maps/conv', 'conv_net/conv'),
        'maps only (ecalib_c10 mask): ecalib - conv net': ('ecalib_maps/ecalib_c10', 'conv_net/ecalib_c10'),
    }
    result = {'mean_psnr': {k: {g: float(v[rows].mean()) for g, rows in groups.items()} for k, v in scores.items()},
              'contrasts': {name: {g: stratum(scores[a] - scores[b], names, rows) for g, rows in groups.items()}
                            for name, (a, b) in contrasts.items()}}

    # slice-by-slice agreement between the contrasts, and which volumes drive the ecalib - conv gap
    own_gap = scores['own/ecalib_c10'] - scores['own/conv']
    mask_gap = scores['ecalib_maps/ecalib_c10'] - scores['ecalib_maps/conv']
    maps_gap = scores['ecalib_maps/conv'] - scores['conv_net/conv']
    result['corr_per_slice'] = {'own gap vs maps-only gap': float(np.corrcoef(own_gap, maps_gap)[0, 1]),
                                'own gap vs mask-only gap': float(np.corrcoef(own_gap, mask_gap)[0, 1]),
                                'own gap vs support': float(np.corrcoef(own_gap, support)[0, 1]),
                                'bilinear-conv vs ecalib-conv (own)': float(np.corrcoef(scores['own/bilinear'] - scores['own/conv'], own_gap)[0, 1])}
    volume_gap = {Path(u).name: float(own_gap[names == u].mean()) for u in np.unique(names)}
    result['volume_mean_own_gap_ecalib_c10_minus_conv'] = dict(sorted(volume_gap.items(), key=lambda kv: kv[1]))

    probe = next(iter(DataLoader(val_loader.dataset, batch_size=1, collate_fn=val_loader.collate_fn)))
    kspace = probe['kspace'].to(args.device)
    probabilities = {n: unscaled(m, kspace, None)[0, 0].float().cpu() for n, m in models.items()}
    result['mask_stats'] = mask_stats(probabilities)
    torch.save(probabilities, OUT_DIR / 'ecalib_comparison_masks.pt')
    np.savez(OUT_DIR / 'ecalib_comparison_psnr.npz', anatomy=anatomy, condition=condition, volume=names, slice_id=slice_ids,
             position=position, support=support, **{k.replace('/', '__'): v for k, v in scores.items()})
    (OUT_DIR / 'ecalib_comparison.json').write_text(json.dumps(result, indent=1) + '\n')

    print('mean PSNR (all):', {k: round(v['all'], 3) for k, v in result['mean_psnr'].items()})
    for name, rows in result['contrasts'].items():
        print(f'{name:<48}', ' | '.join(f"{g.split('=')[-1]} {s['mean']:+.3f}±{s['sd']:.2f}" for g, s in rows.items() if g in ('all', 'anatomy=brain', 'anatomy=knee')))
    print(json.dumps({k: result[k] for k in ('corr_per_slice', 'mask_stats')}, indent=1))


if __name__ == '__main__':
    main()
