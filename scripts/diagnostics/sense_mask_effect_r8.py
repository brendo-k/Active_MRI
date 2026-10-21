"""Masked vs unmasked sense maps (matched 50-epoch bilinear CG-SENSE LOUPE R=8 runs), scored on one common metric.

The two runs differ only in `sampler.sense_mask_method` (null = sense maps are not zeroed outside the body mask). Their logged
validation numbers are not comparable because the masked run was logged before the metric-mask change. Here every
(sense net, mask) pair is scored on the same seeded Bernoulli draws with the current trainer-path metric mask:
  masked / unmasked   each run's own mask + sense net
  masked_sense+unmasked_mask, unmasked_sense+masked_mask   cross pairs
Also reports the unmasked sense net's energy outside the body mask.

    python scripts/diagnostics/sense_mask_effect_r8.py
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
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from anatcond_cgsense_cross_eval_r8 import mask_stats, unscaled
from mask_gradient_anatomy_r8 import load
from sens_upsample_slice_diff_r8 import psnr_per_seed, slice_metadata, summarise

RUNS = {'masked': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep'),
        'unmasked': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_nomask_50ep')}
OUT_DIR = Path('reports/sense_mask_effect_r8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    models = {name: load(path, 'last_model.pt', args.device)[0] for name, path in RUNS.items()}
    cfg = OmegaConf.load(RUNS['masked'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)
    names, slice_ids, position = slice_metadata(val_loader.dataset)

    combos = {'masked': ('masked', 'masked'), 'unmasked': ('unmasked', 'unmasked'),
              'masked_sense+unmasked_mask': ('masked', 'unmasked'), 'unmasked_sense+masked_mask': ('unmasked', 'masked')}
    psnr = {k: [] for k in combos}
    anatomy, support, offset = [], [], 0
    for batch in loader:
        part = {'kspace': batch['kspace'].to(args.device), 'target': batch['target'].to(args.device)}
        part['metric_mask'] = metric_mask_from_real_kspace(part['kspace'], batch['anatomy'])
        maps = {name: unscaled(model, part['kspace'], None) for name, model in models.items()}
        for key, (recon, mask) in combos.items():
            psnr[key].append(psnr_per_seed(models[recon], part, maps[mask], offset))
        anatomy += list(batch['anatomy'])
        support += part['metric_mask'].flatten(1).mean(1).tolist()
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    psnr = {k: torch.cat(v, dim=1).numpy() for k, v in psnr.items()}
    mean_psnr = {k: v.mean(0) for k, v in psnr.items()}
    anatomy, support = np.array(anatomy), np.array(support)
    noise = psnr['masked'][0] - psnr['masked'][1]
    contrasts = {'full (unmasked - masked)': mean_psnr['unmasked'] - mean_psnr['masked'],
                 'mask only (unmasked mask, masked sense)': mean_psnr['masked_sense+unmasked_mask'] - mean_psnr['masked'],
                 'sense only (unmasked sense, masked mask)': mean_psnr['unmasked_sense+masked_mask'] - mean_psnr['masked']}
    groups = {'overall': np.ones(len(support), bool)}
    groups.update({f'anatomy={a}': anatomy == a for a in np.unique(anatomy)})
    edges = np.percentile(support, [33.3, 66.7])
    groups.update({f'support={n}': np.digitize(support, edges) == i for i, n in enumerate(('low', 'mid', 'high'))})
    result = {'absolute_psnr': {g: {k: float(v[rows].mean()) for k, v in mean_psnr.items()} for g, rows in groups.items()},
              'strata': {g: {c: summarise(d[rows], noise[rows]) for c, d in contrasts.items()} for g, rows in groups.items()},
              'noise_floor_sd_same_model_two_draws': float(noise.std())}

    probe = next(iter(DataLoader(val_loader.dataset, batch_size=1, collate_fn=val_loader.collate_fn)))
    kspace = probe['kspace'].to(args.device)
    probabilities = {n: unscaled(m, kspace, None)[0, 0].float().cpu() for n, m in models.items()}
    result['mask_stats'] = mask_stats(probabilities)
    result['mask_stats']['abs_diff_mean'] = float((probabilities['masked'] - probabilities['unmasked']).abs().mean())
    result['mask_stats']['pixel_corr'] = float(np.corrcoef(probabilities['masked'].flatten(), probabilities['unmasked'].flatten())[0, 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    np.savez(OUT_DIR / 'psnr.npz', anatomy=anatomy, volume=names, slice_id=slice_ids, position=position, support=support, **mean_psnr)
    torch.save(probabilities, OUT_DIR / 'final_masks.pt')
    for g, absolute in result['absolute_psnr'].items():
        print(f'{g:<16}', ' '.join(f'{k}={v:.2f}' for k, v in absolute.items()))
    for g, rows in result['strata'].items():
        print(f'{g:<16}', ' | '.join(f"{c.split(' ')[0]} {s['mean']:+.3f}±{s['sd']:.3f}" for c, s in rows.items()), f"n={rows['full (unmasked - masked)']['n']}")
    print(json.dumps({k: result[k] for k in ('noise_floor_sd_same_model_two_draws', 'mask_stats')}, indent=1))


if __name__ == '__main__':
    main()
