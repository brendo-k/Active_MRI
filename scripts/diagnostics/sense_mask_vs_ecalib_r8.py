"""Unmasked vs masked sense net vs BART ecalib c24, and where the reconstruction error sits in brain vs knee.

Runs (matched 50-epoch bilinear CG-SENSE LOUPE R=8): `masked` (sense maps zeroed outside the body mask), `unmasked`
(`sense_mask_method=null`) and `ecalib_c24` (cached BART ecalib maps, 24x24 centre). All scored on the same 688 validation slices,
seeded Bernoulli draws and the current trainer-path metric mask:
  own            each run's own mask + own maps
  masked mask    the masked run's mask reconstructed with each of the three map sources (isolates the maps)
  c24 mask       the ecalib_c24 mask reconstructed with each of the three map sources
Per slice it also records PSNR and NMSE, the oracle (fully sampled) SENSE NMSE of each map source, the share of the signal
mask that lies outside / on the edge of the body mask, and the squared error of the first draw split by those regions
(as a fraction of target energy in the signal mask).

    python scripts/diagnostics/sense_mask_vs_ecalib_r8.py
"""
import argparse
import json
import os
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch
from anatcond_cgsense_cross_eval_r8 import unscaled
from mask_gradient_anatomy_r8 import load
from sens_upsample_slice_diff_r8 import slice_metadata

RUNS = {'masked': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep',
        'unmasked': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_nomask_50ep',
        'ecalib_c24': 'checkpoints/loupe2d_cgsense_r8_brain_knee_ecalib_c24'}
MAP_SOURCES = ('masked', 'unmasked', 'ecalib_c24')
OUT_DIR = Path('reports/sense_mask_effect_r8')
SCORING_SEEDS = (0, 1, 2, 3)
EDGE_HALF_WIDTH = 4


def nmse(output, target):
    return ((target - output) ** 2).flatten(1).sum(1) / (target ** 2).flatten(1).sum(1)


def as_complex(maps):
    return maps if maps.is_complex() else torch.view_as_complex(maps.contiguous())


@torch.no_grad()
def score(model, owner, part, probability, sense_maps, seed_key):
    """Seed-averaged [psnr, nmse] per slice, plus the first draw's squared error image (signal-masked)."""
    old_mask, valid_mask = owner._initial_masks(part['kspace'])
    total, first_error = torch.zeros(2, len(part['kspace'])), None
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = owner.sampler.select_locations(probability, old_mask, owner.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], fixed_mask=mask, sense_maps=sense_maps)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        out_m, tgt_m = _apply_signal_mask(output, target, part['metric_mask'])
        total += torch.stack([compute_psnr_torch(out_m, tgt_m).flatten(), nmse(out_m, tgt_m)]).cpu() / len(SCORING_SEEDS)
        if first_error is None:
            first_error = (out_m - tgt_m) ** 2
    return total, first_error


@torch.no_grad()
def oracle_nmse(maps, kspace_complex, target, signal):
    image = (maps.conj() * ifft_2d_img(kspace_complex, axes=[-1, -2])).sum(1, keepdim=True)
    return nmse(image.abs() * signal, target.reshape_as(signal) * signal).flatten().cpu()


def edge_band(body):
    grow = F.max_pool2d(body, 2 * EDGE_HALF_WIDTH + 1, 1, EDGE_HALF_WIDTH)
    shrink = 1 - F.max_pool2d(1 - body, 2 * EDGE_HALF_WIDTH + 1, 1, EDGE_HALF_WIDTH)
    return grow - shrink


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    models = {name: load(Path(path), 'last_model.pt', args.device)[0] for name, path in RUNS.items()}
    cfg = OmegaConf.load(Path(RUNS['ecalib_c24']) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    loader = DataLoader(val_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=val_loader.collate_fn)
    names, slice_ids, _ = slice_metadata(val_loader.dataset)

    scores, error_share, oracle, geometry = {}, {}, {}, {k: [] for k in ('signal_frac', 'dropped', 'dropped_energy', 'edge_in_signal', 'body_frac')}
    anatomy, offset = [], 0
    for batch in loader:
        kspace = batch['kspace'].to(args.device)
        part = {'kspace': kspace, 'target': batch['target'].to(args.device), 'metric_mask': metric_mask_from_real_kspace(kspace, batch['anatomy'])}
        cache = batch['sense_maps'].to(args.device)
        maps_for = {'ecalib_c24': cache}
        kc = torch.view_as_complex(kspace.contiguous())
        for name in ('masked', 'unmasked'):
            maps_for[name], body = models[name].sensetivity_net(kc, mask_method=models[name].sense_mask_method)
            if name == 'masked':
                masked_body = body
        probability = {name: unscaled(model, kspace, None) for name, model in models.items()}
        net_maps = {'masked': None, 'unmasked': None, 'ecalib_c24': cache}

        pairs = {f'own/{n}': (n, n) for n in MAP_SOURCES}
        pairs.update({f'masked_mask/{m}': (m, 'masked') for m in ('unmasked', 'ecalib_c24')})
        pairs.update({f'c24_mask/{m}': (m, 'ecalib_c24') for m in MAP_SOURCES})
        signal, body_edge = part['metric_mask'], edge_band(masked_body)
        energy = (part['target'].reshape_as(signal) * signal).pow(2).sum((1, 2, 3))
        for key, (recon, owner) in pairs.items():
            value, error = score(models[recon], models[owner], part, probability[owner], net_maps[recon], offset)
            scores.setdefault(key, []).append(value)
            if key.startswith('own/'):
                regions = {'inside_body': masked_body, 'outside_body': 1 - masked_body, 'edge_band': body_edge}
                for region, weight in regions.items():
                    error_share.setdefault(f'{key}/{region}', []).append(((error * weight).sum((1, 2, 3)) / energy).cpu())
        for name in MAP_SOURCES:
            oracle.setdefault(name, []).append(oracle_nmse(as_complex(maps_for[name]), kc, part['target'], signal))
        energy_outside = ((part['target'].reshape_as(signal) * (1 - masked_body)).pow(2).sum((1, 2, 3)) /
                          part['target'].reshape_as(signal).pow(2).sum((1, 2, 3)))
        geometry['signal_frac'].append(signal.flatten(1).mean(1).cpu())
        geometry['dropped'].append(((signal * (1 - masked_body)).sum((1, 2, 3)) / signal.sum((1, 2, 3))).cpu())
        geometry['dropped_energy'].append(energy_outside.cpu())
        geometry['edge_in_signal'].append(((signal * body_edge).sum((1, 2, 3)) / signal.sum((1, 2, 3))).cpu())
        geometry['body_frac'].append(masked_body.flatten(1).mean(1).cpu())
        anatomy += list(batch['anatomy'])
        offset += len(batch['anatomy'])
        print(f'{offset}/{len(loader.dataset)}', flush=True)

    anatomy = np.array(anatomy)
    flat = {f'psnr/{k}': torch.cat(v, dim=1)[0].numpy() for k, v in scores.items()}
    flat.update({f'nmse/{k}': torch.cat(v, dim=1)[1].numpy() for k, v in scores.items()})
    flat.update({f'errshare/{k}': torch.cat(v).numpy() for k, v in error_share.items()})
    flat.update({f'oracle_nmse/{k}': torch.cat(v).numpy() for k, v in oracle.items()})
    flat.update({f'geometry/{k}': torch.cat(v).numpy() for k, v in geometry.items()})
    groups = {'all': np.ones(len(anatomy), bool), 'brain': anatomy == 'brain', 'knee': anatomy == 'knee'}
    result = {'mean': {g: {k: float(v[rows].mean()) for k, v in flat.items()} for g, rows in groups.items()}}
    contrasts = {'own: unmasked - masked': ('own/unmasked', 'own/masked'), 'own: ecalib_c24 - masked': ('own/ecalib_c24', 'own/masked'),
                 'own: ecalib_c24 - unmasked': ('own/ecalib_c24', 'own/unmasked'),
                 'masked mask, maps unmasked - masked': ('masked_mask/unmasked', 'own/masked'),
                 'masked mask, maps ecalib - masked': ('masked_mask/ecalib_c24', 'own/masked'),
                 'masked mask, maps ecalib - unmasked': ('masked_mask/ecalib_c24', 'masked_mask/unmasked'),
                 'c24 mask, maps unmasked - masked': ('c24_mask/unmasked', 'c24_mask/masked'),
                 'c24 mask, maps ecalib - masked': ('c24_mask/ecalib_c24', 'c24_mask/masked'),
                 'c24 mask, maps ecalib - unmasked': ('c24_mask/ecalib_c24', 'c24_mask/unmasked')}
    result['contrast_psnr'] = {c: {g: {'mean': float((flat[f'psnr/{a}'] - flat[f'psnr/{b}'])[rows].mean()),
                                       'sd': float((flat[f'psnr/{a}'] - flat[f'psnr/{b}'])[rows].std()),
                                       'frac_positive': float((flat[f'psnr/{a}'] > flat[f'psnr/{b}'])[rows].mean())}
                                   for g, rows in groups.items()} for c, (a, b) in contrasts.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'vs_ecalib_results.json').write_text(json.dumps(result, indent=1) + '\n')
    np.savez(OUT_DIR / 'vs_ecalib_per_slice.npz', anatomy=anatomy, volume=names, slice_id=slice_ids, **{k.replace('/', '__'): v for k, v in flat.items()})
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
