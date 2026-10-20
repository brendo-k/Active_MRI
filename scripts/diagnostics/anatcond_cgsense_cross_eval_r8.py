"""Why does the per-anatomy LOUPE mask beat the shared one on brain but lose on knee with CG-SENSE (R=8)?

The CG-SENSE runs learn two things: the LOUPE mask(s) and the sensitivity U-Net. A shared-mask run and a
per-anatomy run (`sampler.anatomy_conditioning=true`) differ in both. This crosses them on the full fastMRI
validation set, with common seeded Bernoulli draws: every run's reconstructor (sense net + CG-SENSE) is scored with
every run's masks, so the anatomy gap splits into a mask part and a sense-net part.

Mask sources (probabilities before budget rescaling, from `propose_mask`):
  shared        the shared run's single map
  anat_own      the per-anatomy run's map for the slice's own anatomy
  anat_other    its map for the other anatomy (label swap)
Reconstructors: `shared` and `anat` (each run's own sense net; CG-SENSE itself has no weights).

Also reports, per mask: mean probability, the share of sampling probability by k-space radius, and the correlation
between maps after a Gaussian blur of sigma px (density-space, so the budget is kept). For a warm-started pair
(`sharedinit`) the starting mask is included as `start`.

    python scripts/diagnostics/anatcond_cgsense_cross_eval_r8.py --pair 100ep
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
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import load

PAIRS = {
    '50ep': ('checkpoints/loupe2d_cgsense_r8_brain_knee_shared', 'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond'),
    '100ep': ('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep',
              'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep'),
    'sharedinit': ('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_cont',
                   'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond_sharedinit'),
}
# Warm-started pairs: the mask both arms started from, added to the mask stats as 'start' to read each map's drift.
START = {'sharedinit': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep'}
OUT_DIR = Path('analysis/anatcond_cgsense')
SCORING_SEEDS = (0, 1)
RADIUS_EDGES = (0, 8, 16, 32, 64, 182)
BLUR_SIGMAS = (0, 2, 8)
BOOTSTRAPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pair', choices=tuple(PAIRS), default='100ep')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--split', choices=('val', 'train'), default='val',
                        help='train: the slices the masks were fitted to (in-sample), to separate overfitting from optimisation')
    parser.add_argument('--stride', type=int, default=1, help='score every stride-th slice')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def labels_for(batch, classes, device):
    index = {name: i for i, name in enumerate(classes)}
    return torch.tensor([index[a] for a in batch['anatomy']], device=device)


@torch.no_grad()
def unscaled(model, kspace, labels):
    return model.propose_mask(kspace, anatomy_labels=labels)['unscaled']


@torch.no_grad()
def psnr_with(model, part, probability, seed_key):
    """Per-slice trainer-path PSNR and the training loss (L1 between output and target), over SCORING_SEEDS draws."""
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    psnr, l1 = torch.zeros(len(part['kspace'])), torch.zeros(len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + seed_key)
        mask = model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]
        output = model(part['target'], part['kspace'], fixed_mask=mask)['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        psnr += compute_psnr_torch(*_apply_signal_mask(output, target)).flatten().cpu() / len(SCORING_SEEDS)
        l1 += (output - target).abs().flatten(1).mean(1).cpu() / len(SCORING_SEEDS)
    return psnr, l1


def radius_map(shape):
    ky = torch.arange(shape[0]) - shape[0] // 2
    kx = torch.arange(shape[1]) - shape[1] // 2
    return torch.hypot(ky[:, None].float(), kx[None, :].float())


def blurred(p, sigma):
    if sigma == 0:
        return p
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    out = F.conv2d(p[None, None], kernel.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(out, kernel.view(1, 1, -1, 1), padding=(radius, 0))[0, 0]


def mask_stats(maps):
    """maps: {name: [H, W] centred probability}."""
    radius = radius_map(next(iter(maps.values())).shape)
    stats = {'mean_probability': {k: float(v.mean()) for k, v in maps.items()}, 'radial_share': {}, 'blur_corr': {}}
    for name, p in maps.items():
        total = p.sum()
        stats['radial_share'][name] = [float(p[(radius >= lo) & (radius < hi)].sum() / total)
                                       for lo, hi in zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:])]
    names = list(maps)
    for sigma in BLUR_SIGMAS:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = blurred(maps[a], sigma).flatten(), blurred(maps[b], sigma).flatten()
                stats['blur_corr'][f'{a}|{b}|sigma{sigma}'] = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    return stats


def bootstrap(delta, groups, rng):
    unique = np.unique(groups)
    per_group = np.array([delta[groups == g].mean() for g in unique])
    draws = [per_group[rng.randint(len(per_group), size=len(per_group))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(delta.mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])], 'slices': int(len(delta))}


def main():
    args = parse_args()
    shared_dir, anat_dir = (Path(p) for p in PAIRS[args.pair])
    shared, _ = load(shared_dir, args.checkpoint, args.device)
    anat, anat_ckpt = load(anat_dir, args.checkpoint, args.device)
    classes = list(anat_ckpt['options'].label_classes)
    cfg = OmegaConf.load(anat_dir / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    source_loader = val_loader if args.split == 'val' else train_loader
    indices = list(range(0, len(source_loader.dataset), args.stride))
    loader = DataLoader(Subset(source_loader.dataset, indices), batch_size=args.batch, shuffle=False, num_workers=4,
                        collate_fn=source_loader.collate_fn)

    recons = {'shared': shared, 'anat': anat}
    arms = [(r, m) for r in recons for m in ('shared', 'anat_own', 'anat_other')]
    psnr = {f'{r}_recon+{m}_mask': [] for r, m in arms}
    l1 = {f'{r}_recon+{m}_mask': [] for r, m in arms}
    anatomy, volume, condition = [], [], []
    position = 0
    for batch in loader:
        part = {'kspace': batch['kspace'].to(args.device), 'target': batch['target'].to(args.device)}
        labels = labels_for(batch, classes, args.device)
        sources = {'shared': unscaled(shared, part['kspace'], None),
                   'anat_own': unscaled(anat, part['kspace'], labels),
                   'anat_other': unscaled(anat, part['kspace'], 1 - labels)}
        seed_key = indices[position]
        for r, m in arms:
            slice_psnr, slice_l1 = psnr_with(recons[r], part, sources[m], seed_key)
            psnr[f'{r}_recon+{m}_mask'].append(slice_psnr)
            l1[f'{r}_recon+{m}_mask'].append(slice_l1)
        anatomy += list(batch['anatomy'])
        volume += list(batch['fname'])
        condition += list(batch.get('condition', batch['anatomy']))
        position += len(batch['anatomy'])
        print(f'{position}/{len(indices)}', flush=True)

    psnr = {k: torch.cat(v).numpy() for k, v in psnr.items()}
    l1 = {k: torch.cat(v).numpy() for k, v in l1.items()}
    anatomy, volume = np.array(anatomy), np.array(volume)
    rng = np.random.RandomState(0)
    contrasts = {
        'anat run - shared run (both own masks and sense nets)': ('anat_recon+anat_own_mask', 'shared_recon+shared_mask'),
        'mask effect on shared sense net (anat_own - shared mask)': ('shared_recon+anat_own_mask', 'shared_recon+shared_mask'),
        'mask effect on anat sense net (anat_own - shared mask)': ('anat_recon+anat_own_mask', 'anat_recon+shared_mask'),
        'sense-net effect with shared mask (anat - shared recon)': ('anat_recon+shared_mask', 'shared_recon+shared_mask'),
        'sense-net effect with anat_own mask (anat - shared recon)': ('anat_recon+anat_own_mask', 'shared_recon+anat_own_mask'),
        'label swap on anat sense net (anat_other - anat_own mask)': ('anat_recon+anat_other_mask', 'anat_recon+anat_own_mask'),
        'label swap on shared sense net (anat_other - anat_own mask)': ('shared_recon+anat_other_mask', 'shared_recon+anat_own_mask'),
    }
    result = {'pair': args.pair, 'split': args.split, 'stride': args.stride, 'checkpoints': [str(shared_dir), str(anat_dir)], 'classes': classes,
              'mean_psnr': {}, 'contrasts': {}}
    for name in classes:
        keep = anatomy == name
        result['mean_psnr'][name] = {k: float(v[keep].mean()) for k, v in psnr.items()}
        result['contrasts'][name] = {c: bootstrap(psnr[a][keep] - psnr[b][keep], volume[keep], rng) for c, (a, b) in contrasts.items()}
    for name in sorted(set(condition) - set(classes)):
        keep = np.array(condition) == name
        result['contrasts'][name] = {c: bootstrap(psnr[a][keep] - psnr[b][keep], volume[keep], rng) for c, (a, b) in contrasts.items()}
    result['mean_l1'] = {name: {k: float(v[np.array(condition if name not in classes else anatomy) == name].mean()) for k, v in l1.items()}
                         for name in sorted(set(condition) | set(classes))}

    probe = next(iter(DataLoader(Subset(val_loader.dataset, [0]), batch_size=1, collate_fn=val_loader.collate_fn)))
    kspace = probe['kspace'].to(args.device)
    maps = {'shared': unscaled(shared, kspace, None)[0, 0]}
    for i, name in enumerate(classes):
        maps[f'anat_{name}'] = unscaled(anat, kspace, torch.tensor([i], device=args.device))[0, 0]
    if args.pair in START:
        start, _ = load(Path(START[args.pair]), 'last_model.pt', args.device)
        maps['start'] = unscaled(start, kspace, None)[0, 0]
    maps = {k: v.float().cpu() for k, v in maps.items()}  # masks are already centred (see _init_mask)
    result['mask_stats'] = mask_stats(maps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.pair if args.split == 'val' else f'{args.pair}_train'
    out = OUT_DIR / f'cross_eval_{tag}.json'
    out.write_text(json.dumps(result, indent=1) + '\n')
    np.savez(OUT_DIR / f'cross_eval_{tag}_psnr.npz', anatomy=anatomy, volume=volume, condition=np.array(condition), **psnr,
             **{f'l1__{k}': v for k, v in l1.items()})
    for name, rows in result['contrasts'].items():
        print(f'== {name}')
        for c, r in rows.items():
            print(f"  {c}: {r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] (n={r['slices']})")
    print(json.dumps(result['mask_stats'], indent=1))
    for name, values in result['mean_l1'].items():
        print(f'L1 {name}: ' + ', '.join(f'{k} {v:.5f}' for k, v in values.items() if k.startswith('anat_recon')))


if __name__ == '__main__':
    main()
