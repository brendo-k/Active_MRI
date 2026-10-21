"""Masks of the CG-SENSE R=8 mask variants: shared / per-anatomy / per-protocol, Bernoulli (LOUPE) vs DCM selection.

For each run and class: the sampling probability (LOUPE, after budget rescaling, centre included) or the deterministic
mask (DCM), the share of samples at k-space radius >= 64 px, the pair correlation g(d) against the map's own smooth
density (as in `mask_spacing_r8.py`; < 1 at small d means neighbours are avoided), and the overlap between the classes
of one run (pixel correlation for LOUPE maps, Dice for DCM masks).

    python scripts/diagnostics/mask_variants_figure_r8.py
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from mask_gradient_anatomy_r8 import load
from mask_spacing_r8 import OFFSETS, pair_correlation

RUNS = {
    'LOUPE shared': 'checkpoints/loupe2d_cgsense_r8_brain_knee_shared',
    'LOUPE anatomy': 'checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond',
    'LOUPE protocol': 'checkpoints/loupe2d_cgsense_r8_brain_knee_prototable',
    'DCM shared': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_shared',
    'DCM anatomy': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_anatcond',
    'DCM protocol': 'checkpoints/loupe2d_cgsense_r8_brain_knee_dcm_prototable',
}
OUT_JSON = Path('analysis/cgsense_mask_variants/masks.json')
OUT_FIG = Path('reports/figures/cgsense_mask_variants_r8.png')
OUTER_RADIUS = 64


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smooth', type=float, default=4.0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


@torch.no_grad()
def class_maps(model, options, kspace):
    """{class name: [H, W] map}: rescaled probability + centre for Bernoulli runs, the binary mask for DCM runs."""
    classes = list(options.label_classes) if options.anatomy_conditioning else ['all']
    old_mask, valid_mask = model._initial_masks(kspace)
    maps = {}
    for index, name in enumerate(classes):
        labels = torch.tensor([index], device=kspace.device) if options.anatomy_conditioning else None
        unscaled = model.propose_mask(kspace, anatomy_labels=labels)['unscaled']
        probability, mask = model.sampler.select_locations(unscaled, old_mask, model.budget, valid_mask)
        deterministic = getattr(options, 'selection', 'bernoulli') == 'dcm'
        maps[name] = (mask if deterministic else torch.clamp(probability + old_mask, max=1))[0, 0].float().cpu()
    return maps


def outer_share(p, radius):
    return float(p[radius >= OUTER_RADIUS].sum() / p.sum())


def overlap(a, b, deterministic):
    if deterministic:
        return float(2 * (a * b).sum() / (a.sum() + b.sum()))
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def main():
    args = parse_args()
    cfg = OmegaConf.load(Path(RUNS['LOUPE shared']) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    kspace = next(iter(val_loader))['kspace'][:1].to(args.device)

    maps, result = {}, {'smooth_sigma_px': args.smooth, 'offsets_px': list(OFFSETS), 'outer_radius_px': OUTER_RADIUS, 'runs': {}}
    radius = None
    for run, path in RUNS.items():
        model, checkpoint = load(Path(path), 'last_model.pt', args.device)
        maps[run] = class_maps(model, checkpoint['options'], kspace)
        del model
        torch.cuda.empty_cache()
        if radius is None:
            h, w = next(iter(maps[run].values())).shape
            yy, xx = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing='ij')
            radius = torch.hypot(yy.float(), xx.float())
        deterministic = run.startswith('DCM')
        names = list(maps[run])
        result['runs'][run] = {
            'classes': {name: {'samples': float(p.sum()), 'outer_share': outer_share(p, radius),
                               'g': pair_correlation(p, args.smooth)} for name, p in maps[run].items()},
            'overlap_metric': 'dice' if deterministic else 'pixel_correlation',
            'overlap': {f'{a} vs {b}': overlap(maps[run][a], maps[run][b], deterministic)
                        for i, a in enumerate(names) for b in names[i + 1:]},
        }
    for kind in ('LOUPE', 'DCM'):
        shared = maps[f'{kind} shared']['all']
        result['runs'][f'{kind} shared']['overlap_with'] = {
            f'{run} {name}': overlap(shared, p, kind == 'DCM')
            for run in (f'{kind} anatomy', f'{kind} protocol') for name, p in maps[run].items()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1) + '\n')

    columns = [('shared', 'all'), ('anatomy', 'brain'), ('anatomy', 'knee'),
               ('protocol', 'brain'), ('protocol', 'knee-PD'), ('protocol', 'knee-PDFS')]
    zoom = slice(96, 160)
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(4, len(columns), figsize=(15, 10.5), facecolor='#fcfcfb')
    for col, (variant, name) in enumerate(columns):
        for row, (kind, view) in enumerate((('LOUPE', None), ('LOUPE', zoom), ('DCM', None), ('DCM', zoom))):
            run = f'{kind} {variant}'
            p = maps[run][name]
            stats = result['runs'][run]['classes'][name]
            ax = axes[row, col]
            shown = p if view is None else (p if kind == 'DCM' else (torch.rand(p.shape, generator=torch.Generator().manual_seed(1)) < p).float())
            ax.imshow((shown if view is None else shown[view, view]).numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]), ax.set_yticks([])
            if view is None:
                ax.set_title(f'{run} / {name}\nr>={OUTER_RADIUS}: {100 * stats["outer_share"]:.0f}%  g(1)={stats["g"]["mean"][0]:.2f}',
                             fontsize=8.5, color='#0b0b0b')
            else:
                ax.set_title('central 64 x 64' + (', one draw' if kind == 'LOUPE' else ''), fontsize=8, color='#52514e')
    fig.suptitle('CG-SENSE R=8 masks: LOUPE probability (rows 1-2) vs DCM deterministic mask (rows 3-4)', fontsize=11, color='#0b0b0b')
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, facecolor=fig.get_facecolor())
    print(json.dumps({run: {name: [round(c['outer_share'], 3), round(c['g']['mean'][0], 2), round(c['g']['mean'][1], 2)]
                            for name, c in r['classes'].items()} for run, r in result['runs'].items()}, indent=1))
    print(json.dumps({run: r['overlap'] for run, r in result['runs'].items() if r['overlap']}, indent=1))
    print(json.dumps({run: r['overlap_with'] for run, r in result['runs'].items() if 'overlap_with' in r}, indent=1))


if __name__ == '__main__':
    main()
