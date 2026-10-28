"""Mask-bank pipeline figure for the abstract: ACS block -> router -> sampling mask -> reconstruction, brain and knee rows.

The bank is the two learned LOUPE masks of the per-anatomy R=8 VarNet run (brain, knee); the router's choice is the
anatomy label. Each row is a real validation slice (PSNR-percentile pick among mid-volume slices, outlier volume
excluded) reconstructed from its own seeded Bernoulli draw of the chosen mask. PSNR / SSIM follow the trainer's metric
path. The ACS tile is the log coil-RSS magnitude of the always-acquired centre block.

    CUDA_VISIBLE_DEVICES=0 python scripts/plots/mask_bank_figure.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from mask_gradient_anatomy_r8 import load, to_device

RUN = Path('checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep')
OUTLIER_VOLUME = 'AXT2_201_2010571'
ANATOMIES = ('brain', 'knee')
ACS_SIDE = 10
DRAW_SEED = 0
CANDIDATE_STRIDE = 3
PICK_QUANTILE = 0.6
DENSITY_SIGMA = 4
COLORS = {'brain': '#2a6f97', 'knee': '#c0562b'}
INK = '#222222'


@torch.no_grad()
def reconstruct(model, classes, batch, anatomy, device):
    part = to_device(batch, device)
    labels = torch.tensor([classes.index(anatomy)], device=device)
    old_mask, valid_mask = model._initial_masks(part['kspace'])
    unscaled = model.propose_mask(part['kspace'], anatomy_labels=labels)['unscaled']
    torch.manual_seed(DRAW_SEED)
    probability, mask = model.sampler.select_locations(unscaled, old_mask, model.budget, valid_mask)
    output = model(part['target'], part['kspace'], anatomy_labels=labels, fixed_mask=mask)['output']
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    masked_output, masked_target = _apply_signal_mask(output, target, part['metric_mask'])
    return {'psnr': float(compute_psnr_torch(masked_output, masked_target)[0]),
            'ssim': float(compute_ssim_torch(masked_output, masked_target)[0]),
            'output': masked_output[0, 0].cpu().numpy(), 'target': masked_target[0, 0].cpu().numpy(),
            'mask': mask[0, 0].cpu().numpy(), 'probability': torch.clamp(probability + old_mask, max=1)[0, 0].cpu().numpy()}


def acs_tile(kspace):
    height, width = kspace.shape[-3:-1]
    top, left = (height - ACS_SIDE) // 2, (width - ACS_SIDE) // 2
    block = kspace[0, :, top:top + ACS_SIDE, left:left + ACS_SIDE]
    return np.log1p(torch.linalg.norm(block, dim=(0, -1)).cpu().numpy() / kspace.abs().max().item() * 1e3)


def pick_slice(model, classes, dataset, collate, anatomy, device):
    """Index of the slice at PICK_QUANTILE of PSNR among mid-volume candidates of `anatomy` (outlier volume excluded)."""
    by_volume = {}
    offset = 0
    for part in dataset.datasets:
        in_anatomy = collate([dataset[offset]])['anatomy'][0] == anatomy
        for position, (fname, slice_id) in enumerate(part.examples):
            if in_anatomy:
                by_volume.setdefault(str(fname), []).append((slice_id, offset + position))
        offset += len(part.examples)
    candidates = []
    for fname, entries in by_volume.items():
        if OUTLIER_VOLUME in fname:
            continue
        entries = sorted(entries)
        third = len(entries) // 3
        candidates += [index for _, index in entries[third:2 * third]][::CANDIDATE_STRIDE]
    scored = sorted((reconstruct(model, classes, collate([dataset[i]]), anatomy, device)['psnr'], i) for i in candidates)
    return scored[int(PICK_QUANTILE * (len(scored) - 1))][1]


def box(ax, xy, size, label, color, fontsize=11, sublabel=None):
    ax.add_patch(FancyBboxPatch(xy, *size, boxstyle='round,pad=0.0,rounding_size=0.015', fc='white', ec=color, lw=2, zorder=1))
    cx, cy = xy[0] + size[0] / 2, xy[1] + size[1] / 2
    if sublabel:
        ax.text(cx, cy + 0.012, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color, zorder=3)
        ax.text(cx, cy - 0.025, sublabel, ha='center', va='center', fontsize=8.5, color=INK, zorder=3)
    else:
        ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color, zorder=3)


def arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=16, lw=1.8, color=color, zorder=2, shrinkA=0, shrinkB=0))


def thumbnail(fig, rect, image, cmap, border=None, dim=False, vmin=None, vmax=None):
    ax = fig.add_axes(rect)
    ax.imshow(image, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, alpha=0.35 if dim else 1.0)
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border or '#999999')
        spine.set_linewidth(3 if border else 0.8)
    return ax


def radial_profile(image):
    height, width = image.shape
    yy, xx = np.indices(image.shape)
    radius = np.hypot(yy - height // 2, xx - width // 2).astype(int)
    return np.bincount(radius.ravel(), image.ravel()) / np.bincount(radius.ravel())


def difference_figure(maps, out):
    """Brain map, knee map, knee - brain difference and radial sampling density of the two bank masks."""
    difference = gaussian_filter(maps['knee'], DENSITY_SIGMA) - gaussian_filter(maps['brain'], DENSITY_SIGMA)
    limit = float(np.abs(difference).max())
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), layout='constrained', gridspec_kw={'width_ratios': [1, 1, 1, 1.3]})
    for ax, anatomy in zip(axes[:2], ANATOMIES):
        image = ax.imshow(maps[anatomy], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{anatomy} mask (mean {maps[anatomy].mean():.3f})', color=COLORS[anatomy], fontweight='bold')
    fig.colorbar(image, ax=axes[:2], fraction=0.05, pad=0.02, label='sampling probability')
    shown = axes[2].imshow(difference, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
    axes[2].set_title(f'knee − brain (density, σ = {DENSITY_SIGMA:g} px blur)', fontweight='bold', fontsize=10)
    fig.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.02, label='Δ density')
    for ax in axes[:3]:
        ax.set_xticks([]), ax.set_yticks([])
    radii = np.arange(len(radial_profile(maps['brain'])))
    for anatomy in ANATOMIES:
        axes[3].plot(radii, radial_profile(maps[anatomy]), color=COLORS[anatomy], lw=2, label=anatomy)
    axes[3].set(xlim=(0, 128), xlabel='k-space radius (px)', ylabel='mean sampling probability', yscale='log',
                title='radial density')
    axes[3].legend(frameon=False)
    for suffix, dpi in (('.png', 250), ('.pdf', None)):
        fig.savefig(out.with_suffix(suffix), dpi=dpi, facecolor='white', bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('reports/figures/mask_bank_pipeline'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', default=None, help='override the checkpoint config data_root')
    args = parser.parse_args()

    model, checkpoint = load(RUN, 'last_model.pt', args.device)
    classes = list(checkpoint['options'].label_classes) or list(ANATOMIES)
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    if args.data_root:
        cfg.datasets.data_root = args.data_root
    _, val_loader = build_dataloaders(cfg)
    dataset, collate = val_loader.dataset, val_loader.collate_fn

    rows = {}
    for anatomy in ANATOMIES:
        index = pick_slice(model, classes, dataset, collate, anatomy, args.device)
        batch = collate([dataset[index]])
        rows[anatomy] = {**reconstruct(model, classes, batch, anatomy, args.device), 'acs': acs_tile(batch['kspace']),
                         'name': f'{batch["fname"][0]} slice {int(batch["slice_id"][0])}'}
        print(anatomy, rows[anatomy]['name'], f"PSNR {rows[anatomy]['psnr']:.2f} SSIM {rows[anatomy]['ssim']:.4f}", flush=True)

    fig = plt.figure(figsize=(15, 6.4))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1), canvas.set_ylim(0, 1), canvas.axis('off')
    row_y = {'brain': 0.70, 'knee': 0.27}
    side = 0.27
    aspect = fig.get_figwidth() / fig.get_figheight()

    def rect(cx, cy, scale=1.0):
        return [cx - side * scale / aspect / 2, cy - side * scale / 2, side * scale / aspect, side * scale]

    x_acs, x_router, x_prob, x_mask, x_vn, x_out = 0.06, 0.215, 0.385, 0.57, 0.735, 0.90
    y_vn = sum(row_y.values()) / len(row_y)
    for label, x in (('ACS block', x_acs), ('Distribution bank (R = 8)', x_prob), ('Sampled mask', x_mask),
                     ('Reconstruction', x_out)):
        canvas.text(x, 0.97, label, ha='center', va='center', fontsize=12, fontweight='bold', color=INK)

    canvas.add_patch(FancyBboxPatch((x_prob - 0.075, 0.07), 0.15, 0.85, boxstyle='round,pad=0.0,rounding_size=0.015',
                                    fc='#f4f4f4', ec='#999999', lw=1.5, ls='--', zorder=0))
    box(canvas, (x_router - 0.065, 0.40), (0.13, 0.20), 'Router', INK, fontsize=14, sublabel='selects a probability\ndistribution from the ACS')
    box(canvas, (x_vn - 0.04, y_vn - 0.04), (0.08, 0.08), 'VarNet', INK, fontsize=10)

    thumbnail(fig, rect(x_acs, 0.5, 0.8), rows['brain']['acs'], 'magma')
    canvas.text(x_acs, 0.36, f'{ACS_SIDE}×{ACS_SIDE} k-space centre\n(log |coil RSS|)', ha='center', va='top', fontsize=8.5, color=INK)
    arrow(canvas, (x_acs + 0.05, 0.5), (x_router - 0.065, 0.5))

    for anatomy, data in rows.items():
        y, color = row_y[anatomy], COLORS[anatomy]
        slot = y_vn + (0.03 if anatomy == 'brain' else -0.03)
        canvas.add_patch(FancyArrowPatch((x_router + 0.065, 0.5 + (0.06 if anatomy == 'brain' else -0.06)), (x_prob - 0.052, y),
                                         connectionstyle='arc3,rad=0', arrowstyle='-|>', mutation_scale=16, lw=2.2, color=color, zorder=2))
        canvas.text(x_prob, y + 0.165, f'{anatomy} distribution', ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
        thumbnail(fig, rect(x_prob, y, 0.9), data['probability'], 'gray', border=color, vmin=0, vmax=1)
        arrow(canvas, (x_prob + 0.055, y), (x_mask - 0.05, y), color)
        canvas.text((x_prob + x_mask) / 2 + 0.002, y + 0.025, 'sample', ha='center', va='bottom', fontsize=9, style='italic', color=color)
        thumbnail(fig, rect(x_mask, y, 0.9), data['mask'], 'gray', border=color)
        arrow(canvas, (x_mask + 0.055, y), (x_vn - 0.04, slot), color)
        arrow(canvas, (x_vn + 0.04, slot), (x_out - 0.062, y), color)
        thumbnail(fig, rect(x_out, y, 0.98), data['output'], 'gray', border=color, vmin=0, vmax=data['target'].max())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (('.png', 250), ('.pdf', None)):
        fig.savefig(args.out.with_suffix(suffix), dpi=dpi, facecolor='white')
    difference_figure({a: rows[a]['probability'] for a in ANATOMIES}, args.out.with_name('mask_bank_differences'))
    print('saved', args.out)


if __name__ == '__main__':
    main()
