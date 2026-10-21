"""Do the learned R=8 masks keep sampled k-space points apart? Coil sensitivities multiply the image, so in k-space
each coil's data is the object's k-space convolved with a small kernel. Neighbouring locations within that kernel are
partly redundant, which would favour masks whose samples are spaced (Poisson-disc-like) rather than independent.

LOUPE draws every location independently (Bernoulli), so any spacing has to live in the probability map itself. For
each map p (after budget rescaling, centre included) this computes the pair correlation against the map's own smooth
density p_s (Gaussian blur, sigma `--smooth` px):

    g(d) = mean_k p(k) p(k+d) / mean_k p_s(k) p_s(k+d)

g < 1 at small offsets d means neighbours are avoided (spacing), g > 1 means clustering, g = 1 means no pixel-scale
structure. For binary masks (Poisson-disc reference, one Bernoulli draw) the same formula applies with p binary.

Maps: the VarNet LOUPE run, the three CG-SENSE R=8 100-epoch maps (shared, per-anatomy brain, per-anatomy knee),
the per-slice oracle labels (`/tmp/bkadota_oracle_labels_r8`), and a variable-density Poisson-disc reference drawn
from the VarNet LOUPE density. Also measures the k-space extent of the sensitivity maps (the convolution kernel) on
validation slices: the radius holding 90% / 95% of each coil map's k-space energy.

    python scripts/diagnostics/mask_spacing_r8.py
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
import torch.nn.functional as F
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.training.builders import build_dataloaders
from mask_gradient_anatomy_r8 import load

VARNET_LOUPE = Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05')
CG_SHARED = Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep')
CG_ANAT = Path('checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep')
ORACLE_LABELS = Path('/tmp/bkadota_oracle_labels_r8/val_0_640_s5.pt')
OUT_JSON = Path('reports/mask_spacing_r8/mask_spacing_r8.json')
OUT_FIG = Path('reports/figures/mask_spacing_r8.png')
OFFSETS = tuple(range(1, 9))
SERIES = {  # categorical slots 1-4 of the reference palette, then neutral greys for the references
    'VarNet LOUPE': '#2a78d6', 'CG-SENSE shared': '#eb6834', 'CG-SENSE knee': '#1baf7a', 'oracle labels': '#eda100',
    'Poisson-disc': '#52514e', 'Bernoulli draw': '#9a9893',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smooth', type=float, default=4.0)
    parser.add_argument('--oracle-slices', type=int, default=64)
    parser.add_argument('--sens-slices', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def gaussian_blur(x, sigma):
    radius = int(3 * sigma)
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-t ** 2 / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).to(x)
    out = F.conv2d(x[None, None], kernel.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(out, kernel.view(1, 1, -1, 1), padding=(radius, 0))[0, 0]


def pair_correlation(p, sigma):
    """{'kx'|'ky'|'diag': [g(d) for d in OFFSETS]} for one [H, W] map; zero-padded shifts."""
    p = p.float().cpu()
    smooth = gaussian_blur(p, sigma)
    out = {}
    for name, (dy, dx) in {'kx': (0, 1), 'ky': (1, 0), 'diag': (1, 1)}.items():
        values = []
        for d in OFFSETS:
            a, b = p[:p.shape[0] - d * dy, :p.shape[1] - d * dx], p[d * dy:, d * dx:]
            sa, sb = smooth[:p.shape[0] - d * dy, :p.shape[1] - d * dx], smooth[d * dy:, d * dx:]
            values.append(float((a * b).mean() / (sa * sb).mean()))
        out[name] = values
    out['mean'] = [float(np.mean([out[k][i] for k in ('kx', 'ky', 'diag')])) for i in range(len(OFFSETS))]
    return out


@torch.no_grad()
def rescaled(model, unscaled, kspace):
    old_mask, valid_mask = model._initial_masks(kspace)
    probability = model.sampler.select_locations(unscaled, old_mask, model.budget, valid_mask)[0]
    return torch.clamp(probability + old_mask, max=1)[0, 0].cpu()


def poisson_disc(density, rng):
    """Variable-density Poisson-disc mask with about density.sum() samples: points visited in random order weighted
    by density, accepted if no accepted point lies within r = c / sqrt(density); c bisected to hit the budget."""
    target = density.sum()
    h, w = density.shape
    order = np.argsort(-rng.random(h * w) ** (1.0 / np.clip(density.ravel(), 1e-6, None)))
    ys, xs = np.divmod(order, w)

    def draw(c):
        mask = np.zeros((h, w), dtype=bool)
        radius = c / np.sqrt(np.clip(density, 1e-6, None))
        for y, x in zip(ys, xs):
            r = radius[y, x]
            ri = int(np.ceil(r))
            y0, y1, x0, x1 = max(0, y - ri), min(h, y + ri + 1), max(0, x - ri), min(w, x + ri + 1)
            window = mask[y0:y1, x0:x1]
            if window.any():
                yy, xx = np.nonzero(window)
                if np.min((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2) < r * r:
                    continue
            mask[y, x] = True
            if mask.sum() >= target:
                break
        return mask

    low, high = 0.3, 1.5
    for _ in range(12):
        c = 0.5 * (low + high)
        count = draw(c).sum()
        low, high = (c, high) if count >= target else (low, c)
    return draw(low)


@torch.no_grad()
def sens_kernel_radius(model, loader, count, device):
    """Radius (px) holding 90% and 95% of each coil sensitivity map's k-space energy, median over coils and slices."""
    radius_map = None
    r90, r95 = [], []
    seen = 0
    for batch in loader:
        kspace = torch.view_as_complex(batch['kspace'].to(device).contiguous())
        sense_maps, _ = model.sensetivity_net(kspace, mask_method=model.sense_mask_method)
        spectrum = torch.fft.fftshift(torch.fft.fft2(sense_maps, norm='ortho'), dim=(-2, -1)).abs().pow(2)
        if radius_map is None:
            h, w = spectrum.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing='ij')
            radius_map = torch.hypot(yy.float(), xx.float()).to(device)
            edges = torch.arange(0, 64, 0.5, device=device)
        for coil_spectrum in spectrum.flatten(0, 1):
            total = coil_spectrum.sum()
            if total <= 0:
                continue
            cumulative = torch.stack([coil_spectrum[radius_map <= e].sum() for e in edges]) / total
            r90.append(float(edges[(cumulative >= 0.90).nonzero()[0]]))
            r95.append(float(edges[(cumulative >= 0.95).nonzero()[0]]))
        seen += len(kspace)
        if seen >= count:
            break
    return {'r90_px_median': float(np.median(r90)), 'r95_px_median': float(np.median(r95)), 'maps': len(r90)}


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    varnet, _ = load(VARNET_LOUPE, 'last_model.pt', args.device)
    shared, _ = load(CG_SHARED, 'last_model.pt', args.device)
    anat, _ = load(CG_ANAT, 'last_model.pt', args.device)
    cfg = OmegaConf.load(VARNET_LOUPE / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 2
    _, val_loader = build_dataloaders(cfg)
    probe = next(iter(val_loader))
    kspace = probe['kspace'][:1].to(args.device)

    maps = {
        'VarNet LOUPE': rescaled(varnet, varnet.propose_mask(kspace)['unscaled'], kspace),
        'CG-SENSE shared': rescaled(shared, shared.propose_mask(kspace)['unscaled'], kspace),
        'CG-SENSE brain': rescaled(anat, anat.propose_mask(kspace, anatomy_labels=torch.tensor([0], device=args.device))['unscaled'], kspace),
        'CG-SENSE knee': rescaled(anat, anat.propose_mask(kspace, anatomy_labels=torch.tensor([1], device=args.device))['unscaled'], kspace),
    }
    labels = torch.load(ORACLE_LABELS, map_location='cpu', weights_only=False)
    oracle = []
    for i in range(min(args.oracle_slices, len(labels['label_logits']))):
        logits = labels['label_logits'][i:i + 1].float().to(args.device)
        old_mask, valid_mask = labels['old_mask'][i:i + 1].float().to(args.device), labels['valid_mask'][i:i + 1].float().to(args.device)
        p = varnet.sampler.select_locations(scores_to_prob(logits), old_mask, varnet.budget, valid_mask)[0]
        oracle.append(torch.clamp(p + old_mask, max=1)[0, 0].cpu())
    maps['oracle label (slice 0)'] = oracle[0]
    poisson = torch.from_numpy(poisson_disc(maps['VarNet LOUPE'].numpy().astype(np.float64), rng)).float()
    bernoulli = (torch.rand(maps['VarNet LOUPE'].shape, generator=torch.Generator().manual_seed(0)) < maps['VarNet LOUPE']).float()

    result = {'smooth_sigma_px': args.smooth, 'offsets_px': list(OFFSETS), 'pair_correlation': {}}
    for name, p in maps.items():
        result['pair_correlation'][name] = pair_correlation(p, args.smooth)
    per_slice = [pair_correlation(p, args.smooth)['mean'] for p in oracle]
    result['pair_correlation']['oracle labels'] = {'mean': np.mean(per_slice, axis=0).tolist(),
                                                   'p10': np.percentile(per_slice, 10, axis=0).tolist(),
                                                   'p90': np.percentile(per_slice, 90, axis=0).tolist(),
                                                   'slices': len(per_slice)}
    result['pair_correlation']['Poisson-disc'] = pair_correlation(poisson, args.smooth)
    result['pair_correlation']['Bernoulli draw'] = pair_correlation(bernoulli, args.smooth)
    result['samples'] = {'Poisson-disc': float(poisson.sum()), 'VarNet LOUPE expected': float(maps['VarNet LOUPE'].sum())}
    result['sens_kernel'] = sens_kernel_radius(varnet, val_loader, args.sens_slices, args.device)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1) + '\n')

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    shown = ['VarNet LOUPE', 'CG-SENSE shared', 'CG-SENSE brain', 'CG-SENSE knee', 'oracle label (slice 0)']
    fig = plt.figure(figsize=(15, 9.6), facecolor='#fcfcfb')
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.1], hspace=0.35, wspace=0.3)
    zoom = slice(96, 160)
    for col, name in enumerate(shown):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(maps[name].numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{name}\nsampling probability', fontsize=9, color='#0b0b0b')
        ax.set_xticks([]), ax.set_yticks([])
        ax = fig.add_subplot(grid[1, col])
        draw = (torch.rand(maps[name].shape, generator=torch.Generator().manual_seed(1)) < maps[name]).float()
        ax.imshow(draw[zoom, zoom].numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title('one draw, central 64 x 64', fontsize=9, color='#52514e')
        ax.set_xticks([]), ax.set_yticks([])
    ax = fig.add_subplot(grid[0, 5])
    ax.imshow(poisson.numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Poisson-disc reference\n(VarNet LOUPE density)', fontsize=9, color='#0b0b0b')
    ax.set_xticks([]), ax.set_yticks([])
    ax = fig.add_subplot(grid[1, 5])
    ax.imshow(poisson[zoom, zoom].numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('central 64 x 64', fontsize=9, color='#52514e')
    ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(grid[2, :3])
    for name in ('VarNet LOUPE', 'CG-SENSE shared', 'CG-SENSE knee', 'oracle labels', 'Poisson-disc', 'Bernoulli draw'):
        values = result['pair_correlation'][name]['mean']
        style = '--' if name in ('Poisson-disc', 'Bernoulli draw') else '-'
        ax.plot(OFFSETS, values, style, color=SERIES[name], linewidth=2, marker='o', markersize=4, label=name)
    band = result['pair_correlation']['oracle labels']
    ax.fill_between(OFFSETS, band['p10'], band['p90'], color=SERIES['oracle labels'], alpha=0.15, linewidth=0)
    ax.axhline(1.0, color='#c3c2b7', linewidth=1)
    kernel = result['sens_kernel']['r90_px_median']
    ax.axvline(kernel, color='#c3c2b7', linewidth=1, linestyle=':')
    ax.text(kernel, ax.get_ylim()[1], f' sense-map kernel r90 = {kernel:.1f} px', fontsize=8, color='#52514e', va='top')
    ax.set_xlabel('offset d (px)')
    ax.set_ylabel('pair correlation g(d), mean of kx / ky / diagonal\n(<1: neighbours avoided)')
    ax.set_xlim(0.8, OFFSETS[-1] + 0.2)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)

    ax = fig.add_subplot(grid[2, 3:])
    h, w = maps['VarNet LOUPE'].shape
    yy, xx = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing='ij')
    radius = np.hypot(yy, xx)
    edges = np.arange(0, 130, 4)
    centres = 0.5 * (edges[1:] + edges[:-1])
    for name, colour in (('VarNet LOUPE', SERIES['VarNet LOUPE']), ('CG-SENSE shared', SERIES['CG-SENSE shared']),
                         ('CG-SENSE knee', SERIES['CG-SENSE knee'])):
        p = maps[name].numpy()
        profile = [p[(radius >= lo) & (radius < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centres, profile, color=colour, linewidth=2, label=name)
    p = maps['CG-SENSE brain'].numpy()
    ax.plot(centres, [p[(radius >= lo) & (radius < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])],
            color='#e87ba4', linewidth=2, label='CG-SENSE brain')
    ax.set_xlabel('k-space radius (px)')
    ax.set_ylabel('mean sampling probability')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('R=8 learned masks: pixel structure, spacing and radial density', fontsize=12, color='#0b0b0b')
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(json.dumps({k: v for k, v in result.items() if k != 'pair_correlation'}, indent=1))
    for name, values in result['pair_correlation'].items():
        print(f"{name:24s} g(d), d=1..8: {np.round(values['mean'], 3).tolist()}")


if __name__ == '__main__':
    main()
