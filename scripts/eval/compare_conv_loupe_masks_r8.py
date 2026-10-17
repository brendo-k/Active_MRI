"""Learned masks of the 1-step conv_2d sampler vs LOUPE-2D (R=8, brain+knee VarNet, both floor 0.05).

conv_2d is input-adaptive, so its mask is evaluated per validation slice; LOUPE's is one pattern for
every slice. Both go through `propose_mask`, the single-step path the trainer uses: the effective
sampling probability is the budget-rescaled probability plus the pre-selected centre, DC-centred.

Smoothness is the mean |difference| between 4-neighbours of the effective probability map, divided
by the map's mean, so it is comparable across maps with the same budget. Lower is smoother.

    python scripts/eval/compare_conv_loupe_masks_r8.py
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
from activemri.training.trainer import build_model

RUNS = {
    'loupe': Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05'),
    'conv': Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05'),
}
OUT_DIR = Path('reports/conv_vs_loupe_r8')
ANATOMIES = ('brain', 'knee')
RADIAL_BINS = 64
EXAMPLES_PER_ANATOMY = 3


def load_model(run_dir, checkpoint_name, device):
    checkpoint = torch.load(run_dir / checkpoint_name, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), int(checkpoint['epoch'])


def effective_probability(model, kspace, anatomy_labels):
    with torch.no_grad():
        proposal = model.propose_mask(kspace, anatomy_labels=anatomy_labels)
    return torch.clamp(proposal['rescaled'] + proposal['old_mask'], max=1)[:, 0].cpu().numpy()


def collect(models, val_loader, device):
    """{run: {anatomy: [N, H, W] effective probability}} over the whole validation set."""
    maps = {name: {a: [] for a in ANATOMIES} for name in models}
    for batch in val_loader:
        kspace = batch['kspace'].to(device)
        labels = batch.get('anatomy_labels')
        labels = labels.to(device) if torch.is_tensor(labels) else None
        for name, model in models.items():
            for anatomy, prob in zip(batch['anatomy'], effective_probability(model, kspace, labels)):
                maps[name][anatomy].append(prob)
    return {name: {a: np.stack(v) for a, v in per.items()} for name, per in maps.items()}


def roughness(prob):
    neighbour_diff = np.abs(np.diff(prob, axis=-1)).mean(axis=(-2, -1)) + np.abs(np.diff(prob, axis=-2)).mean(axis=(-2, -1))
    return neighbour_diff / 2 / prob.mean(axis=(-2, -1))


def binary_entropy_bits(prob):
    p = np.clip(prob.astype(np.float64), 1e-12, 1 - 1e-12)
    return (-p * np.log2(p) - (1 - p) * np.log2(1 - p)).mean(axis=(-2, -1))


def radial_profile(image):
    height, width = image.shape
    y, x = np.indices(image.shape)
    radius = np.hypot(y - height // 2, x - width // 2) / (height // 2)
    edges = np.linspace(0, np.sqrt(2), RADIAL_BINS + 1)
    which = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, RADIAL_BINS - 1)
    sums = np.bincount(which, image.ravel(), RADIAL_BINS)
    counts = np.bincount(which, minlength=RADIAL_BINS)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres[counts > 0], (sums / np.maximum(counts, 1))[counts > 0]


def correlation(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def top_overlap(a, b, fraction=0.125):
    k = int(round(a.size * fraction))
    top = lambda m: set(np.argpartition(m.ravel(), -k)[-k:])
    return len(top(a) & top(b)) / k


def statistics(maps):
    loupe_mean = {a: maps['loupe'][a].mean(0) for a in ANATOMIES}
    conv = maps['conv']
    conv_mean = {a: conv[a].mean(0) for a in ANATOMIES}
    stats = {}
    for anatomy in ANATOMIES:
        per_slice = conv[anatomy]
        stats[anatomy] = {
            'slices': len(per_slice),
            'roughness': {'loupe': float(roughness(loupe_mean[anatomy])),
                          'conv_per_slice_mean': float(roughness(per_slice).mean()),
                          'conv_anatomy_mean': float(roughness(conv_mean[anatomy]))},
            'entropy_bits_per_loc': {'loupe': float(binary_entropy_bits(loupe_mean[anatomy])),
                                     'conv_per_slice_mean': float(binary_entropy_bits(per_slice).mean())},
            'frac_p_below_0.01': {'loupe': float((loupe_mean[anatomy] < 0.01).mean()),
                                  'conv_per_slice_mean': float((per_slice < 0.01).mean())},
            'frac_p_above_0.99': {'loupe': float((loupe_mean[anatomy] > 0.99).mean()),
                                  'conv_per_slice_mean': float((per_slice > 0.99).mean())},
            'conv_mean_vs_loupe_corr': correlation(conv_mean[anatomy], loupe_mean[anatomy]),
            'conv_mean_vs_loupe_top12.5pct_overlap': top_overlap(conv_mean[anatomy], loupe_mean[anatomy]),
            'conv_slice_vs_own_anatomy_mean_corr': float(np.mean([correlation(s, conv_mean[anatomy]) for s in per_slice])),
            'conv_across_slice_std_mean': float(per_slice.std(0).mean()),
            'conv_sum_abs_diff_vs_loupe_frac': float(np.abs(conv_mean[anatomy] - loupe_mean[anatomy]).sum()
                                                     / (conv_mean[anatomy].sum() + loupe_mean[anatomy].sum())),
        }
    stats['conv_brain_vs_knee_mean_corr'] = correlation(conv_mean['brain'], conv_mean['knee'])
    stats['conv_brain_minus_knee_max_abs'] = float(np.abs(conv_mean['brain'] - conv_mean['knee']).max())
    return stats, loupe_mean, conv_mean


def figure(maps, loupe_mean, conv_mean, epochs, path):
    conv = maps['conv']
    gray, diverging = dict(cmap='gray', vmin=0, vmax=1), dict(cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    zoom = slice(64, 192)
    fig, axes = plt.subplots(4, 6, figsize=(21, 14.5))
    panels = [
        (axes[0, 0], loupe_mean['brain'], gray, f"LOUPE (epoch {epochs['loupe']})\nfull k-space"),
        (axes[0, 1], conv_mean['brain'], gray, 'conv, mean over brain val'),
        (axes[0, 2], conv_mean['knee'], gray, 'conv, mean over knee val'),
        (axes[0, 3], conv_mean['brain'] - loupe_mean['brain'], diverging, 'conv brain mean - LOUPE'),
        (axes[0, 4], conv_mean['brain'] - conv_mean['knee'], diverging, 'conv brain - knee mean'),
        (axes[0, 5], conv['brain'].std(0), dict(cmap='magma', vmin=0, vmax=0.3), 'conv across-slice std (brain)'),
        (axes[1, 0], loupe_mean['brain'][zoom, zoom], gray, 'LOUPE, centre 128x128'),
        (axes[1, 1], conv_mean['brain'][zoom, zoom], gray, 'conv brain mean, centre'),
        (axes[1, 2], conv_mean['knee'][zoom, zoom], gray, 'conv knee mean, centre'),
        (axes[1, 3], (conv_mean['brain'] - loupe_mean['brain'])[zoom, zoom], diverging, 'conv brain - LOUPE, centre'),
        (axes[1, 4], (conv_mean['brain'] - conv_mean['knee'])[zoom, zoom], diverging, 'conv brain - knee, centre'),
        (axes[1, 5], conv['knee'].std(0), dict(cmap='magma', vmin=0, vmax=0.3), 'conv across-slice std (knee)'),
    ]
    rng = np.random.RandomState(0)
    for column, anatomy in enumerate(ANATOMIES):
        picks = rng.choice(len(conv[anatomy]), EXAMPLES_PER_ANATOMY, replace=False)
        for offset, index in enumerate(picks):
            panels.append((axes[2, column * EXAMPLES_PER_ANATOMY + offset], conv[anatomy][index], gray,
                           f'conv, {anatomy} val slice {index}'))
            panels.append((axes[3, column * EXAMPLES_PER_ANATOMY + offset], conv[anatomy][index][zoom, zoom], gray,
                           f'{anatomy} slice {index}, centre'))
    for ax, image, style, title in panels:
        handle = ax.imshow(image, **style)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if style is diverging:
            fig.colorbar(handle, ax=ax, fraction=0.046)
    fig.suptitle('Effective sampling probability, R=8 (DC-centred, ky vertical, kx horizontal)', fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def radial_figure(maps, loupe_mean, conv_mean, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, anatomy in zip(axes, ANATOMIES):
        for label, image, style in [('LOUPE', loupe_mean[anatomy], '-'), ('conv mean', conv_mean[anatomy], '-')]:
            ax.plot(*radial_profile(image), style, label=label)
        profiles = np.stack([radial_profile(s)[1] for s in maps['conv'][anatomy]])
        centres = radial_profile(conv_mean[anatomy])[0]
        ax.fill_between(centres, *np.percentile(profiles, [5, 95], axis=0), alpha=0.25, color='C1', label='conv slices 5-95%')
        ax.set(xlabel='k-space radius (fraction of half-extent)', ylabel='sampling probability', yscale='log',
               title=f'{anatomy}: radial profile')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    loaded = {name: load_model(run_dir, args.checkpoint, args.device) for name, run_dir in RUNS.items()}
    models = {name: model for name, (model, _) in loaded.items()}
    epochs = {name: epoch + 1 for name, (_, epoch) in loaded.items()}

    cfg = OmegaConf.load(RUNS['conv'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    maps = collect(models, val_loader, args.device)
    stats, loupe_mean, conv_mean = statistics(maps)
    stats['epochs'] = epochs

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'mean_masks.npz', loupe_brain=loupe_mean['brain'], loupe_knee=loupe_mean['knee'],
                        conv_brain=conv_mean['brain'], conv_knee=conv_mean['knee'])
    (args.out / 'mask_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    figure(maps, loupe_mean, conv_mean, epochs, args.out / 'masks.png')
    radial_figure(maps, loupe_mean, conv_mean, args.out / 'radial.png')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
