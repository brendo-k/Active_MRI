"""Compare R=8 brain+knee VarNet runs across sampler families: metrics table and learned masks.

Metrics are parsed from each run's train_sequential.log, i.e. the trainer's own masked
per-anatomy validation path, at the epoch whose weights were saved as best_model.pt.
Masks are what each checkpoint's sampler would acquire in eval mode, on the same
pre-selected centre:

    loupe_2d straight-through   the rescaled probability map (the mask is a random draw of it)
    loupe_2d reinforce          the deterministic top-k of the policy logits
    autosamp_2d                 the positions snapped to the grid

None of these samplers is anatomy-conditioned, so one pattern serves every slice. The
per-anatomy patterns shown are the mean effective mask over the validation set: samples
that land in a slice's zero-padded k-space are discarded, so brain and knee differ only
where their valid regions do.

    python scripts/eval/compare_samplers_r8.py
    python scripts/eval/compare_samplers_r8.py loupe=checkpoints/a reinforce=checkpoints/b ...
"""
import argparse
import re
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
from activemri.training.trainer import build_model
from activemri.training.builders import build_dataloaders

DEFAULT_RUNS = {
    'loupe': 'checkpoints/loupe2d_varnet_r8_brain_knee',
    'reinforce v1': 'checkpoints/reinforce2d_varnet_r8_brain_knee',
    'reinforce v2': 'checkpoints/reinforce2d_varnet_r8_brain_knee_v2',
    'autosamp lr0.01': 'checkpoints/autosamp2d_varnet_r8_brain_knee',
    'autosamp lr0.05': 'checkpoints/autosamp2d_varnet_r8_brain_knee_lr0.05',
    'autosamp normal init': 'checkpoints/autosamp2d_varnet_r8_brain_knee_normalinit',
    'reinforce vd2.5 (100ep)': 'checkpoints/reinforce2d_varnet_r8_brain_knee_vd2.5_100ep',
}
OUT_DIR = Path('reports/sampler_comparison_r8')
RADIAL_EDGES = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])  # in units of the k-space half-extent
ANATOMIES = ('brain', 'knee')

METRIC_LINE = re.compile(r'\[(\w+)\] MSE=\S+ NMSE=(\S+) PSNR=(\S+) SSIM=(\S+)')
EPOCH_LINE = re.compile(r'Epoch = \[\s*(\d+)/\s*(\d+)\]')


def validation_by_epoch(log_path):
    """{epoch: {anatomy: {NMSE, PSNR, SSIM}}}; each epoch's metric line precedes its 'Epoch =' line."""
    epochs, pending = {}, {}
    for line in log_path.read_text().splitlines():
        for anatomy, nmse, psnr, ssim in METRIC_LINE.findall(line):
            pending[anatomy] = {'NMSE': float(nmse), 'PSNR': float(psnr), 'SSIM': float(ssim)}
        epoch_match = EPOCH_LINE.search(line)
        if epoch_match:
            epochs[int(epoch_match.group(1))] = pending
            pending = {}
    return epochs


def learned_mask(checkpoint):
    """Eval-mode new samples of a checkpoint's sampler, [H, W] in DC-centred layout, with the pre-selected centre."""
    options = checkpoint['options']
    model = build_model(options)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    height, width = model.shape
    kspace = torch.zeros(1, 1, height, width, 2)
    old_mask = model._init_mask(kspace)
    with torch.no_grad():
        _, vis, *_ = model.sampler(torch.zeros(1, height, width, 2), old_mask, model.budget)
    return vis['rescaled_mask'].float().numpy(), old_mask[0, 0].numpy()


def radial_density(mask):
    height, width = mask.shape
    rows = (np.arange(height) - height // 2) / (height / 2)
    cols = (np.arange(width) - width // 2) / (width / 2)
    radius = np.hypot(rows[:, None], cols[None, :])
    bins = np.digitize(radius, RADIAL_EDGES) - 1
    return np.array([mask[bins == i].mean() if (bins == i).any() else np.nan for i in range(len(RADIAL_EDGES) - 1)])


def mean_valid_masks(run_dir):
    """{anatomy: [H, W]} fraction of validation slices whose k-space is non-zero at each location.

    The validation set is the same for every run being compared, so any run's config will do.
    """
    cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    totals, counts = {}, {}
    for batch in val_loader:
        valid = (torch.view_as_complex(batch['kspace'].contiguous()).abs().sum(dim=1) > 0).float()  # [B, H, W]
        for anatomy, mask in zip(batch['anatomy'], valid):
            totals[anatomy] = totals.get(anatomy, 0) + mask
            counts[anatomy] = counts.get(anatomy, 0) + 1
    return {anatomy: (totals[anatomy] / counts[anatomy]).numpy() for anatomy in totals}


def anatomy_patterns(mask, preselected, valid_by_anatomy):
    """Mean effective mask per anatomy: the pre-selected centre plus the new samples that survive each slice's valid region."""
    return {anatomy: np.clip(preselected + mask * valid, 0, 1) for anatomy, valid in valid_by_anatomy.items()}


def summarize(name, run_dir, valid_by_anatomy):
    checkpoint = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    best_epoch = checkpoint['epoch'] + 1
    epochs = validation_by_epoch(run_dir / 'train_sequential.log')
    final_epoch = max(epochs)
    mask, preselected = learned_mask(checkpoint)
    return {
        'name': name, 'best_epoch': best_epoch, 'final_epoch': final_epoch,
        'metrics': epochs[best_epoch], 'final': epochs[final_epoch], 'epochs': epochs,
        'mask': mask, 'preselected': preselected,
        'patterns': anatomy_patterns(mask, preselected, valid_by_anatomy),
    }


def metrics_table(summaries):
    header = ['run', 'epochs', 'best'] + [f'{a} {m}' for a in ANATOMIES for m in ('SSIM', 'PSNR', 'NMSE')]
    rows = [header, ['---'] * len(header)]
    for s in summaries:
        cells = [s['name'], str(s['final_epoch']), str(s['best_epoch'])]
        for anatomy in ANATOMIES:
            m = s['metrics'][anatomy]
            cells += [f"{m['SSIM']:.4f}", f"{m['PSNR']:.2f}", f"{m['NMSE']:.4f}"]
        rows.append(cells)
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)


def figure(summaries, path):
    """Masks on top; below, radial density and the validation curves of every run."""
    colors = {s['name']: plt.cm.tab10(i) for i, s in enumerate(summaries)}
    reference = next((s for s in summaries if s['name'] == 'loupe'), summaries[0])
    fig = plt.figure(figsize=(max(2.6 * len(summaries), 14), 15))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.0], hspace=0.1)
    pattern_grid = grid[0].subgridspec(4, len(summaries), hspace=0.05, wspace=0.04)

    gray, diverging = dict(cmap='gray', vmin=0, vmax=1), dict(cmap='RdBu_r', vmin=-1, vmax=1)
    rows = [('brain', lambda s: s['patterns']['brain'], gray),
            ('knee', lambda s: s['patterns']['knee'], gray),
            ('brain - knee', lambda s: s['patterns']['brain'] - s['patterns']['knee'], diverging),
            (f"brain - {reference['name']}", lambda s: s['patterns']['brain'] - reference['patterns']['brain'], diverging)]
    for column, s in enumerate(summaries):
        for row, (label, pattern, style) in enumerate(rows):
            ax = fig.add_subplot(pattern_grid[row, column])
            image = pattern(s)
            ax.imshow(image, **style)
            ax.set_xticks([]); ax.set_yticks([])
            if style is diverging:
                ax.text(0.5, 0.03, f'max |diff| = {np.abs(image).max():.2f}', transform=ax.transAxes, ha='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
            if row == 0:
                ax.set_title(f"{s['name']}\n(epoch {s['final_epoch']})", fontsize=9, color=colors[s['name']])
            if column == 0:
                ax.set_ylabel(label, fontsize=10)

    bottom = grid[1].subgridspec(1, 4, wspace=0.28)
    centres = (RADIAL_EDGES[:-1] + RADIAL_EDGES[1:]) / 2
    density_axis = fig.add_subplot(bottom[0, 0])
    for s in summaries:
        density_axis.plot(centres, radial_density(np.clip(s['mask'] + s['preselected'], 0, 1)),
                          marker='o', ms=3, color=colors[s['name']], label=s['name'])
    density_axis.set(xlabel='k-space radius (fraction of half-extent)', ylabel='sampling density', xscale='symlog', title='Sampling density')
    density_axis.set_xscale('symlog', linthresh=0.05)
    density_axis.legend(fontsize=7)

    for slot, (anatomy, metric) in enumerate([('brain', 'SSIM'), ('knee', 'SSIM'), ('brain', 'NMSE')], start=1):
        ax = fig.add_subplot(bottom[0, slot])
        for s in summaries:
            epochs = sorted(s['epochs'])
            ax.plot(epochs, [s['epochs'][e][anatomy][metric] for e in epochs], color=colors[s['name']], label=s['name'])
        ax.set(xlabel='epoch', title=f'Validation {anatomy} {metric}')
        if metric == 'SSIM':
            ax.set_ylim(0.88, 0.96)
        else:
            ax.set_yscale('log')
        ax.grid(alpha=0.3)
    fig.suptitle('R=8 brain+knee VarNet: sampler comparison', fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', help='name=checkpoint_dir pairs; default is the 50-epoch R=8 set')
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    runs = dict(pair.split('=', 1) for pair in args.runs) if args.runs else DEFAULT_RUNS
    available = {name: Path(d) for name, d in runs.items() if (Path(d) / 'best_model.pt').exists()}
    for name in runs.keys() - available.keys():
        print(f'skipping {name}: no best_model.pt in {runs[name]}')
    valid_by_anatomy = mean_valid_masks(next(iter(available.values())))
    summaries = [summarize(name, directory, valid_by_anatomy) for name, directory in available.items()]

    args.out.mkdir(parents=True, exist_ok=True)
    table = metrics_table(summaries)
    (args.out / 'metrics.md').write_text(table + '\n')
    figure(summaries, args.out / 'overview.png')
    print(table)
    print(f'\nwrote {args.out}/metrics.md and overview.png')


if __name__ == '__main__':
    main()
