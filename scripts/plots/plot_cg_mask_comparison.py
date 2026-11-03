"""CG-SENSE R=4 reconstructions and masks: fully sampled | PI | Poisson | LOUPE | anatomy-conditioned LOUPE.

Four rows: brain (FLAIR) image row, brain mask row, knee-PD image row, knee-PD mask row. Each image panel carries the
trainer-path (signal-masked) PSNR and foreground-only SSIM of the slice in its top-right corner. Per anatomy the slice is the one in the
middle third of its volume (by slice index) where |PSNR(LOUPE) - PSNR(anatomy-conditioned LOUPE)| is largest, over the
validation set, with one seeded Bernoulli draw per LOUPE mask.

LOUPE and anatomy-conditioned LOUPE are the cg_masks `single` / `routed` runs (own mask, own sensitivity U-Net). PI
(regular 2x2 grid, no stagger: plain parallel-imaging R=4) and Poisson-disc masks are fixed, built with the 24x24 centre, and reconstructed with CG-SENSE using the
`single` run's sensitivity U-Net (the fixed masks have no training run of their own).

    python scripts/plots/plot_cg_mask_comparison.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.samplers.caipi import caipi_mask
from activemri.samplers.poisson import poisson_mask
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask, anatomy_labels_from_batch, build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch

LOG_ROOT = Path('/home/remote/bkadota/scratch/logs/active_mri')
SINGLE_RUN = LOG_ROOT / 'cg_masks_single-2413163'
ROUTED_RUN = LOG_ROOT / 'cg_masks_routed-2413164'
CONDITIONS = ('brain', 'knee-PD')
ROW_TITLES = {'brain': 'Brain (FLAIR)', 'knee-PD': 'Knee (PD)'}
METHODS = ('pi', 'poisson', 'loupe', 'loupe_anat')
METHOD_TITLES = {'gt': 'Fully sampled', 'pi': 'PI + CG', 'poisson': 'Poisson + CG',
                 'loupe': 'LOUPE + CG', 'loupe_anat': 'LOUPE anatomy + CG'}
SUPTITLE = 'Sampling pattern comparison with CG-SENSE at R=4'
CALIBRATION = 24
ACCELERATION = 4.0
MASK_UPSAMPLE = 8
BUDGET_TOLERANCE = 0.01  # both LOUPE masks must sample 25% +- 1% of k-space


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/cg_mask_comparison_r4'))
    parser.add_argument('--middle', type=float, default=1 / 3, help='fraction of each volume, centred, searched')
    parser.add_argument('--per-anatomy', type=int, default=60, help='random candidate slices scored per anatomy (knee holds PD and PDFS)')
    parser.add_argument('--max-slices', type=int, default=None, help='cap on scored slices per condition (smoke test)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--flip-knee', action='store_true', help='flip knee reconstructions up-down and left-right for display (masks untouched)')
    parser.add_argument('--flip-knee-axes', type=int, nargs='+', default=[0, 1], help='image axes flipped by --flip-knee (0 up-down, 1 left-right)')
    parser.add_argument('--pin', nargs='+', default=[], metavar='CONDITION=FNAME:SLICE_ID',
                        help='use this validation slice for a condition instead of the largest-gap one, e.g. knee-PD=file1001193.h5:17; the LOUPE masks are one seeded draw')
    parser.add_argument('--sense-threshold', type=float, default=None, help='sensitivity-map support threshold as a fraction of the RSS maximum (default: the trained 0.1)')
    parser.add_argument('--replot', action='store_true', help='redraw the figures from the saved panels.npz / selection.json without running the models')
    parser.add_argument('--dpi', type=int, default=400)
    return parser.parse_args()


def load(run, device):
    checkpoint = torch.load(run / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.mask_train_mode = 'straight_through'
    return model, checkpoint['options']


def run_config(run, num_workers):
    overrides = [o for o in json.loads((run / 'run_info.json').read_text())['argv'][1:]
                 if not o.startswith(('hydra.job.name', 'output.out_dir'))]
    with initialize_config_dir(version_base=None, config_dir=str(Path('configs').resolve())):
        cfg = compose(config_name='config', overrides=overrides)
    cfg.datasets.num_workers = num_workers
    return cfg


def middle_indices(dataset, middle):
    """Global indices of the centred `middle` fraction of every volume's slices, per part of the concatenated dataset."""
    chosen, offset = [], 0
    for part in dataset.datasets:
        by_volume = defaultdict(list)
        for position, (fname, slice_id) in enumerate(part.examples):
            by_volume[fname].append((slice_id, offset + position))
        for slices in by_volume.values():
            slices.sort()
            lo = int(round(len(slices) * (1 - middle) / 2))
            hi = len(slices) - lo
            chosen += [index for _, index in slices[lo:hi]]
        offset += len(part)
    return sorted(chosen)


def subsample(dataset, indices, per_anatomy):
    """Random `per_anatomy` of the candidates from each part of the concatenated dataset (CPU budget)."""
    rng, kept, offset = np.random.RandomState(0), [], 0
    for part in dataset.datasets:
        own = [i for i in indices if offset <= i < offset + len(part)]
        kept += list(rng.choice(own, min(per_anatomy, len(own)), replace=False))
        offset += len(part)
    return sorted(int(i) for i in kept)


def find_slice(dataset, fname, slice_id):
    """Global index of (fname, slice_id) in the concatenated dataset, or None; filenames compare by basename (examples hold Paths)."""
    offset = 0
    for part in dataset.datasets:
        for position, example in enumerate(part.examples):
            if Path(example[0]).name == Path(fname).name and example[1] == slice_id:
                return offset + position
        offset += len(part)
    return None


def fixed_mask(model, kspace, pattern):
    """Centre of the model plus the fixed pattern, as the caipi_2d / poisson_2d samplers build it.

    The pattern is restricted to the acquired k-space (like the learned masks), so a narrower scan's zero-filled columns stay empty."""
    old_mask, valid_mask = model._initial_masks(kspace)
    return torch.clamp(old_mask + pattern.to(old_mask.dtype)[None, None] * valid_mask, max=1)


def metrics(output, target, kspace, anatomy):
    """Trainer-path signal-masked PSNR, and SSIM averaged over the foreground of the full (unzeroed) images.

    The trainer's SSIM scores images whose background was zeroed, which inflates it; here the SSIM map is computed on the
    complete images (data range = target maximum) and only its mean is restricted to the metric mask.
    """
    target4 = target if target.dim() == 4 else target.unsqueeze(1)
    mask = metric_mask_from_real_kspace(kspace, anatomy)
    masked_output, masked_target = _apply_signal_mask(output, target4, mask)
    psnr = compute_psnr_torch(masked_output, masked_target).flatten().cpu()
    ssim = []
    for out, tgt, m in zip(output.reshape(-1, *output.shape[-2:]).cpu().numpy(), target4.reshape(-1, *target4.shape[-2:]).cpu().numpy(),
                           mask.reshape(-1, *mask.shape[-2:]).cpu().numpy()):
        _, ssim_map = structural_similarity(tgt, out, data_range=float(tgt.max()), full=True)
        ssim.append(float(ssim_map[m > 0].mean()))
    return psnr, torch.tensor(ssim)


@torch.no_grad()
def reconstruct(model, batch, labels, mask):
    return model(batch['target'], batch['kspace'], anatomy_labels=labels, fixed_mask=mask)['output']


@torch.no_grad()
def loupe_masks(model, batch, labels, seed):
    probability = model.propose_mask(batch['kspace'], anatomy_labels=labels)['unscaled']
    old_mask, valid_mask = model._initial_masks(batch['kspace'])
    torch.manual_seed(seed)
    return model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]


def to_device(batch, device):
    out = dict(batch)
    for key in ('kspace', 'target'):
        out[key] = batch[key].to(device)
    return out


@torch.no_grad()
def scan(args, models, options, loader, indices):
    """Score LOUPE vs anatomy LOUPE on every candidate slice; keep the largest-gap slice per condition."""
    dataset = loader.dataset
    best = {}
    counts, skipped = defaultdict(int), defaultdict(int)
    gaps = defaultdict(list)
    subset = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, collate_fn=loader.collate_fn)
    for number, batch in enumerate(subset):
        batch_dev = to_device(batch, args.device)
        labels = {name: anatomy_labels_from_batch(batch, options[name].label_classes, args.device, options[name].label_key)
                  for name in ('loupe', 'loupe_anat')}
        masks, outputs, psnr, ssim = {}, {}, {}, {}
        for name in ('loupe', 'loupe_anat'):
            masks[name] = loupe_masks(models[name], batch_dev, labels[name], args.seed * 100003 + number)
            outputs[name] = reconstruct(models[name], batch_dev, labels[name], masks[name])
            psnr[name], ssim[name] = metrics(outputs[name], batch_dev['target'], batch_dev['kspace'], batch['anatomy'])
        for i, condition in enumerate(batch['condition']):
            if condition not in CONDITIONS:
                continue
            fractions = [float(masks[n][i].mean()) for n in masks]
            if any(abs(f - 1 / ACCELERATION) > BUDGET_TOLERANCE for f in fractions):
                skipped[condition] += 1  # an under/over-budget draw would show a sampling failure, not mask quality
                continue
            counts[condition] += 1
            gap = float(psnr['loupe_anat'][i] - psnr['loupe'][i])
            gaps[condition].append(gap)
            if condition not in best or abs(gap) > abs(best[condition]['gap']):
                best[condition] = {'gap': gap, 'index': indices[number * args.batch_size + i], 'fname': batch['fname'][i], 'slice_id': int(batch['slice_id'][i]),
                                   'masks': {n: masks[n][i:i + 1].cpu() for n in masks}}
        if args.max_slices and min((counts[c] for c in CONDITIONS), default=0) >= args.max_slices:
            break
    summary = {c: {'n_slices': counts[c], 'n_skipped_off_budget': skipped[c], 'mean_gap_db': float(np.mean(gaps[c])), 'std_gap_db': float(np.std(gaps[c])),
                   'selected_gap_db': best[c]['gap'], 'fname': best[c]['fname'], 'slice_id': best[c]['slice_id']} for c in CONDITIONS}
    return best, summary


@torch.no_grad()
def pinned_records(args, models, options, dataset):
    """`best`-style records for the slices given by --pin, with one seeded draw per LOUPE mask."""
    records, summary = {}, {}
    for spec in args.pin:
        condition, target = spec.split('=')
        fname, slice_id = target.rsplit(':', 1)
        index = find_slice(dataset, fname, int(slice_id))
        if index is None:
            raise SystemExit(f'--pin {spec}: slice is not in the validation set')
        sample = dataset[index]
        if sample['condition'] != condition:
            raise SystemExit(f'--pin {spec}: slice belongs to condition {sample["condition"]}')
        batch = {'kspace': sample['kspace'][None], 'target': sample['target'][None], 'anatomy': [sample['anatomy']], 'condition': [sample['condition']]}
        batch_dev = to_device(batch, args.device)
        masks = {name: loupe_masks(models[name], batch_dev,
                                   anatomy_labels_from_batch(batch, options[name].label_classes, args.device, options[name].label_key), args.seed * 100003).cpu()
                 for name in ('loupe', 'loupe_anat')}
        records[condition] = {'index': index, 'fname': fname, 'slice_id': int(slice_id), 'masks': masks}
        summary[condition] = {'fname': fname, 'slice_id': int(slice_id), 'pinned': True}
    return records, summary


@torch.no_grad()
def collect_panels(args, models, options, dataset, best):
    """Images, masks and metrics of the five columns for each selected slice."""
    panels = {}
    patterns = None
    for condition, record in best.items():
        sample = dataset[record['index']]
        batch = {'kspace': sample['kspace'][None], 'target': sample['target'][None], 'anatomy': [sample['anatomy']],
                 'condition': [sample['condition']]}
        batch_dev = to_device(batch, args.device)
        shape = tuple(batch_dev['kspace'].shape[-3:-1])
        if patterns is None:
            patterns = {'pi': caipi_mask(shape, accel_y=2, accel_z=2, shift=0), 'poisson': poisson_mask(shape, ACCELERATION, CALIBRATION)}
        labels = {name: anatomy_labels_from_batch(batch, options[name].label_classes, args.device, options[name].label_key)
                  for name in ('loupe', 'loupe_anat')}
        column = {'gt': {'image': batch['target'][0].squeeze().numpy(), 'mask': None}}
        for name in METHODS:
            model = models['loupe'] if name in ('pi', 'poisson') else models[name]
            label = labels['loupe'] if name in ('pi', 'poisson') else labels[name]
            mask = fixed_mask(model, batch_dev['kspace'], patterns[name].to(args.device)) if name in patterns \
                else record['masks'][name].to(args.device)
            output = reconstruct(model, batch_dev, label, mask)
            psnr, ssim = metrics(output, batch_dev['target'], batch_dev['kspace'], batch['anatomy'])
            column[name] = {'image': output[0].squeeze().cpu().numpy(), 'mask': mask[0, 0].cpu().numpy(),
                            'psnr': float(psnr[0]), 'ssim': float(ssim[0]), 'fraction': float(mask.mean())}
        panels[condition] = column
    return panels


def set_sense_threshold(model, threshold):
    """Binary support of the sensitivity maps is |RSS of the centre| > threshold * max (trained with 0.1); a smaller value keeps dimmer tissue."""
    if threshold is not None:
        model.sensetivity_net.forward = partial(model.sensetivity_net.forward, mask_threshold=threshold)


def make_figure(panels, best, path, dpi=400, titles=METHOD_TITLES, suptitle=None):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.6), gridspec_kw={'hspace': 0.04, 'wspace': 0.03})
    columns = ('gt',) + METHODS
    for block, condition in enumerate(CONDITIONS):
        column = panels[condition]
        vmax = float(column['gt']['image'].max())
        for j, name in enumerate(columns):
            image_ax, mask_ax = axes[2 * block, j], axes[2 * block + 1, j]
            entry = column[name]
            image_ax.imshow(entry['image'], cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
            if name == 'gt':
                mask_ax.axis('off')
            else:
                # Each pixel becomes a MASK_UPSAMPLE-square block, then Matplotlib downsamples with its antialiasing filter: a
                # direct nearest-neighbour resize aliases the 1-pixel PI lattice into moire.
                blocks = np.kron(entry['mask'], np.ones((MASK_UPSAMPLE, MASK_UPSAMPLE)))
                mask_ax.imshow(blocks, cmap='gray', vmin=0, vmax=1, interpolation='antialiased')
                image_ax.text(0.97, 0.97, f"SSIM {entry['ssim']:.3f}\nPSNR {entry['psnr']:.2f} dB", transform=image_ax.transAxes,
                              ha='right', va='top', fontsize=10.5, color='white', linespacing=1.3,
                              bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 3, 'edgecolor': 'none'})
            if block == 0:
                image_ax.set_title(titles[name], fontsize=13)
            for ax in (image_ax, mask_ax):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(name != 'gt' or ax is image_ax)
                    spine.set_color('0.7')
        axes[2 * block, 0].set_ylabel(ROW_TITLES[condition], fontsize=13)
        axes[2 * block + 1, 0].set_ylabel('sampling pattern', fontsize=13)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.93)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_error_figure(panels, path):
    """Where anatomy-conditioned LOUPE beats plain LOUPE: |err LOUPE| - |err anatomy LOUPE| (red = anatomy better), foreground only."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={'wspace': 0.04})
    for ax, condition in zip(axes, CONDITIONS):
        column = panels[condition]
        truth = column['gt']['image']
        difference = np.abs(column['loupe']['image'] - truth) - np.abs(column['loupe_anat']['image'] - truth)
        limit = float(np.percentile(np.abs(difference), 99.5))
        shown = ax.imshow(difference, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set_title(f"{ROW_TITLES[condition]}: LOUPE |err| - anatomy LOUPE |err|", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02, label='red: anatomy LOUPE closer to truth')
    fig.savefig(path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        saved, summary = np.load(args.out_dir / 'panels.npz'), json.loads((args.out_dir / 'selection.json').read_text())
        panels = {c: {n: {'image': saved[f'{c}__{n}__image'], 'mask': saved[f'{c}__{n}__mask'] if f'{c}__{n}__mask' in saved else None,
                          **summary['panels'][c].get(n, {})} for n in ('gt',) + METHODS} for c in CONDITIONS}
        if args.flip_knee:
            panels = {c: {n: {**e, 'image': np.flip(e['image'], axis=tuple(args.flip_knee_axes))} for n, e in column.items()} if c.startswith('knee') else column
                      for c, column in panels.items()}
        make_figure(panels, None, args.out_dir / 'cg_mask_comparison', args.dpi, suptitle=SUPTITLE)
        make_error_figure(panels, args.out_dir / 'cg_mask_error_difference')
        return
    models, options = {}, {}
    models['loupe'], options['loupe'] = load(SINGLE_RUN, args.device)
    models['loupe_anat'], options['loupe_anat'] = load(ROUTED_RUN, args.device)
    for model in models.values():
        set_sense_threshold(model, args.sense_threshold)
    _, loader = build_dataloaders(run_config(SINGLE_RUN, args.num_workers))
    best, summary = {}, {}
    if len({spec.split('=')[0] for spec in args.pin} & set(CONDITIONS)) < len(CONDITIONS):
        indices = subsample(loader.dataset, middle_indices(loader.dataset, args.middle), args.per_anatomy)
        print(f'{len(indices)} candidate validation slices (middle {args.middle:.2f} of every volume)', flush=True)
        best, summary = scan(args, models, options, loader, indices)
    pinned, pinned_summary = pinned_records(args, models, options, loader.dataset)
    best |= pinned
    summary |= pinned_summary
    panels = collect_panels(args, models, options, loader.dataset, best)
    make_figure(panels, best, args.out_dir / 'cg_mask_comparison', args.dpi, suptitle=SUPTITLE)
    make_error_figure(panels, args.out_dir / 'cg_mask_error_difference')
    np.savez_compressed(args.out_dir / 'panels.npz', **{f'{c}__{n}__{k}': v for c in CONDITIONS for n, e in panels[c].items() for k, v in e.items() if k in ('image', 'mask') and v is not None})
    summary['panels'] = {c: {n: {k: v for k, v in e.items() if k in ('psnr', 'ssim', 'fraction')} for n, e in panels[c].items() if n != 'gt'}
                         for c in CONDITIONS}
    (args.out_dir / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
