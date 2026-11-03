"""PromptMR R=8 reconstructions and masks: fully sampled | CAIPI | Poisson | LOUPE | anatomy-conditioned LOUPE.

Same layout as plot_cg_mask_comparison.py, but each column is its own end-to-end trained PromptMR run (own mask, own network),
read from `<run>/<checkpoint>` (default epoch20_model.pt, the common epoch the four runs are capped at). The masks are the ones the
model builds in its own forward pass, with the seeded validation mask generator, so the figure shows what the run saw at validation.

The slice per anatomy is the middle-third slice with the largest |PSNR(LOUPE) - PSNR(anatomy-conditioned LOUPE)| among `--per-anatomy`
random candidates. With `--use-cg-slice` the CG figure's slice is used instead whenever it lies in this run's validation set.

    python scripts/plots/plot_promptmr_mask_comparison.py --device cuda
    python scripts/plots/plot_promptmr_mask_comparison.py --replot
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import _path  # noqa: F401
import plot_cg_mask_comparison as cg_figure
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import REINFORCE_VAL_MASK_SEED, anatomy_labels_from_batch, build_model
from plot_cg_mask_comparison import (CONDITIONS, METHODS, make_error_figure, make_figure, metrics, middle_indices, run_config, subsample,
                                     to_device)

LOG_ROOT = Path('/home/remote/bkadota/scratch/logs/active_mri')
RUNS = {'pi': 'promptmr_caipi_trim', 'poisson': 'promptmr_poisson_trim', 'loupe': 'promptmr_loupe_trim', 'loupe_anat': 'promptmr_loupe_anat_trim'}
TITLES = {'gt': 'Fully sampled', 'pi': 'CAIPI + VarNet', 'poisson': 'Poisson + VarNet', 'loupe': 'LOUPE + VarNet',
          'loupe_anat': 'LOUPE anatomy + VarNet'}
ACCELERATION = 8.0
SUPTITLE = 'Sampling pattern comparison with VarNet at R=8'
KNEE_CONDITIONS = ('knee-PD', 'knee-PDFS')
SCAN_CONDITIONS = ('brain',) + KNEE_CONDITIONS
cg_figure.ROW_TITLES['knee-PDFS'] = 'Knee (PDFS)'
BUDGET_TOLERANCE = 0.01
CG_SELECTION = Path('reports/cg_mask_comparison_r4/selection.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/promptmr_mask_comparison_r8'))
    parser.add_argument('--checkpoint', default='epoch20_model.pt')
    parser.add_argument('--middle', type=float, default=1 / 3)
    parser.add_argument('--per-anatomy', type=int, default=30, help='random candidate slices scored per anatomy for the largest LOUPE gap')
    parser.add_argument('--select', choices=('abs', 'anat_better', 'loupe_better'), default='abs',
                        help='abs: largest |gap|; anat_better: slice where anatomy LOUPE beats LOUPE most; loupe_better: the reverse')
    parser.add_argument('--use-cg-slice', action='store_true', help='prefer the CG figure slice over the largest-gap slice when it is in the validation set')
    parser.add_argument('--cg-selection', type=Path, default=CG_SELECTION, help='selection.json of the CG figure read by --use-cg-slice')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dpi', type=int, default=400)
    parser.add_argument('--flip-knee', action='store_true', help='flip knee reconstructions up-down and left-right for display (masks untouched)')
    parser.add_argument('--knee', choices=KNEE_CONDITIONS, help='with --replot: show this knee type instead of the one saved in selection.json')
    parser.add_argument('--sense-threshold', type=float, default=None, help='sensitivity-map support threshold as a fraction of the RSS maximum (default: the trained 0.1)')
    parser.add_argument('--keep-unacquired', action='store_true', help='do not zero the fixed masks and the predicted k-space where the scan acquired nothing (the trained behaviour)')
    parser.add_argument('--replot', action='store_true', help='redraw from the saved panels.npz / selection.json without running the models')
    return parser.parse_args()


def load(run, checkpoint, device):
    ckpt = torch.load(LOG_ROOT / run / checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.requires_grad_(False)
    return model, ckpt['options'], ckpt.get('epoch')


def forward(model, options, batch, device):
    """Model output and mask for a collated batch, with the validation mask generator reseeded so the draw is reproducible."""
    generator = torch.Generator(device=device)
    generator.manual_seed(REINFORCE_VAL_MASK_SEED)
    model.sampler.eval_generator = generator
    extra = {key: batch[key].to(device) for key in ('sense_maps', 'kspace_support') if key in batch}
    labels = anatomy_labels_from_batch(batch, options.label_classes, device, options.label_key)
    with torch.no_grad():
        pred = model(batch['target'], batch['kspace'], anatomy_labels=labels, **extra)
    output = pred['output'][-1] if isinstance(pred['output'], list) else pred['output']
    return output, pred['mask']


def find_slice(dataset, fname, slice_id):
    offset = 0
    for part in dataset.datasets:
        for position, example in enumerate(part.examples):
            if Path(example[0]).name == Path(fname).name and example[1] == slice_id:
                return offset + position
        offset += len(part)
    return None


def scan(args, models, options, loader, indices):
    """Largest |PSNR gap| LOUPE vs anatomy LOUPE per condition over the candidate slices."""
    best, counts, skipped, gaps = {}, defaultdict(int), defaultdict(int), defaultdict(list)
    subset = DataLoader(Subset(loader.dataset, indices), batch_size=args.batch_size, collate_fn=loader.collate_fn)
    for number, batch in enumerate(subset):
        batch = to_device(batch, args.device) | {k: v for k, v in batch.items() if k not in ('kspace', 'target')}
        psnr, masks = {}, {}
        for name in ('loupe', 'loupe_anat'):
            output, masks[name] = forward(models[name], options[name], batch, args.device)
            psnr[name], _ = metrics(output, batch['target'], batch['kspace'], batch['anatomy'])
        for i, condition in enumerate(batch['condition']):
            if condition not in SCAN_CONDITIONS:
                continue
            if any(abs(float(masks[n][i].mean()) - 1 / ACCELERATION) > BUDGET_TOLERANCE for n in masks):
                skipped[condition] += 1
                continue
            counts[condition] += 1
            gap = float(psnr['loupe_anat'][i] - psnr['loupe'][i])
            gaps[condition].append(gap)
            score = {'abs': abs(gap), 'anat_better': gap, 'loupe_better': -gap}[args.select]
            if condition not in best or score > best[condition]['score']:
                best[condition] = {'gap': gap, 'score': score, 'index': indices[number * args.batch_size + i]}
        print(f'scan batch {number + 1}/{len(subset)}', flush=True)
    summary = {c: {'n_slices': counts[c], 'n_skipped_off_budget': skipped[c], 'mean_gap_db': float(np.mean(gaps[c])), 'std_gap_db': float(np.std(gaps[c])),
                   'selected_gap_db': best[c]['gap'], 'score': best[c]['score'], 'max_gap_db': float(np.max(gaps[c])), 'min_gap_db': float(np.min(gaps[c]))} for c in SCAN_CONDITIONS if gaps[c]}
    return {c: best[c]['index'] for c in best}, summary


def collect_panels(args, models, options, dataset, collate_fn, chosen):
    panels = {}
    for condition, index in chosen.items():
        batch = collate_fn([dataset[index]])
        batch = to_device(batch, args.device) | {k: v for k, v in batch.items() if k not in ('kspace', 'target')}
        column = {'gt': {'image': batch['target'][0].squeeze().cpu().numpy(), 'mask': None}}
        for name in METHODS:
            output, mask = forward(models[name], options[name], batch, args.device)
            psnr, ssim = metrics(output, batch['target'], batch['kspace'], batch['anatomy'])
            column[name] = {'image': output[0].squeeze().cpu().numpy(), 'mask': mask[0, 0].cpu().numpy(),
                            'psnr': float(psnr[0]), 'ssim': float(ssim[0]), 'fraction': float(mask.mean())}
        panels[condition] = (column, batch['fname'][0], int(batch['slice_id'][0]))
    return panels


def flip_knee_images(panels):
    return {c: {n: {**e, 'image': np.flip(e['image'], axis=(0, 1))} for n, e in column.items()} if c in KNEE_CONDITIONS else column
            for c, column in panels.items()}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        saved, summary = np.load(args.out_dir / 'panels.npz'), json.loads((args.out_dir / 'selection.json').read_text())
        if args.knee:
            summary['shown'] = ['brain', args.knee]
        panels = {c: {n: {'image': saved[f'{c}__{n}__image'], 'mask': saved[f'{c}__{n}__mask'] if f'{c}__{n}__mask' in saved else None,
                          **summary['panels'][c].get(n, {})} for n in ('gt',) + METHODS} for c in summary.get('shown', list(CONDITIONS))}
        cg_figure.CONDITIONS = tuple(summary.get('shown', list(CONDITIONS)))
        if args.flip_knee:
            panels = flip_knee_images(panels)
        make_figure(panels, None, args.out_dir / 'promptmr_mask_comparison', args.dpi, TITLES, suptitle=SUPTITLE)
        make_error_figure(panels, args.out_dir / 'promptmr_mask_error_difference')
        return
    models, options, epochs = {}, {}, {}
    for name, run in RUNS.items():
        models[name], options[name], epochs[name] = load(run, args.checkpoint, args.device)
        models[name].zero_unacquired = not args.keep_unacquired
        cg_figure.set_sense_threshold(models[name], args.sense_threshold)
    print('checkpoint epochs (0-based as saved):', epochs, flush=True)
    _, loader = build_dataloaders(run_config(LOG_ROOT / RUNS['loupe'], args.num_workers))
    chosen, scan_summary = {}, {}
    cg = json.loads(args.cg_selection.read_text()) if args.use_cg_slice and args.cg_selection.exists() else {}
    for condition in CONDITIONS:
        if condition in cg:
            index = find_slice(loader.dataset, cg[condition]['fname'], cg[condition]['slice_id'])
            if index is not None:
                chosen[condition] = index
                print(f'{condition}: using the CG figure slice {cg[condition]["fname"]}:{cg[condition]["slice_id"]}', flush=True)
    missing = [c for c in SCAN_CONDITIONS if c not in chosen and (c in CONDITIONS or not cg)]
    if missing:
        print(f'scanning candidates for the largest LOUPE vs anatomy LOUPE gap: {missing}', flush=True)
        indices = subsample(loader.dataset, middle_indices(loader.dataset, args.middle), args.per_anatomy)
        picked, scan_summary = scan(args, models, options, loader, indices)
        chosen |= {c: picked[c] for c in missing if c in picked}
    panels = collect_panels(args, models, options, loader.dataset, loader.collate_fn, chosen)
    knees = [c for c in KNEE_CONDITIONS if c in panels]
    knee = max(knees, key=lambda c: scan_summary[c]['score'] if c in scan_summary else float('-inf'))  # knee type with the larger selected effect
    shown = ['brain', knee]
    print(f'knee type shown: {knee} (selection score {({c: scan_summary.get(c, {}).get("score") for c in knees})})', flush=True)
    cg_figure.CONDITIONS = tuple(shown)
    figure_panels = {c: p[0] for c, p in panels.items()}
    make_figure({c: figure_panels[c] for c in shown}, None, args.out_dir / 'promptmr_mask_comparison', args.dpi, TITLES, suptitle=SUPTITLE)
    make_error_figure({c: figure_panels[c] for c in shown}, args.out_dir / 'promptmr_mask_error_difference')
    np.savez_compressed(args.out_dir / 'panels.npz', **{f'{c}__{n}__{k}': v for c in panels for n, e in figure_panels[c].items()
                                                        for k, v in e.items() if k in ('image', 'mask') and v is not None})
    summary = {'checkpoint': args.checkpoint, 'select': args.select, 'shown': shown, 'epochs': epochs, 'scan': scan_summary,
               'slices': {c: {'fname': p[1], 'slice_id': p[2]} for c, p in panels.items()},
               'panels': {c: {n: {k: v for k, v in e.items() if k in ('psnr', 'ssim', 'fraction')} for n, e in figure_panels[c].items() if n != 'gt'}
                          for c in panels}}
    (args.out_dir / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
