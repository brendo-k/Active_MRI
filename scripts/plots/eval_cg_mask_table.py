"""Validation-set-wide CG-SENSE R=4 metrics for PI | Poisson | LOUPE | anatomy-conditioned LOUPE (same models/masks as plot_cg_mask_comparison.py).

Every validation slice of brain and knee-PD is scored (optionally `--stride N` to thin it); per-slice PSNR (trainer-path signal-masked) and
foreground SSIM are saved to per_slice.csv, and summary.md holds mean +- std per method plus paired differences against PI, for all slices
and for the centred middle third of each volume. The LOUPE masks are one seeded Bernoulli draw per slice, as in the figure.

    python scripts/plots/eval_cg_mask_table.py --device cpu
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import _path  # noqa: F401
from activemri.samplers.caipi import caipi_mask
from activemri.samplers.poisson import poisson_mask
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import anatomy_labels_from_batch
from plot_cg_mask_comparison import (ACCELERATION, CALIBRATION, CONDITIONS, METHODS, ROUTED_RUN, SINGLE_RUN, fixed_mask, load, loupe_masks,
                                     metrics, middle_indices, reconstruct, run_config, to_device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/cg_mask_comparison_r4'))
    parser.add_argument('--stride', type=int, default=1, help='score every Nth validation slice')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def score(args, models, options, loader, indices, middle):
    rows, patterns = [], None
    subset = DataLoader(Subset(loader.dataset, indices), batch_size=args.batch_size, collate_fn=loader.collate_fn)
    for number, batch in enumerate(subset):
        batch_dev = to_device(batch, args.device)
        if patterns is None:
            shape = tuple(batch_dev['kspace'].shape[-3:-1])
            patterns = {'pi': caipi_mask(shape, accel_y=2, accel_z=2, shift=0), 'poisson': poisson_mask(shape, ACCELERATION, CALIBRATION)}
        labels = {n: anatomy_labels_from_batch(batch, options[n].label_classes, args.device, options[n].label_key) for n in ('loupe', 'loupe_anat')}
        results = {}
        for name in METHODS:
            model = models['loupe'] if name in patterns else models[name]
            label = labels['loupe'] if name in patterns else labels[name]
            mask = fixed_mask(model, batch_dev['kspace'], patterns[name].to(args.device)) if name in patterns \
                else loupe_masks(model, batch_dev, label, args.seed * 100003 + number)
            output = reconstruct(model, batch_dev, label, mask)
            psnr, ssim = metrics(output, batch_dev['target'], batch_dev['kspace'], batch['anatomy'])
            results[name] = (psnr, ssim, mask.flatten(1).mean(1).cpu())
        for i, condition in enumerate(batch['condition']):
            if condition not in CONDITIONS:
                continue
            for name in METHODS:
                psnr, ssim, fraction = results[name]
                rows.append({'condition': condition, 'fname': batch['fname'][i], 'slice_id': int(batch['slice_id'][i]),
                             'middle': int(indices[number * args.batch_size + i] in middle), 'method': name,
                             'psnr': float(psnr[i]), 'ssim': float(ssim[i]), 'fraction': float(fraction[i])})
        print(f'batch {number + 1}/{len(subset)}', flush=True)
    return rows


def summarize(rows):
    lines = []
    for label, keep in (('all slices', lambda r: True), ('middle third of each volume', lambda r: r['middle'])):
        lines += [f'### {label}', '', '| Anatomy | Method | N | PSNR (dB) | SSIM | Sampled | dPSNR vs PI | dSSIM vs PI |', '|---|---|---|---|---|---|---|---|']
        for condition in CONDITIONS:
            by_method = {m: sorted((r for r in rows if r['condition'] == condition and r['method'] == m and keep(r)),
                                   key=lambda r: (r['fname'], r['slice_id'])) for m in METHODS}
            pi = by_method['pi']
            for method in METHODS:
                sel = by_method[method]
                if not sel:
                    continue
                psnr, ssim = np.array([r['psnr'] for r in sel]), np.array([r['ssim'] for r in sel])
                d_psnr, d_ssim = psnr - np.array([r['psnr'] for r in pi]), ssim - np.array([r['ssim'] for r in pi])
                sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')  # noqa: E731
                delta = '-' if method == 'pi' else f'{d_psnr.mean():+.3f} +- {sem(d_psnr):.3f}'
                delta_s = '-' if method == 'pi' else f'{d_ssim.mean():+.4f} +- {sem(d_ssim):.4f}'
                lines.append(f"| {condition} | {method} | {len(sel)} | {psnr.mean():.2f} +- {psnr.std():.2f} | {ssim.mean():.4f} +- {ssim.std():.4f} "
                             f"| {100 * np.mean([r['fraction'] for r in sel]):.1f}% | {delta} | {delta_s} |")
        lines.append('')
    lines.append('Mean +- std over slices; "vs PI" columns are mean paired difference +- standard error.')
    return '\n'.join(lines)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models, options = {}, {}
    models['loupe'], options['loupe'] = load(SINGLE_RUN, args.device)
    models['loupe_anat'], options['loupe_anat'] = load(ROUTED_RUN, args.device)
    _, loader = build_dataloaders(run_config(SINGLE_RUN, args.num_workers))
    indices = list(range(0, len(loader.dataset), args.stride))
    middle = set(middle_indices(loader.dataset, 1 / 3))
    print(f'{len(indices)} validation slices (stride {args.stride})', flush=True)
    rows = score(args, models, options, loader, indices, middle)
    with open(args.out_dir / 'per_slice.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = summarize(rows)
    (args.out_dir / 'summary.md').write_text(table + '\n')
    print(table)


if __name__ == '__main__':
    main()
