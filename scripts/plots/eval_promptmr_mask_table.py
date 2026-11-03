"""Validation-set-wide PromptMR/VarNet R=8 metrics for CAIPI | Poisson | LOUPE | anatomy-conditioned LOUPE (same runs as plot_promptmr_mask_comparison.py).

Every validation slice is scored with each run's own mask (seeded validation generator); per-slice PSNR and foreground SSIM go to per_slice.csv and
summary.md holds mean +- std per method plus paired differences against CAIPI, for all slices and for the centred middle third of each volume.

    python scripts/plots/eval_promptmr_mask_table.py --device cuda
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
from activemri.training.builders import build_dataloaders
from plot_cg_mask_comparison import METHODS, metrics, middle_indices, run_config, to_device
from plot_promptmr_mask_comparison import LOG_ROOT, RUNS, forward, load

CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/promptmr_mask_comparison_r8'))
    parser.add_argument('--checkpoint', default='epoch20_model.pt')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


@torch.no_grad()
def score(args, models, options, loader, indices, middle):
    rows = []
    subset = DataLoader(Subset(loader.dataset, indices), batch_size=args.batch_size, collate_fn=loader.collate_fn)
    for number, batch in enumerate(subset):
        batch_dev = to_device(batch, args.device) | {k: v for k, v in batch.items() if k not in ('kspace', 'target')}
        results = {}
        for name in METHODS:
            output, mask = forward(models[name], options[name], batch_dev, args.device)
            psnr, ssim = metrics(output, batch_dev['target'], batch_dev['kspace'], batch['anatomy'])
            results[name] = (psnr, ssim, mask.flatten(1).mean(1).cpu())
        for i, condition in enumerate(batch['condition']):
            for name in METHODS:
                psnr, ssim, fraction = results[name]
                rows.append({'condition': condition, 'fname': batch['fname'][i], 'slice_id': int(batch['slice_id'][i]),
                             'middle': int(indices[number * args.batch_size + i] in middle), 'method': name,
                             'psnr': float(psnr[i]), 'ssim': float(ssim[i]), 'fraction': float(fraction[i])})
        print(f'batch {number + 1}/{len(subset)}', flush=True)
    return rows


def summarize(rows):
    lines = []
    conditions = [c for c in CONDITIONS if any(r['condition'] == c for r in rows)]
    sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')  # noqa: E731
    for label, keep in (('all slices', lambda r: True), ('middle third of each volume', lambda r: r['middle'])):
        lines += [f'### {label}', '', '| Anatomy | Method | N | PSNR (dB) | SSIM | Sampled | dPSNR vs PI | dSSIM vs PI |', '|---|---|---|---|---|---|---|---|']
        for condition in conditions:
            by_method = {m: sorted((r for r in rows if r['condition'] == condition and r['method'] == m and keep(r)),
                                   key=lambda r: (r['fname'], r['slice_id'])) for m in METHODS}
            pi = by_method['pi']
            for method in METHODS:
                sel = by_method[method]
                if not sel:
                    continue
                psnr, ssim = np.array([r['psnr'] for r in sel]), np.array([r['ssim'] for r in sel])
                d_psnr, d_ssim = psnr - np.array([r['psnr'] for r in pi]), ssim - np.array([r['ssim'] for r in pi])
                delta = '-' if method == 'pi' else f'{d_psnr.mean():+.3f} +- {sem(d_psnr):.3f}'
                delta_s = '-' if method == 'pi' else f'{d_ssim.mean():+.4f} +- {sem(d_ssim):.4f}'
                lines.append(f"| {condition} | {method} | {len(sel)} | {psnr.mean():.2f} +- {psnr.std():.2f} | {ssim.mean():.4f} +- {ssim.std():.4f} "
                             f"| {100 * np.mean([r['fraction'] for r in sel]):.1f}% | {delta} | {delta_s} |")
        lines.append('')
    lines.append('Mean +- std over slices; "vs PI" columns (CAIPI baseline) are mean paired difference +- standard error.')
    return '\n'.join(lines)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models, options = {}, {}
    for name, run in RUNS.items():
        models[name], options[name], _ = load(run, args.checkpoint, args.device)
    _, loader = build_dataloaders(run_config(LOG_ROOT / RUNS['loupe'], args.num_workers))
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
