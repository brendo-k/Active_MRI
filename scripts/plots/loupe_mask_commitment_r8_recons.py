"""Reconstructions, error maps and sampled masks of the loupe_mask_commitment_r8 arms on the same val slices.

For each condition (brain, knee-PD, knee-PDFS) the slice is the one at the median control PSNR of --seed (ranked from the
eval's per-slice cache, so no outlier is picked). Every arm sees the same seeded mask draw. Errors are |recon - target|
inside the trainer's metric mask, x5, on the target's grey scale; the titles give that slice's PSNR inside the mask
(data range = the slice's max, as in the eval).

    CUDA_VISIBLE_DEVICES=1 python scripts/plots/loupe_mask_commitment_r8_recons.py \
        --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs --report reports/loupe_mask_commitment_r8 \
        --data-root '/export/scratch/bkadota/fastmri_raw/{anatomy}'
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.training.trainer import build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from varnet_loupe_lr_schedule_r8_eval import full_val_loader  # noqa: E402

ARMS = ('control', 'clamp', 'anneal', 'mix')
CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')
ERROR_GAIN = 5
DRAW_SEED = 1234


def pick_slices(cache, seed):
    psnr, conditions = cache[f'control_s{seed}/psnr'], cache[f'control_s{seed}/condition']
    picks = {}
    for condition in CONDITIONS:
        index = np.flatnonzero(conditions == condition)
        picks[condition] = int(index[np.argsort(psnr[index])[len(index) // 2]])
    return picks


@torch.no_grad()
def reconstruct(run_dir, batch, device):
    ckpt = torch.load(run_dir / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if hasattr(model, 'store_visuals'):
        model.store_visuals = False
    kspace, target = batch['kspace'].to(device), batch['target'].to(device)
    torch.manual_seed(DRAW_SEED)
    out = model(target, kspace)
    return out['output'].squeeze().cpu().numpy(), out['mask'].squeeze().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seed', default='43')
    parser.add_argument('--data-root')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--eval', default='eval_last_model', help='per-slice cache <eval>.per_slice.npz in --report')
    parser.add_argument('--name', help='figure name (default recons_s<seed>)')
    args = parser.parse_args()
    arms = tuple(args.arms)
    cache = np.load(args.report / f'{args.eval}.per_slice.npz')
    picks = pick_slices(cache, args.seed)
    loader = full_val_loader(args.runs / f'control_s{args.seed}', args.data_root)
    dataset, collate = loader.dataset, loader.collate_fn
    device = torch.device(args.device)

    columns = 1 + len(arms)
    fig, axes = plt.subplots(3 * len(CONDITIONS), columns, figsize=(2.6 * columns, 2.7 * 3 * len(CONDITIONS)))
    for ax in axes.flat:
        ax.axis('off')
    for block, condition in enumerate(CONDITIONS):
        index = picks[condition]
        batch = collate([dataset[index]])
        target = batch['target'].squeeze().numpy()
        metric = metric_mask_from_real_kspace(batch['kspace'], list(batch['anatomy'])).squeeze().float().numpy()
        vmax = np.percentile(target[metric > 0], 99.5)
        rows = axes[3 * block: 3 * block + 3]
        rows[0, 0].imshow(target, cmap='gray', vmin=0, vmax=vmax)
        rows[0, 0].set_title(f"{condition}\n{cache[f'control_s{args.seed}/volume'][index].split('/')[-1][:22]}", fontsize=8)
        for column, arm in enumerate(arms, start=1):
            recon, mask = reconstruct(args.runs / f'{arm}_s{args.seed}', batch, device)
            error = np.abs(recon - target) * metric
            psnr = 10 * np.log10(target.max() ** 2 / ((error ** 2).sum() / metric.sum()))
            rows[0, column].imshow(recon, cmap='gray', vmin=0, vmax=vmax)
            rows[0, column].set_title(f'{arm}_s{args.seed}\n{psnr:.2f} dB', fontsize=8)
            rows[1, column].imshow(error * ERROR_GAIN, cmap='inferno', vmin=0, vmax=vmax)
            rows[2, column].imshow(mask, cmap='gray', vmin=0, vmax=1, aspect='auto')
            rows[2, column].set_title(f'mask draw, {mask.mean():.3f} sampled', fontsize=7)
        rows[1, 0].text(0.5, 0.5, f'|error| x{ERROR_GAIN}\n(metric mask)', ha='center', va='center', fontsize=9)
        rows[2, 0].text(0.5, 0.5, 'sampled k-space\n(same draw seed)', ha='center', va='center', fontsize=9)
        print(condition, index, cache[f'control_s{args.seed}/volume'][index])
    fig.suptitle(f'Median-PSNR val slices, seed {args.seed}, last_model.pt', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    (args.report / 'figures').mkdir(exist_ok=True)
    out = args.report / 'figures' / f"{args.name or f'recons_s{args.seed}'}.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
