"""Is the learned mask's column ripple present in the brain k-space itself?

Over the brain T2 validation set (k-space exactly as the model sees it) this measures, per phase-encode
column, (a) mean log coil-RSS magnitude and (b) column-to-column phase coherence
|sum k_x conj(k_{x-1})| / sum |k_x||k_{x-1}|, then compares the dataset mean and every individual
scan with the ripple of the learned masks (mean over seeds, detrended mean over rows).

    python scripts/check_kspace_column_ripple.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from plot_mean_masks import load_probability

SMOOTH_WIDTH = 13
COLUMNS = slice(8, 120)
MASK_RUNS = {'brain, 12 casc': 'runs/loupe_anatomy/brain/seed_*', 'brain, 1 casc': 'runs/loupe_casc1/brain/seed_*'}


def detrend(profile):
    return profile - uniform_filter1d(profile, SMOOTH_WIDTH, mode='nearest', axis=-1)


def zscore(v):
    return (v - v.mean()) / v.std()


def corr(a, b):
    return float(np.corrcoef(a[COLUMNS], b[COLUMNS])[0, 1])


def mask_ripple(pattern):
    masks = [load_probability(r) for r in sorted(Path('.').glob(pattern))]
    return detrend(np.mean(masks, axis=0).mean(axis=0))


def measure(loader):
    log_mag_sum, count = 0.0, 0
    mag_profiles, coherence_profiles, names = [], [], []
    for batch in loader:
        k = torch.view_as_complex(batch['kspace'].contiguous()).to(torch.complex128)  # B, C, H, W
        mag = k.abs().pow(2).sum(1).sqrt()                                              # B, H, W
        log_mag = torch.log(mag + 1e-12)
        log_mag_sum = log_mag_sum + log_mag.sum(0).numpy()
        count += k.shape[0]
        mag_profiles.append(log_mag.mean(1).numpy())
        cross = (k[..., 1:] * k[..., :-1].conj()).sum(dim=(1, 2)).abs()
        norm = (k[..., 1:].abs() * k[..., :-1].abs()).sum(dim=(1, 2))
        coherence = torch.zeros(k.shape[0], k.shape[-1], dtype=torch.float64)
        coherence[:, 1:] = cross / norm
        coherence_profiles.append(coherence.numpy())
        names += [f'{f}' for f in batch['fname']]
    return log_mag_sum / count, np.concatenate(mag_profiles), np.concatenate(coherence_profiles), names


def per_scan(profiles, names):
    groups = defaultdict(list)
    for p, n in zip(profiles, names):
        groups[n].append(p)
    return {n: np.mean(v, axis=0) for n, v in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--out', type=Path, default=Path('runs/loupe_anatomy/analysis/kspace_column_ripple.png'))
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    _, loader = build_fastmri_dataloaders(
        data_root=args.data_root, anatomies=['brain'], image_shape=[256, 256], batch_size=8,
        num_coils=16, num_workers=args.num_workers, contrast='T2')
    mean_log_mag, mag_profiles, coherence_profiles, names = measure(loader)
    ripples = {label: mask_ripple(pattern) for label, pattern in MASK_RUNS.items()}

    mean_profiles = {'mean log|k|': detrend(mag_profiles.mean(0)), 'phase coherence': detrend(coherence_profiles.mean(0))}
    scan_corr = {}
    for quantity, profiles in [('mean log|k|', mag_profiles), ('phase coherence', coherence_profiles)]:
        scans = per_scan(profiles, names)
        for label, ripple in ripples.items():
            scan_corr[quantity, label] = np.array([corr(detrend(p), ripple) for p in scans.values()])
    print(f'{len(names)} slices, {len(per_scan(mag_profiles, names))} scans')
    for quantity, profile in mean_profiles.items():
        for label, ripple in ripples.items():
            c = scan_corr[quantity, label]
            print(f'{quantity:16s} vs {label:14s}: dataset-mean r = {corr(profile, ripple):+.2f}; '
                  f'per-scan r median {np.median(c):+.2f}, 90% range [{np.percentile(c, 5):+.2f}, {np.percentile(c, 95):+.2f}], '
                  f'|r|>0.3 in {np.mean(np.abs(c) > 0.3):.0%} of scans')

    fig = plt.figure(figsize=(14, 14))
    grid = fig.add_gridspec(4, 2, height_ratios=[3, 1.2, 1.2, 1.4])
    ax = fig.add_subplot(grid[0, 0])
    shown = ax.imshow(mean_log_mag, cmap='viridis')
    fig.colorbar(shown, ax=ax, fraction=0.046, label='mean log coil-RSS |k|')
    ax.set(title=f'Dataset-mean brain k-space magnitude ({len(names)} val slices)', xlabel='kx column (phase-encode axis)', ylabel='row (readout)')
    ax = fig.add_subplot(grid[0, 1])
    residual = mean_log_mag - uniform_filter1d(mean_log_mag, SMOOTH_WIDTH, mode='nearest', axis=1)
    limit = np.percentile(np.abs(residual), 99)
    shown = ax.imshow(residual, cmap='RdBu', vmin=-limit, vmax=limit)
    fig.colorbar(shown, ax=ax, fraction=0.046, label='residual after removing column-wise trend')
    ax.set_title('Same map minus its moving average along kx: vertical streaks would show here', fontsize=9)
    ax.set_xlabel('kx column (phase-encode axis)')
    columns = np.arange(256)[COLUMNS]
    for row, (quantity, profile) in enumerate(mean_profiles.items(), start=1):
        ax = fig.add_subplot(grid[row, :])
        ax.plot(columns, zscore(profile[COLUMNS]), color='C0', label=f'k-space: detrended {quantity} (dataset mean)')
        ax.plot(columns, zscore(ripples['brain, 12 casc'][COLUMNS]), color='C3', lw=1, alpha=0.8, label='learned mask ripple (12 casc, z-scored)')
        ax.set(title=f'{quantity}: r = {corr(profile, ripples["brain, 12 casc"]):+.2f} with the mask ripple', ylabel='z-score')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('kx column')
    for col, quantity in enumerate(mean_profiles):
        ax = fig.add_subplot(grid[3, col])
        ax.hist(scan_corr[quantity, 'brain, 12 casc'], bins=25, color='C0')
        ax.axvline(0, color='k', lw=0.5)
        ax.set(title=f'per-scan r: {quantity} vs mask ripple', xlabel='Pearson r', ylabel='scans')
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == '__main__':
    main()
