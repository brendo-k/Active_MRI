"""Do column-to-column phase-coherence dips line up with the learned mask's column streaks?

Reuses scripts/check_kspace_column_ripple.py to measure per-column phase coherence over the brain T2
validation set, then (1) prints the mask-ripple peaks and coherence dips, (2) folds the detrended coherence
at a fixed period and ranks the boundary comb phase against all other phases for several column windows,
and (3) repeats the comparison per scan.

The boundary phase (24 mod 19 = 5) was read off the dataset-mean plot, so any window containing it is a
post-hoc choice; the other windows show where the effect does not hold.

    python scripts/coherence_comb_test.py
"""
import numpy as np
from scipy.signal import find_peaks

import check_kspace_column_ripple as ripple_tools
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from pathlib import Path

PERIOD = 19
BOUNDARY_PHASE = 24 % PERIOD
WINDOWS = [(20, 105), (8, 120), (5, 250), (130, 250)]
FIRST_BOUNDARY_COLUMN = 24
LAST_PER_SCAN_COLUMN = 100


def comb_mean(profile, phase, low, high):
    return profile[[c for c in range(low, high) if (c - phase) % PERIOD == 0]].mean()


def main():
    _, loader = build_fastmri_dataloaders(
        data_root=Path('/home/bkadota/data/fastmri'), anatomies=['brain'], image_shape=[256, 256],
        batch_size=8, num_coils=16, num_workers=8, contrast='T2')
    _, _, coherence, names = ripple_tools.measure(loader)
    detrended = ripple_tools.detrend(coherence.mean(0))
    mask_ripple = ripple_tools.mask_ripple(ripple_tools.MASK_RUNS['brain, 12 casc'])

    peaks, _ = find_peaks(mask_ripple[8:112], height=0.03)
    dips, _ = find_peaks(-detrended[8:112], height=1.5 * detrended[8:112].std())
    print('mask ripple peaks (columns):', (peaks + 8).tolist())
    print('coherence dips (columns):   ', (dips + 8).tolist())

    for low, high in WINDOWS:
        by_phase = np.array([comb_mean(detrended, p, low, high) for p in range(PERIOD)])
        others = np.delete(by_phase, BOUNDARY_PHASE)
        rank = int((by_phase < by_phase[BOUNDARY_PHASE]).sum()) + 1
        print(f'window {low}-{high}: boundary-phase coherence {by_phase[BOUNDARY_PHASE]:+.4f} '
              f'(rank {rank} lowest of {PERIOD}; other phases {others.mean():+.4f} +/- {others.std():.4f})')

    boundary_columns = list(range(FIRST_BOUNDARY_COLUMN, LAST_PER_SCAN_COLUMN + 1, PERIOD))
    low, high = WINDOWS[0]
    scans = {}
    for name, profile in zip(names, coherence):
        scans.setdefault(str(name), []).append(profile)
    z = []
    for profiles in scans.values():
        p = ripple_tools.detrend(np.mean(profiles, axis=0))[low:high]
        z.append((p[[c - low for c in boundary_columns]].mean() - p.mean()) / p.std())
    z = np.array(z)
    print(f'per scan, window {low}-{high}: boundary-column z median {np.median(z):+.2f}; '
          f'{np.mean(z < 0):.0%} of {len(z)} scans negative')


if __name__ == '__main__':
    main()
