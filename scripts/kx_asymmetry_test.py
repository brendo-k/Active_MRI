"""Is the left/right (kx) asymmetry of the learned R=8 masks real, or pixel noise?

For each run in plot_reinforce_patterns.RUNS: the share of the policy's inclusion mass in the
kx < 0 half, overall and per ring, against a null that shuffles the run's own inclusion values
within each ring (keeps the radial profile, destroys any direction preference). The ky < 0
(top/bottom) share is reported the same way as a control axis.

z = (observed share - null mean) / null sd, over NULL_SHUFFLES shuffles.

    python scripts/kx_asymmetry_test.py
"""
import json
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from compare_loupe_reinforce_r8 import RADIAL_RINGS, radius_grid
from plot_reinforce_patterns import OUT_DIR, RUNS, run_pattern

NULL_SHUFFLES = 500
FINE_RINGS = 64


def half_shares(inclusion, region):
    h, w = inclusion.shape
    left = np.zeros_like(region); left[:, :w // 2] = True
    top = np.zeros_like(region); top[:h // 2, :] = True
    total = inclusion[region].sum()
    return inclusion[region & left].sum() / total, inclusion[region & top].sum() / total


def shuffled_within_rings(inclusion, ring_index, rng):
    out = inclusion.copy()
    for ring in np.unique(ring_index):
        cells = np.flatnonzero(ring_index == ring)
        out.flat[cells] = inclusion.flat[rng.permutation(cells)]
    return out


def main():
    rng = np.random.default_rng(0)
    results = {}
    for label, path, kind in RUNS:
        inclusion = run_pattern(path, kind)['inclusion']
        h, w = inclusion.shape
        radius = radius_grid(h, w).numpy()
        ring_index = np.minimum((radius / radius.max() * FINE_RINGS).astype(int), FINE_RINGS - 1)
        regions = {'all': radius >= 0.05}
        regions.update({f'{lo:.2f}-{hi:.2f}': (radius >= max(lo, 0.05)) & (radius < hi)
                        for lo, hi in zip(RADIAL_RINGS[:-1], RADIAL_RINGS[1:]) if hi > 0.05})
        null = [shuffled_within_rings(inclusion, ring_index, rng) for _ in range(NULL_SHUFFLES)]
        rows = {}
        for name, region in regions.items():
            kx, ky = half_shares(inclusion, region)
            null_kx, null_ky = np.array([half_shares(n, region) for n in null]).T
            rows[name] = {'mass_share': float(inclusion[region].sum() / inclusion[radius >= 0.05].sum()),
                          'kx_neg_share': float(kx), 'kx_z': float((kx - null_kx.mean()) / null_kx.std()),
                          'ky_neg_share': float(ky), 'ky_z': float((ky - null_ky.mean()) / null_ky.std()),
                          'null_sd': float(null_kx.std())}
        results[label] = rows
        print(f'\n{label}')
        print(f"  {'ring':10s} {'mass':>6s} {'kx<0':>7s} {'z':>7s}   {'ky<0':>7s} {'z':>7s}  (null sd)")
        for name, r in rows.items():
            print(f"  {name:10s} {r['mass_share']:6.1%} {r['kx_neg_share']:7.1%} {r['kx_z']:7.1f}   "
                  f"{r['ky_neg_share']:7.1%} {r['ky_z']:7.1f}  ({r['null_sd']:.4f})")
    (OUT_DIR / 'kx_asymmetry.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
