"""How one-sided is a learned LOUPE map? The layout readout of loupe_loss_partial_fourier (proposal: the share of the map
right of the DC column at 32 to 98 columns from DC, rows 33-287).

The map is p(k) = sigmoid(slope * logits), tied to its point reflection when the run used `loupe_point_symmetric`, before
the sampler's budget rescaling. Every statistic is taken inside the box |column - 160| <= 98, |row - 160| <= 127 that
every FLAIR figure slice acquires, outside the 10 x 10 calibration block. The share right of DC is 0.5 for a symmetric
map; the conjugate overlap sum p(k) p(-k) / sum p(k)^2 is 1 for a point-symmetric map and 0 for complementary halves.

    python scripts/diagnostics/loupe_map_asymmetry.py <run dir> [<run dir> ...] [--out results.json]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CENTRE = 160
CALIBRATION_HALF_WIDTH = 5
COLUMN_HALF_WIDTH = 98
ROW_HALF_WIDTH = 127
BANDS = ((6, 16), (16, 32), (32, 64), (64, 99))
PRIMARY_BAND = (32, 99)


def point_reflected(array):
    return np.roll(np.flip(array, axis=(0, 1)), shift=(1, 1), axis=(0, 1))


def learned_map(run):
    checkpoint = torch.load(Path(run) / 'last_model.pt', map_location='cpu', weights_only=False)
    options = checkpoint['options']
    logits = checkpoint['model']['sampler.mask_net.mask'][0, :, :, 0].float().numpy()
    if getattr(options, 'loupe_point_symmetric', False):
        logits = (logits + point_reflected(logits)) / 2
    return 1 / (1 + np.exp(-float(getattr(options, 'slope', 10.0)) * logits))


def right_share(p, distance, band):
    low, high = band
    inside = (np.abs(distance) >= low) & (np.abs(distance) < high)
    return float(p[inside & (distance > 0)].sum() / p[inside].sum())


def asymmetry(p):
    rows, columns = np.indices(p.shape)
    distance = columns - CENTRE
    box = (np.abs(distance) <= COLUMN_HALF_WIDTH) & (np.abs(rows - CENTRE) <= ROW_HALF_WIDTH)
    outside_calibration = (np.abs(rows - CENTRE) > CALIBRATION_HALF_WIDTH) | (np.abs(distance) > CALIBRATION_HALF_WIDTH)
    p = np.where(box & outside_calibration, p, 0.0)
    return {
        'right_share_primary': right_share(p, distance, PRIMARY_BAND),
        'right_share_overall': float(p[distance > 0].sum() / p[distance != 0].sum()),
        'right_share_by_band': {f'{low}-{high - 1}': right_share(p, distance, (low, high)) for low, high in BANDS},
        'mass_share_by_band': {f'{low}-{high - 1}': float(p[(np.abs(distance) >= low) & (np.abs(distance) < high)].sum()
                                                          / p.sum()) for low, high in BANDS},
        'conjugate_overlap': float((p * point_reflected(p)).sum() / (p * p).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories holding last_model.pt')
    parser.add_argument('--out', help='results .json')
    args = parser.parse_args()
    results = {}
    for run in args.runs:
        results[Path(run).name] = stats = asymmetry(learned_map(run))
        bands = ', '.join(f'{band} {share:.3f}' for band, share in stats['right_share_by_band'].items())
        print(f"{Path(run).name}: right of DC at 32-98 columns {stats['right_share_primary']:.3f} (overall "
              f"{stats['right_share_overall']:.3f}; by band {bands}); conjugate overlap {stats['conjugate_overlap']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
