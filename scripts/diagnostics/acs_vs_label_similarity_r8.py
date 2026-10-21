"""Do slices with near-identical calibration centres want the same oracle mask?

If an ACS -> mask map existed, two slices whose 10 x 10 calibration centres are almost the same would have similar
oracle masks. Over every pair of the 1200 training labels of oracle_labels_r8.py (same anatomy), this compares:

  ACS similarity     the sampler's own input, the SENSE-combined 10 x 10 centre: |<a, b>| / (|a| |b|) of the complex
                     values (phase-sensitive, equal to the correlation of the 10 x 10 low-resolution images), and the
                     correlation of the mean-removed log-magnitudes (the smooth predictor's feature)
  label similarity   corr of the two labels' deviations from the anatomy-mean label over the available locations,
                     per band: unsplit, smooth (Gaussian blur of `--band-sigma` px) and pixel (the deviation minus its
                     normalised blur over the available locations, as in oracle_mask_characteristics_r8.py)

grouped by slice distance within a volume and for other volumes, plus the other-volume pairs whose ACS are as
similar as adjacent slices', and each slice's nearest ACS neighbour in another volume (a 1-NN predictor, in-sample).

Per-volume masks (one mask for every slice of a volume, the SPACE granularity), per band:
  shared fraction        the mean label similarity over all pairs of labelled slices in a volume (all distances). It
                         is the share of a slice's deviation that one per-volume map can serve, so sqrt of it is the
                         predictability of a perfectly known per-volume map, and its rho^2 = shared fraction.
  leave-one-out          corr of each label's deviation with the mean deviation of the other labelled slices of its
                         volume (a per-volume map estimated from the volume's other slices, held out).

    python scripts/diagnostics/acs_vs_label_similarity_r8.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from label_curriculum_r8 import gaussian_blur
from label_group_structure_r8 import slice_sources
from label_predictor_gonogo_r8 import ANATOMIES, anatomy_reference, available_of, label_probability, load_shards
from oracle_labels_r8 import OUT_DIR

OUT = Path('analysis/label_gonogo/acs_vs_label_similarity.json')
DISTANCE_GROUPS = {'1': (1, 1), '2': (2, 2), '3-4': (3, 4), '5-8': (5, 8), '9+': (9, 10 ** 6)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--band-sigma', type=float, default=4.0)
    return parser.parse_args()


def unit_rows(matrix):
    matrix = matrix - matrix.mean(1, keepdim=True)
    return matrix / matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)


def acs_similarities(labels):
    """[N, N] complex-centre similarity and log-magnitude correlation of the 10 x 10 calibration centres."""
    kspace = torch.view_as_complex(labels['kspace_in'].float().contiguous())
    centre = labels['old_mask'].bool().reshape(len(kspace), *kspace.shape[1:])
    values = torch.stack([k[c] for k, c in zip(kspace, centre)])
    values = values / values.norm(dim=1, keepdim=True)
    complex_similarity = (values @ values.conj().T).abs()
    log_magnitude = unit_rows(torch.log(values.abs().clamp(min=1e-8)))
    return complex_similarity, log_magnitude @ log_magnitude.T


def band_of(deviation, available, band, sigma):
    if band == 'unsplit':
        return deviation
    if band == 'smooth':
        return gaussian_blur(deviation, sigma)
    smooth = gaussian_blur(deviation, sigma) / gaussian_blur(available, sigma).clamp(min=1e-6)
    return (deviation - smooth) * available


def deviation_rows(probability, reference, available, band, sigma):
    deviation = band_of((probability - reference) * available, available, band, sigma)
    keep = available[0].bool().flatten()
    assert bool((available.bool().flatten(1) == keep).all()), 'available locations differ between labels'
    return unit_rows(deviation.flatten(1)[:, keep])


def label_similarity(probability, reference, available, band, sigma):
    rows = deviation_rows(probability, reference, available, band, sigma)
    return rows @ rows.T


def volume_leave_one_out(rows, volume):
    """Mean corr of each row with the mean of the other rows of its volume (volumes with at least 2 rows)."""
    correlations, counts = [], []
    for name in np.unique(volume):
        members = np.where(volume == name)[0]
        if len(members) < 2:
            continue
        total = rows[members].sum(0)
        for member in members:
            others = unit_rows((total - rows[member])[None])[0]
            correlations.append(float(rows[member] @ others))
            counts.append(len(members))
    return {'slices': len(correlations), 'mean': float(np.mean(correlations)),
            'median_labelled_slices_per_volume': float(np.median(counts))}


def summarise(values, select):
    picked = values[select]
    return {'pairs': int(select.sum()), 'mean': float(picked.mean()), 'median': float(picked.median())}


def main():
    args = parse_args()
    labels = load_shards(args.label_dir, 'train')
    anatomies = labels['anatomy']
    volume, slice_index, _ = slice_sources(labels['indices'].numpy())
    probability = label_probability(labels)
    anatomy_mean = {a: probability[torch.as_tensor(anatomies == a)].mean(0) for a in ANATOMIES}
    reference = anatomy_reference(anatomy_mean, anatomies)
    available = available_of(labels)

    complex_similarity, log_magnitude = acs_similarities(labels)
    bands = {'unsplit': 'unsplit', f'smooth{args.band_sigma:g}': 'smooth', f'pixel{args.band_sigma:g}': 'pixel'}
    label = {key: label_similarity(probability, reference, available, band, args.band_sigma)
             for key, band in bands.items()}

    n = len(anatomies)
    same_anatomy = torch.as_tensor(anatomies[:, None] == anatomies[None, :])
    same_volume = torch.as_tensor(volume[:, None] == volume[None, :])
    distance = torch.as_tensor(np.abs(slice_index[:, None] - slice_index[None, :]))
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    groups = {f'same volume, {name} slices apart': upper & same_volume & (distance >= low) & (distance <= high)
              for name, (low, high) in DISTANCE_GROUPS.items()}
    groups['other volume, same anatomy'] = upper & same_anatomy & ~same_volume
    adjacent_floor = float(complex_similarity[groups['same volume, 1 slices apart']].median())
    groups['other volume, ACS as similar as the median adjacent pair'] = (
        groups['other volume, same anatomy'] & (complex_similarity >= adjacent_floor))

    table = {name: {'acs_complex': summarise(complex_similarity, select),
                    'acs_log_magnitude': summarise(log_magnitude, select),
                    **{f'label_{key}': summarise(values, select) for key, values in label.items()}}
             for name, select in groups.items()}

    other_volume = same_anatomy & ~same_volume
    masked = complex_similarity.masked_fill(~other_volume, -1)
    nearest = masked.argmax(1)
    rows = torch.arange(n)
    nearest_neighbour = {'acs_complex': float(complex_similarity[rows, nearest].mean()),
                         **{f'label_{key}': float(values[rows, nearest].mean()) for key, values in label.items()}}

    within_volume = upper & same_volume
    per_volume_mask = {}
    for key, values in label.items():
        shared = float(values[within_volume].mean())
        rows = deviation_rows(probability, reference, available, bands[key], args.band_sigma)
        per_volume_mask[key] = {'shared_fraction': shared, 'perfect_volume_map_rho': float(np.sqrt(max(shared, 0.0))),
                                'leave_one_out': volume_leave_one_out(rows, volume)}

    summary = {'slices': n, 'adjacent_median_acs_complex': adjacent_floor, 'groups': table,
               'nearest_acs_neighbour_other_volume': nearest_neighbour, 'per_volume_mask': per_volume_mask}
    print(json.dumps(summary, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps({'summary': summary, 'config': config}, indent=2) + '\n')


if __name__ == '__main__':
    main()
