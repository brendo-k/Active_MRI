"""Encoding census of the local-v1 validation volumes: how much do object support, coil geometry and SENSE noise
amplification differ between scans, and does the best sampling lattice differ per patient? CPU only, no checkpoint.

Each volume is read as the trainer reads it (readout oversampling cropped, 320 x 320 k-space, brain row flip). The
object support is the trainer's apodized metric mask. Sensitivity maps are the Hann-windowed 24 x 24 centre divided by
its RSS. A "lattice" is one of the index-R sublattices of the k-space grid; it folds R image shifts onto each pixel, and
g^2 = [(S^H S)^-1]_pp [S^H S]_pp over the folded object pixels (no noise prewhitening, capped at 100).

    python scripts/diagnostics/encoding_census_r8.py support --out reports/adaptivity_levers_r8/support_census.json
    python scripts/diagnostics/encoding_census_r8.py gfactor --acceleration 8 --out reports/adaptivity_levers_r8/gfactor_census_r8.json
    python scripts/diagnostics/encoding_census_r8.py gfactor --acceleration 4 --out reports/adaptivity_levers_r8/gfactor_census_r4.json
    python scripts/diagnostics/encoding_census_r8.py roi --out reports/adaptivity_levers_r8/roi_gfactor_r8.json
    python scripts/diagnostics/encoding_census_r8.py summarize --dir reports/adaptivity_levers_r8
"""
import argparse
import collections
import csv
import functools
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch

import _path  # noqa: F401
from activemri.datasets.fastmri_data import crop_readout_oversampling, flip_rows_about_centre
from activemri.utils.metric_mask import metric_support_mask

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'configs/datasets/fastmri_local_v1/manifest.csv'
DATA_ROOTS = {'hpc': '/export/scratch/bkadota/fastmri_raw/{anatomy}', 'local_batch2': '/export/scratch/bkadota/fastmri_knee/batch2'}
GRID = 320
CALIBRATION_SIZE = 24
G2_CAP = 100.0
FEW_COILS = 8
ROI_RADIUS = 32
ROI_OFFSETS = {'centre': (0, 0), 'up': (-0.25, 0), 'down': (0.25, 0), 'left': (0, -0.25), 'right': (0, 0.25)}
DEGENERATE_LATTICES = ('1x8s0', '8x1s0')
SLICE_STRIDES = {'support': (0, 2), 'gfactor': (0, 3), 'roi': (1, 4)}
MIN_SUPPORT_PIXELS = {'support': 1, 'gfactor': 500, 'roi': 4000}
MIN_ROI_PIXELS = 1500


def val_rows():
    with open(MANIFEST) as handle:
        return [row for row in csv.DictReader(handle) if row['split'] == 'val']


def volume_path(row):
    return Path(DATA_ROOTS[row['source']].format(anatomy=row['anatomy'])) / row['hpc_relpath']


def centre_crop_or_pad(data, shape):
    out = data.new_zeros(*data.shape[:-2], *shape)
    source, target = [], []
    for size, wanted in zip(data.shape[-2:], shape):
        offset = (size - wanted) // 2
        source.append(slice(max(offset, 0), max(offset, 0) + min(size, wanted)))
        target.append(slice(max(-offset, 0), max(-offset, 0) + min(size, wanted)))
    out[..., target[0], target[1]] = data[..., source[0], source[1]]
    return out


def trainer_slices(row, stage):
    """Yields (slice_id, physical coil count, k-space [C, GRID, GRID], support [GRID, GRID] bool) for one volume."""
    first, stride = SLICE_STRIDES[stage]
    with h5py.File(volume_path(row), 'r') as data:
        for slice_id in range(first, data['kspace'].shape[0], stride):
            kspace = torch.from_numpy(data['kspace'][slice_id])
            coils = int(kspace.shape[0])
            kspace = centre_crop_or_pad(crop_readout_oversampling(kspace), (GRID, GRID))
            if row['anatomy'] == 'brain':
                kspace = flip_rows_about_centre(kspace)
            support = metric_support_mask(kspace[None], [row['anatomy']])[0, 0].numpy() > 0
            if support.sum() >= MIN_SUPPORT_PIXELS[stage]:
                yield slice_id, coils, kspace, support


@functools.lru_cache(maxsize=None)
def lattices(acceleration):
    """Every index-`acceleration` sublattice of the grid, as (name, image-domain alias shifts in pixels)."""
    found = []
    rows, cols = np.arange(GRID)[:, None], np.arange(GRID)[None, :]
    for row_step in (step for step in range(1, acceleration + 1) if acceleration % step == 0):
        col_step = acceleration // row_step
        for shear in range(col_step):
            mask = ((rows % row_step == 0) & ((cols - shear * (rows // row_step)) % col_step == 0)).astype(float)
            point_spread = np.abs(np.fft.fft2(mask))
            shifts = [tuple(int(v) for v in peak) for peak in np.argwhere(point_spread > 0.5 * point_spread.max())]
            assert len(shifts) == acceleration, (row_step, shear, len(shifts))
            found.append((f'{row_step}x{col_step}s{shear}', shifts))
    return found


def calibration_maps(kspace):
    window = torch.zeros(GRID, GRID)
    start = GRID // 2 - CALIBRATION_SIZE // 2
    taper = torch.hann_window(CALIBRATION_SIZE + 2, periodic=False)[1:-1]
    window[start:start + CALIBRATION_SIZE, start:start + CALIBRATION_SIZE] = taper[:, None] * taper[None, :]
    image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace * window, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
    return (image / image.abs().pow(2).sum(0, keepdim=True).sqrt().clamp_min(1e-12)).numpy()


def aliased_object_count(support, shifts):
    return sum(np.roll(support, shift, axis=(0, 1)).astype(float) for shift in shifts)


def gfactor_squared(maps, support, shifts):
    """(g^2, number of folded object pixels) at every object pixel, for one lattice."""
    folded = np.stack([np.roll(maps * support, (-dy, -dx), axis=(1, 2)) for dy, dx in shifts], axis=-1)[:, support]
    present = np.stack([np.roll(support, (-dy, -dx), axis=(0, 1)) for dy, dx in shifts], axis=-1)[support]
    gram = np.einsum('cpi,cpj->pij', folded.conj(), folded)
    diagonal = np.arange(len(shifts))
    gram[:, diagonal, diagonal] += ~present + 1e-6
    own = shifts.index((0, 0))
    g2 = (np.linalg.inv(gram)[:, own, own] * gram[:, own, own]).real
    return np.clip(g2, 1.0, G2_CAP), present.sum(-1)


def volume_header(row, coils):
    return {'file': row['file'], 'cls': row['cls'], 'protocol': row['protocol'], 'field': row['field'], 'model': row['model'], 'coils': coils}


def support_volume(row, acceleration):
    torch.set_num_threads(1)
    slices, coils = [], 0
    for slice_id, coils, kspace, support in trainer_slices(row, 'support'):
        occupied_rows, occupied_cols = np.where(support.any(1))[0], np.where(support.any(0))[0]
        overlap = {name: float((aliased_object_count(support, shifts) * support).sum() / support.sum())
                   for name, shifts in lattices(acceleration)}
        slices.append({'slice': slice_id, 'f': float(support.mean()),
                       'extent_rows': float((occupied_rows[-1] - occupied_rows[0] + 1) / GRID),
                       'extent_cols': float((occupied_cols[-1] - occupied_cols[0] + 1) / GRID),
                       'acquired_cols': int((kspace.abs().sum((0, 1)) > 0).sum()), 'overlap': overlap})
    return {**volume_header(row, coils), 'slices': slices}


def gfactor_volume(row, acceleration):
    torch.set_num_threads(1)
    pooled = {name: {'g2': [], 'folded': []} for name, _ in lattices(acceleration)}
    coils = 0
    for _, coils, kspace, support in trainer_slices(row, 'gfactor'):
        maps = calibration_maps(kspace)
        for name, shifts in lattices(acceleration):
            g2, folded = gfactor_squared(maps, support, shifts)
            pooled[name]['g2'].append(g2)
            pooled[name]['folded'].append(folded)
    summary = {}
    for name, values in pooled.items():
        g2, folded = np.concatenate(values['g2']), np.concatenate(values['folded'])
        summary[name] = {'mean_g2': float(g2.mean()), 'median_g': float(np.sqrt(np.median(g2))), 'p95_g': float(np.sqrt(np.percentile(g2, 95))),
                         'frac_g_gt_2': float((g2 > 4).mean()), 'frac_capped': float((g2 >= G2_CAP).mean()),
                         'frac_folded_gt_coils': float((folded > coils).mean()), 'frac_unfolded': float((folded == 1).mean())}
    return {**volume_header(row, coils), 'lattices': summary}


def roi_volume(row, acceleration):
    torch.set_num_threads(1)
    kept = [(name, shifts) for name, shifts in lattices(acceleration) if name not in DEGENERATE_LATTICES]
    regions = list(ROI_OFFSETS) + ['object']
    sums = {region: np.zeros(len(kept)) for region in regions}
    pixels = dict.fromkeys(regions, 0)
    grid_rows, grid_cols = np.mgrid[:GRID, :GRID]
    coils = 0
    for _, coils, kspace, support in trainer_slices(row, 'roi'):
        occupied_rows, occupied_cols = np.where(support.any(1))[0], np.where(support.any(0))[0]
        centre_row, centre_col = grid_rows[support].mean(), grid_cols[support].mean()
        height, width = occupied_rows[-1] - occupied_rows[0] + 1, occupied_cols[-1] - occupied_cols[0] + 1
        masks = {region: (((grid_rows - centre_row - dy * height) ** 2 + (grid_cols - centre_col - dx * width) ** 2) <= ROI_RADIUS ** 2) & support
                 for region, (dy, dx) in ROI_OFFSETS.items()}
        masks['object'] = support
        if min(mask.sum() for mask in masks.values()) < MIN_ROI_PIXELS:
            continue
        maps = calibration_maps(kspace)
        for index, (_, shifts) in enumerate(kept):
            g2_map = np.zeros((GRID, GRID))
            g2_map[support] = gfactor_squared(maps, support, shifts)[0]
            for region, mask in masks.items():
                sums[region][index] += g2_map[mask].sum()
        for region, mask in masks.items():
            pixels[region] += int(mask.sum())
    return {**volume_header(row, coils), 'lattices': [name for name, _ in kept],
            'mean_g2': {region: (sums[region] / max(pixels[region], 1)).tolist() for region in regions}, 'pixels': pixels}


def run_stage(worker, args):
    rows = val_rows()
    with Pool(args.workers) as pool:
        results = pool.map(functools.partial(worker, acceleration=args.acceleration), rows, chunksize=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results))
    print(f'{len(results)} volumes -> {args.out}')


def decibels(values):
    return 10 * np.log10(np.asarray(values, dtype=float))


def condition(volume):
    return volume['protocol'] if volume['cls'] == 'knee' else volume['cls']


def coil_group(volume):
    if volume['cls'] == 'knee':
        return f'knee {condition(volume)}'
    return f"{condition(volume)} {'few coils' if volume['coils'] <= FEW_COILS else 'many coils'}"


def grouped(volumes, key):
    groups = collections.defaultdict(list)
    for volume in volumes:
        groups[key(volume)].append(volume)
    return dict(sorted(groups.items()))


def r_squared(features, target):
    design = np.c_[np.ones(len(target)), features]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    return coefficients[1:].tolist(), float(1 - (residual ** 2).sum() / ((target - target.mean()) ** 2).sum())


def summarize_support(support):
    table = {}
    for name, volumes in grouped(support, condition).items():
        lattice_names = list(volumes[0]['slices'][0]['overlap'])
        fraction = np.array([np.mean([s['f'] for s in v['slices']]) for v in volumes])
        overlap = np.array([[np.average([s['overlap'][n] for s in v['slices']], weights=[s['f'] for s in v['slices']]) for n in lattice_names]
                            for v in volumes])
        best = overlap.mean(0).argmin()
        table[name] = {'volumes': len(volumes), 'f_mean': float(fraction.mean()), 'f_sd': float(fraction.std()), 'f_min': float(fraction.min()),
                       'f_max': float(fraction.max()), 'slice_f_sd_within_volume': float(np.mean([np.std([s['f'] for s in v['slices']]) for v in volumes])),
                       'slice_f_min': float(min(s['f'] for v in volumes for s in v['slices'])), 'slice_f_max': float(max(s['f'] for v in volumes for s in v['slices'])),
                       'coils': dict(sorted(collections.Counter(v['coils'] for v in volumes).items())),
                       'acquired_cols': dict(sorted(collections.Counter(v['slices'][0]['acquired_cols'] for v in volumes).items())),
                       'best_lattice': lattice_names[best], 'best_lattice_overlap': float(overlap.mean(0)[best]),
                       'overlap_over_bound': float(overlap.mean(0)[best] / (8 * fraction.mean())),
                       'own_minus_group_best_overlap_mean': float((overlap[:, best] - overlap.min(1)).mean()),
                       'own_minus_group_best_overlap_max': float((overlap[:, best] - overlap.min(1)).max())}
    return table


def summarize_gfactor(g8, g4, support):
    fraction = {v['file']: float(np.mean([s['f'] for s in v['slices']])) for v in support}
    g4_by_file = {v['file']: v for v in g4}

    def best_db(volume):
        return float(decibels(min(entry['mean_g2'] for entry in volume['lattices'].values())))

    table = {}
    for name, volumes in grouped(g8, coil_group).items():
        lattice_names = list(volumes[0]['lattices'])
        noise = np.array([[decibels(v['lattices'][n]['mean_g2']) for n in lattice_names] for v in volumes])
        best = noise.mean(0).argmin()
        gap = noise[:, best] - noise.min(1)
        table[name] = {'volumes': len(volumes), 'coils': dict(sorted(collections.Counter(v['coils'] for v in volumes).items())),
                       'f_mean': float(np.mean([fraction[v['file']] for v in volumes])), 'f_min': float(min(fraction[v['file']] for v in volumes)),
                       'f_max': float(max(fraction[v['file']] for v in volumes)),
                       'g2_db_r4_own_best': float(np.mean([best_db(g4_by_file[v['file']]) for v in volumes])),
                       'best_lattice': lattice_names[best], 'g2_db_r8': float(noise[:, best].mean()), 'g2_db_r8_min': float(noise[:, best].min()),
                       'g2_db_r8_max': float(noise[:, best].max()), 'g2_db_r8_volume_sd': float(noise[:, best].std()),
                       'own_minus_group_best_db_mean': float(gap.mean()), 'own_minus_group_best_db_max': float(gap.max()),
                       'frac_folded_gt_coils': float(np.mean([v['lattices'][lattice_names[best]]['frac_folded_gt_coils'] for v in volumes])),
                       'frac_g_gt_2': float(np.mean([v['lattices'][lattice_names[best]]['frac_g_gt_2'] for v in volumes]))}

    level = np.array([best_db(v) for v in g8])
    slope = np.array([10 * np.log10(2) + best_db(v) - best_db(g4_by_file[v['file']]) for v in g8])
    features = np.c_[np.log2([fraction[v['file']] for v in g8]), np.log2([v['coils'] for v in g8])]
    level_fit, level_r2 = r_squared(features, level)
    slope_fit, slope_r2 = r_squared(features, slope)
    allocation = slope / slope.mean()
    penalty = decibels(8) + level
    samples_at_p90 = 2.0 ** (-(np.percentile(penalty, 90) - penalty) / slope)
    budget = {'slope_mean': float(slope.mean()), 'slope_sd': float(slope.std()), 'slope_cv': float(slope.std() / slope.mean()),
              'slope_by_group': {name: float(np.mean([slope[i] for i, v in enumerate(g8) if coil_group(v) == name])) for name in table},
              'level_fit_log2f_log2coils': level_fit, 'level_fit_r2': level_r2, 'slope_fit_log2f_log2coils': slope_fit, 'slope_fit_r2': slope_r2,
              'level_min_db': float(level.min()), 'level_max_db': float(level.max()),
              'waterfilling_mean_gain_db': float(np.mean(slope * np.log2(allocation))),
              'waterfilling_acceleration_min': float(8 / allocation.max()), 'waterfilling_acceleration_max': float(8 / allocation.min()),
              'second_order_estimate_db': float(slope.mean() * (slope.std() / slope.mean()) ** 2 / (2 * np.log(2))),
              'samples_at_p90_penalty_relative_to_r8': float(samples_at_p90.mean())}
    return table, budget


def summarize_roi(roi):
    def anatomy_group(volume):
        if volume['cls'] == 'knee':
            return 'knee'
        return 'brain few coils' if volume['coils'] <= FEW_COILS else 'brain many coils'

    table = {}
    for name, volumes in grouped([v for v in roi if min(v['pixels'].values()) > 0], anatomy_group).items():
        whole = np.array([decibels(v['mean_g2']['object']) for v in volumes])
        whole_best = whole.mean(0).argmin()
        static_gain, patient_gain, zero_gain_share = [], [], []
        for region in ROI_OFFSETS:
            noise = np.array([decibels(v['mean_g2'][region]) for v in volumes])
            region_best = noise.mean(0).argmin()
            static_gain.append((noise[:, whole_best] - noise[:, region_best]).mean())
            patient_gain.append((noise[:, region_best] - noise.min(1)).mean())
            zero_gain_share.append(((noise[:, region_best] - noise.min(1)) < 1e-9).mean())
        table[name] = {'volumes': len(volumes), 'whole_object_patient_gain_db': float((whole[:, whole_best] - whole.min(1)).mean()),
                       'region_patient_gain_db': float(np.mean(patient_gain)), 'region_static_gain_db': float(np.mean(static_gain)),
                       'region_patient_gain_db_by_region': dict(zip(ROI_OFFSETS, map(float, patient_gain))),
                       'share_of_volumes_with_zero_patient_gain': float(np.mean(zero_gain_share))}
    return table


def markdown(summary):
    lines = ['# Encoding census summary (generated by scripts/diagnostics/encoding_census_r8.py summarize)', '',
             '## Support, by condition', '',
             '| Condition | Volumes | f mean (min-max) | 8f range | Slice sd of f within a volume | Best lattice | E[n] / 8f | Own-best minus group-best E[n], mean (max) |',
             '|---|---|---|---|---|---|---|---|']
    for name, row in summary['support'].items():
        lines.append(f"| {name} | {row['volumes']} | {row['f_mean']:.2f} ({row['f_min']:.2f}-{row['f_max']:.2f}) | {8 * row['f_min']:.1f}-{8 * row['f_max']:.1f} | "
                     f"{row['slice_f_sd_within_volume']:.2f} | {row['best_lattice']} | {row['overlap_over_bound']:.2f} | "
                     f"{row['own_minus_group_best_overlap_mean']:.3f} ({row['own_minus_group_best_overlap_max']:.3f}) |")
    lines += ['', '## Noise amplification (10 log10 of object-mean g^2), by coil group', '',
              '| Group | Volumes | Coils | f mean (min-max) | R=4, own-best lattice | R=8, group-best lattice (min-max) | Own-best minus group-best, mean (max) | Object pixels with more aliases than coils | Slope dB/octave |',
              '|---|---|---|---|---|---|---|---|---|']
    for name, row in summary['gfactor'].items():
        coils = ', '.join(f'{count}x{coil}' for coil, count in row['coils'].items())
        lines.append(f"| {name} | {row['volumes']} | {coils} | {row['f_mean']:.2f} ({row['f_min']:.2f}-{row['f_max']:.2f}) | {row['g2_db_r4_own_best']:.1f} dB | "
                     f"{row['g2_db_r8']:.1f} dB ({row['g2_db_r8_min']:.1f}-{row['g2_db_r8_max']:.1f}) | {row['own_minus_group_best_db_mean']:.3f} dB ({row['own_minus_group_best_db_max']:.2f}) | "
                     f"{100 * row['frac_folded_gt_coils']:.1f}% | {summary['budget']['slope_by_group'][name]:.1f} |")
    budget = summary['budget']
    lines += ['', '## Budget under the linear model', '', '| Quantity | Value |', '|---|---|',
              f"| Own-best noise level at R=8, all volumes | {budget['level_min_db']:.1f} to {budget['level_max_db']:.1f} dB |",
              f"| Level fit: a + b log2 f + c log2 coils | b {budget['level_fit_log2f_log2coils'][0]:.2f}, c {budget['level_fit_log2f_log2coils'][1]:.2f}, R^2 {budget['level_fit_r2']:.2f} |",
              f"| Slope, all volumes | {budget['slope_mean']:.2f} dB/octave, sd {budget['slope_sd']:.2f}, CV {budget['slope_cv']:.2f} |",
              f"| Slope fit on the same features | R^2 {budget['slope_fit_r2']:.2f} |",
              f"| Water-filling at a fixed cohort total | +{budget['waterfilling_mean_gain_db']:.2f} dB mean, R from {budget['waterfilling_acceleration_min']:.1f} to {budget['waterfilling_acceleration_max']:.1f} |",
              f"| Second-order estimate, slope x CV^2 / (2 ln 2) | +{budget['second_order_estimate_db']:.2f} dB |",
              f"| Samples to hold every volume at the 90th-percentile penalty | {budget['samples_at_p90_penalty_relative_to_r8']:.2f} x fixed R=8 |",
              '', '## Region-specific noise (32-pixel discs), R=8', '',
              '| Group | Volumes | Patient-specific gain, whole object | Patient-specific gain, region | Static region-specific lattice gain | Volumes with zero region gain |',
              '|---|---|---|---|---|---|']
    for name, row in summary['roi'].items():
        lines.append(f"| {name} | {row['volumes']} | {row['whole_object_patient_gain_db']:.2f} dB | {row['region_patient_gain_db']:.2f} dB | "
                     f"{row['region_static_gain_db']:.2f} dB | {100 * row['share_of_volumes_with_zero_patient_gain']:.0f}% |")
    return '\n'.join(lines) + '\n'


def summarize(args):
    folder = Path(args.dir)
    support = json.loads((folder / 'support_census.json').read_text())
    gfactor_table, budget = summarize_gfactor(json.loads((folder / 'gfactor_census_r8.json').read_text()),
                                              json.loads((folder / 'gfactor_census_r4.json').read_text()), support)
    summary = {'volumes': len(support), 'support': summarize_support(support), 'gfactor': gfactor_table, 'budget': budget,
               'roi': summarize_roi(json.loads((folder / 'roi_gfactor_r8.json').read_text()))}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (folder / 'summary.md').write_text(markdown(summary))
    print(markdown(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('support', 'gfactor', 'roi'):
        stage = stages.add_parser(name)
        stage.add_argument('--out', required=True)
        stage.add_argument('--acceleration', type=int, default=8)
        stage.add_argument('--workers', type=int, default=32)
    stages.add_parser('summarize').add_argument('--dir', required=True)
    args = parser.parse_args()
    if args.stage == 'summarize':
        return summarize(args)
    run_stage({'support': support_volume, 'gfactor': gfactor_volume, 'roi': roi_volume}[args.stage], args)


if __name__ == '__main__':
    main()
