"""Decides the frozen criteria of reports/muon_layer_speed/proposal.md from the eval, the run logs and the step benchmark.

    python scripts/eval/muon_layer_speed_summary.py reports/muon_layer_speed --queue-dir ~/data/seqmri_checkpoints/muon_layer_speed

Reads eval.json and eval.per_slice.npz (scripts/eval/varnet_train_speed_eval.py with --control ref, run <arm>_ema_s<seed>
scored as a run of its own), bench.json, curves.json and curves_reference.json from the report folder, and each run's
TrainTime and ValTime from its log. Writes summary.json and summary.md there. A criterion the frozen proposal states over
two seeds stays undecided on one seed: the verdict then reads 'provisional', and the seed counts as what it would need to be.

Beyond the frozen criteria it reports:
- steady wall-clock: the median epoch's train time times the epochs, plus validation, so a stretch shared with another job
  does not count;
- paired differences between arms, same seed, per slice: exploratory, since no criterion was frozen for them. The swapped
  runs give the 2x2 of 30-epoch against 100-epoch networks and masks;
- the reference epoch each arm matches (trainer val PSNR_Masked; the reference's curve is a centred 5-epoch median, so a
  one-epoch dip does not count), and from it the hours a constant-lr run needs to come within the proposal's bar.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

import varnet_train_speed_eval as evaluation

RUN_NAME = re.compile(r'(?P<arm>.+)_s(?P<seed>\d+)$')
EPOCH_LINE = re.compile(r'Epoch = \[ *\d+/ *\d+\].*TrainTime = ([\d.]+)s ValTime = ([\d.]+)s')
PSNR = 'val/PSNR_Masked'
FASTER_MEAN_DB = -0.10
FASTER_WORST_SEED_DB = -0.20
FASTER_SSIM = -0.002
FASTER_WALL_CLOCK = 0.50
WORSE_MEAN_DB = -0.25
EMA_ADOPT_DB = 0.05
SEEDS_NEEDED = 2
REFERENCE = 'ref'
REFERENCE_RUN = 'muon_layer_s42'
POST_FREEZE_ARMS = {'e30_step_c16'}
SWAPPED = '_maskof_'  # scripts/diagnostics/muon_layer_speed_swap_masks.py: scored runs without a training of their own
BETWEEN_ARMS = [('e30_step', 'e30_cos'), ('e30_cos_muon2e3', 'e30_cos'), ('e30_bs8', 'e30_step'), ('e30_step_c16', 'e30_step'),
                ('e30_step_maskof_ref', 'e30_step'), ('ref', 'ref_maskof_e30_step'),
                ('ref_maskof_e30_step', 'e30_step'), ('ref', 'e30_step_maskof_ref')]
BENCH_CONTROL = 'control'
SMOOTHING_RADIUS = 2
VAL_EVERY = 5


def wall_clock_hours(queue_dir, name):
    rows = EPOCH_LINE.findall((queue_dir / f'{name}.log').read_text(errors='replace'))
    train = [float(t) for t, _ in rows]
    return {'epochs': len(rows), 'train_h': sum(train) / 3600, 'val_h': sum(float(v) for _, v in rows) / 3600,
            'steady_train_h': statistics.median(train) * len(train) / 3600, 'median_train_s': statistics.median(train)}


def verdict(deltas, ssim_deltas, wall_clock_share):
    mean, worst = float(np.mean(deltas)), min(deltas)
    if mean < WORSE_MEAN_DB:
        outcome = 'worse'
    elif mean >= FASTER_MEAN_DB and worst >= FASTER_WORST_SEED_DB and float(np.mean(ssim_deltas)) >= FASTER_SSIM \
            and wall_clock_share <= FASTER_WALL_CLOCK:
        outcome = 'faster'
    else:
        outcome = 'inconclusive'
    return outcome if len(deltas) >= SEEDS_NEEDED else f'provisional {outcome} ({len(deltas)} of {SEEDS_NEEDED} seeds)'


def between_arms(slices, arm, other, seed):
    name, base = f'{arm}_s{seed}', f'{other}_s{seed}'
    if f'{name}/psnr' not in slices or f'{base}/psnr' not in slices:
        return None
    volumes = slices[f'{name}/volume']
    return {'arm': arm, 'other': other, 'seed': seed,
            'psnr': evaluation.paired(slices[f'{name}/psnr'] - slices[f'{base}/psnr'], volumes),
            'ssim': evaluation.paired(slices[f'{name}/ssim'] - slices[f'{base}/ssim'], volumes)}


def smoothed_curve(run):
    """Centred median of the reference's validation curve, as {epoch: dB}."""
    rows = [(row['epoch'], row[PSNR]) for row in run['curve'] if PSNR in row]
    values = [value for _, value in rows]
    return {epoch: float(np.median(values[max(0, i - SMOOTHING_RADIUS):i + SMOOTHING_RADIUS + 1])) for i, (epoch, _) in enumerate(rows)}


def first_epoch_reaching(curve, psnr):
    return next((epoch for epoch, value in sorted(curve.items()) if value >= psnr), None)


def trainer_finals(report):
    """Each arm's last trainer val PSNR_Masked, and the reference's smoothed curve, or empty if the curve files are absent."""
    arms_path, reference_path = report / 'curves.json', report / 'curves_reference.json'
    if not (arms_path.exists() and reference_path.exists()):
        return {}, {}
    finals = {name: [row[PSNR] for row in run['curve'] if PSNR in row][-1] for name, run in json.loads(arms_path.read_text())['runs'].items()
              if any(PSNR in row for row in run['curve'])}
    return finals, smoothed_curve(json.loads(reference_path.read_text())['runs'][REFERENCE_RUN])


def time_to_bar(reference_curve, seconds_per_epoch, val_seconds):
    """Hours for a constant-lr run at this epoch time to reach the reference's final minus the proposal's bar."""
    final = reference_curve[max(reference_curve)]
    epoch = first_epoch_reaching(reference_curve, final + FASTER_MEAN_DB)
    return {'bar_db': final + FASTER_MEAN_DB, 'reference_epoch': epoch,
            'hours': (epoch * seconds_per_epoch + epoch // VAL_EVERY * val_seconds) / 3600}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report', type=Path)
    parser.add_argument('--queue-dir', type=Path, required=True)
    args = parser.parse_args()
    queue_dir = args.queue_dir.expanduser()
    result = json.loads((args.report / 'eval.json').read_text())
    slices = np.load(args.report / 'eval.per_slice.npz', allow_pickle=True)
    finals, reference_curve = trainer_finals(args.report)

    per_run = result['per_run']
    arms = {}
    for arm, seeds in result['paired'].items():
        if arm.endswith('_ema') or SWAPPED in arm:
            continue
        rows = {}
        for seed, entry in seeds.items():
            name, reference = f'{arm}_s{seed}', f'{REFERENCE}_s{seed}'
            clock, reference_clock = wall_clock_hours(queue_dir, name), wall_clock_hours(queue_dir, reference)
            total, reference_total = clock['train_h'] + clock['val_h'], reference_clock['train_h'] + reference_clock['val_h']
            rows[seed] = {'psnr': per_run[name]['all']['psnr_masked'], 'ssim': per_run[name]['all']['ssim_masked'],
                          'delta_psnr': entry['all']['psnr'], 'delta_ssim': entry['all']['ssim'],
                          'per_condition': {c: entry[c]['psnr']['mean'] for c in entry if c != 'all'},
                          **clock, 'wall_clock_h': total, 'wall_clock_share': total / reference_total,
                          'steady_wall_clock_h': clock['steady_train_h'] + clock['val_h'],
                          'steady_wall_clock_share': (clock['steady_train_h'] + clock['val_h']) / reference_total,
                          'schedule_share': clock['epochs'] / reference_clock['epochs']}
            if name in finals:
                rows[seed]['trainer_final_psnr'] = finals[name]
                rows[seed]['reference_epoch_matched'] = first_epoch_reaching(reference_curve, finals[name])
            ema = f'{arm}_ema_s{seed}'
            if f'{ema}/psnr' in slices:
                rows[seed]['ema_minus_last'] = evaluation.paired(slices[f'{ema}/psnr'] - slices[f'{name}/psnr'], slices[f'{name}/volume'])
                rows[seed]['ema_minus_last_ssim'] = float((slices[f'{ema}/ssim'] - slices[f'{name}/ssim']).mean())
        deltas = [row['delta_psnr']['mean'] for row in rows.values()]
        ssims = [row['delta_ssim']['mean'] for row in rows.values()]
        gains = [row['ema_minus_last']['mean'] for row in rows.values() if 'ema_minus_last' in row]
        decided = 'post-freeze, no frozen criterion' if arm in POST_FREEZE_ARMS else verdict(
            deltas, ssims, max(row['wall_clock_share'] for row in rows.values()))
        arms[arm] = {'seeds': rows, 'mean_delta_psnr': float(np.mean(deltas)), 'verdict': decided,
                     'ema_adopt': None if not gains else (
                         (min(gains) >= EMA_ADOPT_DB) if len(gains) >= SEEDS_NEEDED else f'provisional {min(gains) >= EMA_ADOPT_DB}')}

    comparisons = [row for arm, other in BETWEEN_ARMS for seed in sorted({s for entry in arms.values() for s in entry['seeds']})
                   if (row := between_arms(slices, arm, other, seed)) is not None]

    bench_path = args.report / 'bench.json'
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else []
    bench_control = next((row for row in bench if row.get('arm') == BENCH_CONTROL and 'step_s' in row), None)
    for row in bench:
        if bench_control and 'step_s' in row:
            row['step_change'] = row['step_s'] / bench_control['step_s'] - 1

    fast_run = arms.get('e30_step', {}).get('seeds', {}).get('42')
    to_bar = time_to_bar(reference_curve, fast_run['median_train_s'], fast_run['val_h'] * 3600 / max(1, fast_run['epochs'] // VAL_EVERY)) \
        if reference_curve and fast_run else None

    (args.report / 'summary.json').write_text(json.dumps({'arms': arms, 'between_arms': comparisons, 'bench': bench, 'time_to_bar': to_bar,
                                                          'reference': {k: v for k, v in per_run.items() if k.startswith(REFERENCE)}}, indent=1))

    lines = ['# muon_layer_speed summary', '', 'Paired against the same-seed 100-epoch reference on the full validation set, 2 mask draws.', '',
             '| arm | seed | PSNR | SSIM | delta PSNR [95% CI] | delta SSIM | brain / PD / PDFS delta | epochs | wall-clock h (train + val) | share | steady h | EMA - last dB | matches reference epoch | verdict |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, entry in arms.items():
        for seed, row in entry['seeds'].items():
            delta, conditions = row['delta_psnr'], row['per_condition']
            ema = row.get('ema_minus_last')
            lines.append(
                f"| {arm} | {seed} | {row['psnr']:.3f} | {row['ssim']:.4f} | {delta['mean']:+.3f} [{delta['ci'][0]:+.3f}, {delta['ci'][1]:+.3f}] | "
                f"{row['delta_ssim']['mean']:+.4f} | {' / '.join(f'{conditions[c]:+.2f}' for c in sorted(conditions))} | {row['epochs']} | "
                f"{row['wall_clock_h']:.2f} ({row['train_h']:.2f} + {row['val_h']:.2f}) | {row['wall_clock_share']:.0%} | {row['steady_wall_clock_h']:.2f} | "
                f"{'-' if ema is None else format(ema['mean'], '+.3f')} | {row.get('reference_epoch_matched', '-')} | {entry['verdict']} |")
    lines += ['', 'Between arms, same seed, per slice (exploratory):', '', '| arm minus arm | seed | delta PSNR [95% CI] | delta SSIM |', '|---|---|---|---|']
    for row in comparisons:
        psnr, ssim = row['psnr'], row['ssim']
        lines.append(f"| {row['arm']} - {row['other']} | {row['seed']} | {psnr['mean']:+.3f} [{psnr['ci'][0]:+.3f}, {psnr['ci'][1]:+.3f}] | {ssim['mean']:+.4f} |")
    if to_bar:
        lines += ['', f"Constant-lr estimate: the reference's smoothed curve reaches {to_bar['bar_db']:.2f} dB at epoch {to_bar['reference_epoch']}, "
                      f"{to_bar['hours']:.1f} h at e30_step's epoch time."]
    lines += ['', '| bench arm | step s | change | val pass s | steady peak GiB |', '|---|---|---|---|---|']
    for row in bench:
        if 'step_s' in row:
            lines.append(f"| {row['arm']} | {row['step_s']:.3f} | {row.get('step_change', 0):+.1%} | {row['val_pass_s']:.1f} | {row['steady_peak_GiB']:.1f} |")
    (args.report / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
