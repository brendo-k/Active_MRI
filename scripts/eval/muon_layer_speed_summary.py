"""Decides the frozen criteria of reports/muon_layer_speed/proposal.md from the eval, the run logs and the step benchmark.

    python scripts/eval/muon_layer_speed_summary.py reports/muon_layer_speed --queue-dir ~/data/seqmri_checkpoints/muon_layer_speed

Reads eval.json and eval.per_slice.npz (scripts/eval/varnet_train_speed_eval.py with --control ref, run <arm>_ema_s<seed>
scored as a run of its own) and bench.json from the report folder, and each run's TrainTime and ValTime from its log. Writes
summary.json and summary.md there. A criterion the frozen proposal states over two seeds stays undecided on one seed: the
verdict then reads 'provisional', and the seed counts as what it would need to be.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import varnet_train_speed_eval as evaluation

RUN_NAME = re.compile(r'(?P<arm>.+)_s(?P<seed>\d+)$')
EPOCH_LINE = re.compile(r'Epoch = \[ *\d+/ *\d+\].*TrainTime = ([\d.]+)s ValTime = ([\d.]+)s')
FASTER_MEAN_DB = -0.10
FASTER_WORST_SEED_DB = -0.20
FASTER_SSIM = -0.002
FASTER_WALL_CLOCK = 0.50
WORSE_MEAN_DB = -0.25
EMA_ADOPT_DB = 0.05
SEEDS_NEEDED = 2
REFERENCE = 'ref'


def wall_clock_hours(queue_dir, name):
    rows = EPOCH_LINE.findall((queue_dir / f'{name}.log').read_text(errors='replace'))
    return {'epochs': len(rows), 'train_h': sum(float(t) for t, _ in rows) / 3600, 'val_h': sum(float(v) for _, v in rows) / 3600}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report', type=Path)
    parser.add_argument('--queue-dir', type=Path, required=True)
    args = parser.parse_args()
    queue_dir = args.queue_dir.expanduser()
    result = json.loads((args.report / 'eval.json').read_text())
    slices = np.load(args.report / 'eval.per_slice.npz', allow_pickle=True)

    per_run = result['per_run']
    arms = {}
    for arm, seeds in result['paired'].items():
        if arm.endswith('_ema'):
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
                          'schedule_share': clock['epochs'] / reference_clock['epochs']}
            ema = f'{arm}_ema_s{seed}'
            if f'{ema}/psnr' in slices:
                rows[seed]['ema_minus_last'] = evaluation.paired(slices[f'{ema}/psnr'] - slices[f'{name}/psnr'], slices[f'{name}/volume'])
                rows[seed]['ema_minus_last_ssim'] = float((slices[f'{ema}/ssim'] - slices[f'{name}/ssim']).mean())
        deltas = [row['delta_psnr']['mean'] for row in rows.values()]
        ssims = [row['delta_ssim']['mean'] for row in rows.values()]
        gains = [row['ema_minus_last']['mean'] for row in rows.values() if 'ema_minus_last' in row]
        arms[arm] = {'seeds': rows, 'mean_delta_psnr': float(np.mean(deltas)),
                     'verdict': verdict(deltas, ssims, max(row['wall_clock_share'] for row in rows.values())),
                     'ema_adopt': None if not gains else (
                         (min(gains) >= EMA_ADOPT_DB) if len(gains) >= SEEDS_NEEDED else f'provisional {min(gains) >= EMA_ADOPT_DB}')}

    bench_path = args.report / 'bench.json'
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else []
    (args.report / 'summary.json').write_text(json.dumps({'arms': arms, 'bench': bench,
                                                          'reference': {k: v for k, v in per_run.items() if k.startswith(REFERENCE)}}, indent=1))

    lines = ['# muon_layer_speed summary', '', 'Paired against the same-seed 100-epoch reference on the full validation set, 2 mask draws.', '',
             '| arm | seed | PSNR | SSIM | delta PSNR [95% CI] | delta SSIM | brain / PD / PDFS delta | epochs | wall-clock h (train + val) | share | EMA - last dB | verdict |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, entry in arms.items():
        for seed, row in entry['seeds'].items():
            delta, conditions = row['delta_psnr'], row['per_condition']
            ema = row.get('ema_minus_last')
            lines.append(
                f"| {arm} | {seed} | {row['psnr']:.3f} | {row['ssim']:.4f} | {delta['mean']:+.3f} [{delta['ci'][0]:+.3f}, {delta['ci'][1]:+.3f}] | "
                f"{row['delta_ssim']['mean']:+.4f} | {' / '.join(f'{conditions[c]:+.2f}' for c in sorted(conditions))} | {row['epochs']} | "
                f"{row['wall_clock_h']:.2f} ({row['train_h']:.2f} + {row['val_h']:.2f}) | {row['wall_clock_share']:.0%} | "
                f"{'-' if ema is None else format(ema['mean'], '+.3f')} | {entry['verdict']} |")
    lines += ['', '| bench arm | step s | val pass s | steady peak GiB |', '|---|---|---|---|']
    for row in bench:
        if 'step_s' in row:
            lines.append(f"| {row['arm']} | {row['step_s']:.3f} | {row['val_pass_s']:.1f} | {row['steady_peak_GiB']:.1f} |")
    (args.report / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
