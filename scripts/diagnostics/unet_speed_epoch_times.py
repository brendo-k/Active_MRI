"""Training throughput of the reports/unet_speed_varnet2_r8 runs, read from their training logs.

Per run: the median TrainTime and ValTime over epochs (seconds, as the trainer logs them; the median ignores the slower
first epoch, which warms the page cache), the number of epochs, and the wall-clock from the first to the last epoch line.
Per arm and cascade count: those medians averaged over seeds, and the train-time change against base at the same count.
Within a seed the four 2-cascade arms trained at the same time on four GPUs of one machine, so they shared its CPUs and
storage alike. `--extra` adds other runs (e.g. an earlier 12-cascade baseline), reported but not paired.

    python scripts/diagnostics/unet_speed_epoch_times.py \
        --extra ~/data/seqmri_checkpoints/lr_sweep/static_vd_lr0.001_step_50ep_seed42 \
        --out reports/unet_speed_varnet2_r8/epoch_times.json
"""
import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path.home() / 'data/seqmri_checkpoints/unet_speed/runs'
RUN_NAME = re.compile(r'varnet(?P<cascades>\d+)_r8_(?P<arm>.+)_seed(?P<seed>\d+)$')
EPOCH_LINE = re.compile(r'^\[(?P<stamp>[\d\- :,]+)\].*Epoch = \[ *(?P<epoch>\d+)/ *(?P<total>\d+)\].*'
                        r'TrainTime = (?P<train>[\d.]+)s ValTime = (?P<val>[\d.]+)s')


def epoch_times(run_dir):
    rows = [m.groupdict() for line in (run_dir / 'train_sequential.log').read_text().splitlines()
            if (m := EPOCH_LINE.match(line))]
    stamps = [datetime.strptime(r['stamp'], '%Y-%m-%d %H:%M:%S,%f') for r in rows]
    return {
        'epochs': len(rows),
        'num_epochs': int(rows[-1]['total']) if rows else None,
        'median_train_s': statistics.median(float(r['train']) for r in rows),
        'median_val_s': statistics.median(float(r['val']) for r in rows),
        'first_epoch_train_s': float(rows[0]['train']),
        'wall_clock_h': (stamps[-1] - stamps[0]).total_seconds() / 3600 if len(stamps) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=RUNS_DIR)
    parser.add_argument('--extra', type=Path, nargs='*', default=[])
    parser.add_argument('--out', type=Path, default=Path('reports/unet_speed_varnet2_r8/epoch_times.json'))
    args = parser.parse_args()

    runs = {}
    for run_dir in sorted(args.runs_dir.iterdir()):
        if run_dir.is_dir() and (m := RUN_NAME.match(run_dir.name)):
            runs[run_dir.name] = {**epoch_times(run_dir), 'arm': m['arm'], 'cascades': int(m['cascades']),
                                  'seed': int(m['seed'])}
    extra = {run_dir.name: epoch_times(run_dir) for run_dir in args.extra}

    print(f'{"run":36s} {"epochs":>7s} {"train s":>8s} {"val s":>7s} {"epoch1 s":>9s} {"wall h":>7s}')
    for name, r in {**runs, **extra}.items():
        wall = f'{r["wall_clock_h"]:7.2f}' if r['wall_clock_h'] is not None else '      -'
        print(f'{name:36s} {r["epochs"]:3d}/{r["num_epochs"]:<3d} {r["median_train_s"]:8.1f} {r["median_val_s"]:7.1f} '
              f'{r["first_epoch_train_s"]:9.1f} {wall}')

    arms = {}
    for r in runs.values():
        arms.setdefault((r['cascades'], r['arm']), []).append(r)
    summary = {}
    print('\nper arm, mean over seeds of the median epoch time')
    for (cascades, arm), rs in sorted(arms.items()):
        train = statistics.mean(r['median_train_s'] for r in rs)
        base = arms.get((cascades, 'base'))
        base_train = statistics.mean(r['median_train_s'] for r in base) if base else None
        summary[f'{cascades}/{arm}'] = {
            'cascades': cascades, 'arm': arm, 'seeds': sorted(r['seed'] for r in rs),
            'median_train_s': train, 'median_train_s_per_seed': [r['median_train_s'] for r in rs],
            'median_val_s': statistics.mean(r['median_val_s'] for r in rs),
            'train_change_vs_base': train / base_train - 1 if base_train else None,
        }
        change = f'{100 * summary[f"{cascades}/{arm}"]["train_change_vs_base"]:+5.1f}% vs base' if base_train else ''
        per_seed = ', '.join(f'{r["median_train_s"]:.1f}' for r in rs)
        print(f'  {cascades:2d} cascades {arm:10s} train {train:6.1f} s ({per_seed})  '
              f'val {summary[f"{cascades}/{arm}"]["median_val_s"]:5.1f} s  {change}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'runs': runs, 'extra': extra, 'arms': summary}, indent=2))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
