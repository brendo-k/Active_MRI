"""Per-epoch validation curves and epoch times of the varnet_train_speed runs (reports/varnet_train_speed/).

Reads wandb (project SeqMRI, runs named vts_<arm>_s<seed>) for the validation metrics and each run's log for
TrainTime / ValTime, and writes one JSON with every run's curves plus a per-arm summary against the control:
    python scripts/eval/varnet_train_speed_curves.py --out reports/varnet_train_speed/curves.json
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import wandb

QUEUE_DIR = Path('~/data/seqmri_checkpoints/varnet_train_speed').expanduser()
KEYS = ('epoch', 'val/PSNR_Masked', 'val/SSIM_Masked', 'val/loss', 'train/loss',
        'val/brain/PSNR', 'val/knee/PSNR', 'val/brain/SSIM', 'val/knee/SSIM')
EPOCH_LINE = re.compile(r'Epoch = \[ *(\d+)/ *(\d+)\] TrainLoss = (\S+) ValLoss = (\S+) .*TrainTime = ([\d.]+)s ValTime = ([\d.]+)s')
RUN_NAME = re.compile(r'vts_(?P<arm>.+)_s(?P<seed>\d+)$')
CONTROL = 'control'


def log_epochs(name):
    log = QUEUE_DIR / f'{name}.log'
    if not log.exists():
        return {}
    rows = {}
    for line in log.read_text().splitlines():
        match = EPOCH_LINE.search(line)
        if match:
            epoch, total, train_loss, val_loss, train_time, val_time = match.groups()
            rows[int(epoch)] = {'train_time': float(train_time), 'val_time': float(val_time), 'num_epochs': int(total),
                                'train_loss_log': float(train_loss), 'val_loss_log': float(val_loss),
                                'timestamp': line[1:24]}
    return rows


def run_curves(api, path):
    runs = {}
    for run in api.runs(path, filters={'display_name': {'$regex': '^vts_'}}):
        match = RUN_NAME.match(run.name)
        if not match:
            continue
        history = defaultdict(dict)
        for key in KEYS[1:]:
            for row in run.scan_history(keys=['epoch', key]):
                if row.get('epoch') is not None and row.get(key) is not None:
                    history[int(row['epoch']) + 1][key] = row[key]
        name = run.name.removeprefix('vts_')
        times = log_epochs(name)
        epochs = sorted(set(history) | set(times))
        entry = {'arm': match['arm'], 'seed': int(match['seed']), 'wandb_id': run.id, 'state': run.state,
                 'created_at': run.created_at, 'curve': [{'epoch': e, **history.get(e, {}), **times.get(e, {})} for e in epochs]}
        if name not in runs or _rank(entry) > _rank(runs[name]):
            runs[name] = entry  # a crashed attempt logs under the same name as its rerun
    return runs


def _rank(entry):
    """Which of several wandb runs with one name to keep: finished first, then the most epochs, then the newest."""
    return (entry['state'] == 'finished', len(entry['curve']), entry['created_at'])


def summarise(runs):
    by_arm = defaultdict(dict)
    for name, run in runs.items():
        by_arm[run['arm']][run['seed']] = run
    control = by_arm.get(CONTROL, {})
    control_final = [r['curve'][-1].get('val/PSNR_Masked') for r in control.values()
                     if r['curve'] and r['curve'][-1]['epoch'] == r['curve'][-1].get('num_epochs')]
    target = statistics.mean(control_final) if control_final else None
    summary = {'control_final_psnr_masked_mean': target, 'control_seeds_final': len(control_final), 'arms': {}}
    for arm, seeds in sorted(by_arm.items()):
        rows = {}
        for seed, run in sorted(seeds.items()):
            curve = run['curve']
            finished = bool(curve) and curve[-1]['epoch'] == curve[-1].get('num_epochs')
            train_times = [c['train_time'] for c in curve if 'train_time' in c and c['epoch'] > 1]
            epoch_wall = [c['train_time'] + c['val_time'] for c in curve if 'train_time' in c and c['epoch'] > 1]
            reach = next((c['epoch'] for c in curve if target is not None and c.get('val/PSNR_Masked', -1) >= target), None)
            pair = control.get(seed)
            paired = {}
            if pair:
                control_by_epoch = {c['epoch']: c for c in pair['curve']}
                for c in curve:
                    other = control_by_epoch.get(c['epoch'])
                    if other and 'val/PSNR_Masked' in c and 'val/PSNR_Masked' in other:
                        paired[c['epoch']] = round(c['val/PSNR_Masked'] - other['val/PSNR_Masked'], 4)
            rows[seed] = {
                'finished': finished, 'epochs_done': curve[-1]['epoch'] if curve else 0,
                'final': {k: curve[-1].get(k) for k in KEYS[1:]} if curve else {},
                'best_psnr_masked': max((c.get('val/PSNR_Masked', float('-inf')) for c in curve), default=None),
                'median_train_time_s': statistics.median(train_times) if train_times else None,
                'median_epoch_time_s': statistics.median(epoch_wall) if epoch_wall else None,
                'epoch_reaching_control_final': reach,
                'psnr_masked_minus_control_same_seed': paired,
            }
        summary['arms'][arm] = rows
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='reports/varnet_train_speed/curves.json')
    parser.add_argument('--project', default='SeqMRI')
    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    runs = run_curves(api, f'{api.default_entity}/{args.project}')
    out = {'runs': runs, 'summary': summarise(runs)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for arm, seeds in out['summary']['arms'].items():
        for seed, row in seeds.items():
            final = row['final']
            print(f"{arm:>16s} s{seed} ep{row['epochs_done']:>3d} PSNR_M {final.get('val/PSNR_Masked') or float('nan'):.3f} "
                  f"SSIM_M {final.get('val/SSIM_Masked') or float('nan'):.4f} train {row['median_train_time_s'] or float('nan'):.1f}s "
                  f"epoch {row['median_epoch_time_s'] or float('nan'):.1f}s reach {row['epoch_reaching_control_final']}")
    print('control final mean', out['summary']['control_final_psnr_masked_mean'])


if __name__ == '__main__':
    main()
