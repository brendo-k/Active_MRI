"""Summarise per-epoch validation metrics from train_sequential.log files.

For each run prints the epoch count, the best epoch (lowest ValLoss, which is what
best_model.pt tracks), the best brain-PSNR epoch, the last epoch, and optionally the full
per-epoch table. Metrics are the trainer's background-masked val metrics.

    python scripts/summarize_train_logs.py checkpoints/run_a checkpoints/run_b [--table] [--json out.json]
"""
import argparse
import json
import re
import statistics
from pathlib import Path

ANATOMY_LINE = re.compile(
    r"\[(?P<anat>\w+)\] MSE=\S+ NMSE=(?P<nmse>\S+) PSNR=(?P<psnr>\S+) SSIM=(?P<ssim>\S+)")
EPOCH_LINE = re.compile(
    r"Epoch = \[\s*(?P<epoch>\d+)/\s*(?P<total>\d+)\] TrainLoss = (?P<train>\S+) ValLoss = (?P<val>\S+)")


def parse_log(path):
    epochs, pending = [], None
    for line in path.read_text().splitlines():
        if '[brain] MSE=' in line:
            pending = {m['anat']: {k: float(m[k]) for k in ('nmse', 'psnr', 'ssim')}
                       for m in ANATOMY_LINE.finditer(line)}
            continue
        m = EPOCH_LINE.search(line)
        if m and pending is not None:
            epochs.append({'epoch': int(m['epoch']), 'total': int(m['total']),
                           'train_loss': float(m['train']), 'val_loss': float(m['val']), **pending})
            pending = None
    return epochs


def fmt(e):
    return (f"ep {e['epoch']:3d}  brain {e['brain']['psnr']:.2f} dB / {e['brain']['ssim']:.4f}  "
            f"knee {e['knee']['psnr']:.2f} dB / {e['knee']['ssim']:.4f}  val_loss {e['val_loss']:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--json', type=Path)
    parser.add_argument('--at-epoch', type=int, help='also report this epoch (matched-length comparison)')
    parser.add_argument('--window', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='brain PSNR mean / sd / min / max over these epochs, inclusive')
    args = parser.parse_args()

    summary = {}
    for run in args.runs:
        epochs = parse_log(run / 'train_sequential.log')
        if not epochs:
            print(f"{run}: no epochs logged")
            continue
        best = min(epochs, key=lambda e: e['val_loss'])
        best_brain = max(epochs, key=lambda e: e['brain']['psnr'])
        last = epochs[-1]
        summary[str(run)] = {'epochs_done': last['epoch'], 'epochs_total': last['total'],
                             'best_val_loss': best, 'best_brain_psnr': best_brain, 'last': last}
        print(f"\n{run}  ({last['epoch']}/{last['total']} epochs)")
        print(f"  best val_loss : {fmt(best)}")
        print(f"  best brain    : {fmt(best_brain)}")
        print(f"  last          : {fmt(last)}")
        if args.at_epoch is not None:
            matched = next((e for e in epochs if e['epoch'] == args.at_epoch), None)
            summary[str(run)]['at_epoch'] = matched
            if matched:
                print(f"  at epoch      : {fmt(matched)}")
        first, last_epoch = args.window or (0, -1)
        psnr = [e['brain']['psnr'] for e in epochs if first <= e['epoch'] <= last_epoch]
        if len(psnr) >= 2:
            stats = {'epochs': args.window, 'mean': statistics.mean(psnr), 'sd': statistics.stdev(psnr),
                     'min': min(psnr), 'max': max(psnr)}
            summary[str(run)]['brain_psnr_window'] = stats
            print(f"  brain PSNR ep {first}-{last_epoch}: {stats['mean']:.2f} +- {stats['sd']:.2f} "
                  f"({stats['min']:.2f}-{stats['max']:.2f})")
        if args.table:
            for e in epochs:
                print('   ', fmt(e))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
