"""Provisional scaling curves from the trainer's brain val (150 slices, 10 volumes, pinned in every run).

Not the pre-registered score: proposal.md scores best_model.pt and last_model.pt on the full brain val set. These
curves only show the shape early. The trainer's PSNR also uses each batch's own data range.

Reads every <queue>/c<C>_v<V>_s<seed>/train_sequential.log (finished or still running) and writes, into --out:
  trainer_val_curves.png   val PSNR against optimiser step, one panel per data size, one line per cascade count
  train_loss_curves.png    train loss against step, same layout
  peak_vs_cascades.png     peak val PSNR against cascades, one line per data size (finished runs filled, running hollow)
  peak_vs_volumes.png      peak val PSNR against training volumes (log), one line per cascade count
  trainer_val_peaks.csv    one row per run

    python scripts/plots/brain_varnet_scaling_trainer_val.py
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

QUEUE = Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_scaling')
OUT = Path('reports/brain_varnet_scaling/figures/trainer_val')
STEPS_PER_EPOCH = {20: 19, 40: 38, 80: 76}
CASCADES = (2, 4, 6, 8, 12)
VOLUMES = (20, 40, 80)
CASCADE_COLOURS = dict(zip(CASCADES, ('#7fb6dc', '#4f97cc', '#2b78b8', '#145a9a', '#08306b')))
VOLUME_COLOURS = {20: '#e08214', 40: '#5e3c99', 80: '#1b7837'}
EPOCH = re.compile(r'Epoch = \[\s*(\d+)/\s*(\d+)\] TrainLoss = ([\d.e+-]+)')
BRAIN = re.compile(r'\[brain\][^\n]*?PSNR=([\d.]+) SSIM=([\d.]+)')
NAME = re.compile(r'c(\d+)_v(\d+)_s(\d+)$')


def parse(log):
    """Per-epoch train loss and per-eval (epoch, PSNR, SSIM); a [brain] line belongs to the next Epoch line."""
    losses, evals, pending = [], [], None
    for line in log.read_text().splitlines():
        if (m := BRAIN.search(line)):
            pending = (float(m[1]), float(m[2]))
        elif (m := EPOCH.search(line)):
            epoch, total = int(m[1]), int(m[2])
            losses.append((epoch, float(m[3])))
            if pending:
                evals.append((epoch, *pending))
                pending = None
    return losses, evals, total if losses else None


def load_runs(queue, seed):
    runs = {}
    for folder in sorted(queue.glob(f'c*_v*_s{seed}')):
        m = NAME.search(folder.name)
        log = folder / 'train_sequential.log'
        if not m or not log.exists():
            continue
        c, v = int(m[1]), int(m[2])
        losses, evals, total = parse(log)
        if not evals:
            continue
        peak = max(evals, key=lambda e: e[1])
        finished = losses[-1][0] == total or 'Early stop after epoch' in log.read_text()
        runs[c, v] = {'losses': losses, 'evals': evals, 'finished': finished,
                      'peak_psnr': peak[1], 'peak_ssim': peak[2], 'peak_step': peak[0] * STEPS_PER_EPOCH[v],
                      'last_psnr': evals[-1][1], 'last_step': evals[-1][0] * STEPS_PER_EPOCH[v]}
    return runs


def style(ax):
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)


def curves(runs, key, ylabel, path, seed):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, v in zip(axes, VOLUMES):
        for c in CASCADES:
            if (c, v) not in runs:
                continue
            r = runs[c, v]
            if key == 'psnr':
                xs, ys = [e[0] * STEPS_PER_EPOCH[v] for e in r['evals']], [e[1] for e in r['evals']]
            else:
                xs, ys = [e[0] * STEPS_PER_EPOCH[v] for e in r['losses']], [e[1] for e in r['losses']]
            ax.plot(xs, ys, color=CASCADE_COLOURS[c], linewidth=2, label=f'{c} cascades')
        ax.set_title(f'{v} training volumes', fontsize=11)
        ax.set_xlabel('optimiser step')
        ax.set_xlim(0, 7700)
        style(ax)
    axes[0].set_ylabel(ylabel)
    if key == 'psnr':
        top = max(r['peak_psnr'] for r in runs.values())
        axes[0].set_ylim(top - 3, top + 0.3)
    else:
        axes[0].set_yscale('log')
    handles = [plt.Line2D([], [], color=CASCADE_COLOURS[c], linewidth=2, label=f'{c} cascades') for c in CASCADES]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    source = 'trainer brain val, 150 slices (provisional)' if key == 'psnr' else 'train loss per epoch'
    fig.suptitle(f'seed {seed}; {source}; lines end where a run stopped early', y=-0.02, fontsize=9, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def peaks(runs, x_of, line_of, line_values, colours, xlabel, label, path, seed, logx=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for value in line_values:
        points = sorted((x_of(c, v), r) for (c, v), r in runs.items() if line_of(c, v) == value)
        if not points:
            continue
        ax.plot([p[0] for p in points], [p[1]['peak_psnr'] for p in points], color=colours[value], linewidth=2,
                label=label(value))
        for x, r in points:
            ax.plot(x, r['peak_psnr'], 'o', markersize=8, color=colours[value],
                    markerfacecolor=colours[value] if r['finished'] else 'white', markeredgewidth=2)
    if logx:
        ax.set_xscale('log')
        ax.set_xticks(VOLUMES, [str(v) for v in VOLUMES])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_xticks(CASCADES)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('peak trainer val PSNR (dB)')
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    running = '; hollow = still running (peak so far)' if not all(r['finished'] for r in runs.values()) else ''
    ax.set_title(f'seed {seed}; trainer brain val, 150 slices (provisional){running}', fontsize=9, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    runs = load_runs(args.queue, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    curves(runs, 'psnr', 'trainer val PSNR (dB)', args.out / 'trainer_val_curves.png', args.seed)
    curves(runs, 'loss', 'train loss (L1)', args.out / 'train_loss_curves.png', args.seed)
    peaks(runs, lambda c, v: c, lambda c, v: v, VOLUMES, VOLUME_COLOURS, 'VarNet cascades', lambda v: f'{v} volumes',
          args.out / 'peak_vs_cascades.png', args.seed)
    peaks(runs, lambda c, v: v, lambda c, v: c, CASCADES, CASCADE_COLOURS, 'training volumes',
          lambda c: f'{c} cascades', args.out / 'peak_vs_volumes.png', args.seed, logx=True)
    with open(args.out / 'trainer_val_peaks.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cascades', 'volumes', 'finished', 'peak_psnr', 'peak_ssim', 'peak_step', 'last_psnr', 'last_step',
                    'drop_to_last'])
        for (c, v), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            w.writerow([c, v, r['finished'], r['peak_psnr'], r['peak_ssim'], r['peak_step'], r['last_psnr'],
                        r['last_step'], round(r['last_psnr'] - r['peak_psnr'], 3)])
            print(f"c{c:<2d} v{v}: peak {r['peak_psnr']:.2f} @ step {r['peak_step']} (SSIM {r['peak_ssim']:.3f}), "
                  f"last {r['last_psnr']:.2f} @ {r['last_step']}{'' if r['finished'] else '  [running]'}"
                  f"{'  [early stop]' if r['finished'] and r['last_step'] < 7600 else ''}")


if __name__ == '__main__':
    main()
