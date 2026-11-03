"""Validation curves of the varnet_train_speed arms (reports/varnet_train_speed/curves.json, written by
scripts/eval/varnet_train_speed_curves.py): val PSNR_Masked and SSIM_Masked against epoch and against cumulative
wall-clock (train + validation time from the run logs), seed mean with the seed range shaded.

    python scripts/plots/varnet_train_speed_curves.py <curves.json> <out.png> [arm,arm,...] [data description for the title]
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def arm_curves(runs, key, x_key):
    by_arm = defaultdict(list)
    for run in runs.values():
        points = [(c[x_key], c[key]) for c in run['curve'] if key in c and x_key in c]
        if points:
            by_arm[run['arm']].append(points)
    return by_arm


def with_wall_clock(runs):
    for run in runs.values():
        elapsed = 0.0
        for c in run['curve']:
            if 'train_time' in c:
                elapsed += c['train_time'] + c['val_time']
                c['hours'] = elapsed / 3600
    return runs


def main():
    curves_path, out_path = sys.argv[1], sys.argv[2]
    arms = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    data = sys.argv[4] if len(sys.argv) > 4 else '20% of local v1 FLAIR+knee'
    runs = with_wall_clock(json.load(open(curves_path))['runs'])
    if arms:
        runs = {k: v for k, v in runs.items() if v['arm'] in arms}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for col, x_key, x_label in ((0, 'epoch', 'epoch'), (1, 'hours', 'wall-clock hours (train + val, from the run logs)')):
        for row, key, y_label in ((0, 'val/PSNR_Masked', 'val PSNR_Masked (dB)'), (1, 'val/SSIM_Masked', 'val SSIM_Masked')):
            ax = axes[row, col]
            for i, (arm, seeds) in enumerate(sorted(arm_curves(runs, key, x_key).items())):
                length = min(len(s) for s in seeds)
                x = np.mean([[p[0] for p in s[:length]] for s in seeds], axis=0)
                y = np.array([[p[1] for p in s[:length]] for s in seeds])
                color = colors[i % len(colors)]
                ax.plot(x, y.mean(0), color=color, label=f'{arm} (n={len(seeds)})', lw=1.6)
                if len(seeds) > 1:
                    ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=0.15, lw=0)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.3)
            if key == 'val/PSNR_Masked':
                top = max(max(p[1] for p in s) for seeds in arm_curves(runs, key, x_key).values() for s in seeds)
                ax.set_ylim(top - 2.5, top + 0.3)
            else:
                top = max(max(p[1] for p in s) for seeds in arm_curves(runs, key, x_key).values() for s in seeds)
                ax.set_ylim(top - 0.06, top + 0.01)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'VarNet + LOUPE, R=8, {data}: validation curves (seed mean, range shaded)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(out_path)


if __name__ == '__main__':
    main()
