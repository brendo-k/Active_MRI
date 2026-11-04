"""Validation PSNR_Masked of the muon_layer_speed arms beside the 100-epoch reference, against epoch and against wall-clock.

Reads two files written by scripts/eval/varnet_train_speed_curves.py, one for the arms and one for the reference queue.
Wall-clock is train + validation time from the run logs; the arms validate every 5 epochs, the reference every epoch.

    python scripts/plots/muon_layer_speed_curves.py <curves.json> <curves_reference.json> <out.png>
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REFERENCE_RUN = 'muon_layer_s42'
BAR_DB = 0.10
PSNR = 'val/PSNR_Masked'
INK, SECONDARY_INK, MUTED_INK, GRID, BASELINE, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
ARMS = [  # run, legend label, color, marker. Color follows the arm, in the palette's fixed slot order.
    ('e30_cos_s42', 'e30_cos', '#2a78d6', 'o'),
    ('e30_step_s42', 'e30_step', '#eb6834', 's'),
    ('e30_bs8_s42', 'e30_bs8', '#1baf7a', '^'),
    ('e30_cos_muon2e3_s42', 'e30_cos_muon2e3', '#eda100', 'D'),
    ('e30_step_c16_s42', 'e30_step_c16 (post-freeze)', '#e87ba4', 'v'),
]


def points(run):
    """(epoch, wall-clock hours, PSNR) for every validated epoch."""
    elapsed, rows = 0.0, []
    for row in run['curve']:
        elapsed += row.get('train_time', 0.0) + row.get('val_time', 0.0)
        if PSNR in row:
            rows.append((row['epoch'], elapsed / 3600, row[PSNR]))
    return rows


def style(axis, label):
    axis.set_xlabel(label, color=SECONDARY_INK)
    axis.set_ylabel('val PSNR_Masked (dB)', color=SECONDARY_INK)
    axis.grid(axis='y', color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    axis.set_ylim(32.2, 34.0)
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=MUTED_INK)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(BASELINE)


def main():
    curves_path, reference_path, out_path = sys.argv[1:4]
    runs = json.load(open(curves_path))['runs']
    reference = points(json.load(open(reference_path))['runs'][REFERENCE_RUN])
    final = reference[-1][2]
    fig, (by_epoch, by_hours) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for axis, column, label in ((by_epoch, 0, 'epoch'), (by_hours, 1, 'wall-clock hours (train + validation)')):
        style(axis, label)
        axis.plot([row[column] for row in reference], [row[2] for row in reference], color=SECONDARY_INK, lw=2,
                  label='100-epoch reference')
        axis.axhline(final - BAR_DB, color=MUTED_INK, lw=1, ls='--', label=f'reference final - {BAR_DB:.2f} dB')
        for name, arm_label, color, marker in ARMS:
            if name in runs:
                rows = points(runs[name])
                axis.plot([row[column] for row in rows], [row[2] for row in rows], color=color, lw=2, marker=marker,
                          markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm_label)
    by_epoch.set_title('Same epoch: the arms track the reference until the lr drops', color=INK, loc='left', fontsize=11)
    by_hours.set_title('Same wall-clock: a shorter schedule buys time, not the last 0.3 dB', color=INK, loc='left', fontsize=11)
    by_epoch.legend(frameon=False, fontsize=9, loc='lower right', labelcolor=SECONDARY_INK)
    fig.suptitle('VarNet + LOUPE, R=8, Muon + LayerNorm, seed 42, full local v1 FLAIR + knee', color=INK, x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(out_path)


if __name__ == '__main__':
    main()
