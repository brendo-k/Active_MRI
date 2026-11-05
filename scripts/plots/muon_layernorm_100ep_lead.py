"""Each arm's lead over its same-seed control against epoch, one panel per seed, from the trainer's val PSNR_Masked
(reports/muon_layernorm_100ep/curves.json, written by scripts/eval/varnet_train_speed_curves.py).

    python scripts/plots/muon_layernorm_100ep_lead.py <curves.json> <out.png>
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = (42, 43)
ARMS = (('muon', 'Muon', '#2a78d6'), ('layer', 'LayerNorm', '#eb6834'), ('muon_layer', 'Muon + LayerNorm', '#1baf7a'))
SURFACE, INK, SECONDARY_INK, GRID, BASELINE = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
LR_STEP_EPOCH = 90
LABEL_GAP_DB = 0.2
Y_LIMITS = (-1.5, 2.1)


def psnr_by_epoch(curves, name):
    return {row['epoch']: row['val/PSNR_Masked'] for row in curves['runs'][name]['curve'] if 'val/PSNR_Masked' in row}


def spaced(values, gap):
    """Label heights for sorted line ends: each at least `gap` above the one below it."""
    heights = []
    for value in sorted(values):
        heights.append(max(value, heights[-1] + gap) if heights else value)
    return dict(zip(sorted(values), heights))


def main():
    curves = json.load(open(sys.argv[1]))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': SECONDARY_INK,
                         'xtick.color': SECONDARY_INK, 'ytick.color': SECONDARY_INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    for ax, seed in zip(axes, SEEDS):
        control = psnr_by_epoch(curves, f'control_s{seed}')
        ends = {}
        for arm, label, color in ARMS:
            psnr = psnr_by_epoch(curves, f'{arm}_s{seed}')
            epochs = sorted(set(psnr) & set(control))
            leads = [psnr[epoch] - control[epoch] for epoch in epochs]
            ax.plot(epochs, leads, color=color, lw=2, label=label, solid_capstyle='round')
            ends[leads[-1]] = (label, epochs[-1])
        heights = spaced(ends, LABEL_GAP_DB)
        for lead, (label, epoch) in ends.items():
            ax.text(epoch + 1.5, heights[lead], f'{label} {lead:+.2f}', va='center', fontsize=9, color=INK)
        ax.hlines(0, 1, 100, color=SECONDARY_INK, lw=1)
        ax.axvline(LR_STEP_EPOCH, color=BASELINE, lw=1, ls=(0, (4, 3)))
        ax.text(LR_STEP_EPOCH - 1, Y_LIMITS[1], 'lr step', ha='right', va='top', fontsize=9, color=SECONDARY_INK)
        ax.set_title(f'Seed {seed}', loc='left', fontsize=11, color=INK)
        ax.set_xlabel('epoch')
        ax.set_xlim(1, 128)
        ax.set_xticks(range(0, 101, 20))
        ax.set_ylim(*Y_LIMITS)
        ax.set_facecolor(SURFACE)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(BASELINE)
        ax.tick_params(length=0)
    axes[0].set_ylabel('val PSNR_Masked lead over the same-seed control (dB)', fontsize=9)
    axes[1].legend(frameon=False, loc='lower right', fontsize=9, ncol=1)
    fig.suptitle('Lead over the same-seed control: Muon\'s early lead fades, and the combination stays ahead in both seeds',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150, facecolor=SURFACE)
    print(sys.argv[2])


if __name__ == '__main__':
    main()
