"""Figure for reports/ACS_MASK_GATE_CGSENSE_R8.md from the ACS gate outputs (no GPU).

Left: each validation volume's held-out gain of its per-volume 8 x 8 density grid over the other-subject control
(gate v2, `grid - grid_ctrl`), brain coloured by raw coil count from the header (`acs_gate_scanner_check_r8.py`),
knee in grey. Right: share of the sampling budget per k-space ring for the population mask and for each brain
volume's in-sample grid fit, split by coil count.

    python scripts/diagnostics/plot_acs_gate_r8.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GATE = Path('reports/acs_gate_cgsense_r8')
OUT = Path('reports/figures/acs_mask_gate_cgsense_r8.png')
RING_LABELS = ('0-8', '8-16', '16-32', '32-64', '64+')
FEW, MANY, KNEE, INK, MUTED = '#eb6834', '#2a78d6', '#9a9893', '#0b0b0b', '#52514e'


def main():
    theta = json.loads((GATE / 'theta_check_gain.json').read_text())
    brain = sorted(theta['per_volume'], key=lambda v: -v['held_out_gain'])
    scanner = json.loads((GATE / 'scanner_check.json').read_text())['val']
    coils = {r['volume']: r['kspace_coils'] for r in scanner}
    knee = []
    for path in sorted((GATE / 'knee_v2').glob('*.json')):
        r = json.loads(path.read_text())
        knee.append(np.mean(r['psnr']['grid']) - np.mean(r['psnr']['grid_ctrl']))
    knee = sorted(knee, reverse=True)

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor='#fcfcfb', gridspec_kw={'width_ratios': [1.4, 1]})
    colours = [FEW if coils[v['volume']] <= 8 else MANY for v in brain]
    left.bar(range(len(brain)), [v['held_out_gain'] for v in brain], color=colours, width=0.8)
    left.bar(range(len(brain) + 1, len(brain) + 1 + len(knee)), knee, color=KNEE, width=0.8)
    for i, v in enumerate(brain[:6]):
        left.text(i, v['held_out_gain'] + 0.02, f"{coils[v['volume']]}c", ha='center', fontsize=7, color=MUTED)
    left.axhline(0, color='#c3c2b7', linewidth=1)
    left.set_xticks([len(brain) / 2 - 0.5, len(brain) + len(knee) / 2 + 0.5], ['brain volumes (23)', 'knee volumes (10)'])
    left.set_ylabel('held-out gain of per-volume grid\nover its control (dB)')
    left.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (FEW, MANY, KNEE)],
                labels=['brain, 4-6 coils', 'brain, 14-20 coils', 'knee'], frameon=False, fontsize=8)
    left.spines[['top', 'right']].set_visible(False)

    x = np.arange(len(RING_LABELS))
    base = theta['summary']['direction']['base_ring_share']
    for group, colour, offset in (('few', FEW, -0.27), ('many', MANY, 0.0)):
        shares = np.array([v['ring_share_in'] for v in brain if (coils[v['volume']] <= 8) == (group == 'few')])
        right.bar(x + offset, shares.mean(0) - base, width=0.27, color=colour,
                  label=f"brain, {'4-6' if group == 'few' else '14-20'} coils (n={len(shares)})")
        print(f'{group}-coil volumes, ring share change vs population (rings {RING_LABELS}):',
              ' '.join(f'{d:+.4f}' for d in shares.mean(0) - base))
    right.axhline(0, color='#c3c2b7', linewidth=1)
    right.set_xticks(x, RING_LABELS)
    right.set_xlabel('k-space ring (px from centre)')
    right.set_ylabel('change in share of samples\n(in-sample grid fit minus population mask)')
    right.legend(frameon=False, fontsize=8)
    right.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Per-volume density adaptation on CG-SENSE (R=8, 24 x 24 ACS): gain and where the samples move',
                 fontsize=10, color=INK)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == '__main__':
    main()
