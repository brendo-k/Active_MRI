"""Frozen criteria of loupe_mask_commitment_r8 and the trainer val curves of its 8 runs.

Reads the paired eval (scripts/eval/varnet_loupe_lr_schedule_r8_eval.py) and the band counts
(scripts/plots/loupe_mask_commitment_r8_masks.py) from the report folder, applies the proposal's criteria per arm, and
writes summary.json. The curves are the trainer's per-anatomy val PSNR parsed from each run's train_sequential.log (val
every 5 epochs; data range = batch max, so they sit above the eval's numbers and are only compared with each other).

    python scripts/eval/loupe_mask_commitment_r8_summary.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs \
        --report reports/loupe_mask_commitment_r8
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ('clamp', 'anneal', 'mix')
SEEDS = ('42', '43')
CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')
BAND_KEPT = 200
GAIN_DB = 0.10
CONDITION_FLOOR_DB = -0.05
SSIM_FLOOR = -0.002
VAL_LINE = re.compile(r'\[brain\] .*?PSNR=([\d.]+) .*?\| \[knee\] .*?PSNR=([\d.]+)')
EPOCH_LINE = re.compile(r'Epoch = \[ *(\d+)/ *\d+\] .*ValLoss = ([\d.]+)')
COLORS = {'control': '#52514e', 'clamp': '#2a78d6', 'anneal': '#eb6834', 'mix': '#1baf7a'}


def criteria(arm, paired, bands):
    seeds = paired[arm]
    gains = [seeds[s]['all']['psnr']['mean'] for s in SEEDS]
    worst_condition = min(seeds[s][c]['psnr']['mean'] for s in SEEDS for c in CONDITIONS)
    worst_ssim = min(seeds[s]['all']['ssim']['mean'] for s in SEEDS)
    band = [bands[f'{arm}_s{s}']['right_band'] for s in SEEDS]
    band_kept = min(band) >= BAND_KEPT
    fix_works = (sum(gains) / len(gains) >= GAIN_DB and min(gains) > 0
                 and worst_condition >= CONDITION_FLOOR_DB and worst_ssim >= SSIM_FLOOR)
    return {'band': dict(zip(SEEDS, band)), 'band_kept': band_kept,
            'psnr_gain': dict(zip(SEEDS, gains)), 'mean_psnr_gain': sum(gains) / len(gains),
            'worst_condition_gain': worst_condition, 'worst_ssim_gain': worst_ssim, 'fix_works': fix_works,
            'frac_volumes_better': {s: seeds[s]['all']['psnr']['frac_volumes_better'] for s in SEEDS},
            'condition_gain': {s: {c: seeds[s][c]['psnr']['mean'] for c in CONDITIONS} for s in SEEDS}}


def reading(table):
    if table['clamp']['band_kept'] and table['clamp']['fix_works']:
        return 'saturation'
    if not table['clamp']['band_kept'] and any(table[a]['band_kept'] and table[a]['fix_works'] for a in ('anneal', 'mix')):
        return 'race / co-adaptation'
    return 'inconclusive'


def curve(log):
    rows, pending = [], None
    for line in log.read_text().splitlines():
        if (match := VAL_LINE.search(line)):
            pending = (float(match[1]), float(match[2]))
        elif (match := EPOCH_LINE.search(line)) and pending:
            rows.append({'epoch': int(match[1]), 'val_loss': float(match[2]), 'brain_psnr': pending[0], 'knee_psnr': pending[1]})
            pending = None
    return rows


def plot_curves(curves, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, key, title in zip(axes, ('brain_psnr', 'knee_psnr'), ('brain', 'knee (PD + PDFS)')):
        for name, rows in curves.items():
            arm, seed = name.rsplit('_s', 1)
            axis.plot([r['epoch'] for r in rows], [r[key] for r in rows], color=COLORS[arm], lw=1.6,
                      ls='-' if seed == '42' else '--', label=name)
        axis.set_title(f'trainer val PSNR, {title}', loc='left', fontsize=11)
        axis.set_xlabel('epoch')
        axis.set_ylabel('PSNR (dB, batch-max range)')
        axis.grid(axis='y', alpha=0.3)
        axis.set_ylim(min(r[key] for rows in curves.values() for r in rows[2:]) - 0.1, None)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc='lower right')
    fig.suptitle('VarNet 12x18 + LOUPE R=8, full local v1 (solid s42, dashed s43); first two val points cut from the y range',
                 x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    paired = json.loads((args.report / 'eval_last_model.json').read_text())['paired']
    bands = json.loads((args.report / 'masks.json').read_text())['runs']
    table = {arm: criteria(arm, paired, bands) for arm in ARMS}
    curves = {f'{arm}_s{s}': curve(args.runs / f'{arm}_s{s}' / 'train_sequential.log')
              for arm in ('control', *ARMS) for s in SEEDS}
    summary = {'thresholds': {'band_kept': BAND_KEPT, 'mean_gain_db': GAIN_DB, 'condition_floor_db': CONDITION_FLOOR_DB,
                              'ssim_floor': SSIM_FLOOR},
               'control_band': {s: bands[f'control_s{s}']['right_band'] for s in SEEDS},
               'arms': table, 'reading': reading(table), 'trainer_val_curves': curves}
    (args.report / 'summary.json').write_text(json.dumps(summary, indent=1))
    (args.report / 'figures').mkdir(exist_ok=True)
    plot_curves(curves, args.report / 'figures' / 'curves.png')
    for arm, row in table.items():
        print(f"{arm:7s} band {row['band']['42']:6.1f}/{row['band']['43']:6.1f} kept={row['band_kept']!s:5s} "
              f"gain {row['psnr_gain']['42']:+.3f}/{row['psnr_gain']['43']:+.3f} mean {row['mean_psnr_gain']:+.3f} "
              f"worst cond {row['worst_condition_gain']:+.3f} worst ssim {row['worst_ssim_gain']:+.4f} fix={row['fix_works']}")
    print('reading:', summary['reading'])
    for name, rows in curves.items():
        print(f"{name:12s} final trainer val brain {rows[-1]['brain_psnr']:.2f} knee {rows[-1]['knee_psnr']:.2f} (ep {rows[-1]['epoch']})")


if __name__ == '__main__':
    main()
