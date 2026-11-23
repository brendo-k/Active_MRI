"""Frozen criteria of the loupe_mask_commitment_r8 follow-up (followup_proposal.md), meant for seeds 42/43/44.

`mixdecay` is paired against the 100-epoch control (runs_e100: round 1's control_s42/s43 plus control_s44) and
`mix_e150` against control_e150 (runs_e150). Each folder's paired eval (scripts/eval/varnet_loupe_lr_schedule_r8_eval.py)
and band counts (scripts/plots/loupe_mask_commitment_r8_masks.py --name masks_followup_<folder>) are read from the report
folder. The frozen rule needs all three seeds; an arm scored on fewer is reported but makes the reading "not evaluable".
The cross-length readings pair 150-epoch runs against 100-epoch runs of the same seed on the same slices (control_e150 and
mix_e150 against the 100-epoch control, mix_e150 against mixdecay). The curves are the trainer's val PSNR (batch-max data
range); `control_peak` is each run's best trainer val point against its last one, a within-run check of the late decline.

    python scripts/eval/loupe_mask_commitment_r8_followup_summary.py --report reports/loupe_mask_commitment_r8 \
        --runs-e100 /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e100 \
        --runs-e150 /export/scratch/bkadota/loupe_mask_commitment_r8/runs_e150
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from loupe_mask_commitment_r8_summary import BAND_KEPT, CONDITION_FLOOR_DB, CONDITIONS, GAIN_DB, SSIM_FLOOR, criteria, curve
from varnet_train_speed_eval import paired

SEEDS = ('42', '43', '44')
ARM_FOLDERS = {'mixdecay': ('e100', 'mixdecay'), 'mix_e150': ('e150', 'mix')}
COLORS = {'control': '#52514e', 'mixdecay': '#9a5fd0', 'mix': '#1baf7a'}
STYLES = {'42': '-', '43': '--', '44': ':'}


def reading(table):
    incomplete = [a for a in ARM_FOLDERS if table[a] is None or tuple(table[a]['band']) != SEEDS]
    if incomplete:
        return f"not evaluable: {', '.join(incomplete)} not scored on all of seeds {'/'.join(SEEDS)}"
    passes = {a: table[a]['band_kept'] and table[a]['fix_works'] for a in table}
    if passes['mixdecay']:
        return 'band helps once the recon is not taxed'
    if passes['mix_e150']:
        return 'slow convergence'
    if all(table[a]['band_kept'] for a in table):
        return 'band kept, gap not explained'
    return 'inconclusive'


CROSS_LENGTH = {'control_e150_minus_control_e100': ('control', 'control'),
                'mix_e150_minus_control_e100': ('mix', 'control'),
                'mix_e150_minus_mixdecay_e100': ('mix', 'mixdecay')}


def cross_length(report):
    long_cache, short_cache = (report / f'eval_followup_{f}.per_slice.npz' for f in ('e150', 'e100'))
    if not long_cache.exists():
        return {}
    long, short = dict(np.load(long_cache, allow_pickle=True)), dict(np.load(short_cache, allow_pickle=True))
    out = {}
    for key, (long_arm, short_arm) in CROSS_LENGTH.items():
        for seed in SEEDS:
            a, b = f'{long_arm}_s{seed}', f'{short_arm}_s{seed}'
            if f'{a}/psnr' not in long or f'{b}/psnr' not in short:
                continue
            assert (long[f'{a}/volume'] == short[f'{b}/volume']).all()
            conditions, volumes, diff = long[f'{a}/condition'], long[f'{a}/volume'], long[f'{a}/psnr'] - short[f'{b}/psnr']
            out.setdefault(key, {})[seed] = {
                c: paired(diff[keep], volumes[keep])
                for c, keep in [('all', np.ones(len(conditions), bool)), *[(c, conditions == c) for c in CONDITIONS]]}
    return out


def peak_vs_last(rows):
    return {key: {'peak': max(r[key] for r in rows), 'peak_epoch': max(rows, key=lambda r: r[key])['epoch'],
                  'last': rows[-1][key], 'last_epoch': rows[-1]['epoch']} for key in ('brain_psnr', 'knee_psnr')}


def plot_curves(curves, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, key, title in zip(axes, ('brain_psnr', 'knee_psnr'), ('brain', 'knee (PD + PDFS)')):
        for name, rows in curves.items():
            folder, run = name.split('/')
            arm, seed = run.rsplit('_s', 1)
            axis.plot([r['epoch'] for r in rows], [r[key] for r in rows], color=COLORS[arm], lw=1.4 if folder == 'e100' else 2.2,
                      alpha=1 if folder == 'e100' else 0.6, ls=STYLES[seed], label=f'{run} ({folder})')
        axis.set_title(f'trainer val PSNR, {title}', loc='left', fontsize=11)
        axis.set_xlabel('epoch')
        axis.set_ylabel('PSNR (dB, batch-max range)')
        axis.grid(axis='y', alpha=0.3)
        axis.set_ylim(min(r[key] for rows in curves.values() for r in rows[2:]) - 0.1, None)
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc='lower right')
    fig.suptitle('Follow-up: VarNet 12x18 + LOUPE R=8, full local v1 (solid s42, dashed s43, dotted s44; thick = 150 epochs); '
                 'first two val points cut from the y range', x=0.01, ha='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--runs-e100', type=Path, required=True)
    parser.add_argument('--runs-e150', type=Path, required=True)
    args = parser.parse_args()
    runs = {'e100': args.runs_e100, 'e150': args.runs_e150}

    table, control_band, curves = {}, {}, {}
    for label, (folder, arm) in ARM_FOLDERS.items():
        eval_json, masks_json = args.report / f'eval_followup_{folder}.json', args.report / f'masks_followup_{folder}.json'
        if not (eval_json.exists() and masks_json.exists()):
            table[label] = None
            continue
        bands = json.loads(masks_json.read_text())['runs']
        arm_paired = json.loads(eval_json.read_text())['paired']
        seeds = tuple(s for s in SEEDS if s in arm_paired.get(arm, {}))
        table[label] = criteria(arm, arm_paired, bands, seeds)
        control_band[folder] = {s: bands[f'control_s{s}']['right_band'] for s in seeds}
        for name in ('control', arm):
            for s in seeds:
                curves[f'{folder}/{name}_s{s}'] = curve(runs[folder] / f'{name}_s{s}' / 'train_sequential.log')

    summary = {'thresholds': {'band_kept': BAND_KEPT, 'mean_gain_db': GAIN_DB, 'condition_floor_db': CONDITION_FLOOR_DB,
                              'ssim_floor': SSIM_FLOOR},
               'control_band': control_band, 'arms': table, 'reading': reading(table),
               'cross_length': cross_length(args.report),
               'trainer_val_peak_vs_last': {name: peak_vs_last(rows) for name, rows in curves.items()},
               'trainer_val_curves': curves}
    (args.report / 'summary_followup.json').write_text(json.dumps(summary, indent=1))
    (args.report / 'figures').mkdir(exist_ok=True)
    plot_curves(curves, args.report / 'figures' / 'curves_followup.png')

    for arm, row in table.items():
        if row is None:
            print(f'{arm:9s} pending')
            continue
        print(f"{arm:9s} seeds {'/'.join(row['band'])} band " + '/'.join(f"{v:.0f}" for v in row['band'].values())
              + f" kept={row['band_kept']!s:5s} gain " + '/'.join(f"{v:+.3f}" for v in row['psnr_gain'].values()) + f" mean {row['mean_psnr_gain']:+.3f} worst cond "
              f"{row['worst_condition_gain']:+.3f} worst ssim {row['worst_ssim_gain']:+.4f} fix={row['fix_works']}")
        for s in row['condition_gain']:
            print('   s' + s, ' '.join(f"{c} {v:+.3f}" for c, v in row['condition_gain'][s].items()))
    print('control band', control_band)
    print('reading:', summary['reading'])
    for key, seeds in summary['cross_length'].items():
        for s, entry in seeds.items():
            print(f'{key} s{s}: ' + ' '.join(f"{c} {e['mean']:+.3f}" for c, e in entry.items()))
    for name, entry in summary['trainer_val_peak_vs_last'].items():
        b = entry['brain_psnr']
        print(f"{name:20s} brain peak {b['peak']:.2f} @ {b['peak_epoch']} last {b['last']:.2f} @ {b['last_epoch']}")


if __name__ == '__main__':
    main()
