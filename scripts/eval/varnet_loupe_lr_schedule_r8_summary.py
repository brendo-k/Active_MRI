"""Check the varnet_loupe_lr_schedule_r8 eval JSONs against the frozen criteria of proposal.md, followup_proposal.md and
followup2_proposal.md,
and read each run's trainer val curve for the overfitting check.

    python scripts/eval/varnet_loupe_lr_schedule_r8_summary.py \
        --runs /export/scratch/bkadota/varnet_loupe_lr_schedule_r8 --report reports/varnet_loupe_lr_schedule_r8

Inputs: <report>/eval_runs_last_model.json, eval_runs_best_model.json, eval_runs_full_last_model.json,
eval_runs_full_c6_last_model.json (from
varnet_loupe_lr_schedule_r8_eval.py) and <runs>/{runs,runs_full,runs_full_c6}/<name>/train_sequential.log.
Writes <report>/summary.json and prints the tables.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')
ADOPT_DB = 0.10
CONDITION_FLOOR_DB = -0.05
SSIM_FLOOR = -0.002
TWIN_BAR_DB = 0.10
EARLY_PEAK_EPOCH = 80
INTERACTION_DB = 0.08
CASCADE_ARMS = ('lr2e3_step', 'lr4e3_step')

SWEEPS = {
    'first': {'eval': 'eval_runs_last_model.json', 'runs': 'runs', 'twins': {'mask_lr': ('shared_2e3_step', 'lr2e3_step')}},
    'first_best': {'eval': 'eval_runs_best_model.json', 'runs': 'runs', 'twins': {'mask_lr': ('shared_2e3_step', 'lr2e3_step')}},
    'followup': {'eval': 'eval_runs_full_last_model.json', 'runs': 'runs_full',
                 'twins': {'cosine_4e3': ('lr4e3_wcos', 'lr4e3_step'), 'cosine_2e3': ('lr2e3_wcos', 'lr2e3_step')}},
    'followup2': {'eval': 'eval_runs_full_c6_last_model.json', 'runs': 'runs_full_c6', 'twins': {}},
}

VAL_LINE = re.compile(r'\[brain\].*?PSNR=([\d.]+).*\[knee\].*?PSNR=([\d.]+)')
EPOCH_LINE = re.compile(r'Epoch = \[\s*(\d+)/\s*(\d+)\]')


def arm_rows(paired):
    rows = {}
    for arm, seeds in paired.items():
        order = sorted(seeds)
        psnr = [seeds[s]['all']['psnr']['mean'] for s in order]
        rows[arm] = {
            'seeds': order,
            'psnr_per_seed': psnr,
            'psnr_ci_per_seed': [seeds[s]['all']['psnr']['ci'] for s in order],
            'psnr_mean': float(np.mean(psnr)),
            'n_positive': int(sum(d > 0 for d in psnr)),
            'ssim_mean': float(np.mean([seeds[s]['all']['ssim']['mean'] for s in order])),
            'conditions': {c: float(np.mean([seeds[s][c]['psnr']['mean'] for s in order])) for c in CONDITIONS},
        }
        row = rows[arm]
        row['adopt'] = bool(row['psnr_mean'] >= ADOPT_DB and row['n_positive'] == len(order)
                            and min(row['conditions'].values()) >= CONDITION_FLOOR_DB and row['ssim_mean'] >= SSIM_FLOOR)
    return rows


def twin(paired, arm, base):
    seeds = sorted(set(paired[arm]) & set(paired[base]))
    diffs = [paired[arm][s]['all']['psnr']['mean'] - paired[base][s]['all']['psnr']['mean'] for s in seeds]
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    return {'arm': arm, 'base': base, 'seeds': seeds, 'per_seed': diffs, 'mean': float(np.mean(diffs)),
            'conditions': {c: float(np.mean([paired[arm][s][c]['psnr']['mean'] - paired[base][s][c]['psnr']['mean']
                                             for s in seeds])) for c in CONDITIONS},
            'effect': bool(abs(np.mean(diffs)) >= TWIN_BAR_DB and same_sign)}


def interaction(paired6, paired12, arm):
    seeds = sorted(set(paired6[arm]) & set(paired12[arm]))
    d6 = [paired6[arm][s]['all']['psnr']['mean'] for s in seeds]
    d12 = [paired12[arm][s]['all']['psnr']['mean'] for s in seeds]
    diff = float(np.mean(d6) - np.mean(d12))
    if diff <= -INTERACTION_DB and all(a < b for a, b in zip(d6, d12)):
        verdict = 'shrinks'
    elif abs(diff) < INTERACTION_DB:
        verdict = 'holds'
    else:
        verdict = 'inconclusive'
    return {'seeds': seeds, 'd6': d6, 'd12': d12, 'd6_mean': float(np.mean(d6)), 'd12_mean': float(np.mean(d12)),
            'd6_minus_d12': diff, 'verdict': verdict,
            'conditions': {c: float(np.mean([paired6[arm][s][c]['psnr']['mean'] - paired12[arm][s][c]['psnr']['mean']
                                             for s in seeds])) for c in CONDITIONS}}


def cascade_control(runs6, runs12, control):
    seeds = sorted(name.rsplit('_s', 1)[1] for name in runs6 if name.startswith(control))
    out = {}
    for group in ('all', *CONDITIONS):
        per_seed = {s: runs6[f'{control}_s{s}'][group]['psnr_masked'] - runs12[f'{control}_s{s}'][group]['psnr_masked']
                    for s in seeds}
        ssim = {s: runs6[f'{control}_s{s}'][group]['ssim_masked'] - runs12[f'{control}_s{s}'][group]['ssim_masked']
                for s in seeds}
        out[group] = {'psnr_per_seed': per_seed, 'psnr_mean': float(np.mean(list(per_seed.values()))),
                      'ssim_per_seed': ssim}
    return out


def val_curve(log):
    pending, curve = None, []
    for line in log.read_text().splitlines():
        if match := VAL_LINE.search(line):
            pending = (float(match[1]), float(match[2]))
        elif (match := EPOCH_LINE.search(line)) and pending:
            curve.append((int(match[1]), *pending))
            pending = None
    return curve


def curve_summary(curve):
    epochs, brain, knee = (np.array(v) for v in zip(*curve))
    out = {'last_epoch': int(epochs[-1])}
    for name, values in (('brain', brain), ('knee', knee)):
        peak = int(values.argmax())
        out[name] = {'peak_epoch': int(epochs[peak]), 'peak': float(values[peak]), 'final': float(values[-1]),
                     'drop_from_peak': float(values[peak] - values[-1])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True, help='folder holding runs/ and runs_full/')
    parser.add_argument('--report', type=Path, default=Path('reports/varnet_loupe_lr_schedule_r8'))
    args = parser.parse_args()

    summary = {}
    for sweep, spec in SWEEPS.items():
        data = json.loads((args.report / spec['eval']).read_text())
        rows = arm_rows(data['paired'])
        curves = {name: curve_summary(val_curve(args.runs / spec['runs'] / name / 'train_sequential.log'))
                  for name in sorted(data['per_run'])}
        summary[sweep] = {
            'checkpoint': data['checkpoint'],
            'control_absolute': {name: run for name, run in data['per_run'].items() if name.startswith(data['control'])},
            'arms': rows,
            'twins': {key: twin(data['paired'], *pair) for key, pair in spec['twins'].items()},
            'curves': curves,
            'early_brain_peaks': sorted(n for n, c in curves.items() if c['brain']['peak_epoch'] < EARLY_PEAK_EPOCH),
            'early_knee_peaks': sorted(n for n, c in curves.items() if c['knee']['peak_epoch'] < EARLY_PEAK_EPOCH),
        }
        print(f'== {sweep} ({data["checkpoint"]})')
        for arm, row in sorted(rows.items(), key=lambda kv: -kv[1]['psnr_mean']):
            print(f"  {arm:16s} {row['psnr_mean']:+.3f}  seeds " + ' '.join(f'{d:+.3f}' for d in row['psnr_per_seed'])
                  + f"  ssim {row['ssim_mean']:+.4f}  " + ' '.join(f'{c} {v:+.3f}' for c, v in row['conditions'].items())
                  + ('  ADOPT' if row['adopt'] else ''))
        for key, entry in summary[sweep]['twins'].items():
            print(f"  twin {key}: {entry['arm']} - {entry['base']} {entry['mean']:+.3f} seeds "
                  + ' '.join(f'{d:+.3f}' for d in entry['per_seed']) + ('  EFFECT' if entry['effect'] else '  no effect'))
        print(f"  brain peak < epoch {EARLY_PEAK_EPOCH}: {len(summary[sweep]['early_brain_peaks'])}/{len(curves)} runs, "
              f"knee: {len(summary[sweep]['early_knee_peaks'])}/{len(curves)}")

    six, twelve = (json.loads((args.report / SWEEPS[k]['eval']).read_text()) for k in ('followup2', 'followup'))
    summary['followup2']['interaction'] = {arm: interaction(six['paired'], twelve['paired'], arm) for arm in CASCADE_ARMS}
    summary['followup2']['cascade_control'] = cascade_control(six['per_run'], twelve['per_run'], six['control'])
    print('== followup2 interaction (d6 - d12, seed-mean paired gain vs same-cascade control)')
    for arm, entry in summary['followup2']['interaction'].items():
        print(f"  {arm:12s} d6 {entry['d6_mean']:+.3f} ({' '.join(f'{d:+.3f}' for d in entry['d6'])})  "
              f"d12 {entry['d12_mean']:+.3f} ({' '.join(f'{d:+.3f}' for d in entry['d12'])})  "
              f"d6-d12 {entry['d6_minus_d12']:+.3f}  {entry['verdict']}  "
              + ' '.join(f'{c} {v:+.3f}' for c, v in entry['conditions'].items()))
    print('== 6-cascade minus 12-cascade control (unpaired, absolute PSNR)')
    for group, entry in summary['followup2']['cascade_control'].items():
        print(f"  {group:10s} {entry['psnr_mean']:+.3f}  seeds " + ' '.join(f'{s}:{d:+.3f}' for s, d in entry['psnr_per_seed'].items())
              + '  ssim ' + ' '.join(f'{d:+.4f}' for d in entry['ssim_per_seed'].values()))

    (args.report / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
