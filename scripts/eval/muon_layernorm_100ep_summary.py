"""The frozen criteria of reports/muon_layernorm_100ep/proposal.md, decided from the study's result files.

Inputs, all in the report folder:
- eval.json and eval.per_slice.npz: `scripts/eval/varnet_train_speed_eval.py --queue-dir <queue> --control control`.
- curves.json: `scripts/eval/varnet_train_speed_curves.py --queue-dir <queue> --prefix ml100_`.
The run logs are read for skipped optimizer steps and non-finite losses.

Epoch times of the 12-epoch full-data runs come from --reference-curves (reports/varnet_train_speed/curves.json).

Writes criteria.json and summary.md next to them. Arms or seeds still training show as pending.

    python scripts/eval/muon_layernorm_100ep_summary.py reports/muon_layernorm_100ep
"""
import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

QUEUE_DIR = Path('~/data/seqmri_checkpoints/muon_layernorm_100ep').expanduser()
REFERENCE_CURVES = Path('reports/varnet_train_speed/curves.json')
REFERENCE_RUNS = ('full_control_s42', 'full_muon_s42', 'full_norm_layer_s42', 'full_muon_layer_s42')
SEEDS = ('42', '43')
ARMS = ('muon', 'layer', 'muon_layer')
CONDITIONS = ('brain', 'knee-PD', 'knee-PDFS')
EFFECT_DB = 0.15
SPEED_FRACTION = 0.5
SSIM_GUARD = -0.005
CONDITION_GUARD_DB = -0.10
NOT_WORSE_DB = -0.05
SKIP_STREAK_LIMIT = 10
TARGET_EPOCHS = range(96, 101)
TRAILING = 3
LATE_WINDOWS = ((46, 50), (66, 70), (86, 90), (96, 100))
LEAD_EPOCHS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
EPOCH_LINE = re.compile(r'Epoch = \[ *(\d+)/ *(\d+)\] TrainLoss = (\S+) ValLoss = (\S+)')


def paired(delta, volumes):
    names, index = np.unique(volumes, return_inverse=True)
    per_volume = np.bincount(index, weights=delta) / np.bincount(index)
    half = 1.96 * per_volume.std(ddof=1) / np.sqrt(len(names))
    return {'mean': float(delta.mean()), 'ci': [float(delta.mean() - half), float(delta.mean() + half)]}


class Scores:
    """Per-slice PSNR_Masked / SSIM_Masked of every scored run, from the eval cache."""

    def __init__(self, cache):
        stored = dict(np.load(cache, allow_pickle=True))
        self.runs = {key.rsplit('/', 1)[0] for key in stored}
        self.stored = stored

    def has(self, *names):
        return all(name in self.runs for name in names)

    def difference(self, arm, other, metric='psnr', condition=None):
        a, b = self.stored[f'{arm}/{metric}'], self.stored[f'{other}/{metric}']
        volumes, conditions = self.stored[f'{arm}/volume'], self.stored[f'{arm}/condition']
        assert (volumes == self.stored[f'{other}/volume']).all(), f'{arm} and {other} saw the slices in a different order'
        keep = np.ones(len(a), bool) if condition is None else conditions == condition
        return paired(a[keep] - b[keep], volumes[keep])


def run(arm, seed):
    return f'{arm}_s{seed}'


def factor_effects(scores):
    """Each main effect's 4 paired differences (2 seeds x the other factor's 2 levels) and the interaction per seed."""
    pairs = {'muon': [('muon', 'control'), ('muon_layer', 'layer')],
             'layer': [('layer', 'control'), ('muon_layer', 'muon')]}
    effects = {}
    for factor, comparisons in pairs.items():
        rows = [{'arm': arm, 'other': other, 'seed': seed, **scores.difference(run(arm, seed), run(other, seed)),
                 'per_condition': {c: scores.difference(run(arm, seed), run(other, seed), condition=c)['mean'] for c in CONDITIONS}}
                for seed in SEEDS for arm, other in comparisons if scores.has(run(arm, seed), run(other, seed))]
        by_level = {f'{arm} - {other}': statistics.mean(r['mean'] for r in rows if (r['arm'], r['other']) == (arm, other))
                    for arm, other in comparisons}
        effects[factor] = {'differences': rows, 'by_level': by_level, 'complete': len(rows) == 4,
                           'mean': statistics.mean(r['mean'] for r in rows) if rows else None,
                           'positive': sum(r['mean'] > 0 for r in rows), 'negative': sum(r['mean'] < 0 for r in rows)}
    effects['interaction'] = {seed: scores.difference(run('muon_layer', seed), run('layer', seed))['mean']
                              - scores.difference(run('muon', seed), run('control', seed))['mean']
                              for seed in SEEDS if scores.has(*(run(a, seed) for a in ('control', *ARMS)))}
    return effects


def psnr_by_epoch(curves, name):
    return {row['epoch']: row['val/PSNR_Masked'] for row in curves['runs'][name]['curve'] if 'val/PSNR_Masked' in row}


def median_times(curves, name):
    """Median train, val and whole-epoch seconds of one run, epoch 1's compile excluded."""
    rows = [row for row in curves['runs'][name]['curve'] if 'train_time' in row and row['epoch'] > 1]
    if not rows:
        return None
    return {'train_s': statistics.median(row['train_time'] for row in rows),
            'val_s': statistics.median(row['val_time'] for row in rows),
            'epoch_s': statistics.median(row['train_time'] + row['val_time'] for row in rows)}


def timing(curves, reference):
    def with_change_over(control, name, source):
        row, base = median_times(source, name), median_times(source, control)
        return {**row, **{f'{key}_vs_control': row[key] / base[key] - 1 for key in ('train_s', 'epoch_s')}}

    runs = {name: with_change_over(f"control_s{name.rsplit('_s', 1)[1]}", name, curves) for name in sorted(curves['runs'])}
    reference_runs = {name: with_change_over(REFERENCE_RUNS[0], name, reference) for name in REFERENCE_RUNS}
    return {'runs': runs, 'reference_12_epochs': reference_runs}


def reference_leads(reference):
    """The 12-epoch full-data arms' trainer val PSNR_Masked at their last epoch minus the same-seed control's, on all val slices."""
    finals = {name: psnr_by_epoch(reference, name)[12] for name in reference['runs'] if name.startswith('full_') and name[-3:] in ('s42', 's43')}
    return {name: value - finals[f"full_control_{name[-3:]}"] for name, value in sorted(finals.items()) if not name.startswith('full_control')}


def trajectory(curves):
    """Per arm and seed: window means of the trainer's val PSNR_Masked, the lead over the same-seed control in those
    windows and at fixed epochs, and the best epoch."""
    def mean_over(psnr, window):
        return statistics.mean(psnr[epoch] for epoch in range(window[0], window[1] + 1))

    result = {}
    for arm in ('control', *ARMS):
        for seed in SEEDS:
            psnr, control = psnr_by_epoch(curves, run(arm, seed)), psnr_by_epoch(curves, run('control', seed))
            best = max(psnr, key=psnr.get)
            result.setdefault(arm, {})[seed] = {
                'window_mean': {f'{lo}-{hi}': mean_over(psnr, (lo, hi)) for lo, hi in LATE_WINDOWS},
                'window_lead': {f'{lo}-{hi}': mean_over(psnr, (lo, hi)) - mean_over(control, (lo, hi)) for lo, hi in LATE_WINDOWS},
                'lead_at_epoch': {epoch: psnr[epoch] - control[epoch] for epoch in LEAD_EPOCHS},
                'own_gain_66_to_96': mean_over(psnr, (96, 100)) - mean_over(psnr, (66, 70)),
                'best_epoch': best, 'best_psnr': psnr[best], 'final_psnr': psnr[100]}
    return result


def time_to_target(curves):
    """Per arm and seed: the control's epoch 96-100 mean, the first epoch whose trailing 3-epoch mean reaches it, and
    that epoch's share of the control's wall-clock (each run's median epoch time, epoch 1's compile excluded)."""
    runs = curves['runs']

    def median_epoch_time(name):
        times = median_times(curves, name)
        return times['epoch_s'] if times else None

    result = {}
    for seed in SEEDS:
        control = run('control', seed)
        control_psnr = psnr_by_epoch(curves, control) if control in runs else {}
        if not all(epoch in control_psnr for epoch in TARGET_EPOCHS):
            continue
        target = statistics.mean(control_psnr[epoch] for epoch in TARGET_EPOCHS)
        control_time = median_epoch_time(control)
        for arm in ('control', *ARMS):
            name = run(arm, seed)
            if name not in runs:
                continue
            psnr = psnr_by_epoch(curves, name)
            reach = next((epoch for epoch in sorted(psnr) if all(e in psnr for e in range(epoch - TRAILING + 1, epoch + 1))
                          and statistics.mean(psnr[e] for e in range(epoch - TRAILING + 1, epoch + 1)) >= target), None)
            arm_time = median_epoch_time(name)
            result.setdefault(arm, {})[seed] = {
                'target_psnr_masked': target, 'reach_epoch': reach, 'median_epoch_s': arm_time,
                'control_median_epoch_s': control_time, 'epochs_done': max(psnr, default=0),
                'hours_to_target': reach * arm_time / 3600 if reach and arm_time else None,
                'control_hours_100_epochs': 100 * control_time / 3600 if control_time else None,
                'wall_fraction': reach * arm_time / (100 * control_time) if reach and arm_time and control_time else None}
    return result


def stability(queue_dir, name):
    log = queue_dir / f'{name}.log'
    if not log.exists():
        return None
    text = log.read_text(errors='replace')
    losses = [value for match in EPOCH_LINE.finditer(text) for value in match.groups()[2:]]
    return {'skipped_steps': text.count('Skipping optimizer step'),
            'non_finite_epoch_loss': any(not np.isfinite(float(value)) for value in losses)}


def verdict(row):
    if 'passed' not in row:
        return 'info'
    return 'pending' if row['passed'] is None else ('met' if row['passed'] else 'not met')


def decide(scores, effects, speed, evals, queue_dir):
    criteria = {}
    for factor, key in (('muon', 'A1'), ('layer', 'A2')):
        effect = effects[factor]
        criteria[key] = {'value': effect['mean'], 'positive_of_4': effect['positive'],
                         'passed': (effect['mean'] >= EFFECT_DB and effect['positive'] >= 3) if effect['complete'] else None}
    combination = [scores.difference(run('muon_layer', s), run('control', s))['mean'] for s in SEEDS
                   if scores.has(run('muon_layer', s), run('control', s))]
    criteria['B'] = {'value': statistics.mean(combination) if combination else None, 'per_seed': combination,
                     'passed': (min(combination) > 0 and statistics.mean(combination) >= EFFECT_DB) if len(combination) == 2 else None}
    for arm in ARMS:
        rows = [speed.get(arm, {}).get(seed) for seed in SEEDS]
        complete = all(row and row['epochs_done'] >= 100 for row in rows)
        fractions = [row['wall_fraction'] if row else None for row in rows]
        criteria[f'C:{arm}'] = {'wall_fraction': fractions,
                                'passed': all(f is not None and f <= SPEED_FRACTION for f in fractions) if complete else None}
        ready = all(scores.has(run(arm, s), run('control', s)) for s in SEEDS)
        ssim = [scores.difference(run(arm, s), run('control', s), 'ssim')['mean'] for s in SEEDS] if ready else []
        criteria[f'G1:{arm}'] = {'ssim_difference': ssim, 'passed': min(ssim) >= SSIM_GUARD if ready else None}
        worse = {c: [scores.difference(run(arm, s), run('control', s), condition=c)['mean'] for s in SEEDS] for c in CONDITIONS} if ready else {}
        criteria[f'G2:{arm}'] = {'per_condition': worse,
                                 'passed': not any(max(v) <= CONDITION_GUARD_DB for v in worse.values()) if ready else None}
        logs = [stability(queue_dir, run(arm, s)) for s in SEEDS]
        unresolved = any(log and log['skipped_steps'] > SKIP_STREAK_LIMIT for log in logs)
        criteria[f'G3:{arm}'] = {'logs': logs, 'note': 'over 10 skipped steps: check train/skipped_step streaks in wandb' if unresolved else '',
                                 'passed': None if unresolved or not ready else not any(log['non_finite_epoch_loss'] for log in logs)}
        final = [scores.difference(run(arm, s), run('control', s))['mean'] for s in SEEDS] if ready else []
        criteria[f'final:{arm}'] = {'per_seed': final, 'mean': statistics.mean(final) if final else None}
    ssim_all = {name: row['all']['ssim_masked'] for name, row in evals['per_run'].items()}
    control_final = [speed.get('control', {}).get(s, {}).get('target_psnr_masked') for s in SEEDS]
    criteria['S'] = {'min_ssim_masked': min(ssim_all.values()) if ssim_all else None, 'control_final_trainer_psnr': control_final,
                     'control_logs': [stability(queue_dir, run('control', s)) for s in SEEDS],
                     'passed': (min(ssim_all.values()) >= 0.80 and all(v is not None and v >= 31.8 for v in control_final))
                     if len(ssim_all) == 8 else None}
    return criteria


def guard_failed_in_both_seeds(criteria, arm):
    return (max(criteria[f'G1:{arm}']['ssim_difference']) < SSIM_GUARD or criteria[f'G2:{arm}']['passed'] is False
            or all(log['non_finite_epoch_loss'] for log in criteria[f'G3:{arm}']['logs']))


def recommendation(criteria, effects):
    guards = {arm: all(criteria[f'{g}:{arm}']['passed'] for g in ('G1', 'G2', 'G3')) for arm in ARMS}
    lines = []
    for factor, key in (('muon', 'A1'), ('layer', 'A2')):
        if criteria[key]['passed'] is None:
            lines.append(f'{factor}: pending')
        elif criteria[key]['passed'] and guards[factor]:
            lines.append(f'{factor}: flip the default (quality, {key} met, guards hold)')
        elif criteria[f'C:{factor}']['passed'] and guards[factor] and criteria[f'final:{factor}']['mean'] >= NOT_WORSE_DB:
            lines.append(f'{factor}: flip as a speed win ({key} not met, C met, final not worse); retime num_epochs')
        elif (effects[factor]['mean'] <= -EFFECT_DB and effects[factor]['negative'] >= 3) or guard_failed_in_both_seeds(criteria, factor):
            lines.append(f'{factor}: failure, keep the current default')
        else:
            lines.append(f'{factor}: inconclusive on its own, keep the current default')
    if criteria['B']['passed'] is None:
        lines.append('muon_layer: pending')
    elif criteria['B']['passed'] and guards['muon_layer']:
        lines.append('muon_layer: flip both defaults (quality, B met, guards hold)')
    elif criteria['C:muon_layer']['passed'] and guards['muon_layer'] and criteria['final:muon_layer']['mean'] >= NOT_WORSE_DB:
        lines.append('muon_layer: flip both as a speed win (B not met, C met, final not worse); retime num_epochs')
    else:
        lines.append('muon_layer: no flip from the combination')
    flips = [line.split(':')[0] for line in lines if 'flip' in line and 'no flip' not in line]
    if any('pending' in line for line in lines):
        lines.append('overall: pending')
    elif 'muon_layer' in flips:
        lines.append('overall: flip both defaults (the combination is the recipe)')
    else:
        lines.append(f"overall: {'flip ' + ' and '.join(flips) if flips else 'keep both defaults'}")
    return lines


def fmt(value, digits=2, signed=True):
    return '-' if value is None else f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def markdown(evals, effects, speed, times, late, leads_12_epochs, criteria, lines):
    out = ['# muon_layernorm_100ep summary (generated by scripts/eval/muon_layernorm_100ep_summary.py)', '',
           '## Final scores (last_model.pt, full val set, 2 mask draws)', '',
           '| run | epochs | PSNR_M all | trainer ep 100 | SSIM_M all | brain | knee-PD | knee-PDFS | SSIM brain | SSIM knee-PD | SSIM knee-PDFS |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for name, row in sorted(evals['per_run'].items()):
        arm, seed = name.rsplit('_s', 1)
        out.append(f"| `{name}` | {row['epochs']} | {row['all']['psnr_masked']:.3f} | {late[arm][seed]['final_psnr']:.3f} | {row['all']['ssim_masked']:.4f} | "
                   + ' | '.join(f"{row[c]['psnr_masked']:.2f}" if c in row else '-' for c in CONDITIONS) + ' | '
                   + ' | '.join(f"{row[c]['ssim_masked']:.3f}" if c in row else '-' for c in CONDITIONS) + ' |')
    out += ['', '## Paired arm - control, same seed (PSNR_Masked dB, 95% CI over volumes)', '',
            '| arm | seed | all | brain | knee-PD | knee-PDFS | SSIM_M |', '|---|---|---|---|---|---|---|']
    for arm, seeds in sorted(evals['paired'].items()):
        for seed, entry in sorted(seeds.items()):
            psnr = entry['all']['psnr']
            out.append(f"| {arm} | {seed} | {psnr['mean']:+.3f} [{psnr['ci'][0]:+.2f}, {psnr['ci'][1]:+.2f}] | "
                       + ' | '.join(f"{entry[c]['psnr']['mean']:+.2f}" if c in entry else '-' for c in CONDITIONS)
                       + f" | {entry['all']['ssim']['mean']:+.4f} |")
    out += ['', '## Factorial effects (PSNR_Masked dB)', '',
            '| factor | difference | seed | mean [95% CI] | brain | knee-PD | knee-PDFS |', '|---|---|---|---|---|---|---|']
    for factor in ('muon', 'layer'):
        for row in effects[factor]['differences']:
            out.append(f"| {factor} | {row['arm']} - {row['other']} | {row['seed']} | {row['mean']:+.3f} [{row['ci'][0]:+.2f}, {row['ci'][1]:+.2f}] | "
                       + ' | '.join(f"{row['per_condition'][c]:+.2f}" for c in CONDITIONS) + ' |')
        for level, value in effects[factor]['by_level'].items():
            out.append(f'| {factor} | mean of {level} | | {value:+.3f} | | | |')
        out.append(f"| {factor} | **mean of {len(effects[factor]['differences'])}** | | **{fmt(effects[factor]['mean'], 3)}** | | | |")
    for seed, value in sorted(effects['interaction'].items()):
        out.append(f'| interaction | (muon_layer - layer) - (muon - control) | {seed} | {value:+.3f} | | | |')
    if effects['interaction']:
        out.append(f"| interaction | mean over seeds | | {statistics.mean(effects['interaction'].values()):+.3f} | | | |")
    out += ['', '## Time to the control\'s final (trainer val PSNR_Masked, full val set)', '',
            '| arm | seed | target (control ep 96-100) | reach epoch (trailing 3) | median epoch s | control median s | wall fraction | hours to target | control hours, 100 epochs |',
            '|---|---|---|---|---|---|---|---|---|']
    for arm, seeds in sorted(speed.items()):
        for seed, row in sorted(seeds.items()):
            out.append(f"| {arm} | {seed} | {row['target_psnr_masked']:.3f} | {row['reach_epoch'] or 'not reached'} | "
                       f"{fmt(row['median_epoch_s'], 1, False)} | {fmt(row['control_median_epoch_s'], 1, False)} | {fmt(row['wall_fraction'], 3, False)} | "
                       f"{fmt(row['hours_to_target'], 2, False)} | {fmt(row['control_hours_100_epochs'], 2, False)} |")
    out += ['', '## Median epoch time (s, epoch 1 excluded; change against the same-seed control)', '',
            '| run | train | val | epoch | train change | epoch change |', '|---|---|---|---|---|---|']
    for group in ('runs', 'reference_12_epochs'):
        for name, row in times[group].items():
            out.append(f"| `{name}` | {fmt(row['train_s'], 1, False)} | {fmt(row['val_s'], 1, False)} | {fmt(row['epoch_s'], 1, False)} | "
                       f"{row['train_s_vs_control']:+.1%} | {row['epoch_s_vs_control']:+.1%} |")
    out += ['', '## Lead at epoch 12 of the 12-epoch full-data runs (trainer val PSNR_Masked, all val slices, dB)', '',
            '| run | lead over the same-seed control |', '|---|---|']
    out += [f'| `{name}` | {value:+.3f} |' for name, value in leads_12_epochs.items()]
    windows = [f'{lo}-{hi}' for lo, hi in LATE_WINDOWS]
    out += ['', '## Late training (trainer val PSNR_Masked: window mean, then lead over the same-seed control)', '',
            '| arm | seed | ' + ' | '.join(windows) + ' | ' + ' | '.join(f'lead {w}' for w in windows) + ' | own gain 66-70 to 96-100 | best epoch | best | final |',
            '|---|---|' + '---|' * (2 * len(windows) + 4)]
    for arm, seeds in late.items():
        for seed, row in seeds.items():
            out.append(f"| {arm} | {seed} | " + ' | '.join(f"{row['window_mean'][w]:.3f}" for w in windows) + ' | '
                       + ' | '.join(f"{row['window_lead'][w]:+.3f}" for w in windows)
                       + f" | {row['own_gain_66_to_96']:+.3f} | {row['best_epoch']} | {row['best_psnr']:.3f} | {row['final_psnr']:.3f} |")
    out += ['', '## Lead over the same-seed control by epoch (trainer val PSNR_Masked, dB)', '',
            '| arm | seed | ' + ' | '.join(str(epoch) for epoch in LEAD_EPOCHS) + ' |', '|---|---|' + '---|' * len(LEAD_EPOCHS)]
    for arm in ARMS:
        for seed, row in late[arm].items():
            out.append(f"| {arm} | {seed} | " + ' | '.join(f"{row['lead_at_epoch'][epoch]:+.2f}" for epoch in LEAD_EPOCHS) + ' |')
    out += ['', '## Frozen criteria', '', '| criterion | value | status |', '|---|---|---|']
    for key, row in criteria.items():
        value = {k: v for k, v in row.items() if k != 'passed'}
        out.append(f"| {key} | `{json.dumps(value, default=float)[:160]}` | {verdict(row)} |")
    out += ['', '## Mechanical recommendation (the user decides)', '', *[f'- {line}' for line in lines], '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report_dir', type=Path)
    parser.add_argument('--queue-dir', type=Path, default=QUEUE_DIR)
    parser.add_argument('--reference-curves', type=Path, default=REFERENCE_CURVES)
    args = parser.parse_args()
    evals = json.loads((args.report_dir / 'eval.json').read_text())
    scores = Scores(args.report_dir / 'eval.per_slice.npz')
    curves = json.loads((args.report_dir / 'curves.json').read_text())
    effects = factor_effects(scores)
    speed = time_to_target(curves)
    reference = json.loads(args.reference_curves.read_text())
    times = timing(curves, reference)
    leads_12_epochs = reference_leads(reference)
    late = trajectory(curves)
    criteria = decide(scores, effects, speed, evals, args.queue_dir.expanduser())
    lines = recommendation(criteria, effects)
    (args.report_dir / 'criteria.json').write_text(json.dumps(
        {'effects': effects, 'time_to_target': speed, 'timing': times, 'reference_12_epochs_leads': leads_12_epochs,
         'trajectory': late, 'criteria': criteria, 'recommendation': lines}, indent=1, default=float))
    (args.report_dir / 'summary.md').write_text(markdown(evals, effects, speed, times, late, leads_12_epochs, criteria, lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
