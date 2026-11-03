"""Tables for reports/varnet_train_speed/report.md from the study's result files.

Inputs (in the report folder): bench.json (scripts/diagnostics/varnet_train_speed_bench.py), curves.json
(scripts/eval/varnet_train_speed_curves.py), eval.json / eval_vs_short.json / eval_vs_10ep.json
(scripts/eval/varnet_train_speed_eval.py with --control control / control_short / control_10ep), eval_vs_full.json
(--control full_control: the full-data runs). Writes summary.md.

    python scripts/eval/varnet_train_speed_summary.py reports/varnet_train_speed
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CONDITIONS = ('all', 'brain', 'knee-PD', 'knee-PDFS')


def load(folder, name):
    path = folder / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(x, digits=2, sign=False):
    if x is None:
        return 'n/a'
    return f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'


def bench_table(bench):
    rows = {r['variant']: r for r in bench if 'step_s' in r}
    control, fused = rows.get('control'), rows.get('fused')
    lines = ['| variant | step (ms) | vs control | vs fused | val pass (s) | peak GiB |', '|---|---|---|---|---|---|']
    for name, r in rows.items():
        vs_control = 100 * (r['step_s'] / control['step_s'] - 1) if control else None
        vs_fused = 100 * (r['step_s'] / fused['step_s'] - 1) if fused else None
        lines.append(f"| {name} | {1000 * r['step_s']:.0f} | {fmt(vs_control, 1, True)}% | {fmt(vs_fused, 1, True)}% | "
                     f"{r['val_pass_s']:.1f} | {r['peak_GiB']:.1f} |")
    return '\n'.join(lines)


def epoch_time_table(curves):
    by_arm = defaultdict(list)
    for name, run in curves['runs'].items():
        times = [c for c in run['curve'] if 'train_time' in c and c['epoch'] > 1]
        if times:
            by_arm[run['arm']].append((statistics.median(c['train_time'] for c in times),
                                       statistics.median(c['val_time'] for c in times),
                                       statistics.median(c['train_time'] + c['val_time'] for c in times)))
    lines = ['| arm | runs | train s/epoch | val s/epoch | train+val s/epoch |', '|---|---|---|---|---|']
    for arm, rows in sorted(by_arm.items()):
        lines.append(f"| {arm} | {len(rows)} | {statistics.mean(r[0] for r in rows):.1f} | "
                     f"{statistics.mean(r[1] for r in rows):.1f} | {statistics.mean(r[2] for r in rows):.1f} |")
    return '\n'.join(lines)


def time_to_quality_table(curves):
    """First epoch at which an arm's logged val PSNR_Masked reaches its same-seed control's last value, and the wall-clock
    (train + val seconds from the run log, summed to that epoch) as a share of the control's whole run. Full-data arms
    are compared with full_control, the others with control."""
    runs = curves['runs']
    lines = ['| arm | seed | control | control final PSNR | epoch reached / total | wall-clock to reach vs control run |',
             '|---|---|---|---|---|---|']
    for name, run in sorted(runs.items()):
        arm, seed = run['arm'], run['seed']
        control_arm = 'full_control' if arm.startswith('full_') else 'control'
        control = runs.get(f'{control_arm}_s{seed}')
        if arm == control_arm or control is None or not run['curve'] or 'num_epochs' not in run['curve'][-1]:
            continue
        if run['curve'][-1]['epoch'] != run['curve'][-1]['num_epochs']:
            continue
        target = control['curve'][-1].get('val/PSNR_Masked')
        control_seconds = sum(c['train_time'] + c['val_time'] for c in control['curve'] if 'train_time' in c)
        elapsed, reached = 0.0, None
        for c in run['curve']:
            elapsed += c.get('train_time', 0) + c.get('val_time', 0)
            if c.get('val/PSNR_Masked', float('-inf')) >= target:
                reached = (c['epoch'], elapsed)
                break
        total = run['curve'][-1]['num_epochs']
        cell = f'{reached[0]} / {total}' if reached else f'not reached / {total}'
        share = f'{100 * reached[1] / control_seconds:.0f}%' if reached else 'n/a'
        lines.append(f'| {arm} | {seed} | {control_arm} | {target:.2f} | {cell} | {share} |')
    return '\n'.join(lines)


def final_table(evaluation):
    by_arm = defaultdict(dict)
    for name, row in evaluation['per_run'].items():
        arm, seed = name.rsplit('_s', 1)
        by_arm[arm][seed] = row
    lines = ['| arm | epochs | seeds | ' + ' | '.join(f'{c} PSNR / SSIM' for c in CONDITIONS) + ' |',
             '|---|---|---|' + '---|' * len(CONDITIONS)]
    for arm, seeds in sorted(by_arm.items()):
        cells = []
        for c in CONDITIONS:
            psnr = [s[c]['psnr_masked'] for s in seeds.values() if c in s]
            ssim = [s[c]['ssim_masked'] for s in seeds.values() if c in s]
            if not psnr:
                cells.append('n/a')
                continue
            spread = f' [{min(psnr):.2f}, {max(psnr):.2f}]' if len(psnr) > 1 else ''
            cells.append(f'{statistics.mean(psnr):.2f}{spread} / {statistics.mean(ssim):.3f}')
        epochs = sorted({s['epochs'] for s in seeds.values()})
        lines.append(f"| {arm} | {','.join(map(str, epochs))} | {','.join(sorted(seeds))} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def paired_table(evaluation, control_name):
    lines = [f'| arm − {control_name} (same seed) | seed | ' + ' | '.join(f'{c} ΔPSNR [95% CI]' for c in CONDITIONS) + ' | all ΔSSIM |',
             '|---|---|' + '---|' * len(CONDITIONS) + '---|']
    means = {}
    for arm, seeds in sorted(evaluation['paired'].items()):
        for seed, entry in sorted(seeds.items()):
            cells = [f"{fmt(entry[c]['psnr']['mean'], 3, True)} [{fmt(entry[c]['psnr']['ci'][0], 2, True)}, "
                     f"{fmt(entry[c]['psnr']['ci'][1], 2, True)}]" if c in entry else 'n/a' for c in CONDITIONS]
            lines.append(f"| {arm} | {seed} | " + ' | '.join(cells) + f" | {fmt(entry['all']['ssim']['mean'], 4, True)} |")
        all_deltas = [e['all']['psnr']['mean'] for e in seeds.values()]
        means[arm] = (statistics.mean(all_deltas), min(all_deltas), max(all_deltas), len(all_deltas))
    lines += ['', f'| arm | seeds | mean ΔPSNR (all) vs {control_name} | min | max |', '|---|---|---|---|---|']
    for arm, (mean, low, high, n) in sorted(means.items()):
        lines.append(f'| {arm} | {n} | {mean:+.3f} | {low:+.3f} | {high:+.3f} |')
    return '\n'.join(lines)


def main():
    folder = Path(sys.argv[1])
    parts = ['# varnet_train_speed summary tables (generated by scripts/eval/varnet_train_speed_summary.py)', '']
    bench = load(folder, 'bench.json')
    if bench:
        parts += ['## Step time, one idle A6000 (bench.json)', '', bench_table(bench), '']
    curves = load(folder, 'curves.json')
    if curves:
        parts += ['## Epoch times in the training runs (curves.json; median over epochs 2+)', '', epoch_time_table(curves), '']
        parts += ['## Time to the control final quality (curves.json; trainer val PSNR_Masked per epoch)', '',
                  time_to_quality_table(curves), '']
    for name, control in (('eval.json', 'control'), ('eval_vs_short.json', 'control_short'), ('eval_vs_10ep.json', 'control_10ep'),
                          ('eval_vs_full.json', 'full_control')):
        evaluation = load(folder, name)
        if not evaluation:
            continue
        if name == 'eval.json':
            parts += ['## Final validation scores per arm (eval.json; last_model.pt, 2 mask draws)', '', final_table(evaluation), '']
        parts += [f'## Paired against {control} ({name})', '', paired_table(evaluation, control), '']
    (folder / 'summary.md').write_text('\n'.join(parts) + '\n')
    print((folder / 'summary.md').read_text())


if __name__ == '__main__':
    main()
