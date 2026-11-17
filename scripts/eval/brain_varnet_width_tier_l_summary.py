"""Tables and frozen criteria of the tier-L addendum (reports/brain_varnet_width/proposal_tier_l.md).

Inputs are brain_varnet_width_eval.py score files:
  eval_L     the 8 CoreHPC tier-L runs ({6, 12} cascades x {18, 26} channels, seeds 42 and 43)
  eval_hpcS  the 2 CoreHPC tier-S 6 x 18 runs (platform control)
  eval       the local tier-S runs of the original grid
Per run: full-val PSNR / SSIM / NMSE at best_model.pt and last_model.pt, train - val gap on the 80 tier-S volumes,
parameters, median seconds per epoch, the epoch training stopped at, and the trainer-val loss at epoch 25 (the
equal-steps readout). Then validity, platform control, Q3 (capacity at L), Q4 (data gain), the exploratory
gap-growth table and the frozen next-step pick, and the platform / Q3 / cell means repeated at last_model.pt (robustness). Differences are paired per slice; CIs are 95% bootstrap over volumes.

Writes <eval_L>/summary.json and <figures>/tier_l_cells.png.

    python scripts/eval/brain_varnet_width_tier_l_summary.py
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from brain_varnet_width_summary import load_scores, run_dir, volume_ci

REPORT = Path('reports/brain_varnet_width')
QUEUE = Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_width_l')
SEEDS = (42, 43)
BASE = (6, 18)
CAPACITY = [(6, 26), (12, 18), (12, 26)]
CELLS = [BASE, *CAPACITY]
EQUAL_STEPS_EPOCH = 25
PLATFORM_TOL, PAYS, SATURATED, DATA_PAYS, DATA_SATURATED, NEXT_TOL = 0.10, 0.15, 0.05, 0.3, 0.1, 0.05


def log_stats(folder):
    log = folder / 'train_sequential.log'
    state = torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=False)['model']
    out = {'parameters_m': sum(t.numel() for t in state.values() if torch.is_tensor(t)) / 1e6}
    if not log.exists():
        return out
    text = log.read_text()
    epochs = re.findall(r'Epoch = \[\s*(\d+)/\s*\d+\] TrainLoss = (\S+) ValLoss = (\S+) .*?TrainTime = ([\d.]+)s', text)
    times = [float(t) for _, _, _, t in epochs]
    val = {int(e): float(v) for e, _, v, _ in epochs if v != 'nan'}
    out.update(stopped_at_epoch=int(epochs[-1][0]), seconds_per_epoch=float(np.median(times[1:])),
               train_hours=sum(times) / 3600, best_trainer_val=min(val.values()),
               trainer_val_epoch25=val.get(EQUAL_STEPS_EPOCH),
               nonfinite_train=sum(not np.isfinite(float(t)) for _, t, _, _ in epochs),
               skipped_steps=text.count('Skipping optimizer step'))
    return out


def paired(a, b):
    """Mean of a - b over slices with a volume CI; a and b are (scores, volumes) of the same slices."""
    (sa, volumes), (sb, volumes_b) = a, b
    assert (volumes == volumes_b).all()
    diff = sa['psnr'] - sb['psnr']
    return {'diff': float(diff.mean()), 'ci': volume_ci(diff, volumes)}


def plot_cells(table, local, path):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    x = np.arange(len(CELLS))
    for offset, tier, colour in ((-0.12, 'S', '#9ecae1'), (0.12, 'L', '#08519c')):
        for s, marker in zip(SEEDS, ('o', 's')):
            y = [(table if tier == 'L' else local).get((c, k, s), np.nan) for c, k in CELLS]
            ax.plot(x + offset, y, marker, color=colour, markersize=7, linestyle='none',
                    label=f'tier {tier}, seed {s}')
    ax.set_xticks(x, [f'{c}x{k}' for c, k in CELLS])
    ax.set_ylabel('full brain val PSNR at peak checkpoint (dB)')
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title('tier S = local A6000 runs (80 vols); tier L = CoreHPC (300 vols)', fontsize=9, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    L, hpcS, localS = (load_scores(args.report / d) for d in ('eval_L', 'eval_hpcS', 'eval'))

    rows = []
    for tier, scores, queue in (('L', L, QUEUE / 'runs_L'), ('S_hpc', hpcS, QUEUE / 'runs_S')):
        for (split, ckpt, c, k, s) in sorted(scores):
            if split != 'val' or ckpt != 'best_model':
                continue
            best, last = scores['val', 'best_model', c, k, s][0], scores['val', 'last_model', c, k, s][0]
            row = {'tier': tier, 'cascades': c, 'chans': k, 'seed': s,
                   **{f'best_{m}': float(best[m].mean()) for m in ('psnr', 'ssim', 'nmse')},
                   **{f'last_{m}': float(last[m].mean()) for m in ('psnr', 'ssim', 'nmse')},
                   **log_stats(queue / f'c{c}_ch{k}_s{s}')}
            if ('train', 'best_model', c, k, s) in scores:
                row['gap_best'] = float(scores['train', 'best_model', c, k, s][0]['psnr'].mean() - row['best_psnr'])
            rows.append(row)
    local_rows = {(c, k, s): float(localS['val', 'best_model', c, k, s][0]['psnr'].mean())
                  for (split, ckpt, c, k, s) in localS if split == 'val' and ckpt == 'best_model'}
    local_best_trainer_val = {f'{c}x{k}_s{s}': log_stats(run_dir(c, k, s)).get('best_trainer_val')
                     for c, k in CELLS for s in SEEDS if (c, k, s) in local_rows}

    validity = {'all_runs': len(rows) == 10,
                'min_ssim': min(r['best_ssim'] for r in rows),
                'nonfinite_or_skipped': sum(r.get('nonfinite_train', 0) + r.get('skipped_steps', 0) for r in rows)}
    validity['met'] = validity['all_runs'] and validity['min_ssim'] >= 0.80 and validity['nonfinite_or_skipped'] == 0

    platform = {}
    for s in SEEDS:
        d = paired(hpcS['val', 'best_model', *BASE, s], localS['val', 'best_model', *BASE, s])
        platform[f's{s}'] = {**d, 'hpc': float(hpcS['val', 'best_model', *BASE, s][0]['psnr'].mean()),
                             'local': local_rows[(*BASE, s)]}
    platform['met'] = all(abs(platform[f's{s}']['diff']) <= PLATFORM_TOL for s in SEEDS)

    q3 = {f'{c}x{k}': {f's{s}': paired(L['val', 'best_model', c, k, s], L['val', 'best_model', *BASE, s])
                       for s in SEEDS} for c, k in CAPACITY}
    for d in q3.values():
        d['mean'] = float(np.mean([d[f's{s}']['diff'] for s in SEEDS]))
    best_x = max(q3, key=lambda x: q3[x]['mean'])
    if all(q3[best_x][f's{s}']['diff'] >= PAYS for s in SEEDS):
        q3_verdict = 'capacity pays'
    elif all(abs(d[f's{s}']['diff']) < SATURATED for d in q3.values() for s in SEEDS):
        q3_verdict = 'still saturated'
    else:
        q3_verdict = 'inconclusive'
    q3.update(best=best_x, verdict=q3_verdict)

    q4 = {f's{s}': paired(L['val', 'best_model', *BASE, s], hpcS['val', 'best_model', *BASE, s]) for s in SEEDS}
    diffs = [q4[f's{s}']['diff'] for s in SEEDS]
    q4['verdict'] = ('data still pays' if all(d >= DATA_PAYS for d in diffs) else
                     'data saturated' if all(d < DATA_SATURATED for d in diffs) else 'in between')

    gap_growth = {}
    for c, k in CAPACITY:
        for s in SEEDS:
            if (c, k, s) in local_rows:
                at_l = q3[f'{c}x{k}'][f's{s}']['diff']
                at_s = paired(localS['val', 'best_model', c, k, s], localS['val', 'best_model', *BASE, s])['diff']
                gap_growth[f'{c}x{k}_s{s}'] = {'at_L': at_l, 'at_S_local': at_s, 'growth': at_l - at_s}

    l_rows = [r for r in rows if r['tier'] == 'L']
    cells = {}
    for c, k in CELLS:
        rs = [r for r in l_rows if (r['cascades'], r['chans']) == (c, k)]
        cells[f'{c}x{k}'] = {'mean_psnr': float(np.mean([r['best_psnr'] for r in rs])),
                             'seconds_per_epoch': float(np.mean([r['seconds_per_epoch'] for r in rs])),
                             'train_hours': float(np.mean([r['train_hours'] for r in rs])),
                             'parameters_m': rs[0]['parameters_m']}
    top = max(c['mean_psnr'] for c in cells.values())
    eligible = [name for name, c in cells.items() if c['mean_psnr'] >= top - NEXT_TOL]
    next_step = {'best_mean': top, 'within_0.05': eligible,
                 'pick': min(eligible, key=lambda n: cells[n]['seconds_per_epoch'])}

    last_model = {
        'platform': {f's{s}': paired(hpcS['val', 'last_model', *BASE, s], localS['val', 'last_model', *BASE, s])
                     for s in SEEDS},
        'q3': {f'{c}x{k}': {f's{s}': paired(L['val', 'last_model', c, k, s], L['val', 'last_model', *BASE, s])
                            for s in SEEDS} for c, k in CAPACITY},
        'cells_L_mean_psnr': {f'{c}x{k}': float(np.mean([r['last_psnr'] for r in l_rows
                                                          if (r['cascades'], r['chans']) == (c, k)]))
                              for c, k in CELLS}}

    summary = {'runs': rows, 'last_model_robustness': last_model, 'local_S_best_psnr': {f'{c}x{k}_s{s}': v for (c, k, s), v in local_rows.items()},
               'validity': validity, 'platform': platform, 'q3': q3, 'q4': q4, 'gap_growth': gap_growth,
               'cells_L': cells, 'next_step': next_step,
               'equal_steps': {'L_trainer_val_epoch25': {f"{r['cascades']}x{r['chans']}_s{r['seed']}":
                                                         r['trainer_val_epoch25'] for r in l_rows},
                               'S_local_best_trainer_val': local_best_trainer_val}}

    for r in sorted(rows, key=lambda r: (r['tier'], r['seed'], r['cascades'], r['chans'])):
        gap = f"  gap {r['gap_best']:.2f}" if 'gap_best' in r else ''
        print(f"{r['tier']:5s} s{r['seed']} {r['cascades']:>2d}x{r['chans']:<2d}  best {r['best_psnr']:.3f} "
              f"(SSIM {r['best_ssim']:.4f})  last {r['last_psnr']:.3f}{gap}  {r['parameters_m']:.1f} M  "
              f"{r['seconds_per_epoch']:.0f} s/ep  stop ep {r['stopped_at_epoch']}  "
              f"tval25 {r['trainer_val_epoch25']}  tval best {r['best_trainer_val']:.5f}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs',)}, indent=1))
    (args.report / 'eval_L' / 'summary.json').write_text(json.dumps(summary, indent=1))
    (args.report / 'figures').mkdir(exist_ok=True)
    plot_cells({(r['cascades'], r['chans'], r['seed']): r['best_psnr'] for r in l_rows}, local_rows,
               args.report / 'figures' / 'tier_l_cells.png')


if __name__ == '__main__':
    main()
