"""Tables, frozen criteria and figures of reports/brain_varnet_width from the files of brain_varnet_width_eval.py.

Per run: full-val PSNR / SSIM / NMSE at best_model.pt (the brain-val peak) and last_model.pt, the train - val gap at
the peak, parameters, median seconds per step and the step training stopped at. Then, per proposal.md:
  seed43_picks  3 x 36 and 6 x 26, plus the two best new seed-42 cells (not 6 x 18 or 12 x 18)
  q1            per cascade count: best of 26 / 36 channels minus 18, and 12 minus 18, paired per slice, per seed
  q2            at about 30 M parameters: 6 x 26 - 3 x 36 and 12 x 18 - 3 x 36 (and 12 x 18 - 6 x 26), per seed
  diagonals     cells of each equal-size group (about 15 / 30 / 60 M) and their PSNR spread
CIs are 95% bootstrap over val volumes.

Writes <eval>/summary.json and <figures>/full_val_vs_params.png, full_val_grid.png.

    python scripts/eval/brain_varnet_width_summary.py
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

import _path  # noqa: F401

EVAL = Path('reports/brain_varnet_width/eval')
FIGURES = Path('reports/brain_varnet_width/figures')
QUEUES = (Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_width'),
          Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_scaling'))
REUSED = {(6, 18): 'c6_v80_s{seed}', (12, 18): 'c12_v80_s{seed}'}
CASCADES = (3, 6, 12)
CHANS = (12, 18, 26, 36)
NEW = [(c, k) for c in CASCADES for k in CHANS if (c, k) not in REUSED]
DIAGONALS = {'15M': [(3, 26), (6, 18), (12, 12)], '30M': [(3, 36), (6, 26), (12, 18)], '60M': [(6, 36), (12, 26)]}
STEPS_PER_EPOCH = 76
CASCADE_COLOURS = {3: '#7fb6dc', 6: '#2b78b8', 12: '#08306b'}
FILE = re.compile(r'^(val|train)_(best_model|last_model)_c(\d+)_ch(\d+)_s(\d+)\.pt$')


def load_scores(folder):
    scores = {}
    for path in folder.glob('*.pt'):
        m = FILE.match(path.name)
        if m:
            saved = torch.load(path, weights_only=False)
            scores[m[1], m[2], int(m[3]), int(m[4]), int(m[5])] = (
                {k: v.numpy() for k, v in saved['scores'].items()}, np.array(saved['volume']))
    return scores


def run_dir(c, k, seed):
    if (c, k) in REUSED:
        return QUEUES[1] / REUSED[c, k].format(seed=seed)
    return QUEUES[0] / f'c{c}_ch{k}_s{seed}'


def run_cost(folder):
    """Parameters, median seconds per step (epoch 1 skipped) and the step training stopped at."""
    out = {}
    if (folder / 'best_model.pt').exists():
        state = torch.load(folder / 'best_model.pt', map_location='cpu', weights_only=False)['model']
        out['parameters_m'] = sum(t.numel() for t in state.values() if torch.is_tensor(t)) / 1e6
    log = folder / 'train_sequential.log'
    if log.exists():
        text = log.read_text()
        times = [float(t) for t in re.findall(r'TrainTime = ([\d.]+)s', text)]
        epochs = [int(e) for e in re.findall(r'Epoch = \[\s*(\d+)/', text)]
        if len(times) > 1:
            out['seconds_per_step'] = float(np.median(times[1:]) / STEPS_PER_EPOCH)
        if epochs:
            out['stopped_at_step'] = epochs[-1] * STEPS_PER_EPOCH
        out['skipped_steps'] = text.count('Skipping optimizer step')
    return out


def volume_ci(diff, volumes, n=10000):
    rng = np.random.default_rng(0)
    names = np.unique(volumes)
    means = np.array([diff[volumes == v].mean() for v in names])
    counts = np.array([(volumes == v).sum() for v in names])
    draws = rng.integers(0, len(names), (n, len(names)))
    boot = (means[draws] * counts[draws]).sum(1) / counts[draws].sum(1)
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def paired(scores, a, b, seed, ckpt='best_model'):
    """a minus b (each a (cascades, chans) cell) in full-val PSNR, paired per slice, or None if either is missing."""
    ka, kb = ('val', ckpt, *a, seed), ('val', ckpt, *b, seed)
    if ka not in scores or kb not in scores:
        return None
    (sa, volumes), (sb, _) = scores[ka], scores[kb]
    diff = sa['psnr'] - sb['psnr']
    return {'diff': float(diff.mean()), 'ci': volume_ci(diff, volumes)}


def q1_verdict(q1):
    wider = [d for d in q1.values() if d.get('wider_minus_18')]
    narrow = [d for d in q1.values() if d.get('12_minus_18')]
    verdicts = []
    if wider:
        by_cell = {}
        for d in wider:
            by_cell.setdefault((d['cascades'], d['best_wider']), []).append(d['wider_minus_18'])
        two_seed = {cell: v for cell, v in by_cell.items() if len(v) >= 2}
        if any(all(x['diff'] >= 0.15 and x['ci'][0] > 0 for x in v) for v in two_seed.values()):
            verdicts.append('wider helps')
        elif two_seed and all(all(abs(x['diff']) < 0.05 for x in v) for v in two_seed.values()):
            verdicts.append('width saturated')
    if narrow and all(d['12_minus_18']['diff'] <= -0.15 for d in narrow if d['cascades'] in (6, 12)):
        verdicts.append('18 needed')
    return verdicts or ['inconclusive']


def q2_verdict(q2):
    seeds = [s for s, d in q2.items() if d.get('6x26_minus_3x36') and d.get('12x18_minus_3x36')]
    if len(seeds) < 2:
        return 'needs both seeds'
    pairs = [(q2[s]['6x26_minus_3x36']['diff'], q2[s]['12x18_minus_3x36']['diff'], q2[s]['12x18_minus_6x26']['diff'])
             for s in seeds]
    if all(a >= 0.15 and b >= 0.15 for a, b, _ in pairs):
        return 'depth wins'
    if all(a <= -0.15 and b <= -0.15 for a, b, _ in pairs):
        return 'width wins'
    if all(max(abs(a), abs(b), abs(c)) < 0.05 for a, b, c in pairs):
        return 'size alone'
    return 'inconclusive'


def plot_vs_params(table, seed, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for c in CASCADES:
        pts = sorted((r['parameters_m'], r['best_psnr'], r['chans']) for r in table
                     if r['cascades'] == c and r['seed'] == seed and 'parameters_m' in r)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=CASCADE_COLOURS[c], linewidth=2, markersize=7,
                label=f'{c} cascades')
        for p, y, k in pts:
            ax.annotate(f'{k}', (p, y), textcoords='offset points', xytext=(5, -10), fontsize=7, color='#555')
    for name, cells in DIAGONALS.items():
        pts = [next((r for r in table if (r['cascades'], r['chans']) == cell and r['seed'] == seed
                     and 'parameters_m' in r), None) for cell in cells]
        pts = [p for p in pts if p]
        if len(pts) > 1:
            ax.plot([p['parameters_m'] for p in pts], [p['best_psnr'] for p in pts], ':', color='#999', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('parameters (M, log); labels = channels')
    ax.set_ylabel('full brain val PSNR at peak checkpoint (dB)')
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f'seed {seed}; 80 volumes; dotted = equal-size groups', fontsize=9, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(table, seed, path):
    grid = np.full((len(CHANS), len(CASCADES)), np.nan)
    for r in table:
        if r['seed'] == seed:
            grid[CHANS.index(r['chans']), CASCADES.index(r['cascades'])] = r['best_psnr']
    fig, ax = plt.subplots(figsize=(4.6, 4))
    image = ax.imshow(grid, cmap='Blues', origin='lower', aspect='auto')
    for i in range(len(CHANS)):
        for j in range(len(CASCADES)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if grid[i, j] > np.nanmean(grid) else '#222')
    ax.set_xticks(range(len(CASCADES)), [str(c) for c in CASCADES])
    ax.set_yticks(range(len(CHANS)), [str(k) for k in CHANS])
    ax.set_xlabel('cascades')
    ax.set_ylabel('channels')
    fig.colorbar(image, ax=ax, label='PSNR (dB)')
    ax.set_title(f'seed {seed}; full brain val PSNR at peak', fontsize=9, color='#666')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, default=EVAL)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    args = parser.parse_args()
    scores = load_scores(args.eval)
    runs = sorted({(c, k, s) for (split, ckpt, c, k, s) in scores if split == 'val' and ckpt == 'best_model'})
    table = []
    for c, k, s in runs:
        best, _ = scores['val', 'best_model', c, k, s]
        last, _ = scores['val', 'last_model', c, k, s]
        row = {'cascades': c, 'chans': k, 'seed': s, 'reused': (c, k) in REUSED,
               **{f'best_{m}': float(best[m].mean()) for m in ('psnr', 'ssim', 'nmse')},
               **{f'last_{m}': float(last[m].mean()) for m in ('psnr', 'ssim', 'nmse')},
               **run_cost(run_dir(c, k, s))}
        if ('train', 'best_model', c, k, s) in scores:
            row['gap_best'] = float(scores['train', 'best_model', c, k, s][0]['psnr'].mean() - row['best_psnr'])
            row['gap_last'] = float(scores['train', 'last_model', c, k, s][0]['psnr'].mean() - row['last_psnr'])
        table.append(row)
    seeds = sorted({r['seed'] for r in table})
    summary = {'runs': table, 'ranking': {}, 'seed43_picks': [], 'q1': {}, 'q2': {}, 'diagonals': {}}

    for s in seeds:
        rows = sorted((r for r in table if r['seed'] == s), key=lambda r: -r['best_psnr'])
        summary['ranking'][f's{s}'] = [(r['cascades'], r['chans'], round(r['best_psnr'], 3)) for r in rows]
    new42 = [(r['cascades'], r['chans']) for r in sorted(table, key=lambda r: -r['best_psnr'])
             if r['seed'] == 42 and (r['cascades'], r['chans']) in NEW]
    if len(new42) == len(NEW):
        picks = [(3, 36), (6, 26)]
        picks += [cell for cell in new42[:2] if cell not in picks]
        summary['seed43_picks'] = picks

    q1 = {}
    for s in seeds:
        for c in CASCADES:
            entry = {'cascades': c, 'seed': s}
            wider = {k: paired(scores, (c, k), (c, 18), s) for k in (26, 36)}
            wider = {k: v for k, v in wider.items() if v}
            if wider:
                k = max(wider, key=lambda k: wider[k]['diff'])
                entry.update(best_wider=k, wider_minus_18=wider[k], every_wider_minus_18=wider)
            narrow = paired(scores, (c, 12), (c, 18), s)
            if narrow:
                entry['12_minus_18'] = narrow
            if len(entry) > 2:
                q1[f'c{c}_s{s}'] = entry
    summary['q1'] = {**q1, 'verdict': q1_verdict(q1)}

    q2 = {}
    for s in seeds:
        q2[s] = {'6x26_minus_3x36': paired(scores, (6, 26), (3, 36), s),
                 '12x18_minus_3x36': paired(scores, (12, 18), (3, 36), s),
                 '12x18_minus_6x26': paired(scores, (12, 18), (6, 26), s)}
    summary['q2'] = {**{f's{s}': d for s, d in q2.items()}, 'verdict': q2_verdict(q2)}

    for s in seeds:
        for name, cells in DIAGONALS.items():
            values = {f'{c}x{k}': r['best_psnr'] for (c, k) in cells for r in table
                      if (r['cascades'], r['chans'], r['seed']) == (c, k, s)}
            if values:
                summary['diagonals'][f'{name}_s{s}'] = {'psnr': values, 'spread': max(values.values()) - min(values.values())}

    for r in sorted(table, key=lambda r: (r['seed'], r['cascades'], r['chans'])):
        gap = f"  gap {r['gap_best']:.2f}" if 'gap_best' in r else ''
        cost = (f"  {r.get('parameters_m', float('nan')):.1f} M  {r.get('seconds_per_step', float('nan')):.2f} s/step"
                f"  stop {r.get('stopped_at_step', '?')}  skips {r.get('skipped_steps', '?')}")
        print(f"s{r['seed']} {r['cascades']:>2d}x{r['chans']:<2d}  best {r['best_psnr']:.3f} (SSIM {r['best_ssim']:.4f})"
              f"  last {r['last_psnr']:.3f}{gap}{cost}")
    print(json.dumps({k: summary[k] for k in ('ranking', 'seed43_picks', 'q1', 'q2', 'diagonals')}, indent=1))
    (args.eval / 'summary.json').write_text(json.dumps(summary, indent=1))

    args.figures.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        suffix = '' if s == 42 else f'_s{s}'
        plot_vs_params(table, s, args.figures / f'full_val_vs_params{suffix}.png')
        plot_grid(table, s, args.figures / f'full_val_grid{suffix}.png')


if __name__ == '__main__':
    main()
