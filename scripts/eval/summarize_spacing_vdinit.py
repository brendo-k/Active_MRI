"""Summarise the spacing-test / VD-init array (scripts/launch/hpc/spacing_vdinit_r8) from each run's training log.

Reads `<runs>/<arm>-<jobid>/train_sequential.log` (the newest job per arm), takes the last validation epoch's
per-anatomy NMSE / PSNR / SSIM, and prints:
  - one row per arm (flagging an unfinished run, or a val set whose slice count differs between runs)
  - the paired differences the experiment is about: VD-init - random-init (same seed), Poisson-disc - white
    (same mask seed), top-k - white
  - mean val PSNR by epoch for the LOUPE arms, to compare convergence speed

    python code/scripts/eval/summarize_spacing_vdinit.py --runs runs --out summary.json
"""
import argparse
import json
import re
from pathlib import Path

ARMS = ('random_s1', 'random_s2', 'vdinit_s1', 'vdinit_s2', 'pd_loupe_s0', 'pd_loupe_s1',
        'white_loupe_s0', 'white_loupe_s1', 'topk_loupe')
PAIRS = (('vdinit_s1', 'random_s1'), ('vdinit_s2', 'random_s2'), ('pd_loupe_s0', 'white_loupe_s0'),
         ('pd_loupe_s1', 'white_loupe_s1'), ('topk_loupe', 'white_loupe_s0'))
CONVERGENCE_EPOCHS = (1, 5, 10, 20, 30, 40, 50)
STATS = re.compile(r'\[(\w+)\] MSE=\S+ NMSE=(\S+) PSNR=(\S+) SSIM=(\S+) Sparsity=(\S+) \(n=(\d+)\)')
EPOCH = re.compile(r'Epoch = \[\s*(\d+)/\s*(\d+)\]')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, default=None, help='also write the summary as JSON')
    return parser.parse_args()


def parse_log(path):
    """[(epoch, total, {anatomy: metrics})]: the per-anatomy stats line is logged just before its epoch line."""
    epochs, pending = [], None
    for line in path.read_text(errors='replace').splitlines():
        stats = STATS.findall(line)
        if stats:
            pending = {anatomy: {'nmse': float(nmse), 'psnr': float(psnr), 'ssim': float(ssim),
                                 'sparsity': float(sparsity), 'n': int(n)}
                       for anatomy, nmse, psnr, ssim, sparsity, n in stats}
        match = EPOCH.search(line)
        if match and pending is not None:
            epochs.append((int(match.group(1)), int(match.group(2)), pending))
            pending = None
    return epochs


def newest_run(runs, arm):
    candidates = [d for d in runs.glob(f'{arm}-*') if (d / 'train_sequential.log').exists()]
    return max(candidates, key=lambda d: int(d.name.rsplit('-', 1)[1])) if candidates else None


def mean_psnr(metrics):
    return sum(m['psnr'] for m in metrics.values()) / len(metrics)


def main():
    args = parse_args()
    results = {}
    for arm in ARMS:
        run = newest_run(args.runs, arm)
        epochs = parse_log(run / 'train_sequential.log') if run else []
        if epochs:
            epoch, total, metrics = epochs[-1]
            results[arm] = {'run': str(run), 'epoch': epoch, 'total_epochs': total, 'final': metrics,
                            'psnr_by_epoch': {e: round(mean_psnr(m), 3) for e, _, m in epochs}}

    val_sizes = {tuple(sorted((a, m['n']) for a, m in r['final'].items())) for r in results.values()}
    print('| arm | epochs | brain PSNR / SSIM | knee PSNR / SSIM | mean PSNR | val slices |')
    print('|---|---|---|---|---|---|')
    for arm in ARMS:
        if arm not in results:
            print(f'| {arm} | not found | | | | |')
            continue
        r, final = results[arm], results[arm]['final']
        done = f"{r['epoch']}/{r['total_epochs']}" + ('' if r['epoch'] == r['total_epochs'] else ' (unfinished)')
        cells = [f"{final[a]['psnr']:.2f} / {final[a]['ssim']:.4f}" if a in final else '-' for a in ('brain', 'knee')]
        sizes = ', '.join(f"{a} {m['n']}" for a, m in final.items())
        print(f"| {arm} | {done} | {cells[0]} | {cells[1]} | {mean_psnr(final):.3f} | {sizes} |")
    if len(val_sizes) > 1:
        print('\nWARNING: the runs validated on different slice counts; the comparisons below are not paired.')

    print('\n| comparison | brain dPSNR | knee dPSNR | mean dPSNR | brain dSSIM | knee dSSIM |')
    print('|---|---|---|---|---|---|')
    differences = {}
    for a, b in PAIRS:
        if a in results and b in results:
            fa, fb = results[a]['final'], results[b]['final']
            row = {f'{k}_{anatomy}': fa[anatomy][k] - fb[anatomy][k] for anatomy in fa for k in ('psnr', 'ssim')}
            row['psnr_mean'] = mean_psnr(fa) - mean_psnr(fb)
            differences[f'{a} - {b}'] = row
            print(f"| {a} - {b} | {row['psnr_brain']:+.3f} | {row['psnr_knee']:+.3f} | {row['psnr_mean']:+.3f} | "
                  f"{row['ssim_brain']:+.4f} | {row['ssim_knee']:+.4f} |")

    loupe_arms = [arm for arm in ARMS[:4] if arm in results]
    if loupe_arms:
        print('\nmean val PSNR by epoch (LOUPE arms)\n')
        print('| arm | ' + ' | '.join(f'ep {e}' for e in CONVERGENCE_EPOCHS) + ' |')
        print('|---|' + '---|' * len(CONVERGENCE_EPOCHS))
        for arm in loupe_arms:
            by_epoch = results[arm]['psnr_by_epoch']
            print(f'| {arm} | ' + ' | '.join(f'{by_epoch[e]:.2f}' if e in by_epoch else '-' for e in CONVERGENCE_EPOCHS) + ' |')

    if args.out:
        args.out.write_text(json.dumps({'runs': results, 'differences': differences}, indent=1) + '\n')


if __name__ == '__main__':
    main()
