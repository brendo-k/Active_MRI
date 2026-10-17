"""Training dynamics of LOUPE-2D with vs without the gradient floor, from the full wandb history.

Per epoch: val PSNR/SSIM per anatomy (trainer metrics), LOUPE logit saturation (|slope * w| > log 99),
max |logit|, and per-step sampler gradient norm and global-norm-clip firing rate (GRAD_CLIP_NORM = 1).
Uses scan_history, which is unsampled.

    python scripts/plots/plot_grad_floor_dynamics.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import wandb

RUNS = {'floor 0.05': 'uo8d68n0', 'no floor': 'ijhnsmnm'}
PROJECT = 'SeqMRI'
VAL_KEYS = ['val/brain/PSNR', 'val/knee/PSNR', 'val/brain/SSIM', 'val/knee/SSIM', 'val/loss']
MASK_KEYS = ['mask/saturated_frac', 'mask/saturated_abs_mean', 'mask/logit_abs_max',
             'mask/saturated_moving_outward_frac']
STEP_KEYS = ['grad/sampler_norm', 'grad/total_norm', 'grad/clipped']


def load_history(api, run_id):
    run = api.run(f'{api.default_entity}/{PROJECT}/{run_id}')
    history = pd.DataFrame(list(run.scan_history(page_size=2000)))
    per_epoch = history.dropna(subset=['val/brain/PSNR']).set_index('epoch')[VAL_KEYS]
    mask = history.dropna(subset=['mask/saturated_frac'])[MASK_KEYS].reset_index(drop=True)
    mask.index = pd.Index(range(len(mask)), name='epoch')  # one row per epoch, logged before that epoch's val
    steps = history.dropna(subset=['grad/sampler_norm'])[['train/step'] + STEP_KEYS].reset_index(drop=True)
    return run, per_epoch.join(mask, how='inner'), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('reports/grad_floor_r8/dynamics.png'))
    args = parser.parse_args()
    api = wandb.Api()

    data = {label: load_history(api, run_id) for label, run_id in RUNS.items()}
    panels = [('val/brain/PSNR', 'val brain PSNR (dB)'), ('val/knee/PSNR', 'val knee PSNR (dB)'),
              ('val/brain/SSIM', 'val brain SSIM'), ('val/knee/SSIM', 'val knee SSIM'),
              ('mask/saturated_frac', 'saturated logit fraction'), ('mask/logit_abs_max', 'max |slope * w|'),
              ('mask/saturated_moving_outward_frac', 'saturated logits moving outward'), (None, 'sampler grad norm')]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for label, (_, per_epoch, steps) in data.items():
            if key is None:
                smoothed = steps.set_index('train/step')['grad/sampler_norm'].rolling(200, min_periods=1).median()
                ax.plot(smoothed.index, smoothed.values, label=label)
                ax.set(xlabel='train step', yscale='log')
            else:
                ax.plot(per_epoch.index, per_epoch[key], label=label)
                ax.set(xlabel='epoch')
        ax.set_title(title)
        ax.legend()
    fig.suptitle('LOUPE-2D R=8 brain+knee: gradient floor 0.05 vs none (seed 42)')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches='tight')

    summary = {}
    for label, (run, per_epoch, steps) in data.items():
        best_epoch = int(per_epoch['val/loss'].idxmin())
        final_epoch = int(per_epoch.index.max())
        last5 = per_epoch.loc[per_epoch.index > final_epoch - 5]
        summary[label] = {
            'run_id': run.id,
            'state': run.state,
            'epochs_logged': len(per_epoch),
            'best_val_loss_epoch': best_epoch,
            'at_best': per_epoch.loc[best_epoch].to_dict(),
            'at_final': per_epoch.loc[final_epoch].to_dict(),
            'last5_mean': last5[VAL_KEYS].mean().to_dict(),
            'grad_clipped_rate': float(steps['grad/clipped'].mean()),
            'sampler_grad_norm_median_last_epoch': float(steps['grad/sampler_norm'].iloc[-171:].median()),
            'runtime_hours': run.summary.get('_runtime', 0) / 3600,
        }
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2, default=float))
    per_epoch_all = pd.concat({label: per_epoch for label, (_, per_epoch, _) in data.items()}, names=['run'])
    per_epoch_all.to_csv(args.out.with_name(args.out.stem + '_per_epoch.csv'))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    main()
