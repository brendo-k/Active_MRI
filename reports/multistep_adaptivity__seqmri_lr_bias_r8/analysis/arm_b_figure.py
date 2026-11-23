"""Arm B in one figure: what the route changes, what the per-step maps add, and both on one dev slice.

Row 1: gradient norm reaching each step's logit map without and with the route (the numbers of
       `addendum_2026-11-19_route.md`, from scripts/diagnostics/seqmri_step_bias_gradients.py), and every arm-B run's
       pooled PSNR from summary.json.
Row 2: the learned per-step logit maps (sampler.mask_net.step_bias) of F_routebias and Fb_routebias, on one colour scale.
Row 3: on one dev slice and draw seed 1234: each run's final mask, reconstruction and error.

    python reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/arm_b_figure.py \
        --runs-dir /export/scratch/bkadota/lrb_r8_sync/runs \
        --fscr1-runs-dir /export/scratch/bkadota/multistep_adaptivity_sync/runs \
        --data-root 'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}' --train-tier S
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts' / 'eval')]

import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_multistep_adaptivity as ema  # noqa: E402
from activemri.utils.multistep import masked_slice_metrics, step_thresholds  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
GRADIENT_NORMS = {  # addendum_2026-11-19_route.md, steps 0-3
    ('F', 'no route'): [1.8e-6, 3.0e-6, 4.1e-6, 7.0e-6],
    ('F', 'route'): [7.3e-6, 7.7e-6, 7.1e-6, 7.0e-6],
    ('Fb', 'no route'): [0.0, 0.0, 0.0, 5.5e-6],
    ('Fb', 'route'): [4.2e-6, 4.6e-6, 4.6e-6, 5.5e-6],
}
RUNS = [  # (label, summary key, run name, from the fscr1 sync)
    ('F', 'F_tF', 'F_4_s42_fscr1_tF', True),
    ('F_route', 'F_route', 'F_4_s42_route_tF', False),
    ('F_routebias', 'F_routebias', 'F_4_s42_routebias_tF', False),
    ('Fb', 'Fb_tF', 'Fb_4_s42_fscr1_tF', True),
    ('Fb_route', 'Fb_route', 'Fb_4_s42_route_tF', False),
    ('Fb_routebias', 'Fb_routebias', 'Fb_4_s42_routebias_tF', False),
]
BLUE, ORANGE, GREEN, GREY = '#2563c9', '#d4761f', '#1f9a6b', '#8a8f98'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--fscr1-runs-dir', type=Path, required=True)
    parser.add_argument('--data-root', action='append', default=[], metavar='SOURCE=DIR')
    parser.add_argument('--train-tier', default=None, help='load this train tier (the val set does not depend on it)')
    parser.add_argument('--volume', default='file_brain_AXFLAIR_203_6000925.h5')
    parser.add_argument('--slice', type=int, default=8)
    parser.add_argument('--out', type=Path, default=REPORT / 'figures' / 'arm_b.png')
    return parser.parse_args()


@torch.no_grad()
def slice_outputs(run, args):
    model, checkpoint = ema.load_run(run, 'last_model.pt', 'cuda')
    options = checkpoint['options']
    dataset, collate = ema.val_data(run, args)
    parts = dataset.datasets if hasattr(dataset, 'datasets') else [dataset]
    examples = [example for part in parts for example in part.examples]
    index = next(i for i, (fname, sid) in enumerate(examples) if Path(str(fname)).name == args.volume and sid == args.slice)
    part = ema.labelled(collate([dataset[index]]), options, 'cuda')
    maps, _ = model.sense_maps_for(part['kspace'])
    masks, _ = model.sample_trajectory(part['kspace'], part['labels'],
                                       step_thresholds(1234, [index], model.num_step, model.shape, 'cuda'))
    output = model.reconstruct_from_mask(part['kspace'], masks[-1], maps, part['labels'])
    scores = masked_slice_metrics(output, part['target'], part['metric_mask'])
    bias = getattr(model.sampler.mask_net, 'step_bias', None)
    result = {'mask': masks[-1][0, 0].cpu().numpy(), 'recon': output[0, 0].float().cpu().numpy(),
              'target': part['target'][0, 0].float().cpu().numpy(), 'metric_mask': part['metric_mask'][0, 0].cpu().numpy(),
              'psnr': scores['psnr'].item(), 'bias': None if bias is None else bias[:, 0].float().cpu().numpy()}
    del model
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    run_config = ema.datasets_config

    def local_config(run, namespace):
        cfg = run_config(run, namespace)
        if args.train_tier:
            cfg.datasets.train_tier = args.train_tier
        return cfg

    ema.datasets_config = local_config
    data_args = SimpleNamespace(workers=0, val_half='dev', data_root=args.data_root, chunk=1)
    data_args.volume, data_args.slice = args.volume, args.slice
    summary = json.loads((REPORT / 'summary.json').read_text())
    outputs = {label: slice_outputs((args.fscr1_runs_dir if fscr1 else args.runs_dir) / run, data_args)
               for label, _, run, fscr1 in RUNS}

    figure = plt.figure(figsize=(19, 21), dpi=110)
    grid = figure.add_gridspec(6, 7, height_ratios=[1.1, 1, 1, 1, 1, 1], hspace=0.38, wspace=0.08, top=0.935)

    bars = figure.add_subplot(grid[0, 0:3])
    steps, width = np.arange(4), 0.2
    for i, ((arm, route), values) in enumerate(GRADIENT_NORMS.items()):
        color = BLUE if arm == 'F' else ORANGE
        bars.bar(steps + (i - 1.5) * width, np.array(values) * 1e6, width * 0.92, color=color,
                 alpha=1.0 if route == 'route' else 0.4, label=f"{arm}, {route}")
    for i, value in enumerate(GRADIENT_NORMS[('Fb', 'no route')][:3]):
        bars.text(i + 0.5 * width, 0.25, '0', ha='center', fontsize=9, color=ORANGE)
    bars.set_xticks(steps, [f'step {s}' for s in steps])
    bars.set_ylabel('|gradient| on the step map (×1e-6)')
    bars.set_title('Route: the loss now credits every step\'s draws\n'
                   'without it, the twin\'s steps 0-2 get no gradient (draws are detached between steps)', fontsize=11)
    bars.legend(fontsize=9, ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    bars.spines[['top', 'right']].set_visible(False)

    dots = figure.add_subplot(grid[0, 4:7])
    order = [('F', 'F_tF'), ('F_route', 'F_route'), ('F_routebias', 'F_routebias'),
             ('Fb', 'Fb_tF'), ('Fb_route', 'Fb_route'), ('Fb_routebias', 'Fb_routebias'), ('L (LOUPE)', 'L_tF')]
    for y, (label, key) in enumerate(order):
        color = GREEN if key.startswith('L') else ORANGE if key.startswith('Fb') else BLUE
        value = summary[key]['psnr']
        dots.plot(value, -y, 'o', color=color, markersize=9)
        dots.text(value + 0.012, -y, f'{value:.3f}', va='center', fontsize=10)
    dots.set_yticks([-y for y in range(len(order))], [label for label, _ in order])
    dots.set_xlim(33.2, 33.56)
    dots.set_xlabel('pooled PSNR on the dev half (dB, trainer convention)')
    dots.set_title('Route costs F 0.15 dB; the maps win back 0.06.\nThe maps lift the twin to LOUPE level', fontsize=11)
    dots.grid(axis='x', alpha=0.3)
    dots.spines[['top', 'right']].set_visible(False)

    maps = {label: outputs[label]['bias'] for label in ('F_routebias', 'Fb_routebias')}
    top = float(np.percentile(np.abs(np.concatenate([m.ravel() for m in maps.values()])), 99.5))
    for r, (label, bias) in enumerate(maps.items(), start=1):
        for step in range(4):
            axis = figure.add_subplot(grid[r, step])
            image = axis.imshow(bias[step], cmap='RdBu_r', vmin=-top, vmax=top)
            axis.set_xticks([])
            axis.set_yticks([])
            rms = float(np.sqrt((bias[step] ** 2).mean()))
            axis.set_title(f'{label}, step {step} map\nrms {rms:.2f}', fontsize=10)
        note = figure.add_subplot(grid[r, 4:7])
        note.axis('off')
        if r == 1:
            note.text(0.2, 0.5, 'Routebias: one learned 320 × 320 logit map per step,\n'
                                 'added before the softplus. Red raises a pixel\'s\n'
                                 'sampling probability at that step, blue lowers it.\n'
                                 'They start at zero, so the maps show what training\n'
                                 'learned. LOUPE has one such map; F had none.',
                      fontsize=11, va='center')
            bar = figure.colorbar(image, cax=note.inset_axes([0.04, 0.05, 0.025, 0.9]), label='logit offset')
            bar.ax.yaxis.set_label_position('left')

    target = outputs['F']['target']
    vmax = float(np.percentile(target, 99.5))
    labels = [label for label, *_ in RUNS]
    for c, label in enumerate(['target'] + labels):
        for r, row in enumerate(('mask', 'recon', 'error'), start=3):
            if row == 'error' and c == 0:
                continue
            if row == 'mask' and c == 0:
                axis = figure.add_subplot(grid[r, c])
                axis.imshow(outputs['F']['metric_mask'], cmap='gray')
                axis.set_title('metric mask', fontsize=10)
                axis.axis('off')
                continue
            axis = figure.add_subplot(grid[r if c else 4, c])
            if c == 0:
                axis.imshow(target, cmap='gray', vmin=0, vmax=vmax)
                axis.set_title('target (fully sampled)', fontsize=10)
            else:
                out = outputs[label]
                if row == 'mask':
                    axis.imshow(out['mask'], cmap='gray', interpolation='nearest')
                    axis.set_title(f'{label}: final mask', fontsize=10, color=ORANGE if label.startswith('Fb') else BLUE)
                elif row == 'recon':
                    axis.imshow(out['recon'], cmap='gray', vmin=0, vmax=vmax)
                    axis.set_title(f'recon, {out["psnr"]:.2f} dB on this slice', fontsize=10)
                else:
                    axis.imshow(np.abs(out['recon'] - target) * out['metric_mask'], cmap='magma', vmin=0, vmax=0.25 * vmax)
                    axis.set_title('|recon − target| (×4)', fontsize=10)
            axis.axis('off')
    figure.suptitle(f'Arm B: route, then route + per-step maps. Seed 42, FLAIR, R = 8. Bottom rows: {args.volume} '
                    f'slice {args.slice}, draw seed 1234', fontsize=13, y=0.99)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches='tight')
    print(args.out, {label: round(out['psnr'], 3) for label, out in outputs.items()})


if __name__ == '__main__':
    main()
