"""Rotation probe: does rotating a scan by 90 degrees rotate the sampling density that a run draws for it?

For every dev slice, the k-space, the target and the draw thresholds are rotated by T, an exact 90-degree rotation about
the k-space centre (index 160 of the centred 320 x 320 grid; T commutes with the FFT, so the image rotates the same way).
Each run draws its trajectory for the slice as it is and for T(slice), with the thresholds rotated too, so a sampler that
is exactly equivariant draws exactly rotated masks.

Density: each step's sampling scores before the budget rescaling (`unscaled`: LOUPE's sigmoid map, seqmri_2d's softplus
over its slice maximum), per step and summed over steps. The budget rescaling is not used: it depends on the acquired
region, which rotates, so it moves even a fixed map. On the region Omega acquired in both orientations, outside both
calibration centres, p (the slice), q (T(slice)) and T(p) are each standardised over Omega (z-scores), so a change of
scale or offset alone counts as no movement, and with d = z(T(p)) - z(p):

    w = <z(q) - z(p), d> / <d, d>     0: the density stays where it was; 1: it rotates with the anatomy.

Pooled w = sum over slices of the numerators over the sum of the denominators; 95% CI from resampling volumes (2,000).
A data-blind twin and LOUPE read no data, so they should give w near 0 (exactly 0 for LOUPE). Also scores PSNR (trainer
convention) on both orientations, since the VarNets never saw rotated brains.

    python reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/rotation_probe.py score \
        --runs-dir /export/scratch/bkadota/lrb_r8_sync/runs --fscr1-runs-dir /export/scratch/bkadota/multistep_adaptivity_sync/runs \
        --data-root 'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}' --train-tier S
    python reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/rotation_probe.py summarise
    python reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/rotation_probe.py figure
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts' / 'eval')]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_multistep_adaptivity as ema  # noqa: E402
from activemri.utils.multistep import masked_slice_metrics, step_thresholds  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
OUT = REPORT / 'rotation'
DRAW_SEED = 1234
BOOTSTRAP = 2000
RUNS = [  # (short name, run directory name, in the fscr1 sync)
    ('F', 'F_4_s42_fscr1_tF', True), ('Fb', 'Fb_4_s42_fscr1_tF', True), ('L', 'L_1_s42_fscr1_tF', True),
    ('F_r150', 'F_4_s42_fscr1_tF_r150', False), ('Fb_r150', 'Fb_4_s42_fscr1_tF_r150', False),
    ('L_r150', 'L_1_s42_fscr1_tF_r150', False),
    ('F_lrs', 'F_4_s42_lrs_tF', False), ('Fb_lrs', 'Fb_4_s42_lrs_tF', False),
    ('F_route', 'F_4_s42_route_tF', False), ('Fb_route', 'Fb_4_s42_route_tF', False),
    ('F_routebias', 'F_4_s42_routebias_tF', False), ('Fb_routebias', 'Fb_4_s42_routebias_tF', False),
]


def rotate(x, image_dims=(-2, -1)):
    """90-degree rotation about index 160 of both axes; commutes with the centred FFT."""
    return torch.roll(torch.rot90(x, 1, dims=image_dims), 1, dims=image_dims[0])


def rotate_kspace(kspace):
    return rotate(kspace, image_dims=(-3, -2))  # [B, C, H, W, 2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--runs-dir', type=Path, required=True)
    scoring.add_argument('--fscr1-runs-dir', type=Path, required=True)
    scoring.add_argument('--data-root', action='append', default=[], metavar='SOURCE=DIR')
    scoring.add_argument('--train-tier', default=None, help='load this train tier (the val set does not depend on it)')
    scoring.add_argument('--runs', nargs='*', default=None, help='short names; default all twelve')
    scoring.add_argument('--chunk', type=int, default=4)
    scoring.add_argument('--limit', type=int, default=None, help='first N dev slices only (smoke tests)')
    scoring.add_argument('--out', type=Path, default=OUT)
    summarising = commands.add_parser('summarise')
    summarising.add_argument('--out', type=Path, default=OUT)
    drawing = commands.add_parser('figure')
    drawing.add_argument('--out', type=Path, default=OUT)
    drawing.add_argument('--figure', type=Path, default=REPORT / 'figures' / 'rotation_probe.png')
    return parser.parse_args()


def densities(details):
    """[steps, B, H, W] sampling scores of each step before the budget rescaling."""
    return torch.stack([d['unscaled'][:, 0].float() for d in details])


def standardised(x, region):
    """z-scores of x [..., B, H, W] over `region` [B, H, W], zero outside it."""
    count = region.flatten(-2).sum(-1).clamp(min=1)[..., None, None]
    mean = (x * region).flatten(-2).sum(-1)[..., None, None] / count
    std = ((((x - mean) * region) ** 2).flatten(-2).sum(-1)[..., None, None] / count).sqrt().clamp(min=1e-12)
    return (x - mean) / std * region


def projections(p, q, region):
    """(numerator, denominator) of w for [..., B, H, W] densities."""
    zp, zq, zt = standardised(p, region), standardised(q, region), standardised(rotate(p), region)
    d = zt - zp
    return (zq - zp).mul(d).flatten(-2).sum(-1), d.mul(d).flatten(-2).sum(-1)


@torch.no_grad()
def score_run(run, args, data_args):
    model, checkpoint = ema.load_run(run, 'last_model.pt', 'cuda')
    options = checkpoint['options']
    dataset, collate = ema.val_data(run, data_args)
    indices = np.arange(len(dataset) if args.limit is None else min(args.limit, len(dataset)))
    rows = {key: [] for key in ('num', 'den', 'num_steps', 'den_steps', 'psnr', 'psnr_rotated', 'volume')}
    example = None
    for start in range(0, len(indices), args.chunk):
        chunk = indices[start:start + args.chunk]
        batch = collate([dataset[int(i)] for i in chunk])
        part = ema.labelled(batch, options, 'cuda')
        kspace, rotated = part['kspace'], rotate_kspace(part['kspace'])
        thresholds = step_thresholds(DRAW_SEED, chunk, model.num_step, model.shape, 'cuda')
        masks, details = model.sample_trajectory(kspace, part['labels'], thresholds)
        masks_r, details_r = model.sample_trajectory(rotated, part['labels'], [rotate(t) for t in thresholds])
        p, q = densities(details), densities(details_r)
        centre, valid = (m[:, 0].float() for m in model._initial_masks(kspace))
        centre_r, valid_r = (m[:, 0].float() for m in model._initial_masks(rotated))
        region = valid_r * rotate(valid) * (1 - centre_r) * (1 - rotate(centre))
        num_steps, den_steps = projections(p, q, region)
        num, den = projections(p.sum(0), q.sum(0), region)
        rows['num'] += num.tolist()
        rows['den'] += den.tolist()
        rows['num_steps'] += num_steps.T.tolist()
        rows['den_steps'] += den_steps.T.tolist()
        rows['volume'] += [str(v) for v in batch['fname']]
        maps, maps_r = model.sense_maps_for(kspace)[0], model.sense_maps_for(rotated)[0]
        mask_r_metric = ema.metric_mask_from_real_kspace(rotated, batch['anatomy']).bool().float()
        rows['psnr'] += masked_slice_metrics(model.reconstruct_from_mask(kspace, masks[-1], maps, part['labels']),
                                             part['target'], part['metric_mask'])['psnr'].tolist()
        rows['psnr_rotated'] += masked_slice_metrics(
            model.reconstruct_from_mask(rotated, masks_r[-1], maps_r, part['labels']),
            rotate(part['target']), mask_r_metric)['psnr'].tolist()
        if example is None:
            example = {'p': p.sum(0)[0].cpu().numpy(), 'q': q.sum(0)[0].cpu().numpy(),
                       'region': region[0].cpu().numpy(), 'volume': str(batch['fname'][0])}
    del model
    torch.cuda.empty_cache()
    return {key: np.array(value) for key, value in rows.items()}, example


def pooled(num, den, volumes, rng):
    unique = np.unique(volumes)
    groups = [np.flatnonzero(volumes == v) for v in unique]
    estimate = num.sum(0) / den.sum(0)
    draws = []
    for _ in range(BOOTSTRAP):
        chosen = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(num[chosen].sum(0) / den[chosen].sum(0))
    low, high = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return estimate, low, high


def summarise(out):
    rng = np.random.default_rng(20261122)
    summary = {}
    for short, _, _ in RUNS:
        path = out / f'{short}.npz'
        if not path.exists():
            continue
        rows = np.load(path)
        w, low, high = pooled(rows['num'], rows['den'], rows['volume'], rng)
        ws, lows, highs = pooled(rows['num_steps'], rows['den_steps'], rows['volume'], rng)
        summary[short] = {'w': [float(w), float(low), float(high)],
                          'w_by_step': [[float(a), float(b), float(c)] for a, b, c in zip(ws, lows, highs)],
                          'psnr': float(rows['psnr'].mean()), 'psnr_rotated': float(rows['psnr_rotated'].mean()),
                          'n_slices': int(len(rows['num'])), 'n_volumes': int(len(np.unique(rows['volume'])))}
        steps = '  '.join(f'{a:+.3f}' for a, _, _ in summary[short]['w_by_step'])
        print(f"{short:13s} w {w:+.3f} [{low:+.3f}, {high:+.3f}]  by step {steps}  "
              f"PSNR {summary[short]['psnr']:.3f} -> rotated {summary[short]['psnr_rotated']:.3f}")
    pairs = [('F', 'Fb'), ('F_r150', 'Fb_r150'), ('F_lrs', 'Fb_lrs'), ('F_route', 'Fb_route'),
             ('F_routebias', 'Fb_routebias')]
    summary['adaptive_minus_twin'] = {}
    for adaptive, twin in pairs:
        if adaptive not in summary or twin not in summary:
            continue
        a, b = np.load(out / f'{adaptive}.npz'), np.load(out / f'{twin}.npz')
        if not np.array_equal(a['volume'], b['volume']):
            raise ValueError(f'{adaptive} and {twin} were scored on different slices')
        groups = [np.flatnonzero(a['volume'] == v) for v in np.unique(a['volume'])]
        def gap(rows):
            return a['num'][rows].sum() / a['den'][rows].sum() - b['num'][rows].sum() / b['den'][rows].sum()
        draws = [gap(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
                 for _ in range(BOOTSTRAP)]
        estimate, (low, high) = gap(np.arange(len(a['num']))), np.percentile(draws, [2.5, 97.5])
        summary['adaptive_minus_twin'][f'{adaptive} - {twin}'] = [float(estimate), float(low), float(high)]
        print(f'{adaptive} - {twin}: w gap {estimate:+.3f} [{low:+.3f}, {high:+.3f}]')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))


def figure(out, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((out / 'summary.json').read_text())
    colours = {'F': '#2563c9', 'Fb': '#d4761f', 'L': '#1f9a6b'}
    family = lambda name: 'Fb' if name.startswith('Fb') else 'F' if name.startswith('F') else 'L'
    fig = plt.figure(figsize=(18, 12), dpi=110)
    grid = fig.add_gridspec(2, 9, height_ratios=[1, 0.62], hspace=0.28, wspace=0.12)

    dots = fig.add_subplot(grid[0, 0:4])
    names = [short for short, _, _ in RUNS if short in summary]  # noqa: E501
    for y, name in enumerate(names):
        w, low, high = summary[name]['w']
        dots.plot([low, high], [-y, -y], color=colours[family(name)], lw=2)
        dots.plot(w, -y, 'o', color=colours[family(name)], ms=8)
        dots.text(1.02, -y, f'{w:+.3f}', va='center', fontsize=9, transform=dots.get_yaxis_transform())
    dots.axvline(0, color='0.5', lw=1)
    dots.axvline(1, color='0.5', lw=1, ls=':')
    dots.set_yticks([-y for y in range(len(names))], names)
    dots.set_xlim(-0.05, 1.05)
    dots.set_xlabel('rotation index w (0 = density stays put, 1 = rotates with the scan)')
    dots.set_title('Does the density follow a 90° rotation? (all steps summed, 95% volume CI)', fontsize=11)
    dots.spines[['top', 'right']].set_visible(False)

    steps = fig.add_subplot(grid[0, 5:9])
    pairs = [('F', 'Fb'), ('F_r150', 'Fb_r150'), ('F_lrs', 'Fb_lrs'), ('F_route', 'Fb_route'), ('F_routebias', 'Fb_routebias')]
    styles = ['-', '--', '-.', ':', (0, (5, 1))]
    for (adaptive, twin), style in zip(pairs, styles):
        for name in (adaptive, twin):
            if name in summary:
                values = [v[0] for v in summary[name]['w_by_step']]
                steps.plot(range(4), values, ls=style, marker='o', color=colours[family(name)], ms=4,
                           label=name)
    steps.set_xticks(range(4), [f'step {i}' for i in range(4)])
    steps.set_ylim(-0.05, 1.0)
    steps.set_ylabel('w')
    steps.set_title('By step: adaptive F (blue) against its data-blind twin (orange)', fontsize=11)
    steps.legend(fontsize=8, ncol=2, frameon=False, loc='upper left')
    steps.spines[['top', 'right']].set_visible(False)

    for c, name in enumerate(('F_lrs', 'Fb_lrs', 'L')):
        path_example = out / f'{name}_example.npz'
        if not path_example.exists():
            continue
        example = np.load(path_example)
        p, q, region = (torch.from_numpy(example[k]) for k in ('p', 'q', 'region'))
        panels = [('p: original input', p), ('q: rotated input', q), ('R·p: p rotated', rotate(p))]
        for k, (title, image) in enumerate(panels):
            axis = fig.add_subplot(grid[1, 3 * c + k])
            shown = image.numpy()
            inside = shown[region.numpy() > 0]
            axis.imshow(shown, cmap='magma', vmin=np.percentile(inside, 1), vmax=np.percentile(inside, 99))
            axis.contour(region.numpy(), levels=[0.5], colors='cyan', linewidths=0.8)
            axis.set_title(f'{name}\n{title}', fontsize=10, color=colours[family(name)])
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle('Rotation probe: each dev slice rotated 90° (k-space, target and draw thresholds); scores before budget '
                 'rescaling, on k-space acquired in both orientations (cyan outline; bottom row: first dev slice)',
                 fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(path)


def main():
    args = parse_args()
    if args.command == 'summarise':
        summarise(args.out)
        return
    if args.command == 'figure':
        figure(args.out, args.figure)
        return
    run_config = ema.datasets_config

    def local_config(run, namespace):
        cfg = run_config(run, namespace)
        if args.train_tier:
            cfg.datasets.train_tier = args.train_tier
        return cfg

    ema.datasets_config = local_config
    data_args = SimpleNamespace(workers=4, val_half='dev', data_root=args.data_root, chunk=args.chunk)
    args.out.mkdir(parents=True, exist_ok=True)
    for short, name, fscr1 in RUNS:
        if args.runs and short not in args.runs:
            continue
        rows, example = score_run((args.fscr1_runs_dir if fscr1 else args.runs_dir) / name, args, data_args)
        np.savez(args.out / f'{short}.npz', **rows)
        np.savez(args.out / f'{short}_example.npz', **example)
        print(f"{short}: w {rows['num'].sum() / rows['den'].sum():+.3f}, PSNR {rows['psnr'].mean():.3f} -> "
              f"rotated {rows['psnr_rotated'].mean():.3f}", flush=True)
    summarise(args.out)


if __name__ == '__main__':
    main()
