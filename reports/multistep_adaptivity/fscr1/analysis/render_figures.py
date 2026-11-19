"""Figures for the fscr1 report: recons, sampling densities and step grids of the six fscr1 runs on the scorer's three
fixed figure slices, rendered after the fact from their final checkpoints, and training curves from the slurm logs in
this folder.

    render  one forward per run and slice (draw seed 1234, gradients on, as `score --gradients` draws its figures).
            Checks the local val set against the committed scores (slice hash, order) and each slice's PSNR, SSIM and
            per-step new samples against the scored values, then caches the panels in --cache. Needs a GPU and the
            six `last_model.pt` in --runs-dir.
    plot    the committed figure PNGs from the cache into ../figures: the density figures of both tiers, and at tier F
            the recon and sampling figures and F's step grids. --all-figures adds tier S5's recon and sampling
            figures and every run's step grids. No GPU.
    curves  per-epoch trainer metrics of the six runs to ../curves.json and ../figures/curves.png.

Run from the repository root, e.g. `python reports/multistep_adaptivity/fscr1/analysis/render_figures.py plot`.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parents[1]
FIGURES = REPORT / 'figures'
DEFAULT_RUNS = '/export/scratch/bkadota/multistep_adaptivity_sync/runs'
DEFAULT_DATA_ROOT = 'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}'
DEFAULT_CACHE = '/export/scratch/bkadota/multistep_adaptivity_sync/fscr1_render_cache'
ARMS = ('F', 'Fb', 'L')
TIERS = ('tF', 'tS5')
ARM_LABELS = {'F': 'F, SeqMRI 4-step', 'Fb': 'Fb, data-blind twin', 'L': 'L, LOUPE'}
TIER_LABELS = {'tF': 'tier F', 'tS5': 'tier S5'}
ARM_COLOURS = {'F': '#2a78d6', 'Fb': '#eb6834', 'L': '#1baf7a'}
DRAW_SEED = 1234
FIGURE_SLICES = 3
STEP_KEYS = ('measured', 'rescaled', 'residual', 'grad', 'new', 'mask_after', 'zero_filled', 'recon', 'error')
PLOT_PATTERNS = ('recon_*.png', 'sampling_*.png', 'steps_*.png', 'density_*.png')


def run_name(arm, tier):
    return f"{arm}_{1 if arm == 'L' else 4}_s42_fscr1_{tier}"


def scores_of(name):
    return (np.load(REPORT / 'scores' / f'{name}.npz', allow_pickle=True),
            json.loads((REPORT / 'scores' / f'{name}.json').read_text()))


def local_val_data(ev, run, args):
    cfg = ev.datasets_config(run, args)
    try:
        _, val_loader = ev.build_dataloaders(cfg)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f'tier-F train set does not build here ({error!r}); building with train tier S, val unchanged', flush=True)
        cfg.datasets.train_tier = 'S'
        _, val_loader = ev.build_dataloaders(cfg)
    return val_loader.dataset, val_loader.collate_fn


def forward_panels(ev, model, options, dataset, collate, index, args):
    import torch
    from activemri.utils.multistep import StepRecorder, masked_slice_metrics, step_panels, step_thresholds
    part = ev.labelled(collate([dataset[int(index)]]), options, args.device)
    thresholds = step_thresholds(DRAW_SEED, [index], model.num_step, model.shape, args.device)
    recorder = StepRecorder(slices=[0])
    with torch.set_grad_enabled(True):
        model.sampler.requires_grad_(True)
        model.step_recorder = recorder
        pred = model(part['target'], part['kspace'], anatomy_labels=part['labels'], thresholds=thresholds)
        model.step_recorder = None
        loss, _ = model.loss(pred, {'target': part['target'], 'kspace': part['kspace']}, ev.LOSS_META,
                             options.loss_type)
        loss.backward()
        model.sampler.requires_grad_(False)
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        panels = step_panels(model, part['kspace'], part['target'], pred['sense_maps'].detach(),
                             [mask.detach() for mask in pred['masks']], 0, part['labels'], part['metric_mask'],
                             recorder)
        sense_maps, _ = model.sense_maps_for(part['kspace'])
        calibration, valid = model._initial_masks(part['kspace'])
        coil_rss = torch.view_as_complex(part['kspace'][0].contiguous()).abs().pow(2).sum(dim=0).sqrt()
        masks, _ = model.sample_trajectory(part['kspace'], part['labels'], thresholds)
        recon = model.reconstruct_from_mask(part['kspace'], masks[-1], sense_maps, part['labels'])
        scored = {key: float(value[0]) for key, value in
                  masked_slice_metrics(recon.float(), part['target'].float(), part['metric_mask']).items()}
        previous = [calibration] + masks[:-1]
        scoring_new = [float((after - before).sum()) for before, after in zip(previous, masks)]
        same_final_mask = bool(torch.equal(masks[-1], pred['masks'][-1].detach()))
    cached = {'target': panels['target'], 'metric_mask': part['metric_mask'][0, 0].cpu().numpy(),
              'calibration': calibration[0, 0].cpu().numpy(), 'valid': valid[0, 0].float().cpu().numpy(),
              'kspace_rss': coil_rss.float().cpu().numpy(), 'num_step': model.num_step,
              'sampler_type': options.sampler_type, 'data_blind': bool(getattr(options, 'data_blind', False)),
              'scored_psnr': scored['psnr'], 'scored_ssim': scored['ssim'], 'scored_nmse': scored['nmse'],
              'scoring_new': np.array(scoring_new), 'same_final_mask': same_final_mask}
    for step, row in enumerate(panels['rows']):
        for key in STEP_KEYS:
            if key in row:
                cached[f'row{step}_{key}'] = row[key]
        cached[f'row{step}_psnr'] = row['psnr']
    return cached


def render(args):
    sys.path[:0] = [str(REPO / 'scripts' / 'eval'), str(REPO / 'scripts'), str(REPO)]
    import torch
    import eval_multistep_adaptivity as ev
    import activemri
    if not Path(activemri.__file__).resolve().is_relative_to(REPO):
        raise SystemExit(f'activemri imported from {activemri.__file__}, not {REPO}')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runs_dir, cache_dir = Path(args.runs_dir), Path(args.cache)
    ev_args = argparse.Namespace(workers=4, val_half=None, data_root=[args.data_root], chunk=8, limit=None,
                                 draw_seeds=[DRAW_SEED, 4321], device=args.device, figures=FIGURE_SLICES,
                                 gradients=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset, collate = local_val_data(ev, runs_dir / run_name('F', 'tF'), ev_args)
    indices = ev.scored_indices(dataset, ev_args)
    meta = ev.header_pass(dataset, indices, collate, ev_args)
    positions = ev.figure_positions(meta, FIGURE_SLICES)
    local_hash = ev.slices_hash(meta, indices)
    checks = {'local_slices_hash': local_hash, 'positions': positions, 'runs': {}}
    for tier in TIERS:
        for arm in ARMS:
            name = run_name(arm, tier)
            scores, summary = scores_of(name)
            if local_hash != summary['slices']['hash']:
                raise SystemExit(f'{name}: local val slices hash {local_hash} != scored {summary["slices"]["hash"]}')
            if not (np.array_equal(scores['dataset_index'], indices) and np.array_equal(scores['volume'], meta['volume'])
                    and np.array_equal(scores['slice_id'], meta['slice_id'])):
                raise SystemExit(f'{name}: local val order differs from the scored order')
            seed_column = list(scores['draw_seeds']).index(DRAW_SEED)
            model, checkpoint = ev.load_run(runs_dir / name, 'last_model.pt', ev_args.device)
            options = checkpoint['options']
            checks['runs'][name] = {'epoch': int(checkpoint['epoch'])}
            for label, position in positions.items():
                cached = forward_panels(ev, model, options, dataset, collate, indices[position], ev_args)
                cached.update(volume=str(meta['volume'][position]), slice_id=int(meta['slice_id'][position]),
                              dataset_index=int(indices[position]), name=name)
                last = cached['num_step'] - 1
                check = {
                    'psnr_scored': float(scores['psnr'][position, seed_column]),
                    'psnr_local_scoring_path': cached['scored_psnr'],
                    'psnr_local_figure_path': float(cached[f'row{last}_psnr']),
                    'ssim_scored': float(scores['ssim'][position, seed_column]),
                    'ssim_local': cached['scored_ssim'],
                    'new_scored': scores['step_new'][position, :, seed_column].tolist(),
                    'new_local_scoring_path': cached['scoring_new'].tolist(),
                    'new_local_figure_path': [float(cached[f'row{s}_new'].sum()) for s in range(cached['num_step'])],
                    'figure_and_scoring_final_masks_equal': cached['same_final_mask'],
                }
                check['psnr_difference'] = check['psnr_local_scoring_path'] - check['psnr_scored']
                checks['runs'][name][label] = check
                np.savez_compressed(cache_dir / f'{name}__{label}.npz', **cached)
                print(f"{name} {label} {cached['volume']} s{cached['slice_id']}: PSNR scored "
                      f"{check['psnr_scored']:.3f} local {check['psnr_local_scoring_path']:.3f} figure "
                      f"{check['psnr_local_figure_path']:.3f}; new scored {check['new_scored']} local "
                      f"{check['new_local_scoring_path']} figure {check['new_local_figure_path']}; masks equal "
                      f"{cached['same_final_mask']}", flush=True)
            del model
            torch.cuda.empty_cache()
    (REPORT / 'render_checks.json').write_text(json.dumps(checks, indent=2) + '\n')


def load_cache(cache_dir):
    cache = {}
    for path in sorted(Path(cache_dir).glob('*.npz')):
        name, label = path.stem.split('__')
        with np.load(path, allow_pickle=True) as data:
            cache[(name, label)] = {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}
    if not cache:
        raise SystemExit(f'no cached panels in {cache_dir}; run the render stage first')
    return cache


def per_volume_gap(tier):
    first, _ = scores_of(run_name('F', tier))
    second, _ = scores_of(run_name('L', tier))
    gap = np.nanmean(first['psnr'], axis=1) - np.nanmean(second['psnr'], axis=1)
    return {volume: float(gap[first['volume'] == volume].mean()) for volume in np.unique(first['volume'])}


def slice_descriptions(cache):
    import csv
    with open(REPO / 'configs/datasets/fastmri_local_v1/manifest.csv') as f:
        rows = {row['file']: row for row in csv.DictReader(f)}
    gap = per_volume_gap('tF')
    descriptions = {}
    for (name, label), cached in cache.items():
        if name != run_name('F', 'tF'):
            continue
        volume = cached['volume']
        row = rows[volume]
        width = row['kspace_shape'].strip('[]').split(',')[-1].strip()
        short = volume.replace('file_brain_AXFLAIR_', '').replace('.h5', '')
        descriptions[label] = {
            'volume': volume, 'short': short, 'slice_id': int(cached['slice_id']), 'field': row['field'],
            'model': row['model'], 'coils': int(row['coils']), 'width': int(width), 'volume_gap_tF': gap[volume],
            'title': (f"AXFLAIR_{short} slice {int(cached['slice_id'])}: {row['field']} {row['model']}, "
                      f"{row['coils']} coils, k-space {width} columns wide; this volume's F - L at tier F "
                      f"{gap[volume]:+.2f} dB")}
    return descriptions


def blue_ramp():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('blue_ramp', ['#fcfcfb', '#8fbaf0', '#2a78d6', '#0f3a73'])


def last_row(cached, key):
    return cached[f"row{int(cached['num_step']) - 1}_{key}"]


DENSITY_SIGMA = 3.0
ZOOM_FRACTION = 0.36
OUTLINE = '#2a78d6'
NOT_SCORED = '#e6e5df'


def local_density(mask):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(mask.astype(np.float64), DENSITY_SIGMA)


def no_ticks(axis):
    axis.set_xticks([])
    axis.set_yticks([])


def bare(axis):
    no_ticks(axis)
    for spine in axis.spines.values():
        spine.set_visible(False)


def zoom_box(metric_mask):
    rows, columns = np.nonzero(metric_mask)
    side = int(round(ZOOM_FRACTION * metric_mask.shape[0]))
    top = int(np.clip(rows.mean() - side / 2, 0, metric_mask.shape[0] - side))
    left = int(np.clip(columns.mean() - side / 2, 0, metric_mask.shape[1] - side))
    return top, left, side


def recon_figure(cache, label, tier, description, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    first = cache[(run_name('F', 'tF'), label)]
    reference, metric_mask = first['target'], first['metric_mask'] > 0
    image_top = float(np.percentile(reference, 99.5))
    errors = {(arm, t): np.abs(last_row(cache[(run_name(arm, t), label)], 'recon') - reference)
              for arm in ARMS for t in TIERS}
    error_top = float(np.percentile(np.concatenate([e[metric_mask] for e in errors.values()]), 99.5))
    top, left, side = zoom_box(metric_mask)
    error_map = plt.get_cmap('magma').copy()
    error_map.set_bad(NOT_SCORED)
    figure, axes = plt.subplots(3, 4, figsize=(4 * 3.0, 3 * 3.0 + 0.75), dpi=150)
    images = [('reference', None, reference)] + [
        (arm, cache[(run_name(arm, tier), label)], last_row(cache[(run_name(arm, tier), label)], 'recon'))
        for arm in ARMS]
    for c, (arm, cached, image) in enumerate(images):
        full, zoom, error = axes[0, c], axes[1, c], axes[2, c]
        full.imshow(image, cmap='gray', vmin=0, vmax=image_top, interpolation='nearest')
        full.add_patch(Rectangle((left - 0.5, top - 0.5), side, side, fill=False, edgecolor=OUTLINE, linewidth=0.9))
        zoom.imshow(image[top:top + side, left:left + side], cmap='gray', vmin=0, vmax=image_top,
                    interpolation='nearest')
        if cached is None:
            full.contour(metric_mask, levels=[0.5], colors=OUTLINE, linewidths=0.6, linestyles='dotted')
            full.set_title('reference\n(dotted: metric mask, box: zoom)', fontsize=9)
            bare(error)
            continue
        full.set_title(f"{ARM_LABELS[arm]}\n{cached['scored_psnr']:.2f} dB, SSIM {cached['scored_ssim']:.3f}",
                       fontsize=9)
        shown = error.imshow(np.where(metric_mask, errors[(arm, tier)], np.nan), cmap=error_map, vmin=0,
                             vmax=error_top, interpolation='nearest')
    colorbar_axis = axes[2, 0].inset_axes([0.45, 0.06, 0.09, 0.88])
    figure.colorbar(shown, cax=colorbar_axis)
    colorbar_axis.tick_params(labelsize=7.5)
    axes[0, 0].set_ylabel('reconstruction', fontsize=9)
    axes[1, 0].set_ylabel(f'zoom, {side} x {side} px', fontsize=9)
    axes[2, 0].set_ylabel('|recon - reference|\ninside the metric mask', fontsize=9)
    for axis in axes.flat:
        no_ticks(axis)
    figure.suptitle(f"{TIER_LABELS[tier]} runs. {description['title']}.\nDraw seed {DRAW_SEED}. One grey window "
                    "from the reference; one error scale shared by all six runs on this slice, grey outside the "
                    "metric mask", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path)
    plt.close(figure)


def kspace_panel(axis, cached):
    rss = cached['kspace_rss']
    shown = np.where(rss > 0, np.log10(np.maximum(rss, 1e-30)), np.nan)
    finite = shown[np.isfinite(shown)]
    ramp = blue_ramp().copy()
    ramp.set_bad(NOT_SCORED)
    axis.imshow(shown, cmap=ramp, vmin=float(np.percentile(finite, 1)), vmax=float(finite.max()),
                interpolation='nearest')
    axis.contour(cached['valid'], levels=[0.5], colors=OUTLINE, linewidths=0.7)


def mask_panel(axis, cached, mask, kind):
    if kind == 'mask':
        axis.imshow(mask, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    else:
        axis.imshow(local_density(mask), cmap=blue_ramp(), vmin=0, vmax=0.5, interpolation='nearest')
    axis.contour(cached['valid'], levels=[0.5], colors=OUTLINE, linewidths=0.7)
    centre = np.nonzero(cached['calibration'])
    axis.plot([centre[1].mean()], [centre[0].mean()], marker='+', color='#eb6834', markersize=7, markeredgewidth=1.2)


def sampling_figure(cache, label, tier, description, path):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 4, figsize=(4 * 3.0, 2 * 3.3 + 0.9), dpi=150, layout='constrained')
    loupe = cache[(run_name('L', tier), label)]
    kspace_panel(axes[0, 0], loupe)
    axes[0, 0].set_title('fully sampled k-space\nlog10 coil RSS', fontsize=9)
    loupe_map = loupe.get('row0_rescaled')
    if loupe_map is not None:
        axes[1, 0].imshow(loupe_map, cmap=blue_ramp(), vmin=0, vmax=1, interpolation='nearest')
        axes[1, 0].contour(loupe['valid'], levels=[0.5], colors=OUTLINE, linewidths=0.7)
        axes[1, 0].set_title("LOUPE's draw probability\n(its learned map, 0 to 1)", fontsize=9)
    for c, arm in enumerate(ARMS, start=1):
        cached = cache[(run_name(arm, tier), label)]
        final = last_row(cached, 'mask_after')
        right = final[:, final.shape[1] // 2 + 1:].sum() / max(final.sum(), 1)
        mask_panel(axes[0, c], cached, final, 'mask')
        axes[0, c].set_title(f"{ARM_LABELS[arm]}\nfinal mask, {int(final.sum())} samples, {100 * right:.0f}% right "
                             "of centre", fontsize=9)
        mask_panel(axes[1, c], cached, final, 'density')
        axes[1, c].set_title(f'local sampling density\n(mask blurred, sigma {DENSITY_SIGMA:g} px, 0 to 0.5)', fontsize=9)
    for axis in axes.flat:
        no_ticks(axis)
    figure.suptitle(f"{TIER_LABELS[tier]} runs. {description['title']}.\nk-space on the 320 x 320 grid, draw seed "
                    f"{DRAW_SEED}. Blue outline: the acquired region; orange +: the centre of the 10 x 10 calibration "
                    "block", fontsize=9)
    figure.savefig(path)
    plt.close(figure)


def density_across_slices_figure(cache, descriptions, tier, path):
    import matplotlib.pyplot as plt
    labels = sorted(descriptions)
    figure, axes = plt.subplots(3, len(labels), figsize=(len(labels) * 3.2, 3 * 3.3 + 1.0), dpi=150,
                                layout='constrained')
    for r, arm in enumerate(ARMS):
        for c, label in enumerate(labels):
            cached = cache[(run_name(arm, tier), label)]
            final = last_row(cached, 'mask_after')
            right = final[:, final.shape[1] // 2 + 1:].sum() / max(final.sum(), 1)
            mask_panel(axes[r, c], cached, final, 'density')
            no_ticks(axes[r, c])
            d = descriptions[label]
            heading = (f"AXFLAIR_{d['short']} s{d['slice_id']}\n{d['field']} {d['model']}, {d['width']} columns\n"
                       if r == 0 else '')
            axes[r, c].set_title(f'{heading}{100 * right:.0f}% of samples right of centre', fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel(ARM_LABELS[arm], fontsize=9.5)
    figure.suptitle(f'{TIER_LABELS[tier]} runs: local sampling density of the final mask on the three slices '
                    f'(blurred, sigma {DENSITY_SIGMA:g} px, 0 to 0.5), draw seed {DRAW_SEED}.\nBlue outline: the '
                    'acquired region; orange +: the centre of the calibration block', fontsize=9)
    figure.savefig(path)
    plt.close(figure)


def step_grid(cache, name, label, description, path):
    import matplotlib.pyplot as plt
    sys.path[:0] = [str(REPO)]
    from activemri.utils.multistep import step_grid_figure
    cached = cache[(name, label)]
    rows = []
    for step in range(int(cached['num_step'])):
        row = {key: cached[f'row{step}_{key}'] for key in STEP_KEYS if f'row{step}_{key}' in cached}
        row['psnr'] = float(cached[f'row{step}_psnr'])
        rows.append(row)
    kind = 'data-blind twin' if cached['data_blind'] else str(cached['sampler_type'])
    title = (f"{name} ({kind}, K={int(cached['num_step'])}), AXFLAIR_{description['short']} slice "
             f"{description['slice_id']} ({description['field']} {description['model']}, {description['width']} "
             f"columns), draw seed {DRAW_SEED}. Rows before the last reconstruct masks the reconstructor never "
             "trained on")
    figure = step_grid_figure({'rows': rows, 'target': cached['target']}, title, panel_inches=2.0, dpi=120)
    figure.savefig(path)
    plt.close(figure)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    FIGURES.mkdir(exist_ok=True)
    cache = load_cache(args.cache)
    descriptions = slice_descriptions(cache)
    (REPORT / 'slice_descriptions.json').write_text(json.dumps(descriptions, indent=2) + '\n')
    for pattern in PLOT_PATTERNS:
        for stale in FIGURES.glob(pattern):
            stale.unlink()
    tiers = TIERS if args.all_figures else ('tF',)
    step_arms = ARMS if args.all_figures else ('F',)
    for label, description in descriptions.items():
        for tier in tiers:
            recon_figure(cache, label, tier, description, FIGURES / f'recon_{label}_{tier}.png')
            sampling_figure(cache, label, tier, description, FIGURES / f'sampling_{label}_{tier}.png')
            for arm in step_arms:
                name = run_name(arm, tier)
                step_grid(cache, name, label, description, FIGURES / f'steps_{arm}_{tier}_{label}.png')
    for tier in TIERS:
        density_across_slices_figure(cache, descriptions, tier, FIGURES / f'density_{tier}.png')


EPOCH_LINE = re.compile(r'Epoch = \[\s*(\d+)/\s*\d+\] TrainLoss = ([\d.eE+-]+) ValLoss = ([\d.eE+-]+)')
VAL_LINE = re.compile(r'\[brain\] MSE=([\d.eE+-]+) NMSE=([\d.eE+-]+) PSNR=([\d.eE+-]+) SSIM=([\d.eE+-]+) '
                      r'Sparsity=([\d.eE+-]+) \(n=(\d+)\)')
CURVE_PANELS = (('val_psnr', 'validation PSNR, dB (trainer)'), ('val_loss', 'validation L1 loss'),
                ('train_loss', 'training L1 loss'))
CURVE_FIRST_EPOCH = 10
LR_STEP_EPOCH = 90


def read_curves():
    jobs = dict(line.split() for line in (REPORT / 'fscr1_jobs.txt').read_text().splitlines() if line.strip())
    result = {}
    for tier in TIERS:
        for arm in ARMS:
            name = run_name(arm, tier)
            series = {key: [] for key in ('epoch', 'train_loss', 'val_loss', 'val_psnr', 'val_ssim', 'val_nmse')}
            pending = None
            for line in (REPORT / f'multistep-{jobs[name]}.log').read_text().splitlines():
                if (match := VAL_LINE.search(line)):
                    pending = match
                    if int(match.group(6)) != 307:
                        raise SystemExit(f'{name}: val line with n={match.group(6)}')
                elif (match := EPOCH_LINE.search(line)):
                    if pending is None:
                        raise SystemExit(f'{name}: epoch {match.group(1)} without a val line before it')
                    series['epoch'].append(int(match.group(1)))
                    series['train_loss'].append(float(match.group(2)))
                    series['val_loss'].append(float(match.group(3)))
                    series['val_psnr'].append(float(pending.group(3)))
                    series['val_ssim'].append(float(pending.group(4)))
                    series['val_nmse'].append(float(pending.group(2)))
                    pending = None
            if series['epoch'] != list(range(1, 101)):
                raise SystemExit(f"{name}: epochs {series['epoch'][:3]}...{series['epoch'][-3:]}")
            _, summary = scores_of(name)
            series['scored_psnr'] = summary['means']['psnr']['all'] if isinstance(summary['means']['psnr'], dict) \
                else summary['means']['psnr']
            result[name] = series
    return result


def curves_figure(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#1f1f1d', '#6b6a64', '#e4e3dd'
    figure, axes = plt.subplots(len(CURVE_PANELS), len(TIERS), figsize=(11.5, 10.5), dpi=150, layout='constrained')
    for c, tier in enumerate(TIERS):
        for r, (key, title) in enumerate(CURVE_PANELS):
            axis = axes[r, c]
            shown = []
            for arm in ARMS:
                series = result[run_name(arm, tier)]
                epochs = np.array(series['epoch'])
                values = np.array(series[key])
                keep = epochs >= CURVE_FIRST_EPOCH
                axis.plot(epochs[keep], values[keep], color=ARM_COLOURS[arm], linewidth=1.6, label=ARM_LABELS[arm])
                shown.append((values[keep][-1], arm))
            low, high = axis.get_ylim()
            spacing = 0.06 * (high - low)
            placed = []
            for value, arm in sorted(shown):
                position = value if not placed or value - placed[-1] >= spacing else placed[-1] + spacing
                placed.append(position)
                axis.annotate(arm, xy=(100, value), xytext=(104, position), textcoords='data', fontsize=8.5,
                              color=ink, va='center', annotation_clip=False,
                              arrowprops=dict(arrowstyle='-', color=muted, linewidth=0.6, shrinkA=0, shrinkB=2))
            axis.axvline(LR_STEP_EPOCH, color=muted, linewidth=0.8, linestyle=(0, (3, 3)))
            rising = key == 'val_psnr'
            axis.text(LR_STEP_EPOCH - 1, low + 0.03 * (high - low) if rising else high - 0.03 * (high - low),
                      'lr x0.1', fontsize=7.5, color=muted, ha='right', va='bottom' if rising else 'top')
            axis.set_xlim(CURVE_FIRST_EPOCH, 100)
            axis.grid(True, color=grid, linewidth=0.6)
            axis.set_axisbelow(True)
            for side in ('top', 'right'):
                axis.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                axis.spines[side].set_color(muted)
            axis.tick_params(colors=muted, labelsize=8)
            axis.set_title(f'{TIER_LABELS[tier]}: {title}', fontsize=9.5, color=ink, loc='left')
            if r == len(CURVE_PANELS) - 1:
                axis.set_xlabel('epoch', fontsize=8.5, color=muted)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=9)
    figure.suptitle('fscr1 training curves, seed 42, epochs 10 to 100. Validation is the 307 dev slices every epoch. '
                    "Trainer PSNR uses the last batch's data range,\nso it reads about 1.2 to 1.4 dB above the scored "
                    'PSNR; compare runs within a panel. Each panel has its own y-axis.', fontsize=9, color=ink)
    figure.savefig(path)
    plt.close(figure)


def curves(args):
    result = read_curves()
    (REPORT / 'curves.json').write_text(json.dumps(result) + '\n')
    FIGURES.mkdir(exist_ok=True)
    curves_figure(result, FIGURES / 'curves.png')
    for name, series in result.items():
        late = slice(80, 100)
        print(f"{name}: val PSNR e100 {series['val_psnr'][-1]:.3f} (scored {series['scored_psnr']}); "
              f"train loss e81-100 {np.mean(series['train_loss'][late]):.5f} vs e61-80 "
              f"{np.mean(series['train_loss'][60:80]):.5f}; val loss e81-100 {np.mean(series['val_loss'][late]):.5f} "
              f"vs e61-80 {np.mean(series['val_loss'][60:80]):.5f}; min val loss epoch "
              f"{series['epoch'][int(np.argmin(series['val_loss']))]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('render', 'plot', 'curves'))
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS, help="folder holding the six runs' last_model.pt")
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT, help='datasets.roots override for the val volumes')
    parser.add_argument('--cache', default=DEFAULT_CACHE, help='rendered panels (npz), kept outside the repository')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--all-figures', action='store_true',
                        help="plot: also tier S5's recon and sampling figures and every run's step grids (not committed)")
    arguments = parser.parse_args()
    {'render': render, 'plot': plot, 'curves': curves}[arguments.stage](arguments)
