"""K-step adaptivity eval for exp/multistep_adaptivity (reports/multistep_adaptivity/proposal.md, section 5).

`score` evaluates one run's checkpoint on every val slice. Every draw comes from a map keyed by (draw seed, slice, step)
(activemri.utils.multistep.step_thresholds), so all runs scored on the same slices share their random numbers. It
writes <out>/<name>.npz (per slice) and <name>.json (summary), and logs both to a wandb eval run:
  - PSNR_Masked, SSIM_Masked and NMSE as the trainer scores them; realised fraction; new samples at every step.
  - Trajectory swap cost. Each slice gets a donor: another volume's slice with the same condition, acquired-location
    mask and coil count. The run's whole trajectory is drawn on the donor with the target's thresholds, and its final
    mask is reconstructed on the target. Own minus donor is the cost. A data-blind twin reads nothing that differs
    between the two, so its donor mask must equal its own bit for bit (the null that tests this code).
  - Per-step data-swap sensitivity (seq_adapt_2d): the share of step i's new samples that move when only the data
    channels are the donor's, M_i, header, budget and thresholds held. Exactly 0 for the twin.
  - Guards: pooled SSIM_Masked, the realised fraction against 1/R, slices below 20 dB.
  - Step-grid figures, one row per step (activemri.utils.multistep.step_grid_figure), for fixed mid-volume slices
    of each condition. `--gradients` adds the training loss's gradient on each step's probabilities.
`contrast` gives paired per-slice differences between scored runs, e.g. G_K = A_K - T_K, the twin floor T_K - T_K',
T_K - T_1 and U_K = A_K - T_1. CIs are 95% volume-cluster bootstraps of the slice-weighted mean, volumes resampled
within each condition. It refuses runs scored on different slices or draw seeds.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_multistep_adaptivity.py score --run <run dir> --name A_4_s42 \
        --out reports/multistep_adaptivity/scores
    python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/scores \
        --pair G_4_s42=A_4_s42-T_4_s42 --pair F_16=T_16_s42-T_16b_s42 --out reports/multistep_adaptivity/contrasts.json
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import _path  # noqa: F401,E402
from activemri.training.builders import build_dataloaders  # noqa: E402
from activemri.training.trainer import build_model  # noqa: E402
from activemri.utils.metric_mask import metric_mask_from_real_kspace  # noqa: E402
from activemri.utils.multistep import (StepRecorder, cluster_bootstrap, header_signature, masked_slice_metrics,  # noqa: E402
                                       match_donors, step_grid_figure, step_panels, step_thresholds)

DRAW_SEEDS = (1234, 4321)
METRICS = ('psnr', 'ssim', 'nmse')
FAILED_RECON_DB = 20.0
SSIM_GUARD = 0.80
REALISED_TOLERANCE = 0.002
FIGURE_VOLUME_QUANTILES = (0.25, 0.5, 0.75)
LOSS_META = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
REPO = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    scoring = commands.add_parser('score', help="score one run's checkpoint on the val set")
    scoring.add_argument('--run', type=Path, required=True, help='run directory (checkpoint, .hydra/config.yaml)')
    scoring.add_argument('--name', required=True, help='name of this run in the scores folder, e.g. A_4_s42')
    scoring.add_argument('--out', type=Path, required=True, help='scores folder')
    scoring.add_argument('--checkpoint', default='last_model.pt')
    scoring.add_argument('--draw-seeds', type=int, nargs='+', default=list(DRAW_SEEDS))
    scoring.add_argument('--val-half', choices=('all', 'dev', 'holdout'), default=None,
                         help="score this half of the val volumes instead of the run's own datasets.val_half")
    scoring.add_argument('--chunk', type=int, default=8)
    scoring.add_argument('--workers', type=int, default=4)
    scoring.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    scoring.add_argument('--no-swap', action='store_true', help='skip the trajectory swap cost')
    scoring.add_argument('--no-sensitivity', action='store_true', help='skip the per-step data-swap sensitivity')
    scoring.add_argument('--figures', type=int, default=len(FIGURE_VOLUME_QUANTILES),
                         help='step-grid figures per condition (0 = none)')
    scoring.add_argument('--gradients', action='store_true', help="figures: add the loss gradient on each step's draw")
    scoring.add_argument('--no-wandb', action='store_true', help='write files only (smoke tests)')
    scoring.add_argument('--device', default='cuda')
    contrasting = commands.add_parser('contrast', help='paired differences between scored runs')
    contrasting.add_argument('--scores', type=Path, required=True, help='scores folder')
    contrasting.add_argument('--pair', action='append', required=True, metavar='LABEL=A-B',
                             help='per-slice A minus B, e.g. G_4_s42=A_4_s42-T_4_s42 (repeatable)')
    contrasting.add_argument('--out', type=Path, required=True, help='contrasts .json')
    return parser.parse_args()


def git_state(path):
    def git(*args):
        return subprocess.run(['git', *args], cwd=path, capture_output=True, text=True).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'dirty_tracked_files': git('status', '--short', '--untracked-files=no').splitlines()}


def load_run(run, checkpoint_name, device):
    """(model in eval mode with every weight frozen, checkpoint)."""
    checkpoint = torch.load(run / checkpoint_name, map_location='cpu', weights_only=False)
    options = checkpoint['options']
    if getattr(options, 'pseudo_target', False) or getattr(options, 'phase_scramble_std', 0.0) > 0:
        raise ValueError(f"{run} trains on pseudo targets or scrambled phase; this eval scores the dataset's targets")
    if getattr(options, 'budget_over', 'grid') != 'acquired':
        raise ValueError(f"{run} counts R over the grid; the realised-fraction guard counts it over acquired k-space")
    model = build_model(options)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint


def labelled(batch, options, device):
    kspace = batch['kspace'].to(device)
    target = batch['target'].to(device)
    classes = list(options.label_classes)
    return {'kspace': kspace, 'target': target if target.dim() == 4 else target.unsqueeze(1),
            'metric_mask': metric_mask_from_real_kspace(kspace, batch['anatomy']).bool().float(),
            'labels': torch.as_tensor([classes.index(name) for name in batch[options.label_key]], device=device)}


def loader(dataset, indices, collate, args):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [int(i) for i in indices]),
                                       batch_size=args.chunk, shuffle=False, num_workers=args.workers,
                                       collate_fn=collate)


def header_pass(dataset, indices, collate, args):
    """Per scored slice: condition, anatomy, volume, slice id and header signature (one pass over the data)."""
    meta = {'condition': [], 'anatomy': [], 'volume': [], 'slice_id': [], 'signature': []}
    for batch in loader(dataset, indices, collate, args):
        meta['condition'] += list(batch['condition'])
        meta['anatomy'] += list(batch['anatomy'])
        meta['volume'] += list(batch['fname'])
        meta['slice_id'] += batch['slice_id'].tolist()
        meta['signature'] += header_signature(batch['kspace'])
    return {key: np.array(value) if key != 'signature' else value for key, value in meta.items()}


def slices_hash(meta, indices):
    keys = '\n'.join(f'{i}:{v}:{s}' for i, v, s in zip(indices, meta['volume'], meta['slice_id']))
    return hashlib.sha1(keys.encode()).hexdigest()


def per_slice_mean(values):
    """Mean over draw seeds (the last axis), NaN only where every seed is NaN."""
    finite = np.isfinite(values)
    total = np.where(finite, values, 0.0).sum(axis=-1)
    count = finite.sum(axis=-1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


@torch.no_grad()
def score_slices(model, options, dataset, collate, indices, donors, args):
    """The per-slice arrays, [slices, draw seeds] or [slices, steps, draw seeds]."""
    count, steps, seeds = len(indices), model.num_step, len(args.draw_seeds)
    has_donor = donors >= 0
    arrays = {key: np.full((count, seeds), np.nan) for key in (*METRICS, 'realised_fraction',
                                                              *(f'swap_{m}' for m in METRICS))}
    arrays['swap_mask_equal'] = np.zeros((count, seeds), dtype=bool)
    arrays.update({key: np.full((count, steps, seeds), np.nan)
                   for key in ('step_new', 'step_budget', 'residual_rms', 'moved', 'residual_swap_rms')})
    sensitivity = model.sampler.reads_context and not args.no_sensitivity
    self_check = None
    donor_indices = indices[np.where(has_donor, donors, np.arange(count))]
    start = 0
    for batch, donor_batch in zip(loader(dataset, indices, collate, args), loader(dataset, donor_indices, collate, args)):
        part, donor = labelled(batch, options, args.device), labelled(donor_batch, options, args.device)
        rows = np.arange(start, start + len(part['kspace']))
        start += len(rows)
        matched = has_donor[rows]
        sense_maps, _ = model.sense_maps_for(part['kspace'])
        calibration, valid = model._initial_masks(part['kspace'])
        budget = (model.mask_budget(calibration, valid) / steps).float()
        for s, seed in enumerate(args.draw_seeds):
            thresholds = step_thresholds(seed, indices[rows], steps, model.shape, args.device)
            masks, details = model.sample_trajectory(part['kspace'], part['labels'], thresholds)
            previous = [calibration] + masks[:-1]
            scores = masked_slice_metrics(model.reconstruct_from_mask(part['kspace'], masks[-1], sense_maps,
                                                                      part['labels']),
                                          part['target'], part['metric_mask'])
            for metric in METRICS:
                arrays[metric][rows, s] = scores[metric].cpu().numpy()
            arrays['realised_fraction'][rows, s] = (
                (masks[-1] * valid).flatten(1).sum(1) / valid.flatten(1).sum(1)).cpu().numpy()
            arrays['step_new'][rows, :, s] = torch.stack(
                [(after - before).flatten(1).sum(1) for before, after in zip(previous, masks)], dim=1).cpu().numpy()
            arrays['step_budget'][rows, :, s] = budget.cpu().numpy()[:, None]
            if details[0].get('residual') is not None:
                arrays['residual_rms'][rows, :, s] = torch.stack(
                    [d['residual'].pow(2).flatten(1).mean(1).sqrt() for d in details], dim=1).cpu().numpy()
            if not args.no_swap:
                donor_masks, _ = model.sample_trajectory(donor['kspace'], donor['labels'], thresholds)
                donor_final = donor_masks[-1] * valid
                swapped = masked_slice_metrics(model.reconstruct_from_mask(part['kspace'], donor_final, sense_maps,
                                                                           part['labels']),
                                               part['target'], part['metric_mask'])
                for metric in METRICS:
                    arrays[f'swap_{metric}'][rows, s] = np.where(matched, swapped[metric].cpu().numpy(), np.nan)
                arrays['swap_mask_equal'][rows, s] = (donor_final == masks[-1]).flatten(1).all(1).cpu().numpy() & matched
            if sensitivity:
                for step in range(steps):
                    swap = model.counterfactual_step(part['kspace'], previous[step], step, donor['kspace'],
                                                     thresholds[step], part['labels'])
                    own_new = (masks[step] - previous[step]) > 0
                    kept = (own_new & (swap['new'] > 0)).flatten(1).sum(1).double()
                    total = own_new.flatten(1).sum(1).double()
                    moved = torch.where(total > 0, 1 - kept / total.clamp(min=1), torch.nan)
                    arrays['moved'][rows, step, s] = np.where(matched, moved.cpu().numpy(), np.nan)
                    change = (swap['residual'] - details[step]['residual']).pow(2).flatten(1).mean(1).sqrt()
                    arrays['residual_swap_rms'][rows, step, s] = np.where(matched, change.cpu().numpy(), np.nan)
                if self_check is None:
                    self_check = all(torch.equal(
                        model.counterfactual_step(part['kspace'], previous[step], step, part['kspace'],
                                                  thresholds[step], part['labels'])['new'],
                        masks[step] - previous[step]) for step in range(steps))
        print(f'  scored {start}/{count} slices', flush=True)
    return arrays, self_check


def summarise(arrays, meta, options, has_donor, self_check, args):
    conditions, volumes = meta['condition'], meta['volume']
    groups = {**{c: conditions == c for c in np.unique(conditions)}, 'all': np.ones(len(conditions), dtype=bool)}
    slice_means = {key: per_slice_mean(value) for key, value in arrays.items() if key != 'swap_mask_equal'}
    target_fraction = 1.0 / options.accelerations[0]
    summary = {
        'means': {key: {g: float(np.nanmean(slice_means[key][k])) for g, k in groups.items()}
                  for key in (*METRICS, 'realised_fraction')},
        'by_step': {key: {g: np.nanmean(slice_means[key][k], axis=0).tolist() for g, k in groups.items()}
                    for key in ('step_new', 'step_budget', 'residual_rms') if np.isfinite(slice_means[key]).any()},
        'donors': {'matched': int(has_donor.sum()),
                   'unmatched': {c: int((~has_donor & (conditions == c)).sum()) for c in np.unique(conditions)}},
    }
    pooled_ssim, realised = summary['means']['ssim']['all'], summary['means']['realised_fraction']['all']
    failed = np.flatnonzero(slice_means['psnr'] < FAILED_RECON_DB)
    summary['guards'] = {
        'ssim_pooled': pooled_ssim, 'ssim_at_least_0.80': bool(pooled_ssim >= SSIM_GUARD),
        'realised_fraction_target': target_fraction,
        'realised_fraction_within_0.002': bool(abs(realised - target_fraction) <= REALISED_TOLERANCE),
        'failed_reconstructions_below_20dB': [
            {'volume': str(volumes[i]), 'slice_id': int(meta['slice_id'][i]), 'condition': str(conditions[i]),
             'psnr': float(slice_means['psnr'][i])} for i in failed],
    }
    if not args.no_swap:
        summary['swap_cost'] = {
            metric: {g: cluster_bootstrap((slice_means[metric] - slice_means[f'swap_{metric}'])[k], volumes[k],
                                          conditions[k]) for g, k in groups.items()} for metric in METRICS}
        equal = arrays['swap_mask_equal'][has_donor]
        summary['swap_null'] = {'donor_final_mask_equals_own': float(equal.mean()) if equal.size else float('nan'),
                                'must_be_1_for_a_data_blind_twin': bool(getattr(options, 'data_blind', False))}
    if self_check is not None:
        summary['counterfactual_with_own_data_reproduces_every_step'] = self_check
    if np.isfinite(slice_means['moved']).any():
        summary['data_swap'] = {
            'moved_by_step': {g: np.nanmean(slice_means['moved'][k], axis=0).tolist() for g, k in groups.items()},
            'residual_change_rms_by_step': {g: np.nanmean(slice_means['residual_swap_rms'][k], axis=0).tolist()
                                            for g, k in groups.items()},
        }
    return summary


def figure_positions(meta, per_condition):
    """Positions (in score order) of the mid-volume slice of volumes at fixed quantiles of each condition's volumes,
    the same slices for every run scored on this val set."""
    positions = {}
    quantiles = FIGURE_VOLUME_QUANTILES[:per_condition] if per_condition <= len(FIGURE_VOLUME_QUANTILES) else \
        np.linspace(0.1, 0.9, per_condition)
    for condition in np.unique(meta['condition']):
        names = sorted(set(meta['volume'][meta['condition'] == condition]))
        for quantile in quantiles:
            volume = names[min(int(quantile * len(names)), len(names) - 1)]
            slices = np.flatnonzero(meta['volume'] == volume)
            slices = slices[np.argsort(meta['slice_id'][slices])]
            positions[f'{condition}_q{int(round(100 * quantile)):02d}'] = int(slices[len(slices) // 2])
    return positions


def render_figures(model, options, dataset, collate, indices, meta, args, out_dir):
    """Step-grid figures, {label: (path, figure)}, for `figure_positions`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    blind = getattr(options, 'data_blind', False)
    for label, position in figure_positions(meta, args.figures).items():
        part = labelled(collate([dataset[int(indices[position])]]), options, args.device)
        thresholds = step_thresholds(args.draw_seeds[0], [indices[position]], model.num_step, model.shape, args.device)
        recorder = StepRecorder(slices=[0])
        with torch.set_grad_enabled(args.gradients):
            model.sampler.requires_grad_(args.gradients)
            model.step_recorder = recorder
            pred = model(part['target'], part['kspace'], anatomy_labels=part['labels'], thresholds=thresholds)
            model.step_recorder = None
            if args.gradients:
                loss, _ = model.loss(pred, {'target': part['target'], 'kspace': part['kspace']}, LOSS_META,
                                     options.loss_type)
                loss.backward()
            model.sampler.requires_grad_(False)
            model.zero_grad(set_to_none=True)
        panels = step_panels(model, part['kspace'], part['target'], pred['sense_maps'].detach(), pred['masks'], 0,
                             part['labels'], part['metric_mask'], recorder)
        title = (f"{args.name} ({'data-blind twin' if blind else options.sampler_type}, K={model.num_step}), "
                 f"{meta['condition'][position]} {meta['volume'][position]} slice {meta['slice_id'][position]}, "
                 f"draw seed {args.draw_seeds[0]}. Rows before the last reconstruct masks the reconstructor never "
                 "trained on")
        figure = step_grid_figure(panels, title)
        path = out_dir / f'{label}.png'
        figure.savefig(path)
        figures[label] = (path, figure)
    return figures


def flat_scalars(tree, prefix):
    """{'prefix/a/b': number} for every number in a nested dict, lists by index, a CI as _lo/_hi."""
    flat = {}
    for key, value in tree.items():
        name = f'{prefix}/{key}'
        if isinstance(value, dict):
            flat.update(flat_scalars(value, name))
        elif isinstance(value, (bool, int, float)) and not isinstance(value, str):
            flat[name] = float(value)
        elif isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            if key == 'ci95':
                flat[f'{prefix}/ci95_lo'], flat[f'{prefix}/ci95_hi'] = value
            else:
                flat.update({f'{name}/{i}': float(v) for i, v in enumerate(value)})
    return flat


def score(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, checkpoint = load_run(args.run, args.checkpoint, args.device)
    options = checkpoint['options']
    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = args.workers
    if args.val_half is not None:
        cfg.datasets.val_half = args.val_half
    _, val_loader = build_dataloaders(cfg)
    dataset, collate = val_loader.dataset, val_loader.collate_fn
    indices = np.arange(len(dataset)) if args.limit is None else \
        np.unique(np.linspace(0, len(dataset) - 1, args.limit).astype(int))
    print(f"{args.name}: {args.run / args.checkpoint} epoch {checkpoint['epoch']}, {options.sampler_type}, "
          f"K={model.num_step}, data_blind={getattr(options, 'data_blind', False)}, {len(indices)} slices", flush=True)
    meta = header_pass(dataset, indices, collate, args)
    donors = match_donors(meta['condition'], meta['volume'], meta['signature'])
    arrays, self_check = score_slices(model, options, dataset, collate, indices, donors, args)
    summary = summarise(arrays, meta, options, donors >= 0, self_check, args)
    run_info_path = args.run / 'run_info.json'
    summary = {
        'name': args.name, 'run': str(args.run), 'checkpoint': args.checkpoint, 'epoch': int(checkpoint['epoch']),
        'training_code': json.loads(run_info_path.read_text()) if run_info_path.exists() else None,
        'eval_code': git_state(REPO),
        'options': {key: getattr(options, key, None) for key in (
            'sampler_type', 'num_step', 'data_blind', 'accelerations', 'budget_over', 'seed', 'residual_chans',
            'residual_depth', 'residual_limit', 'residual_lr', 'anatomy_conditioning', 'loss_type')},
        'draw_seeds': list(args.draw_seeds), 'slices': {'count': len(indices), 'hash': slices_hash(meta, indices)},
        **summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(args.out / f'{args.name}.npz', dataset_index=indices, donor=donors,
                        draw_seeds=np.array(args.draw_seeds), **{k: v for k, v in meta.items() if k != 'signature'},
                        **arrays)
    psnr = summary['means']['psnr']
    print('  PSNR_Masked ' + '  '.join(f'{g} {v:.3f}' for g, v in psnr.items()), flush=True)
    for key in ('donors', 'swap_null', 'counterfactual_with_own_data_reproduces_every_step', 'guards'):
        if key in summary:
            value = summary[key]
            print(f'  {key}: ' + json.dumps({k: v for k, v in value.items() if k != 'failed_reconstructions_below_20dB'}
                                            if isinstance(value, dict) else value), flush=True)
    if 'data_swap' in summary:
        print('  data-swap moved by step (all): ' + ' '.join(
            f'{v:.3f}' for v in summary['data_swap']['moved_by_step']['all']), flush=True)
    figures = render_figures(model, options, dataset, collate, indices, meta, args, args.out / 'figures' / args.name) \
        if args.figures > 0 else {}
    if not args.no_wandb:
        import wandb
        wandb.init(project='SeqMRI', name=f'eval_{args.name}', job_type='eval', group='multistep_adaptivity',
                   config={'run': str(args.run), 'checkpoint': args.checkpoint, **summary['options']},
                   dir=str(args.out))
        wandb.log({**flat_scalars({key: summary[key] for key in ('means', 'guards', 'swap_cost', 'swap_null',
                                                                   'data_swap', 'by_step', 'donors') if key in summary},
                                  'eval'),
                   **{f'eval/figures/{label}': wandb.Image(str(path)) for label, (path, _) in figures.items()}})
        wandb.finish()
    for _, figure in figures.values():
        plt.close(figure)


def contrast(args):
    pairs = []
    for spec in args.pair:
        label, _, difference = spec.partition('=')
        minuend, _, subtrahend = difference.partition('-')
        if not (label and minuend and subtrahend):
            raise SystemExit(f'--pair {spec!r}: expected LABEL=A-B')
        pairs.append((label, minuend, subtrahend))
    names = sorted({name for _, a, b in pairs for name in (a, b)})
    scores = {name: np.load(args.scores / f'{name}.npz', allow_pickle=False) for name in names}
    summaries = {name: json.loads((args.scores / f'{name}.json').read_text()) for name in names}
    reference = summaries[names[0]]
    for name in names[1:]:
        if summaries[name]['slices'] != reference['slices'] or summaries[name]['draw_seeds'] != reference['draw_seeds']:
            raise SystemExit(f'{name} was scored on other slices or draw seeds than {names[0]}; they cannot be paired')
    conditions, volumes = scores[names[0]]['condition'], scores[names[0]]['volume']
    groups = {**{c: conditions == c for c in np.unique(conditions)}, 'all': np.ones(len(conditions), dtype=bool)}
    results = {}
    for label, minuend, subtrahend in pairs:
        result = {'difference': f'{minuend} - {subtrahend}',
                  'num_step': [summaries[minuend]['options']['num_step'], summaries[subtrahend]['options']['num_step']],
                  'data_blind': [summaries[minuend]['options']['data_blind'],
                                 summaries[subtrahend]['options']['data_blind']]}
        for metric in METRICS:
            delta = per_slice_mean(scores[minuend][metric]) - per_slice_mean(scores[subtrahend][metric])
            result[metric] = {g: cluster_bootstrap(delta[k], volumes[k], conditions[k]) for g, k in groups.items()}
        results[label] = result
        print(f'{label:16s} PSNR ' + '  '.join(
            f"{g} {r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" for g, r in result['psnr'].items()),
            flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'scores': str(args.scores), 'slices': reference['slices'],
                                    'draw_seeds': reference['draw_seeds'], 'eval_code': git_state(REPO),
                                    'contrasts': results}, indent=2) + '\n')


if __name__ == '__main__':
    parsed = parse_args()
    score(parsed) if parsed.command == 'score' else contrast(parsed)
