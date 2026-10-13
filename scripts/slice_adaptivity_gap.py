"""How much does adapting a k-space mask to a slice buy, with the reconstructor held fixed?

Step 0 of reports/ADAPTIVE_MASK_CONDITIONING.md, measured through a frozen VarNet (the one trained
jointly with the shared LOUPE mask) rather than CG-SENSE. Three rungs, all optimised with the same
procedure (hard top-k forward, sigmoid-relaxed backward, Adam on the scores) and the same budget:

  shared  top-k of the trained LOUPE logits: one mask for every slice
  class   scores fitted on train slices of one anatomy, starting from the shared logits
  slice   scores fitted on the evaluated slice itself, starting from the class (or shared) scores

Gaps are reported in dB of masked MSE (equal to a PSNR difference at fixed data range), per slice,
stratified by sense-support area. The slice rung is scored on the slice it was fitted to, so it is an
oracle: an upper bound on what a perfect per-slice mask predictor could win under this VarNet.

    CUDA_VISIBLE_DEVICES=3 python scripts/slice_adaptivity_gap.py --out-dir runs/slice_adaptivity
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.training.trainer import _apply_signal_mask, _signal_mask, build_model

ANATOMIES = ('brain', 'knee')
SUPPORT_QUANTILE_EDGES = (0.0, 0.05, 0.2, 0.5, 0.8, 1.0)
TRACE_STEPS = (10, 25, 50, 100)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=Path, default=Path('checkpoints/loupe2d_varnet_r8_brain_knee/best_model.pt'))
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--per-bin', type=int, default=3, help='slices per support bin per anatomy')
    parser.add_argument('--slice-steps', type=int, default=100)
    parser.add_argument('--class-steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=0.3, help='relaxation width, in std of the shared scores')
    parser.add_argument('--slice-batch', type=int, default=2)
    parser.add_argument('--class-batch', type=int, default=4)
    parser.add_argument('--skip-shared-init', action='store_true',
                        help='only fit slice masks from the class scores, not also from the shared scores')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_frozen_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval().requires_grad_(False)
    return model


def to_device(batch, device):
    return batch['kspace'].to(device), batch['target'].to(device)


def reconstruct(model, target, kspace, mask):
    """Reconstruction through the frozen VarNet with `mask` ([B, 1, H, W], centre block included)."""
    model.mask_policy = lambda _: mask
    try:
        return model(target, kspace)['output']
    finally:
        model.mask_policy = None


def masked_errors(output, target):
    """Per-slice masked MSE, NMSE and PSNR with the trainer's signal mask."""
    target = target.unsqueeze(1)
    masked_output, masked_target = _apply_signal_mask(output, target)
    squared = (masked_output - masked_target).pow(2).flatten(1)
    mse = squared.mean(1)
    nmse = squared.sum(1) / masked_target.pow(2).flatten(1).sum(1).clamp(min=1e-12)
    psnr = 10 * torch.log10(target.flatten(1).max(1).values.pow(2) / mse)
    return mse, nmse, psnr


def centre_block(model, device):
    return model._init_mask(torch.zeros(1, 1, *model.shape, 2, device=device))


def standardised_scores(model, centre):
    """The shared LOUPE logits, shifted so the top-k threshold sits at 0 and scaled to unit std."""
    logits = model.sampler.mask_net.raw_logits().reshape(1, *model.shape).detach()
    allowed = centre[:, 0] == 0
    threshold = logits[allowed].topk(round(model.budget)).values[-1]
    return (logits - threshold) / logits[allowed].std()


def straight_through_mask(scores, allowed, k, tau, centre):
    """Hard top-k of `scores` over `allowed`; gradient flows through a sigmoid around the k-th score."""
    flat = scores.masked_fill(~allowed, float('-inf')).flatten(1)
    top = flat.topk(k, dim=1)
    hard = torch.zeros_like(flat).scatter_(1, top.indices, 1.0).view_as(scores)
    threshold = top.values[:, -1].detach().view(-1, 1, 1)
    soft = torch.sigmoid((scores - threshold) / tau) * allowed
    return (hard + soft - soft.detach()).unsqueeze(1) + centre


def fit_scores(nmse_fn, k, batches, init_scores, centre, args, steps, per_example, trace=None):
    """Adam on the scores, with batches drawn from the `batches` generator.

    Per-example fits return, for each row, the iterate with the lowest loss on its own slice.
    The shared-row (class) fit returns its last iterate, since minibatch losses are too noisy to select on.
    `nmse_fn(kspace, target, mask)` is the reconstructor under study: per-slice NMSE from a [B, 1, H, W] mask.
    The learning rate decays by cosine to a tenth. `trace`, if given, receives the best NMSE so far at TRACE_STEPS.
    """
    allowed = (centre[:, 0] == 0)
    scores = init_scores.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([scores], lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps, eta_min=0.1 * args.lr)
    best_nmse = torch.full((init_scores.shape[0],), float('inf'), device=init_scores.device)
    best_scores = init_scores.clone()
    for step in range(steps):
        kspace, target = next(batches)
        rows = scores if per_example else scores.expand(kspace.shape[0], -1, -1)
        with torch.enable_grad():
            mask = straight_through_mask(rows, allowed, k, args.tau, centre)
            nmse = nmse_fn(kspace, target, mask)
            loss = nmse.sum() if per_example else nmse.mean()
            optimiser.zero_grad()
            loss.backward()
        if per_example:
            improved = nmse.detach() < best_nmse
            best_nmse = torch.where(improved, nmse.detach(), best_nmse)
            best_scores[improved] = scores.detach()[improved]
        optimiser.step()
        schedule.step()
        if trace is not None and step + 1 in TRACE_STEPS:
            trace[step + 1] = best_nmse.clone()
    return best_scores if per_example else scores.detach()


def varnet_nmse_fn(model):
    return lambda kspace, target, mask: masked_errors(reconstruct(model, target, kspace, mask), target)[1]


def repeat_forever(loader, device):
    while True:
        for batch in loader:
            yield to_device(batch, device)


def collect_support(model, dataset, collate_fn, device, cache):
    if cache.exists():
        loaded = np.load(cache, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    rows = {'area': [], 'valid': [], 'anatomy': [], 'fname': [], 'slice_id': []}
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            kspace, _ = to_device(batch, device)
            _, sense_mask = model.sensetivity_net(torch.view_as_complex(kspace.contiguous()),
                                                  mask_method=model.sense_mask_method)
            rows['area'] += sense_mask.flatten(1).float().mean(1).cpu().tolist()
            rows['valid'] += model._valid_kspace_mask(kspace).flatten(1).mean(1).cpu().tolist()
            rows['anatomy'] += batch['anatomy']
            rows['fname'] += batch['fname']
            rows['slice_id'] += batch['slice_id'].tolist()
    support = {key: np.array(value) for key, value in rows.items()}
    np.savez(cache, **support)
    return support


def select_slices(support, per_bin, rng):
    """`per_bin` slices from each support-area quantile bin of each anatomy, from distinct volumes where possible."""
    chosen = []
    for anatomy in ANATOMIES:
        indices = np.flatnonzero(support['anatomy'] == anatomy)
        areas = support['area'][indices]
        edges = np.quantile(areas, SUPPORT_QUANTILE_EDGES)
        for bin_index in range(len(edges) - 1):
            in_bin = (areas >= edges[bin_index]) & (areas <= edges[bin_index + 1])
            candidates = rng.permutation(indices[in_bin])
            seen, picked = set(), []
            for index in candidates:
                if support['fname'][index] not in seen:
                    seen.add(support['fname'][index])
                    picked.append(int(index))
                if len(picked) == per_bin:
                    break
            chosen += [(index, bin_index) for index in picked]
    return chosen


def fit_on_train(model, train_dataset, indices, collate_fn, centre, init, args, device):
    loader = DataLoader(Subset(train_dataset, indices), batch_size=args.class_batch, shuffle=True,
                        num_workers=4, collate_fn=collate_fn, drop_last=True)
    return fit_scores(varnet_nmse_fn(model), round(model.budget), repeat_forever(loader, device), init, centre,
                      args, args.class_steps, per_example=False)


def train_scores(model, train_dataset, collate_fn, centre, shared, args, device):
    """Global scores fitted on every train slice, then per-anatomy scores continued from them."""
    train_anatomy = np.array(list(_train_anatomies(train_dataset)))
    print(f'fitting global scores on {len(train_anatomy)} train slices', flush=True)
    fitted = {'global': fit_on_train(model, train_dataset, list(range(len(train_anatomy))), collate_fn,
                                     centre, shared, args, device)}
    for anatomy in ANATOMIES:
        indices = np.flatnonzero(train_anatomy == anatomy).tolist()
        print(f'fitting {anatomy} class scores on {len(indices)} train slices', flush=True)
        fitted[anatomy] = fit_on_train(model, train_dataset, indices, collate_fn, centre, fitted['global'],
                                       args, device)
    return fitted


def _train_anatomies(concat_dataset):
    for name, size in zip(ANATOMIES, np.diff([0, *concat_dataset.cumulative_sizes])):
        yield from [name] * int(size)


def mask_dice(a, b, centre):
    a, b = a * (1 - centre), b * (1 - centre)
    return (2 * (a * b).sum() / (a.sum() + b.sum())).item()


def evaluate_slices(model, dataset, collate_fn, selection, shared, fitted, centre, args, device):
    results, saved_masks = [], []
    k = round(model.budget)
    allowed = (centre[:, 0] == 0)
    for start in range(0, len(selection), args.slice_batch):
        group = selection[start:start + args.slice_batch]
        batch = collate_fn([dataset[index] for index, _ in group])
        kspace, target = to_device(batch, device)
        anatomies = batch['anatomy']
        valid = model._valid_kspace_mask(kspace)
        class_init = torch.cat([fitted[a] for a in anatomies])

        trace = {}

        def run_fit(init, trace=None):
            return fit_scores(varnet_nmse_fn(model), k, repeat_forever_fixed(kspace, target), init, centre,
                              args, args.slice_steps, per_example=True, trace=trace)

        scores = {
            'shared': shared.expand(len(group), -1, -1),
            'global': fitted['global'].expand(len(group), -1, -1),
            'class': class_init,
        }
        scores['slice_from_class'] = run_fit(class_init, trace)
        if not args.skip_shared_init:
            scores['slice_from_shared'] = run_fit(shared.expand(len(group), -1, -1).contiguous())

        with torch.no_grad():
            masks = {name: straight_through_mask(s, allowed, k, 1.0, centre).detach()
                     for name, s in scores.items()}
            draw = model(target, kspace)['output']
            errors = {name: masked_errors(reconstruct(model, target, kspace, mask), target)
                      for name, mask in masks.items()}
            errors['model_draw'] = masked_errors(draw, target)
        for row, (index, bin_index) in enumerate(group):
            entry = {'index': index, 'bin': bin_index, 'anatomy': anatomies[row], 'fname': batch['fname'][row],
                     'slice_id': int(batch['slice_id'][row]), 'valid_fraction': valid[row].mean().item(),
                     'mask_points': {name: masks[name][row].sum().item() for name in masks}}
            for name, (mse, nmse, psnr) in errors.items():
                entry[name] = {'mse': mse[row].item(), 'nmse': nmse[row].item(), 'psnr': psnr[row].item()}
            entry['trace_nmse'] = {step: values[row].item() for step, values in trace.items()}
            entry['dice_slice_vs_class'] = mask_dice(masks['slice_from_class'][row], masks['class'][row], centre[0])
            entry['dice_class_vs_global'] = mask_dice(masks['class'][row], masks['global'][row], centre[0])
            results.append(entry)
        saved_masks.append({name: masks[name].cpu() for name in ('class', 'slice_from_class')})
        print(f'{len(results)}/{len(selection)} slices done', flush=True)
    return results, saved_masks


def mse_of_masks(model, sample, masks, device, chunk=5):
    """Masked MSE of one slice reconstructed under each of `masks` [M, 1, H, W]."""
    kspace, target = to_device(sample, device)
    errors = []
    with torch.no_grad():
        for part in masks.split(chunk):
            count = part.shape[0]
            errors.append(masked_errors(reconstruct(model, target.expand(count, -1, -1).contiguous(),
                                                    kspace.expand(count, -1, -1, -1, -1).contiguous(),
                                                    part.to(device)), target.expand(count, -1, -1))[0])
    return torch.cat(errors)


def transfer_gains(model, dataset, collate_fn, support, selection, results, saved_masks, device):
    """Does a slice-fitted mask help slices it was not fitted on, relative to the class mask?

    For each selected slice j: gain (dB) of every same-anatomy slice-fitted mask i on slice j, plus the
    gain of j's own mask on its neighbours j-1 and j+1 in the same volume.
    """
    positions = {(str(f), int(i)): n for n, (f, i) in enumerate(zip(support['fname'], support['slice_id']))}
    slice_masks = torch.cat([m['slice_from_class'] for m in saved_masks])
    class_masks = torch.cat([m['class'] for m in saved_masks])
    rows = []
    for j, (index, bin_index) in enumerate(selection):
        anatomy = results[j]['anatomy']
        same = [i for i in range(len(selection)) if results[i]['anatomy'] == anatomy]
        sample = collate_fn([dataset[index]])
        errors = mse_of_masks(model, sample, torch.cat([class_masks[j:j + 1], slice_masks[same]]), device)
        gains = 10 * torch.log10(errors[0] / errors[1:]).cpu().numpy()
        row = {'target': j, 'anatomy': anatomy, 'bin': bin_index,
               'gain_from': {int(i): float(g) for i, g in zip(same, gains)}, 'neighbour_gains': []}
        for offset in (-1, 1):
            neighbour = positions.get((str(support['fname'][index]), int(support['slice_id'][index]) + offset))
            if neighbour is None:
                continue
            errors = mse_of_masks(model, collate_fn([dataset[neighbour]]),
                                  torch.cat([class_masks[j:j + 1], slice_masks[j:j + 1]]), device)
            row['neighbour_gains'].append(float(10 * torch.log10(errors[0] / errors[1])))
        rows.append(row)
        print(f'transfer {j + 1}/{len(selection)}', flush=True)
    return rows


def summarise_transfer(rows, selection):
    lines = ['\nTransfer of slice-fitted masks (gain in dB over the class mask, on the slice named)']
    for anatomy in (*ANATOMIES, 'all'):
        subset = [r for r in rows if anatomy in (r['anatomy'], 'all')]
        own = [r['gain_from'][r['target']] for r in subset]
        neighbour = [g for r in subset for g in r['neighbour_gains']]
        same_bin, other_bin = [], []
        for r in subset:
            for i, gain in r['gain_from'].items():
                if i == r['target']:
                    continue
                (same_bin if selection[i][1] == r['bin'] else other_bin).append(gain)
        lines.append(f'  [{anatomy}] own {np.mean(own):+.3f} | adjacent slices {np.mean(neighbour):+.3f} '
                     f'(n={len(neighbour)}) | other slices, same support bin {np.mean(same_bin):+.3f} '
                     f'(n={len(same_bin)}) | other slices, different bin {np.mean(other_bin):+.3f} '
                     f'(n={len(other_bin)})')
    return lines


def repeat_forever_fixed(kspace, target):
    while True:
        yield kspace, target


def db_gain(results, better, worse, metric='mse'):
    return np.array([10 * np.log10(r[worse][metric] / r[better][metric]) for r in results])


def summarise(results, support, out_dir, transfer_lines):
    comparisons = [('global vs shared', 'global', 'shared'), ('class vs global', 'class', 'global'),
                   ('slice vs class', 'slice_from_class', 'class'), ('slice vs global', 'slice_from_class', 'global')]
    lines = []
    for anatomy in (*ANATOMIES, 'all'):
        rows = [r for r in results if anatomy in (r['anatomy'], 'all')]
        lines.append(f'\n[{anatomy}] n={len(rows)}  mean support area '
                     f'{np.mean([support["area"][r["index"]] for r in rows]):.3f}')
        for bin_index in (None, *range(len(SUPPORT_QUANTILE_EDGES) - 1)):
            subset = [r for r in rows if bin_index is None or r['bin'] == bin_index]
            if not subset:
                continue
            area = np.mean([support['area'][r['index']] for r in subset])
            label = 'all bins' if bin_index is None else f'bin {bin_index}'
            cells = []
            for title, better, worse in comparisons:
                gain = db_gain(subset, better, worse)
                cells.append(f'{title}: {gain.mean():+.3f} dB ({(gain > 0).mean():.0%} better)')
            lines.append(f'  {label:9} area {area:.3f} n={len(subset):2d} | ' + ' | '.join(cells))
    draws = db_gain(results, 'shared', 'model_draw')
    lines.append(f'\nshared top-k vs the model\'s own Bernoulli draw: {draws.mean():+.3f} dB '
                 f'(positive = top-k better)')
    lines.append(f'global vs the model\'s own Bernoulli draw: {db_gain(results, "global", "model_draw").mean():+.3f} dB')
    trace = {step: np.mean([10 * np.log10(r['trace_nmse'][step]) for r in results]) for step in TRACE_STEPS}
    lines.append('mean slice-fit NMSE (dB) after 10/25/50/100 steps: '
                 + ' / '.join(f'{trace[step]:.2f}' for step in TRACE_STEPS))
    lines.append(f'mean PSNR (data range = slice max): shared top-k '
                 f'{np.mean([r["shared"]["psnr"] for r in results]):.2f}, draw '
                 f'{np.mean([r["model_draw"]["psnr"] for r in results]):.2f}')
    lines.append(f'mean Dice slice-vs-class {np.mean([r["dice_slice_vs_class"] for r in results]):.3f}, '
                 f'class-vs-global {np.mean([r["dice_class_vs_global"] for r in results]):.3f} '
                 f'(over the free points, excluding the centre block)')
    text = '\n'.join(lines + transfer_lines)
    (out_dir / 'summary.txt').write_text(text)
    print(text)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, anatomy in zip(axes, ANATOMIES):
        rows = [r for r in results if r['anatomy'] == anatomy]
        area = [support['area'][r['index']] for r in rows]
        axis.scatter(area, db_gain(rows, 'class', 'global'), label='class vs global')
        axis.scatter(area, db_gain(rows, 'slice_from_class', 'class'), label='slice vs class')
        axis.axhline(0, color='k', lw=0.5)
        axis.set_title(anatomy)
        axis.set_xlabel('sense-support area (fraction of image)')
    axes[0].set_ylabel('gain (dB masked MSE)')
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(out_dir / 'gain_vs_support.png', dpi=120)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_loader, val_loader = build_fastmri_dataloaders(
        args.data_root, list(ANATOMIES), [256, 256], 8, num_coils=16, num_workers=0, contrast='T2', balance=True)
    collate_fn = val_loader.collate_fn
    model = load_frozen_model(args.ckpt, device)
    centre = centre_block(model, device)
    print(f'budget {model.budget:.0f} free points + {int(centre.sum())} centre points', flush=True)

    support = collect_support(model, val_loader.dataset, collate_fn, device, args.out_dir / 'support.npz')
    print(f'padded k-space: {(support["valid"] < 0.999).mean():.1%} of val slices have valid fraction < 1')
    selection = select_slices(support, args.per_bin, rng)
    (args.out_dir / 'selection.json').write_text(json.dumps(
        [{'index': i, 'bin': b, 'anatomy': str(support['anatomy'][i]), 'fname': str(support['fname'][i]),
          'slice_id': int(support['slice_id'][i]), 'area': float(support['area'][i])} for i, b in selection],
        indent=1))

    shared = standardised_scores(model, centre)
    fitted = train_scores(model, train_loader.dataset, collate_fn, centre, shared, args, device)
    results, saved_masks = evaluate_slices(model, val_loader.dataset, collate_fn, selection, shared, fitted,
                                           centre, args, device)
    (args.out_dir / 'results.json').write_text(json.dumps(results, indent=1))
    torch.save({'scores': {name: tensor.cpu() for name, tensor in fitted.items()}, 'shared': shared.cpu(),
                'masks': saved_masks}, args.out_dir / 'tensors.pt')
    transfer = transfer_gains(model, val_loader.dataset, collate_fn, support, selection, results, saved_masks,
                              device)
    (args.out_dir / 'transfer.json').write_text(json.dumps(transfer, indent=1))
    summarise(results, support, args.out_dir, summarise_transfer(transfer, selection))


if __name__ == '__main__':
    main()
