"""Same global -> class -> slice mask ladder as slice_adaptivity_gap.py, with CG-SENSE as the reconstructor.

Reuses the optimiser, mask parametrisation, and the exact val slices of the VarNet run (its selection.json),
so the two gaps are directly comparable and any disagreement is down to the reconstructor. Masks are
fitted through 10 backpropagated CG iterations and scored with 32.

    CUDA_VISIBLE_DEVICES=3 python scripts/diagnostics/slice_adaptivity_gap_cgsense.py \\
        --selection runs/slice_adaptivity/selection.json --out-dir runs/slice_adaptivity_cgsense
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.samplers.oracle import estimate_sense_maps
from slice_adaptivity_gap import (
    ANATOMIES, SUPPORT_QUANTILE_EDGES, _train_anatomies, centre_block, db_gain, fit_scores, load_frozen_model,
    masked_errors, repeat_forever, repeat_forever_fixed, standardised_scores, straight_through_mask,
    summarise_transfer, to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=Path, default=Path('checkpoints/loupe2d_varnet_r8_brain_knee/best_model.pt'),
                        help='only supplies the initial scores, centre block and budget')
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--slice-steps', type=int, default=100)
    parser.add_argument('--class-steps', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--slice-batch', type=int, default=2)
    parser.add_argument('--class-batch', type=int, default=4)
    parser.add_argument('--fit-iters', type=int, default=10)
    parser.add_argument('--score-iters', type=int, default=32)
    parser.add_argument('--lam', type=float, default=1e-2)
    parser.add_argument('--acs', type=int, default=24)
    parser.add_argument('--limit', type=int, default=0, help='only the first N selected slices (smoke tests)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def cgsense_nmse_fn(iters, lam, acs):
    def nmse(kspace, target, mask):
        kspace_complex = torch.view_as_complex(kspace.contiguous())
        with torch.no_grad():
            maps = torch.stack([estimate_sense_maps(k, acs=acs)[0] for k in kspace_complex])
        recon = conjugate_gradient_sense(kspace_complex * mask, maps, mask, num_iters=iters, lamda=lam)
        return masked_errors(recon.abs().unsqueeze(1), target)[1]
    return nmse


def fit_on_train(nmse_fn, k, train_dataset, indices, collate_fn, centre, init, args, device):
    loader = DataLoader(Subset(train_dataset, indices), batch_size=args.class_batch, shuffle=True,
                        num_workers=4, collate_fn=collate_fn, drop_last=True)
    return fit_scores(nmse_fn, k, repeat_forever(loader, device), init, centre, args, args.class_steps,
                      per_example=False)


def nmse_of_masks(score_fn, sample, masks, device, chunk=5):
    kspace, target = to_device(sample, device)
    errors = []
    with torch.no_grad():
        for part in masks.split(chunk):
            count = part.shape[0]
            errors.append(score_fn(kspace.expand(count, -1, -1, -1, -1).contiguous(),
                                   target.expand(count, -1, -1).contiguous(), part.to(device)))
    return torch.cat(errors)


def transfer_rows(score_fn, val_dataset, collate_fn, support, selection, class_masks, slice_masks, device):
    positions = {(str(f), int(i)): n for n, (f, i) in enumerate(zip(support['fname'], support['slice_id']))}
    rows = []
    for j, entry in enumerate(selection):
        same = [i for i, other in enumerate(selection) if other['anatomy'] == entry['anatomy']]
        errors = nmse_of_masks(score_fn, collate_fn([val_dataset[entry['index']]]),
                               torch.cat([class_masks[j:j + 1], slice_masks[same]]), device)
        gains = 10 * torch.log10(errors[0] / errors[1:]).cpu().numpy()
        row = {'target': j, 'anatomy': entry['anatomy'], 'bin': entry['bin'],
               'gain_from': {int(i): float(g) for i, g in zip(same, gains)}, 'neighbour_gains': []}
        for offset in (-1, 1):
            neighbour = positions.get((entry['fname'], entry['slice_id'] + offset))
            if neighbour is None:
                continue
            errors = nmse_of_masks(score_fn, collate_fn([val_dataset[neighbour]]),
                                   torch.cat([class_masks[j:j + 1], slice_masks[j:j + 1]]), device)
            row['neighbour_gains'].append(float(10 * torch.log10(errors[0] / errors[1])))
        rows.append(row)
        print(f'transfer {j + 1}/{len(selection)}', flush=True)
    return rows


def summarise(results, out_dir, transfer_lines):
    comparisons = [('class vs global', 'class', 'global'), ('slice vs class', 'slice', 'class'),
                   ('slice vs global', 'slice', 'global'), ('global vs VarNet-shared', 'global', 'shared')]
    lines = []
    for anatomy in (*ANATOMIES, 'all'):
        rows = [r for r in results if anatomy in (r['anatomy'], 'all')]
        lines.append(f'\n[{anatomy}] n={len(rows)}')
        for bin_index in (None, *range(len(SUPPORT_QUANTILE_EDGES) - 1)):
            subset = [r for r in rows if bin_index is None or r['bin'] == bin_index]
            if not subset:
                continue
            area = np.mean([r['area'] for r in subset])
            label = 'all bins' if bin_index is None else f'bin {bin_index}'
            cells = []
            for title, better, worse in comparisons:
                gain = db_gain(subset, better, worse, metric='nmse')
                cells.append(f'{title}: {gain.mean():+.3f} dB ({(gain > 0).mean():.0%} better)')
            lines.append(f'  {label:9} area {area:.3f} n={len(subset):2d} | ' + ' | '.join(cells))
    text = '\n'.join(lines + transfer_lines)
    (out_dir / 'summary.txt').write_text(text)
    print(text)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    torch.manual_seed(args.seed)

    train_loader, val_loader = build_fastmri_dataloaders(
        args.data_root, list(ANATOMIES), [256, 256], 8, num_coils=16, num_workers=0, contrast='T2', balance=True)
    collate_fn, val_dataset = val_loader.collate_fn, val_loader.dataset
    model = load_frozen_model(args.ckpt, device)
    centre = centre_block(model, device)
    k = round(model.budget)
    shared = standardised_scores(model, centre)
    allowed = centre[:, 0] == 0
    fit_fn = cgsense_nmse_fn(args.fit_iters, args.lam, args.acs)
    score_fn = cgsense_nmse_fn(args.score_iters, args.lam, args.acs)
    selection = json.loads(args.selection.read_text())
    if args.limit:
        selection = selection[:args.limit]

    sample = collate_fn([val_dataset[selection[0]['index']]])
    kspace, target = to_device(sample, device)
    with torch.no_grad():
        full = score_fn(kspace, target, torch.ones(1, 1, 256, 256, device=device))
    print(f'sanity: fully sampled CG-SENSE NMSE vs target {full.item():.5f}', flush=True)

    train_anatomy = np.array(list(_train_anatomies(train_loader.dataset)))
    print(f'fitting global scores on {len(train_anatomy)} train slices', flush=True)
    fitted = {'global': fit_on_train(fit_fn, k, train_loader.dataset, list(range(len(train_anatomy))),
                                     collate_fn, centre, shared, args, device)}
    for anatomy in ANATOMIES:
        indices = np.flatnonzero(train_anatomy == anatomy).tolist()
        print(f'fitting {anatomy} class scores on {len(indices)} train slices', flush=True)
        fitted[anatomy] = fit_on_train(fit_fn, k, train_loader.dataset, indices, collate_fn, centre,
                                       fitted['global'], args, device)

    results, class_masks, slice_masks = [], [], []
    for start in range(0, len(selection), args.slice_batch):
        group = selection[start:start + args.slice_batch]
        batch = collate_fn([val_dataset[entry['index']] for entry in group])
        kspace, target = to_device(batch, device)
        count = len(group)
        class_init = torch.cat([fitted[a] for a in batch['anatomy']])
        slice_scores = fit_scores(fit_fn, k, repeat_forever_fixed(kspace, target), class_init, centre, args,
                                  args.slice_steps, per_example=True)
        scores = {'shared': shared.expand(count, -1, -1), 'global': fitted['global'].expand(count, -1, -1),
                  'class': class_init, 'slice': slice_scores}
        with torch.no_grad():
            nmse = {name: score_fn(kspace, target, straight_through_mask(s, allowed, k, 1.0, centre).detach())
                    for name, s in scores.items()}
        with torch.no_grad():
            class_masks.append(straight_through_mask(class_init, allowed, k, 1.0, centre).detach().cpu())
            slice_masks.append(straight_through_mask(slice_scores, allowed, k, 1.0, centre).detach().cpu())
        for row, entry in enumerate(group):
            results.append({**entry, **{name: {'nmse': value[row].item()} for name, value in nmse.items()}})
        print(f'{len(results)}/{len(selection)} slices done', flush=True)

    (args.out_dir / 'results.json').write_text(json.dumps(results, indent=1))
    support = np.load(args.selection.parent / 'support.npz', allow_pickle=True)
    transfer = transfer_rows(score_fn, val_dataset, collate_fn, support, selection, torch.cat(class_masks),
                             torch.cat(slice_masks), device)
    (args.out_dir / 'transfer.json').write_text(json.dumps(transfer, indent=1))
    summarise(results, args.out_dir, summarise_transfer(transfer, [(e['index'], e['bin']) for e in selection]))


if __name__ == '__main__':
    main()
