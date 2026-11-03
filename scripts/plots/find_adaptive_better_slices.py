"""Validation slices where the anatomy-conditioned (adaptive) LOUPE beats Poisson and plain LOUPE for BOTH reconstructors.

Each candidate slice is reconstructed with the CG-SENSE R=4 models and the PromptMR VarNet R=8 runs, exactly as the two mask-comparison
figures do (same masks, seeds, sense thresholds and metrics). A slice qualifies for a condition when, for both reconstructors, the
adaptive LOUPE has higher PSNR and SSIM than Poisson and than the single-mask LOUPE. Qualifying slices are ranked by their smallest PSNR margin.

    python scripts/plots/find_adaptive_better_slices.py --device cpu
Writes <out-dir>/candidates.json and prints the --pin strings for plot_cg_mask_comparison.py.
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import torch

import _path  # noqa: F401
import plot_cg_mask_comparison as cg
import plot_promptmr_mask_comparison as vn
from activemri.samplers.poisson import poisson_mask
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import anatomy_labels_from_batch

MARGIN_KEYS = ('poisson', 'loupe')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/adaptive_better_slices'))
    parser.add_argument('--per-anatomy', type=int, default=40)
    parser.add_argument('--middle', type=float, default=1 / 3)
    parser.add_argument('--cg-sense-threshold', type=float, default=0.02)
    parser.add_argument('--checkpoint', default='epoch20_model.pt')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


@torch.no_grad()
def cg_scores(models, options, poisson, sample, device):
    batch = {'kspace': sample['kspace'][None], 'target': sample['target'][None], 'anatomy': [sample['anatomy']], 'condition': [sample['condition']]}
    dev = cg.to_device(batch, device)
    labels = {n: anatomy_labels_from_batch(batch, options[n].label_classes, device, options[n].label_key) for n in ('loupe', 'loupe_anat')}
    masks = {n: cg.loupe_masks(models[n], dev, labels[n], 0) for n in ('loupe', 'loupe_anat')}  # the seed --pin uses
    masks['poisson'] = cg.fixed_mask(models['loupe'], dev['kspace'], poisson.to(device))
    scores = {}
    for name in ('poisson', 'loupe', 'loupe_anat'):
        model, label = (models['loupe'], labels['loupe']) if name == 'poisson' else (models[name], labels[name])
        psnr, ssim = cg.metrics(cg.reconstruct(model, dev, label, masks[name]), dev['target'], dev['kspace'], batch['anatomy'])
        scores[name] = {'psnr': float(psnr[0]), 'ssim': float(ssim[0]), 'fraction': float(masks[name].mean())}
    return scores


@torch.no_grad()
def varnet_scores(models, options, sample, collate_fn, device):
    batch = collate_fn([sample])
    dev = cg.to_device(batch, device) | {k: v for k, v in batch.items() if k not in ('kspace', 'target')}
    scores = {}
    for name in ('poisson', 'loupe', 'loupe_anat'):
        output, mask = vn.forward(models[name], options[name], dev, device)
        psnr, ssim = cg.metrics(output, dev['target'], dev['kspace'], batch['anatomy'])
        scores[name] = {'psnr': float(psnr[0]), 'ssim': float(ssim[0]), 'fraction': float(mask.mean())}
    return scores


def margins(scores):
    return {metric: min(scores['loupe_anat'][metric] - scores[k][metric] for k in MARGIN_KEYS) for metric in ('psnr', 'ssim')}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    cg_models, cg_options = {}, {}
    cg_models['loupe'], cg_options['loupe'] = cg.load(cg.SINGLE_RUN, device)
    cg_models['loupe_anat'], cg_options['loupe_anat'] = cg.load(cg.ROUTED_RUN, device)
    for model in cg_models.values():
        cg.set_sense_threshold(model, args.cg_sense_threshold)
    vn_models, vn_options = {}, {}
    for name in ('poisson', 'loupe', 'loupe_anat'):
        vn_models[name], vn_options[name], _ = vn.load(vn.RUNS[name], args.checkpoint, device)
        vn_models[name].zero_unacquired = True
    _, cg_loader = build_dataloaders(cg.run_config(cg.SINGLE_RUN, args.num_workers))
    _, vn_loader = build_dataloaders(cg.run_config(vn.LOG_ROOT / vn.RUNS['loupe'], args.num_workers))
    shape = tuple(next(iter(cg_loader.dataset.datasets))[0]['kspace'].shape[-3:-1])
    poisson = poisson_mask(shape, cg.ACCELERATION, cg.CALIBRATION)

    indices = cg.subsample(vn_loader.dataset, cg.middle_indices(vn_loader.dataset, args.middle), args.per_anatomy)
    rows = []
    for number, index in enumerate(indices):
        vn_sample = vn_loader.dataset[index]
        condition = vn_sample['condition']
        if condition not in cg.CONDITIONS:
            continue
        fname, slice_id = Path(vn_sample['fname']).name, int(vn_sample['slice_id'])
        cg_index = cg.find_slice(cg_loader.dataset, fname, slice_id)
        if cg_index is None:
            continue
        cg_scores_ = cg_scores(cg_models, cg_options, poisson, cg_loader.dataset[cg_index], device)
        vn_scores_ = varnet_scores(vn_models, vn_options, vn_sample, vn_loader.collate_fn, device)
        on_budget = all(abs(s[n]['fraction'] - 1 / a) <= cg.BUDGET_TOLERANCE
                        for s, a in ((cg_scores_, cg.ACCELERATION), (vn_scores_, vn.ACCELERATION)) for n in ('loupe', 'loupe_anat'))
        row = {'condition': condition, 'fname': fname, 'slice_id': slice_id, 'on_budget': on_budget,
               'cg': cg_scores_, 'varnet': vn_scores_, 'cg_margin': margins(cg_scores_), 'varnet_margin': margins(vn_scores_)}
        row['qualifies'] = on_budget and all(row[k][m] > 0 for k in ('cg_margin', 'varnet_margin') for m in ('psnr', 'ssim'))
        row['score'] = min(row['cg_margin']['psnr'], row['varnet_margin']['psnr'])
        rows.append(row)
        print(f'{number + 1}/{len(indices)} {condition} {fname}:{slice_id} cg {row["cg_margin"]["psnr"]:+.2f} varnet {row["varnet_margin"]["psnr"]:+.2f} '
              f'{"QUALIFIES" if row["qualifies"] else ""}', flush=True)
        (args.out_dir / 'candidates.json').write_text(json.dumps(rows, indent=1))
    print()
    for condition in cg.CONDITIONS:
        good = sorted((r for r in rows if r['condition'] == condition and r['qualifies']), key=lambda r: -r['score'])
        print(f'{condition}: {len(good)} qualifying of {sum(r["condition"] == condition for r in rows)} scored')
        for r in good[:5]:
            print(f'   --pin {condition}={r["fname"]}:{r["slice_id"]}  min PSNR margin {r["score"]:.2f} dB  (cg {r["cg_margin"]["psnr"]:.2f}, varnet {r["varnet_margin"]["psnr"]:.2f})')


if __name__ == '__main__':
    main()
