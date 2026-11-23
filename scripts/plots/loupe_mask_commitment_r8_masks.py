"""Learned LOUPE masks of loupe_mask_commitment_r8: does each arm keep the right k-space band?

The probability is the run's own sampler forward (so the clamp arm's soft clamp is applied), at the checkpoint's slope,
before the per-slice budget rescale, as in varnet_loupe_lr_schedule_r8_masks.py. The right band is that script's: total p in
rows within 12 of the centre row and columns more than 72 right of the centre column. `--reference` re-scores the
previous experiment's 12-cascade lr 1e-3 step controls with the same code, as a check against its masks.json (8.4, 0.2).

    python scripts/plots/loupe_mask_commitment_r8_masks.py --runs /export/scratch/bkadota/loupe_mask_commitment_r8/runs \
        --reference /export/scratch/bkadota/varnet_loupe_lr_schedule_r8/runs_full --report reports/loupe_mask_commitment_r8

The follow-up (followup_proposal.md) scores its run folders with --arms, --seeds and --name, e.g.
    ... --runs .../runs_e100 --arms control mixdecay --seeds 42 43 44 --name masks_followup_e100
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.samplers.loupe import ProbMask
from varnet_loupe_lr_schedule_r8_masks import BANDS, RIGHT_BAND, SUPPORT_P, radius_map

ARMS = ('control', 'clamp', 'anneal', 'mix')
SEEDS = (42, 43)
REFERENCE_RUNS = ('lr1e3_step_s42', 'lr1e3_step_s43')
BAND_KEPT = 200
SATURATED_LOGIT = np.log(99)


def probability(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    options = checkpoint['options']
    logits = checkpoint['model']['sampler.mask_net.mask'].float()
    logit_limit = getattr(options, 'loupe_logit_limit', 0.0)
    clamp = {'logit_limit': logit_limit, 'logit_leak': options.logit_leak} if logit_limit > 0 else {}
    mask_net = ProbMask(tuple(logits.shape[1:3]), options.slope, **clamp)
    with torch.no_grad():
        mask_net.mask.copy_(logits)
        p = mask_net(torch.zeros(1, *logits.shape[1:3], 2)).squeeze().numpy()
    return p, (options.slope * logits).squeeze().numpy(), int(checkpoint['epoch']) + 1


def describe(path):
    p, slope_logits, epoch = probability(path)
    radius = radius_map(p.shape)
    total = p.sum()
    return {
        'epoch': epoch,
        'right_band': float(p[RIGHT_BAND].sum()),
        'mean_p': float(p.mean()),
        'binarization': float(np.abs(2 * p - 1).mean()),
        'support': float((p > SUPPORT_P).mean()),
        'saturated_frac': float((np.abs(slope_logits) > SATURATED_LOGIT).mean()),
        'band_share': {f'{lo}-{hi}': float(p[(radius >= lo) & (radius < hi)].sum() / total) for lo, hi in BANDS},
    }, p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--name', default='masks', help='writes <name>.json and figures/<name>.png')
    args = parser.parse_args()
    arms, seeds = tuple(args.arms), tuple(args.seeds)

    runs, maps = {}, {}
    for arm in arms:
        for seed in seeds:
            path = args.runs / f'{arm}_s{seed}' / 'last_model.pt'
            if path.exists():
                runs[f'{arm}_s{seed}'], maps[f'{arm}_s{seed}'] = describe(path)
    reference = {}
    if args.reference:
        for name in REFERENCE_RUNS:
            reference[name], _ = describe(args.reference / name / 'last_model.pt')

    for name, entry in {**runs, **{f'reference/{k}': v for k, v in reference.items()}}.items():
        print(f"{name:28s} ep {entry['epoch']:3d} band {entry['right_band']:7.1f} mean_p {entry['mean_p']:.4f} "
              f"bin {entry['binarization']:.3f} support {entry['support']:.3f} sat {entry['saturated_frac']:.3f}")

    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / f'{args.name}.json').write_text(json.dumps(
        {'right_band_rows_cols': [[RIGHT_BAND[0].start, RIGHT_BAND[0].stop], [RIGHT_BAND[1].start, None]],
         'band_kept_threshold': BAND_KEPT, 'runs': runs, 'reference_lr1e3_step_12c': reference}, indent=1))

    names = list(maps)
    fig, axes = plt.subplots(len(seeds), len(arms), figsize=(3.2 * len(arms), 3.3 * len(seeds)), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for name in names:
        arm, seed = name.rsplit('_s', 1)
        ax = axes[seeds.index(int(seed)), arms.index(arm)]
        ax.imshow(maps[name], cmap='gray', vmin=0, vmax=1)
        rows, cols = RIGHT_BAND
        ax.add_patch(plt.Rectangle((cols.start - 0.5, rows.start - 0.5), maps[name].shape[1] - cols.start, rows.stop - rows.start,
                                   fill=False, edgecolor='tab:orange', linewidth=1))
        ax.set_title(f"{name}\nband {runs[name]['right_band']:.0f}", fontsize=9)
    fig.suptitle('LOUPE p = sigmoid(slope * logit) at last_model (orange: right band)', fontsize=10)
    fig.tight_layout()
    (args.report / 'figures').mkdir(exist_ok=True)
    fig.savefig(args.report / 'figures' / f'{args.name}.png', dpi=120)


if __name__ == '__main__':
    main()
