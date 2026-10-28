"""Spacing test (R=8, brain + knee VarNet): reconstructors trained on fixed masks of one density, cross-evaluated.

The runs come from scripts/launch/train_varnet_r8_spacing_masks.sh: one VarNet per fixed mask (Poisson-disc vs
independent draw, mask seeds 0 and 1, and the deterministic top-k), all with training seed 42. Every reconstructor is
scored on the whole val set with every run's mask (its saved file, passed as `fixed_mask`), with the trainer's metric
mask. Paired per-slice differences with 95% volume-bootstrap CIs, per anatomy and knee protocol:
  run - run       pd_loupe_s<k> recon + own mask  minus  white_loupe_s<k> recon + own mask (and topk - white_s0)
  mask effect     pd mask minus white mask of the same seed, on each of the two reconstructors
  seed mean       the run - run difference averaged over the two mask seeds

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_spacing_masks_r8.py --runs-root ~/data/seqmri_checkpoints/spacing_r8/runs
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from eval_adaptive_samplers_r8 import ANATOMIES, bootstrap
from mask_gradient_anatomy_r8 import load, to_device

OUT = Path('reports/mask_spacing_r8')
ARMS = ('pd_loupe_s0', 'pd_loupe_s1', 'white_loupe_s0', 'white_loupe_s1', 'topk_loupe')
METRICS = ('psnr', 'ssim')
SAMPLED_FRACTION = 0.125


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--run-name', default='varnet_r8_brain_knee_fixedmask_{arm}_50ep_seed42',
                        help='run directory under --runs-root, with {arm} for the arm')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--tag', default='spacing')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def saved_mask(model, options):
    """The run's mask as its file holds it, checked against the mask the run trained with."""
    saved = torch.load(options.fixed_mask_path, map_location='cpu', weights_only=False)
    mask = torch.as_tensor(saved['mask'] if isinstance(saved, dict) else saved).float()
    assert torch.equal(mask, model.mask_policy.mask[0, 0].cpu()), f'{options.fixed_mask_path} changed since training'
    assert abs(float(mask.mean()) - SAMPLED_FRACTION) < 1e-6, f'{options.fixed_mask_path} samples {float(mask.mean())}'
    return mask


@torch.no_grad()
def score(model, loader, masks, device):
    """Per-slice PSNR / SSIM of one reconstructor on each named [H, W] mask."""
    count = len(loader.dataset)
    scores = {name: {m: torch.zeros(count) for m in METRICS} for name in masks}
    start = 0
    for batch in loader:
        part = to_device(batch, device)
        rows = slice(start, start + len(part['kspace']))
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        for name, mask in masks.items():
            fixed = mask.to(device).expand(len(part['kspace']), 1, -1, -1)
            output = model(part['target'], part['kspace'], fixed_mask=fixed)['output']
            output = output[-1] if isinstance(output, list) else output
            masked = _apply_signal_mask(output, target, part['metric_mask'])
            scores[name]['psnr'][rows] = compute_psnr_torch(*masked).flatten().cpu()
            scores[name]['ssim'][rows] = compute_ssim_torch(*masked).flatten().cpu()
        start = rows.stop
    return scores


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    runs = {arm: args.runs_root / args.run_name.format(arm=arm) for arm in ARMS}
    runs = {arm: run for arm, run in runs.items() if (run / args.checkpoint).exists()}
    print('runs:', {arm: str(run) for arm, run in runs.items()}, flush=True)
    cfg = OmegaConf.load(next(iter(runs.values())) / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset if args.limit is None else torch.utils.data.Subset(
        val_loader.dataset, np.linspace(0, len(val_loader.dataset) - 1, args.limit).astype(int).tolist())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.chunk, shuffle=False, num_workers=4,
                                         collate_fn=val_loader.collate_fn)
    anatomy = np.array(sum((list(b['anatomy']) for b in loader), []))
    condition = np.array(sum((list(b['condition']) for b in loader), []))
    volumes = np.array(sum((list(b['fname']) for b in loader), []))

    models, masks = {}, {}
    for arm, run in runs.items():
        models[arm], checkpoint = load(run, args.checkpoint, args.device)
        masks[arm] = saved_mask(models[arm], checkpoint['options'])
        print(f'{arm}: {run} {args.checkpoint} epoch {checkpoint["epoch"]}', flush=True)

    scores = {}
    for arm, model in models.items():
        scores[arm] = score(model, loader, masks, args.device)
        print(arm, 'recon PSNR', {m: {a: round(float(scores[arm][m]['psnr'][anatomy == a].mean()), 3) for a in ANATOMIES}
                                  for m in masks}, flush=True)
        models[arm] = None
        torch.cuda.empty_cache()

    contrasts = {}
    for seed in (0, 1):
        pd, white = f'pd_loupe_s{seed}', f'white_loupe_s{seed}'
        if pd in scores and white in scores:
            contrasts[f'run_minus_run_s{seed}'] = ((pd, pd), (white, white))
            contrasts[f'mask_effect_on_pd_recon_s{seed}'] = ((pd, pd), (pd, white))
            contrasts[f'mask_effect_on_white_recon_s{seed}'] = ((white, pd), (white, white))
    if 'topk_loupe' in scores and 'white_loupe_s0' in scores:
        contrasts['topk_minus_white_s0'] = (('topk_loupe', 'topk_loupe'), ('white_loupe_s0', 'white_loupe_s0'))

    groups = {**{a: anatomy == a for a in ANATOMIES}, **{c: condition == c for c in ('knee-PD', 'knee-PDFS')}}
    summary = {'runs': {arm: str(run) for arm, run in runs.items()}, 'checkpoint': args.checkpoint,
               'n_slices': {g: int(k.sum()) for g, k in groups.items()},
               'n_volumes': {g: int(len(np.unique(volumes[k]))) for g, k in groups.items()},
               'means': {f'{r}/{m}': {metric: {g: float(scores[r][m][metric][k].mean()) for g, k in groups.items()}
                                      for metric in METRICS} for r in scores for m in masks},
               'contrasts': {}}
    deltas = {}
    for contrast, ((run_a, mask_a), (run_b, mask_b)) in contrasts.items():
        deltas[contrast] = {metric: (scores[run_a][mask_a][metric] - scores[run_b][mask_b][metric]).numpy()
                            for metric in METRICS}
    if {'run_minus_run_s0', 'run_minus_run_s1'} <= deltas.keys():
        deltas['run_minus_run_seed_mean'] = {metric: (deltas['run_minus_run_s0'][metric]
                                                      + deltas['run_minus_run_s1'][metric]) / 2 for metric in METRICS}
    for contrast, delta in deltas.items():
        summary['contrasts'][contrast] = {metric: {g: bootstrap(delta[metric][k], volumes[k], rng) for g, k in groups.items()}
                                          for metric in METRICS}
        psnr = summary['contrasts'][contrast]['psnr']
        print(f'{contrast:32s} ' + '  '.join(f'{g} {psnr[g]["mean"]:+.3f} [{psnr[g]["ci95"][0]:+.3f}, {psnr[g]["ci95"][1]:+.3f}]'
                                              for g in groups), flush=True)

    if args.limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'eval_{args.tag}.json').write_text(json.dumps(summary, indent=2) + '\n')
        torch.save({'anatomy': anatomy.tolist(), 'condition': condition.tolist(), 'volume': volumes.tolist(),
                    'scores': {r: {m: {k: v.half() for k, v in s.items()} for m, s in d.items()} for r, d in scores.items()}},
                   OUT / f'eval_{args.tag}.pt')


if __name__ == '__main__':
    main()
