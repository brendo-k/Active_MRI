"""Brain-only VarNets with a frozen shared mask vs a learned LOUPE mask (R=8): score on the full brain val set.

Each run is a brain-only VarNet trained on a quarter of local v1 (`datasets.limit_volumes=0.25`, which also cuts val).
Here every run is re-scored on all 40 brain val volumes (limit_volumes=1). Each run uses its own brain probability map,
thresholded with common per-slice Bernoulli draws (seeds 1234 and 4321, as in eval_adaptive_samplers_r8), and is
scored with the trainer's metric mask. Brain-only runs never see knee data, and their val loss mixes in knee, so
`last_model.pt` is scored rather than `best_model.pt`.

Outputs, per seed:
  arm means            PSNR / SSIM / NMSE / realised sampling fraction
  contrasts            arm - arm, paired per slice, 95% volume-bootstrap CI
  budget relocated     1/2 sum|p_arm - p_shared| / sum p_shared, on the mean brain map

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_static_vs_loupe_eval.py \
        --queue ~/data/seqmri_checkpoints/brain_static_vs_loupe --seeds 42 43 44
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
from eval_adaptive_samplers_r8 import SEEDS, bootstrap, thresholds
from mask_gradient_anatomy_r8 import load, to_device

ARMS = ('static', 'loupe_shared', 'loupe_uniform')
CONTRASTS = (('loupe_shared', 'static'), ('loupe_uniform', 'static'), ('loupe_shared', 'loupe_uniform'))
METRICS = ('psnr', 'ssim', 'nmse', 'fraction')
OUT = Path('reports/brain_static_vs_loupe')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, required=True, help='folder holding <arm>_s<seed>/ run folders')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def full_brain_val_loader(run, chunk, limit):
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    cfg.datasets.limit_volumes = 1
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset
    brain_parts = [part for part in dataset.datasets if part[0]['anatomy'] == 'brain']
    dataset = torch.utils.data.ConcatDataset(brain_parts)
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, np.linspace(0, len(dataset) - 1, limit).astype(int).tolist())
    return torch.utils.data.DataLoader(dataset, batch_size=chunk, shuffle=False, num_workers=4,
                                       collate_fn=val_loader.collate_fn)


def with_brain_label(batch, classes, device):
    part = to_device(batch, device)
    part['label'] = torch.full((len(part['kspace']),), classes.index('brain'), device=device)
    return part


@torch.no_grad()
def score_run(model, classes, loader, device):
    """Per-slice metrics of the run's own brain map (mean over SEEDS), and the run's mean brain probability map."""
    scores = {m: [] for m in METRICS}
    map_sum, count, start = 0.0, 0, 0
    for batch in loader:
        part = with_brain_label(batch, classes, device)
        proposal = model.propose_mask(part['kspace'], anatomy_labels=part['label'])
        rescaled = proposal['rescaled'][:, 0].float()
        old, valid = proposal['old_mask'][:, 0].bool(), proposal['valid_mask'][:, 0].bool()
        map_sum = map_sum + rescaled.sum(0).cpu()
        rows = torch.arange(start, start + len(rescaled))
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        batch_scores = {m: torch.zeros(len(rows)) for m in METRICS}
        for seed in SEEDS:
            threshold = torch.stack([thresholds(seed, int(i), old.shape[1:]) for i in rows]).to(device)
            mask = (old | ((rescaled > threshold) & valid)).float().unsqueeze(1)
            output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
            output = output[-1] if isinstance(output, list) else output
            out_m, tgt_m = _apply_signal_mask(output, target, part['metric_mask'])
            nmse = ((out_m - tgt_m) ** 2).flatten(1).sum(1) / (tgt_m ** 2).flatten(1).sum(1)
            batch_scores['psnr'] += compute_psnr_torch(out_m, tgt_m).flatten().cpu() / len(SEEDS)
            batch_scores['ssim'] += compute_ssim_torch(out_m, tgt_m).flatten().cpu() / len(SEEDS)
            batch_scores['nmse'] += nmse.cpu() / len(SEEDS)
            batch_scores['fraction'] += mask.flatten(1).mean(1).cpu() / len(SEEDS)
        for m in METRICS:
            scores[m].append(batch_scores[m])
        count += len(rows)
        start += len(rows)
    return {m: torch.cat(v) for m, v in scores.items()}, map_sum / count


def budget_relocated(p, reference):
    return float(0.5 * (p - reference).abs().sum() / reference.sum())


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    runs = {(arm, seed): args.queue / f'{arm}_s{seed}' for seed in args.seeds for arm in args.arms}
    loader = full_brain_val_loader(next(iter(runs.values())), args.chunk, args.limit)
    volumes = np.array(sum((list(b['fname']) for b in loader), []))

    scores, maps, summary = {}, {}, {'runs': {}, 'checkpoint': args.checkpoint, 'thresholds': list(SEEDS),
                                     'n_slices': int(len(volumes)), 'n_volumes': int(len(np.unique(volumes)))}
    for key, run in runs.items():
        model, checkpoint = load(run, args.checkpoint, args.device)
        classes = list(checkpoint['options'].label_classes)
        scores[key], maps[key] = score_run(model, classes, loader, args.device)
        summary['runs']['%s_s%d' % key] = {'path': str(run), 'epoch': int(checkpoint['epoch']),
                                           **{m: float(scores[key][m].mean()) for m in METRICS}}
        print(key, {m: round(float(scores[key][m].mean()), 4) for m in METRICS}, flush=True)
        del model
        torch.cuda.empty_cache()

    summary['contrasts'], summary['budget_relocated_vs_static'] = {}, {}
    for seed in args.seeds:
        for a, b in CONTRASTS:
            if (a, seed) not in scores or (b, seed) not in scores:
                continue
            name = f'{a}-{b}_s{seed}'
            summary['contrasts'][name] = {
                m: bootstrap((scores[(a, seed)][m] - scores[(b, seed)][m]).numpy(), volumes, rng) for m in METRICS}
            psnr = summary['contrasts'][name]['psnr']
            print(f'{name:36s} PSNR {psnr["mean"]:+.3f} [{psnr["ci95"][0]:+.3f}, {psnr["ci95"][1]:+.3f}]', flush=True)
        if ('static', seed) in maps:
            for arm in args.arms:
                if arm != 'static' and (arm, seed) in maps:
                    summary['budget_relocated_vs_static'][f'{arm}_s{seed}'] = budget_relocated(
                        maps[(arm, seed)], maps[('static', seed)])
    print('budget relocated vs static map:', summary['budget_relocated_vs_static'], flush=True)

    if args.limit is None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'eval.json').write_text(json.dumps(summary, indent=2) + '\n')
        torch.save({'volume': volumes.tolist(), 'maps': {'%s_s%d' % k: v.half() for k, v in maps.items()},
                    'scores': {'%s_s%d' % k: {m: v.half() for m, v in s.items()} for k, s in scores.items()}},
                   args.out / 'eval_per_slice.pt')


if __name__ == '__main__':
    main()
