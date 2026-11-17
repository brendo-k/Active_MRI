"""Score the brain_varnet_width runs (reports/brain_varnet_width/proposal.md) on full brain val or their train set.

Every run uses the frozen shared mask S. Its brain probability map is thresholded with the common per-slice Bernoulli
draws of eval_adaptive_samplers_r8 (SEEDS 1234 and 4321), so all runs see identical masks on each slice and are
paired per slice. Scores use the trainer's apodized metric mask.

  --split val     the full brain val set: 40 volumes, 605 slices (the trainer validated on a 10-volume subset)
  --split train   the 80 training volumes (1,211 slices) every run trained on, for the train - val gap

Writes <out>/<split>_<checkpoint stem>_<run>.pt per run and checkpoint, holding per-slice PSNR / SSIM / NMSE and the
slice volumes. A run whose file exists is skipped. All runs share one pass over the data.

The 6 x 18 and 12 x 18 cells are brain_varnet_scaling's c6_v80_* and c12_v80_* runs, scored there with this same
recipe; run.sh copies their files in under the names c6_ch18_* and c12_ch18_*.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/brain_varnet_scaling_eval.py --seed 42 --split val
"""
import argparse
import re
from collections import defaultdict
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
from eval_adaptive_samplers_r8 import SEEDS, thresholds
from mask_gradient_anatomy_r8 import load, to_device

QUEUE = Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_width')
OUT = Path('reports/brain_varnet_width/eval')
METRICS = ('psnr', 'ssim', 'nmse')
LOCAL_DATASET = Path('configs/datasets/fastmri_local_v1_base.yaml')
NAME = re.compile(r'^c(\d+)_ch(\d+)_s(\d+)$')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--runs', nargs='*', default=None, help='run names; default every finished run of --seed')
    parser.add_argument('--split', choices=('val', 'train'), default='val')
    parser.add_argument('--checkpoints', nargs='+', default=['best_model.pt', 'last_model.pt'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local-data', action='store_true',
                        help='runs trained elsewhere (CoreHPC): read the local data copies, and score --split train on '
                             'the tier-S volumes, which every tier-L run also trained on')
    return parser.parse_args()


def brain_loader(run, split, chunk, limit, local_data=False):
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    if local_data:
        cfg.datasets.roots = OmegaConf.load(LOCAL_DATASET).roots
        cfg.datasets.train_tier = 'S'
    if split == 'val':
        cfg.datasets.limit_volumes = 1
        cfg.datasets.val_limit_volumes = None
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    dataset = (train_loader if split == 'train' else val_loader).dataset
    dataset = torch.utils.data.ConcatDataset([part for part in dataset.datasets if part[0]['anatomy'] == 'brain'])
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, np.linspace(0, len(dataset) - 1, limit).astype(int).tolist())
    return torch.utils.data.DataLoader(dataset, batch_size=chunk, shuffle=False, num_workers=4,
                                       collate_fn=val_loader.collate_fn)


@torch.no_grad()
def score(models, loader, device):
    """Per-slice metrics of each model with its own brain map, averaged over the SEEDS draws."""
    scores = {key: {m: [] for m in METRICS} for key in models}
    volumes, start = [], 0
    for batch in loader:
        volumes += [str(v) for v in batch['fname']]
        part = None
        for key, (model, classes) in models.items():
            part = to_device(batch, device) if part is None else part
            labels = torch.full((len(part['kspace']),), classes.index('brain'), device=device)
            proposal = model.propose_mask(part['kspace'], anatomy_labels=labels)
            rescaled = proposal['rescaled'][:, 0].float()
            old, valid = proposal['old_mask'][:, 0].bool(), proposal['valid_mask'][:, 0].bool()
            rows = torch.arange(start, start + len(rescaled))
            target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
            batch_scores = {m: torch.zeros(len(rows)) for m in METRICS}
            for seed in SEEDS:
                threshold = torch.stack([thresholds(seed, int(i), old.shape[1:]) for i in rows]).to(device)
                mask = (old | ((rescaled > threshold) & valid)).float().unsqueeze(1)
                output = model(part['target'], part['kspace'], anatomy_labels=labels, fixed_mask=mask)['output']
                output = output[-1] if isinstance(output, list) else output
                out_m, tgt_m = _apply_signal_mask(output, target, part['metric_mask'])
                nmse = ((out_m - tgt_m) ** 2).flatten(1).sum(1) / (tgt_m ** 2).flatten(1).sum(1)
                batch_scores['psnr'] += compute_psnr_torch(out_m, tgt_m).flatten().cpu() / len(SEEDS)
                batch_scores['ssim'] += compute_ssim_torch(out_m, tgt_m).flatten().cpu() / len(SEEDS)
                batch_scores['nmse'] += nmse.cpu() / len(SEEDS)
            for m in METRICS:
                scores[key][m].append(batch_scores[m])
        start += len(batch['fname'])
        print(f'{start} slices', flush=True)
    return {key: {m: torch.cat(v) for m, v in by_metric.items()} for key, by_metric in scores.items()}, volumes


def main():
    args = parse_args()
    names = args.runs or sorted(p.name for p in args.queue.iterdir()
                                if NAME.match(p.name) and int(NAME.match(p.name)[3]) in args.seed
                                and (p / 'last_model.pt').exists())
    args.out.mkdir(parents=True, exist_ok=True)
    by_volumes = defaultdict(list)
    for name in names:
        by_volumes[0].append(name)
    for group in by_volumes.values():
        todo = [(name, ckpt) for name in group for ckpt in args.checkpoints
                if not (args.out / f'{args.split}_{Path(ckpt).stem}_{name}.pt').exists() or args.limit]
        if not todo:
            continue
        models = {}
        for name, ckpt in todo:
            model, checkpoint = load(args.queue / name, ckpt, args.device)
            models[name, ckpt] = (model, list(checkpoint['options'].label_classes))
        loader = brain_loader(args.queue / todo[0][0], args.split, args.chunk, args.limit, args.local_data)
        scores, volumes = score(models, loader, args.device)
        for (name, ckpt), s in scores.items():
            print(f'{args.split} {name:12s} {ckpt:14s} PSNR {s["psnr"].mean():.3f} SSIM {s["ssim"].mean():.4f} '
                  f'NMSE {s["nmse"].mean():.5f}', flush=True)
            if args.limit is None:
                torch.save({'run': name, 'checkpoint': ckpt, 'split': args.split, 'thresholds': list(SEEDS),
                            'volume': volumes, 'scores': s}, args.out / f'{args.split}_{Path(ckpt).stem}_{name}.pt')
        del models
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
