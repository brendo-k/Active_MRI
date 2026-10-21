"""Paired per-slice comparison of two finished runs on the same val slices and the same mask-draw RNG streams.

    CUDA_VISIBLE_DEVICES=1 python scripts/eval/paired_two_runs.py checkpoints/A checkpoints/B   # reports A - B
"""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from compare_anatcond_r8 import evaluate, paired_stats, METRICS
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model


def score(run, val_loader, device, seeds, checkpoint):
    ckpt = torch.load(run / checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    draws = [evaluate(model, ckpt['options'], val_loader, device, seed, False) for seed in seeds]
    return {'rows': draws[0][1], **{m: torch.stack([d[1][m] for d in draws]).mean(0) for m in METRICS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_a', type=Path)
    parser.add_argument('run_b', type=Path)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    _, val_loader = build_dataloaders(OmegaConf.load(args.run_a / '.hydra' / 'config.yaml'))
    a = score(args.run_a, val_loader, device, args.seeds, args.checkpoint)
    b = score(args.run_b, val_loader, device, args.seeds, args.checkpoint)
    assert a['rows']['volume'] == b['rows']['volume'], 'runs saw slices in a different order'

    summary = {}
    for anatomy in sorted(set(a['rows']['anatomy'])):
        keep = torch.tensor([x == anatomy for x in a['rows']['anatomy']])
        volumes = [v for v, k in zip(a['rows']['volume'], keep) if k]
        summary[anatomy] = {}
        for metric in METRICS:
            stats = paired_stats(a[metric][keep] - b[metric][keep], volumes)
            stats['a_mean'], stats['b_mean'] = a[metric][keep].mean().item(), b[metric][keep].mean().item()
            summary[anatomy][metric] = stats
            print(f'{anatomy:6s} {metric}  A {stats["a_mean"]:.4f}  B {stats["b_mean"]:.4f}  A-B {stats["mean"]:+.4f}  '
                  f'95% CI [{stats["ci"][0]:+.4f}, {stats["ci"][1]:+.4f}]  A better on '
                  f'{100 * stats["frac_volumes_better"]:.0f}% of {stats["n_volumes"]} volumes')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'a': str(args.run_a), 'b': str(args.run_b), **summary}, indent=2))


if __name__ == '__main__':
    main()
