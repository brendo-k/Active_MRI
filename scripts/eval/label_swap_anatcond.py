"""Label swap on one anatomy-conditioned LOUPE run: score every val slice with its own anatomy's mask
and with the other anatomy's mask, same reconstructor and same Bernoulli RNG stream.

    CUDA_VISIBLE_DEVICES=3 python scripts/eval/label_swap_anatcond.py checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321])
    parser.add_argument('--out', type=Path, default=Path('reports/anatcond_r8/label_swap_100ep.json'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    _, val_loader = build_dataloaders(OmegaConf.load(args.run / '.hydra' / 'config.yaml'))
    ckpt = torch.load(args.run / args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"{args.run.name} {args.checkpoint}: epoch {ckpt['epoch']}, classes {list(ckpt['options'].label_classes)}")

    draws = {swap: [evaluate(model, ckpt['options'], val_loader, device, seed, swap) for seed in args.seeds]
             for swap in (False, True)}
    rows = draws[False][0][1]
    per_slice = {swap: {m: torch.stack([d[1][m] for d in runs]).mean(0) for m in METRICS} for swap, runs in draws.items()}

    summary = {'checkpoint': str(args.run / args.checkpoint), 'epoch': ckpt['epoch'], 'seeds': args.seeds}
    for anatomy in sorted(set(rows['anatomy'])):
        keep = torch.tensor([a == anatomy for a in rows['anatomy']])
        volumes = [v for v, k in zip(rows['volume'], keep) if k]
        summary[anatomy] = {}
        for metric in METRICS:
            own, other = per_slice[False][metric][keep], per_slice[True][metric][keep]
            stats = paired_stats(other - own, volumes)
            stats['own_mean'], stats['other_mean'] = own.mean().item(), other.mean().item()
            summary[anatomy][metric] = stats
            print(f'{anatomy:6s} {metric}  own {stats["own_mean"]:.4f}  other-label mask {stats["other_mean"]:.4f}  '
                  f'delta {stats["mean"]:+.4f}  95% CI [{stats["ci"][0]:+.4f}, {stats["ci"][1]:+.4f}]  '
                  f'worse on {100 * (1 - stats["frac_volumes_better"]):.0f}% of {stats["n_volumes"]} volumes')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
