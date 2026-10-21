"""Paired per-slice comparison of finished runs against one reference run, on the same val slices and the same
mask-draw RNG streams. Each run is scored once; every run's sampler parameters are compared with the reference's
so a mask difference is reported rather than mistaken for a reconstructor difference.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval/paired_vs_reference.py checkpoints/REF checkpoints/A checkpoints/B \
        --out reports/x/paired.json   # reports A - REF and B - REF
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
    sampler = {k: v for k, v in ckpt['model'].items() if k.startswith('sampler.')}
    return {'epoch': ckpt['epoch'], 'sampler': sampler, 'aggregate': [d[0] for d in draws], 'rows': draws[0][1],
            **{m: torch.stack([d[1][m] for d in draws]).mean(0) for m in METRICS}}


def same_sampler(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    _, val_loader = build_dataloaders(OmegaConf.load(args.reference / '.hydra' / 'config.yaml'))
    scored = {run: score(run, val_loader, device, args.seeds, args.checkpoint) for run in [args.reference, *args.runs]}
    ref = scored[args.reference]
    for run, s in scored.items():
        assert s['rows']['volume'] == ref['rows']['volume'], f'{run} saw slices in a different order'
        aggregate = '  '.join(f'[{a}] PSNR={v["PSNR"]:.3f} SSIM={v["SSIM"]:.4f}' for a, v in sorted(s['aggregate'][0].items()))
        print(f'{run.name}  epoch {s["epoch"]}  same sampler as reference: {same_sampler(s["sampler"], ref["sampler"])}  '
              f'aggregate (seed {args.seeds[0]}): {aggregate}', flush=True)

    anatomies = sorted(set(ref['rows']['anatomy']))
    summary = {'reference': str(args.reference), 'checkpoint': args.checkpoint, 'seeds': args.seeds, 'runs': {}}
    for run in args.runs:
        s = scored[run]
        summary['runs'][str(run)] = {'same_sampler': same_sampler(s['sampler'], ref['sampler'])}
        print(f'\n{run.name} - {args.reference.name}  (per-slice, mean over {len(args.seeds)} mask draws; CI over volumes)')
        for anatomy in anatomies:
            keep = torch.tensor([a == anatomy for a in ref['rows']['anatomy']])
            volumes = [v for v, k in zip(ref['rows']['volume'], keep) if k]
            summary['runs'][str(run)][anatomy] = {}
            for metric in METRICS:
                stats = paired_stats(s[metric][keep] - ref[metric][keep], volumes)
                stats['run_mean'], stats['reference_mean'] = s[metric][keep].mean().item(), ref[metric][keep].mean().item()
                summary['runs'][str(run)][anatomy][metric] = stats
                print(f'  {anatomy:6s} {metric}  run {stats["run_mean"]:.4f}  ref {stats["reference_mean"]:.4f}  '
                      f'diff {stats["mean"]:+.4f}  95% CI [{stats["ci"][0]:+.4f}, {stats["ci"][1]:+.4f}]  '
                      f'run better on {100 * stats["frac_volumes_better"]:.0f}% of {stats["n_volumes"]} volumes')
    summary['aggregate'] = {str(run): s['aggregate'] for run, s in scored.items()}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
