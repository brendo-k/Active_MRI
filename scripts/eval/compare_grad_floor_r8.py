"""Paired val comparison: LOUPE-2D with vs without the sigmoid/STE gradient floor (R=8, brain+knee VarNet).

  floor    checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05  (wandb uo8d68n0)
           sampler.mask_grad_floor=0.05, sampler.ste_grad_floor=0.05
  nofloor  checkpoints/grad_floor_compare/loupe2d_r8_slope5_nofloor    (wandb ijhnsmnm)
           both floors 0

Same config otherwise (slope 5, straight_through, seed 42, 50 epochs, lr step 40). Each run is scored
at its best-val-loss checkpoint and at its last (epoch-50) checkpoint. Every model sees the same val
slices and the same RNG stream for the Bernoulli mask draws. Metrics follow the trainer (body-masked
output and target); paired per-slice deltas get 95% CIs over per-volume means.

    CUDA_VISIBLE_DEVICES=3 python scripts/eval/compare_grad_floor_r8.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from compare_anatcond_r8 import METRICS, evaluate, paired_stats
from activemri.training.trainer import build_model

RUN_DIRS = {
    'floor': Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05'),
    'nofloor': Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_nofloor'),
}
CHECKPOINTS = {'best': 'best_model.pt', 'last': 'last_model.pt'}
COMPARISONS = [('floor_best', 'nofloor_best'), ('floor_last', 'nofloor_last'),
               ('floor_last', 'floor_best'), ('nofloor_last', 'nofloor_best')]


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['options'], int(ckpt['epoch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321], help='RNG streams for mask draws')
    parser.add_argument('--out', type=Path, default=Path('reports/grad_floor_r8'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(RUN_DIRS['nofloor'] / '.hydra' / 'config.yaml')
    _, val_loader = build_dataloaders(cfg)

    results, epochs = {}, {}
    for run, run_dir in RUN_DIRS.items():
        for tag, filename in CHECKPOINTS.items():
            name = f'{run}_{tag}'
            model, options, epochs[name] = load_model(run_dir / filename, device)
            draws = [evaluate(model, options, val_loader, device, seed, swap_labels=False) for seed in args.seeds]
            results[name] = {
                'aggregate': [d[0] for d in draws],
                'rows': draws[0][1],
                **{m: torch.stack([d[1][m] for d in draws]).mean(0) for m in METRICS},
            }
            for seed, (aggregate, _) in zip(args.seeds, draws):
                print(f'{name:13s} epoch {epochs[name]:2d} seed {seed}  ' + '  '.join(
                    f'[{a}] PSNR={v["PSNR"]:.3f} SSIM={v["SSIM"]:.4f}' for a, v in sorted(aggregate.items())),
                    flush=True)
            del model
            torch.cuda.empty_cache()

    reference = results['nofloor_best']['rows']
    for name in results:
        assert results[name]['rows']['volume'] == reference['volume'], f'{name} saw slices in a different order'

    summary = {}
    for a_name, b_name in COMPARISONS:
        key = f'{a_name} - {b_name}'
        summary[key] = {}
        print(f'\n{key}  (per-slice, mean over {len(args.seeds)} mask draws; CI over volumes)')
        for anatomy in sorted(set(reference['anatomy'])):
            keep = torch.tensor([a == anatomy for a in reference['anatomy']])
            volumes = [v for v, k in zip(reference['volume'], keep) if k]
            summary[key][anatomy] = {}
            for metric in METRICS:
                delta = results[a_name][metric][keep] - results[b_name][metric][keep]
                s = paired_stats(delta, volumes)
                summary[key][anatomy][metric] = s
                print(f'  {anatomy:6s} {metric}  {s["mean"]:+.4f}  95% CI [{s["ci"][0]:+.4f}, {s["ci"][1]:+.4f}]'
                      f'  better on {100 * s["frac_volumes_better"]:.0f}% of {s["n_volumes"]} volumes')

    summary['epochs'] = epochs
    summary['aggregate'] = {name: r['aggregate'] for name, r in results.items()}
    (args.out / 'paired.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
