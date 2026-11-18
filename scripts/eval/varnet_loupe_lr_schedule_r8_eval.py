"""Score the varnet_loupe_lr_schedule_r8 runs on the FULL local v1 FLAIR+knee val set, paired per seed vs the control.

The runs trained on `datasets.limit_volumes=0.2`, which also cuts val. Here each finished run (its last_model.pt
reached training.num_epochs) is scored on limit_volumes=1 with the per-slice scoring of varnet_train_speed_eval
(trainer metric mask, slice-max data range, common seeded mask draws). --checkpoint best_model.pt scores the peak on the training-time val instead (for the 20% runs that val is a
subset of the full val, so the peak is chosen partly on the scored slices). Writes the same JSON layout
(per_run, paired[arm][seed][condition]) and a per-slice cache next to --out.

    python scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir ~/scratch/varnet_loupe_lr_schedule_r8/runs \
        --out reports/varnet_loupe_lr_schedule_r8/eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from varnet_train_speed_eval import RUN_NAME, paired, score

CONTROL = 'lr1e3_step'


def is_finished(run_dir):
    checkpoint = run_dir / 'last_model.pt'
    if not checkpoint.exists():
        return False
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return ckpt['epoch'] + 1 == ckpt['options'].num_epochs


def full_val_loader(run_dir, data_root=None):
    cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    cfg.datasets.limit_volumes = 1
    if data_root:
        cfg.datasets.roots = {name: data_root for name in cfg.datasets.roots}
    return build_dataloaders(cfg)[1]


def condition_rows(scores):
    conditions = scores['condition']
    for condition in ['all', *sorted(set(conditions))]:
        yield condition, np.ones(len(conditions), bool) if condition == 'all' else conditions == condition


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('reports/varnet_loupe_lr_schedule_r8/eval.json'))
    parser.add_argument('--control', default=CONTROL)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321])
    parser.add_argument('--checkpoint', default='last_model.pt', help='last_model.pt or best_model.pt (peak on the training val)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', help='replace every datasets.roots entry, e.g. /export/scratch/bkadota/fastmri_raw/{anatomy} to score CoreHPC runs here')
    args = parser.parse_args()
    runs_dir = args.runs_dir.expanduser()
    names = sorted(p.name for p in runs_dir.iterdir()
                   if p.is_dir() and RUN_NAME.match(p.name) and '.attempt' not in p.name and is_finished(p))
    if not names:
        raise SystemExit(f'no finished runs under {runs_dir}')

    cache = args.out.with_suffix('.per_slice.npz')
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    val_loader = full_val_loader(runs_dir / names[0], args.data_root)
    scores = {}
    for name in names:
        if f'{name}/psnr' in stored:
            scores[name] = {k: stored[f'{name}/{k}'] for k in ('psnr', 'ssim', 'condition', 'volume', 'epochs')}
            continue
        print(f'scoring {name}', flush=True)
        scores[name] = score(runs_dir / name, val_loader, torch.device(args.device), args.seeds, args.checkpoint)
        stored.update({f'{name}/{k}': v for k, v in scores[name].items()})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **stored)

    per_run = {name: {'epochs': int(s['epochs']),
                      **{c: {'psnr_masked': float(s['psnr'][keep].mean()), 'ssim_masked': float(s['ssim'][keep].mean()),
                             'n_slices': int(keep.sum())} for c, keep in condition_rows(s)}}
               for name, s in scores.items()}
    paired_out = {}
    for name, s in scores.items():
        match = RUN_NAME.match(name)
        control = f"{args.control}_s{match['seed']}"
        if match['arm'] == args.control or control not in scores:
            continue
        c = scores[control]
        assert (s['volume'] == c['volume']).all(), f'{name} and {control} saw the slices in a different order'
        paired_out.setdefault(match['arm'], {})[match['seed']] = {
            condition: {'psnr': paired(s['psnr'][keep] - c['psnr'][keep], s['volume'][keep]),
                        'ssim': paired(s['ssim'][keep] - c['ssim'][keep], s['volume'][keep])}
            for condition, keep in condition_rows(s)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'seeds': args.seeds, 'control': args.control, 'val': 'full local v1 (limit_volumes=1)', 'checkpoint': args.checkpoint,
                                    'per_run': per_run, 'paired': paired_out}, indent=1))
    for arm, seeds in sorted(paired_out.items()):
        for seed, entry in sorted(seeds.items()):
            print(f"{arm:>16s} s{seed} - control: " + ' '.join(
                f"{c} {e['psnr']['mean']:+.3f}" for c, e in entry.items()))


if __name__ == '__main__':
    main()
