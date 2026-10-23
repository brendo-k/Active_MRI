"""Reconstruction quality of the faster-U-Net arms against the fastMRI U-Net, 2-cascade VarNet, R=8 static VD.

Runs come from scripts/launch/train_unet_speed_varnet2_r8.sh: arms base, bf16 and sens64 (the sensitivity net's U-Net
at 64x64), each at every seed, all on one saved mask. The lite and lite_bf16 runs need the code snapshot they trained on
(see the launcher). Within a seed the arms share the mask and the data order; their initialisations differ wherever their
U-Nets do. Every val slice is scored inside the apodized per-anatomy metric mask: PSNR / SSIM per slice (data range =
the slice's max), averaged over slices.

The script reports, per anatomy:
  arm - base     per seed and for the per-slice difference averaged over seeds, with a 95% CI over per-volume means
  seed spread    each arm's mean per seed, and the base arm's range over seeds, the yardstick for "no loss"

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_unet_speed_varnet2_r8.py --arms base sens64 --seeds 42 43 44 \
        --out reports/unet_speed_varnet2_r8/eval_sens64.json 2>&1 | tee reports/unet_speed_varnet2_r8/eval_sens64_stdout.txt

`--cascades 12` scores the 12-cascade runs instead. `--run ARM=DIR` (single seed) scores another run directory as that
arm, e.g. an earlier 12-cascade baseline trained on the same mask by the code before this study. eval_12cascades.json
came from this command while the lite options existed (as in the code snapshot; lite_bf16 does not build now):

    python scripts/eval/eval_unet_speed_varnet2_r8.py --cascades 12 --seeds 42 --arms base lite_bf16 \
        --run base=~/data/seqmri_checkpoints/lr_sweep/static_vd_lr0.001_step_50ep_seed42 \
        --out reports/unet_speed_varnet2_r8/eval_12cascades.json
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from compare_anatcond_r8 import METRICS, paired_stats
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask, build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch

RUN_PATTERN = str(Path.home() / 'data/seqmri_checkpoints/unet_speed/runs/varnet{cascades}_r8_{arm}_seed{seed}')
BASE = 'base'
ARMS = (BASE, 'bf16', 'sens64')


def load(run, checkpoint, device):
    ckpt = torch.load(run / checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['epoch']


@torch.no_grad()
def score(model, val_loader, device):
    rows = {'anatomy': [], 'volume': [], 'psnr': [], 'ssim': []}
    for batch in val_loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        output = model(target, kspace)['output']
        target4 = target if output.dim() == target.dim() else target.unsqueeze(1)
        masked_output, masked_target = _apply_signal_mask(output, target4,
                                                          metric_mask_from_real_kspace(kspace, batch['anatomy']))
        rows['anatomy'] += list(batch['anatomy'])
        rows['volume'] += list(batch['fname'])
        rows['psnr'].append(compute_psnr_torch(masked_output, masked_target).flatten().cpu())
        rows['ssim'].append(compute_ssim_torch(masked_output, masked_target).flatten().cpu())
    rows['psnr'], rows['ssim'] = torch.cat(rows['psnr']), torch.cat(rows['ssim'])
    return rows


def by_anatomy(rows):
    return {name: torch.tensor([a == name for a in rows['anatomy']]) for name in sorted(set(rows['anatomy']))}


def compare(delta, reference):
    """Paired statistics of a per-slice difference, per anatomy."""
    out = {}
    for name, keep in by_anatomy(reference).items():
        volumes = [v for v, k in zip(reference['volume'], keep) if k]
        out[name] = {metric: paired_stats(delta[metric][keep], volumes) for metric in METRICS}
    return out


def means(rows):
    return {name: {metric: rows[metric][keep].mean().item() for metric in METRICS}
            for name, keep in by_anatomy(rows).items()}


def print_gap(title, gap):
    print(f'\n{title}')
    for name, metrics in gap.items():
        cells = '  '.join(f'{metric} {s["mean"]:+.4f} [{s["ci"][0]:+.4f}, {s["ci"][1]:+.4f}] '
                          f'(better on {100 * s["frac_volumes_better"]:.0f}% of {s["n_volumes"]} vols)'
                          for metric, s in metrics.items())
        print(f'  {name:6s} {cells}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--cascades', type=int, default=2)
    parser.add_argument('--run', action='append', default=[], metavar='ARM=DIR',
                        help='score DIR as ARM instead of the run named by the pattern (needs a single seed)')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, default=Path('reports/unet_speed_varnet2_r8/eval.json'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.arms[0] != BASE:
        raise ValueError(f'the first arm is the reference and must be {BASE}')

    overrides = {arm: Path(run).expanduser() for arm, run in (item.split('=', 1) for item in args.run)}
    if overrides and len(args.seeds) != 1:
        raise ValueError('--run needs a single seed')
    runs = {(arm, seed): overrides.get(arm, Path(RUN_PATTERN.format(cascades=args.cascades, arm=arm, seed=seed)))
            for seed in args.seeds for arm in args.arms}
    _, val_loader = build_dataloaders(OmegaConf.load(runs[BASE, args.seeds[0]] / '.hydra' / 'config.yaml'))

    scored, summary = {}, {'checkpoint': args.checkpoint, 'cascades': args.cascades, 'seeds': args.seeds,
                           'arms': args.arms, 'runs':{f'{arm}/{seed}': str(run) for (arm, seed), run in runs.items()},
                           'means': {}, 'arm_minus_base': {}, 'seed_mean_arm_minus_base': {}}
    for seed in args.seeds:
        for arm in args.arms:
            model, epoch = load(runs[arm, seed], args.checkpoint, device)
            scored[arm, seed] = score(model, val_loader, device)
            del model
            assert scored[arm, seed]['volume'] == scored[BASE, args.seeds[0]]['volume'], f'{arm} seed {seed}: slice order'
            summary['means'].setdefault(arm, {})[seed] = means(scored[arm, seed])
            cells = '  '.join(f'{name} PSNR {m["psnr"]:.3f} SSIM {m["ssim"]:.4f}'
                              for name, m in summary['means'][arm][seed].items())
            print(f'{runs[arm, seed].name} {args.checkpoint} (epoch {epoch}): {cells}', flush=True)

    reference = scored[BASE, args.seeds[0]]
    for arm in args.arms[1:]:
        deltas = []
        for seed in args.seeds:
            delta = {m: scored[arm, seed][m] - scored[BASE, seed][m] for m in METRICS}
            deltas.append(delta)
            gap = compare(delta, reference)
            summary['arm_minus_base'].setdefault(arm, {})[seed] = gap
            print_gap(f'seed {seed}: {arm} - {BASE}', gap)
        mean_delta = {m: torch.stack([d[m] for d in deltas]).mean(0) for m in METRICS}
        gap = compare(mean_delta, reference)
        summary['seed_mean_arm_minus_base'][arm] = gap
        print_gap(f'{arm} - {BASE}, per-slice difference averaged over seeds {args.seeds}', gap)

    print('\nseed spread: per-anatomy mean over seeds (min, max)')
    summary['seed_spread'] = {}
    for arm in args.arms:
        for name in by_anatomy(reference):
            for metric in METRICS:
                values = [summary['means'][arm][seed][name][metric] for seed in args.seeds]
                summary['seed_spread'].setdefault(arm, {}).setdefault(name, {})[metric] = values
                print(f'  {arm:10s} {name:6s} {metric}  mean {sum(values) / len(values):.4f}  '
                      f'range [{min(values):.4f}, {max(values):.4f}]  spread {max(values) - min(values):.4f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
