"""Per-condition validation scores of the varnet_train_speed runs, and paired arm - control differences.

Every finished run (its log reached its last epoch) under the queue directory is scored from its last_model.pt on the
run's validation set (20% of the local v1 FLAIR+knee val volumes). Per slice it computes PSNR_Masked and SSIM_Masked as
the trainer does (`MaskedMetricAccumulator`: inside the apodized per-anatomy metric mask, data range = the slice's
max). Each run uses the same seeded mask draws (`--seeds`, the global RNG seeded before each pass), so two runs differ
only by their weights and learned mask probabilities.

Output (one JSON):
  per_run[name][condition]  mean PSNR_Masked / SSIM_Masked over slices, averaged over the draws
  paired[arm][seed][condition]  arm - control (same seed; the control arm is --control) per slice: mean, 95% CI over
                                per-volume means, share of volumes where the arm is better
  Slices with condition brain / knee-PD / knee-PDFS; 'all' pools them.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/varnet_train_speed_eval.py --out reports/varnet_train_speed/eval.json
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.metrics import masked_ssim

QUEUE_DIR = Path('~/data/seqmri_checkpoints/varnet_train_speed').expanduser()
RUN_NAME = re.compile(r'(?P<arm>.+)_s(?P<seed>\d+)$')
EPOCH_LINE = re.compile(r'Epoch = \[ *(\d+)/ *(\d+)\]')


def finished(run_dir):
    log = run_dir.with_suffix('.log')
    if not (run_dir / 'last_model.pt').exists() or not log.exists():
        return False
    epochs = [m.groups() for m in EPOCH_LINE.finditer(log.read_text())]
    return bool(epochs) and epochs[-1][0] == epochs[-1][1]


@torch.no_grad()
def score(run_dir, val_loader, device, seeds):
    ckpt = torch.load(run_dir / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if hasattr(model, 'store_visuals'):
        model.store_visuals = False
    options = ckpt['options']
    classes = list(options.label_classes)
    psnr, ssim, conditions, volumes = [], [], [], []
    for draw, seed in enumerate(seeds):
        torch.manual_seed(seed)
        draw_psnr, draw_ssim = [], []
        for batch in val_loader:
            kspace, target = batch['kspace'].to(device), batch['target'].to(device)
            anatomies = list(batch['anatomy'])
            labels = torch.tensor([classes.index(c) for c in batch[options.label_key]], device=device) if classes else None
            output = model(target, kspace, anatomy_labels=labels)['output']
            target4 = target if target.dim() == 4 else target.unsqueeze(1)
            output = output if output.dim() == 4 else output.unsqueeze(1)
            mask = metric_mask_from_real_kspace(kspace, anatomies).float()
            mask = mask if mask.dim() == 4 else mask.unsqueeze(1)
            mse = ((target4 - output) ** 2 * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
            draw_psnr.append((10 * torch.log10(target4.amax(dim=(1, 2, 3)) ** 2 / mse)).double().cpu())
            draw_ssim.append(masked_ssim(output, target4, mask).cpu())
            if draw == 0:
                conditions += list(batch['condition'])
                volumes += [f"{c}/{f}" for c, f in zip(batch['condition'], batch['fname'])]
        psnr.append(torch.cat(draw_psnr))
        ssim.append(torch.cat(draw_ssim))
    return {'psnr': torch.stack(psnr).mean(0).numpy(), 'ssim': torch.stack(ssim).mean(0).numpy(),
            'condition': np.array(conditions), 'volume': np.array(volumes),
            'epochs': ckpt['epoch'] + 1}


def paired(delta, volumes):
    names, index = np.unique(volumes, return_inverse=True)
    per_volume = np.bincount(index, weights=delta) / np.bincount(index)
    half = 1.96 * per_volume.std(ddof=1) / np.sqrt(len(names)) if len(names) > 1 else float('nan')
    return {'mean': float(delta.mean()), 'ci': [float(delta.mean() - half), float(delta.mean() + half)],
            'n_volumes': int(len(names)), 'frac_volumes_better': float((per_volume > 0).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('reports/varnet_train_speed/eval.json'))
    parser.add_argument('--runs', nargs='*', help='run names (default: every finished run)')
    parser.add_argument('--control', default='control', help='arm the others are paired against, per seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 4321])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache', type=Path, help='per-slice score cache shared between calls (default: next to --out)')
    args = parser.parse_args()
    device = torch.device(args.device)

    names = args.runs or sorted(p.name for p in QUEUE_DIR.iterdir()
                                if p.is_dir() and RUN_NAME.match(p.name) and '.attempt' not in p.name and finished(p))
    cache = args.cache or args.out.with_suffix('.per_slice.npz')
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    _, val_loader = build_dataloaders(OmegaConf.load(QUEUE_DIR / names[0] / '.hydra' / 'config.yaml'))
    scores = {}
    for name in names:
        if f'{name}/psnr' in stored:
            scores[name] = {k: stored[f'{name}/{k}'] for k in ('psnr', 'ssim', 'condition', 'volume', 'epochs')}
            continue
        print(f'scoring {name}', flush=True)
        scores[name] = score(QUEUE_DIR / name, val_loader, device, args.seeds)
        stored.update({f'{name}/{k}': v for k, v in scores[name].items()})
        np.savez(cache, **stored)

    per_run = {}
    for name, s in scores.items():
        per_run[name] = {'epochs': int(s['epochs'])}
        for condition in ['all', *sorted(set(s['condition']))]:
            keep = np.ones(len(s['psnr']), bool) if condition == 'all' else s['condition'] == condition
            per_run[name][condition] = {'psnr_masked': float(s['psnr'][keep].mean()), 'ssim_masked': float(s['ssim'][keep].mean()),
                                        'n_slices': int(keep.sum())}
    paired_out = {}
    for name, s in scores.items():
        match = RUN_NAME.match(name)
        control = f"{args.control}_s{match['seed']}"
        if match['arm'] == args.control or control not in scores:
            continue
        c = scores[control]
        assert (s['volume'] == c['volume']).all(), f'{name} and {control} saw the slices in a different order'
        entry = {}
        for condition in ['all', *sorted(set(s['condition']))]:
            keep = np.ones(len(s['psnr']), bool) if condition == 'all' else s['condition'] == condition
            entry[condition] = {'psnr': paired(s['psnr'][keep] - c['psnr'][keep], s['volume'][keep]),
                                'ssim': paired(s['ssim'][keep] - c['ssim'][keep], s['volume'][keep])}
        paired_out.setdefault(match['arm'], {})[match['seed']] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'seeds': args.seeds, 'control': args.control, 'per_run': per_run, 'paired': paired_out}, indent=1))
    for name, row in per_run.items():
        print(f"{name:>22s} ep{row['epochs']:>3d} " + ' '.join(
            f"{c}: {row[c]['psnr_masked']:.2f}/{row[c]['ssim_masked']:.3f}" for c in row if c != 'epochs'))
    for arm, seeds in paired_out.items():
        for seed, entry in seeds.items():
            print(f"{arm:>16s} s{seed} - control: " + ' '.join(
                f"{c} {e['psnr']['mean']:+.3f} [{e['psnr']['ci'][0]:+.3f},{e['psnr']['ci'][1]:+.3f}]" for c, e in entry.items()))


if __name__ == '__main__':
    main()
