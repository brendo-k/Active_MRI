"""Re-score checkpoints with the trainer's current metric mask next to the legacy target-threshold mask.

Runs each checkpoint through the trainer's validation path (anatomy labels, seeded eval generator, sense maps) on
the run's own validation set, then accumulates the same per-anatomy PSNR/SSIM/NMSE twice from the same outputs:
under the legacy mask (target > 0.1 * max) and under the new apodized per-anatomy mask.

    python scripts/eval/reeval_metric_mask_r8.py --runs 'loupe2d_cgsense_r8_*' --device cuda:0
    python scripts/eval/reeval_metric_mask_r8.py --table
"""
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import fnmatch
import json
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import REINFORCE_VAL_MASK_SEED, _AnatomyStats, _apply_signal_mask, _signal_mask, build_model
from activemri.utils.metric_mask import metric_mask_from_real_kspace

CHECKPOINT_ROOT = Path('checkpoints')
OUT_DIR = Path('reports/metric_mask_reeval_r8')
METRIC_NAMES = ('PSNR', 'SSIM', 'NMSE')


def find_runs(patterns, checkpoint):
    runs = sorted(path for path in CHECKPOINT_ROOT.iterdir() if (path / checkpoint).exists())
    return [run for run in runs if any(fnmatch.fnmatch(run.name, pattern) for pattern in patterns)]


def model_inputs(batch, options, device):
    inputs = {}
    if options.label_classes:
        index = {name: i for i, name in enumerate(options.label_classes)}
        inputs['anatomy_labels'] = torch.tensor([index[name] for name in batch[options.label_key]], device=device)
    else:
        inputs['anatomy_labels'] = None
    for key in ('sense_maps', 'kspace_support'):
        if key in batch:
            inputs[key] = batch[key].to(device)
    return inputs


@torch.no_grad()
def reevaluate(run, checkpoint, device, num_workers):
    ckpt = torch.load(run / checkpoint, map_location='cpu', weights_only=False)
    options = ckpt['options']
    model = build_model(options).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(REINFORCE_VAL_MASK_SEED)
    model.sampler.eval_generator = generator

    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = num_workers
    _, val_loader = build_dataloaders(cfg)
    stats = {'legacy': _AnatomyStats(device), 'new': _AnatomyStats(device)}
    mask_fraction = {'legacy': [], 'new': []}
    skipped = 0
    for batch in val_loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        anatomies = list(batch['anatomy'])
        try:
            pred = model(target, kspace, **model_inputs(batch, options, device))
        except RuntimeError as err:
            if 'Binarization failed to match target density' not in str(err):
                raise
            skipped += 1
            continue
        output = pred['output'][-1] if isinstance(pred['output'], list) else pred['output']
        target = target if output.dim() == target.dim() else target.unsqueeze(1)
        masks = {'legacy': _signal_mask(target), 'new': metric_mask_from_real_kspace(kspace, anatomies)}
        for name, mask in masks.items():
            stats[name].update(anatomies, *_apply_signal_mask(output, target, mask), pred['mask'])
            mask_fraction[name] += mask.flatten(1).mean(1).tolist()
    result = {'run': run.name, 'checkpoint': checkpoint, 'epoch': ckpt.get('epoch'), 'total_epochs': options.num_epochs,
              'skipped_batches': skipped, 'mask_fraction': {k: sum(v) / len(v) for k, v in mask_fraction.items()}}
    for name, accumulator in stats.items():
        result[name] = {anatomy: {metric: float(metrics[metric].compute()) for metric in METRIC_NAMES}
                        for anatomy, metrics in accumulator.metrics.items()}
    return result


def table(out_dir):
    rows = ['| run | epoch | anatomy | PSNR legacy | PSNR new | SSIM legacy | SSIM new | NMSE legacy | NMSE new |',
            '|---|---|---|---|---|---|---|---|---|']
    for path in sorted(out_dir.glob('*.json')):
        result = json.loads(path.read_text())
        epoch = f"{result['epoch'] + 1}/{result['total_epochs']}" if result['epoch'] is not None else '?'
        for anatomy in result['new']:
            old, new = result['legacy'][anatomy], result['new'][anatomy]
            rows.append(f"| {result['run']} | {epoch} | {anatomy} | {old['PSNR']:.2f} | {new['PSNR']:.2f} | {old['SSIM']:.4f} | "
                        f"{new['SSIM']:.4f} | {old['NMSE']:.4f} | {new['NMSE']:.4f} |")
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', default=['*'], help='checkpoint directory names or globs')
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--table', action='store_true', help='only print the markdown table of the JSONs already in --out')
    args = parser.parse_args()
    if args.table:
        print(table(args.out))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for run in find_runs(args.runs, args.checkpoint):
        target = args.out / f'{run.name}.json'
        if target.exists():
            print(f'skip {run.name} (done)', flush=True)
            continue
        try:
            result = reevaluate(run, args.checkpoint, args.device, args.num_workers)
        except Exception:
            print(f'FAILED {run.name}\n{traceback.format_exc()}', flush=True)
            continue
        target.write_text(json.dumps(result, indent=2) + '\n')
        summary = {anatomy: round(metrics['PSNR'], 2) for anatomy, metrics in result['new'].items()}
        print(f'done {run.name}: new-mask PSNR {summary}', flush=True)


if __name__ == '__main__':
    main()
