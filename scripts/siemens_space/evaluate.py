"""Score SSDU checkpoints on the held-out readout positions, one reconstructor under several masks.

    python scripts/siemens_space/evaluate.py loupe=<run>/best_model.pt random=<run2>/best_model.pt

For each checkpoint's reconstructor and sense maps, on the val set with its fixed hold-out set Lambda:
    own      the checkpoint's own sampler mask
    random   mean +- std over seeded uniform draws from the same input pool and budget
    pool     every acquired point outside Lambda (~7.8x on MID00295): the most this reconstructor can be given

Reported: SSDU loss on Lambda (the self-supervised objective, no reference involved) and PSNR / SSIM / NMSE
against the PICS image of everything acquired, over the signal region as the trainer scores it. That reference
is itself a compressed-sensing reconstruction, so read the image metrics as agreement with it, not accuracy.
"""
import _path  # noqa: F401
import argparse
import json
import pathlib

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from activemri.datasets.siemens_space_data import SiemensSpaceDataset
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.training.trainer import _apply_signal_mask, build_model
from activemri.utils.evaluate import build_metrics

CACHE_DIR = pathlib.Path.home() / 'data/my_scans/oct_7_2026_cache'
RANDOM_SEEDS = range(5)
CG_ITERATIONS, CG_LAMDA = 30, 0.01  # the classical baseline given the identical mask and data
OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / 'analysis/oct7_scan'


def load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    options = checkpoint['options']
    model = build_model(options).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint['model'].items()})
    return model.eval(), options, checkpoint['epoch']


@torch.no_grad()
def score(model, loader, device, seed=None, use_pool=False):
    generator = torch.Generator(device=device)
    if seed is not None:
        model.sampler.mask_train_mode = 'random'
        model.sampler.eval_generator = generator
        generator.manual_seed(seed)
    model.mask_policy = (lambda kspace: model._valid_kspace_mask(kspace)) if use_pool else None

    metrics, cg_metrics = build_metrics(device=device), build_metrics(device=device)
    ssdu, cg_ssdu, counts, first = [], [], [], None
    for batch in loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        out = model(target, kspace, sense_maps=batch['sense_maps'].to(device), kspace_support=batch['kspace_support'].to(device))
        target_dict = {'holdout_mask': batch['holdout_mask'].to(device), 'holdout_kspace': batch['holdout_kspace'].to(device)}
        ssdu.append(model.ssdu_loss(out, target_dict).cpu())
        sense, mask = out['sense_maps'], out['mask']
        cg = conjugate_gradient_sense(torch.view_as_complex(kspace.contiguous()) * mask, sense, mask, CG_ITERATIONS, CG_LAMDA)
        cg_ssdu.append(model.ssdu_loss({'recon_complex': cg.unsqueeze(1), 'sense_maps': sense}, target_dict).cpu())
        cg_image = (cg.abs() * (sense.abs().pow(2).sum(1) > 0).float()).unsqueeze(1)
        cg_metrics['PSNR'].data_range = cg_metrics['SSIM'].data_range = target.max()
        cg_metrics.update(*_apply_signal_mask(cg_image, target.unsqueeze(1)))
        counts.append(out['mask'].sum((1, 2, 3)).cpu())
        metrics['PSNR'].data_range = metrics['SSIM'].data_range = target.max()
        masked_output, masked_target = _apply_signal_mask(out['output'], target.unsqueeze(1))
        metrics.update(masked_output, masked_target)
        if first is None:
            first = {'recon': out['output'][0, 0].cpu(), 'mask': out['mask'][0, 0].cpu(), 'target': target[0].cpu()}
    result = {name: value.item() for name, value in metrics.compute().items()}
    result['SSDU'] = torch.cat(ssdu).mean().item()
    result.update({f'CG_{name}': value.item() for name, value in cg_metrics.compute().items()})
    result['CG_SSDU'] = torch.cat(cg_ssdu).mean().item()
    result['points'] = torch.cat(counts).mean().item()
    return result, first


def mean_std(rows):
    return {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows]))) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', help='name=path pairs')
    parser.add_argument('--volumes', nargs='+', default=['meas_MID00295_FID30725_bk_space'])
    parser.add_argument('--holdout-fraction', type=float, default=0.2)
    parser.add_argument('--tag', default='eval')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    results, images = {}, {}
    for pair in args.checkpoints:
        name, path = pair.split('=', 1)
        model, options, epoch = load(path, args.device)
        dataset = SiemensSpaceDataset([CACHE_DIR / f'{v}.h5' for v in args.volumes], 'val', args.holdout_fraction, tuple(options.resolution))
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        own, images[name] = score(model, loader, args.device)
        random_rows = [score(model, loader, args.device, seed=s)[0] for s in RANDOM_SEEDS]
        pool, _ = score(model, loader, args.device, use_pool=True)
        results[name] = {'epoch': epoch, 'own': own, 'random': mean_std(random_rows), 'pool': pool}
        print(f'{name} (best epoch {epoch}, R={options.accelerations[0]:g}, sampler={options.sampler_type}/{options.mask_train_mode})')
        for label, row in (('own', own), ('random', mean_std(random_rows)), ('pool', pool)):
            fmt = lambda v: f'{v[0]:.4f}±{v[1]:.4f}' if isinstance(v, tuple) else f'{v:.4f}'
            print(f'  {label:7s} ' + '  '.join(f'{k}={fmt(row[k])}' for k in ('SSDU', 'PSNR', 'SSIM', 'CG_SSDU', 'CG_PSNR', 'CG_SSIM', 'points')))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f'{args.tag}.json').write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(len(images), 3, figsize=(10, 4.2 * len(images)), squeeze=False)
    for row, (name, shown) in zip(axes, images.items()):
        for ax, (key, title) in zip(row, (('target', 'PICS reference'), ('recon', f'{name} recon'), ('mask', f'{name} mask'))):
            ax.imshow(shown[key], cmap='gray', aspect='auto')
            ax.set_title(title)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{args.tag}.png', dpi=90)


if __name__ == '__main__':
    main()
