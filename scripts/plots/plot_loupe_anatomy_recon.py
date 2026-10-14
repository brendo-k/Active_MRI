"""Learned mask, validation metrics and reconstructed slices for the per-anatomy LOUPE 2D runs.

Reads <root>/<anatomy>/seed_<n>/best_model.pt written by scripts/launch/train_loupe_anatomy.sh.
Metrics follow the trainer's validation path (background zeroed by the target's signal
mask, data_range = batch max) so they match the logged val numbers, not the full-image
scores of the older eval scripts.

    CUDA_VISIBLE_DEVICES=0 python scripts/plots/plot_loupe_anatomy_recon.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.training.trainer import _apply_signal_mask, build_model
from activemri.utils.evaluate import build_metrics

CONFIG_NUM_COILS = {'brain': 16, 'knee': 15}
SLICE_QUANTILES = (0.2, 0.4, 0.6, 0.8)
ERROR_GAIN = 5
RNG_SEED = 1234


def load_model(run_dir, device):
    ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['epoch'] + 1


def masked_pair(output, target):
    target = target if target.dim() == output.dim() else target.unsqueeze(1)
    return _apply_signal_mask(output, target)


def mean_metrics(metric_collection):
    return {name: float(value) for name, value in metric_collection.compute().items()}


@torch.no_grad()
def validation_metrics(model, loader, device):
    metrics = build_metrics(device=device)
    for batch in loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        output = model(target, kspace)['output']
        masked_output, masked_target = masked_pair(output, target)
        data_range = masked_target.max()
        metrics['PSNR'].data_range = data_range
        metrics['SSIM'].data_range = data_range
        metrics.update(masked_output, masked_target)
    return mean_metrics(metrics)


@torch.no_grad()
def reconstruct_slice(model, dataset, loader, index, device):
    batch = loader.collate_fn([dataset[index]])
    kspace, target = batch['kspace'].to(device), batch['target'].to(device)
    torch.manual_seed(RNG_SEED)
    pred = model(target, kspace)
    masked_output, masked_target = masked_pair(pred['output'], target)
    metrics = build_metrics(device=device)
    data_range = masked_target.max()
    metrics['PSNR'].data_range = data_range
    metrics['SSIM'].data_range = data_range
    metrics.update(masked_output, masked_target)
    return {
        'target': masked_target[0, 0].cpu().numpy(),
        'output': masked_output[0, 0].cpu().numpy(),
        'mask': pred['mask'].squeeze().cpu().numpy(),
        'metrics': mean_metrics(metrics),
        'label': f'{batch["fname"][0]} slice {int(batch["slice_id"][0])}',
    }


def learned_probability(model):
    logits = model.sampler.mask_net.mask.detach().float().squeeze().cpu().numpy()
    slope = model.sampler.mask_net.slope if hasattr(model.sampler.mask_net, 'slope') else 10.0
    return 1.0 / (1.0 + np.exp(-slope * logits))


def plot_anatomy(anatomy, probability, slices, val_means, out):
    n = len(slices)
    fig, axes = plt.subplots(3, n + 1, figsize=(3.2 * (n + 1), 9.6))
    axes[0, 0].imshow(probability, cmap='viridis')
    axes[0, 0].set_title(f'learned probability\n(mean {probability.mean():.3f})')
    axes[1, 0].imshow(slices[0]['mask'], cmap='gray')
    axes[1, 0].set_title(f'binary mask drawn for slice 1\n(R=4, {slices[0]["mask"].mean():.1%} sampled)')
    axes[2, 0].plot(np.linspace(-1, 1, probability.shape[1]), probability.mean(axis=0), label='mean over ky (kx profile)')
    axes[2, 0].plot(np.linspace(-1, 1, probability.shape[0]), probability.mean(axis=1), label='mean over kx (ky profile)')
    axes[2, 0].set(title='mean profiles', ylim=(0, 1.02), xlabel='normalised frequency')
    axes[2, 0].legend()

    for col, s in enumerate(slices, start=1):
        vmax = s['target'].max()
        axes[0, col].imshow(s['target'], cmap='gray', vmin=0, vmax=vmax)
        axes[0, col].set_title(f'ground truth\n{s["label"]}', fontsize=8)
        axes[1, col].imshow(s['output'], cmap='gray', vmin=0, vmax=vmax)
        axes[1, col].set_title(f'LOUPE recon  PSNR {s["metrics"]["PSNR"]:.1f} dB  SSIM {s["metrics"]["SSIM"]:.3f}', fontsize=8)
        error = np.abs(s['output'] - s['target'])
        axes[2, col].imshow(error, cmap='inferno', vmin=0, vmax=vmax / ERROR_GAIN)
        axes[2, col].set_title(f'|error| (colour scale = max/{ERROR_GAIN})  NMSE {s["metrics"]["NMSE"]:.4f}', fontsize=8)
    for ax in axes.ravel():
        if ax is not axes[2, 0]:
            ax.axis('off')
    fig.suptitle(
        f'LOUPE 2D trained on {anatomy} only (seed 0, R=4) — full val set: '
        f'PSNR {val_means["PSNR"]:.2f} dB, SSIM {val_means["SSIM"]:.3f}, NMSE {val_means["NMSE"]:.4f}', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/loupe_anatomy'))
    parser.add_argument('--data-root', type=Path, default=Path('/home/bkadota/data/fastmri'))
    parser.add_argument('--anatomies', nargs='+', default=['brain', 'knee'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = args.root / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for anatomy in args.anatomies:
        _, loader = build_fastmri_dataloaders(
            data_root=args.data_root, anatomies=[anatomy], image_shape=[256, 256],
            batch_size=args.batch_size, num_coils=CONFIG_NUM_COILS[anatomy],
            num_workers=args.num_workers, contrast='T2',
        )
        dataset = loader.dataset
        indices = [int(q * len(dataset)) for q in SLICE_QUANTILES]
        results[anatomy] = {}
        for seed in args.seeds:
            model, epochs = load_model(args.root / anatomy / f'seed_{seed}', device)
            torch.manual_seed(RNG_SEED)
            val_means = validation_metrics(model, loader, device)
            results[anatomy][f'seed_{seed}'] = {'checkpoint_epoch': epochs, 'val_slices': len(dataset), **val_means}
            print(f'{anatomy} seed {seed} (epoch {epochs}): ' + ' '.join(f'{k}={v:.4g}' for k, v in val_means.items()), flush=True)
            if seed == args.seeds[0]:
                slices = [reconstruct_slice(model, dataset, loader, i, device) for i in indices]
                plot_anatomy(anatomy, learned_probability(model), slices, val_means, out_dir / f'recon_{anatomy}_seed{seed}.png')
        del model
    (out_dir / 'recon_metrics.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
