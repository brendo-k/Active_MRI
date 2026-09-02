import os
import sys
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import random
from pathlib import Path

import numpy as np
import torch

from activemri.training.trainer import build_model, _apply_signal_mask
from activemri.utils.evaluate import build_metrics
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.config import TrainConfig


def evaluate_checkpoint(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    options: TrainConfig = checkpoint['options']

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    state_dict = checkpoint['model']

    # Detect if checkpoint uses old ConvTranspose2d (key: upsample.weight)
    # vs current bilinear Sequential (key: upsample.1.weight)
    needs_conv_mode = any(
        'up_sample_layers' in k and '.upsample.weight' in k and '.upsample.1.' not in k
        for k in state_dict.keys()
    )

    if needs_conv_mode:
        # Temporarily monkey-patch the default upsample_mode
        import activemri.reconstructors.UNet as unet_module
        OrigUpsampleBlock = unet_module.upsample_block
        class ConvUpsampleBlock(OrigUpsampleBlock):
            def __init__(self, in_chans, out_chans, drop_prob, upsample_mode='conv'):
                super().__init__(in_chans, out_chans, drop_prob, upsample_mode='conv')
        unet_module.upsample_block = ConvUpsampleBlock

        model = build_model(options).to(device)
        unet_module.upsample_block = OrigUpsampleBlock
    else:
        model = build_model(options).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    _, val_loader = build_fastmri_dataloaders(
        data_root=Path('/home/bkadota/data/fastmri'),
        anatomies=['brain'],
        image_shape=options.resolution,
        batch_size=4,
        num_coils=16,
        num_workers=4,
        contrast='T2',
        min_energy_fraction=options.min_energy_fraction,
    )

    metrics_masked = build_metrics(device=device)
    metrics_unmasked = build_metrics(device=device)

    with torch.no_grad():
        for iter_idx, batch in enumerate(val_loader):
            kspace = batch['kspace'].to(device)
            target = batch['target'].to(device)

            pred_dict = model(target, kspace)
            output = pred_dict['output']

            target_for_metrics = target
            if output.dim() != target_for_metrics.dim():
                target_for_metrics = target_for_metrics.unsqueeze(1)

            data_range = target_for_metrics.max()

            metrics_unmasked["PSNR"].data_range = data_range
            metrics_unmasked["SSIM"].data_range = data_range
            metrics_unmasked.update(output, target_for_metrics)

            metrics_masked["PSNR"].data_range = data_range
            metrics_masked["SSIM"].data_range = data_range
            masked_output, masked_target = _apply_signal_mask(output, target_for_metrics)
            metrics_masked.update(masked_output, masked_target)

    masked_results = metrics_masked.compute()
    unmasked_results = metrics_unmasked.compute()
    return masked_results, unmasked_results, options


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoints = {
        'nomask (0ug80cd5)': 'checkpoints/brain_ablation_nomask_320/best_model.pt',
        'baseline (xz1n8bul)': 'checkpoints/brain_ablation_baseline_320/best_model.pt',
    }

    print("=" * 80)
    print("Re-evaluation with signal masking comparison")
    print("=" * 80)

    for name, ckpt_path in checkpoints.items():
        print(f"\n{'─' * 40}")
        print(f"Checkpoint: {name}")
        print(f"Path: {ckpt_path}")
        print(f"{'─' * 40}")

        masked, unmasked, options = evaluate_checkpoint(ckpt_path, device)

        print(f"  sense_mask_method: {options.sense_mask_method}")
        print(f"  min_energy_fraction: {options.min_energy_fraction}")
        print()
        print(f"  {'Metric':<8} {'Masked':<12} {'Unmasked':<12}")
        print(f"  {'─' * 32}")
        for key in masked.keys():
            m_val = masked[key].item()
            u_val = unmasked[key].item()
            print(f"  {key:<8} {m_val:<12.4f} {u_val:<12.4f}")

    print("\n" + "=" * 80)
