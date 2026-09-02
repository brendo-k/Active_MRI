"""Single-volume inference script for SequentialSampling (LLS/CG-SENSE).

Usage:
    python scripts/run_inference.py \
        --checkpoint <path/to/model.pt> \
        --data-path  /home/bkadota/data/fastmri/multicoil_train/T2 \
        --volume-idx 0 \
        --device cuda:3 \
        --out-dir /tmp/inference_out
"""
import argparse
import pathlib
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from activemri.datasets.real_brain_data import RealBrainData
from activemri.training.trainer import build_model


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"Loaded checkpoint epoch={checkpoint['epoch']}, ") #val_loss={checkpoint['best_val_loss']:.6f}")
    options = checkpoint['options']
    options.device = device
    model = build_model(options).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, options


def run_inference(model, kspace, target, device):
    kspace = kspace.unsqueeze(0).to(device)   # [1, C, H, W, 2]
    target = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        pred_dict = model(target, kspace)

    return pred_dict, target


def save_results(pred_dict, target, sense_mask, out_dir, volume_idx):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recon   = np.flip(pred_dict['output'][0, 0].cpu().numpy(), 0)
    gt      = np.flip(target[0, 0].cpu().numpy(), 0)
    masks   = np.flip(pred_dict['masks'], 0)          # list of [1,1,H,W] per step
    num_steps = len(masks)

    fig, axes = plt.subplots(2, 2 + num_steps, figsize=[4 * (2 + num_steps), 8])

    axes[0, 0].imshow(gt, cmap='gray')
    axes[0, 0].set_title('Ground truth')

    axes[0, 1].imshow(recon, cmap='gray')
    axes[0, 1].set_title('Reconstruction')

    for s, m in enumerate(masks):
        axes[0, 2 + s].imshow(m[0, 0].cpu().numpy(), cmap='gray')
        axes[0, 2 + s].set_title(f'Sampling mask step {s+1}')

    if sense_mask is not None:
        axes[1, 0].imshow(sense_mask[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Sense body mask')

    axes[1, 1].imshow(np.abs(recon - gt), cmap='hot')
    axes[1, 1].set_title('Abs error')

    for s, zf in enumerate(pred_dict['zero_filled_recon']):
        axes[1, 2 + s].imshow(zf[0, 0].cpu().numpy(), cmap='gray')
        axes[1, 2 + s].set_title(f'Zero-filled step {s+1}')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    out_path = out_dir / f'inference_vol{volume_idx}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")

    nmse = (np.linalg.norm(recon - gt) ** 2) / (np.linalg.norm(gt) ** 2)
    print(f"NMSE: {nmse:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path',  type=str, required=True)
    parser.add_argument('--volume-idx', type=int, default=0)
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--out-dir',    type=str, default='/tmp/inference_out')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, options = load_checkpoint(args.checkpoint, device)

    # Use the flat training folder directly (same as original args.data_path)
    dataset = RealBrainData(
        pathlib.Path(args.data_path) / 'train',
        options.resolution,
        transform=None,
        noise_type='none',
        noise_level=0.0,
        random_rotate=False,
    )

    kspace, _, target, attrs, fname, slice_id = dataset[args.volume_idx]
    print(f"File: {fname}, slice: {slice_id}")
    print(f"Volume {args.volume_idx}: kspace {kspace.shape}, target {target.shape}")

    pred_dict, target_t = run_inference(model, kspace, target, device)

    # sense_mask is stored in data_for_vis by step_forward
    sense_mask_np = model.data_for_vis[0].get('sense_mask')
    sense_mask = torch.from_numpy(sense_mask_np).unsqueeze(0).unsqueeze(0) if sense_mask_np is not None else None

    save_results(pred_dict, target_t, sense_mask, args.out_dir, args.volume_idx)


if __name__ == '__main__':
    main()
