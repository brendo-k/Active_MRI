"""Quick eval: load one batch and test both models."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from pathlib import Path
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.reconstructors.VarNet import VarNet
from activemri.models.sequential_sampling import SequentialSampling
from activemri.utils.util import compute_ssim_torch, compute_psnr_torch

device = torch.device('cuda:0')

print("Loading val set...")
_, val_loader = build_fastmri_dataloaders(
    data_root=Path('/home/bkadota/data/fastmri'),
    anatomies=['brain', 'knee'],
    image_shape=(256, 256),
    batch_size=8,
    num_coils=16,
    num_workers=0,  # Disable workers to avoid hanging
    balance=True,
)
print(f"val set: {len(val_loader)} batches")

for model_name, ckpt_path, mask_mode in [
    ('LOUPE', 'checkpoints/loupe2d_varnet_r8_brain_knee/best_model.pt', 'straight_through'),
    ('REINFORCE', 'checkpoints/reinforce2d_varnet_r8_brain_knee_v2/best_model.pt', 'reinforce'),
]:
    print(f"\n=== {model_name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('best_val_loss', '?'):.4g}")
    
    model = SequentialSampling(
        reconstructor=VarNet(num_cascades=12, chans=18, pools=4),
        sparsity=1.0/8.0, shape=(256, 256), num_step=1, preselect_num=10, sampler_type='loupe_2d',
        slope=10.0, with_uncertainty=False, recon_type='varnet', sense_mask_method='pre_network',
        mask_train_mode=mask_mode, sampler_chans=64, sampler_depth=4, reinforce_vd_power=0.0,
    ).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    psnrs, ssims = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 5: break  # Just first 5 batches for quick test
            kspace, target = batch['kspace'].to(device), batch['target'].to(device)
            pred = model(target, kspace)['output']
            if pred.dim() != target.dim(): 
                target = target.unsqueeze(1)
            ssim = compute_ssim_torch(pred, target).mean().item()
            psnr = compute_psnr_torch(pred, target).mean().item()
            ssims.append(ssim)
            psnrs.append(psnr)
            print(f"  batch {batch_idx}: PSNR={psnr:6.2f} SSIM={ssim:.4f}")
    
    if psnrs:
        print(f"Quick test: PSNR {np.mean(psnrs):.2f}±{np.std(psnrs):.2f}  SSIM {np.mean(ssims):.4f}±{np.std(ssims):.4f}  batches={len(psnrs)}")
