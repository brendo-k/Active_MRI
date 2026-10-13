"""Evaluate LOUPE vs REINFORCE on val set (using same split as training)."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from pathlib import Path
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.reconstructors.VarNet import VarNet
from activemri.models.sequential_sampling import SequentialSampling
from activemri.utils.util import compute_ssim_torch, compute_psnr_torch

def evaluate_run(name, ckpt_path, val_loader, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f'\n=== {name}'); print(f'epoch: {ckpt.get("epoch", "?")}, val loss: {ckpt.get("best_val_loss", "?"):.4g}')
    
    model = SequentialSampling(
        reconstructor=VarNet(num_cascades=12, chans=18, pools=4),
        sparsity=1.0/8.0, shape=(256, 256), num_step=1, preselect_num=10, sampler_type='loupe_2d',
        slope=10.0, with_uncertainty=False, recon_type='varnet', sense_mask_method='pre_network',
        mask_train_mode='straight_through' if 'loupe2d' in ckpt_path else 'reinforce',
        sampler_chans=64, sampler_depth=4, reinforce_vd_power=0.0,
    ).to(device)
    
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    psnrs, ssims = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            kspace, target = batch['kspace'].to(device), batch['target'].to(device)
            try:
                pred = model(target, kspace)['output']
                if pred.dim() != target.dim(): target = target.unsqueeze(1)
                ssims.append(compute_ssim_torch(pred, target).mean().item())
                psnrs.append(compute_psnr_torch(pred, target).mean().item())
                if batch_idx % 20 == 0: print(f'  batch {batch_idx:3d}: PSNR={psnrs[-1]:6.2f} SSIM={ssims[-1]:.4f}')
            except Exception as e: 
                print(f'  batch {batch_idx}: {type(e).__name__}')
    
    if psnrs: 
        mean_p, std_p = np.mean(psnrs), np.std(psnrs)
        mean_s, std_s = np.mean(ssims), np.std(ssims)
        print(f'final: PSNR {mean_p:.2f}±{std_p:.2f}  SSIM {mean_s:.4f}±{std_s:.4f}  n={len(psnrs)}')
    return psnrs, ssims

device = torch.device('cuda:0')

_, val_loader = build_fastmri_dataloaders(
    data_root=Path('/home/bkadota/data/fastmri'),
    anatomies=['brain', 'knee'],
    image_shape=(256, 256),
    batch_size=8,
    num_coils=16,
    num_workers=4,
    balance=True,
)
print(f'val set: {len(val_loader)} batches')

loupe_p, loupe_s = evaluate_run('LOUPE', 'checkpoints/loupe2d_varnet_r8_brain_knee/best_model.pt', val_loader, device)
rf_p, rf_s = evaluate_run('REINFORCE', 'checkpoints/reinforce2d_varnet_r8_brain_knee_v2/best_model.pt', val_loader, device)

print('\n=== SUMMARY')
print(f'LOUPE:      PSNR {np.mean(loupe_p):.2f}  SSIM {np.mean(loupe_s):.4f}')
print(f'REINFORCE:  PSNR {np.mean(rf_p):.2f}  SSIM {np.mean(rf_s):.4f}')
print(f'Winner: REINFORCE' if np.mean(rf_p) > np.mean(loupe_p) else 'Winner: LOUPE')
