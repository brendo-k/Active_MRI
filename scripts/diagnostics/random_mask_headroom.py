"""Is there anything for a sampler policy to learn at R=8 (brain + knee)?

For val slices spread over both anatomies, scores NUM_MASKS random R=8 masks per slice (a
centre block plus uniformly random points, i.e. PPO's initial policy) with two frozen
reconstructors, and compares them with LOUPE's learned mask:

  loupe   the R=8 VarNet trained jointly with LOUPE (reconstructor matched to R=8 masks)
  caipi   the R=4 CAIPI VarNet whose NMSE is PPO's reward in the sweep

Within-slice spread across random masks is what a policy gradient has to work with (the
critic removes the between-slice difficulty); the LOUPE gap is what it would need to find.

    CUDA_VISIBLE_DEVICES=3 python scripts/diagnostics/random_mask_headroom.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.envs.mri_env import MRIEnv, center_block_mask
from activemri.training.trainer import build_model

LOUPE = Path('checkpoints/loupe2d_varnet_r8_brain_knee/best_model.pt')
CAIPI = 'checkpoints/varnet_caipi2d_r4_brain_knee/best_model.pt'
NUM_SLICES = 32
NUM_MASKS = 64
TOTAL_POINTS = 256 * 256 // 8
CENTER_BLOCK = 16
BATCH = 8
DEVICE = torch.device('cuda')


def random_masks(num, generator):
    """`[num, 1, H, W]` centred masks: the centre block plus random points up to TOTAL_POINTS."""
    centre = center_block_mask(1, [256, 256], CENTER_BLOCK, DEVICE).flatten()
    free = (centre == 0).nonzero().squeeze(1)
    chosen = torch.rand(num, free.numel(), device=DEVICE, generator=generator).topk(
        TOTAL_POINTS - CENTER_BLOCK**2, dim=1).indices
    masks = centre.repeat(num, 1)
    masks.scatter_(1, free[chosen], 1.0)
    return masks.view(num, 1, 256, 256)


@torch.no_grad()
def nmse(reconstructor, sense_maps, kspace_complex, masks, target):
    recon = reconstructor(kspace_complex * masks.squeeze(1).unsqueeze(1), sense_maps, masks).abs()
    return ((recon.squeeze(1) - target).pow(2).flatten(1).sum(1) / target.pow(2).flatten(1).sum(1)).cpu()


def main():
    _, val_loader = build_fastmri_dataloaders(
        Path('/home/bkadota/data/fastmri'), ['brain', 'knee'], [256, 256], BATCH, num_coils=16,
        num_workers=2, contrast='T2', balance=True)
    size = len(val_loader.dataset)
    indices = np.linspace(0, size - 1, NUM_SLICES).round().astype(int)
    loader = DataLoader(Subset(val_loader.dataset, indices.tolist()), batch_size=BATCH, shuffle=False,
                        collate_fn=val_loader.collate_fn)
    is_brain = indices < size // 2

    checkpoint = torch.load(LOUPE, map_location='cpu', weights_only=False)
    loupe = build_model(checkpoint['options'])
    loupe.load_state_dict(checkpoint['model'])
    loupe.to(DEVICE).eval()
    caipi = MRIEnv([], [256, 256], 1, recon_checkpoint=CAIPI, device='cuda')

    generator = torch.Generator(device=DEVICE).manual_seed(0)
    rows = {name: [] for name in ('loupe_random', 'loupe_mask', 'caipi_random', 'caipi_loupe_mask')}
    for batch in loader:
        kspace = batch['kspace'].to(DEVICE)
        target = batch['target'].to(DEVICE).squeeze(1)
        kspace_complex = torch.view_as_complex(kspace.contiguous())
        with torch.no_grad():
            out = loupe(batch['target'].to(DEVICE), kspace)
            loupe_sense = out['sense_maps']
            caipi_sense, _ = caipi.sensetivity_net(kspace_complex, mask_method='pre_network')
        loupe_mask = out['mask']
        rows['loupe_mask'].append(nmse(loupe.reconstructor, loupe_sense, kspace_complex, loupe_mask, target))
        rows['caipi_loupe_mask'].append(nmse(caipi.reconstructor, caipi_sense, kspace_complex, loupe_mask, target))
        for key, reconstructor, sense in (('loupe_random', loupe.reconstructor, loupe_sense),
                                          ('caipi_random', caipi.reconstructor, caipi_sense)):
            scores = []
            generator.manual_seed(0)
            for _ in range(NUM_MASKS):
                masks = random_masks(kspace.shape[0], generator)
                scores.append(nmse(reconstructor, sense, kspace_complex, masks, target))
            rows[key].append(torch.stack(scores, dim=1))  # [B, NUM_MASKS]
        print(f"batch done, loupe mask points {loupe_mask.sum((1, 2, 3)).mean():.0f}", flush=True)

    result = {k: torch.cat(v).numpy() for k, v in rows.items()}
    print(f"\n{NUM_SLICES} val slices, {NUM_MASKS} random R=8 masks each (NMSE of |recon| vs target, unmasked)\n")
    for recon, random_key, loupe_key in (('loupe', 'loupe_random', 'loupe_mask'),
                                         ('caipi', 'caipi_random', 'caipi_loupe_mask')):
        print(f"reconstructor: {recon}")
        print(f"{'anatomy':8} {'rand mean':>10} {'within-slice std':>17} {'(CV)':>6} {'between-slice std':>18} "
              f"{'LOUPE mask':>11} {'gap':>8} {'gap / within std':>17} {'slices LOUPE better':>20}")
        for name, selection in (('brain', is_brain), ('knee', ~is_brain), ('all', np.ones_like(is_brain))):
            r, m = result[random_key][selection], result[loupe_key][selection]
            slice_mean, within = r.mean(1), r.std(1)
            gap = slice_mean - m
            print(f"{name:8} {slice_mean.mean():10.4f} {within.mean():17.4f} {(within / slice_mean).mean():6.2f} "
                  f"{slice_mean.std():18.4f} {m.mean():11.4f} {gap.mean():8.4f} {(gap / within).mean():17.1f} "
                  f"{(m < slice_mean).mean():20.0%}")
        best = result[random_key].min(1)
        print(f"best of {NUM_MASKS} random masks vs LOUPE mask: random best beats LOUPE on "
              f"{(best < result[loupe_key]).mean():.0%} of slices\n")


if __name__ == '__main__':
    main()
