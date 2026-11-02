"""Save the R=8 static variable-density mask of a frozen-autosamp run as a `sampler.fixed_mask_path` file.

The static-VD recipe draws AutoSamp's positions from the global RNG after the reconstructor's weights are initialised,
so two runs with the same seed but different reconstructors get different masks. A saved mask removes that: every run
that takes it sees the same pattern. The mask is read off the run's forward pass on val slices (centre included), and
must be the same on every slice of both anatomies.

    python scripts/diagnostics/export_static_vd_mask_r8.py RUN_DIR OUT.pt
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--checkpoint', default='last_model.pt')
    args = parser.parse_args()

    ckpt = torch.load(args.run / args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()
    _, val_loader = build_dataloaders(OmegaConf.load(args.run / '.hydra' / 'config.yaml'))

    masks, anatomies = [], set()
    for batch in val_loader:
        if set(batch['anatomy']) <= anatomies:
            continue
        anatomies |= set(batch['anatomy'])
        output = model(batch['target'].cuda(), batch['kspace'].cuda())
        masks.append(output['mask'][:, 0].cpu())
        if anatomies >= {'brain', 'knee'}:
            break
    masks = torch.cat(masks)
    mask = masks[0]
    if not bool((masks == mask).all()):
        raise ValueError(f"{args.run} gives different masks on different slices; it is not a static mask")
    print(f"{args.run.name} epoch {ckpt['epoch']}: {masks.shape[0]} slices from {sorted(anatomies)} share one mask, "
          f"{int(mask.sum())} of {mask.numel()} locations (R = {mask.numel() / mask.sum():.3f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'mask': mask, 'source': str(args.run / args.checkpoint), 'epoch': ckpt['epoch']}, args.out)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
