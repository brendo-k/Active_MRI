"""Crosses the LOUPE masks of a short run and the 100-epoch reference, to tell which part of the short run falls behind.

Writes two checkpoints, each with one run's reconstructor and sensitivity net and the other run's mask, as runs the eval
scores like any other:
    <reference>_maskof_<short>_s<seed>   the reference's networks, the short run's mask
    <short>_maskof_<reference>_s<seed>   the short run's networks, the reference's mask
With the two original runs this makes a 2x2. A network sees a mask it did not train on, so each swapped run also carries a
mismatch cost, and the two directions bracket the mask's own effect.

    python scripts/diagnostics/muon_layer_speed_swap_masks.py <queue dir> e30_step ref 42
"""
import sys
from pathlib import Path

import torch

MASK_KEY = 'sampler.mask_net.mask'


def write_swapped(queue_dir, networks_from, mask_from, seed):
    networks = torch.load(queue_dir / f'{networks_from}_s{seed}' / 'last_model.pt', map_location='cpu', weights_only=False)
    mask = torch.load(queue_dir / f'{mask_from}_s{seed}' / 'last_model.pt', map_location='cpu', weights_only=False)
    swapped = {'epoch': networks['epoch'], 'options': networks['options'], 'best_val_loss': networks['best_val_loss'],
               'model': {**networks['model'], MASK_KEY: mask['model'][MASK_KEY]}}
    out_dir = queue_dir / f'{networks_from}_maskof_{mask_from}_s{seed}'
    out_dir.mkdir(exist_ok=True)
    torch.save(swapped, out_dir / 'last_model.pt')
    print(out_dir)


def main():
    queue_dir, short, reference, seed = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3], sys.argv[4]
    write_swapped(queue_dir, reference, short, seed)
    write_swapped(queue_dir, short, reference, seed)


if __name__ == '__main__':
    main()
