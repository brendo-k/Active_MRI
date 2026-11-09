"""Which losses reward subtracting the target's noise floor from the RSS-L1 run's outputs (inside the output support)?
Reads cross_loss_examples.pt written by cross_loss.py."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, '.')
from activemri.utils.image_processing import fft_2d_img

d = torch.load(Path(sys.argv[1]) / 'cross_loss_examples.pt', weights_only=False)
t, o, xe = d['rss']['t'], d['rss']['out'], d['rss']['xe']
omega = (o > 1e-3).float()
floor = torch.stack([t[i][o[i] < 1e-3].median() for i in range(len(t))])[:, None, None]
o_minus_floor = (o - floor).clamp(min=0) * omega
M = (xe.abs() > 0).float()


def ksp(a, b, mask=1.0):
    return (fft_2d_img(a * mask) - fft_2d_img(b * mask)).abs().mean(dim=(-2, -1))


for name, fn in (('image L1', lambda a: (a - t).abs().mean(dim=(-2, -1))), ('kspace L1 (as run)', lambda a: ksp(a, t)),
                 ('kspace L1 in M (as run)', lambda a: ksp(a, t, M)), ('kspace L1 in output support', lambda a: ksp(a, t, omega))):
    keep, drop = fn(o), fn(o_minus_floor)
    print(f"{name:28s} rss output {keep.mean():.5f}   minus floor {drop.mean():.5f}   "
          f"floor removal preferred on {(drop < keep).float().mean():.0%} of slices")
