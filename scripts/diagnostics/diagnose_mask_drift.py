"""How far did autosamp positions / REINFORCE-vd logits move from their seed init?"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path('/home/bkadota/code/active_mri/SeqMRI')
sys.path[:0] = [str(ROOT)]
os.chdir(ROOT)
from activemri.training.trainer import build_model  # noqa: E402
from activemri.training.builders import build_dataloaders  # noqa: E402

RINGS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]


def recover_init(run):
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    build_dataloaders(cfg)
    ckpt = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=False)
    model = build_model(ckpt['options'])
    return model, ckpt


def radius(h, w):
    ky = (torch.arange(h) - h // 2) / (h / 2)
    kx = (torch.arange(w) - w // 2) / (w / 2)
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def ring_table(maps, r):
    print('ring        ' + ' '.join(f'{k:>14s}' for k in maps))
    for lo, hi in zip(RINGS[:-1], RINGS[1:]):
        sel = (r >= lo) & (r < hi)
        print(f'{lo:.2f}-{hi:.2f}   ' + ' '.join(f'{v[sel].float().mean().item():14.4f}' for v in maps.values()))


def autosamp(run):
    model, ckpt = recover_init(run)
    init = model.sampler.mask_net.positions.detach().clone()
    final = ckpt['model']['sampler.mask_net.positions'].float()
    h, w = model.sampler.mask_net.shape
    size = torch.tensor([h, w]).float()
    delta = final - init
    delta = torch.remainder(delta + size / 2, size) - size / 2  # wrapped displacement
    dist = delta.norm(dim=1)
    centre = torch.tensor([h // 2, w // 2]).float()
    r_init = (init - centre).norm(dim=1)
    r_final = (torch.remainder(final + 0.5, size) - 0.5 - centre).norm(dim=1)
    print(f'\n=== {run.name}  (epoch {ckpt.get("epoch")}) points={len(init)}')
    print('displacement quantiles (cells) 50/90/99/max:',
          [round(dist.quantile(q).item(), 3) for q in (0.5, 0.9, 0.99)], round(dist.max().item(), 3))
    print('frac points on different cell than init:', (final.round() != init.round()).any(1).float().mean().item())
    print('mean radial change (cells, - = toward DC):', (r_final - r_init).mean().item())
    rad = radius(h, w)
    def density(pos):
        m = torch.zeros(h * w)
        c = torch.remainder(pos.round(), size).long()
        m[c[:, 0] * w + c[:, 1]] = 1
        return m.view(h, w)
    ring_table({'init density': density(init), 'final density': density(final)}, rad)


def reinforce(run):
    model, ckpt = recover_init(run)
    init = model.sampler.mask_net.mask.detach().float().squeeze()
    final = ckpt['model']['sampler.mask_net.mask'].float().squeeze()
    print(f'\n=== {run.name}  (epoch {ckpt.get("epoch")})')
    h, w = init.shape
    rad = radius(h, w)
    k = 8092
    def topk(x):
        m = torch.zeros(h * w); m[x.flatten().topk(k).indices] = 1; return m.view(h, w)
    print('global corr(init, final):', torch.corrcoef(torch.stack([init.flatten(), final.flatten()]))[0, 1].item())
    ring_table({'init logit': init, 'final logit': final, '|drift|': (final - init).abs(),
                'init topk dens': topk(init), 'final topk dens': topk(final)}, rad)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        run = ROOT / 'checkpoints' / name
        (autosamp if 'autosamp' in name else reinforce)(run)
