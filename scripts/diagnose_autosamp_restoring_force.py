"""(a) init points inside the preselect; (b) does the STE gradient pull each point back
to its x.5 snap boundary?  (c) REINFORCE-vd2.5 PL inclusion density, init vs final."""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from diagnose_mask_drift import ROOT, RINGS, radius, recover_init  # noqa: E402
from activemri.samplers.oracle import plackett_luce_sample_and_logprob  # noqa: E402

device = 'cuda'
PRESELECT = 10


def preselect_cells(h, w):
    region = torch.zeros(h, w, dtype=torch.bool)
    n = PRESELECT // 2
    region[h // 2 - n:h // 2 + n, w // 2 - n:w // 2 + n] = True
    return region


def autosamp(name, n_batches=4, kernel_sigma=0.0):
    run = ROOT / 'checkpoints' / name
    model, ckpt = recover_init(run)
    h, w = model.sampler.mask_net.shape
    init = model.sampler.mask_net.positions.detach().round().long()
    inside = preselect_cells(h, w)[init[:, 0], init[:, 1]]
    print(f'\n=== {name}: init points inside {PRESELECT}x{PRESELECT} preselect: {int(inside.sum())}')

    from train_sequential import build_dataloaders
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 2
    cfg.training.batch_size = 4
    train_loader = build_dataloaders(cfg)[0]

    model.load_state_dict(ckpt['model'])
    model.to(device)
    if kernel_sigma > 0:  # the soft forward only runs in training
        model.train()
        model.sampler.mask_net.kernel_sigma = kernel_sigma
    else:
        model.eval()  # eval: no in-place projection; grads still flow
    print(f'  kernel_sigma={kernel_sigma}')
    positions = model.sampler.mask_net.positions
    total = torch.zeros_like(positions)
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        positions.grad = None
        pred = model(batch['target'].to(device), batch['kspace'].to(device))
        target = batch['target'].to(device)
        loss, _ = model.loss(pred, {'target': target, 'kspace': batch['kspace'].to(device)},
                             {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0},
                             cfg.training.get('loss_type', 'l1'))
        loss.backward()
        total += positions.grad
    frac = positions.detach() - positions.detach().floor()
    side = torch.sign(frac - 0.5)              # +1: snapped to the upper cell, -1: lower
    nonzero = total != 0
    restoring = (torch.sign(total) == side) & nonzero  # descent step -grad points back toward x.5
    centre = positions.new_tensor([h // 2, w // 2])
    offset = positions.detach() - centre
    r = offset.norm(dim=1) / (h / 2)
    inward = (-total * -offset).sum(dim=1) > 0  # descent step has a component towards DC
    moving = total.abs().sum(dim=1) > 0
    print('  fraction whose descent step points towards DC, per ring (null 0.5):')
    for lo, hi in zip(RINGS[:-1], RINGS[1:]):
        sel = moving & (r.cpu() >= lo).to(moving.device) & (r < hi)
        print(f'    {lo:.2f}-{hi:.2f}: {inward[sel].float().mean():.3f} (n={int(sel.sum())})')
    for axis, label in enumerate(('ky', 'kx')):
        nz = nonzero[:, axis]
        near = nz & ((frac[:, axis] - 0.5).abs() < 0.02)
        far = nz & ((frac[:, axis] - 0.5).abs() > 0.2)
        print(f'  {label}: grad points back to boundary for {restoring[nz, axis].float().mean():.3f} of coords '
              f'(near boundary {restoring[near, axis].float().mean():.3f}, n={int(near.sum())}; '
              f'far from it {restoring[far, axis].float().mean():.3f}, n={int(far.sum())})')


def reinforce_inclusion(name, draws=64, k=8092):
    run = ROOT / 'checkpoints' / name
    model, ckpt = recover_init(run)
    init = model.sampler.mask_net.mask.detach().float().squeeze()
    final = ckpt['model']['sampler.mask_net.mask'].float().squeeze()
    h, w = init.shape
    excluded = preselect_cells(h, w).flatten()

    def inclusion(logits):
        flat = logits.flatten().masked_fill(excluded, float('-inf')).to(device)
        order, _ = plackett_luce_sample_and_logprob(flat.expand(draws, -1).contiguous(), k)
        hits = torch.zeros(draws, h * w, device=device).scatter_(1, order, 1.0)
        return hits.mean(0).view(h, w).cpu()

    a, b = inclusion(init), inclusion(final)
    rad = radius(h, w)
    print(f'\n=== {name}: Plackett-Luce inclusion density ({draws} draws)')
    print('ring          init     final   share-of-budget init/final')
    for lo, hi in zip(RINGS[:-1], RINGS[1:]):
        sel = (rad >= lo) & (rad < hi) & ~excluded.view(h, w)
        print(f'{lo:.2f}-{hi:.2f}   {a[sel].mean():.3f}   {b[sel].mean():.3f}   {a[sel].sum() / k:.3f} / {b[sel].sum() / k:.3f}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--kernel-sigmas']:
        for sigma in map(float, sys.argv[2:]):
            autosamp('autosamp2d_varnet_r8_brain_knee', kernel_sigma=sigma)
    else:
        reinforce_inclusion('reinforce2d_varnet_r8_brain_knee_vd2.5_100ep')
        reinforce_inclusion('reinforce2d_varnet_r8_brain_knee_v2')
        for name in ('autosamp2d_varnet_r8_brain_knee', 'autosamp2d_varnet_r8_brain_knee_normalinit'):
            autosamp(name)
