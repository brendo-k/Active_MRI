"""Why does straight-through LOUPE beat 2D REINFORCE at R=8 (brain+knee VarNet)?

Two analyses of checkpoints/loupe2d_varnet_r8_brain_knee (L) and
checkpoints/reinforce2d_varnet_r8_brain_knee_v2 (R):

  static   mask-only: recovered seed-42 init logits, how far each run's logits moved
           from init per k-space ring, REINFORCE policy sharpness from Monte Carlo
           Plackett-Luce draws, overlap and radial density of the two learned masks.
  crossval each reconstructor x each mask source on the full val set, scored with the
           trainer's own masked per-anatomy metric path (reproduces the logged numbers).

    CUDA_VISIBLE_DEVICES=1 python scripts/eval/compare_loupe_reinforce_r8.py static
    CUDA_VISIBLE_DEVICES=1 python scripts/eval/compare_loupe_reinforce_r8.py crossval
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.samplers.oracle import plackett_luce_sample_and_logprob
from activemri.training.trainer import _AnatomyStats, _apply_signal_mask, build_model
from activemri.utils.mask_binarization import RescaleProbMap
from activemri.training.builders import build_dataloaders

RUNS = {
    'loupe': Path('checkpoints/loupe2d_varnet_r8_brain_knee'),
    'reinforce': Path('checkpoints/reinforce2d_varnet_r8_brain_knee_v2'),
}
RADIAL_RINGS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
MC_DRAWS = 64
OUT_DIR = Path('reports/loupe_vs_reinforce_r8')


def load(run):
    return torch.load(RUNS[run] / 'best_model.pt', map_location='cpu', weights_only=False)


def mask_logits(ckpt):
    return ckpt['model']['sampler.mask_net.mask'].float().squeeze()


def recover_init_logits(run):
    """Replays train.py's run_sequential RNG consumption up to model construction."""
    cfg = OmegaConf.load(RUNS[run] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    build_dataloaders(cfg)
    model = build_model(load(run)['options'])
    return model.sampler.mask_net.mask.detach().float().squeeze(), model


def radius_grid(h, w):
    ky = (torch.arange(h) - h // 2) / (h / 2)
    kx = (torch.arange(w) - w // 2) / (w / 2)
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def preselect_region(h, w, preselect_num):
    region = torch.zeros(h, w, dtype=torch.bool)
    n = preselect_num // 2
    region[h // 2 - n:h // 2 + n, w // 2 - n:w // 2 + n] = True
    return region


def topk_mask(logits, k, excluded):
    flat = logits.flatten().masked_fill(excluded.flatten(), float('-inf'))
    mask = torch.zeros_like(flat)
    mask[torch.topk(flat, k).indices] = 1.0
    return mask.view_as(logits)


def pl_draws(logits, k, excluded, n):
    flat = logits.flatten().masked_fill(excluded.flatten(), float('-inf'))
    order, _ = plackett_luce_sample_and_logprob(flat.expand(n, -1).contiguous(), k)
    draws = torch.zeros(n, flat.numel())
    draws.scatter_(1, order, 1.0)
    return draws.view(n, *logits.shape)


def dice(a, b):
    return float(2 * (a * b).sum() / (a.sum() + b.sum()))


def static_analysis():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = {run: load(run) for run in RUNS}
    logits = {run: mask_logits(ckpts[run]) for run in RUNS}
    inits, model = {}, None
    for run in RUNS:
        inits[run], model = recover_init_logits(run)
    h, w = logits['loupe'].shape
    opts = ckpts['loupe']['options']
    excluded = preselect_region(h, w, opts.preselect_num)
    budget = int(round(model.budget))
    radius = radius_grid(h, w)
    rings = list(zip(RADIAL_RINGS[:-1], RADIAL_RINGS[1:]))
    report = {'budget': budget, 'preselect': int(excluded.sum())}

    report['init_identical_across_runs'] = bool(torch.equal(inits['loupe'], inits['reinforce']))
    report['init_logit_range'] = [float(inits['loupe'].min()), float(inits['loupe'].max())]

    masks = {run: topk_mask(logits[run], budget, excluded) for run in RUNS}
    init_mask = topk_mask(inits['loupe'], budget, excluded)
    draws = pl_draws(logits['reinforce'], budget, excluded, MC_DRAWS)
    inclusion = draws.mean(0)

    loupe_prob = RescaleProbMap(torch.sigmoid(opts.slope * logits['loupe'])[None, None], budget / (h * w))[0, 0]

    report['dice'] = {
        'loupe_topk_vs_reinforce_topk': dice(masks['loupe'], masks['reinforce']),
        'reinforce_topk_vs_init_topk': dice(masks['reinforce'], init_mask),
        'loupe_topk_vs_init_topk': dice(masks['loupe'], init_mask),
        'reinforce_draw_vs_topk_mean': float(np.mean([dice(d, masks['reinforce']) for d in draws])),
        'reinforce_draw_vs_draw_mean': float(np.mean([dice(draws[i], draws[i + 1]) for i in range(MC_DRAWS - 1)])),
        'chance': budget / float((~excluded).sum()),
    }

    free = ~excluded
    report['policy_sharpness'] = {
        'reinforce_frac_inclusion_gt_0.9': float((inclusion[free] > 0.9).float().mean()),
        'reinforce_frac_inclusion_lt_0.1': float((inclusion[free] < 0.1).float().mean()),
        'reinforce_frac_uncommitted_0.1_0.9': float(((inclusion[free] > 0.1) & (inclusion[free] < 0.9)).float().mean()),
        'loupe_frac_prob_gt_0.9': float((loupe_prob[free] > 0.9).float().mean()),
        'loupe_frac_prob_lt_0.1': float((loupe_prob[free] < 0.1).float().mean()),
        'loupe_frac_uncommitted_0.1_0.9': float(((loupe_prob[free] > 0.1) & (loupe_prob[free] < 0.9)).float().mean()),
        'reinforce_logit_std': float(logits['reinforce'][free].std()),
        'reinforce_logit_range': [float(logits['reinforce'][free].min()), float(logits['reinforce'][free].max())],
    }

    per_ring = []
    for lo, hi in rings:
        ring = (radius >= lo) & (radius < hi) & free
        if not ring.any():
            continue
        row = {'ring': f'{lo:.2f}-{hi:.2f}', 'n': int(ring.sum())}
        for run in RUNS:
            init_r, final_r = inits[run][ring], logits[run][ring]
            row[f'{run}_density'] = float(masks[run][ring].mean())
            row[f'{run}_corr_with_init'] = float(torch.corrcoef(torch.stack([init_r, final_r]))[0, 1])
            row[f'{run}_mean_abs_drift'] = float((final_r - init_r).abs().mean())
            row[f'{run}_topk_kept_from_init_topk'] = float(
                (masks[run][ring] * init_mask[ring]).sum() / init_mask[ring].sum().clamp(min=1))
        row['reinforce_expected_density'] = float(inclusion[ring].mean())
        row['reinforce_uncommitted'] = float(((inclusion[ring] > 0.1) & (inclusion[ring] < 0.9)).float().mean())
        row['loupe_expected_density'] = float(loupe_prob[ring].clamp(max=1).mean())
        row['init_density'] = float(init_mask[ring].mean())
        per_ring.append(row)
    report['per_ring'] = per_ring

    torch.save({'logits': logits, 'inits': inits, 'masks': masks, 'inclusion': inclusion,
                'loupe_prob': loupe_prob, 'excluded': excluded}, OUT_DIR / 'static.pt')
    (OUT_DIR / 'static.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def ring_constant_logits(prob, slope, bins=32, eps=1e-4):
    """Logits whose Bernoulli draw keeps `prob`'s density in each k-space ring but is
    uniform within the ring: removes every bit of fine (within-ring) structure."""
    h, w = prob.shape
    ring = (radius_grid(h, w) * bins).long().clamp(max=bins)
    ring_mean = torch.bincount(ring.flatten(), weights=prob.flatten()) / torch.bincount(ring.flatten())
    flat = ring_mean[ring].clamp(eps, 1 - eps)
    return torch.log(flat / (1 - flat)) / slope


def set_mask_source(model, source, logits):
    """Points the sampler at a mask source without touching the reconstructor."""
    sampler = model.sampler
    with torch.no_grad():
        sampler.mask_net.mask.copy_(logits.view_as(sampler.mask_net.mask))
    model.eval()
    if source == 'bernoulli':
        sampler.mask_train_mode = 'straight_through'
    elif source == 'topk':
        sampler.mask_train_mode = 'reinforce'
        sampler.reinforce_eval = 'topk'
    elif source == 'pl_sample':
        sampler.mask_train_mode = 'reinforce'
        sampler.train()
    elif source == 'radial':
        sampler.mask_train_mode = 'straight_through'
    elif source == 'uniform':
        sampler.mask_train_mode = 'random'
    else:
        raise ValueError(source)


@torch.no_grad()
def evaluate(model, val_loader, device, seed):
    torch.manual_seed(seed)
    stats = _AnatomyStats(device)
    per_slice = []
    for batch in val_loader:
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        anatomies = list(batch['anatomy'])
        output = model(target, kspace)['output']
        if isinstance(output, list):
            output = output[-1]
        target4 = target if output.dim() == target.dim() else target.unsqueeze(1)
        masked_output, masked_target = _apply_signal_mask(output, target4)
        stats.update(anatomies, masked_output, masked_target)
        err = (masked_output - masked_target).pow(2).flatten(1).sum(1)
        ref = masked_target.pow(2).flatten(1).sum(1).clamp(min=1e-12)
        per_slice.extend(zip(anatomies, (err / ref).tolist(), batch['fname'], batch['slice_id'].tolist()))
    result = {a: {k: float(v) for k, v in m.compute().items()} for a, m in stats.metrics.items()}
    return result, per_slice


CONDITIONS = [
    # (reconstructor run, mask logits run, mask source, seed)
    ('loupe', 'loupe', 'bernoulli', 0),
    ('loupe', 'loupe', 'bernoulli', 1),
    ('loupe', 'loupe', 'topk', 0),
    ('loupe', 'reinforce', 'topk', 0),
    ('loupe', 'loupe', 'uniform', 0),
    ('reinforce', 'reinforce', 'topk', 0),
    ('reinforce', 'reinforce', 'pl_sample', 0),
    ('reinforce', 'reinforce', 'pl_sample', 1),
    ('reinforce', 'loupe', 'topk', 0),
    ('reinforce', 'loupe', 'bernoulli', 0),
    ('reinforce', 'reinforce', 'uniform', 0),
]


RADIAL_CONDITIONS = [
    # (reconstructor run, which per-ring density to keep, seed); all Bernoulli draws
    ('loupe', 'loupe_prob', 0),
    ('loupe', 'loupe_prob', 1),
    ('reinforce', 'reinforce_inclusion', 0),
    ('reinforce', 'reinforce_topk', 0),
]


def radial(args):
    device = torch.device('cuda')
    cfg = OmegaConf.load(RUNS['loupe'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = args.num_workers
    _, val_loader = build_dataloaders(cfg)
    static = torch.load(OUT_DIR / 'static.pt')
    excluded = static['excluded']
    densities = {
        'loupe_prob': static['loupe_prob'],
        'reinforce_inclusion': static['inclusion'].masked_fill(excluded, 1.0),
        'reinforce_topk': static['masks']['reinforce'].masked_fill(excluded, 1.0),
    }
    results = []
    for recon_run in RUNS:
        ckpt = load(recon_run)
        model = build_model(ckpt['options']).to(device)
        model.load_state_dict(ckpt['model'])
        slope = ckpt['options'].slope
        for recon, density, seed in RADIAL_CONDITIONS:
            if recon != recon_run:
                continue
            set_mask_source(model, 'radial', ring_constant_logits(densities[density], slope).to(device))
            metrics, _ = evaluate(model, val_loader, device, seed)
            results.append({'recon': recon, 'density': density, 'seed': seed, **{
                f'{a}_{k}': v for a, m in metrics.items() for k, v in m.items()}})
            print(f'recon={recon} radial({density}) seed={seed}', ' '.join(
                f'[{a}] PSNR={m["PSNR"]:.3f} SSIM={m["SSIM"]:.4f} NMSE={m["NMSE"]:.5f}' for a, m in metrics.items()), flush=True)
    (OUT_DIR / 'radial.json').write_text(json.dumps(results, indent=2))


def crossval(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    cfg = OmegaConf.load(RUNS['loupe'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = args.num_workers
    _, val_loader = build_dataloaders(cfg)
    ckpts = {run: load(run) for run in RUNS}
    logits = {run: mask_logits(ckpts[run]) for run in RUNS}
    results, slices = [], {}
    for recon_run in RUNS:
        model = build_model(ckpts[recon_run]['options']).to(device)
        model.load_state_dict(ckpts[recon_run]['model'])
        for recon, mask_run, source, seed in CONDITIONS:
            if recon != recon_run:
                continue
            set_mask_source(model, source, logits[mask_run].to(device))
            metrics, per_slice = evaluate(model, val_loader, device, seed)
            name = f'recon={recon} mask={mask_run}/{source} seed={seed}'
            slices[name] = per_slice
            results.append({'recon': recon, 'mask': mask_run, 'source': source, 'seed': seed, **{
                f'{a}_{k}': v for a, m in metrics.items() for k, v in m.items()}})
            print(name, ' '.join(f'[{a}] PSNR={m["PSNR"]:.3f} SSIM={m["SSIM"]:.4f} NMSE={m["NMSE"]:.5f}'
                                 for a, m in metrics.items()), flush=True)
    (OUT_DIR / 'crossval.json').write_text(json.dumps(results, indent=2))
    torch.save(slices, OUT_DIR / 'crossval_per_slice.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['static', 'crossval', 'radial'])
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    {'static': lambda: static_analysis(), 'crossval': lambda: crossval(args), 'radial': lambda: radial(args)}[args.mode]()


if __name__ == '__main__':
    main()
