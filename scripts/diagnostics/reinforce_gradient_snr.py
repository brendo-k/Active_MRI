"""How much signal does one REINFORCE update carry per mask logit?

Replays the training-time policy gradient of a trained 2D REINFORCE checkpoint
(PL draws, reward = -NMSE, per-group normalized advantages, `group` draws per slice)
on fixed validation slices, keeping every draw's own score-function gradient so its
mean (signal) and spread (noise) can be compared per k-space position and per ring.

Per-step SNR = |mean| / std * sqrt(draws per training step). Under Adam, a logit whose
per-step SNR is s drifts ~lr*s per step but diffuses ~lr*sqrt(steps), so signal only
wins after ~1/s^2 steps.

    CUDA_VISIBLE_DEVICES=1 python scripts/diagnostics/reinforce_gradient_snr.py
"""
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
import activemri.samplers.sampler as sampler_module
from activemri.training.trainer import build_model
from compare_loupe_reinforce_r8 import radius_grid
from activemri.training.builders import build_dataloaders

RINGS = [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .7), (.7, 1), (1, 2)]


def cached_val_slices(cfg, num_slices, cache):
    if cache.exists():
        return torch.load(cache)
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset
    picks = np.linspace(0, len(dataset) - 1, num_slices).astype(int)
    slices = [(dataset[i]['kspace'], dataset[i]['target']) for i in picks]
    torch.save(slices, cache)
    return slices


class ScoreRecorder:
    """Wraps the sampler's PL draw so each row's logits are a separate grad leaf."""

    def __init__(self):
        self.original = sampler_module.plackett_luce_sample_and_logprob
        self.leaf = self.logp = None

    def __call__(self, logits, k):
        with torch.enable_grad():  # the reconstructor forward runs under no_grad
            self.leaf = logits.detach().clone().requires_grad_(True)
            order, self.logp = self.original(self.leaf, k)
        return order, self.logp.detach()

    def per_draw_score(self):
        return torch.autograd.grad(self.logp.sum(), self.leaf)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('checkpoints/reinforce2d_varnet_r8_brain_knee_v2_100ep'))
    parser.add_argument('--num-slices', type=int, default=24)
    parser.add_argument('--draws', type=int, default=16, help='draws per slice (forward batch)')
    parser.add_argument('--group', type=int, default=4, help='draws sharing a baseline, as in training')
    parser.add_argument('--step-draws', type=int, default=16, help='draws averaged per training step')
    parser.add_argument('--repeats', type=int, default=4, help='forward batches per slice')
    parser.add_argument('--null-permutations', type=int, default=8)
    parser.add_argument('--out', type=Path, default=Path('reports/loupe_vs_reinforce_r8/gradient_snr.json'))
    args = parser.parse_args()
    device = torch.device('cuda')

    ckpt = torch.load(args.run / 'best_model.pt', map_location='cpu', weights_only=False)
    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 8
    slices = cached_val_slices(cfg, args.num_slices, args.out.parent / f'snr_val_slices_{args.num_slices}.pt')

    model = build_model(ckpt['options']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.sampler.train()  # policy draws, deterministic reconstructor
    recorder = ScoreRecorder()
    sampler_module.plackett_luce_sample_and_logprob = recorder

    advantages, scores = [], []  # per draw: normalized advantage, score function [N]
    for kspace, target in slices:
        kspace = kspace.unsqueeze(0).expand(args.draws, *kspace.shape).contiguous().to(device)
        target = target.unsqueeze(0).expand(args.draws, *target.shape).contiguous().to(device)
        for _ in range(args.repeats):
            with torch.no_grad():
                output = model(target, kspace)['output']
            output = output[-1] if isinstance(output, list) else output
            target4 = target if output.dim() == target.dim() else target.unsqueeze(1)
            nmse = (output - target4).pow(2).flatten(1).sum(1) / target4.pow(2).flatten(1).sum(1).clamp(min=1e-12)
            reward = -nmse.view(-1, args.group)
            advantage = reward - reward.mean(1, keepdim=True)
            advantages.append((advantage / (advantage.std(1, keepdim=True) + 1e-8)).flatten().cpu())
            scores.append(recorder.per_draw_score().cpu())
    advantage, score = torch.cat(advantages), torch.cat(scores)
    n = advantage.shape[0]

    h, w = model.sampler.shape
    radius = radius_grid(h, w).flatten()
    active = score.std(0) > 1e-6  # drops preselected / out-of-bounds positions
    rings = [(lo, hi, (radius >= lo) & (radius < hi) & active) for lo, hi in RINGS]
    kx = torch.arange(w).repeat(h)
    # "raise the left half, lower the right": the partial-Fourier direction along kx
    directions = {'kx_left_minus_right': torch.where(kx < w // 2, 1.0, -1.0) * active}

    def directional_z(g, direction):
        projected = g @ direction
        return float(projected.mean() / (projected.std() / np.sqrt(n)))

    def z_scores(adv):
        g = adv[:, None] * score
        pixel_z = g.mean(0) / (g.std(0).clamp(min=1e-12) / np.sqrt(n))
        ring_z = []
        for _, _, ring in rings:
            ring_g = g[:, ring].sum(1)  # gradient along "raise this whole ring"
            ring_z.append(float(ring_g.mean() / (ring_g.std() / np.sqrt(n))))
        direction_z = {name: directional_z(g, d) for name, d in directions.items()}
        return pixel_z, ring_z, direction_z

    def permuted_within_groups(adv):
        grouped = adv.view(-1, args.group)
        perm = torch.rand_like(grouped).argsort(1)
        return grouped.gather(1, perm).flatten()

    pixel_z, ring_z, direction_z = z_scores(advantage)
    nulls = [z_scores(permuted_within_groups(advantage)) for _ in range(args.null_permutations)]
    # z^2 = 1 + n*snr^2 on average, so the excess of z^2 over the permutation null
    # estimates the true per-draw SNR even when no single position is significant.
    to_step = np.sqrt(args.step_draws)
    report = {'run': str(args.run), 'draws_total': n, 'step_draws': args.step_draws, 'rings': []}
    for i, (lo, hi, ring) in enumerate(rings):
        null_pixel_z2 = float(np.mean([null[0][ring].pow(2).mean() for null in nulls]))
        excess = max(float(pixel_z[ring].pow(2).mean()) - null_pixel_z2, 0.0)
        pixel_snr_step = np.sqrt(excess / n) * to_step
        null_ring_z_rms = float(np.sqrt(np.mean([null[1][i] ** 2 for null in nulls])))
        ring_snr_step = abs(ring_z[i]) / np.sqrt(n) * to_step
        report['rings'].append({
            'ring': f'{lo}-{hi}', 'n_positions': int(ring.sum()),
            'pixel_frac_abs_z_gt_3': float((pixel_z[ring].abs() > 3).float().mean()),
            'pixel_frac_abs_z_gt_3_null': float(np.mean([(null[0][ring].abs() > 3).float().mean() for null in nulls])),
            'pixel_snr_per_step_est': pixel_snr_step,
            'pixel_steps_for_drift_to_beat_diffusion': (1 / pixel_snr_step ** 2) if pixel_snr_step > 0 else float('inf'),
            'ring_z': ring_z[i], 'ring_z_null_rms': null_ring_z_rms,
            'ring_snr_per_step': ring_snr_step,
            'ring_steps_for_drift_to_beat_diffusion': 1 / ring_snr_step ** 2,
        })
    report['directions'] = {}
    for name, z in direction_z.items():
        snr_step = abs(z) / np.sqrt(n) * to_step
        report['directions'][name] = {
            'z': z, 'z_null_rms': float(np.sqrt(np.mean([null[2][name] ** 2 for null in nulls]))),
            'snr_per_step': snr_step, 'steps_for_drift_to_beat_diffusion': 1 / snr_step ** 2,
        }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
