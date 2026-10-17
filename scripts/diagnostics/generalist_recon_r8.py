"""Fine-tune a VarNet on a broad family of variable-density R=8 masks: a "generalist" reconstructor that has seen the
per-slice mask changes an adaptive sampler would make, for training samplers against a reconstructor that does not
favour one mask (reports/ADAPTIVITY_MECHANISMS_R8.md, next steps).

Starts from a jointly trained run's reconstructor and sensitivity net (default: the LOUPE floor-0.05 run). Each
training slice gets its own mask, half of them from each family:

  reshaped   a sampler's own map (the LOUPE mask, its logits bounded to +/-6, or the conv ep50 sampler's per-slice
             map) plus a smooth Gaussian random field: sigma ~ U(0, 2.5) logits, length 8, 16 or 32 px
  parametric density (1 + (r / r0)^2)^(-s / 2) on an anisotropic radius (axis ratio 1/2 to 2), r0 log-uniform in
             [3, 40] px, s ~ U(1.5, 4), plus a uniform floor in [0, 0.02]

Every map goes through the conv sampler's budget rescale and a Bernoulli draw. Quality gate: on unperturbed LOUPE
masks the generalist should be within about 0.2 dB of the LOUPE run's own reconstructor. Use last_model.pt;
best_model.pt is the best trained step on LOUPE masks, which favours steps that drifted least from the start run. Validation (every
`--eval-every` steps, every 8th val slice, common seeded draws) scores LOUPE masks, conv masks and LOUPE masks reshaped
at sigma 1, and logs to wandb.

    python scripts/diagnostics/generalist_recon_r8.py --out /tmp/bkadota_runs/generalist_vd_varnet_r8
"""
import argparse
import copy
import json
import math
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.utils.mask_binarization import RescaleProbMap
from mask_gradient_anatomy_r8 import CONV_RUN, ScoreModel, load, radius_map, to_device
from recon_mask_tolerance_r8 import LOUPE_RUN, psnr_of, smooth_field, val_subset

LENGTHS = (8.0, 16.0, 32.0)
MAX_SIGMA = 2.5
SCORING_SEEDS = (0, 1)
VALIDATION_SIGMA = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start-run', type=Path, default=LOUPE_RUN)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--val-stride', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


class MaskFamily:
    """Per-slice masks from the reshaped and parametric families, through the conv run's budget and estimator."""

    def __init__(self, loupe_model, conv_model, device):
        self.estimator = conv_model
        self.loupe_scores, self.conv_scores = ScoreModel(loupe_model), ScoreModel(conv_model)
        self.loupe_model = loupe_model
        self.radius_grid = None
        self.device = device

    def _rescale_and_draw(self, probability, old_mask, valid_mask, generator):
        available = (1 - old_mask) * valid_mask
        sparsity = self.estimator.budget / probability[0, 0].numel()
        rescaled = RescaleProbMap(probability * available, sparsity, available)
        threshold = torch.rand(rescaled.shape, generator=generator).to(rescaled.device)
        return torch.clamp(old_mask + (rescaled > threshold).float() * valid_mask, max=1)

    def reshaped_probability(self, part, base, sigma, length, generator):
        if base == 'loupe':
            z = self.loupe_scores.scores(part).clamp(-6.0, 6.0)
            probability = torch.sigmoid
        else:
            z = self.conv_scores.scores(part)
            probability = None
        if sigma > 0:
            z = z + sigma * smooth_field(z.shape, length, generator, z.device)
        old_mask, _ = self.estimator._initial_masks(part['kspace'])
        if probability is not None:
            return probability(z) * (1 - old_mask)
        return self.conv_scores.probability(z, old_mask)

    def parametric_probability(self, count, shape, generator):
        if self.radius_grid is None:
            ky, kx = torch.meshgrid(torch.arange(shape[0]) - shape[0] // 2, torch.arange(shape[1]) - shape[1] // 2,
                                    indexing='ij')
            self.radius_grid = (ky.float().to(self.device), kx.float().to(self.device))
        ky, kx = self.radius_grid
        maps = []
        for _ in range(count):
            u = torch.rand(5, generator=generator).tolist()
            ratio = math.exp((2 * u[0] - 1) * math.log(2.0))
            r0 = math.exp(math.log(3.0) + u[1] * (math.log(40.0) - math.log(3.0)))
            power = 1.5 + 2.5 * u[2]
            floor = 0.02 * u[3]
            radius = torch.sqrt((kx * math.sqrt(ratio)) ** 2 + (ky / math.sqrt(ratio)) ** 2)
            maps.append((1 + (radius / r0) ** 2) ** (-power / 2) + floor)
        return torch.stack(maps).unsqueeze(1)

    @torch.no_grad()
    def masks(self, part, generator):
        old_mask, valid_mask = self.estimator._initial_masks(part['kspace'])
        count = len(part['kspace'])
        probability = torch.zeros_like(old_mask)
        kinds = torch.rand(count, generator=generator)
        for row in range(count):
            single = {k: (v[row:row + 1] if v is not None else None) for k, v in part.items()}
            if kinds[row] < 0.5:
                base = 'loupe' if torch.rand(1, generator=generator).item() < 0.5 else 'conv'
                sigma = MAX_SIGMA * torch.rand(1, generator=generator).item()
                length = LENGTHS[int(torch.randint(len(LENGTHS), (1,), generator=generator))]
                probability[row:row + 1] = self.reshaped_probability(single, base, sigma, length, generator)
            else:
                probability[row:row + 1] = self.parametric_probability(1, old_mask.shape[-2:], generator)
        return self._rescale_and_draw(probability, old_mask, valid_mask, generator)

    @torch.no_grad()
    def validation_mask(self, part, kind, seed):
        generator = torch.Generator().manual_seed(seed)
        old_mask, valid_mask = self.estimator._initial_masks(part['kspace'])
        if kind == 'loupe':
            probability = self.loupe_scores.probability(self.loupe_scores.scores(part), old_mask)
        elif kind == 'conv':
            probability = self.conv_scores.probability(self.conv_scores.scores(part), old_mask)
        else:
            probability = self.reshaped_probability(part, 'loupe', VALIDATION_SIGMA, 16.0, generator)
        return self._rescale_and_draw(probability, old_mask, valid_mask, generator)


@torch.no_grad()
def validate(recon, family, loader, device):
    out = {}
    for kind in ('loupe', 'conv', 'loupe_reshaped'):
        values = []
        for index, batch in enumerate(loader):
            part = to_device(batch, device)
            total = torch.zeros(len(part['kspace']))
            for seed in SCORING_SEEDS:
                total += psnr_of(recon, part, family.validation_mask(part, kind, seed * 100003 + index)) / len(SCORING_SEEDS)
            values.append(total)
        out[kind] = float(torch.cat(values).mean())
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    start, start_checkpoint = load(args.start_run, 'last_model.pt', args.device)
    conv, _ = load(CONV_RUN, 'last_model.pt', args.device)
    reference = copy.deepcopy(start)  # the start run's own reconstructor, for the quality gate
    family = MaskFamily(start, conv, args.device)

    train_loader, val_loader, indices = val_subset(args.val_stride)
    val = DataLoader(Subset(val_loader.dataset, indices), batch_size=8, shuffle=False, num_workers=4,
                     collate_fn=val_loader.collate_fn)
    train = DataLoader(train_loader.dataset, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True,
                       collate_fn=train_loader.collate_fn, generator=torch.Generator().manual_seed(args.seed))

    recon = start
    parameters = [p for n, p in recon.named_parameters() if not n.startswith('sampler.')]
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 1)

    wandb.init(project='SeqMRI', name='generalist-vd-recon_R8_from-loupe', group='adaptivity_next_steps',
               tags=['R8', 'brain', 'knee', 'varnet', 'generalist'], dir=str(args.out),
               config={k: str(v) for k, v in vars(args).items()})
    reference_scores = validate(reference, family, val, args.device)
    print('reference (start run, frozen)', json.dumps(reference_scores), flush=True)
    wandb.log({f'reference/{k}': v for k, v in reference_scores.items()}, step=0)
    del reference
    torch.cuda.empty_cache()

    history, step, best = [], 0, -np.inf
    while step < args.steps:
        for batch in train:
            if step % args.eval_every == 0:
                recon.eval()
                scores = validate(recon, family, val, args.device)
                scores['gap_to_reference_on_loupe'] = scores['loupe'] - reference_scores['loupe']
                history.append({'step': step, **scores})
                wandb.log({f'val/{k}': v for k, v in scores.items()}, step=step)
                print(json.dumps(history[-1]), flush=True)
                if step > 0 and scores['loupe'] > best:  # step 0 is the unmodified start run, not a generalist
                    best = scores['loupe']
                    torch.save({'model': recon.state_dict(), 'options': start_checkpoint['options'], 'epoch': step,
                                'generalist': vars(args) | {'out': str(args.out), 'start_run': str(args.start_run)}},
                               args.out / 'best_model.pt')
                recon.train()
            part = to_device(batch, args.device)
            mask = family.masks(part, generator)
            output = recon(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
            target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
            loss = (output - target).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if step % 50 == 0:
                wandb.log({'train/loss': loss.item()}, step=step)
            step += 1
            if step >= args.steps:
                break

    recon.eval()
    scores = validate(recon, family, val, args.device)
    scores['gap_to_reference_on_loupe'] = scores['loupe'] - reference_scores['loupe']
    history.append({'step': step, **scores})
    wandb.log({f'val/{k}': v for k, v in scores.items()}, step=step)
    torch.save({'model': recon.state_dict(), 'options': start_checkpoint['options'], 'epoch': step,
                'generalist': vars(args) | {'out': str(args.out), 'start_run': str(args.start_run)}},
               args.out / 'last_model.pt')
    (args.out / 'history.json').write_text(json.dumps({'reference': reference_scores, 'history': history},
                                                      indent=1, default=str))
    wandb.finish()
    print('done', json.dumps(history[-1]), flush=True)


if __name__ == '__main__':
    main()
