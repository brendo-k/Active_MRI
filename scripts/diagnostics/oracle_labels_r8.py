"""Per-slice oracle mask labels for the 1-step conv_2d sampler (R=8, brain+knee), for supervised sampler training.

Mask-backward of Huang & Ravishankar (TCI 2022) with the fixes their alternating loop lacked here: the
reconstructor (baseline conv run's VarNet + sensitivity net) is frozen, every slice starts from one shared
prior map instead of the sampler's own mask, there is no consistency term, and the loss of each step is
averaged over `--draws` Bernoulli draws. Per slice a free score map is optimised for `--steps` Adam steps
with the trainer's estimator (budget rescale, Bernoulli draw, straight-through gradient).

The prior is the logit of the native sampler's mean score map over `--prior-slices` training slices; it is
computed once and saved as prior.pt so every shard shares it. Each shard saves, per slice: the sampler's
network input (SENSE-combined k-space of the 10x10 centre), the centre and valid masks, the label logits,
and the trainer-path PSNR of the native, prior and label masks over fixed seeded draws.

    python scripts/diagnostics/oracle_labels_r8.py --split train --start 0 --count 1350
    python scripts/diagnostics/oracle_labels_r8.py --split val --count 128 --stride 5
"""
import argparse
import json
import time
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask, build_model
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
from activemri.utils.util import compute_psnr_torch

RUN = Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05')
OUT_DIR = Path('/tmp/bkadota_oracle_labels_r8')
LOSS_WEIGHTS = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
SCORING_SEEDS = (0, 1, 2, 3)
PROBABILITY_EPS = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--start', type=int, default=0, help='first position in the seeded slice order')
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--stride', type=int, default=1, help='take every stride-th slice of the order')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--draws', type=int, default=2, help='Bernoulli draws averaged per step')
    parser.add_argument('--chunk', type=int, default=4, help='slices optimised together')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--prior-slices', type=int, default=256)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_frozen(device):
    checkpoint = torch.load(RUN / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint['options'].loss_type


def slice_order(dataset, split, seed):
    """Seeded permutation of a training set (so shards are disjoint and anatomy-mixed); validation keeps its order."""
    return np.random.RandomState(seed).permutation(len(dataset)) if split == 'train' else np.arange(len(dataset))


def loader_over(loader, indices, chunk):
    return DataLoader(Subset(loader.dataset, indices.tolist()), batch_size=chunk, shuffle=False,
                      num_workers=4, collate_fn=loader.collate_fn)


def to_device(batch, device):
    labels = batch.get('anatomy_labels')
    return {'kspace': batch['kspace'].to(device), 'target': batch['target'].to(device),
            'label': labels.to(device) if torch.is_tensor(labels) else None}


@torch.no_grad()
def sampler_input(model, kspace):
    """The conv sampler's observed k-space: the 10x10 centre, SENSE-combined, back in k-space [B, H, W, 2]."""
    sense_maps, _ = model.sensetivity_net(torch.view_as_complex(kspace.contiguous()), mask_method=model.sense_mask_method)
    old_mask, valid_mask = model._initial_masks(kspace)
    observed = torch.view_as_complex((kspace * old_mask.unsqueeze(-1)).contiguous())
    combined = (ifft_2d_img(observed) * sense_maps).sum(dim=1)
    return torch.view_as_real(fft_2d_img(combined)), old_mask, valid_mask


def masks_from_logits(model, logits, kspace):
    old_mask, valid_mask = model._initial_masks(kspace)
    return model.sampler.select_locations(scores_to_prob(logits), old_mask, model.budget, valid_mask)[1]


def recon_loss(model, loss_type, part, mask):
    output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)
    target_dict = {'target': part['target'], 'kspace': part['kspace']}
    if part['label'] is not None:
        target_dict['anatomy_labels'] = part['label']
    return model.loss(output, target_dict, LOSS_WEIGHTS, loss_type)[0]


def tiled(part, draws):
    return {key: (value.repeat(draws, *([1] * (value.dim() - 1))) if value is not None else None)
            for key, value in part.items()}


@torch.no_grad()
def seeded_psnr(model, part, mask_fn, chunk_index):
    """Trainer-path PSNR per slice, averaged over SCORING_SEEDS draws (identical across mask_fns)."""
    psnr = torch.zeros(len(part['kspace']))
    for seed in SCORING_SEEDS:
        torch.manual_seed(seed * 100003 + chunk_index)
        output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask_fn())['output']
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        masked_output, masked_target = _apply_signal_mask(output, target)
        psnr += compute_psnr_torch(masked_output, masked_target).flatten().cpu() / len(SCORING_SEEDS)
    return psnr


@torch.no_grad()
def build_prior(model, loader, args):
    """Logit of the native sampler's mean score map over the first `prior_slices` of the seeded training order."""
    indices = slice_order(loader.dataset, 'train', args.seed)[:args.prior_slices]
    total, count = 0.0, 0
    for batch in loader_over(loader, indices, 16):
        part = to_device(batch, args.device)
        total = total + model.propose_mask(part['kspace'], anatomy_labels=part['label'])['unscaled'].sum(0, keepdim=True)
        count += len(part['kspace'])
    return torch.logit((total / count).clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS)).cpu()


def fit_labels(model, loss_type, part, prior, args):
    logits = prior.to(args.device).expand(len(part['kspace']), -1, -1, -1).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    many = tiled(part, args.draws)
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = recon_loss(model, loss_type, many, masks_from_logits(model, logits.repeat(args.draws, 1, 1, 1), many['kspace']))
        loss.backward()
        optimizer.step()
    return logits.detach()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, loss_type = load_frozen(args.device)
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    loader = train_loader if args.split == 'train' else val_loader
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prior_path = args.out_dir / 'prior.pt'
    if not prior_path.exists():
        torch.save(build_prior(model, train_loader, args), prior_path)
    prior = torch.load(prior_path)

    indices = slice_order(loader.dataset, args.split, args.seed)[args.start:][::args.stride][:args.count]
    rows = {key: [] for key in ('kspace_in', 'old_mask', 'valid_mask', 'label_logits', 'anatomy_labels',
                                'psnr_native', 'psnr_prior', 'psnr_label')}
    anatomies, started = [], time.time()
    for chunk_index, batch in enumerate(loader_over(loader, indices, args.chunk)):
        part = to_device(batch, args.device)
        kspace_in, old_mask, valid_mask = sampler_input(model, part['kspace'])
        logits = fit_labels(model, loss_type, part, prior, args)
        rows['psnr_native'].append(seeded_psnr(model, part, lambda: model.propose_mask(
            part['kspace'], anatomy_labels=part['label'])['mask'], chunk_index))
        rows['psnr_prior'].append(seeded_psnr(model, part, lambda: masks_from_logits(
            model, prior.to(args.device).expand_as(logits), part['kspace']), chunk_index))
        rows['psnr_label'].append(seeded_psnr(model, part, lambda: masks_from_logits(model, logits, part['kspace']), chunk_index))
        rows['kspace_in'].append(kspace_in.half().cpu())
        rows['old_mask'].append(old_mask.bool().cpu())
        rows['valid_mask'].append(valid_mask.bool().cpu())
        rows['label_logits'].append(logits.half().cpu())
        rows['anatomy_labels'].append(part['label'].cpu() if part['label'] is not None else torch.zeros(0))
        anatomies += list(batch['anatomy'])
        done = (chunk_index + 1) * args.chunk
        gain = torch.cat(rows['psnr_label']) - torch.cat(rows['psnr_prior'])
        print(f"{min(done, len(indices))}/{len(indices)} slices, {(time.time() - started) / done:.1f} s/slice, "
              f"label - prior {gain.mean():+.3f} dB, native - prior "
              f"{(torch.cat(rows['psnr_native']) - torch.cat(rows['psnr_prior'])).mean():+.3f} dB", flush=True)

    shard = {key: torch.cat(value) for key, value in rows.items()}
    shard.update({'indices': torch.as_tensor(indices), 'anatomy': anatomies, 'split': args.split,
                  'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}})
    name = f"{args.split}_{args.start}_{args.start + len(indices) * args.stride}_s{args.stride}.pt"
    torch.save(shard, args.out_dir / name)
    print(json.dumps({'saved': str(args.out_dir / name), 'slices': len(indices),
                      'label_minus_prior_db': float((shard['psnr_label'] - shard['psnr_prior']).mean()),
                      'native_minus_prior_db': float((shard['psnr_native'] - shard['psnr_prior']).mean())}), flush=True)


if __name__ == '__main__':
    main()
