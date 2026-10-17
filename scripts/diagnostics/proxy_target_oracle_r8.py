"""How much of the per-slice oracle gain survives fitting the label against a target the scanner could have?

The oracle labels of oracle_labels_r8.py are fitted against the ground-truth image. Here the same fit (same
prior init, steps, draws, lr, frozen VarNet) runs against a proxy target: the RSS of the zero-filled n x n
k-space centre, for each `--centres` n. n = 10 is exactly the calibration centre a 1-step sampler sees; larger
n trace how much target resolution the gain needs. The proxy keeps its natural scale, as a scanner's would.

Every arm is then scored against the ground truth on the same labelled validation slices, through the trainer's
masked PSNR over seeded Bernoulli draws shared by every arm, against the anatomy-mean label with
volume-bootstrap 95% intervals (as oracle_blur_psnr_r8.py and oracle_transfer_r8.py):

  anatomy_mean_label   the strongest non-adaptive control of label_predictor_gonogo_r8.py
  oracle               the ground-truth-fitted label (the stored validation shard, identical fit settings)
  proxy_centre<n>      the label fitted against the n x n centre proxy

Fitted proxy logits go to --fit-dir, never to the label directory the go/no-go scripts read.

    python scripts/diagnostics/proxy_target_oracle_r8.py --centres 10 24 64
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

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.utils.image_processing import ifft_2d_img
from label_predictor_gonogo_r8 import ANATOMIES, anatomy_reference, bootstrap_ci, label_probability, load_shards, logit_of
from oracle_labels_r8 import (OUT_DIR, RUN, load_frozen, loader_over, masks_from_logits, recon_loss, seeded_psnr,
                              tiled, to_device)

OUT = Path('analysis/label_gonogo/proxy_target_oracle.json')
FIT_DIR = Path('/tmp/bkadota_proxy_oracle')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--fit-dir', type=Path, default=FIT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--centres', type=int, nargs='+', default=[10, 24, 64])
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def centre_box(old_mask, size):
    """An size x size box on the centre of the preselected calibration region."""
    rows, cols = torch.nonzero(old_mask[0, 0], as_tuple=True)
    centre_row, centre_col = (rows.min() + rows.max() + 1) // 2, (cols.min() + cols.max() + 1) // 2
    box = torch.zeros_like(old_mask)
    box[..., centre_row - size // 2:centre_row + size // 2, centre_col - size // 2:centre_col + size // 2] = 1
    return box


@torch.no_grad()
def centre_proxy(model, kspace, size):
    """RSS of the zero-filled centre, in the target's [B, H, W] form and scale."""
    old_mask, _ = model._initial_masks(kspace)
    coil_images = ifft_2d_img(torch.view_as_complex(kspace.contiguous()) * centre_box(old_mask, size))
    return coil_images.abs().pow(2).sum(dim=1).sqrt()


def fit_against(model, loss_type, part, prior, fit_config):
    logits = prior.to(part['kspace'].device).expand(len(part['kspace']), -1, -1, -1).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=fit_config['lr'])
    many = tiled(part, fit_config['draws'])
    for _ in range(fit_config['steps']):
        optimizer.zero_grad()
        masks = masks_from_logits(model, logits.repeat(fit_config['draws'], 1, 1, 1), many['kspace'])
        recon_loss(model, loss_type, many, masks).backward()
        optimizer.step()
    return logits.detach()


def fit_proxy_labels(model, loss_type, val_loader, indices, prior, fit_config, size, args):
    path = args.fit_dir / f'centre{size}.pt'
    if path.exists():
        return torch.load(path)
    fitted, started = [], time.time()
    for batch in loader_over(val_loader, indices, args.chunk):
        part = to_device(batch, args.device)
        part['target'] = centre_proxy(model, part['kspace'], size)
        fitted.append(fit_against(model, loss_type, part, prior, fit_config).half().cpu())
        done = sum(len(f) for f in fitted)
        print(f"centre {size}: fitted {done}/{len(indices)}, {(time.time() - started) / done:.1f} s/slice", flush=True)
    fitted = torch.cat(fitted)
    args.fit_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, path)
    return fitted


@torch.no_grad()
def score(model, val_loader, indices, arms, chunk, device):
    psnr, volumes = {name: [] for name in arms}, []
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, chunk)):
        part = to_device(batch, device)
        rows = slice(chunk_index * chunk, chunk_index * chunk + len(part['kspace']))
        for name, logits in arms.items():
            chunk_logits = logits[rows].float().to(device)
            psnr[name].append(seeded_psnr(model, part, lambda: masks_from_logits(model, chunk_logits, part['kspace']),
                                          chunk_index))
        volumes += list(batch['fname'])
    return {name: torch.cat(values).numpy() for name, values in psnr.items()}, np.array(volumes)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, loss_type = load_frozen(args.device)
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    fit_config = torch.load(next(args.label_dir.glob('val_*.pt')), weights_only=False)['config']
    prior = torch.load(args.label_dir / 'prior.pt')
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}

    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    indices = val['indices'].numpy()
    arms = {'anatomy_mean_label': logit_of(anatomy_reference(anatomy_mean, val['anatomy'])),
            'oracle': val['label_logits'].float()}
    for size in args.centres:
        arms[f'proxy_centre{size}'] = fit_proxy_labels(model, loss_type, val_loader, indices, prior, fit_config, size, args)

    psnr, volumes = score(model, val_loader, indices, arms, args.chunk, args.device)
    anatomies, rng = val['anatomy'], np.random.RandomState(args.seed)
    control, oracle_gain = psnr['anatomy_mean_label'], float((psnr['oracle'] - psnr['anatomy_mean_label']).mean())
    summary = {}
    for name in arms:
        gain = psnr[name] - control
        summary[name] = {'mean_psnr': float(psnr[name].mean()),
                         'vs_anatomy_mean': bootstrap_ci(gain, volumes, rng),
                         'by_anatomy': {a: float(gain[anatomies == a].mean()) for a in ANATOMIES},
                         'fraction_of_oracle_gain': float(gain.mean()) / oracle_gain}
    print(json.dumps(summary, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps({'summary': summary, 'fit_config': fit_config, 'config': config}, indent=2) + '\n')


if __name__ == '__main__':
    main()
