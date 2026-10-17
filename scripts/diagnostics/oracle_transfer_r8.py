"""Is the per-slice oracle gain of oracle_labels_r8.py a property of the slice or of the frozen VarNet it was fitted through?

The labelled validation slices are scored with other reconstructors, each with its own frozen checkpoint and the
same masks, through the trainer's masked PSNR over seeded Bernoulli draws shared by every arm:

  anatomy_mean_label   the strongest non-adaptive control of label_predictor_gonogo_r8.py
  oracle               the slice's own label
  other_slice_oracle   the label of a same-anatomy slice from another volume: a sharp, label-like mask
                       that is not this slice's, separating slice-specific gain from label-mask statistics

Gains are against anatomy_mean_label with volume-bootstrap 95% intervals. The labels' own reconstructor
(the first checkpoint) should reproduce the oracle's +0.9 dB.

    python scripts/diagnostics/oracle_transfer_r8.py
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model
from label_predictor_gonogo_r8 import ANATOMIES, anatomy_reference, bootstrap_ci, label_probability, load_shards, logit_of
from oracle_labels_r8 import OUT_DIR, RUN, loader_over, masks_from_logits, seeded_psnr, to_device

OUT = Path('analysis/label_gonogo/oracle_transfer.json')
RECONSTRUCTORS = (
    RUN,
    Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_draws2'),
    Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput'),
    Path('checkpoints/random_mask_varnet_r8_brain_knee'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--reconstructors', type=Path, nargs='+', default=list(RECONSTRUCTORS))
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_reconstructor(run, device):
    checkpoint = torch.load(run / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval().requires_grad_(False)


def other_slice_rows(anatomies, volumes, rng):
    """For each row, a random row of the same anatomy from a different volume."""
    return np.array([rng.choice(np.flatnonzero((anatomies == anatomy) & (volumes != volume)))
                     for anatomy, volume in zip(anatomies, volumes)])


@torch.no_grad()
def score(model, val_loader, indices, arms, chunk):
    psnr = {name: [] for name in arms}
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, chunk)):
        part = to_device(batch, next(model.parameters()).device)
        rows = slice(chunk_index * chunk, chunk_index * chunk + len(part['kspace']))
        for name, logits in arms.items():
            chunk_logits = logits[rows].to(part['kspace'].device)
            psnr[name].append(seeded_psnr(model, part, lambda: masks_from_logits(model, chunk_logits, part['kspace']),
                                          chunk_index))
    return {name: torch.cat(values).numpy() for name, values in psnr.items()}


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}

    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    indices = val['indices'].numpy()
    volumes = np.array([fname for batch in loader_over(val_loader, indices, args.chunk) for fname in batch['fname']])
    anatomies = val['anatomy']
    label_logits = val['label_logits'].float()
    arms = {'anatomy_mean_label': logit_of(anatomy_reference(anatomy_mean, anatomies)),
            'oracle': label_logits,
            'other_slice_oracle': label_logits[other_slice_rows(anatomies, volumes, rng)]}

    results = {}
    for run in args.reconstructors:
        model = load_reconstructor(run, args.device)
        psnr = score(model, val_loader, indices, arms, args.chunk)
        del model
        torch.cuda.empty_cache()
        control = psnr['anatomy_mean_label']
        results[run.name] = {name: {'mean_psnr': float(values.mean()),
                                    'vs_anatomy_mean': bootstrap_ci(values - control, volumes, rng),
                                    'by_anatomy': {a: float((values - control)[anatomies == a].mean()) for a in ANATOMIES}}
                             for name, values in psnr.items()}
        print(run.name, json.dumps({name: round(r['vs_anatomy_mean']['mean'], 3) for name, r in results[run.name].items()}),
              f"control psnr {results[run.name]['anatomy_mean_label']['mean_psnr']:.2f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'results': results, 'config': vars(args)}, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
