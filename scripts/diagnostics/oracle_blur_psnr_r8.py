"""How much of the per-slice oracle gain survives blurring the label's deviation from the anatomy mean?

For each labelled validation slice of oracle_labels_r8.py, the oracle logit is
logit(anatomy mean) + G_sigma * (label logit - logit(anatomy mean)), a Gaussian low-pass in k-space location.
Each sigma is scored with the frozen baseline VarNet through the trainer's masked PSNR over seeded Bernoulli
draws shared by every arm, against the anatomy-mean label (the strongest non-adaptive control of
label_predictor_gonogo_r8.py) with volume-bootstrap 95% intervals. If a coarse sigma keeps most of the gain, a
smooth predictor (label_curriculum_r8.py) has a target worth reaching; if not, the gain is in fine structure.

    python scripts/diagnostics/oracle_blur_psnr_r8.py
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
from label_curriculum_r8 import gaussian_blur
from label_predictor_gonogo_r8 import (ANATOMIES, anatomy_reference, bootstrap_ci, label_probability, load_shards,
                                       logit_of)
from oracle_labels_r8 import OUT_DIR, RUN, load_frozen, loader_over, masks_from_logits, seeded_psnr, to_device

OUT = Path('analysis/label_gonogo/oracle_blur.json')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0, 1, 2, 4, 8, 16])
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    model, _ = load_frozen(args.device)
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    reference_logit = logit_of(anatomy_reference(anatomy_mean, val['anatomy']))
    deviation = val['label_logits'].float() - reference_logit
    arms = {'anatomy_mean_label': reference_logit,
            **{f'oracle_sigma{s:g}': reference_logit + gaussian_blur(deviation, s) for s in args.sigmas}}

    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    indices = val['indices'].numpy()
    psnr = {name: [] for name in arms}
    volumes = []
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, args.chunk)):
        part = to_device(batch, args.device)
        rows = slice(chunk_index * args.chunk, chunk_index * args.chunk + len(part['kspace']))
        for name, logits in arms.items():
            chunk_logits = logits[rows].to(args.device)
            psnr[name].append(seeded_psnr(model, part, lambda: masks_from_logits(model, chunk_logits, part['kspace']),
                                          chunk_index))
        volumes += list(batch['fname'])
        print(f"scored {len(volumes)}/{len(indices)}", flush=True)
    psnr = {name: torch.cat(values).numpy() for name, values in psnr.items()}
    volumes, anatomies = np.array(volumes), val['anatomy']

    rng = np.random.RandomState(args.seed)
    control = psnr['anatomy_mean_label']
    full_gain = float((psnr['oracle_sigma0'] - control).mean()) if 0 in args.sigmas else None
    summary = {}
    for name in arms:
        gain = psnr[name] - control
        summary[name] = {'mean_psnr': float(psnr[name].mean()),
                         'vs_anatomy_mean': bootstrap_ci(gain, volumes, rng),
                         'by_anatomy': {a: float(gain[anatomies == a].mean()) for a in ANATOMIES},
                         'fraction_of_unblurred_gain': float(gain.mean()) / full_gain if full_gain else None}
    print(json.dumps(summary, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps({'summary': summary, 'config': config}, indent=2) + '\n')


if __name__ == '__main__':
    main()
