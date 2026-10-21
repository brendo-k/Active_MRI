"""Does blurring an oracle label keep its sampling budget in place? Logit-space vs density-space blur.

For the labelled validation slices of oracle_labels_r8.py, the share of the sampling probability (through the
sampler's scores_to_prob, over the available locations) in each radial ring of k-space, for the label itself, the
anatomy-mean reference, and the label blurred as in oracle_blur_psnr_r8.py with `--space logit` and `--space density`.
A blur that only removes fine structure should leave every ring's share where the label had it.

    python scripts/diagnostics/blur_budget_shift_r8.py
"""
import argparse
import json
from pathlib import Path

import torch

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from label_curriculum_r8 import gaussian_blur
from label_predictor_gonogo_r8 import ANATOMIES, anatomy_reference, available_of, label_probability, load_shards, logit_of
from oracle_blur_psnr_r8 import density_blurred
from oracle_labels_r8 import OUT_DIR

OUT = Path('reports/label_gonogo_r8/blur_budget_shift.json')
RINGS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 256))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--eps', type=float, default=1e-3)
    return parser.parse_args()


def ring_shares(probability, available):
    """Mean over slices of each radial ring's share of the slice's total probability on available locations."""
    height, width = probability.shape[-2:]
    rows, cols = torch.meshgrid(torch.arange(height) - height // 2, torch.arange(width) - width // 2, indexing='ij')
    radius = (rows ** 2 + cols ** 2).float().sqrt()
    weighted = (probability * available)[:, 0]
    total = weighted.sum((1, 2))
    return [float((weighted[:, (radius >= low) & (radius < high)].sum(1) / total).mean()) for low, high in RINGS]


def main():
    args = parse_args()
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    reference = anatomy_reference(anatomy_mean, val['anatomy'])
    label, available = label_probability(val), available_of(val)
    deviation = val['label_logits'].float() - logit_of(reference)
    shares = {'label': ring_shares(label, available), 'anatomy_mean_label': ring_shares(scores_to_prob(logit_of(reference)), available)}
    for sigma in args.sigmas:
        shares[f'logit_sigma{sigma:g}'] = ring_shares(scores_to_prob(logit_of(reference) + gaussian_blur(deviation, sigma)), available)
        shares[f'density_sigma{sigma:g}'] = ring_shares(
            scores_to_prob(logit_of(density_blurred(label, reference, available, sigma, args.eps))), available)
    for name, values in shares.items():
        print(f'{name:22s}', ' '.join(f'{v:.3f}' for v in values))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'rings_px': RINGS, 'shares': shares, 'eps': args.eps}, indent=2) + '\n')


if __name__ == '__main__':
    main()
