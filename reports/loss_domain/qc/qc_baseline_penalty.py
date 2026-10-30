"""On an RSS-L1 LOUPE VarNet checkpoint: per-condition masked metrics, the RSS-metric cost of replacing the RSS target's
magnitude by the ESPIRiT image's (|x| + the run's own error), and the maps' common phase relative to ESPIRiT.

    python qc_baseline_penalty.py RUN_DIR --out out.json
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path('/home/bkadota/code/active_mri/SeqMRI')
sys.path.insert(0, str(REPO))

from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature  # noqa: E402
from activemri.training.builders import build_dataloaders  # noqa: E402
from activemri.training.trainer import build_model  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402
from activemri.utils.metric_mask import metric_support_mask  # noqa: E402
from activemri.utils.metrics import masked_ssim  # noqa: E402

CACHE = Path('/export/scratch/bkadota/fastmri_complex_espirit')


def masked_psnr(pred, target, mask):
    mse = ((target - pred) ** 2 * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return 10 * torch.log10(target.amax(dim=(-2, -1)) ** 2 / mse)


def circular_std(phase, weight):
    """Weighted circular std sqrt(-2 ln R) of `phase` over the pixels with weight > 0, per slice."""
    resultant = (weight * torch.exp(1j * phase)).sum(dim=(-2, -1)).abs() / weight.sum(dim=(-2, -1)).clamp(min=1e-12)
    return torch.sqrt(-2 * torch.log(resultant.clamp(min=1e-12)))


def neighbour_jump_fraction(phase, mask):
    """Share of 4-neighbour pixel pairs inside `mask` whose phase differs by more than pi/2."""
    fractions = []
    for dim in (-2, -1):
        a, b = phase.narrow(dim, 0, phase.shape[dim] - 1), phase.narrow(dim, 1, phase.shape[dim] - 1)
        ma, mb = mask.narrow(dim, 0, mask.shape[dim] - 1), mask.narrow(dim, 1, mask.shape[dim] - 1)
        pair = (ma & mb).float()
        jump = (torch.angle(torch.exp(1j * (a - b))).abs() > math.pi / 2).float()
        fractions.append(((jump * pair).sum(dim=(-2, -1)), pair.sum(dim=(-2, -1))))
    return (fractions[0][0] + fractions[1][0]) / (fractions[0][1] + fractions[1][1]).clamp(min=1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    device = torch.device('cuda')

    ckpt = torch.load(args.run / args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(ckpt['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ckpt['model'].items()})
    model.eval()
    _, val_loader = build_dataloaders(OmegaConf.load(args.run / '.hydra' / 'config.yaml'))
    cache = ComplexImageCache(CACHE)
    torch.manual_seed(args.seed)
    rows = defaultdict(list)
    for batch in val_loader:
        kspace_real = batch['kspace'].to(device)
        kspace = torch.view_as_complex(kspace_real.contiguous())
        target = batch['target'].to(device)
        anatomies = list(batch['anatomy'])
        out = model(target, kspace_real)
        output = out['output'][:, 0].float()
        x = torch.stack([cache.load(a, f, int(s), cache_signature(a, [320, 320], 20, 0.2))
                         for a, f, s in zip(anatomies, batch['fname'], batch['slice_id'])]).to(device)
        metric = metric_support_mask(kspace, anatomies)[:, 0]
        maps = out['sense_maps']
        sense = maps.abs().pow(2).sum(1) > 0.5
        full_combined = (ifft_2d_img(kspace) * maps.conj()).sum(1)
        recon_complex = out['recon_complex'][:, 0]
        pseudo = x.abs() + (output - target)
        support = metric.bool() & sense & (x.abs() > 0)
        weight = x.abs() * support
        values = {
            'condition': list(batch['condition']),
            'volume': [f'{a}/{f}' for a, f in zip(anatomies, batch['fname'])],
            'psnr_masked': masked_psnr(output, target, metric),
            'ssim_masked': masked_ssim(output[:, None], target[:, None], metric[:, None]).float(),
            'nmse_masked': ((output - target) ** 2 * metric).sum(dim=(-2, -1)) / (target ** 2 * metric).sum(dim=(-2, -1)),
            'psnr_masked_pseudo_complex': masked_psnr(pseudo, target, metric),
            'ssim_masked_pseudo_complex': masked_ssim(pseudo[:, None], target[:, None], metric[:, None]).float(),
            'frac_grid_sense': sense.float().mean(dim=(-2, -1)),
            'frac_grid_M': (x.abs() > 0).float().mean(dim=(-2, -1)),
            'xabs_l1_outside_sense_over_total': (x.abs() * ~sense).sum(dim=(-2, -1)) / x.abs().sum(dim=(-2, -1)),
            'phase_circstd_full_vs_espirit': circular_std(torch.angle(full_combined * x.conj()), weight),
            'phase_circstd_recon_vs_espirit': circular_std(torch.angle(recon_complex * x.conj()), weight),
            'phase_jump_frac_full_vs_espirit': neighbour_jump_fraction(torch.angle(full_combined * x.conj()), support),
            'phase_jump_frac_espirit_itself': neighbour_jump_fraction(torch.angle(x), support),
            'phase_jump_frac_full_learned_itself': neighbour_jump_fraction(torch.angle(full_combined), support),
        }
        for key, value in values.items():
            rows[key] += value if isinstance(value, list) else value.cpu().tolist()

    summary = {}
    for condition in sorted(set(rows['condition'])) + ['all']:
        keep = [c == condition or condition == 'all' for c in rows['condition']]
        summary[condition] = {}
        for key in rows:
            if key in ('condition', 'volume'):
                continue
            v = np.array([r for r, k in zip(rows[key], keep) if k], dtype=float)
            summary[condition][key] = {'mean': float(np.nanmean(v)), 'median': float(np.nanmedian(v))}
    args.out.write_text(json.dumps({'run': str(args.run), 'summary': summary}, indent=2))
    for condition, stats in summary.items():
        print(f'== {condition}')
        for key, value in stats.items():
            print(f"  {key:40s} mean {value['mean']:.4f}  median {value['median']:.4f}")


if __name__ == '__main__':
    main()
