"""QC before loss_domain item 1: how the ESPIRiT support M (|x| > 0 of the cached complex image) overlaps the
trainer's apodized metric mask and the learned maps' support, and how far |x| is from the RSS target.

    python qc_support_mask.py --out qc.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path('/home/bkadota/code/active_mri/SeqMRI')
sys.path.insert(0, str(REPO))

from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature  # noqa: E402
from activemri.datasets.fastmri_data import build_manifest_dataloaders  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402
from activemri.utils.metric_mask import metric_support_mask  # noqa: E402
from activemri.utils.metrics import masked_ssim  # noqa: E402

CACHE = Path('/export/scratch/bkadota/fastmri_complex_espirit')
SENSE_THRESHOLD = 0.1  # SensetivityModel_mc pre_network mask


def sense_support(kspace):
    """SensetivityModel_mc's pre_network mask: RSS of the IFFT'd 10x10 centre > 0.1 of its max. [B, H, W] bool."""
    h, w = kspace.shape[-2:]
    centre = torch.zeros_like(kspace)
    centre[..., h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5] = 1
    rss = ifft_2d_img(kspace * centre).abs().pow(2).sum(1).sqrt()
    return rss > SENSE_THRESHOLD * rss.amax(dim=(-2, -1), keepdim=True)


def masked_psnr(pred, target, mask):
    mse = ((target - pred) ** 2 * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return 10 * torch.log10(target.amax(dim=(-2, -1)) ** 2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)

    train_loader, val_loader = build_manifest_dataloaders(
        manifest='configs/datasets/fastmri_local_v1/manifest.csv',
        roots={'local_batch2': '/export/scratch/bkadota/fastmri_knee/batch2',
               'hpc': '/export/scratch/bkadota/fastmri_raw/{anatomy}'},
        classes=['FLAIR', 'knee'], image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8)
    loader = val_loader if args.split == 'val' else train_loader
    cache = ComplexImageCache(CACHE)
    rows = defaultdict(list)
    for batch in loader:
        kspace = torch.view_as_complex(batch['kspace'].to(device).contiguous())
        target = batch['target'].to(device)
        anatomies = list(batch['anatomy'])
        x = torch.stack([
            cache.load(a, f, int(s), cache_signature(a, [320, 320], 20, 0.2))
            for a, f, s in zip(anatomies, batch['fname'], batch['slice_id'])
        ]).to(device)
        support = x.abs() > 0
        metric = metric_support_mask(kspace, anatomies)[:, 0].bool()
        sense = sense_support(kspace)
        magnitude = x.abs()
        inside = metric & support
        ratio = torch.where(inside, magnitude / target.clamp(min=1e-12), torch.nan)
        # Energy of the RSS target that the complex target zeroes inside the scored region.
        target_energy_metric = (target ** 2 * metric).sum(dim=(-2, -1))
        rows_b = {
            'condition': list(batch['condition']),
            'volume': [f'{a}/{f}' for a, f in zip(anatomies, batch['fname'])],
            'frac_grid_in_M': support.float().mean(dim=(-2, -1)),
            'frac_metric_outside_M': (metric & ~support).float().sum(dim=(-2, -1)) / metric.float().sum(dim=(-2, -1)),
            'frac_M_outside_metric': (support & ~metric).float().sum(dim=(-2, -1)) / support.float().sum(dim=(-2, -1)).clamp(min=1),
            'frac_sense_outside_M': (sense & ~support).float().sum(dim=(-2, -1)) / sense.float().sum(dim=(-2, -1)),
            'frac_metric_outside_sense': (metric & ~sense).float().sum(dim=(-2, -1)) / metric.float().sum(dim=(-2, -1)),
            'frac_grid_sense_outside_M': (sense & ~support).float().mean(dim=(-2, -1)),
            'target_energy_metric_outside_M': (target ** 2 * (metric & ~support)).sum(dim=(-2, -1)) / target_energy_metric,
            'target_energy_outside_M_over_metric': (target ** 2 * ~support).sum(dim=(-2, -1)) / target_energy_metric,
            'psnr_masked_absx_vs_target': masked_psnr(magnitude, target, metric.float()),
            'nmse_metric_absx_vs_target': ((magnitude - target) ** 2 * metric).sum(dim=(-2, -1)) / target_energy_metric,
            'ssim_masked_absx_vs_target': masked_ssim(magnitude[:, None], target[:, None], metric[:, None].float()).float(),
            'ratio_median_in_metric_and_M': torch.nanquantile(ratio.flatten(1), 0.5, dim=1),
            'ratio_p05_in_metric_and_M': torch.nanquantile(ratio.flatten(1), 0.05, dim=1),
            'ratio_p95_in_metric_and_M': torch.nanquantile(ratio.flatten(1), 0.95, dim=1),
            'l1_ratio_x_over_target': x.abs().sum(dim=(-2, -1)) / target.sum(dim=(-2, -1)),
        }
        for key, value in rows_b.items():
            rows[key] += value if isinstance(value, list) else value.cpu().tolist()

    summary = {}
    conditions = sorted(set(rows['condition']))
    for condition in conditions + ['all']:
        keep = [c == condition or condition == 'all' for c in rows['condition']]
        summary[condition] = {'n_slices': sum(keep), 'n_volumes': len({v for v, k in zip(rows['volume'], keep) if k})}
        for key in rows:
            if key in ('condition', 'volume'):
                continue
            values = np.array([v for v, k in zip(rows[key], keep) if k], dtype=float)
            summary[condition][key] = {'mean': float(np.nanmean(values)), 'p05': float(np.nanquantile(values, 0.05)),
                                       'median': float(np.nanmedian(values)), 'p95': float(np.nanquantile(values, 0.95))}
    args.out.write_text(json.dumps({'split': args.split, 'summary': summary}, indent=2))
    for condition, stats in summary.items():
        print(f"== {condition}  slices {stats['n_slices']}  volumes {stats['n_volumes']}")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key:40s} mean {value['mean']:.4f}  p05 {value['p05']:.4f}  median {value['median']:.4f}  p95 {value['p95']:.4f}")


if __name__ == '__main__':
    main()
