"""Score PISCO sensitivity maps as a forward model of the processed (cropped, normalised) val k-space.

For each val slice: projection residual of the fully sampled coil images onto the maps, fully sampled
SENSE vs the RSS target through the trainer's masked metric, and how much of the anatomy the map
support zeroes. Baselines: the centre-block ACS estimate and sigpy's ESPIRiT on the same block.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from activemri.datasets.fastmri_data import FastMRIDataset, _resolve_anatomy_dirs, _split_volumes, list_anatomy_files
from activemri.reconstructors.CG_Sense import conjugate_gradient_sense
from activemri.samplers.oracle import estimate_sense_maps
from activemri.training.trainer import _apply_signal_mask, _signal_mask
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.metrics import MaskedMetricAccumulator
from activemri.utils.pisco import PiscoConfig, centre_block, pisco_sense_maps

ANATOMIES = ('brain', 'knee')


def val_dataset(anatomy, data_root, num_coils, resolution, volumes, slices_per_volume):
    train_dir, val_dir = _resolve_anatomy_dirs(pathlib.Path(data_root), anatomy, 'T2')
    if train_dir == val_dir:
        _, val_files = _split_volumes(list_anatomy_files(train_dir, anatomy), 0.2, 1234)
    else:
        val_files = list_anatomy_files(val_dir, anatomy)
    names = [f.name for f in val_files[:volumes]]
    dataset = FastMRIDataset(val_dir, anatomy, resolution, num_coils=num_coils, file_names=names)
    by_volume = {}
    for example in dataset.examples:
        by_volume.setdefault(example[0], []).append(example)
    dataset.examples = [e for v in by_volume.values() for e in v[:: max(1, len(v) // slices_per_volume)][:slices_per_volume]]
    return dataset


def sigpy_maps(kspace, calib, kernel, thresh=0.02, crop=0.95):
    import sigpy.mri as mr
    live = kspace.abs().flatten(1).amax(dim=1) > 0
    maps = mr.app.EspiritCalib(kspace[live].cpu().numpy(), calib_width=calib, kernel_width=kernel, thresh=thresh, crop=crop,
                               show_pbar=False, device=-1).run()
    out = torch.zeros_like(kspace)
    out[live] = torch.from_numpy(maps).to(kspace.device, kspace.dtype)
    return out, (out.abs().pow(2).sum(0) > 0.5).float()


def vd_mask(shape, acceleration, acs, seed, device):
    """Seeded 2D variable-density random mask at `acceleration`, with a fully sampled centre block of side `acs`."""
    H, W = shape
    generator = torch.Generator(device='cpu').manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing='ij')
    radius = ((yy / (H / 2)) ** 2 + (xx / (W / 2)) ** 2).sqrt()
    density = (1 + radius * 10) ** -2
    centre = torch.zeros(H, W, dtype=torch.bool)
    centre[H // 2 - acs // 2:H // 2 + acs // 2, W // 2 - acs // 2:W // 2 + acs // 2] = True
    budget = H * W / acceleration - centre.sum()
    density = density.masked_fill(centre, 0)
    probability = (density * budget / density.sum()).clamp(max=1)
    probability = probability * budget / probability.sum()
    mask = torch.rand(H, W, generator=generator) < probability.clamp(max=1)
    return (mask | centre).float().to(device)


def cg_sense_scores(kspace, target, maps, support, mask):
    recon = conjugate_gradient_sense(kspace[None] * mask, maps[None], mask[None, None], num_iters=10)[0]
    return recon.abs() * support


def forward_model_scores(kspace, target, maps, support):
    coil_images = ifft_2d_img(kspace)
    combined = (coil_images * maps.conj()).sum(dim=0)
    projection = maps * combined
    anatomy = _signal_mask(target[None])[0]
    sense = combined.abs() * support
    return {
        'proj_resid_anatomy': float(((coil_images - projection).abs().pow(2) * anatomy).sum().sqrt() / (coil_images.abs().pow(2) * anatomy).sum().sqrt()),
        'proj_resid_all': float((coil_images - projection).abs().pow(2).sum().sqrt() / coil_images.abs().pow(2).sum().sqrt()),
        'anatomy_zeroed': float((anatomy * (1 - support)).sum() / anatomy.sum()),
        'sense_nmse': float(((sense - target) ** 2 * anatomy).sum() / (target ** 2 * anatomy).sum()),
        'sense': sense,
    }


def evaluate(dataset, estimators, device, max_slices, acceleration, mask_acs):
    rows = {name: [] for name in estimators}
    metrics = {name: MaskedMetricAccumulator(device=device) for name in estimators}
    cg_metrics = {name: MaskedMetricAccumulator(device=device) for name in estimators}
    timings = {name: 0.0 for name in estimators}
    indices = np.linspace(0, len(dataset) - 1, min(max_slices, len(dataset))).astype(int)
    for i in indices:
        sample = dataset[int(i)]
        kspace = torch.view_as_complex(sample['kspace']).to(device)
        target = sample['target'].to(device)
        anatomy = _signal_mask(target[None])
        for name, estimate in estimators.items():
            torch.cuda.synchronize(); start = time.perf_counter()
            maps, support = estimate(kspace)
            torch.cuda.synchronize(); timings[name] += time.perf_counter() - start
            scores = forward_model_scores(kspace, target, maps, support)
            masked_sense, masked_target = _apply_signal_mask(scores.pop('sense')[None], target[None])
            metrics[name].update(masked_sense, masked_target, anatomy)
            scores['cg_nmse'] = 0.0
            recon = cg_sense_scores(kspace, target, maps, support, vd_mask(target.shape, acceleration, mask_acs, int(i), device))
            masked_recon, masked_target = _apply_signal_mask(recon[None], target[None])
            cg_metrics[name].update(masked_recon, masked_target, anatomy)
            scores['cg_nmse'] = float(((masked_recon - masked_target) ** 2).sum() / (masked_target ** 2).sum())
            rows[name].append(scores)
    summary = {}
    for name in estimators:
        means = {k: float(np.mean([r[k] for r in rows[name]])) for k in rows[name][0]}
        masked = metrics[name].compute()
        cg = cg_metrics[name].compute()
        summary[name] = {**means, 'psnr': float(masked['PSNR_Masked']), 'ssim': float(masked['SSIM_Masked']),
                         'cg_psnr': float(cg['PSNR_Masked']), 'cg_ssim': float(cg['SSIM_Masked']), 'sec_per_slice': timings[name] / len(indices)}
    return summary


def pisco_estimator(config):
    def estimate(kspace):
        maps, support, _ = pisco_sense_maps(kspace, config)
        return maps, support
    return estimate


def acs_estimator(acs, threshold=0.05):
    def estimate(kspace):
        return estimate_sense_maps(kspace, acs=acs, threshold=threshold)
    return estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/bkadota/data/fastmri')
    parser.add_argument('--num_coils', type=int, default=16)
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--volumes', type=int, default=4)
    parser.add_argument('--slices_per_volume', type=int, default=4)
    parser.add_argument('--max_slices', type=int, default=24)
    parser.add_argument('--calibs', type=int, nargs='+', default=[10, 24])
    parser.add_argument('--kernels', type=int, nargs='+', default=[5, 7])
    parser.add_argument('--singular_rtols', type=float, nargs='+', default=[0.02])
    parser.add_argument('--support_thresholds', type=float, nargs='+', default=[0.95])
    parser.add_argument('--grid_extras', type=int, nargs='+', default=[24])
    parser.add_argument('--acceleration', type=float, default=8.0)
    parser.add_argument('--mask_acs', type=int, default=10, help='fully sampled centre of the R-fold test mask')
    parser.add_argument('--with_sigpy', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    device = torch.device('cuda')
    estimators = {}
    for calib in args.calibs:
        estimators[f'acs{calib}'] = acs_estimator(calib)
        estimators[f'acs{calib}_nosupport'] = acs_estimator(calib, threshold=0.0)
        for kernel in args.kernels:
            if args.with_sigpy:
                estimators[f'sigpy_c{calib}_k{kernel}'] = lambda k, c=calib, w=kernel: sigpy_maps(k, c, w)
            for rtol in args.singular_rtols:
                for support in args.support_thresholds:
                    for extra in args.grid_extras:
                        config = PiscoConfig(calib, kernel, rtol, extra, support)
                        estimators[f'pisco_c{calib}_k{kernel}_t{rtol}_s{support}_g{extra}'] = pisco_estimator(config)

    results = {}
    for anatomy in ANATOMIES:
        dataset = val_dataset(anatomy, args.data_root, args.num_coils, args.resolution, args.volumes, args.slices_per_volume)
        results[anatomy] = evaluate(dataset, estimators, device, args.max_slices, args.acceleration, args.mask_acs)
        print(f'== {anatomy} ({len(dataset)} slices available)')
        for name, s in results[anatomy].items():
            print(f"{name:42s} resid_anat={s['proj_resid_anatomy']:.4f} resid_all={s['proj_resid_all']:.4f} "
                  f"zeroed={s['anatomy_zeroed']:.4f} R{args.acceleration:g}: psnr={s['cg_psnr']:.2f} ssim={s['cg_ssim']:.4f} nmse={s['cg_nmse']:.4f} s/slice={s['sec_per_slice']:.3f}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
