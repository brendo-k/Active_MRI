"""Tune PISCO map estimation and support per anatomy against R=8 CG-SENSE, with a 24x24 calibration and a 24x24 test-mask centre.

Stage 1 sweeps the support rule (eigenvalue / calibration-RSS thresholds, dilation) for a fixed estimator; stage 2 sweeps the
estimator (kernel, singular-value threshold, low-res grid) with the stage-1 winner. Tune on --split train, confirm on --split val.
Baselines: ACS24 maps (with the pipeline-style 0.05 support and with none) and the trained CG-SENSE run's sensitivity net.
"""
import argparse
import itertools
import json
import pathlib

import numpy as np
import torch

import _path  # noqa: F401
from pisco_sense_maps import ANATOMIES, cg_sense_scores, vd_mask
from activemri.datasets.fastmri_data import FastMRIDataset, _resolve_anatomy_dirs, _split_volumes, list_anatomy_files
from activemri.samplers.oracle import estimate_sense_maps
from activemri.training.trainer import _apply_signal_mask, _signal_mask
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.metrics import MaskedMetricAccumulator
from activemri.utils.pisco import PiscoConfig, pisco_estimate

DATA_ROOT = '/home/bkadota/data/fastmri'
NET_RUN = pathlib.Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared')
END_FRACTION = 0.15


def split_dataset(anatomy, split, volumes, slices_per_volume):
    train_dir, val_dir = _resolve_anatomy_dirs(pathlib.Path(DATA_ROOT), anatomy, 'T2')
    if train_dir == val_dir:
        train_files, val_files = _split_volumes(list_anatomy_files(train_dir, anatomy), 0.2, 1234)
        files, root = (train_files if split == 'train' else val_files), train_dir
    else:
        root = train_dir if split == 'train' else val_dir
        files = list_anatomy_files(root, anatomy)
    rng = np.random.RandomState(0)
    chosen = sorted(rng.permutation(len(files))[:volumes])
    dataset = FastMRIDataset(root, anatomy, [256, 256], num_coils=16, file_names=[files[i].name for i in chosen])
    by_volume = {}
    for example in dataset.examples:
        by_volume.setdefault(example[0], []).append(example)
    examples, position = [], []
    for volume_examples in by_volume.values():
        keep = np.linspace(0, len(volume_examples) - 1, min(slices_per_volume, len(volume_examples))).astype(int)
        for k in keep:
            examples.append(volume_examples[k])
            position.append(k / max(1, len(volume_examples) - 1))
    dataset.examples = examples
    return dataset, np.array(position)


def net_baseline(device):
    from mask_gradient_anatomy_r8 import load
    model = load(NET_RUN, 'last_model.pt', device)[0]

    @torch.no_grad()
    def estimate(kspace):
        maps, mask = model.sensetivity_net(kspace[None], mask_method=model.sense_mask_method)
        return maps[0], mask[0, 0]
    return estimate


def clean_anatomy(target, factor=4.0, width=9):
    """Object pixels: the box-smoothed target above `factor` times its air level (10th percentile), so noise speckle in weak slices is excluded."""
    smooth = torch.nn.functional.avg_pool2d(target[None, None], width, stride=1, padding=width // 2)[0, 0]
    return (smooth > factor * torch.quantile(smooth.flatten(), 0.1)).float()


def score_slice(kspace, target, anatomy, maps, support, mask):
    coil_images = ifft_2d_img(kspace)
    combined = (coil_images * maps.conj()).sum(dim=0)
    resid = (coil_images - maps * combined).abs().pow(2)
    recon = cg_sense_scores(kspace, target, maps, support, mask)
    masked_recon, masked_target = _apply_signal_mask(recon[None], target[None])
    accumulator = MaskedMetricAccumulator(device=kspace.device)
    accumulator.update(masked_recon, masked_target, anatomy[None])
    clean = clean_anatomy(target)
    clean_accumulator = MaskedMetricAccumulator(device=kspace.device)
    clean_recon, clean_target = recon[None] * clean, target[None] * clean
    clean_accumulator.update(clean_recon, clean_target, clean[None])
    return {
        'psnr': accumulator.psnr_values[0], 'ssim': accumulator.ssim_values[0],
        'psnr_clean': clean_accumulator.psnr_values[0], 'ssim_clean': clean_accumulator.ssim_values[0],
        'zeroed_clean': float((clean * (1 - support)).sum() / clean.sum()),
        'zeroed': float((anatomy * (1 - support)).sum() / anatomy.sum()),
        'bloat': float(support.sum() / anatomy.sum()),
        'resid': float((resid * anatomy).sum().sqrt() / (coil_images.abs().pow(2) * anatomy).sum().sqrt()),
    }


def run(dataset, estimators, pisco_configs, device, mask_acs):
    """estimators: name -> kspace -> (maps, support); pisco_configs: name -> PiscoConfig (estimates shared across equal estimator keys)."""
    estimator_key = lambda c: (c.calib, c.kernel, c.singular_rtol, c.grid_extra)
    rows = {name: [] for name in [*estimators, *pisco_configs]}
    for i in range(len(dataset)):
        sample = dataset[i]
        kspace = torch.view_as_complex(sample['kspace']).to(device)
        target = sample['target'].to(device)
        anatomy = _signal_mask(target[None])[0]
        mask = vd_mask(target.shape, 8.0, mask_acs, i, device)
        for name, estimate in estimators.items():
            maps, support = estimate(kspace)
            rows[name].append(score_slice(kspace, target, anatomy, maps, support, mask))
        cache = {}
        for name, config in pisco_configs.items():
            key = estimator_key(config)
            if key not in cache:
                cache[key] = pisco_estimate(kspace, config)
            maps, support = cache[key].with_support(config)
            rows[name].append(score_slice(kspace, target, anatomy, maps, support, mask))
    return rows


def summarise(rows, position):
    end = (position < END_FRACTION) | (position > 1 - END_FRACTION)
    out = {}
    for name, slice_rows in rows.items():
        column = lambda k: np.array([r[k] for r in slice_rows])
        out[name] = {'psnr': column('psnr').mean(), 'psnr_end': column('psnr')[end].mean(), 'psnr_mid': column('psnr')[~end].mean(),
                     'clean': column('psnr_clean').mean(), 'clean_end': column('psnr_clean')[end].mean(), 'clean_mid': column('psnr_clean')[~end].mean(),
                     'clean_ssim': column('ssim_clean').mean(), 'zeroed_clean': column('zeroed_clean').mean(), 'zeroed_clean_end': column('zeroed_clean')[end].mean(),
                     'ssim': column('ssim').mean(), 'zeroed': column('zeroed').mean(), 'zeroed_end': column('zeroed')[end].mean(),
                     'bloat': column('bloat').mean(), 'resid': column('resid').mean()}
    return out


def show(summary, top=None, title='', key='clean'):
    print(f'-- {title} (sorted by {key})')
    ordered = sorted(summary.items(), key=lambda kv: -kv[1][key])
    for name, s in ordered[:top] if top else ordered:
        print(f"{name:34s} clean={s['clean']:.3f} (end {s['clean_end']:.2f} mid {s['clean_mid']:.2f}) ssim={s['clean_ssim']:.4f} zeroed_clean={s['zeroed_clean']:.4f} (end {s['zeroed_clean_end']:.4f}) "
              f"| trainer psnr={s['psnr']:.3f} (end {s['psnr_end']:.2f}) zeroed={s['zeroed']:.4f} | bloat={s['bloat']:.2f} resid={s['resid']:.4f}")


def support_sweep(base):
    """One criterion at a time (eigenvalue, peak-relative RSS, air-relative RSS), each with and without dilation, plus no support."""
    families = [('eig', (0.5, 0.8, 0.9, 0.95)), ('rss', (0.005, 0.01, 0.03, 0.1, 0.2, 0.3)), ('noise', (1.5, 2, 3, 5, 8, 12, 20))]
    configs = {'none': PiscoConfig(base.calib, base.kernel, base.singular_rtol, base.grid_extra, 0.0)}
    for (family, values), dilate in itertools.product(families, (0, 3)):
        for value in values:
            support = {'support_threshold': 0.0, 'support_rss': 0.0, 'support_noise': 0.0, 'support_dilate': dilate}
            support[{'eig': 'support_threshold', 'rss': 'support_rss', 'noise': 'support_noise'}[family]] = value
            configs[f'{family}{value}_dil{dilate}'] = PiscoConfig(base.calib, base.kernel, base.singular_rtol, base.grid_extra, **support)
    return configs


def estimator_sweep(base, support):
    configs = {}
    for kernel, rtol, extra in itertools.product((3, 5, 7), (0.003, 0.01, 0.03), (12, 24, 48)):
        configs[f'k{kernel}_t{rtol}_g{extra}'] = PiscoConfig(base.calib, kernel, rtol, extra, **support)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--volumes', type=int, default=10)
    parser.add_argument('--slices_per_volume', type=int, default=10)
    parser.add_argument('--calib', type=int, default=24)
    parser.add_argument('--stage', type=int, default=1, help='1 support sweep, 2 estimator sweep (--chosen support), 3 confirm (--chosen full config)')
    parser.add_argument('--chosen', default=None, help='json {anatomy: {support_threshold, support_rss, support_noise, support_dilate}} (stage 2) or {anatomy: [kernel, rtol, grid, {support}]} (confirmation)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    device = torch.device('cuda')

    base_estimators = {'net_cgsense(10x10)': net_baseline(device),
                       f'acs{args.calib}_support0.05': lambda k: estimate_sense_maps(k, acs=args.calib, threshold=0.05),
                       f'acs{args.calib}_nosupport': lambda k: estimate_sense_maps(k, acs=args.calib, threshold=0.0)}
    results = {}
    for anatomy in ANATOMIES:
        dataset, position = split_dataset(anatomy, args.split, args.volumes, args.slices_per_volume)
        base = PiscoConfig(calib=args.calib, kernel=5, singular_rtol=0.01, grid_extra=24)
        chosen = json.loads(args.chosen)[anatomy] if args.chosen else None
        if args.stage == 1:
            configs = support_sweep(base)
        elif args.stage == 2:
            configs = estimator_sweep(base, chosen)
        else:
            kernel, rtol, extra, support = chosen
            configs = {'chosen': PiscoConfig(args.calib, int(kernel), rtol, int(extra), **support),
                       'default_eig0.9': PiscoConfig(args.calib, 5, 0.01, 24, 0.9)}
        rows = run(dataset, base_estimators, configs, device, mask_acs=args.calib)
        summary = summarise(rows, position)
        print(f'===== {anatomy} [{args.split}] {len(dataset)} slices, {len({e[0] for e in dataset.examples})} volumes')
        show({k: v for k, v in summary.items() if k in base_estimators}, title='baselines')
        show({k: v for k, v in summary.items() if k not in base_estimators}, top=12, title='best PISCO configs')
        results[anatomy] = {'summary': summary, 'rows': rows, 'position': position.tolist(), 'volumes': [e[0].name for e in dataset.examples]}
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, default=float))


if __name__ == '__main__':
    main()
