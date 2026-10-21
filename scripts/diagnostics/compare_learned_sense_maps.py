"""How do the trained models' sensitivity nets differ from PISCO / ACS maps on the val set?

Per val slice (brain, knee; 16 coils; the processed k-space the models see) for each map source: projection residual of the
fully sampled coil images, R=8 CG-SENSE PSNR with a common seeded mask, sum|S|^2 and support behaviour, and per-pixel
agreement |<S_a, S_b>| (phase-free) with the other sources, split by radius from the image centre.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from mask_gradient_anatomy_r8 import load
from pisco_sense_maps import acs_estimator, cg_sense_scores, forward_model_scores, pisco_estimator, val_dataset, vd_mask
from activemri.utils.image_processing import fft_2d_img
from activemri.training.trainer import _apply_signal_mask, _signal_mask
from activemri.utils.metrics import MaskedMetricAccumulator
from activemri.utils.pisco import PiscoConfig

MODELS = {
    'net_cgsense': Path('checkpoints/loupe2d_cgsense_r8_brain_knee_shared'),
    'net_varnet': Path('checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep'),
}
RADIUS_EDGES = (0, 24, 48, 72, 96, 128)


def with_support(estimate, support_source):
    """`estimate`'s maps restricted to `support_source`'s support."""
    def combined(kspace):
        maps, _ = estimate(kspace)
        _, support = support_source(kspace)
        return maps * support, support
    return combined


def high_frequency_fraction(maps, cutoff):
    """Share of the maps' spectral energy outside the central cutoff x cutoff block."""
    spectrum = fft_2d_img(maps).abs().pow(2)
    H, W = spectrum.shape[-2:]
    inner = spectrum[..., H // 2 - cutoff // 2:H // 2 + cutoff // 2, W // 2 - cutoff // 2:W // 2 + cutoff // 2].sum()
    return float(1 - inner / spectrum.sum())


def net_estimator(model):
    @torch.no_grad()
    def estimate(kspace):
        maps, mask = model.sensetivity_net(kspace[None], mask_method=model.sense_mask_method)
        support = (maps.abs().pow(2).sum(dim=1) > 0.5).float()[0] if mask is None else mask[0, 0]
        return maps[0], support
    return estimate


def agreement(a, b, anatomy):
    """Per-pixel |<a, b>| / (|a||b|) on the anatomy."""
    inner = (a.conj() * b).sum(0).abs()
    norm = a.abs().pow(2).sum(0).sqrt() * b.abs().pow(2).sum(0).sqrt()
    return (inner / norm.clamp(min=1e-8))[anatomy > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volumes', type=int, default=8)
    parser.add_argument('--slices_per_volume', type=int, default=8)
    parser.add_argument('--max_slices', type=int, default=64)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    device = torch.device('cuda')

    estimators = {name: net_estimator(load(path, 'last_model.pt', device)[0]) for name, path in MODELS.items()}
    estimators['pisco_c10_k5'] = pisco_estimator(PiscoConfig(10, 5, 0.01, 12, 0.0))
    estimators['acs10'] = acs_estimator(10, threshold=0.0)
    estimators['pisco_c10_k5+netsupport'] = with_support(estimators['pisco_c10_k5'], estimators['net_cgsense'])
    estimators['acs10+netsupport'] = with_support(estimators['acs10'], estimators['net_cgsense'])
    names = list(estimators)

    results = {}
    for anatomy_name in ('brain', 'knee'):
        dataset = val_dataset(anatomy_name, '/home/bkadota/data/fastmri', 16, [256, 256], args.volumes, args.slices_per_volume)
        indices = np.linspace(0, len(dataset) - 1, min(args.max_slices, len(dataset))).astype(int)
        scores = {n: [] for n in names}
        cg = {n: MaskedMetricAccumulator(device=device) for n in names}
        core = [n for n in names if '+' not in n]
        pair = {(a, b): [] for a in core for b in core if a < b}
        radial = {(a, b): [[] for _ in RADIUS_EDGES[1:]] for (a, b) in pair}
        sumsq = {n: [] for n in names}
        highfreq = {n: [] for n in names}
        yy, xx = torch.meshgrid(torch.arange(256, device=device) - 128, torch.arange(256, device=device) - 128, indexing='ij')
        radius = (yy ** 2 + xx ** 2).float().sqrt()
        for i in indices:
            sample = dataset[int(i)]
            kspace = torch.view_as_complex(sample['kspace']).to(device)
            target = sample['target'].to(device)
            anatomy = _signal_mask(target[None])[0]
            mask = vd_mask(target.shape, 8.0, 10, int(i), device)
            maps = {}
            for n, estimate in estimators.items():
                maps[n], support = estimate(kspace)
                row = forward_model_scores(kspace, target, maps[n], support)
                row.pop('sense')
                scores[n].append(row)
                recon, tgt = _apply_signal_mask(cg_sense_scores(kspace, target, maps[n], support, mask)[None], target[None])
                cg[n].update(recon, tgt, anatomy[None])
                sumsq[n].append(float(maps[n].abs().pow(2).sum(0)[anatomy > 0].mean()))
                highfreq[n].append(high_frequency_fraction(maps[n], 12))
            for (a, b) in pair:
                pair[(a, b)].append(float(agreement(maps[a], maps[b], anatomy).mean()))
                value = (maps[a].conj() * maps[b]).sum(0).abs() / (maps[a].abs().pow(2).sum(0).sqrt() * maps[b].abs().pow(2).sum(0).sqrt()).clamp(min=1e-8)
                for k, (lo, hi) in enumerate(zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:])):
                    ring = (anatomy > 0) & (radius >= lo) & (radius < hi)
                    if ring.any():
                        radial[(a, b)][k].append(float(value[ring].mean()))
        results[anatomy_name] = {
            'per_source': {n: {**{k: float(np.mean([r[k] for r in scores[n]])) for k in scores[n][0]},
                               'r8_psnr': float(cg[n].compute()['PSNR_Masked']), 'r8_ssim': float(cg[n].compute()['SSIM_Masked']),
                               'mean_sumsq_on_anatomy': float(np.mean(sumsq[n])), 'energy_beyond_12x12': float(np.mean(highfreq[n]))} for n in names},
            'agreement': {f'{a}|{b}': float(np.mean(v)) for (a, b), v in pair.items()},
            'agreement_by_radius': {f'{a}|{b}': [float(np.mean(v)) if v else None for v in rings] for (a, b), rings in radial.items()},
        }
        print(f'== {anatomy_name}')
        for n, s in results[anatomy_name]['per_source'].items():
            print(f"{n:24s} resid={s['proj_resid_anatomy']:.4f} R8 psnr={s['r8_psnr']:.2f} ssim={s['r8_ssim']:.4f} zeroed={s['anatomy_zeroed']:.4f} sum|S|^2={s['mean_sumsq_on_anatomy']:.3f} E>12x12={s['energy_beyond_12x12']:.4f}")
        for k, v in results[anatomy_name]['agreement'].items():
            print(f'  agreement {k:28s} {v:.4f}  by radius {[round(x, 3) if x else None for x in results[anatomy_name]["agreement_by_radius"][k]]}')
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
