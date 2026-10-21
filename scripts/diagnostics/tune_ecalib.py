"""Tune BART ecalib (ESPIRiT) on the same slices/metrics as tune_pisco_support.py (24x24 calibration, 24x24 test-mask centre).

Stage A: kernel x null-space threshold with no crop. Stage B (--base 'k,t'): crop value and soft-SENSE. Reference arms: ACS24 without
support and the tuned PISCO preset.
"""
import argparse
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import torch

import _path  # noqa: F401
from pisco_sense_maps import ANATOMIES, vd_mask
from tune_pisco_support import clean_anatomy, net_baseline, score_slice, split_dataset
from activemri.samplers.oracle import estimate_sense_maps
from activemri.training.trainer import _signal_mask
from activemri.utils.bart_ecalib import ecalib_sense_maps
from activemri.utils.pisco import PRESETS, pisco_estimate


def evaluate(samples, make_maps, device, mask_acs, workers=1):
    kspaces = [s['k'] for s in samples]
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            produced = list(pool.map(make_maps, kspaces))
    else:
        produced = [make_maps(k) for k in kspaces]
    rows = [score_slice(s['k'], s['t'], s['an'], maps.to(device), support.to(device), s['mask'])
            for s, (maps, support) in zip(samples, produced)]
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    return {'rows': [r['psnr_clean'] for r in rows], 'clean': mean('psnr_clean'), 'ssim': mean('ssim_clean'), 'zeroed_clean': mean('zeroed_clean'), 'bloat': mean('bloat'), 'resid': mean('resid')}


def show(name, s):
    print(f"{name:34s} clean={s['clean']:.3f} ssim={s['ssim']:.4f} zeroed_clean={s['zeroed_clean']:.4f} bloat={s['bloat']:.2f} resid={s['resid']:.4f}", flush=True)


def paired(reference, other, volumes, rng=np.random.RandomState(0)):
    diff = np.array(reference['rows']) - np.array(other['rows'])
    volumes = np.array(volumes)
    per_volume = np.array([diff[volumes == v].mean() for v in sorted(set(volumes))])
    boots = [per_volume[rng.randint(0, len(per_volume), len(per_volume))].mean() for _ in range(2000)]
    return per_volume.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--volumes', type=int, default=10)
    parser.add_argument('--slices_per_volume', type=int, default=5)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--stage', default='A', choices=['A', 'B', 'C'], help='C: --base is {anatomy: [[kernel, threshold, crop, soft], ...]}; also scores the trained net and paired CIs vs the first config')
    parser.add_argument('--base', default=None, help='json {anatomy: [kernel, threshold]} (stage B) or [kernel, threshold, crop, soft] (stage C)')
    args = parser.parse_args()
    device = torch.device('cuda')
    base = __import__('json').loads(args.base) if args.base else {}

    for anatomy in ANATOMIES:
        dataset, _ = split_dataset(anatomy, args.split, args.volumes, args.slices_per_volume)
        samples = []
        for i in range(len(dataset)):
            item = dataset[i]
            target = item['target'].to(device)
            samples.append({'k': torch.view_as_complex(item['kspace']).to(device), 't': target, 'an': _signal_mask(target[None])[0],
                            'mask': vd_mask(target.shape, 8.0, 24, i, device)})
        print(f'===== {anatomy} [{args.split}] {len(samples)} slices')
        volumes = [e[0].name for e in dataset.examples]
        results = {'acs24 none': evaluate(samples, lambda k: estimate_sense_maps(k, acs=24, threshold=0.0), device, 24)}
        preset = PRESETS[anatomy]
        results['pisco preset'] = evaluate(samples, lambda k: pisco_estimate(k, preset).with_support(preset), device, 24)
        if args.stage == 'C':
            net = net_baseline(device)
            results['net_cgsense(10x10)'] = evaluate(samples, net, device, 24)
        if args.stage == 'A':
            grid = [(k, t, 0.0, False) for k, t in itertools.product((4, 5, 6, 8), (0.001, 0.005, 0.02, 0.05, 0.1))]
        elif args.stage == 'B':
            k, t = base[anatomy]
            grid = [(k, t, c, soft) for c, soft in itertools.product((0.0, 0.3, 0.5, 0.8, 0.95), (False, True))]
        else:
            grid = [tuple(c) for c in base[anatomy]]
        for k, t, c, soft in grid:
            results[f'ecalib k{k} t{t} c{c}{" S" if soft else ""}'] = evaluate(
                samples, lambda x, k=k, t=t, c=c, soft=soft: ecalib_sense_maps(x, 24, k, t, c, soft), device, 24, args.workers)
        for name, r in results.items():
            show(name, r)
        if args.stage == 'C':
            first = f'ecalib k{grid[0][0]} t{grid[0][1]} c{grid[0][2]}{" S" if grid[0][3] else ""}'
            for name, r in results.items():
                if name != first:
                    mean, lo, hi = paired(results[first], r, volumes)
                    print(f'   {first} - {name:28s} {mean:+.3f} dB [{lo:+.3f}, {hi:+.3f}]  ({len(set(volumes))} volumes)', flush=True)


if __name__ == '__main__':
    main()
