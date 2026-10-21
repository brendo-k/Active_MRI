"""Do the brain volumes that gain from a per-volume density change want the same change? (ACS gate, follow-up)

Input: the v2 gate results (`acs_mask_gate_cgsense_r8.py --tag v2`). Each brain volume has three fitted 8 x 8
log-density grids theta (population density times exp(bicubic(theta))): `grid` fitted on F, `grid_in` on F and H,
`grid_ctrl` on other subjects' training slices. theta is used raw: its mean is not absorbed by the budget rescale,
because the population map's mean probability sits just above the rescale target, so a negative offset switches
RescaleProbMap to its inverted branch (a uniform floor). On the two top volumes the offset alone loses 0.11-0.12 dB
and centred theta keeps about two thirds of the gain.

1. Direction. For each fit, the effective change r = log(p / p_base) of the budget-rescaled sampling probability
   over the available locations, and its share of samples per k-space ring. Similarity between two volumes is the
   cosine of their r weighted by p_base (where samples actually move). Also the reliability of a volume's change:
   cos(r_grid, r_grid_in) (F fit vs all-slice fit) against cos(r_grid, r_grid_ctrl).
2. One direction. PC1 of the `grid_in` thetas over volumes (the variance share it explains, about the mean and raw).
   Leave one volume out: u = the first right-singular vector of the other volumes' raw `grid_in` thetas, signed to
   their mean (`--direction pc1`), or the other volumes' `grid_in` thetas averaged with weights max(in-sample gain,
   0) (`--direction gain`: the change the volumes that gain want; on training volumes these gains are known);
   scored on H with the gate's seeded draws:
     full_fit     the volume's own F-fit theta (a sanity check: reproduces the gate's `grid` arm)
     fit_centred  that theta minus its mean; fit_offset  only its mean (the two parts of full_fit)
     dir_oracle   theta = a_v u, a_v the projection of the volume's own F-fit theta (a perfect amplitude predictor)
     dir_static   theta = the mean amplitude of the other volumes' `grid_in` along u (a static refit along u)
     dir_<scalar> theta = a_hat u, a_hat from a leave-one-out linear fit of the others' `grid_in` amplitude on one
                  per-volume ACS scalar (mean over the volume's F and H slices of the 24 x 24 centre):
                    acs_spread  share of ACS energy outside the central 8 x 8
                    acs_coils   coil participation ratio (sum E_c)^2 / sum E_c^2
                    acs_object  share of the 24 x 24 ACS RSS image above 10% of its max
                  and, not deployable, base_psnr (the population mask's PSNR on H) as a reference.
3. Coil confound. Per volume the number of physical coils (non-zero coils of the 16 the loader pads to) and the
   scanner code of the file name (AXT2_<code>_...); held-out gain (grid - grid_ctrl) by coil class (<= 8 or more).

Reading, written before running:
  - "same direction" if PC1 explains >= 50% of the variance and the 5 highest-gain volumes' pairwise cosine is
    >= 0.5; otherwise the gain is volume-specific in shape, and stage 2 (a 24 x 24 predictor) is a no-go;
  - a scalar predictor would pass stage 2 only at >= +0.1 dB over dir_static with a volume-bootstrap interval above 0.

    python scripts/diagnostics/acs_gate_theta_check_r8.py
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
from acs_mask_gate_cgsense_r8 import ANATOMY_INDEX, OUT_DIR, RUN, Slices, logit_of, rescaled, score
from mask_gradient_anatomy_r8 import load
from subject_coadapt_ceiling_r8 import DensityFamily, volume_groups

RINGS = (0, 8, 16, 32, 64, 1000)
ACS = 24
INNER = 8
SCALARS = ('acs_spread', 'acs_coils', 'acs_object', 'base_psnr')
TOP = 5
BOOTSTRAPS = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default='v2')
    parser.add_argument('--direction', choices=('pc1', 'gain'), default='pc1')
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def weighted_cosine(a, b, w):
    a, b = a - (w * a).sum() / w.sum(), b - (w * b).sum() / w.sum()
    return float((w * a * b).sum() / torch.sqrt((w * a * a).sum() * (w * b * b).sum()).clamp(min=1e-12))


@torch.no_grad()
def acs_scalars(slices, locals_):
    """Per-volume ACS features, averaged over the volume's slices, from the 24 x 24 k-space centre only."""
    spread, coils, objects = [], [], []
    for local in locals_:
        kspace = torch.view_as_complex(slices.item(local)['kspace'][0].contiguous())  # [C, H, W], centred
        h, w = kspace.shape[-2:]
        centre = kspace[:, h // 2 - ACS // 2:h // 2 + ACS // 2, w // 2 - ACS // 2:w // 2 + ACS // 2]
        energy = centre.abs().pow(2)
        lo = ACS // 2 - INNER // 2
        spread.append(float(1 - energy[:, lo:lo + INNER, lo:lo + INNER].sum() / energy.sum()))
        per_coil = energy.sum(dim=(-2, -1))
        coils.append(float(per_coil.sum() ** 2 / per_coil.pow(2).sum().clamp(min=1e-30)))
        image = torch.fft.ifft2(torch.fft.ifftshift(centre, dim=(-2, -1))).abs().pow(2).sum(0).sqrt()
        objects.append(float((image > 0.1 * image.max()).float().mean()))
    return {'acs_spread': np.mean(spread), 'acs_coils': np.mean(coils), 'acs_object': np.mean(objects)}


@torch.no_grad()
def physical_coils(slices, locals_):
    kspace = slices.item(locals_[len(locals_) // 2])['kspace'][0]
    return int((kspace.pow(2).sum(dim=(-3, -2, -1)) > 0).sum())


def bootstrap(per_volume, rng):
    draws = [per_volume[rng.randint(len(per_volume), size=len(per_volume))].mean() for _ in range(BOOTSTRAPS)]
    return {'mean': float(per_volume.mean()), 'ci95': [float(x) for x in np.percentile(draws, [2.5, 97.5])],
            'median': float(np.median(per_volume)), 'positive': int((per_volume > 0).sum()), 'volumes': len(per_volume)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = np.random.RandomState(args.seed)
    base, _ = load(args.run, 'last_model.pt', args.device)
    cfg = OmegaConf.load(args.run / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    _, val_loader = build_dataloaders(cfg)
    a = ANATOMY_INDEX['brain']
    val_part = val_loader.dataset.datasets[a]
    val = Slices(val_part, sum(len(d) for d in val_loader.dataset.datasets[:a]), val_loader.collate_fn, args.device)
    reference_kspace = val.batch([0])['kspace']
    with torch.no_grad():
        base_logits = logit_of(base.propose_mask(reference_kspace)['unscaled'])
    base_p, available = rescaled(base, base_logits, reference_kspace)
    grid = DensityFamily('grid', base_logits, 8.0, 8)
    with torch.no_grad():
        unscaled = torch.sigmoid(base_logits)
    rescale = {'unscaled_mean_over_available': float((unscaled * available).sum() / available.sum()),
               'rescale_target': float(base.budget / available.sum())}
    keep = available[0, 0].cpu() > 0
    weight = base_p[0, 0].cpu()[keep]
    h, w = base_p.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing='ij')
    radius = torch.hypot(yy.float(), xx.float())[keep]

    results = [json.loads(p.read_text()) for p in sorted((OUT_DIR / f'brain_{args.tag}').glob('*.json'))]
    names = [r['volume'] for r in results]
    groups = volume_groups(val_part)
    theta = {arm: torch.tensor([r['grid_theta'][arm] for r in results]) for arm in ('grid', 'grid_in', 'grid_ctrl')}

    def change(t):
        p = rescaled(base, grid(t.view(1, 1, 8, 8).to(args.device)), reference_kspace)[0][0, 0].cpu()[keep]
        return torch.log((p + args.eps) / (weight + args.eps)), p

    def rings(p):
        return [float(p[(radius >= lo) & (radius < hi)].sum() / p.sum()) for lo, hi in zip(RINGS[:-1], RINGS[1:])]

    psnr = {arm: np.array([np.mean(r['psnr'][arm]) for r in results]) for arm in ('base', 'grid', 'grid_ctrl', 'grid_in')}
    held_out_gain = psnr['grid'] - psnr['grid_ctrl']
    changes = {arm: [change(t) for t in theta[arm]] for arm in theta}
    per_volume = []
    for i, name in enumerate(names):
        per_volume.append({
            'volume': name, 'held_out_gain': float(held_out_gain[i]), 'in_sample_gain': float(psnr['grid_in'][i] - psnr['base'][i]),
            'base_psnr': float(psnr['base'][i]),
            'reliability_fit_vs_in': weighted_cosine(changes['grid'][i][0], changes['grid_in'][i][0], weight),
            'fit_vs_ctrl': weighted_cosine(changes['grid'][i][0], changes['grid_ctrl'][i][0], weight),
            'ring_share_in': rings(changes['grid_in'][i][1]),
        })
    top = np.argsort(-held_out_gain)[:TOP]
    similarity = np.array([[weighted_cosine(changes['grid_in'][i][0], changes['grid_in'][j][0], weight)
                            for j in range(len(names))] for i in range(len(names))])
    upper = np.triu_indices(len(names), 1)
    top_pairs = [similarity[i, j] for k, i in enumerate(top) for j in top[k + 1:]]
    centred = theta['grid_in'] - theta['grid_in'].mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    pc1_share = float(singular[0] ** 2 / (singular ** 2).sum())
    raw_singular = torch.linalg.svdvals(theta['grid_in'])
    direction = {
        'pc1_share_of_variance_about_mean': pc1_share,
        'pc1_share_of_raw_second_moment': float(raw_singular[0] ** 2 / (raw_singular ** 2).sum()),
        'mean_pairwise_cosine_all': float(similarity[upper].mean()),
        'mean_pairwise_cosine_top': float(np.mean(top_pairs)),
        'top_volumes': [names[i] for i in top],
        'mean_reliability_fit_vs_in': float(np.mean([v['reliability_fit_vs_in'] for v in per_volume])),
        'mean_fit_vs_ctrl': float(np.mean([v['fit_vs_ctrl'] for v in per_volume])),
        'base_ring_share': rings(weight),
        'rescale': rescale,
    }
    print(json.dumps(direction, indent=1), flush=True)

    features = {name: acs_scalars(val, groups[name]) for name in names}
    for i, name in enumerate(names):
        features[name]['base_psnr'] = float(psnr['base'][i])
        features[name]['physical_coils'] = physical_coils(val, groups[name])
        features[name]['scanner'] = name.split('_')[3]
    arms = {'full_fit': [], 'fit_centred': [], 'fit_offset': [], 'dir_oracle': [], 'dir_static': [], **{f'dir_{s}': [] for s in SCALARS}}
    amplitudes = []
    for i, name in enumerate(names):
        others = [j for j in range(len(names)) if j != i]
        if args.direction == 'pc1':
            u = torch.linalg.svd(theta['grid_in'][others], full_matrices=False).Vh[0]
        else:
            gains = torch.tensor(np.clip(psnr['grid_in'][others] - psnr['base'][others], 0, None), dtype=torch.float32)
            u = (gains[:, None] * theta['grid_in'][others]).sum(0)
            u = u / u.norm()
        u = u if float(theta['grid_in'][others].mean(0) @ u) >= 0 else -u
        amplitude_in = theta['grid_in'][others] @ u
        own = float(theta['grid'][i] @ u)
        amplitudes.append({'volume': name, 'own_fit': own, 'own_in': float(theta['grid_in'][i] @ u)})
        thetas = {'dir_oracle': own, 'dir_static': float(amplitude_in.mean())}
        full_fit = theta['grid'][i]
        for s in SCALARS:
            x = np.array([features[names[j]][s] for j in others])
            slope, intercept = np.polyfit(x, amplitude_in.numpy(), 1)
            thetas[f'dir_{s}'] = float(slope * features[name][s] + intercept)
        score_set = results[i]['score']
        for arm, t in (('full_fit', full_fit), ('fit_centred', full_fit - full_fit.mean()),
                       ('fit_offset', torch.full_like(full_fit, float(full_fit.mean())))):
            arms[arm].append(np.mean(score(base, grid(t.view(1, 1, 8, 8).to(args.device)), val, score_set)))
        for arm, amplitude in thetas.items():
            logits = grid((amplitude * u).view(1, 1, 8, 8).to(args.device))
            arms[arm].append(np.mean(score(base, logits, val, score_set)))
        val.forget()
        print(f'{name}: ' + ', '.join(f'{k} {np.mean(v[-1]) - psnr["base"][i]:+.3f}' for k, v in arms.items()), flush=True)

    arms = {k: np.array(v) for k, v in arms.items()}
    in_amplitude = np.array([x['own_in'] for x in amplitudes])
    summary = {
        'direction': direction,
        'vs_base': {arm: bootstrap(v - psnr['base'], rng) for arm, v in arms.items()},
        'vs_dir_static': {arm: bootstrap(v - arms['dir_static'], rng) for arm, v in arms.items() if arm != 'dir_static'},
        'grid_vs_ctrl': bootstrap(held_out_gain, rng),
        'corr_in_amplitude_with': {s: float(np.corrcoef([features[n][s] for n in names], in_amplitude)[0, 1]) for s in SCALARS},
        'corr_in_amplitude_with_held_out_gain': float(np.corrcoef(in_amplitude, held_out_gain)[0, 1]),
        'corr_held_out_gain_with_physical_coils': float(np.corrcoef([features[n]['physical_coils'] for n in names], held_out_gain)[0, 1]),
        'held_out_gain_by_coils': {label: bootstrap(held_out_gain[np.array([test(features[n]['physical_coils']) for n in names])], rng)
                                   for label, test in (('<=8 coils', lambda c: c <= 8), ('>8 coils', lambda c: c > 8))},
    }
    out = OUT_DIR / f'theta_check_{args.direction}.json'
    out.write_text(json.dumps({'summary': summary, 'per_volume': per_volume, 'amplitudes': amplitudes,
                               'features': features, 'similarity_grid_in': similarity.tolist()}, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in summary if k != 'direction'}, indent=1))


if __name__ == '__main__':
    main()
