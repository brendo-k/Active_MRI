"""Does a trained reconstructor use conjugate (Hermitian) symmetry, i.e. partial Fourier? The antilinearity test of
scripts/diagnostics/probe_conjugate_symmetry.py, ported to today's runs (manifest loader, multi-step samplers).

For each probed slice the run's own final mask is drawn once (draw seed 1234) and then held fixed, so the probe
measures the reconstructor, not the sampler. One measured location k outside the calibration block, whose partner -k
is not sampled, is perturbed by d = eps * y(k) in every coil, four times: +d, -d, +i d, -i d. Of the change in the
reconstruction's k-space (the coil-combined complex image's spectrum), the part linear in d and the part antilinear in d
(proportional to conj(d)) are separated. A conjugate mechanism answers at the unsampled -k antilinearly; interpolation
answers linearly. Placebo points mirror k in one axis only, so they share its geometry but have no conjugate relation.

Per run it reports, over every perturbation, the median antilinear fraction |anti|^2 / (|lin|^2 + |anti|^2) at -k and
at the placebos, and each point's response |change| relative to the response at k itself.

    CUDA_VISIBLE_DEVICES=0 python scripts/diagnostics/conjugate_symmetry_probe.py --run <run dir> [--run ...] --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'scripts' / 'eval')]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_multistep_adaptivity as ev  # noqa: E402
from activemri.utils.image_processing import fft_2d_img  # noqa: E402
from activemri.utils.multistep import step_thresholds  # noqa: E402

DRAW_SEED = 1234
PERTURBATIONS = ((1, 1), (-1, 1), (1, 1j), (-1, 1j))
POINTS = ('self', 'partner', 'placebo_rows', 'placebo_columns')


def val_data(run, args):
    ev_args = argparse.Namespace(workers=0, val_half=None, data_root=[args.data_root], chunk=8, limit=None,
                                 draw_seeds=[DRAW_SEED], device=args.device, figures=0, gradients=False)
    cfg = ev.datasets_config(run, ev_args)
    try:
        _, val_loader = ev.build_dataloaders(cfg)
    except (FileNotFoundError, OSError, ValueError):
        cfg.datasets.train_tier = 'S'
        _, val_loader = ev.build_dataloaders(cfg)
    return val_loader.dataset, val_loader.collate_fn


def probe_slice(model, part, index, args, generator, results):
    kspace, labels = part['kspace'], part['labels']
    thresholds = step_thresholds(DRAW_SEED, [index], model.num_step, model.shape, args.device)
    sense_maps, _ = model.sense_maps_for(kspace)
    masks, _ = model.sample_trajectory(kspace, labels, thresholds, sense_maps=sense_maps)
    mask = masks[-1]
    calibration, valid = model._initial_masks(kspace)
    support = model._recon_support(None, model._kspace_extent(kspace))
    h, w = mask.shape[-2:]
    rows = torch.arange(h, device=mask.device)[:, None] - h // 2
    columns = torch.arange(w, device=mask.device)[None, :] - w // 2
    radius = torch.hypot(rows.float(), columns.float())
    sampled = (mask[0, 0] > 0) & (calibration[0, 0] == 0)
    partner_sampled = torch.roll(torch.flip(mask[0, 0] > 0, (0, 1)), (1, 1), (0, 1))
    candidates = sampled & ~partner_sampled & (radius > args.min_radius) & (radius < args.max_radius) & (valid[0, 0] > 0)
    positions = candidates.nonzero()
    if len(positions) == 0:
        return
    picks = positions[torch.randperm(len(positions), generator=generator)[:args.locations].to(positions.device)]
    coil = torch.view_as_complex(kspace.contiguous())
    for iy, ix in picks.tolist():
        delta = args.eps * coil[0, :, iy, ix]
        batch = coil.repeat(len(PERTURBATIONS), 1, 1, 1)
        for i, (sign, rotation) in enumerate(PERTURBATIONS):
            batch[i, :, iy, ix] = batch[i, :, iy, ix] + sign * rotation * delta
        recon, _, _ = model._run_reconstructor(torch.view_as_real(batch).contiguous(), mask.expand(len(PERTURBATIONS), -1, -1, -1),
                                               sense_maps.expand(len(PERTURBATIONS), -1, -1, -1),
                                               labels.expand(len(PERTURBATIONS)) if labels is not None else None,
                                               None if support is None else support.expand(len(PERTURBATIONS), -1, -1, -1))
        spectrum = fft_2d_img(recon[:, 0])
        d0, d1 = (spectrum[0] - spectrum[1]) / 2, (spectrum[2] - spectrum[3]) / 2
        linear, anti = (d0 - 1j * d1) / 2, (d0 + 1j * d1) / 2
        points = {'self': (iy, ix), 'partner': ((h - iy) % h, (w - ix) % w), 'placebo_rows': ((h - iy) % h, ix),
                  'placebo_columns': (iy, (w - ix) % w)}
        reference = d0[iy, ix].abs().item()
        for name, (ty, tx) in points.items():
            lin, ant = linear[ty, tx].abs().item() ** 2, anti[ty, tx].abs().item() ** 2
            results[name]['anti_fraction'].append(ant / max(lin + ant, 1e-30))
            results[name]['response'].append(d0[ty, tx].abs().item() / max(reference, 1e-30))


def probe_run(run, args):
    model, checkpoint = ev.load_run(run, 'last_model.pt', args.device)
    options = checkpoint['options']
    dataset, collate = val_data(run, args)
    indices = list(range(0, len(dataset), max(1, len(dataset) // args.slices)))[:args.slices]
    generator = torch.Generator().manual_seed(args.seed)
    results = {name: {'anti_fraction': [], 'response': []} for name in POINTS}
    with torch.no_grad():
        for index in indices:
            part = ev.labelled(collate([dataset[index]]), options, args.device)
            probe_slice(model, part, index, args, generator, results)
    summary = {name: {'n': len(values['response']),
                      'anti_fraction_median': float(np.median(values['anti_fraction'])) if values['response'] else None,
                      'response_median': float(np.median(values['response'])) if values['response'] else None}
               for name, values in results.items()}
    print(f"{run.name}: " + '; '.join(f"{name} anti {s['anti_fraction_median']:.4f} response {s['response_median']:.4f}"
                                     for name, s in summary.items() if s['n']) + f" (n={summary['self']['n']})", flush=True)
    return {'sampler_type': options.sampler_type, 'num_step': int(options.num_step), 'slices': indices,
            'summary': summary, 'values': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, help='run directory with last_model.pt (repeatable)')
    parser.add_argument('--out', required=True, help='results .json')
    parser.add_argument('--data-root', default='hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}')
    parser.add_argument('--slices', type=int, default=24, help='dev slices probed, evenly spaced')
    parser.add_argument('--locations', type=int, default=6, help='perturbed locations per slice')
    parser.add_argument('--eps', type=float, default=0.05, help='perturbation size relative to y(k)')
    parser.add_argument('--min-radius', type=float, default=20.0)
    parser.add_argument('--max-radius', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    results = {Path(run).name: probe_run(Path(run), args) for run in args.run}
    settings = {key: value for key, value in vars(args).items() if key not in ('run', 'out')}
    Path(args.out).write_text(json.dumps({'settings': settings, 'runs': results}) + '\n')


if __name__ == '__main__':
    main()
