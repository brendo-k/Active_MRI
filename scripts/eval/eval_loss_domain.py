"""Score the loss-domain runs (reports/loss_domain/proposal.md) on the local v1 val set and apply the frozen bars.

Every run is scored from `last_model.pt` on the same val slices with the same two mask-draw seeds:
- per-slice PSNR_Masked, SSIM_Masked and NMSE under the trainer's apodized metric mask;
- the maps' common phase against the cached ESPIRiT image: the circular std of angle(sum_c conj(S_c) coil_c * conj(x)),
  weighted by |x| on metric mask ∩ map support ∩ M, and the share of neighbouring pairs that jump by more than pi/2;
- the learned LOUPE probability map, sigmoid(slope * w).
Per-run results are cached in OUT/runs/, so runs that finish later can be added by running the script again.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_loss_domain.py --runs-root ~/data/seqmri_checkpoints/loss_domain/runs \
        --out reports/loss_domain/eval
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature  # noqa: E402
from activemri.datasets.fastmri_data import build_manifest_dataloaders  # noqa: E402
from activemri.training.trainer import build_model  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402
from activemri.utils.metric_mask import metric_support_mask  # noqa: E402
from activemri.utils.metrics import masked_ssim  # noqa: E402

ARMS = ('rss', 'complex', 'kspace_sense', 'kspace_rss', 'kspace_rss_M')
TWIN = 'rss'
SEEDS = (42, 43, 44)
MASK_DRAW_SEEDS = (1234, 4321)
METRICS = ('psnr', 'ssim', 'nmse')
LOSS_BAR_DB = 0.10
MASK_BAR_DB = 0.10
SSIM_GATE = 0.01
CONTROL_SSIM = 0.839  # patient_specific_loupe_r8_flair_knee_100ep_seed42, the 10-27 RSS-L1 control (reports/loss_domain/qc)
BOOTSTRAP = 2000
ROOTS = {'local_batch2': '/export/scratch/bkadota/fastmri_knee/batch2', 'hpc': '/export/scratch/bkadota/fastmri_raw/{anatomy}'}
CACHE = '/export/scratch/bkadota/fastmri_complex_espirit'


def run_name(arm, seed, epochs=100):
    return f'loss_domain_{arm}_r8_flair_knee_{epochs}ep_seed{seed}'


def masked_psnr(output, target, mask):
    mse = ((target - output) ** 2 * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return 10 * torch.log10(target.amax(dim=(-2, -1)) ** 2 / mse)


def circular_std(phase, weight):
    resultant = (weight * torch.exp(1j * phase)).sum(dim=(-2, -1)).abs() / weight.sum(dim=(-2, -1)).clamp(min=1e-12)
    return torch.sqrt(-2 * torch.log(resultant.clamp(min=1e-12)))


def neighbour_jump_fraction(phase, mask):
    jumps, pairs = 0, 0
    for dim in (-2, -1):
        n = phase.shape[dim] - 1
        pair = (mask.narrow(dim, 0, n) & mask.narrow(dim, 1, n)).float()
        jump = (torch.angle(torch.exp(1j * (phase.narrow(dim, 0, n) - phase.narrow(dim, 1, n)))).abs() > math.pi / 2).float()
        jumps = jumps + (jump * pair).sum(dim=(-2, -1))
        pairs = pairs + pair.sum(dim=(-2, -1))
    return jumps / pairs.clamp(min=1)


@torch.no_grad()
def score_run(path, val_loader, cache, device):
    checkpoint = torch.load(path / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options']).to(device)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in checkpoint['model'].items()})
    model.eval()
    model.sampler.eval_generator = None
    mask_net = model.sampler.mask_net
    probability = torch.sigmoid(mask_net.slope * mask_net.mask.detach())[0, ..., 0].cpu()
    result = {'epoch': checkpoint['epoch'], 'probability': probability, 'volume': [], 'condition': [],
              'phase_circstd': [], 'phase_jump': [], **{m: [[] for _ in MASK_DRAW_SEEDS] for m in METRICS}}
    for draw, draw_seed in enumerate(MASK_DRAW_SEEDS):
        torch.manual_seed(draw_seed)
        for batch in val_loader:
            kspace_real = batch['kspace'].to(device)
            kspace = torch.view_as_complex(kspace_real.contiguous())
            target = batch['target'].to(device)
            anatomies = list(batch['anatomy'])
            pred = model(target, kspace_real)
            output = pred['output'][:, 0].float()
            metric = metric_support_mask(kspace, anatomies)[:, 0]
            result['psnr'][draw] += masked_psnr(output, target, metric).cpu().tolist()
            result['ssim'][draw] += masked_ssim(output[:, None], target[:, None], metric[:, None]).cpu().tolist()
            nmse = ((output - target) ** 2 * metric).sum(dim=(-2, -1)) / (target ** 2 * metric).sum(dim=(-2, -1))
            result['nmse'][draw] += nmse.cpu().tolist()
            if draw:
                continue
            result['volume'] += [f'{a}/{f}' for a, f in zip(anatomies, batch['fname'])]
            result['condition'] += list(batch['condition'])
            espirit = torch.stack([cache.load(a, f, int(s), cache_signature(a, [320, 320], 20, 0.2))
                                   for a, f, s in zip(anatomies, batch['fname'], batch['slice_id'])]).to(device)
            maps = pred['sense_maps']
            combined = (ifft_2d_img(kspace) * maps.conj()).sum(dim=1)
            support = metric.bool() & (maps.abs().pow(2).sum(dim=1) > 0.5) & (espirit.abs() > 0)
            relative = torch.angle(combined * espirit.conj())
            result['phase_circstd'] += circular_std(relative, espirit.abs() * support).cpu().tolist()
            result['phase_jump'] += neighbour_jump_fraction(relative, support).cpu().tolist()
    for key in METRICS:
        result[key] = torch.tensor(result[key], dtype=torch.float64).mean(dim=0)
    for key in ('phase_circstd', 'phase_jump'):
        result[key] = torch.tensor(result[key], dtype=torch.float64)
    return result


def paired(delta, volumes, rng):
    """Mean over slices and a 95% bootstrap CI over volumes."""
    names = sorted(set(volumes))
    index = np.array([names.index(v) for v in volumes])
    per_volume = np.array([delta[index == i].mean() for i in range(len(names))])
    weights = np.bincount(index)
    draws = rng.integers(0, len(names), size=(BOOTSTRAP, len(names)))
    boot = (per_volume[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1)
    return {'mean': float(delta.mean()), 'ci': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            'n_volumes': len(names), 'frac_volumes_positive': float((per_volume > 0).mean())}


def by_condition(result, values):
    conditions = np.array(result['condition'])
    out = {'all': float(values.mean())}
    for condition in sorted(set(result['condition'])):
        out[condition] = float(values[conditions == condition].mean())
    return out


def compare(a, b, rng):
    if a['volume'] != b['volume']:
        raise ValueError('runs were scored on different slices')
    conditions = np.array(a['condition'])
    out = {}
    for metric in METRICS:
        delta = (a[metric] - b[metric]).numpy()
        out[metric] = {'all': paired(delta, a['volume'], rng)}
        for condition in sorted(set(a['condition'])):
            keep = conditions == condition
            out[metric][condition] = paired(delta[keep], [v for v, k in zip(a['volume'], keep) if k], rng)
    return out


def correlation(a, b):
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def summarise(results, rng):
    """`results` maps (arm, seed) to a scored run."""
    results = {run_name(arm, seed): result for (arm, seed), result in results.items()}
    runs = {name: {'epoch': r['epoch'], **{m: by_condition(r, r[m].numpy()) for m in METRICS},
                   'phase_circstd': by_condition(r, r['phase_circstd'].numpy()),
                   'phase_jump': by_condition(r, r['phase_jump'].numpy())}
            for name, r in results.items()}
    pairs, loss_bar, health = {}, {}, {}
    for arm in ARMS[1:]:
        per_seed = {}
        for seed in SEEDS:
            a, b = results.get(run_name(arm, seed)), results.get(run_name(TWIN, seed))
            if a is None or b is None:
                continue
            pairs[f'{arm} - {TWIN} seed{seed}'] = compare(a, b, rng)
            per_seed[seed] = pairs[f'{arm} - {TWIN} seed{seed}']['psnr']['all']['mean']
            ssim_gap = {c: runs[run_name(arm, seed)]['ssim'][c] - runs[run_name(TWIN, seed)]['ssim'][c]
                        for c in runs[run_name(arm, seed)]['ssim'] if c != 'all'}
            health[run_name(arm, seed)] = {'ssim_minus_twin': ssim_gap, 'pass': all(g >= -SSIM_GATE for g in ssim_gap.values())}
        if per_seed:
            mean = float(np.mean(list(per_seed.values())))
            loss_bar[arm] = {'per_seed': per_seed, 'mean': mean, 'complete': len(per_seed) == len(SEEDS),
                             'pass': len(per_seed) == len(SEEDS) and mean >= LOSS_BAR_DB and all(d > 0 for d in per_seed.values())}
    mask_seeds = {}
    for seed in SEEDS:
        a, b = results.get(run_name('kspace_rss_M', seed)), results.get(run_name('kspace_rss', seed))
        if a is not None and b is not None:
            pairs[f'kspace_rss_M - kspace_rss seed{seed}'] = compare(a, b, rng)
            mask_seeds[seed] = pairs[f'kspace_rss_M - kspace_rss seed{seed}']['psnr']['all']['mean']
    mask_bar = None
    if mask_seeds:
        mean = float(np.mean(list(mask_seeds.values())))
        signs = {np.sign(d) for d in mask_seeds.values()}
        mask_bar = {'per_seed': mask_seeds, 'mean': mean, 'complete': len(mask_seeds) == len(SEEDS),
                    'pass': len(mask_seeds) == len(SEEDS) and abs(mean) >= MASK_BAR_DB and len(signs) == 1}
    twins = {seed: runs[run_name(TWIN, seed)]['ssim']['all'] for seed in SEEDS if run_name(TWIN, seed) in runs}
    health['twins'] = {'ssim': twins, 'pass': all(abs(s - CONTROL_SSIM) <= SSIM_GATE for s in twins.values())}
    mask_correlation = {f'{arm} vs {TWIN} seed{seed}': correlation(results[run_name(arm, seed)]['probability'],
                                                                 results[run_name(TWIN, seed)]['probability'])
                        for arm in ARMS[1:] for seed in SEEDS
                        if run_name(arm, seed) in results and run_name(TWIN, seed) in results}
    twin_seeds = [s for s in SEEDS if run_name(TWIN, s) in results]
    mask_correlation.update({f'{TWIN} seed{s} vs seed{t}': correlation(results[run_name(TWIN, s)]['probability'],
                                                                      results[run_name(TWIN, t)]['probability'])
                             for i, s in enumerate(twin_seeds) for t in twin_seeds[i + 1:]})
    return {'runs': runs, 'loss_bar': loss_bar, 'mask_bar': mask_bar, 'health': health,
            'mask_correlation': mask_correlation, 'pairs': pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-root', type=Path, nargs='+', default=[Path.home() / 'data/seqmri_checkpoints/loss_domain/runs'])
    parser.add_argument('--out', type=Path, default=Path('reports/loss_domain/eval'))
    parser.add_argument('--epochs', type=int, default=100, help='the runs\' epoch count, as in their folder names')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    (args.out / 'runs').mkdir(parents=True, exist_ok=True)

    _, val_loader = build_manifest_dataloaders(
        manifest='configs/datasets/fastmri_local_v1/manifest.csv', roots=ROOTS, classes=['FLAIR', 'knee'],
        image_shape=[320, 320], batch_size=16, num_coils=20, num_workers=8)
    cache = ComplexImageCache(CACHE)
    results = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = run_name(arm, seed, args.epochs)
            found = [root / name for root in args.runs_root if (root / name / 'last_model.pt').exists()]
            if not found:
                print(f'{name}: no checkpoint, skipped')
                continue
            path, cached = found[0], args.out / 'runs' / f'{name}.pt'
            if cached.exists() and cached.stat().st_mtime > (path / 'last_model.pt').stat().st_mtime:
                results[arm, seed] = torch.load(cached, weights_only=False)
            else:
                results[arm, seed] = score_run(path, val_loader, cache, device)
                torch.save(results[arm, seed], cached)
            r = results[arm, seed]
            print(f"{name}: epoch {r['epoch'] + 1}  PSNR_Masked {r['psnr'].mean():.3f}  SSIM_Masked {r['ssim'].mean():.4f}  "
                  f"NMSE {r['nmse'].mean():.5f}  phase std {r['phase_circstd'].mean():.3f}", flush=True)

    summary = summarise(results, np.random.default_rng(0))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nLoss bar (arm - rss, PSNR_Masked dB, all val slices):')
    for arm, bar in summary['loss_bar'].items():
        seeds = '  '.join(f's{s} {d:+.3f}' for s, d in bar['per_seed'].items())
        print(f"  {arm:13s} {seeds}  mean {bar['mean']:+.3f}  {'PASS' if bar['pass'] else 'miss'}{'' if bar['complete'] else ' (incomplete)'}")
    if summary['mask_bar']:
        bar = summary['mask_bar']
        seeds = '  '.join(f's{s} {d:+.3f}' for s, d in bar['per_seed'].items())
        print(f"Mask bar (kspace_rss_M - kspace_rss): {seeds}  mean {bar['mean']:+.3f}  {'PASS' if bar['pass'] else 'miss'}")
    print('Health:', json.dumps(summary['health'], indent=None))
    print('Mask correlation:', json.dumps(summary['mask_correlation'], indent=None))


if __name__ == '__main__':
    main()
