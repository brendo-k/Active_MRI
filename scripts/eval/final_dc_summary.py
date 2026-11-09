"""Decide reports/final_dc/proposal.md's frozen criteria from the per-slice scores of final_dc_eval.py, and tabulate the
post-freeze diagnostics.

Order: (1) lambda = 0 reproduces each checkpoint's published eval; (2) the frozen stage-1 criteria on the 2x2; (3) the side
readout (loss_domain kspace_gt against rss, same loader and draws); (4) the lambda curve and the mechanism readouts by condition.
Paired differences are per slice, with a 95% CI of +-1.96 SE over per-volume means (as varnet_train_speed_eval.py).

    python scripts/eval/final_dc_summary.py reports/final_dc
"""
import argparse
import json
from pathlib import Path

import numpy as np

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RECIPES = ('control', 'muon', 'layer', 'muon_layer')
SEEDS = (42, 43)
CONDITIONS = ('all', 'brain', 'knee-PD', 'knee-PDFS')
LOSS_DOMAIN_RUNS = {'rss': 'loss_domain_rss_r8_flair_knee_100ep_seed42', 'kspace_gt': 'loss_domain_kspace_gt_r8_flair_knee_100ep_seed42'}
REPRODUCE_DB = 0.003
GAIN_BAR, FAIL_BAR, SSIM_GUARD, CONDITION_GUARD = 0.10, 0.05, -0.002, -0.10
SIDE_SHRINK, SIDE_PDFS_RATIO = 0.5, 0.95
DIAGNOSTICS = ('phase_offset', 'phase_estimate_error', 'dc_error', 'dc_error_aligned', 'dc_error_estaligned', 'maps_floor',
               'error_share_sampled', 'sampled_fraction', 'mean_ratio_l0', 'mean_ratio_l1', 'mean_ratio_estaligned_l1')


def lam(l):
    return f'l{l:g}'


def keep(run, condition):
    return np.ones(len(run['condition']), bool) if condition == 'all' else run['condition'] == condition


def paired(delta, volumes):
    names, index = np.unique(volumes, return_inverse=True)
    per_volume = np.bincount(index, weights=delta) / np.bincount(index)
    half = 1.96 * per_volume.std(ddof=1) / np.sqrt(len(names))
    return {'mean': float(delta.mean()), 'ci': [float(delta.mean() - half), float(delta.mean() + half)],
            'frac_volumes_positive': float((per_volume > 0).mean()), 'n_volumes': int(len(names))}


def difference(a, key_a, b, key_b, condition):
    if not (a['volume'] == b['volume']).all():
        raise ValueError('scored on different slices')
    k = keep(a, condition)
    return paired(a[key_a][k] - b[key_b][k], a['volume'][k])


def means(run, key):
    return {c: float(run[key][keep(run, c)].mean()) for c in CONDITIONS}


def reproduction(scores, root):
    rows = {}
    reference = np.load(root / 'reports/muon_layernorm_100ep/eval.per_slice.npz', allow_pickle=True)
    for recipe in RECIPES:
        for seed in SEEDS:
            name = f'{recipe}_s{seed}'
            run = scores.get(f'ml100_{name}')
            if run is None:
                continue
            if not (reference[f'{name}/volume'] == run['volume']).all():
                raise ValueError(f'{name}: slice order differs from the published eval')
            gap = run['psnr_l0'] - reference[f'{name}/psnr']
            rows[f'ml100_{name}'] = {'reference': float(reference[f'{name}/psnr'].mean()), 'lambda0': float(run['psnr_l0'].mean()),
                                     'mean_gap': float(gap.mean()), 'max_abs_slice_gap': float(np.abs(gap).max()),
                                     'pass': bool(abs(gap.mean()) <= REPRODUCE_DB)}
    summary = json.loads(Path(root.parent / 'exp-loss_domain/reports/loss_domain/eval_v2/summary.json').read_text())
    for arm, name in LOSS_DOMAIN_RUNS.items():
        run = scores.get(f'loss_domain_{name}')
        if run is None:
            continue
        published = summary['runs'][name]['psnr']['all']
        rows[f'loss_domain_{arm}'] = {'reference': published, 'lambda0': float(run['psnr_l0'].mean()),
                                      'mean_gap': float(run['psnr_l0'].mean() - published), 'max_abs_slice_gap': None,
                                      'pass': bool(abs(run['psnr_l0'].mean() - published) <= REPRODUCE_DB)}
    return rows


def frozen_criteria(scores):
    deltas, ssim, conditions = {}, {}, {}
    for recipe in RECIPES:
        for seed in SEEDS:
            run = scores[f'ml100_{recipe}_s{seed}']
            deltas[recipe, seed] = difference(run, 'psnr_l1', run, 'psnr_l0', 'all')['mean']
            ssim[recipe, seed] = difference(run, 'ssim_l1', run, 'ssim_l0', 'all')['mean']
            for c in CONDITIONS[1:]:
                conditions[recipe, seed, c] = difference(run, 'psnr_l1', run, 'psnr_l0', c)['mean']
    mean = float(np.mean(list(deltas.values())))
    c1 = mean >= GAIN_BAR and all(d > 0 for d in deltas.values())
    c2 = all(ssim['muon_layer', s] >= SSIM_GUARD for s in SEEDS)
    worse = [f'{r} {c}' for r in RECIPES for c in CONDITIONS[1:] if all(conditions[r, s, c] <= CONDITION_GUARD for s in SEEDS)]
    c3 = not worse
    failure = mean < FAIL_BAR or any(deltas['muon_layer', s] < 0 for s in SEEDS)
    verdict = 'success' if c1 and c2 and c3 else 'failure' if failure else 'inconclusive'
    return {'delta_psnr': {f'{r}_s{s}': d for (r, s), d in deltas.items()}, 'mean_delta_psnr': mean,
            'delta_ssim': {f'{r}_s{s}': d for (r, s), d in ssim.items()},
            'delta_psnr_by_condition': {f'{r}_s{s} {c}': d for (r, s, c), d in conditions.items()},
            'C1_gain': bool(c1), 'C2_ssim_guard': bool(c2), 'C3_condition_guard': bool(c3), 'C3_worse_in_both_seeds': worse,
            'failure_rule': bool(failure), 'verdict': verdict}


def side_readout(scores):
    rss, kgt = (scores.get(f'loss_domain_{LOSS_DOMAIN_RUNS[a]}') for a in ('rss', 'kspace_gt'))
    if rss is None or kgt is None:
        return None
    gap0 = {c: difference(kgt, 'psnr_l0', rss, 'psnr_l0', c) for c in CONDITIONS}
    gap1 = {c: difference(kgt, 'psnr_l1', rss, 'psnr_l1', c) for c in CONDITIONS}
    against_raw_rss = {c: difference(kgt, 'psnr_l1', rss, 'psnr_l0', c) for c in CONDITIONS}
    pdfs_ratio = {f'{arm} {lam(l)}': means(run, f'mean_ratio_{lam(l)}')['knee-PDFS']
                  for arm, run in (('rss', rss), ('kspace_gt', kgt)) for l in (0.0, 1.0)}
    shrink = gap1['all']['mean'] / gap0['all']['mean']
    deficit0, deficit1 = max(0.0, -gap0['all']['mean']), max(0.0, -gap1['all']['mean'])
    a = deficit1 <= SIDE_SHRINK * deficit0
    b = pdfs_ratio['kspace_gt l1'] >= SIDE_PDFS_RATIO
    return {'kspace_gt_l1_minus_rss_l0': against_raw_rss, 'gap_same_lambda_l0': gap0, 'gap_same_lambda_l1': gap1,
            'gap_ratio_l1_over_l0': shrink, 'deficit_l0': deficit0, 'deficit_l1': deficit1, 'knee_PDFS_mean_ratio': pdfs_ratio,
            'a_deficit_shrinks_to_half': bool(a), 'b_pdfs_ratio': bool(b), 'recovers': bool(a and b),
            'self_delta': {arm: {c: difference(run, 'psnr_l1', run, 'psnr_l0', c) for c in CONDITIONS}
                           for arm, run in (('rss', rss), ('kspace_gt', kgt))}}


def lambda_curve(runs):
    """Mean over the given runs of PSNR(lambda) - PSNR(0) and SSIM(lambda) - SSIM(0), by condition."""
    curve = {}
    for c in CONDITIONS:
        curve[c] = {lam(l): {'psnr': float(np.mean([difference(r, f'psnr_{lam(l)}', r, 'psnr_l0', c)['mean'] for r in runs])),
                             'ssim': float(np.mean([difference(r, f'ssim_{lam(l)}', r, 'ssim_l0', c)['mean'] for r in runs]))}
                    for l in LAMBDAS[1:]}
    return curve


def mechanism(run):
    out = {k: means(run, k) for k in DIAGNOSTICS}
    for name, key in (('oracle_aligned_l0', 'psnr_aligned_l0'), ('oracle_aligned_l1', 'psnr_aligned_l1'),
                      ('estimated_aligned_l0', 'psnr_estaligned_l0'), ('estimated_aligned_l1', 'psnr_estaligned_l1'),
                      ('raw_l1', 'psnr_l1'), ('gain_offset_fit_l0', 'fit_psnr_l0'), ('gain_offset_fit_l1', 'fit_psnr_l1')):
        out[f'delta_psnr_{name}'] = {c: difference(run, key, run, 'psnr_l0', c)['mean'] for c in CONDITIONS}
    out['delta_ssim_oracle_aligned_l1'] = {c: difference(run, 'ssim_aligned_l1', run, 'ssim_l0', c)['mean'] for c in CONDITIONS}
    out['psnr_l0'] = means(run, 'psnr_l0')
    out['ssim_l0'] = means(run, 'ssim_l0')
    return out


def average(blocks):
    """Mean of the mechanism blocks of several runs, key by key."""
    return {k: {c: float(np.mean([b[k][c] for b in blocks])) for c in CONDITIONS} for k in blocks[0]}


def fmt_ci(p):
    return f"{p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"


def write_markdown(result, path):
    lines = ['# final_dc summary', '', '## Reproduction (lambda = 0 against the published eval, PSNR_Masked dB)', '',
             '| run | published | lambda = 0 | mean gap | max slice gap | within 0.003 dB |', '|---|---|---|---|---|---|']
    for name, row in result['reproduction'].items():
        slice_gap = '—' if row['max_abs_slice_gap'] is None else f"{row['max_abs_slice_gap']:.4f}"
        lines.append(f"| {name} | {row['reference']:.3f} | {row['lambda0']:.3f} | {row['mean_gap']:+.4f} | {slice_gap} | "
                     f"{'yes' if row['pass'] else '**no**'} |")
    f = result['frozen']
    lines += ['', '## Frozen criteria (2x2, lambda = 1 minus lambda = 0)', '',
              f"Mean over the 8 checkpoints: {f['mean_delta_psnr']:+.3f} dB. Verdict: **{f['verdict']}** "
              f"(C1 {f['C1_gain']}, C2 {f['C2_ssim_guard']}, C3 {f['C3_condition_guard']}, failure rule {f['failure_rule']}).", '',
              '| checkpoint | PSNR λ=0 | ΔPSNR all [95% CI] | ΔSSIM all | Δ brain | Δ knee-PD | Δ knee-PDFS |', '|---|---|---|---|---|---|---|']
    for name, block in result['per_run'].items():
        if not name.startswith('ml100_'):
            continue
        d = block['delta_l1']
        lines.append(f"| {name.removeprefix('ml100_')} | {block['psnr_l0']['all']:.2f} | {fmt_ci(d['psnr']['all'])} | "
                     f"{d['ssim']['all']['mean']:+.4f} | " + ' | '.join(f"{d['psnr'][c]['mean']:+.3f}" for c in CONDITIONS[1:]) + ' |')
    if f['C3_worse_in_both_seeds']:
        lines.append(f"\nWorse by >= 0.10 dB in both seeds: {', '.join(f['C3_worse_in_both_seeds'])}.")
    lines += ['', '## Lambda curve (mean over the 8 checkpoints; ΔPSNR dB / ΔSSIM against lambda = 0)', '',
              '| condition | ' + ' | '.join(lam(l) for l in LAMBDAS[1:]) + ' |', '|---|' + '---|' * (len(LAMBDAS) - 1)]
    for c, row in result['lambda_curve_2x2'].items():
        lines.append(f'| {c} | ' + ' | '.join(f"{v['psnr']:+.3f} / {v['ssim']:+.4f}" for v in row.values()) + ' |')
    lines += ['', '## Mechanism (post-freeze diagnostics; means by condition)', '',
              'Phase offsets are tissue-weighted mean |angle| in rad; errors are relative norms at the sampled points; Δ are PSNR dB '
              'against the same run at lambda = 0. Oracle alignment takes the phase from the fully sampled data (noise phase '
              'included), so it is an upper bound with leakage; the estimated alignment uses the sampled data only (σ = 8 px).', '']
    groups = {'2x2 mean (old flip)': result['mechanism_2x2'], **{f'loss_domain {a} (fixed flip)': m
                                                                 for a, m in result['mechanism_loss_domain'].items()}}
    rows = (('phase_offset', 'phase offset (rad)', '.3f'), ('dc_error', 'error at samples', '.3f'), ('maps_floor', 'maps floor', '.3f'),
            ('dc_error_aligned', 'error at samples, oracle-aligned', '.3f'), ('error_share_sampled', 'share of error at samples', '.3f'),
            ('delta_psnr_raw_l1', 'Δ hard DC', '+.3f'), ('delta_psnr_oracle_aligned_l0', 'Δ oracle-aligned, no DC', '+.3f'),
            ('delta_psnr_oracle_aligned_l1', 'Δ oracle-aligned + hard DC', '+.3f'),
            ('delta_psnr_estimated_aligned_l1', 'Δ estimated-aligned + hard DC', '+.3f'),
            ('phase_estimate_error', 'estimate phase error (rad)', '.3f'),
            ('mean_ratio_l0', 'output/target mean, λ=0', '.3f'), ('mean_ratio_l1', 'output/target mean, λ=1', '.3f'),
            ('delta_psnr_gain_offset_fit_l0', 'Δ oracle gain+offset fit, λ=0', '+.3f'),
            ('delta_psnr_gain_offset_fit_l1', 'Δ oracle gain+offset fit, λ=1', '+.3f'))
    for group, block in groups.items():
        lines += [f'### {group}', '', '| readout | ' + ' | '.join(CONDITIONS) + ' |', '|---|' + '---|' * len(CONDITIONS)]
        lines += [f'| {label} | ' + ' | '.join(format(block[key][c], spec) for c in CONDITIONS) + ' |' for key, label, spec in rows]
        lines.append('')
    s = result['side_readout']
    if s:
        lines += ['## Side readout: does the step recover kspace_gt\'s intensity gap? (seed 42, descriptive)', '',
                  '| comparison | ' + ' | '.join(CONDITIONS) + ' |', '|---|' + '---|' * len(CONDITIONS),
                  '| kspace_gt λ=1 − rss λ=0 | ' + ' | '.join(fmt_ci(s['kspace_gt_l1_minus_rss_l0'][c]) for c in CONDITIONS) + ' |',
                  '| kspace_gt − rss, both λ=0 | ' + ' | '.join(fmt_ci(s['gap_same_lambda_l0'][c]) for c in CONDITIONS) + ' |',
                  '| kspace_gt − rss, both λ=1 | ' + ' | '.join(fmt_ci(s['gap_same_lambda_l1'][c]) for c in CONDITIONS) + ' |',
                  '| rss λ=1 − λ=0 | ' + ' | '.join(fmt_ci(s['self_delta']['rss'][c]) for c in CONDITIONS) + ' |',
                  '| kspace_gt λ=1 − λ=0 | ' + ' | '.join(fmt_ci(s['self_delta']['kspace_gt'][c]) for c in CONDITIONS) + ' |', '',
                  f"knee-PDFS output/target mean: " + ', '.join(f'{k} {v:.3f}' for k, v in s['knee_PDFS_mean_ratio'].items()) + '.',
                  f"Same-λ deficit: {s['deficit_l0']:.3f} dB at λ=0, {s['deficit_l1']:.3f} dB at λ=1 (signed gap ratio "
                  f"{s['gap_ratio_l1_over_l0']:+.2f}). (a) {s['a_deficit_shrinks_to_half']}, "
                  f"(b) {s['b_pdfs_ratio']}: recovers = **{s['recovers']}**.", '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report_dir', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    scores = {p.stem: dict(np.load(p, allow_pickle=True)) for p in sorted((args.report_dir / 'scores').glob('*.npz'))}
    per_run = {}
    for name, run in scores.items():
        per_run[name] = {'psnr_l0': means(run, 'psnr_l0'), 'ssim_l0': means(run, 'ssim_l0'),
                         'delta_l1': {m: {c: difference(run, f'{m}_l1', run, f'{m}_l0', c) for c in CONDITIONS} for m in ('psnr', 'ssim')},
                         'own_output_max_abs_gap': float(run['own_output_max_abs_gap'])}
    ml100 = [scores[f'ml100_{r}_s{s}'] for r in RECIPES for s in SEEDS]
    result = {'reproduction': reproduction(scores, args.repo_root), 'frozen': frozen_criteria(scores), 'per_run': per_run,
              'lambda_curve_2x2': lambda_curve(ml100), 'mechanism_2x2': average([mechanism(r) for r in ml100]),
              'mechanism_2x2_per_run': {f'{r}_s{s}': mechanism(scores[f'ml100_{r}_s{s}']) for r in RECIPES for s in SEEDS},
              'mechanism_loss_domain': {a: mechanism(scores[f'loss_domain_{n}']) for a, n in LOSS_DOMAIN_RUNS.items()
                                        if f'loss_domain_{n}' in scores},
              'side_readout': side_readout(scores)}
    (args.report_dir / 'summary.json').write_text(json.dumps(result, indent=1))
    write_markdown(result, args.report_dir / 'summary.md')
    print((args.report_dir / 'summary.md').read_text())


if __name__ == '__main__':
    main()
