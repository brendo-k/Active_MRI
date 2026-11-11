"""Decide the criteria of reports/brain_static_vs_loupe/followup_proposal.md from mask-swap scores.

The follow-up trains a fresh VarNet on each seed's learned loupe_shared brain mask M, frozen (frozen_learned_s<seed>),
and repeats static_s42 (static_repeat_s42). brain_static_vs_loupe_swap.py scores the new VarNets with their own mask and
the same seed's other masks (files in --followup); the main experiment's swap files (--swap) hold the static and
loupe_shared VarNets. Both score the same slices with the same draws (asserted). frozen_learned's brain logits are
loupe_shared's (checked here), so its mask is loupe_shared's, and a VarNet's score on one stands for the other.

  validity       every new run reached its last epoch; frozen brain logits bit-identical to their source; own-mask SSIM
  primary        frozen_learned - static, own masks, paired per slice (95% volume-bootstrap CI) per seed; mean and
                 paired t over seeds (mean / (SD / sqrt(n)), n - 1 df); verdict by the frozen thresholds
  secondary      frozen_learned - loupe_shared, own masks (both M), same thresholds
  mechanism      G = (train - val own-mask PSNR) of frozen_learned minus the same for static, in s43 and s44
  decomposition  mask effect, VarNet effect and co-adaptation of the (frozen_learned, static) pair, as in
                 brain_static_vs_loupe_swap_summary.py
  repeat         static_repeat_s42 - static_s42 on S and on M: the repeat-run floor at a fixed seed
  uniform mask   each VarNet's PSNR on its seed's loupe_uniform mask minus on its own

    python scripts/eval/brain_static_vs_loupe_followup_summary.py --swap reports/brain_static_vs_loupe/swap \
        --followup reports/brain_static_vs_loupe/followup --queue ~/data/seqmri_checkpoints/brain_static_vs_loupe \
        --shared <S best_model.pt>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _path  # noqa: F401
from brain_static_vs_loupe_diagnostics import brain_w, checkpoint_of
from brain_static_vs_loupe_swap_summary import decompose, over_seeds
from eval_adaptive_samplers_r8 import bootstrap

SEEDS = (42, 43, 44)
WIN, TIE, MIN_T = 0.10, 0.05, 2.92
MIN_SSIM = 0.80
MECHANISM_SEEDS = (43, 44)
MOVING_MASK_MAX_G, MASK_MIN_G = 0.15, 0.30
NEW_RUNS = tuple(f'frozen_learned_s{s}' for s in SEEDS) + ('static_repeat_s42',)


def load_scores(folders, split):
    """{recipient: {donor: {metric: per-slice array}}} over the swap files of every folder, and the slice volumes."""
    scores, volumes = {}, None
    for folder in folders:
        for path in sorted(folder.glob(f'swap_{split}_*.pt')):
            saved = torch.load(path, weights_only=False)
            volumes = np.array(saved['volume']) if volumes is None else volumes
            assert list(volumes) == list(saved['volume']), f'{path} scored different slices'
            scores[saved['recipient']] = {d: {m: v.double().numpy() for m, v in by_metric.items()}
                                          for d, by_metric in saved['scores'].items()}
    return scores, volumes


def verdict(per_seed):
    values = np.array(per_seed)
    mean, t = values.mean(), over_seeds(values)['t']
    if mean >= WIN and (values > 0).all() and t is not None and t >= MIN_T:
        return 'first arm wins'
    if mean <= -WIN and (values < 0).all():
        return 'second arm wins'
    if abs(mean) < TIE:
        return 'no effect'
    return 'inconclusive'


def contrast(psnr, a, b, volumes, rng):
    """Arm a minus arm b, each VarNet with its own mask, per seed, with the over-seed mean, t and verdict."""
    per_seed = {str(s): bootstrap(psnr[f'{a}_s{s}'][f'{a}_s{s}'] - psnr[f'{b}_s{s}'][f'{b}_s{s}'], volumes, rng)
                for s in SEEDS}
    means = [entry['mean'] for entry in per_seed.values()]
    return {'per_seed': per_seed, **over_seeds(means), 'verdict': verdict(means)}


def validity(args, val):
    shared = brain_w(checkpoint_of(args.shared))
    runs = {}
    for name in NEW_RUNS:
        checkpoint = checkpoint_of(args.queue / name / 'last_model.pt')
        source = shared if name.startswith('static') else brain_w(checkpoint_of(
            args.queue / name.replace('frozen_learned', 'loupe_shared') / 'last_model.pt'))
        runs[name] = {'finished': int(checkpoint['epoch']) + 1 == checkpoint['options'].num_epochs,
                      'brain_logits_identical_to_source': torch.equal(brain_w(checkpoint), source),
                      'own_mask_ssim': float(val[name][name]['ssim'].mean())}
    same_mask = {f'frozen_learned_s{s}': float(np.abs(val[f'frozen_learned_s{s}'][f'frozen_learned_s{s}']['psnr']
                                                      - val[f'frozen_learned_s{s}'][f'loupe_shared_s{s}']['psnr']).max())
                 for s in SEEDS}
    met = all(r['finished'] and r['brain_logits_identical_to_source'] and r['own_mask_ssim'] >= MIN_SSIM
              for r in runs.values())
    return {'runs': runs, 'max_abs_psnr_own_minus_loupe_shared_mask': same_mask, 'met': met}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--swap', type=Path, required=True, help="the main experiment's swap_<split>_<run>.pt folder")
    parser.add_argument('--followup', type=Path, required=True, help="the follow-up runs' swap files; the summary goes here")
    parser.add_argument('--queue', type=Path, required=True, help='folder holding <arm>_s<seed>/ run folders')
    parser.add_argument('--shared', type=Path, required=True, help='checkpoint of the shared mask S')
    args = parser.parse_args()
    rng = np.random.RandomState(0)
    scores = {split: load_scores([args.swap, args.followup], split) for split in ('val', 'train')}
    (val, volumes), (train, _) = scores['val'], scores['train']
    psnr = {split: {r: {d: m['psnr'] for d, m in by_donor.items()} for r, by_donor in s.items()}
            for split, (s, _) in scores.items()}
    for split in psnr:
        for s in SEEDS:
            psnr[split][f'static_s{s}'][f'frozen_learned_s{s}'] = psnr[split][f'static_s{s}'][f'loupe_shared_s{s}']

    own = {split: {r: float(p[r][r].mean()) for r in p if r in p[r]} for split, p in psnr.items()}
    gap = {r: own['train'][r] - own['val'][r] for r in own['val'] if r in own['train']}
    g = {str(s): gap[f'frozen_learned_s{s}'] - gap[f'static_s{s}'] for s in MECHANISM_SEEDS}
    if all(v < MOVING_MASK_MAX_G for v in g.values()):
        reading = '(a) moving mask'
    elif all(v > MASK_MIN_G for v in g.values()):
        reading = '(b) the mask'
    else:
        reading = 'unresolved'

    summary = {
        'n_slices': {'val': int(len(volumes)), 'train': int(len(scores['train'][1]))},
        'validity': validity(args, val),
        'own_mask_psnr': own,
        'primary_frozen_learned-static': contrast(psnr['val'], 'frozen_learned', 'static', volumes, rng),
        'secondary_frozen_learned-loupe_shared': contrast(psnr['val'], 'frozen_learned', 'loupe_shared', volumes, rng),
        'mechanism': {'train_minus_val_own_mask': gap, 'G_frozen_learned_minus_static': g,
                      'loupe_shared_minus_static': {str(s): gap[f'loupe_shared_s{s}'] - gap[f'static_s{s}'] for s in SEEDS},
                      'reading': reading},
        'decomposition_frozen_learned-static': {str(s): decompose(psnr['val'], f'frozen_learned_s{s}', f'static_s{s}', volumes, rng)
                                                for s in SEEDS},
        'repeat_static_s42': {mask: bootstrap(psnr['val']['static_repeat_s42'][donor] - psnr['val']['static_s42'][donor], volumes, rng)
                              for mask, donor in (('S', 'static_s42'), ('M', 'loupe_shared_s42'))},
        'uniform_mask_minus_own_mask': {r: float((psnr['val'][r][f'loupe_uniform_s{r.rsplit("_s", 1)[1]}'] - psnr['val'][r][r]).mean())
                                        for arm in ('static', 'loupe_shared', 'frozen_learned', 'static_repeat') for s in SEEDS
                                        if (r := f'{arm}_s{s}') in psnr['val']},
    }
    decomposition = summary['decomposition_frozen_learned-static']
    summary['decomposition_frozen_learned-static']['over_seeds'] = {
        part: over_seeds([decomposition[str(s)][part]['mean'] for s in SEEDS]) for part in decomposition['42']}
    (args.followup / 'followup_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    print('validity:', json.dumps(summary['validity']))
    for name in ('primary_frozen_learned-static', 'secondary_frozen_learned-loupe_shared'):
        c = summary[name]
        cells = '  '.join(f"s{s} {c['per_seed'][str(s)]['mean']:+.3f} [{c['per_seed'][str(s)]['ci95'][0]:+.2f},"
                          f"{c['per_seed'][str(s)]['ci95'][1]:+.2f}]" for s in SEEDS)
        print(f"{name:40s} {cells}  mean {c['mean']:+.3f} t {c['t']:+.2f} -> {c['verdict']}")
    print('mechanism:', json.dumps(summary['mechanism']))
    for part, entry in decomposition['over_seeds'].items():
        print(f"(frozen_learned, static) {part:14s} by seed {[round(v, 3) for v in entry['by_seed']]} mean {entry['mean']:+.3f}")
    print('repeat static_s42:', json.dumps(summary['repeat_static_s42']))
    print('uniform mask minus own mask:', {k: round(v, 3) for k, v in summary['uniform_mask_minus_own_mask'].items()})


if __name__ == '__main__':
    main()
