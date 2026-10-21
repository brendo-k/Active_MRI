"""Seed checks for the ISMRM 2027 anatomy pair (R=8, brain + knee VarNet): run-to-run noise floor and layout control.

Two seeds of the shared run (S1, S2) and of the per-anatomy run (A1, A2). Each reconstructor is scored on the whole val set
with common per-slice Bernoulli thresholds (seeds 1234 / 4321, as in eval_adaptive_samplers_r8) and the trainer's metric
mask, given:
  shared recon S_i        its own map; the other seed's shared map
  per-anatomy recon A_i   its own-anatomy map; its other-anatomy map (label swap); the other seed's own-anatomy map
                          (same anatomy, different layout); the other seed's other-anatomy map
Paired per-slice contrasts with 95% volume-bootstrap CIs and the count of volumes below 0:
  noise floor        S2 - S1 and A2 - A1, each with its own maps (run - run between seeds)
  pair               A_i - S_i, each with its own maps
  layout swap        A_i with the other seed's own-anatomy map minus A_i own: a new layout of the same anatomy's mask
  label swap         A_i with its other-anatomy map minus A_i own
  anatomy-specific   label swap minus layout swap (= A_i other-anatomy map minus the other seed's own-anatomy map)
  shared layout swap S_i with the other seed's shared map minus S_i own

    CUDA_VISIBLE_DEVICES=2 python scripts/eval/eval_anatomy_seeds_ismrm_r8.py \
        --shared checkpoints/loupe2d_varnet_r8_brain_knee_shared_100ep_lr1e-3 checkpoints/loupe2d_varnet_r8_brain_knee_shared_100ep_lr1e-3_seed43 \
        --anatcond checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep_lr1e-3_seed43
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
from eval_adaptive_samplers_r8 import ANATOMIES, SEEDS, bootstrap
from eval_anatomy_pair_ismrm_r8 import METRICS, OUT, sampling_maps, score_maps
from mask_gradient_anatomy_r8 import load


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shared', type=Path, nargs=2, required=True, metavar=('SEED_A', 'SEED_B'))
    parser.add_argument('--anatcond', type=Path, nargs=2, required=True, metavar=('SEED_A', 'SEED_B'))
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--tag', default='seeds')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def contrast(scores, plus, minus, metric, groups, volumes, rng):
    delta = (scores[plus[0]][plus[1]][metric] - scores[minus[0]][minus[1]][metric]).numpy()
    out = {}
    for group, keep in groups.items():
        stats = bootstrap(delta[keep], volumes[keep], rng)
        per_volume = [delta[keep][volumes[keep] == v].mean() for v in np.unique(volumes[keep])]
        out[group] = {**stats, 'volumes_below_0': int(np.sum(np.array(per_volume) < 0)), 'n_volumes': len(per_volume)}
    return out


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    cfg = OmegaConf.load(args.anatcond[0] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset if args.limit is None else torch.utils.data.Subset(
        val_loader.dataset, np.linspace(0, len(val_loader.dataset) - 1, args.limit).astype(int).tolist())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.chunk, shuffle=False, num_workers=4,
                                         collate_fn=val_loader.collate_fn)
    anatomy = np.array(sum((list(b['anatomy']) for b in loader), []))
    condition = np.array(sum((list(b['condition']) for b in loader), []))
    volumes = np.array(sum((list(b['fname']) for b in loader), []))

    runs = {'S1': args.shared[0], 'S2': args.shared[1], 'A1': args.anatcond[0], 'A2': args.anatcond[1]}
    models, classes = {}, {}
    for name, run in runs.items():
        models[name], checkpoint = load(run, args.checkpoint, args.device)
        classes[name] = list(checkpoint['options'].label_classes) or list(ANATOMIES)
        conditioned = getattr(checkpoint['options'], 'anatomy_conditioning', None)
        assert conditioned is None or bool(conditioned) == name.startswith('A'), f'{run}: unexpected conditioning'
        print(f'{name}: {run} {args.checkpoint} epoch {checkpoint["epoch"]} seed {checkpoint["options"].seed}', flush=True)

    maps = {}
    for name in ('S1', 'S2'):
        maps[name], old, valid = sampling_maps(models[name], classes[name], loader, args.device)
    for name in ('A1', 'A2'):
        maps[f'{name}_own'], _, _ = sampling_maps(models[name], classes[name], loader, args.device)
        maps[f'{name}_other'], _, _ = sampling_maps(models[name], classes[name], loader, args.device, swap=True)

    given = {'S1': {'own': 'S1', 'other_seed': 'S2'},
             'S2': {'own': 'S2', 'other_seed': 'S1'},
             'A1': {'own': 'A1_own', 'label_swap': 'A1_other', 'other_seed_own': 'A2_own', 'other_seed_other': 'A2_other'},
             'A2': {'own': 'A2_own', 'label_swap': 'A2_other', 'other_seed_own': 'A1_own', 'other_seed_other': 'A1_other'}}
    scores = {}
    for name, model in models.items():
        scores[name] = score_maps(model, classes[name], loader, {k: maps[v] for k, v in given[name].items()}, old, valid,
                                  args.device)
        print(name, {k: {a: round(float(v['psnr'][anatomy == a].mean()), 3) for a in ANATOMIES} for k, v in scores[name].items()},
              flush=True)

    contrasts = {
        'noise_floor_shared': (('S2', 'own'), ('S1', 'own')),
        'noise_floor_anatcond': (('A2', 'own'), ('A1', 'own')),
        'pair_seed_a': (('A1', 'own'), ('S1', 'own')),
        'pair_seed_b': (('A2', 'own'), ('S2', 'own')),
        'layout_swap_a': (('A1', 'other_seed_own'), ('A1', 'own')),
        'layout_swap_b': (('A2', 'other_seed_own'), ('A2', 'own')),
        'label_swap_a': (('A1', 'label_swap'), ('A1', 'own')),
        'label_swap_b': (('A2', 'label_swap'), ('A2', 'own')),
        'anatomy_specific_a': (('A1', 'label_swap'), ('A1', 'other_seed_own')),
        'anatomy_specific_b': (('A2', 'label_swap'), ('A2', 'other_seed_own')),
        'shared_layout_swap_a': (('S1', 'other_seed'), ('S1', 'own')),
        'shared_layout_swap_b': (('S2', 'other_seed'), ('S2', 'own')),
    }
    groups = {**{a: anatomy == a for a in ANATOMIES}, **{c: condition == c for c in ('knee-PD', 'knee-PDFS')}}
    summary = {'runs': {k: str(v) for k, v in runs.items()}, 'checkpoint': args.checkpoint, 'seeds': list(SEEDS),
               'n_slices': {g: int(k.sum()) for g, k in groups.items()},
               'means': {f'{r}/{m}': {metric: {g: float(scores[r][m][metric][k].mean()) for g, k in groups.items()}
                                      for metric in (*METRICS, 'fraction')} for r in scores for m in scores[r]},
               'contrasts': {}}
    for name, (plus, minus) in contrasts.items():
        summary['contrasts'][name] = {metric: contrast(scores, plus, minus, metric, groups, volumes, rng) for metric in METRICS}
        psnr = summary['contrasts'][name]['psnr']
        print(f'{name:22s} ' + '  '.join(f'{g} {psnr[g]["mean"]:+.3f} [{psnr[g]["ci95"][0]:+.3f}, {psnr[g]["ci95"][1]:+.3f}] '
                                          f'{psnr[g]["volumes_below_0"]}/{psnr[g]["n_volumes"]}<0' for g in ('brain', 'knee')),
              flush=True)

    if args.limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'anatomy_{args.tag}.json').write_text(json.dumps(summary, indent=2) + '\n')
        torch.save({'anatomy': anatomy.tolist(), 'condition': condition.tolist(), 'volume': volumes.tolist(),
                    'scores': {r: {m: {k: v.half() for k, v in s.items()} for m, s in d.items()} for r, d in scores.items()}},
                   OUT / f'anatomy_{args.tag}.pt')


if __name__ == '__main__':
    main()
