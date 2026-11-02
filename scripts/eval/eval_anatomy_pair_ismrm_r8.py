"""Per-anatomy vs shared LOUPE masks (R=8, brain + knee VarNet): mask x reconstructor cross-evaluation for ISMRM 2027.

Both runs share every setting except `sampler.anatomy_conditioning`. On the whole val set, each run's reconstructor is
given these probability maps with common per-slice Bernoulli thresholds (seeds 1234 and 4321, as in
eval_adaptive_samplers_r8), scored with the trainer's metric mask:
  shared   the shared run's map
  as_<c>   the per-class run's map for class c, on every slice (classes: the run's anatomies or condition classes)
  own      as_<c> of the slice's own class
  other    with two classes, as_<c> of the other class (label swap)
Paired per-slice differences with 95% volume-bootstrap CIs, per anatomy and knee protocol:
  run - run      per-anatomy recon + own map  minus  shared recon + shared map
  mask effect    own map minus shared map, on each reconstructor
  label swap     other map (with more than two classes, each as_<c>) minus own map, on each reconstructor
  recon effect   per-anatomy recon minus shared recon, on the shared map and on the own map
Also saved: each anatomy's mean effective sampling map per condition (figure) and the realised sampling fraction.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_anatomy_pair_ismrm_r8.py \
        --shared checkpoints/loupe2d_varnet_r8_brain_knee_shared_100ep_lr1e-3 \
        --anatcond checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep --tag seed42
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
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch, compute_ssim_torch
from eval_adaptive_samplers_r8 import ANATOMIES, SEEDS, bootstrap, thresholds
from mask_gradient_anatomy_r8 import load, to_device

OUT = Path('reports/ismrm2027')
METRICS = ('psnr', 'ssim')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--anatcond', type=Path, required=True)
    parser.add_argument('--checkpoint', default='last_model.pt')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out', type=Path, default=OUT, help='folder for anatomy_pair_<tag>.json / .pt')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def labelled(batch, classes, label_key, device, forced=None):
    """The batch on `device`, every slice labelled with its own class or, given `forced`, with that class."""
    part = to_device(batch, device)
    names = [forced or name for name in batch[label_key]]
    part['label'] = torch.as_tensor([classes.index(n) for n in names], device=device)
    return part


@torch.no_grad()
def sampling_maps(model, classes, label_key, loader, device, forced=None):
    rescaled, old, valid = [], [], []
    for batch in loader:
        part = labelled(batch, classes, label_key, device, forced)
        proposal = model.propose_mask(part['kspace'], anatomy_labels=part['label'])
        rescaled.append(proposal['rescaled'][:, 0].half().cpu())
        old.append(proposal['old_mask'][:, 0].bool().cpu())
        valid.append(proposal['valid_mask'][:, 0].bool().cpu())
    return torch.cat(rescaled), torch.cat(old), torch.cat(valid)


@torch.no_grad()
def score_maps(model, classes, label_key, loader, maps, old, valid, device):
    """Per-slice PSNR / SSIM for each named map, mean over SEEDS; also the realised sampling fraction per map."""
    count = len(old)
    scores = {name: {m: torch.zeros(count) for m in (*METRICS, 'fraction')} for name in maps}
    start = 0
    for batch in loader:
        part = labelled(batch, classes, label_key, device)
        rows = torch.arange(start, start + len(part['kspace']))
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        for seed in SEEDS:
            threshold = torch.stack([thresholds(seed, int(i), old.shape[1:]) for i in rows])
            for name, rescaled in maps.items():
                new = (rescaled[rows].float() > threshold) & valid[rows]
                mask = (old[rows] | new).float().unsqueeze(1).to(device)
                output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
                output = output[-1] if isinstance(output, list) else output
                masked = _apply_signal_mask(output, target, part['metric_mask'])
                scores[name]['psnr'][rows] += compute_psnr_torch(*masked).flatten().cpu() / len(SEEDS)
                scores[name]['ssim'][rows] += compute_ssim_torch(*masked).flatten().cpu() / len(SEEDS)
                scores[name]['fraction'][rows] += mask.flatten(1).mean(1).cpu() / len(SEEDS)
        start += len(rows)
    return scores


def main():
    args = parse_args()
    rng = np.random.RandomState(0)
    cfg = OmegaConf.load(args.anatcond / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    dataset = val_loader.dataset if args.limit is None else torch.utils.data.Subset(
        val_loader.dataset, np.linspace(0, len(val_loader.dataset) - 1, args.limit).astype(int).tolist())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.chunk, shuffle=False, num_workers=4,
                                         collate_fn=val_loader.collate_fn)
    anatomy = np.array(sum((list(b['anatomy']) for b in loader), []))
    condition = np.array(sum((list(b['condition']) for b in loader), []))
    volumes = np.array(sum((list(b['fname']) for b in loader), []))

    models, classes = {}, {}
    for name, run in (('shared', args.shared), ('anatcond', args.anatcond)):
        models[name], checkpoint = load(run, args.checkpoint, args.device)
        classes[name] = list(checkpoint['options'].label_classes) or list(ANATOMIES)
        conditioned = getattr(checkpoint['options'], 'anatomy_conditioning', None)
        assert conditioned is None or bool(conditioned) == (name == 'anatcond'), f'{run}: unexpected conditioning'
        print(f'{name}: {run} {args.checkpoint} epoch {checkpoint["epoch"]}', flush=True)
    label_key = checkpoint['options'].label_key
    labels = np.array(sum((list(b[label_key]) for b in loader), []))
    routed_classes = classes['anatcond']

    shared_map, old, valid = sampling_maps(models['shared'], classes['shared'], label_key, loader, args.device)
    maps = {'shared': shared_map, **{f'as_{c}': sampling_maps(models['anatcond'], routed_classes, label_key, loader,
                                                              args.device, forced=c)[0] for c in routed_classes}}

    scores = {}
    for name, model in models.items():
        scores[name] = score_maps(model, classes[name], label_key, loader, maps, old, valid, args.device)
    # own: each slice's map for its own class; other (two classes only): the other class's map, the label swap.
    derived = {'own': [f'as_{n}' for n in labels]}
    if len(routed_classes) == 2:
        derived['other'] = [f'as_{routed_classes[1 - routed_classes.index(n)]}' for n in labels]
    for name, per_slice in derived.items():
        pick = {m: torch.as_tensor(np.array(per_slice) == m) for m in set(per_slice)}
        maps[name] = torch.zeros_like(shared_map)
        for m, rows in pick.items():
            maps[name][rows] = maps[m][rows]
        for run_scores in scores.values():
            run_scores[name] = {metric: sum(run_scores[m][metric] * rows for m, rows in pick.items())
                                for metric in (*METRICS, 'fraction')}
    for name in models:
        print(name, 'recon PSNR', {m: {a: round(float(scores[name][m]['psnr'][anatomy == a].mean()), 3) for a in ANATOMIES}
                                   for m in maps}, flush=True)

    swap_maps = ['other'] if 'other' in maps else [f'as_{c}' for c in routed_classes]
    contrasts = {
        'run_minus_run': (('anatcond', 'own'), ('shared', 'shared')),
        'mask_effect_on_shared_recon': (('shared', 'own'), ('shared', 'shared')),
        'mask_effect_on_anatcond_recon': (('anatcond', 'own'), ('anatcond', 'shared')),
        **{f'label_swap_on_{run}_recon' + ('' if m == 'other' else f'/{m}'): ((run, m), (run, 'own'))
           for run in ('anatcond', 'shared') for m in swap_maps},
        'recon_effect_on_shared_map': (('anatcond', 'shared'), ('shared', 'shared')),
        'recon_effect_on_own_map': (('anatcond', 'own'), ('shared', 'own')),
    }
    groups = {**{a: anatomy == a for a in ANATOMIES}, **{c: condition == c for c in ('knee-PD', 'knee-PDFS')}}
    summary = {'shared': str(args.shared), 'anatcond': str(args.anatcond), 'checkpoint': args.checkpoint,
               'seeds': list(SEEDS), 'n_slices': {g: int(k.sum()) for g, k in groups.items()},
               'n_volumes': {g: int(len(np.unique(volumes[k]))) for g, k in groups.items()},
               'means': {f'{r}/{m}': {metric: {g: float(scores[r][m][metric][k].mean()) for g, k in groups.items()}
                                      for metric in (*METRICS, 'fraction')} for r in models for m in maps},
               'contrasts': {}}
    for contrast, ((run_a, map_a), (run_b, map_b)) in contrasts.items():
        summary['contrasts'][contrast] = {
            metric: {g: bootstrap((scores[run_a][map_a][metric] - scores[run_b][map_b][metric]).numpy()[k], volumes[k], rng)
                     for g, k in groups.items()}
            for metric in METRICS}
        psnr = summary['contrasts'][contrast]['psnr']
        print(f'{contrast:32s} ' + '  '.join(f'{g} {psnr[g]["mean"]:+.3f} [{psnr[g]["ci95"][0]:+.3f}, {psnr[g]["ci95"][1]:+.3f}]'
                                              for g in groups), flush=True)

    effective = {name: (m.float() + old.float()).clamp(max=1) for name, m in maps.items()}
    mean_maps = {f'{name}/{g}': effective[name][torch.as_tensor(k)].mean(0) for name in maps for g, k in groups.items()}
    if args.limit is None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f'anatomy_pair_{args.tag}.json').write_text(json.dumps(summary, indent=2) + '\n')
        torch.save({'mean_maps': mean_maps, 'anatomy': anatomy.tolist(), 'condition': condition.tolist(),
                    'volume': volumes.tolist(), 'scores': {r: {m: {k: v.half() for k, v in s.items()} for m, s in d.items()}
                                                           for r, d in scores.items()}},
                   args.out / f'anatomy_pair_{args.tag}.pt')


if __name__ == '__main__':
    main()
