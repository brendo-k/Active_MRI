"""Per-anatomy vs shared LOUPE masks (R=8, brain + knee VarNet): mask x reconstructor cross-evaluation for ISMRM 2027.

Both runs share every setting except `sampler.anatomy_conditioning`. On the whole val set, each run's reconstructor is
given three probability maps with common per-slice Bernoulli thresholds (seeds 1234 and 4321, as in
eval_adaptive_samplers_r8), scored with the trainer's metric mask:
  shared   the shared run's map
  own      the per-anatomy run's map for the slice's anatomy
  other    the per-anatomy run's map for the other anatomy (label swap)
Paired per-slice differences with 95% volume-bootstrap CIs, per anatomy:
  run - run      per-anatomy recon + own map  minus  shared recon + shared map
  mask effect    own map minus shared map, on each reconstructor
  label swap     other map minus own map, on each reconstructor
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
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='N evenly spaced val slices only (smoke tests)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def labelled(batch, classes, device, swap=False):
    part = to_device(batch, device)
    names = [ANATOMIES[1 - ANATOMIES.index(a)] if swap else a for a in batch['anatomy']]
    part['label'] = torch.as_tensor([classes.index(n) for n in names], device=device)
    return part


@torch.no_grad()
def sampling_maps(model, classes, loader, device, swap=False):
    rescaled, old, valid = [], [], []
    for batch in loader:
        part = labelled(batch, classes, device, swap)
        proposal = model.propose_mask(part['kspace'], anatomy_labels=part['label'])
        rescaled.append(proposal['rescaled'][:, 0].half().cpu())
        old.append(proposal['old_mask'][:, 0].bool().cpu())
        valid.append(proposal['valid_mask'][:, 0].bool().cpu())
    return torch.cat(rescaled), torch.cat(old), torch.cat(valid)


@torch.no_grad()
def score_maps(model, classes, loader, maps, old, valid, device):
    """Per-slice PSNR / SSIM for each named map, mean over SEEDS; also the realised sampling fraction per map."""
    count = len(old)
    scores = {name: {m: torch.zeros(count) for m in (*METRICS, 'fraction')} for name in maps}
    start = 0
    for batch in loader:
        part = labelled(batch, classes, device)
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

    shared_map, old, valid = sampling_maps(models['shared'], classes['shared'], loader, args.device)
    own_map, _, _ = sampling_maps(models['anatcond'], classes['anatcond'], loader, args.device)
    other_map, _, _ = sampling_maps(models['anatcond'], classes['anatcond'], loader, args.device, swap=True)
    maps = {'shared': shared_map, 'own': own_map, 'other': other_map}

    scores = {}
    for name, model in models.items():
        scores[name] = score_maps(model, classes[name], loader, maps, old, valid, args.device)
        print(name, 'recon PSNR', {m: {a: round(float(scores[name][m]['psnr'][anatomy == a].mean()), 3) for a in ANATOMIES}
                                   for m in maps}, flush=True)

    contrasts = {
        'run_minus_run': (('anatcond', 'own'), ('shared', 'shared')),
        'mask_effect_on_shared_recon': (('shared', 'own'), ('shared', 'shared')),
        'mask_effect_on_anatcond_recon': (('anatcond', 'own'), ('anatcond', 'shared')),
        'label_swap_on_anatcond_recon': (('anatcond', 'other'), ('anatcond', 'own')),
        'label_swap_on_shared_recon': (('shared', 'other'), ('shared', 'own')),
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
    mean_maps = {f'{name}/{a}': effective[name][torch.as_tensor(anatomy == a)].mean(0) for name in maps for a in ANATOMIES}
    if args.limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'anatomy_pair_{args.tag}.json').write_text(json.dumps(summary, indent=2) + '\n')
        torch.save({'mean_maps': mean_maps, 'anatomy': anatomy.tolist(), 'condition': condition.tolist(),
                    'volume': volumes.tolist(), 'scores': {r: {m: {k: v.half() for k, v in s.items()} for m, s in d.items()}
                                                           for r, d in scores.items()}},
                   OUT / f'anatomy_pair_{args.tag}.pt')


if __name__ == '__main__':
    main()
