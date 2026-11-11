"""Why the brain_static_vs_loupe arms did not separate: convergence, the reach of the shared mask's support, a reference.

Each section writes <out>/<section>.json and prints a short summary.
  train_loss     each run's train loss (train_sequential.log) at epochs 10/30/50/58/65 and its relative drop over epochs
                 50-58 and 58-65 (the lr steps down x0.1 after epoch 58); the trainer's brain val PSNR at epochs 50-65
                 (150 slices, per-batch data range, so a trend only, not comparable with eval.json)
  reference      the joint full-data VarNet that S was learned with, scored as in eval.json (full brain val, its own
                 brain map, common per-slice draws)
  masks          within the acquired extent (where any loupe_uniform map is > 0):
                   S's dead region (p = sigmoid(slope w) < DEAD_P) and its logits slope * w;
                   Adam's reach, sum over steps of the mask lr: about the most any w can move;
                   each run's change in w from its start (S, or the uniform start rebuilt as train.py builds it);
                   per-pixel entropy of each mean brain map (bits, mean over the extent);
                   pairwise budget relocated between mean maps; radial density; seed-to-seed correlation of the maps
                   after Gaussian blurs of sigma 0/2/4/8 px
  trajectory     each LOUPE run's mean brain map (64 val slices) at its start, at best_model.pt (an earlier epoch) and
                 at last_model.pt: centre (4-16 px) and outer (>= 128 px) density, budget moved between snapshots,
                 and the share of the extent with p outside the uniform start's range [0.01, 0.99]
  dead_gradient  gradient of a training loss (batch-mean L1, straight-through masks) with respect to the brain mask w,
                 on the first 6 batches of 4 training slices, in S's dead region and in the rest of the extent: at the
                 end of loupe_shared_s42 and at its start (the seed-42 random VarNet with mask S)

    python scripts/eval/brain_static_vs_loupe_diagnostics.py --report reports/brain_static_vs_loupe \
        --queue ~/data/seqmri_checkpoints/brain_static_vs_loupe \
        --shared-run ~/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42 \
        --sections masks trajectory --device cpu
"""
import argparse
import itertools
import json
import random
import re
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F

import _path  # noqa: F401
from activemri.training.trainer import build_model
from brain_static_vs_loupe_eval import ARMS, budget_relocated, full_brain_val_loader, score_run
from brain_static_vs_loupe_summary import DEAD_P
from brain_static_vs_loupe_swap import brain_loader
from mask_gradient_anatomy_r8 import load

SEEDS = (42, 43, 44)
SECTIONS = ('train_loss', 'reference', 'masks', 'trajectory', 'dead_gradient')
LOSS_EPOCHS = (10, 30, 50, 58, 65)
VAL_EPOCHS = (50, 55, 60, 65)
RADIUS_EDGES = (4, 8, 16, 32, 64, 96, 128, 160, 230)  # the always-acquired centre (< 4 px) is outside the extent
CORRELATION_BANDS = ((4, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 230))
BLUR_SIGMAS = (0, 2, 4, 8)
UNIFORM_START_RANGE = (0.01, 0.99)
ADAM_EPS = 1e-8
EPOCH_LINE = re.compile(r'Epoch = \[\s*(\d+)/\s*\d+\] TrainLoss = (\S+)')
BRAIN_VAL_LINE = re.compile(r'\[brain\] .*?PSNR=(\S+)')
TRAIN_SET_LINE = re.compile(r"Manifest train set \(volumes, slices\): \{'brain': \((\d+), (\d+)\)\}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, required=True, help='report folder holding eval_per_slice.pt')
    parser.add_argument('--queue', type=Path, required=True, help='folder holding <arm>_s<seed>/ run folders')
    parser.add_argument('--shared-run', type=Path, required=True, help='run folder whose best_model.pt holds S')
    parser.add_argument('--sections', nargs='+', choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument('--out', type=Path, default=None, help='default: <report>/diagnostics')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=32, help='torch CPU threads')
    return parser.parse_args()


def checkpoint_of(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def brain_w(checkpoint):
    """The brain mask's w (p = sigmoid(slope * w)) of a loupe_2d checkpoint, shared or one mask per anatomy."""
    w = checkpoint['model']['sampler.mask_net.mask'][..., 0].float()
    classes = list(getattr(checkpoint['options'], 'label_classes', None) or [])
    return w[classes.index('brain')] if w.shape[0] > 1 else w[0]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def uniform_start(checkpoint, seed):
    """A loupe_uniform run's starting brain w: train.py seeds, then builds the model. The loaders it builds in between
    draw from no global random state, so the start is reproduced exactly; the knee rows, which brain-only training
    never updates, show it."""
    seed_everything(seed)
    start = build_model(checkpoint['options']).sampler.mask_net.mask.detach()[..., 0].float()
    end = checkpoint['model']['sampler.mask_net.mask'][..., 0].float()
    brain = list(checkpoint['options'].label_classes).index('brain')
    knee_rows_reproduced = all(torch.equal(start[i], end[i]) for i in range(len(start)) if i != brain)
    return start[brain], knee_rows_reproduced


def quantiles(x, qs):
    return [float(v) for v in torch.quantile(x.flatten().float(), torch.tensor(qs))]


def entropy_bits(p):
    p = p.clamp(1e-12, 1 - 1e-12)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def radius(shape):
    rows, cols = shape
    return torch.hypot((torch.arange(rows) - rows // 2).float()[:, None], (torch.arange(cols) - cols // 2).float()[None, :])


def blur(image, sigma):
    if sigma == 0:
        return image
    x = torch.arange(-3 * sigma, 3 * sigma + 1).float()
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    image = F.conv2d(image[None, None], kernel.view(1, 1, 1, -1), padding=(0, len(x) // 2))
    return F.conv2d(image, kernel.view(1, 1, -1, 1), padding=(len(x) // 2, 0))[0, 0]


def correlation(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])


def run_names():
    return [f'{arm}_s{seed}' for seed in SEEDS for arm in ARMS]


def train_loss(args):
    result = {}
    for name in run_names():
        loss, brain_psnr, pending = {}, {}, None
        for line in (args.queue / name / 'train_sequential.log').read_text(errors='replace').splitlines():
            if (match := BRAIN_VAL_LINE.search(line)):
                pending = float(match.group(1))
            elif (match := EPOCH_LINE.search(line)):
                epoch = int(match.group(1))
                loss[epoch] = float(match.group(2))
                if pending is not None:
                    brain_psnr[epoch], pending = pending, None
        result[name] = {'loss_at_epoch': {e: loss[e] for e in LOSS_EPOCHS},
                        'relative_drop_50_to_58': 1 - loss[58] / loss[50],
                        'relative_drop_58_to_65': 1 - loss[65] / loss[58],
                        'trainer_brain_val_psnr': {e: brain_psnr[e] for e in VAL_EPOCHS if e in brain_psnr}}
        print(f'{name:20s} loss ' + ' '.join(f'{loss[e]:.5f}' for e in LOSS_EPOCHS)
              + f"  drop 50-58 {result[name]['relative_drop_50_to_58']:.1%}  58-65 {result[name]['relative_drop_58_to_65']:.1%}"
              + '  brain val PSNR ' + ' '.join(f'{v:.2f}' for v in result[name]['trainer_brain_val_psnr'].values()))
    return result


def reference(args):
    loader = full_brain_val_loader(args.queue / 'static_s42', args.chunk, None)
    model, checkpoint = load(args.shared_run, 'best_model.pt', args.device)
    options = checkpoint['options']
    scores, _ = score_run(model, list(options.label_classes), loader, args.device)
    result = {'checkpoint': str(args.shared_run / 'best_model.pt'), 'epoch': int(checkpoint['epoch']), 'device': args.device,
              'label_classes': list(options.label_classes), 'optimizer': getattr(options, 'optimizer', None),
              'num_epochs': getattr(options, 'num_epochs', None), 'n_slices': int(len(scores['psnr'])),
              **{m: float(v.mean()) for m, v in scores.items()}}
    print('reference (S run, best_model.pt):', {k: result[k] for k in ('epoch', 'optimizer', 'psnr', 'ssim', 'nmse', 'fraction')})
    return result


def adam_reach(args, name):
    """Steps per epoch times the mask lr, summed over the step schedule: about the most Adam can move any w."""
    options = checkpoint_of(args.queue / name / 'last_model.pt')['options']
    if options.lr_scheduler != 'step' or getattr(options, 'sampler_lr', None) is not None:
        raise ValueError(f'{name}: expected the step schedule and the mask at the base lr')
    slices = int(TRAIN_SET_LINE.search((args.queue / f'{name}.log').read_text(errors='replace')).group(2))
    steps = -(-slices // options.batch_size)
    lr_by_epoch = [options.lr * options.lr_gamma ** (epoch // options.lr_step_size) for epoch in range(options.num_epochs)]
    return {'train_slices': slices, 'steps_per_epoch': steps, 'lr': options.lr, 'lr_step_size': options.lr_step_size,
            'lr_gamma': options.lr_gamma, 'num_epochs': options.num_epochs, 'reach_w': steps * sum(lr_by_epoch)}


def masks(args):
    maps = {k: v.float() for k, v in torch.load(args.report / 'eval_per_slice.pt', weights_only=False)['maps'].items()}
    extent = torch.stack([maps[f'loupe_uniform_s{s}'] > 0 for s in SEEDS]).any(0)
    shared = checkpoint_of(args.shared_run / 'best_model.pt')
    w_shared, slope = brain_w(shared), shared['options'].slope
    dead = (torch.sigmoid(slope * w_shared) < DEAD_P) & extent
    alive = extent & ~dead
    r = radius(extent.shape)
    result = {'extent_pixels': int(extent.sum()), 'dead_p_threshold': DEAD_P, 'dead_share_of_extent': float(dead.sum() / extent.sum()),
              'shared_logit_quantiles_0_10_50_90_100': {
                  'dead': quantiles(slope * w_shared[dead], [0, .1, .5, .9, 1]),
                  'alive': quantiles(slope * w_shared[alive], [0, .1, .5, .9, 1])},
              'adam_reach': adam_reach(args, 'loupe_shared_s42'), 'runs': {}}
    for seed in SEEDS:
        for arm in ARMS:
            name = f'{arm}_s{seed}'
            checkpoint = checkpoint_of(args.queue / name / 'last_model.pt')
            w = brain_w(checkpoint)
            if arm == 'static':
                entry = {'max_abs_dw_from_S': float((w - w_shared).abs().max())}
            elif arm == 'loupe_shared':
                dw = w - w_shared
                entry = {'abs_dw_alive_q50_90_99_100': quantiles(dw[alive].abs(), [.5, .9, .99, 1]),
                         'dw_dead_q0_50_90_100': quantiles(dw[dead], [0, .5, .9, 1]),
                         'max_p_dead': float(torch.sigmoid(slope * w[dead]).max())}
            else:
                start, reproduced = uniform_start(checkpoint, seed)
                band_corr = {f'{lo}-{hi}': correlation(start[extent & (r >= lo) & (r < hi)], w[extent & (r >= lo) & (r < hi)])
                             for lo, hi in CORRELATION_BANDS}
                entry = {'knee_rows_reproduced': reproduced, 'corr_start_end': correlation(start[extent], w[extent]),
                         'corr_start_end_by_radius': band_corr,
                         'abs_dw_q50_90_99_100': quantiles((w - start)[extent].abs(), [.5, .9, .99, 1]),
                         'share_outside_start_range': float((w[extent].abs() > start[extent].abs().max()).float().mean())}
            result['runs'][name] = entry
    names = run_names()
    result['entropy_bits'] = {name: float(entropy_bits(maps[name])[extent].mean()) for name in names}
    result['relocated'] = {a: {b: budget_relocated(maps[a], maps[b]) for b in names} for a in names}
    result['relocated_between_seeds'] = {arm: {f's{s}-s{t}': budget_relocated(maps[f'{arm}_s{s}'], maps[f'{arm}_s{t}'])
                                               for s, t in itertools.combinations(SEEDS, 2)} for arm in ARMS}
    result['radial_density'] = {'edges_px': list(RADIUS_EDGES), **{
        name: [float(maps[name][extent & (r >= lo) & (r < hi)].mean()) for lo, hi in zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:])]
        for name in names}}
    pairs = [(f'{arm}_s{s}', f'{arm}_s{t}') for arm in ('loupe_shared', 'loupe_uniform') for s, t in itertools.combinations(SEEDS, 2)]
    result['seed_map_correlation'] = {'sigmas_px': list(BLUR_SIGMAS), **{
        f'{a} vs {b}': [correlation(blur(maps[a], s)[extent], blur(maps[b], s)[extent]) for s in BLUR_SIGMAS] for a, b in pairs}}
    print(f"extent {result['extent_pixels']} px, S dead share {result['dead_share_of_extent']:.1%}, "
          f"Adam reach {result['adam_reach']['reach_w']:.2f} in w")
    for name, entry in result['runs'].items():
        print(f'  {name:20s} entropy {result["entropy_bits"][name]:.3f} bits ', json.dumps(entry))
    print('relocated between seeds:', json.dumps(result['relocated_between_seeds']))
    return result


@torch.no_grad()
def mean_brain_map(model, brain, kspaces):
    total, count = 0.0, 0
    for kspace in kspaces:
        old, valid = model._initial_masks(kspace)
        labels = torch.full((len(kspace),), brain)
        unscaled, _, _ = model.sampler.unscaled_probabilities(
            torch.zeros(len(kspace), kspace.shape[2], kspace.shape[3], 2), old, anatomy_labels=labels)
        rescaled, _ = model.sampler.select_locations(unscaled, old, model.mask_budget(old, valid), valid)
        total, count = total + rescaled[:, 0].sum(0), count + len(kspace)
    return total / count


def model_from(checkpoint, brain_w_override=None):
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    brain = list(checkpoint['options'].label_classes).index('brain')
    if brain_w_override is not None:
        with torch.no_grad():
            model.sampler.mask_net.mask[brain, ..., 0] = brain_w_override
    return model, brain


def trajectory(args):
    kspaces = [batch['kspace'] for batch in full_brain_val_loader(args.queue / 'static_s42', args.chunk, 64)]
    shared = checkpoint_of(args.shared_run / 'best_model.pt')
    w_shared, slope = brain_w(shared), shared['options'].slope
    static_map = mean_brain_map(*model_from(checkpoint_of(args.queue / 'static_s42' / 'last_model.pt')), kspaces)
    extent = static_map > 0
    r = radius(extent.shape)
    low, high = UNIFORM_START_RANGE

    def describe(epoch, p_map, w):
        p = torch.sigmoid(slope * w)[extent]
        return {'epoch': epoch, 'centre_4_16': float(p_map[extent & (r >= 4) & (r < 16)].mean()),
                'outer_128_plus': float(p_map[extent & (r >= 128)].mean()),
                'relocated_vs_S': budget_relocated(p_map, static_map),
                'share_p_outside_start_range': float(((p < low) | (p > high)).float().mean())}

    result = {}
    for arm in ('loupe_shared', 'loupe_uniform'):
        for seed in SEEDS:
            name = f'{arm}_s{seed}'
            last = checkpoint_of(args.queue / name / 'last_model.pt')
            start_w = w_shared if arm == 'loupe_shared' else uniform_start(last, seed)[0]
            snapshots, maps = [], []
            for epoch, checkpoint, override in ((0, last, start_w), (None, checkpoint_of(args.queue / name / 'best_model.pt'), None),
                                                (None, last, None)):
                model, brain = model_from(checkpoint, override)
                maps.append(mean_brain_map(model, brain, kspaces))
                epoch = int(checkpoint['epoch']) + 1 if epoch is None else epoch
                snapshots.append(describe(epoch, maps[-1], model.sampler.mask_net.mask[brain, ..., 0].detach()))
            moved = [{'epochs': f"{a['epoch']}-{b['epoch']}", 'relocated': budget_relocated(mb, ma)}
                     for a, b, ma, mb in zip(snapshots[:-1], snapshots[1:], maps[:-1], maps[1:])]
            result[name] = {'snapshots': snapshots, 'moved': moved}
            print(f'{name:20s}', ' | '.join(f"ep{s['epoch']}: centre {s['centre_4_16']:.2f} outer {s['outer_128_plus']:.3f} "
                                            f"vs S {s['relocated_vs_S']:.2f} sat {s['share_p_outside_start_range']:.3f}"
                                            for s in snapshots),
                  '| moved', ', '.join(f"{m['epochs']}: {m['relocated']:.3f}" for m in moved))
    return result


def gradient_summary(model, brain, batches, dead, alive, device):
    model.to(device).train()
    model.requires_grad_(False)
    mask = model.sampler.mask_net.mask
    mask.requires_grad_(True)
    grads = []
    for index, batch in enumerate(batches):
        torch.manual_seed(index)
        kspace, target = batch['kspace'].to(device), batch['target'].to(device)
        labels = torch.full((len(kspace),), brain, dtype=torch.long, device=device)
        output = model(target, kspace, anatomy_labels=labels)['output']
        output = output[-1] if isinstance(output, list) else output
        target = target if target.dim() == output.dim() else target.unsqueeze(1)
        grads.append(torch.autograd.grad(F.l1_loss(output, target), mask)[0][brain, ..., 0].float().cpu())
    model.cpu()
    grads = torch.stack(grads)
    mean_abs, mean = grads.abs().mean(0), grads.mean(0)
    consistency = mean.abs() / mean_abs.clamp_min(1e-30)
    return {region: {'abs_grad_q50_90_99': quantiles(mean_abs[where], [.5, .9, .99]),
                     'share_above_adam_eps': float((mean_abs[where] > ADAM_EPS).float().mean()),
                     'median_sign_consistency': float(consistency[where].median()),
                     'share_descent_raises_p': float((mean[where] < 0).float().mean())}
            for region, where in (('dead', dead), ('alive', alive))}


def dead_gradient(args):
    maps = torch.load(args.report / 'eval_per_slice.pt', weights_only=False)['maps']
    extent = torch.stack([maps[f'loupe_uniform_s{s}'].float() > 0 for s in SEEDS]).any(0)
    shared = checkpoint_of(args.shared_run / 'best_model.pt')
    w_shared = brain_w(shared)
    dead = (torch.sigmoid(shared['options'].slope * w_shared) < DEAD_P) & extent
    alive = extent & ~dead
    batches = list(itertools.islice(brain_loader(args.queue / 'loupe_shared_s42', 'train', 4, None), 6))
    end = checkpoint_of(args.queue / 'loupe_shared_s42' / 'last_model.pt')
    result = {'batches': len(batches), 'slices_per_batch': 4, 'adam_eps': ADAM_EPS, 'device': args.device,
              'end_of_loupe_shared_s42': gradient_summary(*model_from(end), batches, dead, alive, args.device)}
    seed_everything(42)
    start = build_model(end['options'])
    brain = list(end['options'].label_classes).index('brain')
    with torch.no_grad():
        start.sampler.mask_net.mask[:, ..., 0] = w_shared
    result['start_random_varnet_with_S'] = gradient_summary(start, brain, batches, dead, alive, args.device)
    for state in ('end_of_loupe_shared_s42', 'start_random_varnet_with_S'):
        print(state, json.dumps(result[state]))
    return result


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    out = args.out or args.report / 'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    for section in args.sections:
        print(f'=== {section}', flush=True)
        result = globals()[section](args)
        (out / f'{section}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
