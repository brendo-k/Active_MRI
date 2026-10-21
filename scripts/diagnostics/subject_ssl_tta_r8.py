"""Deployable per-subject VarNet adaptation: zero-shot self-supervised fine-tuning on the acquired k-space only (R=8).

Starts from the LOUPE run's VarNet and static mask. Each slice is acquired once (seeded draw 0, the same draw for every
arm), giving the acquired set Omega. A fixed `--gamma` share of Omega's non-centre points is held out (Gamma) to pick
the stopping step (0 disables it); the rest is the training pool. Inference always uses all of Omega. No ground truth
is used for fitting or for stopping. Two methods (`--method`):

  ssdu   ZS-SSL (Yaman et al., ICLR 2022): every step splits the pool at random into a loss set Lambda (`--rho` of it)
         and the data-consistency input Theta (the rest plus the calibration centre). Theta is sparser than Omega, so
         the VarNet is trained at a higher acceleration than it is used at.
  echo   ECHO (Shang et al.): the current VarNet, without gradients, estimates the full k-space from the pool; the
         acquired values replace the estimate at the pool (the hybrid observation); a fresh mask from the acquisition
         mask's own distribution (another LOUPE draw, same R) re-undersamples the hybrid as the input, and the loss
         is on the acquired pool (`--echo-loss acquired`) or on the pool points the new mask left out (`unseen`).
         Input and inference masks share one distribution, so there is no acceleration mismatch. Both losses score
         points whose values shaped the estimate; `heldout` removes a `--rho` share Lambda of the pool from the
         estimator's input and from the hybrid's acquired part, and scores only Lambda.

Arms, scored on the same held-out half H of each validation volume as subject_coadapt_ceiling_r8.py:

  base        LOUPE VarNet
  ssl_vol     one VarNet per volume, fitted on every slice of the volume (F and H: all acquired data)
  ssl_slice   one VarNet per slice, fitted on that slice alone (first `--slice-count` slices of H)
  ssl_ctrl    the same fit on as many training slices of the same anatomy from other subjects

Each fitted arm is scored at every `--eval-at` step and at the step with the lowest Gamma loss (`@stop`).

    python scripts/diagnostics/subject_ssl_tta_r8.py --anatomy brain --volumes 0:4
"""
import argparse
import copy
import json
import time
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from mask_gradient_anatomy_r8 import load
from subject_coadapt_ceiling_r8 import (ANATOMY_INDEX, LOUPE_RUN, PROBABILITY_EPS, Slices, control_slices, evenly,
                                        masks_from_logits, volume_groups)

OUT_DIR = Path('reports/subject_coadaptation_r8/ssl')
ACQUISITION_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anatomy', choices=tuple(ANATOMY_INDEX), required=True)
    parser.add_argument('--volumes', default=':', help='start:stop over the anatomy\'s volumes')
    parser.add_argument('--max-score', type=int, default=8, help='at most this many H slices (evenly spaced)')
    parser.add_argument('--max-fit', type=int, default=16, help='control slices drawn (matches the ceiling\'s F)')
    parser.add_argument('--slice-count', type=int, default=4, help='H slices that get their own per-slice fit')
    parser.add_argument('--batch', type=int, default=4, help='slices per step')
    parser.add_argument('--splits', type=int, default=2, help='Theta/Lambda splits (ssdu) or new masks (echo) per slice per step')
    parser.add_argument('--method', choices=('ssdu', 'echo'), default='ssdu')
    parser.add_argument('--echo-loss', choices=('acquired', 'unseen', 'heldout'), default='acquired')
    parser.add_argument('--train-cascades', type=int, default=0, help='fine-tune only the last N VarNet cascades (0: all)')
    parser.add_argument('--rho', type=float, default=0.4, help='share of the training pool used as the loss set')
    parser.add_argument('--gamma', type=float, default=0.15, help='share of acquired non-centre points held out to stop')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--loss-space', choices=('image', 'coil'), default='image')
    parser.add_argument('--freeze-sens', action='store_true', help='fine-tune the VarNet only, not the sensitivity net')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--eval-at', default='50,100,200,300')
    parser.add_argument('--stop-every', type=int, default=25, help='Gamma loss checked every this many steps')
    parser.add_argument('--arms', default='base,ssl_vol,ssl_slice,ssl_ctrl')
    parser.add_argument('--tag', default='', help='suffix of the output folder')
    parser.add_argument('--checkpoint-cascades', action='store_true', help='recompute VarNet cascades to save memory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    return parser.parse_args()


class Acquisition:
    """Per slice: the acquired mask Omega (one seeded draw), the stop holdout Gamma, and the calibration centre."""

    def __init__(self, model, logits, slices, gamma, seed):
        self.model, self.logits, self.slices, self.gamma, self.seed = model, logits, slices, gamma, seed
        self.cache = {}

    @torch.no_grad()
    def get(self, local):
        if local not in self.cache:
            part = self.slices.batch([local])
            torch.manual_seed(ACQUISITION_SEED * 100003 + self.slices.global_index(local))
            omega = masks_from_logits(self.model, self.logits, part['kspace'])
            centre, _ = self.model._initial_masks(part['kspace'])
            generator = torch.Generator(device=omega.device).manual_seed(self.seed + self.slices.global_index(local))
            outer = omega * (1 - centre)
            gamma = outer * (torch.rand(outer.shape, generator=generator, device=omega.device) < self.gamma)
            self.cache[local] = {'omega': omega, 'gamma': gamma, 'centre': centre}
        return self.cache[local]

    def stacked(self, locals_, key):
        return torch.cat([self.get(i)[key] for i in locals_])


def last_cascade_kspace(model):
    """The VarNet's per-coil k-space output, captured by a hook on its last cascade (installed once per model)."""
    if not hasattr(model, '_captured_kspace'):
        model._captured_kspace = {}
        model.reconstructor.cascades[-1].register_forward_hook(
            lambda module, inputs, output: model._captured_kspace.__setitem__('k', output))
    return model._captured_kspace


def ssdu_step_loss(model, part, theta, loss_mask, loss_space, measured_part=None):
    """SSDU loss on `loss_mask`: on the image re-projected through the sense maps (`image`, the trainer's
    ssdu_loss), or on the VarNet's own per-coil k-space (`coil`), which the sense-map model does not floor.
    The input is `theta` of part['kspace']; the measurements compared are those of `measured_part` (default `part`)."""
    measured_part = measured_part or part
    captured = last_cascade_kspace(model)
    output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=theta)
    target_dict = {'holdout_kspace': measured_part['kspace'], 'holdout_mask': loss_mask}
    if loss_space == 'image':
        return model.ssdu_loss(output, target_dict).mean()
    measured = torch.view_as_complex(measured_part['kspace'].contiguous())
    error = ((captured['k'] - measured) * loss_mask).abs().flatten(1)
    reference = (measured * loss_mask).abs().flatten(1)
    relative_l2 = error.pow(2).sum(1).sqrt() / reference.pow(2).sum(1).sqrt().clamp(min=1e-12)
    relative_l1 = error.sum(1) / reference.sum(1).clamp(min=1e-12)
    return (relative_l2 + relative_l1).mean()


def echo_step_loss(model, part, pool, centre, logits, echo_loss, rho):
    """ECHO: hybrid of the acquired pool and the VarNet's own no-grad estimate, re-undersampled by a fresh draw of
    the acquisition mask distribution; coil-space loss against the acquired measurements."""
    captured = last_cascade_kspace(model)
    measured = torch.view_as_complex(part['kspace'].contiguous())
    _, valid = model._initial_masks(part['kspace'])
    held_out = (pool - centre) * (torch.rand_like(pool) < rho) if echo_loss == 'heldout' else torch.zeros_like(pool)
    observed = pool - held_out
    with torch.no_grad():
        model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=observed)
        hybrid = observed * measured + (1 - observed) * valid * captured['k']
        secondary = masks_from_logits(model, logits, part['kspace'])
    loss_mask = {'acquired': pool - centre, 'unseen': (pool - centre) * (1 - secondary), 'heldout': held_out}[echo_loss]
    hybrid_part = dict(part, kspace=torch.view_as_real(hybrid))
    return ssdu_step_loss(model, hybrid_part, secondary, loss_mask, 'coil', measured_part=part)


@torch.no_grad()
def stop_loss(model, acquisition, locals_, loss_space):
    """Mean SSDU loss on Gamma with the rest of Omega as input: the stopping criterion (no ground truth)."""
    model.eval()
    total = 0.0
    for local in locals_:
        part, a = acquisition.slices.batch([local]), acquisition.get(local)
        total += float(ssdu_step_loss(model, part, a['omega'] - a['gamma'], a['gamma'], loss_space)) / len(locals_)
    return total


@torch.no_grad()
def score(model, acquisition, slices, locals_):
    """Per-slice trainer-path PSNR with all of Omega as input."""
    model.eval()
    psnr = []
    for local in locals_:
        part = slices.batch([local])
        output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=acquisition.get(local)['omega'])
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        psnr.append(float(compute_psnr_torch(*_apply_signal_mask(output['output'], target, part['metric_mask'])).flatten()[0]))
    return psnr


def ssl_fit(base, acquisition, fit_locals, score_fn, args):
    """ZS-SSL on `fit_locals`; returns {'@step': scores} for every eval step and the Gamma-selected stop."""
    model = copy.deepcopy(base)
    first_cascade = len(model.reconstructor.cascades) - args.train_cascades if args.train_cascades else 0
    trained_cascades = tuple(f'reconstructor.cascades.{i}.' for i in range(first_cascade, len(model.reconstructor.cascades)))
    parameters = [p for name, p in model.named_parameters()
                  if name.startswith(trained_cascades) or (name.startswith('sensetivity_net.') and not args.freeze_sens)]
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / args.warmup))
    eval_at = {int(s) for s in args.eval_at.split(',')}
    generator = np.random.RandomState(args.seed)
    scores, curve = {}, []
    best = (stop_loss(model, acquisition, fit_locals, args.loss_space), 0, None)
    curve.append((0, best[0]))
    for step in range(1, args.steps + 1):
        model.train()
        model.sampler.eval()
        chosen = generator.choice(fit_locals, size=min(args.batch, len(fit_locals)), replace=False).tolist()
        part = acquisition.slices.batch(chosen, copies=args.splits)
        omega = acquisition.stacked(chosen, 'omega').repeat(args.splits, 1, 1, 1)
        pool = omega - acquisition.stacked(chosen, 'gamma').repeat(args.splits, 1, 1, 1)
        centre = acquisition.stacked(chosen, 'centre').repeat(args.splits, 1, 1, 1)
        optimizer.zero_grad()
        if args.method == 'echo':
            loss = echo_step_loss(model, part, pool, centre, acquisition.logits, args.echo_loss, args.rho)
        else:
            loss_mask = (pool - centre) * (torch.rand_like(pool) < args.rho)
            loss = ssdu_step_loss(model, part, pool - loss_mask, loss_mask, args.loss_space)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        if step % args.stop_every == 0:
            current = stop_loss(model, acquisition, fit_locals, args.loss_space)
            curve.append((step, current))
            if current < best[0]:
                best = (current, step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if step in eval_at:
            scores[f'@{step}'] = score_fn(model)
    if best[2] is not None:
        model.load_state_dict(best[2])
    scores['@stop'] = score_fn(model)
    del model
    torch.cuda.empty_cache()
    return scores, {'stop_step': best[1], 'stop_curve': curve}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    base, _ = load(LOUPE_RUN, 'last_model.pt', args.device)
    base.reconstructor.checkpoint_cascades = args.checkpoint_cascades
    cfg = OmegaConf.load(LOUPE_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    train_loader, val_loader = build_dataloaders(cfg)
    a = ANATOMY_INDEX[args.anatomy]
    val_part, train_part = val_loader.dataset.datasets[a], train_loader.dataset.datasets[a]
    val = Slices(val_part, sum(len(d) for d in val_loader.dataset.datasets[:a]), val_loader.collate_fn, args.device)
    train = Slices(train_part, 10 ** 6, train_loader.collate_fn, args.device)  # offset keeps its seeds apart from val

    with torch.no_grad():
        unscaled = base.propose_mask(val.batch([0])['kspace'])['unscaled']
    logits = torch.logit(unscaled.clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS))
    val_acq = Acquisition(base, logits, val, args.gamma, args.seed)
    train_acq = Acquisition(base, logits, train, args.gamma, args.seed)
    arms = args.arms.split(',')

    groups = volume_groups(val_part)
    start, _, stop = args.volumes.partition(':')
    names = list(groups)[int(start or 0):int(stop) if stop else None]
    out_dir = args.out_dir / (args.anatomy + (f'_{args.tag}' if args.tag else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = out_dir / f'{name}.json'
        if path.exists():
            print(f'skip {name}: done', flush=True)
            continue
        started = time.time()
        members = groups[name]
        fit_set, score_set = evenly(members[0::2], args.max_fit), evenly(members[1::2], args.max_score)
        rng = np.random.RandomState(args.seed + len(name) + sum(map(ord, name)))
        ctrl_set = control_slices(train_part, args.anatomy, name, len(fit_set), rng)
        result = {'volume': name, 'anatomy': args.anatomy, 'score': score_set, 'ctrl': ctrl_set,
                  'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  'psnr': {}, 'stop': {}}
        result['psnr']['base'] = score(base, val_acq, val, score_set)
        print(f'{name} base: {np.mean(result["psnr"]["base"]):.3f} dB', flush=True)

        def report(arm, scores, stop_info):
            result['stop'][arm] = stop_info
            for key, values in scores.items():
                result['psnr'][f'{arm}{key}'] = values
                print(f'{name} {arm}{key}: {np.mean(values) - np.mean(result["psnr"]["base"][:len(values)]):+.3f} dB vs base', flush=True)

        score_h = lambda m: score(m, val_acq, val, score_set)  # noqa: E731
        if 'ssl_vol' in arms:
            report('ssl_vol', *ssl_fit(base, val_acq, members, score_h, args))
        if 'ssl_ctrl' in arms:
            report('ssl_ctrl', *ssl_fit(base, train_acq, ctrl_set, score_h, args))
        if 'ssl_slice' in arms:
            per_slice, stops = {}, []
            for local in score_set[:args.slice_count]:
                scores, stop_info = ssl_fit(base, val_acq, [local], lambda m, local=local: score(m, val_acq, val, [local]), args)
                for key, values in scores.items():
                    per_slice.setdefault(key, []).extend(values)
                stops.append(stop_info)
            report('ssl_slice', per_slice, stops)
        result['seconds'] = time.time() - started
        path.write_text(json.dumps(result, indent=1) + '\n')
        print(f'{name} done in {result["seconds"] / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
