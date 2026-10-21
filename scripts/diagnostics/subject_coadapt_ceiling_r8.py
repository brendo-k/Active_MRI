"""Ground-truth ceiling for per-subject co-adaptation: one mask and one VarNet per volume (R=8, brain+knee).

Starting point: the LOUPE run's static mask and VarNet (`uo8d68n0`). Each validation volume's slices, in slice
order, are split into alternating halves: the even half F is fitted with the ground truth and the odd half H is scored,
so a fitted VarNet can never have seen the image it is scored on (`--split block` fits the first half and scores the
second, after a `--gap`, so the gain can be read against slice distance). Arms, all scored on H with common seeded draws:

  base       LOUPE mask, LOUPE VarNet
  mask       one free logit map per volume, fitted on F through the frozen VarNet
  mask_in    the same map fitted on F and H (a per-volume mask is applied to the whole scan, so this is the mask
             oracle a SPACE acquisition could use); frozen VarNet
  recon      the VarNet + sensitivity net fine-tuned on F with the LOUPE mask
  joint      mask and VarNet fitted together on F
  recon_on_mask_in  the VarNet fine-tuned on F with the `mask_in` map held fixed (needs `mask_in` earlier in --arms):
             does a co-adapted VarNet amplify the per-volume mask oracle?
  param, grid  a few-parameter per-volume density change, the pattern family an ACS -> parameters predictor would
             output (DensityFamily), fitted on F through the frozen VarNet. `*_in` fits F and H; `*_h` fits H alone and
             is not scored (its parameters, against the F fit's, give the reliability of a volume's parameters).

Every fitted arm has a control (`*_ctrl`): the same fit on as many training slices of the same anatomy from other
subjects. `recon_generic` is plain continued training: the same steps on random slices of the whole training set. The base VarNet is a 50-epoch model, so any fine-tune can gain from extra training alone; the
subject-specific gain is arm minus its control. Fine-tuned arms are scored after every `--eval-at` step count.

    python scripts/diagnostics/subject_coadapt_ceiling_r8.py --anatomy brain --volumes 0:4
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
import torch.nn.functional as F
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _apply_signal_mask
from activemri.utils.util import compute_psnr_torch
from label_curriculum_r8 import gaussian_blur
from mask_gradient_anatomy_r8 import load, to_device

LOUPE_RUN = Path('checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05')
OUT_DIR = Path('reports/subject_coadaptation_r8')
LOGITS_DIR = Path('/tmp/bkadota_subject_coadapt')
LOSS_WEIGHTS = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
SCORING_SEEDS = (0, 1, 2, 3)
PROBABILITY_EPS = 1e-4
ANATOMY_INDEX = {'brain': 0, 'knee': 1}
FAMILY_EPS = 1e-6
TILT_RADII = (8.0, 64.0)
FAMILY_ARMS = {'': 'fit', '_in': 'fit+score', '_ctrl': 'ctrl', '_h': 'score'}
TRACE_EVERY = 25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anatomy', choices=tuple(ANATOMY_INDEX), required=True)
    parser.add_argument('--volumes', default=':', help='start:stop over the anatomy\'s volumes')
    parser.add_argument('--reverse', action='store_true',
                        help='work through the volumes last first, so a second process can share a range')
    parser.add_argument('--max-fit', type=int, default=16, help='at most this many F slices (evenly spaced)')
    parser.add_argument('--max-score', type=int, default=8, help='at most this many H slices (evenly spaced)')
    parser.add_argument('--batch', type=int, default=4, help='slices per step')
    parser.add_argument('--draws', type=int, default=2, help='Bernoulli draws per slice per step')
    parser.add_argument('--mask-steps', type=int, default=150)
    parser.add_argument('--mask-lr', type=float, default=0.05)
    parser.add_argument('--param-lr', type=float, default=0.02, help='Adam lr of the param and grid families')
    parser.add_argument('--envelope-sigma', type=float, default=8.0, help='param family: blur of the base density, px')
    parser.add_argument('--grid-size', type=int, default=8, help='grid family: offset grid side')
    parser.add_argument('--recon-lr', type=float, default=1e-4)
    parser.add_argument('--recon-warmup', type=int, default=50, help='steps of linear lr warmup for the VarNet')
    parser.add_argument('--eval-at', default='100,200,400', help='fine-tune step counts at which H is scored')
    parser.add_argument('--arms', default='base,mask,mask_in,mask_ctrl,recon,recon_ctrl,joint,joint_ctrl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=('alternate', 'block'), default='alternate',
                        help='alternate: F even, H odd slices; block: F the first half, H the second half after a gap')
    parser.add_argument('--gap', type=int, default=2, help='block split: slices skipped between F and H')
    parser.add_argument('--resume-adam', action='store_true', help="start the VarNet's Adam moments from the checkpoint")
    parser.add_argument('--tag', default='', help='suffix of the output folder')
    parser.add_argument('--checkpoint-cascades', action='store_true', help='recompute VarNet cascades to save memory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    return parser.parse_args()


def volume_groups(dataset):
    """{volume name: dataset-local indices in slice order}."""
    groups = {}
    for local, (path, slice_id) in enumerate(dataset.examples):
        groups.setdefault(Path(path).name, []).append((slice_id, local))
    return {name: [local for _, local in sorted(members)] for name, members in groups.items()}


def evenly(indices, count):
    if len(indices) <= count:
        return list(indices)
    return [indices[i] for i in np.linspace(0, len(indices) - 1, count).round().astype(int)]


class Slices:
    """Dataset items of one anatomy (cached unless `cache=False`), collated on demand into device batches."""

    def __init__(self, dataset, offset, collate, device, cache=True):
        self.dataset, self.offset, self.collate, self.device = dataset, offset, collate, device
        self.cache = {} if cache else None

    def item(self, local):
        if self.cache is None:
            return self.dataset[local]
        if local not in self.cache:
            self.cache[local] = self.dataset[local]
        return self.cache[local]

    def batch(self, locals_, copies=1):
        part = to_device(self.collate([dict(self.item(i)) for i in locals_]), self.device)  # collate pops keys
        if copies > 1:
            part = {k: (v.repeat(copies, *([1] * (v.dim() - 1))) if v is not None else None) for k, v in part.items()}
        return part

    def global_index(self, local):
        return self.offset + local


def masks_from_logits(model, logits, kspace):
    old_mask, valid_mask = model._initial_masks(kspace)
    probability = torch.sigmoid(logits).expand(len(kspace), -1, -1, -1)
    return model.sampler.select_locations(probability, old_mask, model.budget, valid_mask)[1]


class DensityFamily:
    """Per-volume density change on the base mask: the logits of p_base * exp(delta(theta)), exactly the base at
    theta = 0 (a density ratio, so the base's pixel structure is kept and no budget moves between rings by itself).

      param  delta = log E(W k) - log E(k) + tilt * log((1 + |k| / 8) / (1 + 64 / 8)). E is the base density blurred
             by `envelope_sigma` px, k the offset from the k-space centre, W = exp(scale) * expm([[a, b], [b, -a]]):
             theta = (scale, a, b, tilt) is a radial scale, an anisotropy along any axis, and a radial density tilt.
      grid   delta = a `grid_size` x `grid_size` grid, bicubically upsampled.
    """

    def __init__(self, kind, base_logits, envelope_sigma, grid_size):
        self.kind, self.base_logits = kind, base_logits
        self.base_probability = torch.sigmoid(base_logits)
        self.size = tuple(base_logits.shape[-2:])
        self.shape = (4,) if kind == 'param' else (1, 1, grid_size, grid_size)
        if kind == 'param':
            height, width = self.size
            rows = torch.arange(height, device=base_logits.device) - height // 2
            cols = torch.arange(width, device=base_logits.device) - width // 2
            self.offsets = torch.stack(torch.meshgrid(rows, cols, indexing='ij'), dim=-1).float()
            self.log_envelope = torch.log(gaussian_blur(self.base_probability, envelope_sigma).clamp(min=FAMILY_EPS))
            self.unwarped = self.warped_log_envelope(torch.eye(2, device=base_logits.device))
            inner, outer = TILT_RADII
            self.tilt = torch.log((1 + self.offsets.norm(dim=-1) / inner) / (1 + outer / inner))

    def init(self):
        return torch.zeros(self.shape, device=self.base_logits.device)

    def warped_log_envelope(self, matrix):
        """log E sampled at W k (bilinear, border), as grid_sample's align_corners=False coordinates."""
        height, width = self.size
        warped = self.offsets @ matrix.T
        rows = (warped[..., 0] + height // 2 + 0.5) / height * 2 - 1
        cols = (warped[..., 1] + width // 2 + 0.5) / width * 2 - 1
        grid = torch.stack([cols, rows], dim=-1)[None]
        return F.grid_sample(self.log_envelope, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def delta(self, theta):
        if self.kind == 'grid':
            return F.interpolate(theta, size=self.size, mode='bicubic', align_corners=True)
        scale, a, b, tilt = theta
        generator = torch.stack([torch.stack([a, b]), torch.stack([b, -a])])
        matrix = torch.exp(scale) * torch.matrix_exp(generator)
        return self.warped_log_envelope(matrix) - self.unwarped + tilt * self.tilt

    def __call__(self, theta):
        base = self.base_probability
        scaled = (base * torch.exp(self.delta(theta))).clamp(FAMILY_EPS, 1 - FAMILY_EPS)
        logit_change = (torch.log(scaled) - torch.log(base)) - (torch.log1p(-scaled) - torch.log1p(-base))
        return self.base_logits + logit_change


def recon_loss(model, loss_type, part, mask):
    output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)
    target_dict = {'target': part['target'], 'kspace': part['kspace']}
    if part['label'] is not None:
        target_dict['anatomy_labels'] = part['label']
    return model.loss(output, target_dict, LOSS_WEIGHTS, loss_type)[0]


def recon_parameters(model):
    return [p for name, p in model.named_parameters() if not name.startswith('sampler.')]


@torch.no_grad()
def score(model, logits, slices, locals_):
    """Per-slice trainer-path PSNR on `locals_`, averaged over SCORING_SEEDS draws keyed to the slice's global index,
    plus the share of the target's signal pixels outside the sense support (the mask-independent tail)."""
    model.eval()
    psnr, unsupported = [], []
    for local in locals_:
        part = slices.batch([local])
        total = 0.0
        for seed in SCORING_SEEDS:
            torch.manual_seed(seed * 100003 + slices.global_index(local))
            output = model(part['target'], part['kspace'], anatomy_labels=part['label'],
                           fixed_mask=masks_from_logits(model, logits, part['kspace']))
            target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
            masked_output, masked_target = _apply_signal_mask(output['output'], target, part['metric_mask'])
            total += float(compute_psnr_torch(masked_output, masked_target).flatten()[0]) / len(SCORING_SEEDS)
        support = output['sense_maps'].abs().pow(2).sum(dim=1, keepdim=True) > 0
        signal = masked_target > 0
        unsupported.append(float((signal & ~support).sum() / signal.sum().clamp(min=1)))
        psnr.append(total)
    return psnr, unsupported


def fit(model, loss_type, logits, slices, locals_, steps, args, fit_mask, fit_recon, eval_at=(), on_eval=None,
        adam_state=None, to_logits=None, mask_lr=None, trace=None):
    """Adam on the volume logit map and/or the reconstructor over random `--batch` subsets of `locals_`. With
    `adam_state` ({parameter name: Adam state}), the VarNet's moments resume from the training run's optimizer.
    With `to_logits`, `logits` are the parameters of a mask family and `to_logits` maps them to the logit map;
    `trace` then collects them every TRACE_EVERY steps."""
    as_logits = to_logits or (lambda z: z)
    groups = []
    if fit_mask:
        logits = logits.clone().requires_grad_(True)
        groups.append({'params': [logits], 'lr': mask_lr or args.mask_lr})
    parameters = recon_parameters(model)
    for p in parameters:
        p.requires_grad_(fit_recon)
    if fit_recon:
        groups.append({'params': parameters, 'lr': args.recon_lr})
    optimizer = torch.optim.Adam(groups)
    if fit_recon and adam_state:
        for name, p in model.named_parameters():
            if name in adam_state and not name.startswith('sampler.'):
                optimizer.state[p] = {k: v.clone() if k == 'step' else v.to(p.device).clone() for k, v in adam_state[name].items()}
    # A fresh Adam moves every weight by ~lr on its first steps whatever the gradient; ramp the VarNet's lr so that
    # transient does not masquerade as a fine-tune effect. Resumed moments need no ramp.
    warmup = 1 if adam_state else args.recon_warmup
    ramps = [lambda step: 1.0] * fit_mask + [lambda step: min(1.0, (step + 1) / warmup)] * fit_recon
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, ramps)
    generator = np.random.RandomState(args.seed)
    for step in range(1, steps + 1):
        model.train() if fit_recon else model.eval()
        model.sampler.eval()
        chosen = generator.choice(locals_, size=min(args.batch, len(locals_)), replace=False).tolist()
        part = slices.batch(chosen, copies=args.draws)
        optimizer.zero_grad()
        loss = recon_loss(model, loss_type, part, masks_from_logits(model, as_logits(logits), part['kspace']))
        loss.backward()
        if fit_recon:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        if step in eval_at:
            on_eval(step, model, as_logits(logits).detach())
        if trace is not None and step % TRACE_EVERY == 0:
            trace.append([step] + logits.detach().flatten().tolist())
    for p in parameters:
        p.requires_grad_(False)
    model.eval()
    return logits.detach()


def control_slices(train_dataset, anatomy, exclude_volume, count, rng):
    """`count` training slices of the anatomy, from distinct volumes where possible (other subjects)."""
    groups = volume_groups(train_dataset)
    names = [n for n in groups if n != exclude_volume]
    rng.shuffle(names)
    chosen = [groups[n][rng.randint(len(groups[n]))] for n in names[:count]]
    return chosen


def main():
    args = parse_args()
    eval_at = tuple(int(s) for s in args.eval_at.split(','))
    arms = args.arms.split(',')
    torch.manual_seed(args.seed)
    base, checkpoint = load(LOUPE_RUN, 'last_model.pt', args.device)
    loss_type = checkpoint['options'].loss_type
    base.reconstructor.checkpoint_cascades = args.checkpoint_cascades
    adam_state = None
    if args.resume_adam:
        names = [name for name, _ in base.named_parameters()]
        saved = checkpoint['optimizer']['state']
        assert len(saved) == len(names) and all(saved[i]['exp_avg'].shape == p.shape for i, (_, p) in enumerate(base.named_parameters()))
        adam_state = {name: saved[i] for i, name in enumerate(names)}
    cfg = OmegaConf.load(LOUPE_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 0
    train_loader, val_loader = build_dataloaders(cfg)
    a = ANATOMY_INDEX[args.anatomy]
    val_part, train_part = val_loader.dataset.datasets[a], train_loader.dataset.datasets[a]
    val = Slices(val_part, sum(len(d) for d in val_loader.dataset.datasets[:a]), val_loader.collate_fn, args.device)
    train = Slices(train_part, 0, train_loader.collate_fn, args.device)
    train_uncached = Slices(train_part, 0, train_loader.collate_fn, args.device, cache=False)

    with torch.no_grad():
        unscaled = base.propose_mask(val.batch([0])['kspace'])['unscaled']
    base_logits = torch.logit(unscaled.clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS))
    families = {kind: DensityFamily(kind, base_logits, args.envelope_sigma, args.grid_size) for kind in ('param', 'grid')}
    for kind, family in families.items():
        assert torch.equal(family(family.init()), base_logits), f'{kind} family at zero is not the base mask'

    groups = volume_groups(val_part)
    start, _, stop = args.volumes.partition(':')
    names = list(groups)[int(start or 0):int(stop) if stop else None]
    if args.reverse:
        names = names[::-1]
    out_dir = args.out_dir / ((args.anatomy if args.split == 'alternate' else f'{args.anatomy}_block') + (f'_{args.tag}' if args.tag else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = out_dir / f'{name}.json'
        if path.exists():
            print(f'skip {name}: done', flush=True)
            continue
        started = time.time()
        members = groups[name]
        if args.split == 'alternate':
            fit_part, score_part = members[0::2], members[1::2]
        else:
            fit_part, score_part = members[:len(members) // 2], members[len(members) // 2 + args.gap:]
        fit_set, score_set = evenly(fit_part, args.max_fit), evenly(score_part, args.max_score)
        position = {local: i for i, local in enumerate(members)}
        distance = [min(abs(position[h] - position[f]) for f in fit_set) for h in score_set]
        rng = np.random.RandomState(args.seed + len(name) + sum(map(ord, name)))
        ctrl_set = control_slices(train_part, args.anatomy, name, len(fit_set), rng)
        result = {'volume': name, 'anatomy': args.anatomy, 'fit': fit_set, 'score': score_set, 'ctrl': ctrl_set,
                  'distance_to_fit': distance,
                  'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'psnr': {}}
        logits_out = {}

        def record(arm, model, logits):
            result['psnr'][arm], unsupported = score(model, logits, val, score_set)
            result.setdefault('unsupported', unsupported)
            delta = np.mean(result['psnr'][arm]) - np.mean(result['psnr']['base']) if 'base' in result['psnr'] else 0.0
            print(f'{name} {arm}: {np.mean(result["psnr"][arm]):.3f} dB ({delta:+.3f} vs base)', flush=True)

        record('base', base, base_logits)
        sets = {'fit': fit_set, 'fit+score': fit_set + score_set, 'ctrl': ctrl_set, 'score': score_set}
        for arm in arms:
            if arm == 'base':
                continue
            kind, _, rest = arm.partition('_')
            if kind in families:
                suffix = f'_{rest}' if rest else ''
                role = FAMILY_ARMS[suffix]
                trace = []
                theta = fit(base, loss_type, families[kind].init(), train if role == 'ctrl' else val, sets[role],
                            args.mask_steps, args, True, False, to_logits=families[kind], mask_lr=args.param_lr,
                            trace=trace)
                result.setdefault('theta', {})[arm] = theta.flatten().tolist()
                result.setdefault('theta_trace', {})[arm] = trace
                if role != 'score':
                    logits = families[kind](theta)
                    logits_out[arm] = logits.cpu()
                    record(arm, base, logits)
                continue
            if arm.endswith('_ctrl'):
                source, locals_ = train, ctrl_set
            elif arm == 'recon_on_mask_in':
                source, locals_ = val, fit_set
            elif arm.endswith('_generic'):
                source, locals_ = train_uncached, list(range(len(train_part)))
            else:
                source, locals_ = val, fit_set + score_set if arm == 'mask_in' else fit_set
            kind = 'recon' if arm == 'recon_on_mask_in' else arm.removesuffix('_ctrl').removesuffix('_generic').removesuffix('_in')
            start_logits = logits_out['mask_in'].float().to(args.device) if arm == 'recon_on_mask_in' else base_logits
            if kind == 'mask':
                logits = fit(base, loss_type, base_logits, source, locals_, args.mask_steps, args, True, False)
                logits_out[arm] = logits.cpu()
                record(arm, base, logits)
            else:
                model = copy.deepcopy(base)
                fitted = fit(model, loss_type, start_logits, source, locals_, max(eval_at), args,
                             fit_mask=(kind == 'joint'), fit_recon=True, eval_at=eval_at,
                             on_eval=lambda step, m, z, arm=arm: record(f'{arm}@{step}', m, z), adam_state=adam_state)
                if kind == 'joint':
                    logits_out[arm] = fitted.cpu()
                del model
                torch.cuda.empty_cache()
        for arm, logits in logits_out.items():
            result.setdefault('mask_corr_with_base', {})[arm] = float(torch.corrcoef(torch.stack([
                torch.sigmoid(logits.float()).flatten(), torch.sigmoid(base_logits.cpu()).flatten()]))[0, 1])
        result['seconds'] = time.time() - started
        path.write_text(json.dumps(result, indent=1) + '\n')
        torch.save({k: v.half() for k, v in logits_out.items()}, LOGITS_DIR / f'{args.anatomy}_{name}.pt')
        print(f'{name} done in {result["seconds"] / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
