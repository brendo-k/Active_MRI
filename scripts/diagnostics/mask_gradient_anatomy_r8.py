"""Where does the per-slice signal in a sampler's gradient go? (R=8, brain+knee, 1-step, frozen checkpoints)

Stage 1, gradients. On adjacent-slice pairs from training volumes, the gradient of each slice's L1 reconstruction
loss with respect to its own score map z (the logit the sampler emits; LOUPE: slope * w), through the trainer's
estimator: the sampler's sigmoid (clamp, leak and floor of the run), budget rescale, Bernoulli draw and
straight-through backward (floor of the run). `--draws` independent draws per slice, split into halves A and B.
The reconstructor can come from another run (`--recon-run`), at the sampler run's operating point.

The gradient of slice i on draw d is split as
    g_id = gbar + a_anatomy(i) + s_i + e_id
population mean, anatomy offset, slice-specific part (within anatomy) and draw noise. Energies of the non-noise
parts are estimated free of noise bias from cross-half products <x^A, x^B>. Neighbour sharing is the
disattenuated correlation of s between the two slices of a pair, against pairs from different volumes.

The same split is made for the gradient with respect to the binary mask m the reconstructor saw (the raw
"value of sampling here" signal before the estimator's sigmoid, clamp and straight-through factors), and located
by the sampler's own probability: centre, in band (0.01 < p < 0.99), saturated, dead, clamped.

Stage 2, transport (conv samplers only, own reconstructor). How the sampler network turns slice gradients into
score changes: the kernel J_j P J_i^T with P the checkpoint's Adam preconditioner (or 1, SGD), computed with
backward passes and forward-mode JVPs (see `transport`).

    python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag conv_last
    python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag conv_ep4 --sampler-ckpt resume_from_epoch4.pt
"""
import argparse
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.func import functional_call, jvp
from torch.utils.data import DataLoader, Subset

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import _signal_mask, build_model
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
from activemri.utils.mask_binarization import SIGMOID_PEAK_DERIVATIVE, floored_grad_sigmoid

CONV_RUN = Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05')
OUT_DIR = Path('/tmp/bkadota_adaptivity_r8')
REPORT_DIR = Path('reports/adaptivity_mechanisms_r8')
ANATOMIES = ('brain', 'knee')
RADIUS_EDGES = (0, 8, 16, 24, 32, 48, 64, 80, 96, 128, 182)
BAND = (0.01, 0.99)
TRANSPORT_BATCH = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--sampler-run', type=Path, default=CONV_RUN)
    parser.add_argument('--sampler-ckpt', default='last_model.pt')
    parser.add_argument('--recon-run', type=Path, default=None, help='default: the sampler run')
    parser.add_argument('--recon-ckpt', default=None, help='default: the sampler checkpoint name')
    parser.add_argument('--pairs-per-anatomy', type=int, default=12)
    parser.add_argument('--draws', type=int, default=32)
    parser.add_argument('--slices-per-pass', type=int, default=4)
    parser.add_argument('--draws-per-pass', type=int, default=4)
    parser.add_argument('--no-transport', action='store_true')
    parser.add_argument('--from-saved', action='store_true', help='reuse the saved stage-1 gradients of this tag')
    parser.add_argument('--full-update-only', action='store_true', help='transport: skip the one-slice pushes')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load(run, ckpt, device):
    checkpoint = torch.load(run / ckpt, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    model.requires_grad_(False)
    # A reconstructor trained under `random`/`reinforce` masks skips autograd through its forward when frozen;
    # every mask here is passed in as `fixed_mask`, so the straight-through setting only re-enables that path.
    model.mask_train_mode = 'straight_through'
    return model, checkpoint


def adjacent_pairs(dataset, pairs_per_anatomy, rng):
    """Global indices of `pairs_per_anatomy` (slice, slice + 1) pairs per anatomy, one pair per volume."""
    chosen, offset = [], 0
    for part in dataset.datasets:
        by_volume = {}
        for position, (fname, slice_id) in enumerate(part.examples):
            by_volume.setdefault(fname, {})[slice_id] = offset + position
        volumes = [v for v in sorted(by_volume) if any(s + 1 in by_volume[v] for s in by_volume[v])]
        for volume in rng.choice(len(volumes), pairs_per_anatomy, replace=False):
            slices = by_volume[volumes[volume]]
            starts = sorted(s for s in slices if s + 1 in slices)
            start = starts[rng.randint(len(starts))]
            chosen += [slices[start], slices[start + 1]]
        offset += len(part)
    return chosen


def to_device(batch, device):
    labels = batch.get('anatomy_labels')
    return {'kspace': batch['kspace'].to(device), 'target': batch['target'].to(device),
            'label': labels.to(device) if torch.is_tensor(labels) else None}


@torch.no_grad()
def conv_input(model, kspace):
    """The conv sampler's network input [B, 5, H, W]: SENSE-combined centre k-space (twice, as the sampler builds it) and the centre mask."""
    sense_maps, _ = model.sensetivity_net(torch.view_as_complex(kspace.contiguous()), mask_method=model.sense_mask_method)
    old_mask, _ = model._initial_masks(kspace)
    observed = torch.view_as_complex((kspace * old_mask.unsqueeze(-1)).contiguous())
    combined = torch.view_as_real(fft_2d_img((ifft_2d_img(observed) * sense_maps).sum(dim=1)))
    stacked = torch.cat([combined, combined * old_mask.permute(0, 2, 3, 1)], dim=-1)
    if model.sampler.image_domain_input:
        stacked = model.sampler._to_image_domain(stacked)
    return torch.cat([stacked.permute(0, 3, 1, 2), old_mask], dim=1)


def check_conv_supported(mask_net):
    if (mask_net.location_head is not None or mask_net.anatomy_embedding is not None or mask_net.with_recon
            or mask_net.with_gfactor or mask_net.uncertainty_mode != 'none' or mask_net.num_anatomy_classes
            or mask_net.fixed_input):
        raise NotImplementedError('only the plain conv_2d score path (no head, conditioning or extra inputs)')


class ScoreModel:
    """Maps a slice to its score map z and z to the unscaled probability, exactly as the sampler does."""

    def __init__(self, model):
        self.model = model
        sampler = model.sampler
        self.is_loupe = sampler.is_loupe
        if self.is_loupe:
            self.slope, self.floor = sampler.mask_net.slope, sampler.mask_net.grad_floor
            self.limit, self.leak = float('inf'), 1.0
        else:
            check_conv_supported(sampler.mask_net)
            net = sampler.mask_net
            self.floor, self.limit, self.leak = net.grad_floor, net.logit_limit, net.logit_leak

    @torch.no_grad()
    def scores(self, part):
        if self.is_loupe:
            logits = self.model.sampler.mask_net.raw_logits(part['label']) * self.slope
            return logits.expand(len(part['kspace']), -1, -1).unsqueeze(1).clone()
        return self.model.sampler.mask_net.unet(conv_input(self.model, part['kspace']))

    def probability(self, z, old_mask):
        if self.is_loupe:
            return floored_grad_sigmoid(z, 1.0, self.floor) * (1 - old_mask)
        return scores_to_prob(z, self.floor, self.limit, self.leak) * (1 - old_mask)

    def derivative(self, z):
        """d(unscaled probability)/dz as the backward pass uses it (floor and leak included)."""
        clamped = z.clamp(-self.limit, self.limit)
        squashed = torch.sigmoid(clamped + self.leak * (z - clamped))
        local = torch.clamp(squashed * (1 - squashed), min=self.floor * SIGMOID_PEAK_DERIVATIVE)
        return local * torch.where(z.abs() > self.limit, torch.full_like(z, self.leak), torch.ones_like(z))

    @torch.no_grad()
    def check(self, part):
        old_mask, _ = self.model._initial_masks(part['kspace'])
        reference = self.model.propose_mask(part['kspace'], anatomy_labels=part['label'])['unscaled']
        error = (self.probability(self.scores(part), old_mask) - reference).abs().max().item()
        if error > 1e-5:
            raise RuntimeError(f'score path does not reproduce the sampler (max error {error:.2e})')


def per_sample_l1(recon, part, mask):
    output = recon(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)['output']
    target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
    return (output - target).abs().flatten(1).mean(1)


def tiled(part, copies):
    return {key: (value.repeat(copies, *([1] * (value.dim() - 1))) if value is not None else None)
            for key, value in part.items()}


def slice_gradients(score_model, recon, part, z, args):
    """Per-draw gradients of each slice's loss w.r.t. its own score map z and w.r.t. the binary mask m it drew.
    Masks use the sampler run's estimator. Returns half-A and half-B means, per-location per-draw variances,
    the first two single-draw z gradients and the per-draw losses."""
    count, half = len(z), args.draws // 2
    grads = {space: torch.zeros(args.draws, count, *z.shape[2:], device=z.device) for space in ('z', 'm')}
    losses = torch.zeros(args.draws, count)
    sampler_model = score_model.model
    old_mask, valid_mask = sampler_model._initial_masks(part['kspace'])
    many = tiled(part, args.draws_per_pass)
    for first in range(0, args.draws, args.draws_per_pass):
        leaf = z.repeat(args.draws_per_pass, 1, 1, 1).requires_grad_(True)
        olds, valids = old_mask.repeat(args.draws_per_pass, 1, 1, 1), valid_mask.repeat(args.draws_per_pass, 1, 1, 1)
        _, mask = sampler_model.sampler.select_locations(score_model.probability(leaf, olds), olds, sampler_model.budget, valids)
        mask.retain_grad()
        loss = per_sample_l1(recon, many, mask)
        loss.sum().backward()
        rows = slice(first, first + args.draws_per_pass)
        grads['z'][rows] = leaf.grad.view(args.draws_per_pass, count, *z.shape[2:])
        grads['m'][rows] = mask.grad.view(args.draws_per_pass, count, *z.shape[2:])
        losses[rows] = loss.detach().view(args.draws_per_pass, count).cpu()
    out = {'losses': losses.T, 'single': grads['z'][:2].transpose(0, 1)}
    for space, values in grads.items():
        out.update({f'{space}_a': values[:half].mean(0), f'{space}_b': values[half:].mean(0), f'{space}_var': values.var(0)})
    return out


def collect(score_model, recon, loader, args):
    rows, meta = {}, {'anatomy': [], 'fname': [], 'slice_id': []}

    def add(key, value):
        rows.setdefault(key, []).append(value.cpu())

    for index, batch in enumerate(loader):
        part = to_device(batch, args.device)
        if index == 0:
            score_model.check(part)
        z = score_model.scores(part)
        sampler_model = score_model.model
        old_mask, valid_mask = sampler_model._initial_masks(part['kspace'])
        with torch.no_grad():
            rescaled, _ = sampler_model.sampler.select_locations(
                score_model.probability(z, old_mask), old_mask, sampler_model.budget, valid_mask)
        for key, value in slice_gradients(score_model, recon, part, z, args).items():
            add(key, value)
        target = part['target'] if part['target'].dim() == 4 else part['target'].unsqueeze(1)
        add('z', z[:, 0])
        add('p_eff', torch.clamp(rescaled + old_mask, max=1)[:, 0])
        add('centre', old_mask[:, 0].bool())
        add('valid', valid_mask[:, 0].bool())
        add('support', _signal_mask(target).flatten(1).mean(1))
        meta['anatomy'] += list(batch['anatomy'])
        meta['fname'] += list(batch['fname'])
        meta['slice_id'] += [int(s) for s in batch['slice_id']]
        print(f'{len(meta["anatomy"])} slices, mean loss {rows["losses"][-1].mean():.4f}', flush=True)
    data = {k: torch.cat(v) for k, v in rows.items()}
    data.update(meta)
    return data


def inner(x, y):
    return (x * y).flatten(1).sum(1)


def decompose(data, space):
    """Population / anatomy / slice parts of the half-A and half-B mean gradients in `space` ('z' or 'm')."""
    anatomy = np.array(data['anatomy'])
    parts = {}
    for half in ('a', 'b'):
        grads = data[f'{space}_{half}']
        population = grads.mean(0, keepdim=True)
        offset = torch.zeros_like(grads)
        for name in ANATOMIES:
            rows = torch.as_tensor(anatomy == name)
            offset[rows] = grads[rows].mean(0, keepdim=True) - population
        parts[half] = {'pop': population.expand_as(grads), 'anat': offset, 'slice': grads - population - offset}
    return parts


def energies(parts, data, space, where=None):
    """Noise-free energies of each part (cross-half products) and the per-draw noise energy, per-slice means."""
    weight = torch.ones_like(data['z']) if where is None else where.float()
    out = {name: float(inner(parts['a'][name] * weight, parts['b'][name]).mean()) for name in ('pop', 'anat', 'slice')}
    out['noise_per_draw'] = float((data[f'{space}_var'] * weight).flatten(1).sum(1).mean())
    total = out['pop'] + out['anat'] + out['slice']
    out.update({f'{name}_frac': out[name] / total for name in ('pop', 'anat', 'slice')})
    out['slice_snr_per_draw'] = out['slice'] / out['noise_per_draw']
    out['anat_snr_per_draw'] = out['anat'] / out['noise_per_draw']
    out['pop_snr_batch16'] = out['pop'] / ((out['anat'] + out['slice'] + out['noise_per_draw']) / TRANSPORT_BATCH)
    return out


def correlation(x, y):
    x, y = x.flatten(1), y.flatten(1)
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    return inner(x, y) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-30)


def unit_rows(x):
    x = x.flatten(1)
    x = x - x.mean(1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-30)


def pair_kinds(data):
    """[N, N] array of 'self', 'neighbour' (adjacent slice, same volume), 'other' (same anatomy) or 'cross_anatomy'."""
    anatomy, fnames, slice_ids = np.array(data['anatomy']), np.array(data['fname']), np.array(data['slice_id'])
    same_anatomy = anatomy[:, None] == anatomy[None, :]
    neighbour = (fnames[:, None] == fnames[None, :]) & (np.abs(slice_ids[:, None] - slice_ids[None, :]) == 1)
    kinds = np.where(same_anatomy, np.where(neighbour, 'neighbour', 'other'), 'cross_anatomy').astype(object)
    np.fill_diagonal(kinds, 'self')
    return kinds


def upper(matrix, kinds, kind):
    rows, cols = np.triu_indices(len(matrix), 1)
    keep = kinds[rows, cols] == kind
    return matrix[rows[keep], cols[keep]]


def sharing(parts, data):
    """Split-half reliability of s_i, and correlation of s between adjacent slices vs other volumes (raw and disattenuated)."""
    s_a, s_b = parts['a']['slice'], parts['b']['slice']
    self_energy = inner(s_a, s_b)
    reliability = correlation(s_a, s_b)
    cross = (s_a.flatten(1) @ s_b.flatten(1).T + s_b.flatten(1) @ s_a.flatten(1).T) / 2
    scale = torch.sqrt(self_energy.clamp(min=1e-30)[:, None] * self_energy.clamp(min=1e-30)[None, :])
    disattenuated = (cross / scale).numpy()
    raw = unit_rows(s_a) @ unit_rows(s_b).T
    raw = ((raw + raw.T) / 2).numpy()
    kinds = pair_kinds(data)
    return {'split_half_reliability_16_draws': float(reliability.mean()),
            'positive_self_energy_frac': float((self_energy > 0).float().mean()),
            'neighbour_corr_raw': float(upper(raw, kinds, 'neighbour').mean()),
            'other_volume_corr_raw': float(upper(raw, kinds, 'other').mean()),
            'neighbour_corr_disattenuated_median': float(np.median(upper(disattenuated, kinds, 'neighbour'))),
            'neighbour_corr_disattenuated_quartiles': np.percentile(upper(disattenuated, kinds, 'neighbour'), [25, 50, 75]).tolist(),
            'other_volume_corr_disattenuated_median': float(np.median(upper(disattenuated, kinds, 'other'))),
            'pairs': int((upper(raw, kinds, 'neighbour')).size)}


SCALES = (1, 2, 4, 8)


def smoothed(x, sigma):
    """Gaussian blur (std `sigma` px) of [N, H, W] maps, zero padded."""
    radius = int(3 * sigma)
    grid = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-grid ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    out = F.conv2d(x[:, None], kernel.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(out, kernel.view(1, 1, -1, 1), padding=(radius, 0))[:, 0]


def scales(parts, data):
    """Share of the noise-free slice-specific energy at spatial scales >= sigma (kept by a Gaussian blur), and the
    disattenuated neighbour / other-volume correlation of the blurred signal."""
    kinds = pair_kinds(data)
    total = inner(parts['a']['slice'], parts['b']['slice']).sum()
    rows = []
    for sigma in SCALES:
        a, b = smoothed(parts['a']['slice'], sigma), smoothed(parts['b']['slice'], sigma)
        self_energy = inner(a, b).clamp(min=1e-30)
        cross = (a.flatten(1) @ b.flatten(1).T + b.flatten(1) @ a.flatten(1).T) / 2
        disattenuated = (cross / torch.sqrt(self_energy[:, None] * self_energy[None, :])).numpy()
        rows.append({'sigma_px': sigma, 'energy_kept': float(inner(a, b).sum() / total),
                     'neighbour_corr_disattenuated_median': float(np.median(upper(disattenuated, kinds, 'neighbour'))),
                     'other_volume_corr_disattenuated_median': float(np.median(upper(disattenuated, kinds, 'other')))})
    population = {f'pop_energy_kept_sigma{sigma}': float(
        inner(smoothed(parts['a']['pop'][:1], sigma), smoothed(parts['b']['pop'][:1], sigma)).sum()
        / inner(parts['a']['pop'][:1], parts['b']['pop'][:1]).sum()) for sigma in SCALES}
    return {'slice': rows, **population}


def radius_map(shape):
    ky, kx = torch.meshgrid(torch.arange(shape[0]) - shape[0] // 2, torch.arange(shape[1]) - shape[1] // 2, indexing='ij')
    return torch.hypot(ky.float(), kx.float())


def regions(data, score_model):
    centre = data['centre']
    clamped = (data['z'].abs() > score_model.limit) & ~centre
    dead = (data['p_eff'] <= BAND[0]) & ~centre
    in_band = (data['p_eff'] > BAND[0]) & (data['p_eff'] < BAND[1]) & ~centre
    saturated = (data['p_eff'] >= BAND[1]) & ~centre
    return {'centre': centre, 'in_band': in_band, 'saturated': saturated, 'dead': dead, 'clamped': clamped}


def region_shares(parts, data, score_model):
    """Where (by the sampler's own probability) the noise-free mask-space signal of each part lies."""
    out = {}
    for name, where in regions(data, score_model).items():
        row = {'frac_locations': float(where.float().mean())}
        for part in ('pop', 'anat', 'slice'):
            total = inner(parts['a'][part], parts['b'][part]).sum()
            row[f'{part}_share'] = float(inner(parts['a'][part] * where, parts['b'][part]).sum() / total)
        out[name] = row
    return out


def radial(z_parts, m_parts, data, score_model):
    radius = radius_map(data['z'].shape[1:])
    marks = regions(data, score_model)
    rows = []
    for low, high in zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:]):
        ring = ((radius >= low) & (radius < high)).expand_as(data['z'])
        row = {'r_low': low, 'r_high': high}
        row.update({f'z_{name}': float(inner(z_parts['a'][name] * ring, z_parts['b'][name]).mean()) for name in ('pop', 'anat', 'slice')})
        row['z_noise_per_draw'] = float((data['z_var'] * ring).flatten(1).sum(1).mean())
        row.update({f'm_{name}': float(inner(m_parts['a'][name] * ring, m_parts['b'][name]).mean()) for name in ('pop', 'anat', 'slice')})
        row['m_noise_per_draw'] = float((data['m_var'] * ring).flatten(1).sum(1).mean())
        row['mean_p_eff'] = float(data['p_eff'][ring].mean())
        row.update({f'frac_{name}': float(where[ring].float().mean()) for name, where in marks.items()})
        row['z_across_slice_std'] = float(data['z'].std(0)[ring[0]].mean())
        rows.append(row)
    return rows


def adam_preconditioner(model, checkpoint, names):
    """1 / (sqrt(v_hat) + eps) of the checkpoint's Adam state for the named sampler U-Net parameters."""
    trainable = list(model.named_parameters())
    state = checkpoint['optimizer']['state']
    group = checkpoint['optimizer']['param_groups'][0]
    if len(trainable) != len(group['params']):
        raise RuntimeError(f'{len(trainable)} parameters vs {len(group["params"])} in the optimizer')
    beta2, eps = group['betas'][1], group['eps']
    out = {}
    for index, (name, parameter) in zip(group['params'], trainable):
        short = name.removeprefix('sampler.mask_net.unet.')
        if short not in names:
            continue
        entry = state[index]
        if entry['exp_avg_sq'].shape != parameter.shape:
            raise RuntimeError(f'optimizer state shape mismatch at {name}')
        v_hat = entry['exp_avg_sq'] / (1 - beta2 ** float(entry['step']))
        out[short] = 1.0 / (v_hat.sqrt() + eps)
    if set(out) != set(names):
        raise RuntimeError('missing optimizer state for some sampler parameters')
    return out


TRANSPORT_RINGS = (0, 16, 32, 64, 96, 182)


def transport(model, checkpoint, loader, data, z_parts, args):
    """How a network update moves the scores of every slice: the kernel J_j P J_i^T (P: Adam preconditioner or 1).

    full_update   update from all slices' gradients (half A); the slice-specific part of the score change, and its
                  alignment with each slice's own slice-specific gradient (half B). A free per-slice map has
                  alignment = split-half reliability.
    single_push   update from one slice's slice-specific gradient s_i alone; response of every slice j along -s_i,
                  r_ji = <dz_j, -s_i> / |s_i|^2, for j = i, its neighbour, other volumes, the other anatomy, and per
                  k-space ring. Specificity 1 - r_other / r_self: 1 for a free per-slice map, 0 for LOUPE.
    anatomy_push  update from one anatomy's offset on that anatomy's slices; response along -offset on both anatomies.
    """
    unet = model.sampler.mask_net.unet
    params = {n: p.detach() for n, p in unet.named_parameters()}
    preconditioner = {k: v.to(args.device) for k, v in adam_preconditioner(model, checkpoint, set(params)).items()}
    inputs = torch.cat([conv_input(model, to_device(b, args.device)['kspace']).cpu() for b in loader])
    anatomy = np.array(data['anatomy'])
    kinds = pair_kinds(data)
    radius = radius_map(data['z'].shape[1:])
    rings = [((radius >= lo) & (radius < hi)).float() for lo, hi in zip(TRANSPORT_RINGS[:-1], TRANSPORT_RINGS[1:])]

    def scores(p, x):
        return functional_call(unet, p, (x,))

    def update(directions, rows, precondition):
        total = {n: torch.zeros_like(p) for n, p in params.items()}
        for chunk in torch.as_tensor(rows).split(8):
            live = {n: p.clone().requires_grad_(True) for n, p in params.items()}
            z = scores(live, inputs[chunk].to(args.device))
            grads = torch.autograd.grad((z * directions[chunk].unsqueeze(1).to(args.device)).sum(), list(live.values()))
            for name, g in zip(live, grads):
                total[name] += g
        return {n: -(g * preconditioner[n] if precondition else g) for n, g in total.items()}

    def score_change(step):
        changes = []
        for chunk in torch.arange(len(inputs)).split(8):
            _, tangent = jvp(lambda p: scores(p, inputs[chunk].to(args.device)), (params,), (step,))
            changes.append(tangent[:, 0].cpu())
        return torch.cat(changes)

    def slice_specific(change):
        out = change - change.mean(0, keepdim=True)
        for name in ANATOMIES:
            rows = torch.as_tensor(anatomy == name)
            out[rows] = out[rows] - out[rows].mean(0, keepdim=True)
        return out

    results = {}
    for precondition in (True, False):
        key = 'adam' if precondition else 'sgd'
        change = score_change(update(data['z_a'], list(range(len(inputs))), precondition))
        specific = slice_specific(change)
        common = change.mean(0, keepdim=True)
        offsets = change.clone()
        for name in ANATOMIES:
            rows = torch.as_tensor(anatomy == name)
            offsets[rows] = change[rows].mean(0, keepdim=True) - common
        results[f'{key}_full_update'] = {
            'common_alignment_with_population': float(correlation(common, -z_parts['b']['pop'][:1]).mean()),
            'anatomy_offset_alignment': float(correlation(offsets, -z_parts['b']['anat']).mean()),
            'slice_specific_alignment': float(correlation(specific, -z_parts['b']['slice']).mean()),
            'free_map_alignment': float(correlation(z_parts['a']['slice'], z_parts['b']['slice']).mean()),
            'slice_specific_norm_frac': float(specific.norm() / change.norm())}
        if args.full_update_only:
            continue

        responses = np.zeros((len(inputs), len(inputs)))
        ring_responses = np.zeros((len(rings), len(inputs), len(inputs)))
        self_cosine, common_frac = [], []
        smooth_cosine = {sigma: [] for sigma in SCALES}
        for i in range(len(inputs)):
            direction = torch.zeros_like(data['z_a'])
            direction[i] = z_parts['a']['slice'][i]
            change = score_change(update(direction, [i], precondition))
            push = -direction[i]
            responses[:, i] = (inner(change, push.expand_as(change)) / push.pow(2).sum()).numpy()
            self_cosine.append(float(inner(change[i:i + 1], push[None]) / (change[i].norm() * push.norm())))
            for sigma in SCALES:
                target = smoothed(-z_parts['b']['slice'][i:i + 1], sigma)
                smooth_cosine[sigma].append(float(inner(change[i:i + 1], target) / (change[i].norm() * target.norm())))
            others = torch.as_tensor(kinds[:, i] == 'other')
            common_frac.append(float(change[others].mean(0).norm() / change[i].norm()))
            for r, ring in enumerate(rings):
                ring_responses[r, :, i] = (inner(change * ring, push.expand_as(change) * ring) / (push * ring).pow(2).sum()).numpy()

        def summary(matrix):
            self_response = float(np.diag(matrix).mean())
            row = {'self': self_response}
            for kind in ('neighbour', 'other', 'cross_anatomy'):
                row[kind] = float(matrix[kinds == kind].mean())
            row['specificity'] = 1 - row['other'] / self_response
            return row

        results[f'{key}_single_push'] = summary(responses)
        results[f'{key}_single_push']['self_cosine'] = float(np.mean(self_cosine))
        results[f'{key}_single_push'].update({f'self_cosine_with_blurred_half_b_sigma{sigma}': float(np.mean(v))
                                              for sigma, v in smooth_cosine.items()})
        results[f'{key}_single_push']['other_mean_change_over_self_change_norm'] = float(np.mean(common_frac))
        results[f'{key}_single_push_by_ring'] = [{'r_low': lo, 'r_high': hi, **summary(m)}
                                                 for lo, hi, m in zip(TRANSPORT_RINGS[:-1], TRANSPORT_RINGS[1:], ring_responses)]

        # Control: the same one-slice push along the population direction or the anatomy offset. If the network follows
        # these but not s_i, the low self cosine is specific to per-slice directions, not a property of any one step.
        controls = {}
        for part_name in ('pop', 'anat', 'slice'):
            pushed, independent = [], []
            for i in range(len(inputs)):
                direction = torch.zeros_like(data['z_a'])
                direction[i] = z_parts['a'][part_name][i]
                change = score_change(update(direction, [i], precondition))[i:i + 1]
                pushed.append(float(inner(change, -direction[i:i + 1]) / (change.norm() * direction[i].norm())))
                target = -z_parts['b'][part_name][i:i + 1]
                independent.append(float(inner(change, target) / (change.norm() * target.norm())))
            controls[part_name] = {'self_cosine_with_push': float(np.mean(pushed)),
                                   'self_cosine_with_half_b': float(np.mean(independent))}
        results[f'{key}_single_push_controls'] = controls

        for name in ANATOMIES:
            rows = np.flatnonzero(anatomy == name)
            direction = torch.zeros_like(data['z_a'])
            direction[rows] = z_parts['a']['anat'][rows]
            change = score_change(update(direction, rows.tolist(), precondition))
            push = -z_parts['a']['anat'][rows[0]]
            response = (inner(change, push.expand_as(change)) / push.pow(2).sum()).numpy()
            results[f'{key}_anatomy_push_{name}'] = {f'response_on_{a}': float(response[anatomy == a].mean()) for a in ANATOMIES}
        print(key, json.dumps({k: v for k, v in results.items() if k.startswith(key) and 'ring' not in k}), flush=True)
    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model, checkpoint = load(args.sampler_run, args.sampler_ckpt, args.device)
    recon_run = args.recon_run or args.sampler_run
    recon_ckpt = args.recon_ckpt or args.sampler_ckpt
    same = recon_run == args.sampler_run and recon_ckpt == args.sampler_ckpt
    recon = model if same else load(recon_run, recon_ckpt, args.device)[0]
    score_model = ScoreModel(model)

    cfg = OmegaConf.load(CONV_RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, _ = build_dataloaders(cfg)
    indices = adjacent_pairs(train_loader.dataset, args.pairs_per_anatomy, rng)
    loader = DataLoader(Subset(train_loader.dataset, indices), batch_size=args.slices_per_pass, shuffle=False,
                        num_workers=4, collate_fn=train_loader.collate_fn)

    saved = OUT_DIR / f'gradients_{args.tag}.pt'
    if args.from_saved:
        data = torch.load(saved, weights_only=False)
    else:
        data = collect(score_model, recon, loader, args)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data, saved)

    outside_centre = ~data['centre']
    for key in ('m_a', 'm_b', 'm_var'):
        data[key] = data[key] * outside_centre
    z_parts, m_parts = decompose(data, 'z'), decompose(data, 'm')
    results = {'config': {'sampler': f'{args.sampler_run}/{args.sampler_ckpt}', 'recon': f'{recon_run}/{recon_ckpt}',
                          'slices': len(indices), 'draws': args.draws, 'seed': args.seed},
               'mean_loss': float(data['losses'].mean()),
               'z_energies': energies(z_parts, data, 'z'),
               'm_energies': energies(m_parts, data, 'm'),
               'z_sharing': sharing(z_parts, data),
               'm_sharing': sharing(m_parts, data),
               'm_region_shares': region_shares(m_parts, data, score_model),
               'z_scales': scales(z_parts, data),
               'radial': radial(z_parts, m_parts, data, score_model)}
    print(json.dumps({k: v for k, v in results.items() if k != 'radial'}, indent=1), flush=True)
    if not args.no_transport and not score_model.is_loupe and same:
        results['transport'] = transport(model, checkpoint, loader, data, z_parts, args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = '_full_update' if args.full_update_only else ''
    (REPORT_DIR / f'gradients_{args.tag}{suffix}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
