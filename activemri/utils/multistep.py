"""Diagnostics of K-step sampling runs, shared by the trainer's logging and scripts/eval/eval_multistep_adaptivity.py.

    step_thresholds       one U(0, 1) map per (draw seed, slice, step), so runs scored on the same slices share draws
    masked_slice_metrics  PSNR_Masked, SSIM_Masked and NMSE per slice, as the trainer scores them
    cluster_bootstrap     slice-weighted mean with a CI from resampling volumes within each condition
    header_signature,     donors for the swap tests: another volume's slice with the same condition, acquired-location
    match_donors          mask and coil count, on which a data-blind sampler draws exactly the slice's own trajectory
    StepRecorder          what each step of one forward chose, and the gradient the loss sends each step's probabilities
    step_panels,          the per-step figure of one slice: one row per step, one column per quantity
    step_grid_figure
"""
import hashlib
import math
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.metrics import masked_ssim

BOOTSTRAP_DRAWS = 2000
LOG_MAGNITUDE_FLOOR = 1e-12


def _draw_seed(seed, slice_id, step):
    return int(np.random.SeedSequence([seed, slice_id, step]).generate_state(1, dtype=np.uint64)[0])


def step_thresholds(seed, slice_ids, num_steps, shape, device='cpu'):
    """`num_steps` maps [B, 1, H, W] of U(0, 1) values, one per step for the slices `slice_ids`.

    A (seed, slice, step) triple always gives the same map, so two runs scored on the same slices share every draw,
    whatever the batching, and no two steps of a slice share a map.
    """
    return [torch.stack([torch.rand(shape, generator=torch.Generator().manual_seed(_draw_seed(seed, int(s), step)))
                         for s in slice_ids]).unsqueeze(1).to(device) for step in range(num_steps)]


def masked_slice_metrics(output, target, mask):
    """{'psnr', 'ssim', 'nmse'}: [B] float64 per slice of [B, 1, H, W] images, as the trainer scores them.

    PSNR and SSIM are MaskedMetricAccumulator's: the error over the metric `mask`, with the slice's own target maximum as
    the range, and `masked_ssim`. NMSE is the trainer's NMSE of the masked images.
    """
    output, target, mask = (x.unsqueeze(1) if x.dim() == 3 else x for x in (output, target, mask))
    mse = ((target - output) ** 2 * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    psnr = 10 * torch.log10(target.amax(dim=(1, 2, 3)) ** 2 / mse)
    error, reference = ((target - output) * mask).flatten(1), (target * mask).flatten(1)
    nmse = error.norm(dim=1) ** 2 / reference.norm(dim=1) ** 2
    return {'psnr': psnr.double(), 'ssim': masked_ssim(output, target, mask), 'nmse': nmse.double()}


def cluster_bootstrap(values, volumes, strata, draws=BOOTSTRAP_DRAWS, seed=0):
    """Slice-weighted mean of `values` [N] with a 95% CI from resampling whole volumes within each stratum.

    Every draw keeps each stratum's (condition's) volume count and weights slices as the mean does. Non-finite values
    (e.g. slices with no donor) are left out and counted.
    """
    values, volumes, strata = np.asarray(values, dtype=float), np.asarray(volumes), np.asarray(strata)
    finite = np.isfinite(values)
    values, volumes, strata = values[finite], volumes[finite], strata[finite]
    if not len(values):
        return {'mean': float('nan'), 'ci95': [float('nan')] * 2, 'n_slices': 0, 'n_volumes': 0,
                'n_left_out': int((~finite).sum())}
    rng = np.random.default_rng(seed)
    sums, counts = np.zeros(draws), np.zeros(draws)
    for stratum in np.unique(strata):
        inside = strata == stratum
        names, position = np.unique(volumes[inside], return_inverse=True)
        per_sum = np.bincount(position, weights=values[inside], minlength=len(names))
        per_count = np.bincount(position, minlength=len(names))
        picks = rng.integers(len(names), size=(draws, len(names)))
        sums += per_sum[picks].sum(axis=1)
        counts += per_count[picks].sum(axis=1)
    return {'mean': float(values.mean()), 'ci95': np.percentile(sums / counts, [2.5, 97.5]).tolist(),
            'n_slices': int(len(values)), 'n_volumes': int(len(np.unique(volumes))), 'n_left_out': int((~finite).sum())}


def header_signature(kspace):
    """Per slice of real-view k-space [B, C, H, W, 2]: (hash of its acquired-location mask, active coil count).

    The acquired region sets the available locations and the budget, and with the coil count it is all a data-blind
    sampler reads besides its own draws, so two slices of one condition and signature get the same trajectory from it.
    """
    coil_energy = torch.view_as_complex(kspace.contiguous()).abs()
    acquired = (coil_energy.sum(dim=1) > 0).cpu().numpy()
    coils = (coil_energy.flatten(2).sum(dim=-1) > 0).sum(dim=1).tolist()
    return [(hashlib.sha1(np.packbits(region).tobytes()).hexdigest()[:16], int(count))
            for region, count in zip(acquired, coils)]


def match_donors(conditions, volumes, signatures, seed=7):
    """For each slice, a random slice of another volume with the same condition and header signature, else -1."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    keys = list(zip(conditions, signatures))
    for index, key in enumerate(keys):
        groups[key].append(index)
    donors = np.full(len(keys), -1)
    for index, key in enumerate(keys):
        candidates = [other for other in groups[key] if volumes[other] != volumes[index]]
        if candidates:
            donors[index] = candidates[rng.integers(len(candidates))]
    return donors


class StepRecorder:
    """What each step of one forward chose, and, once backward has run, the gradient its probabilities received.

    Set as `SequentialSampling.step_recorder` around a forward. Keeps per-step scalars for the batch and [H, W] maps
    only for the `slices` asked for, so recording K=16 steps of a batch adds a few small tensors, not K full ones.
    """

    def __init__(self, slices=()):
        self.slices = list(slices)
        self.steps = []

    def add(self, step, old_mask, mask, details):
        new = (mask - old_mask).detach()
        record = {'step': step, 'new_samples': new.flatten(1).sum(dim=1), 'grad_norm': None, 'maps': {}}
        residual = details.get('residual')
        if residual is not None:
            record['residual_rms'] = residual.detach().pow(2).flatten(1).mean(dim=1).sqrt()
        for index in self.slices:
            maps = {'old_mask': old_mask[index, 0], 'new': new[index, 0]}
            maps.update({key: details[key][index, 0] for key in ('rescaled', 'residual') if details.get(key) is not None})
            record['maps'][index] = {key: value.detach().float().cpu() for key, value in maps.items()}
        unscaled = details.get('unscaled')
        if unscaled is not None and unscaled.requires_grad:
            unscaled.register_hook(partial(self._receive_gradient, record))
        self.steps.append(record)

    @staticmethod
    def _receive_gradient(record, grad):
        record['grad_norm'] = grad.detach().flatten(1).norm(dim=1)
        for index, maps in record['maps'].items():
            maps['grad'] = grad[index, 0].detach().float().cpu()

    def scalars(self, prefix):
        """wandb scalars `{prefix}/steps/...`: per step the mean new samples per slice, the residual's RMS and the norm of
        the gradient reaching the step's probabilities; and the last step's share of the summed squared gradient norms
        (1 when only the last step learns, 1/K when every step gets an equal pull)."""
        log, squared = {}, []
        for record in self.steps:
            step = record['step']
            log[f'{prefix}/steps/new_samples_{step}'] = record['new_samples'].float().mean().item()
            if 'residual_rms' in record:
                log[f'{prefix}/steps/residual_rms_{step}'] = record['residual_rms'].mean().item()
            if record['grad_norm'] is not None:
                squared.append(record['grad_norm'].pow(2).sum().item())
                log[f'{prefix}/steps/grad_norm_{step}'] = math.sqrt(squared[-1])
        if squared and sum(squared) > 0:
            log[f'{prefix}/steps/last_grad_share'] = squared[-1] / sum(squared)
        return log


@torch.no_grad()
def step_panels(model, kspace, target, sense_maps, masks, index, anatomy_labels=None, metric_mask=None, recorder=None,
                chunk=4):
    """Per-step images of slice `index` for `step_grid_figure`, from a forward's after-step `masks` [M_1, ..., M_K].

    Each row is one step i: the coil-RSS magnitude of the k-space measured before it (y * M_i), what the recorder kept
    (rescaled probabilities, residual, gradient, new samples), the mask after it, and the zero-filled image,
    reconstruction and error of that mask. Only the final mask is a training input, so the earlier reconstructions are
    of masks off the reconstructor's training distribution.
    """
    coil_kspace = kspace[index:index + 1]
    maps = sense_maps[index:index + 1]
    labels = None if anatomy_labels is None else anatomy_labels[index:index + 1]
    after = [mask[index:index + 1] for mask in masks]
    before = [model._initial_masks(coil_kspace)[0]] + after[:-1]
    recons = []
    for start in range(0, len(after), chunk):
        stacked = torch.cat(after[start:start + chunk])
        count = len(stacked)
        recons.append(model.reconstruct_from_mask(
            coil_kspace.expand(count, -1, -1, -1, -1).contiguous(), stacked, maps.expand(count, -1, -1, -1),
            None if labels is None else labels.expand(count)))
    recons = torch.cat(recons)
    coils = torch.view_as_complex(coil_kspace.contiguous())[0]
    truth = (target[index:index + 1] if target.dim() == 4 else target[index:index + 1].unsqueeze(1)).float()
    rows = []
    for step, (mask_before, mask_after) in enumerate(zip(before, after)):
        measured = (coils * mask_before[0]).abs().pow(2).sum(dim=0).sqrt()
        zero_filled = (ifft_2d_img(coils * mask_after[0]) * maps[0].conj()).sum(dim=0).abs()
        recon = recons[step:step + 1].float()
        row = {'measured': torch.log10(measured + LOG_MAGNITUDE_FLOOR) * (mask_before[0, 0] > 0),
               'mask_after': mask_after[0, 0], 'zero_filled': zero_filled, 'recon': recon[0, 0],
               'error': (recon - truth).abs()[0, 0]}
        if metric_mask is not None:
            row['psnr'] = masked_slice_metrics(recon, truth, metric_mask[index:index + 1])['psnr'].item()
        if recorder is not None and step < len(recorder.steps):
            row.update(recorder.steps[step]['maps'].get(index, {}))
        rows.append({key: value.cpu().float().numpy() if torch.is_tensor(value) else value for key, value in row.items()})
    return {'rows': rows, 'target': truth[0, 0].cpu().numpy()}


STEP_COLUMNS = (
    ('measured', 'seen before step i\nlog10 |y M_i| (coil RSS)', 'magma'),
    ('rescaled', 'p_i, rescaled\n(step i draw)', 'inferno'),
    ('residual', 'residual r_i\n(logits)', 'RdBu_r'),
    ('grad', 'dL/dp_i\n(loss gradient)', 'RdBu_r'),
    ('new', 'new samples b_i', 'gray'),
    ('mask_after', 'mask M_{i+1}', 'gray'),
    ('zero_filled', 'zero-filled\nfrom M_{i+1}', 'gray'),
    ('recon', 'reconstruction\nfrom M_{i+1}', 'gray'),
    ('error', '|reconstruction\n- target|', 'magma'),
)
SHARED_SYMMETRIC = ('residual', 'grad')
SYMMETRIC_PERCENTILE = 99.5  # a few near-DC gradients are orders above the rest; a max-based range would blank the map


def step_grid_figure(panels, title, panel_inches=1.45, dpi=80):
    """One row per step and one column per quantity (STEP_COLUMNS), for `step_panels` output.

    The residual and gradient columns share one symmetric colour range over all steps (the 99.5th percentile of their
    magnitude, so a few outliers saturate instead of blanking the rest), so their size can be compared between steps;
    the row label gives each step's new samples, gradient norm and reconstruction PSNR.
    """
    rows = panels['rows']
    columns = [column for column in STEP_COLUMNS if any(column[0] in row for row in rows)]
    figure, axes = plt.subplots(len(rows), len(columns), squeeze=False, dpi=dpi,
                                figsize=(panel_inches * len(columns), panel_inches * len(rows) + 0.9))
    image_top = max(float(np.percentile(panels['target'], 99.5)), 1e-12)
    error_top = max(max(float(np.percentile(row['error'], 99.5)) for row in rows), 1e-12)
    symmetric = {key: max(float(np.percentile(np.abs(np.stack([row[key] for row in rows if key in row])),
                                              SYMMETRIC_PERCENTILE)) if any(key in row for row in rows) else 0.0, 1e-12)
                 for key in SHARED_SYMMETRIC}
    measured = [row['measured'][row['measured'] != 0] for row in rows]
    measured_low = min((float(values.min()) for values in measured if values.size), default=0.0)
    for r, row in enumerate(rows):
        for c, (key, label, cmap) in enumerate(columns):
            axis = axes[r, c]
            axis.set_xticks([])
            axis.set_yticks([])
            if r == 0:
                axis.set_title(label, fontsize=6)
            if key not in row:
                axis.set_facecolor('0.92')
                continue
            limits = {'measured': (measured_low, None), 'zero_filled': (0, image_top), 'recon': (0, image_top),
                      'error': (0, error_top), 'new': (0, 1), 'mask_after': (0, 1)}.get(key)
            if key in SHARED_SYMMETRIC:
                limits = (-symmetric[key], symmetric[key])
            vmin, vmax = limits if limits else (None, None)
            value = np.where(row[key] == 0, np.nan, row[key]) if key == 'measured' else row[key]
            axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        label = [f"step {r}", f"+{int(row['new'].sum())} new" if 'new' in row else '']
        if 'grad' in row:
            label.append(f"|g| {np.linalg.norm(row['grad']):.2e}")
        if 'psnr' in row:
            label.append(f"{row['psnr']:.2f} dB")
        axes[r, 0].set_ylabel('\n'.join(part for part in label if part), fontsize=6)
    figure.suptitle(title, fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (panel_inches * len(rows) + 0.9)))
    return figure
