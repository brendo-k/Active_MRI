"""What do the per-slice oracle masks look like, and what in the slice do they follow?

On the 128 labelled validation slices of oracle_labels_r8.py; d is a label's probability minus the anatomy-mean label
(training labels) over the available locations. Three blocks:

  mask       - how binary the labels are: share of available locations with p in (0.05, 0.95), against the
               anatomy-mean label;
             - the radial share of |d|^2 against the label's share of the budget;
             - d's autocorrelation at 1-2 px along each axis (negative: the oracle moves samples by a pixel), with a
               Bernoulli draw of the anatomy mean as a control;
             - corr(d(k), d(k*)) for the Hermitian partner k* = -k, against the two one-axis mirrors (placebos).
               Negative: the oracle avoids sampling both of a conjugate pair;
             - on the 32 refit slices of oracle_init_identifiability_r8.py, the cross-start agreement of the pair
               sum d(k) + d(k*) and of the pair difference d(k) - d(k*). A high pair-sum agreement next to a low
               difference agreement would mean the starts disagree mostly about which partner to take.
  slice      what d follows in the slice's own k-space, through the frozen conv ep50 VarNet at the anatomy-mean label
             (`--draws` seeded draws):
             - the residual power |fft(x_hat) - y|^2 of the SENSE-combined k-space, averaged over the draws in which
               k was not sampled;
             - the ground-truth power |y|^2.
             Both are taken as logs minus their anatomy mean over the labelled slices. Every field, d included, is
             split into a smooth band (normalised Gaussian blur, sigma `--band-sigma` px) and a pixel band (the
             rest); corr(d, .) is reported per band. The residual depends on the draws, so its split-half
             reliability (first vs second half of the draws, Spearman-Brown corrected to all draws) is reported and
             its correlations are also given disattenuated by it.
  neighbour  the same log residual and log power for the next slice of the same volume: their per-band correlation
             between the two slices, with the pair's ACS similarity. For comparison, the labels of adjacent training
             slices correlate 0.059 (pixel) / 0.53 (4 px) (acs_vs_label_similarity_r8.py).

    python scripts/diagnostics/oracle_mask_characteristics_r8.py
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
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
from label_curriculum_r8 import gaussian_blur
from label_predictor_gonogo_r8 import ANATOMIES, available_of, label_probability, load_shards, logit_of
from oracle_init_identifiability_r8 import REFITS, leave_one_out_reference, matched_rows
from oracle_labels_r8 import OUT_DIR, RUN, load_frozen, loader_over, masks_from_logits, to_device

OUT = Path('reports/label_gonogo_r8/oracle_mask_characteristics.json')
RINGS = (0, 8, 16, 32, 64, 1000)
LAGS = {'axis0_lag1': (1, 0), 'axis0_lag2': (2, 0), 'axis1_lag1': (0, 1), 'axis1_lag2': (0, 2), 'diagonal_lag1': (1, 1)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--band-sigma', type=float, default=4.0)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def corr(x, y, keep):
    x, y = x[keep].double(), y[keep].double()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-30))


def mean_corr(xs, ys, keeps):
    return float(np.nanmean([corr(x, y, k) for x, y, k in zip(xs, ys, keeps)]))


def partner(field, axes=(-2, -1)):
    """field at -k for centred indices (DC at N // 2): index i -> (N - i) % N along each axis in `axes`."""
    return torch.roll(torch.flip(field, dims=axes), shifts=(1,) * len(axes), dims=axes)


def bands(field, keep, sigma):
    """(smooth, pixel) parts of a field over the kept locations, by normalised Gaussian blur."""
    weight = keep.float()
    smooth = gaussian_blur(torch.nan_to_num(field) * weight, sigma) / gaussian_blur(weight, sigma).clamp(min=1e-6)
    return smooth, torch.nan_to_num(field) - smooth


def lagged_corr(field, keep, lag):
    a, b = lag
    height, width = field.shape[-2:]
    first = field[..., :height - a, :width - b]
    second = field[..., a:, b:]
    both = keep[..., :height - a, :width - b] & keep[..., a:, b:]
    return mean_corr(first, second, both)


def radius_map(shape):
    rows = torch.arange(shape[0]) - shape[0] // 2
    cols = torch.arange(shape[1]) - shape[1] // 2
    return torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2).float()


def mask_block(probability, reference, available, args):
    deviation = (probability - reference) * available
    keep = available.bool()
    generator = torch.Generator().manual_seed(args.seed)
    draw = (torch.rand(reference.shape, generator=generator) < reference).float()
    control = (draw - reference) * available
    in_between = ((probability > 0.05) & (probability < 0.95) & keep).float().sum((1, 2, 3)) / keep.float().sum((1, 2, 3))
    reference_between = ((reference > 0.05) & (reference < 0.95) & keep).float().sum((1, 2, 3)) / keep.float().sum((1, 2, 3))

    radius = radius_map(reference.shape[-2:])
    energy = (deviation ** 2).sum(0)[0]
    budget = (probability * available).sum(0)[0]
    rings = {}
    for low, high in zip(RINGS[:-1], RINGS[1:]):
        ring = (radius >= low) & (radius < high)
        rings[f'{low}-{high}'] = {'deviation_energy_share': float(energy[ring].sum() / energy.sum()),
                                  'label_budget_share': float(budget[ring].sum() / budget.sum())}

    lags = {name: {'oracle_deviation': lagged_corr(deviation, keep, lag), 'bernoulli_control': lagged_corr(control, keep, lag)}
            for name, lag in LAGS.items()}
    mirrors = {'hermitian_partner': (-2, -1), 'mirror_axis0_placebo': (-2,), 'mirror_axis1_placebo': (-1,)}
    pairs = {}
    for name, axes in mirrors.items():
        both = keep & partner(keep, axes)
        pairs[name] = {'oracle_deviation': mean_corr(deviation, partner(deviation, axes), both),
                       'bernoulli_control': mean_corr(control, partner(control, axes), both)}
    return {'share_of_available_with_p_in_0.05_0.95': {'label': float(in_between.mean()),
                                                       'anatomy_mean_label': float(reference_between.mean())},
            'radial': rings, 'autocorrelation': lags, 'conjugate_pairs': pairs}


def pair_agreement(label_dir, sigma):
    """Cross-start agreement of the conjugate pair sum and difference, with each set's leave-one-out reference."""
    refits = {name: load_shards(path, 'val') for name, path in REFITS.items()}
    indices = refits['retest']['indices']
    sets = {'original': matched_rows(load_shards(label_dir, 'val'), indices)}
    sets.update({name: matched_rows(shard, indices) for name, shard in refits.items()})
    anatomies = sets['original']['anatomy']
    available = available_of(sets['original'])
    keep = available.bool() & partner(available.bool())
    deviation = {}
    for name, labels in sets.items():
        probability = label_probability(labels)
        deviation[name] = (probability - leave_one_out_reference(probability, anatomies)) * available
    result = {}
    for other in ('retest', 'init_noise', 'init_loupe'):
        first, second = deviation['original'], deviation[other]
        first_smooth, first_pixel = bands(first, keep, sigma)
        second_smooth, second_pixel = bands(second, keep, sigma)
        result[f'original_vs_{other}'] = {
            'per_location': mean_corr(first, second, keep),
            'per_location_pixel_band': mean_corr(first_pixel, second_pixel, keep),
            'per_location_smooth_band': mean_corr(first_smooth, second_smooth, keep),
            'pair_sum': mean_corr(first + partner(first), second + partner(second), keep),
            'pair_difference': mean_corr(first - partner(first), second - partner(second), keep)}
    return result


def combined_kspace(kspace, sense_maps):
    """SENSE-combined k-space in the sampler's convention (oracle_labels_r8.sampler_input without the centre mask)."""
    return fft_2d_img((ifft_2d_img(torch.view_as_complex(kspace.contiguous())) * sense_maps.conj()).sum(dim=1))


def log_mean(residual, unsampled, floor):
    """log of the residual averaged over the draws that left k unsampled ([B, 1, H, W], nan where none did)."""
    mean = torch.where(unsampled > 0, residual / unsampled.clamp(min=1), torch.full_like(residual, float('nan')))
    return torch.log(mean + floor).unsqueeze(1).cpu()


def split_half_reliability(first, second, keep, sigma):
    """Per band, the Spearman-Brown reliability of the full-draw field from the corr of its two draw halves."""
    both = keep & torch.isfinite(first) & torch.isfinite(second)
    first_smooth, first_pixel = bands(first, both, sigma)
    second_smooth, second_pixel = bands(second, both, sigma)
    halves = {'pixel_band': mean_corr(first_pixel, second_pixel, both),
              'smooth_band': mean_corr(first_smooth, second_smooth, both)}
    return {band: {'split_half': value, 'reliability': 2 * value / (1 + value)} for band, value in halves.items()}


@torch.no_grad()
def slice_fields(model, val_loader, indices, anatomy_logit, args):
    """Per slice: log residual power (unsampled draws only), log ground-truth power, ACS centre values, anatomy."""
    rows = {'log_residual': [], 'log_residual_half0': [], 'log_residual_half1': [], 'log_power': [], 'acs': [],
            'anatomy': [], 'fname': []}
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, args.chunk)):
        part = to_device(batch, args.device)
        anatomies = list(batch['anatomy'])
        logits = torch.stack([anatomy_logit[a] for a in anatomies]).to(args.device)
        residual, unsampled, target_kspace = [0.0, 0.0], [0.0, 0.0], None
        for draw in range(args.draws):
            torch.manual_seed(draw * 100003 + chunk_index)
            mask = masks_from_logits(model, logits, part['kspace'])
            output = model(part['target'], part['kspace'], anatomy_labels=part['label'], fixed_mask=mask)
            if target_kspace is None:
                target_kspace = combined_kspace(part['kspace'], output['sense_maps'])
            recon = output['recon_complex']
            recon = recon[:, 0] if recon.dim() == 4 else recon
            assert torch.is_complex(recon), 'expected a complex reconstruction'
            free = 1.0 - mask[:, 0].float()
            half = int(draw >= args.draws // 2)
            residual[half] = residual[half] + (fft_2d_img(recon) - target_kspace).abs() ** 2 * free
            unsampled[half] = unsampled[half] + free
        power = target_kspace.abs() ** 2
        floor = 1e-10 * power.mean((-2, -1), keepdim=True)
        rows['log_residual'].append(log_mean(residual[0] + residual[1], unsampled[0] + unsampled[1], floor))
        for half in (0, 1):
            rows[f'log_residual_half{half}'].append(log_mean(residual[half], unsampled[half], floor))
        rows['log_power'].append(torch.log(power + floor).unsqueeze(1).cpu())
        old_mask, _ = model._initial_masks(part['kspace'])
        centre = old_mask.bool().reshape(target_kspace.shape)
        rows['acs'].append(torch.stack([k[c] for k, c in zip(target_kspace, centre)]).cpu())
        rows['anatomy'] += anatomies
        rows['fname'] += [str(name) for name in batch['fname']]
        print(f"fields {len(rows['anatomy'])}/{len(indices)}", flush=True)
    fields = {key: torch.cat(rows[key]) for key in rows if key not in ('anatomy', 'fname')}
    fields['anatomy'], fields['fname'] = np.array(rows['anatomy']), np.array(rows['fname'])
    return fields


def anatomy_centred(field, anatomies, labelled):
    """field minus its nan-mean over the labelled slices of the same anatomy."""
    centred = field.clone()
    for anatomy in ANATOMIES:
        members = torch.as_tensor(anatomies == anatomy)
        centred[members] = field[members] - torch.nanmean(field[members & labelled], dim=0, keepdim=True)
    return centred


def next_slices(val_loader, indices):
    """For each index, the dataset index of the next slice of the same volume, or -1."""
    examples = [example for dataset in val_loader.dataset.datasets for example in dataset.examples]
    following = []
    for index in indices:
        index = int(index)
        has_next = (index + 1 < len(examples) and examples[index + 1][0] == examples[index][0]
                    and examples[index + 1][1] == examples[index][1] + 1)
        following.append(index + 1 if has_next else -1)
    return np.array(following)


def main():
    args = parse_args()
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    anatomy_logit = {a: logit_of(p) for a, p in anatomy_mean.items()}
    probability = label_probability(val)
    reference = torch.stack([anatomy_mean[a] for a in val['anatomy']])
    available = available_of(val)

    summary = {'mask': mask_block(probability, reference, available, args),
               'refit_conjugate_pair_agreement': pair_agreement(args.label_dir, args.band_sigma)}
    print(json.dumps(summary, indent=2), flush=True)

    model, _ = load_frozen(args.device)
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)
    labelled_indices = val['indices'].numpy()
    following = next_slices(val_loader, labelled_indices)
    neighbours = following[following >= 0]
    fields = slice_fields(model, val_loader, np.concatenate([labelled_indices, neighbours]), anatomy_logit, args)
    count = len(labelled_indices)
    labelled = torch.zeros(len(fields['anatomy']), dtype=torch.bool)
    labelled[:count] = True
    assert list(fields['anatomy'][:count]) == list(val['anatomy']), 'field order does not match the labels'

    keep = available.bool()
    deviation_smooth, deviation_pixel = bands((probability - reference) * available, keep, args.band_sigma)
    follows, centred = {}, {}
    for name in ('log_residual', 'log_power'):
        centred[name] = anatomy_centred(fields[name], fields['anatomy'], labelled)
        field_keep = torch.isfinite(centred[name][:count]) & keep
        smooth, pixel = bands(centred[name][:count], field_keep, args.band_sigma)
        follows[name] = {'pixel_band': mean_corr(deviation_pixel, pixel, field_keep),
                         'smooth_band': mean_corr(deviation_smooth, smooth, field_keep),
                         'unsplit': mean_corr((probability - reference) * available, centred[name][:count], field_keep)}
    halves = [anatomy_centred(fields[f'log_residual_half{h}'], fields['anatomy'], labelled)[:count] for h in (0, 1)]
    reliability = split_half_reliability(*halves, keep, args.band_sigma)
    summary['log_residual_reliability'] = reliability
    for band in ('pixel_band', 'smooth_band'):
        follows['log_residual'][f'{band}_disattenuated'] = (
            follows['log_residual'][band] / np.sqrt(reliability[band]['reliability']))
    summary['slice_fields_followed_by_oracle_deviation'] = follows

    row_of_neighbour = {int(index): count + row for row, index in enumerate(neighbours)}
    pairs = [(row, row_of_neighbour[int(index)]) for row, index in enumerate(following) if index >= 0]
    acs = fields['acs'] / fields['acs'].norm(dim=1, keepdim=True)
    neighbour = {'pairs': len(pairs),
                 'acs_complex_similarity': float(np.mean([float((acs[a].conj() * acs[b]).sum().abs()) for a, b in pairs]))}
    full_keep = keep[0]
    for name, field in centred.items():
        per_pair = {'pixel_band': [], 'smooth_band': []}
        for a, b in pairs:
            both = torch.isfinite(field[a]) & torch.isfinite(field[b]) & full_keep
            smooth_a, pixel_a = bands(field[a:a + 1], both[None], args.band_sigma)
            smooth_b, pixel_b = bands(field[b:b + 1], both[None], args.band_sigma)
            per_pair['pixel_band'].append(corr(pixel_a[0], pixel_b[0], both))
            per_pair['smooth_band'].append(corr(smooth_a[0], smooth_b[0], both))
        neighbour[name] = {band: float(np.nanmean(values)) for band, values in per_pair.items()}
    for band in ('pixel_band', 'smooth_band'):
        neighbour['log_residual'][f'{band}_disattenuated'] = (
            neighbour['log_residual'][band] / reliability[band]['reliability'])
    summary['adjacent_slice_similarity'] = neighbour

    print(json.dumps(summary, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.write_text(json.dumps({'summary': summary, 'config': config}, indent=2) + '\n')


if __name__ == '__main__':
    main()
