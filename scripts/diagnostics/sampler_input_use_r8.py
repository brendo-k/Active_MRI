"""Does the 1-step conv_2d sampler's mask depend on its input, and how much of the across-slice variation is the network's?

All quantities are probability maps (no Bernoulli draws), computed through `Sampler.unscaled_probabilities`
so `image_input` switches the domain exactly as in training. For each checkpoint and anatomy:

  input diversity      how alike the sampler's network inputs are across slices: mean correlation of a slice's
                       input with its anatomy's mean input, the brain-vs-knee mean-input correlation, and the
                       spread (coefficient of variation) of input norms
  output diversity     the same correlation / across-slice std for the unscaled scores and the effective
                       probability (rescaled + pre-selected centre)
  variation split      across-slice std of the effective probability when the network's scores are replaced by
                       the anatomy-mean scores (what the per-slice valid mask and budget rescaling alone
                       produce), against the real one
  perturbation         mean |change in scores| / mean score when the input is zeroed, scaled x0.5 / x2, or
                       replaced by the anatomy-mean input, against the real slice-to-slice deviation
  saturation           fraction of scores beyond the soft clamp (|logit| > 6, p < 0.0025), where the
                       gradient is only the 0.01 leak

    python scripts/diagnostics/sampler_input_use_r8.py
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
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img as torch_fft2, ifft_2d_img as torch_ifft2

RUNS = {
    'baseline': Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05'),
    'image_input': Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput'),
    'draws2': Path('checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_draws2'),
}
OUT = Path('reports/conv_vs_loupe_r8/sampler_input_use.json')
ANATOMIES = ('brain', 'knee')
CLAMPED_PROBABILITY = 0.0025  # sigmoid(-6)
CHUNK = 16


def load_model(run_dir, device):
    checkpoint = torch.load(run_dir / 'last_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['options'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), int(checkpoint['epoch']) + 1


@torch.no_grad()
def collect(model, val_loader, device):
    """Per validation slice: the sampler's k-space input, pre-selected centre, valid region, anatomy and label."""
    rows = {key: [] for key in ('kspace_in', 'old_mask', 'valid_mask', 'anatomy', 'label')}
    for batch in val_loader:
        full_kspace = batch['kspace'].to(device)
        sense_maps, _ = model.sensetivity_net(torch.view_as_complex(full_kspace.contiguous()),
                                              mask_method=model.sense_mask_method)
        old_mask, valid_mask = model._initial_masks(full_kspace)
        observed = torch.view_as_complex((full_kspace * old_mask.unsqueeze(-1)).contiguous())
        combined = (torch_ifft2(observed) * sense_maps).sum(dim=1)
        rows['kspace_in'].append(torch.view_as_real(torch_fft2(combined)).cpu())
        rows['old_mask'].append(old_mask.cpu())
        rows['valid_mask'].append(valid_mask.cpu())
        rows['anatomy'] += list(batch['anatomy'])
        labels = batch.get('anatomy_labels')
        rows['label'].append(labels.cpu() if torch.is_tensor(labels) else None)
    labels = rows['label']
    return {'kspace_in': torch.cat(rows['kspace_in']), 'old_mask': torch.cat(rows['old_mask']),
            'valid_mask': torch.cat(rows['valid_mask']), 'anatomy': np.array(rows['anatomy']),
            'label': torch.cat(labels) if all(l is not None for l in labels) else None}


def network_input(sampler, kspace_in, old_mask):
    channels = torch.cat([kspace_in, kspace_in * old_mask.permute(0, 2, 3, 1)], dim=-1)
    return sampler._to_image_domain(channels) if sampler.image_domain_input else channels


@torch.no_grad()
def scores(model, data, device, rows=None, kspace_in=None):
    """Unscaled scores [n, H, W] for the stored slices (or the `rows` subset); `kspace_in` replaces their input, aligned with `rows`."""
    rows = torch.arange(len(data['kspace_in'])) if rows is None else torch.as_tensor(rows)
    source = data['kspace_in'][rows] if kspace_in is None else kspace_in
    old_mask = data['old_mask'][rows]
    labels = data['label'][rows] if data['label'] is not None else None
    out = []
    for position in torch.arange(len(rows)).split(CHUNK):
        unscaled, _, _ = model.sampler.unscaled_probabilities(
            source[position].to(device), old_mask[position].to(device),
            anatomy_labels=labels[position].to(device) if labels is not None else None)
        out.append(unscaled[:, 0].cpu())
    return torch.cat(out)


@torch.no_grad()
def effective(model, unscaled, data, device):
    """Effective probability: budget-rescaled scores over the available region plus the centre."""
    out = []
    for chunk in torch.arange(len(unscaled)).split(CHUNK):
        old_mask = data['old_mask'][chunk].to(device)
        rescaled, _ = model.sampler.select_locations(
            unscaled[chunk].unsqueeze(1).to(device), old_mask, model.budget, data['valid_mask'][chunk].to(device))
        out.append(torch.clamp(rescaled + old_mask, max=1)[:, 0].cpu())
    return torch.cat(out)


def correlation(a, b):
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def mean_correlation_with_mean(maps):
    centre = maps.mean(0)
    return float(np.mean([correlation(m, centre) for m in maps]))


def analyse(model, data, device):
    effective_real = effective(model, scores(model, data, device), data, device)
    unscaled_real = scores(model, data, device)
    result = {'effective_overall_std': float(effective_real.std(0).mean())}
    anatomy_mean_input, anatomy_mean_scores = {}, {}
    for anatomy in ANATOMIES:
        rows = np.where(data['anatomy'] == anatomy)[0]
        inputs = network_input(model.sampler, data['kspace_in'][rows], data['old_mask'][rows]).numpy()
        norms = np.linalg.norm(inputs.reshape(len(rows), -1), axis=1)
        anatomy_mean_input[anatomy] = data['kspace_in'][rows].mean(0, keepdim=True)
        anatomy_mean_scores[anatomy] = unscaled_real[rows].mean(0)
        unscaled, eff = unscaled_real[rows].numpy(), effective_real[rows].numpy()

        replaced = effective(model, anatomy_mean_scores[anatomy].expand(len(rows), -1, -1).contiguous(),
                             {k: (v[rows] if torch.is_tensor(v) else v) for k, v in data.items()}, device)
        deviation = float(np.abs(unscaled - unscaled.mean(0)).mean() / unscaled.mean())

        def response(replacement):
            changed = scores(model, data, device, rows=rows, kspace_in=replacement)
            return float((changed - unscaled_real[rows]).abs().mean() / unscaled_real[rows].mean())

        kspace_in = data['kspace_in'][rows]
        result[anatomy] = {
            'slices': len(rows),
            'input': {'slice_vs_mean_corr': mean_correlation_with_mean(inputs),
                      'norm_cv': float(norms.std() / norms.mean())},
            'unscaled': {'across_slice_std': float(unscaled.std(0).mean()),
                         'slice_vs_mean_corr': mean_correlation_with_mean(unscaled),
                         'frac_below_0.01': float((unscaled < 0.01).mean()),
                         'frac_clamped': float((unscaled < CLAMPED_PROBABILITY * 1.05).mean())},
            'effective': {'across_slice_std': float(eff.std(0).mean()),
                          'slice_vs_mean_corr': mean_correlation_with_mean(eff)},
            'effective_std_with_mean_scores': float(replaced.std(0).mean()),
            'real_slice_deviation_rel': deviation,
            'response_rel': {'zero_input': response(torch.zeros_like(kspace_in)),
                             'half_input': response(kspace_in * 0.5),
                             'double_input': response(kspace_in * 2.0),
                             'anatomy_mean_input': response(anatomy_mean_input[anatomy].expand_as(kspace_in).contiguous())},
        }
    mean_inputs = {a: network_input(model.sampler, anatomy_mean_input[a], data['old_mask'][:1]).numpy() for a in ANATOMIES}
    result['brain_vs_knee_mean_input_corr'] = correlation(mean_inputs['brain'], mean_inputs['knee'])
    result['brain_vs_knee_mean_scores_corr'] = correlation(anatomy_mean_scores['brain'].numpy(), anatomy_mean_scores['knee'].numpy())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()

    cfg = OmegaConf.load(RUNS['baseline'] / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    _, val_loader = build_dataloaders(cfg)

    results = {}
    for name, run_dir in RUNS.items():
        model, epochs = load_model(run_dir, args.device)
        data = collect(model, val_loader, args.device)
        results[name] = {'epochs_trained': epochs, **analyse(model, data, args.device)}
        print(name, json.dumps(results[name], indent=2), flush=True)
        del model, data
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
