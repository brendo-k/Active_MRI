"""Why is the conv_2d sampler's mask smooth? Where the smoothness comes from, and what would remove it.

On the trained image-input conv sampler (checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput) and its matched
LOUPE run, fed what they trained on: T2 + knee at 256x256 and the old sampler input sum_c S_c IFFT(k_c) (before the
conj fix). CPU only.

  decompose   the logits are final_conv(F) + input[:, 0] (LOUPEUNet's residual): the share of each term, and the
              high-frequency fraction (spectral power above 1/8 cycles/px) of the input, the last features F (what the
              1x1 final conv mixes) and the logits; the repo's roughness (compare_conv_loupe_masks_r8.py) of the
              effective probability
  filter      one linearised gradient step on the logits towards a requested per-pixel change d (white noise, or a
              Gaussian random field of length sigma): correlation of the realised change with d, and its
              high-frequency fraction. LOUPE realises d exactly (correlation 1). Arms: the trained net (all weights /
              the 1x1 final conv only), the same architecture at random init, and at random init with Fourier-feature
              coordinate channels. float64 throughout, so finite differences don't add high frequencies
  fit         supervised regression of LOUPE's logit map from the same inputs (Adam, MSE outside the centre), for the
              plain U-Net, the U-Net with Fourier-feature coordinate channels and the U-Net with head=bias, all from
              random init: R^2 overall and of the pixel band (the map minus its 2 px Gaussian blur)

    python scripts/diagnostics/conv_sampler_smoothness_r8.py --part decompose

Any other conv/LOUPE pair, fed the input its own training used (`--input model`: propose_mask, so the run's coil
combination and budget), e.g. the RSS-input patient-specific pair on local v1 FLAIR + knee:

    RUNS=~/data/seqmri_checkpoints/patient_specific_r8_local_v1/runs
    python scripts/diagnostics/conv_sampler_smoothness_r8.py --part decompose --input model \
        --conv-run $RUNS/patient_specific_conv_r8_flair_knee_100ep_seed42 \
        --loupe-run $RUNS/patient_specific_loupe_r8_flair_knee_100ep_seed42 \
        --dataset fastmri_local_v1_flair_knee --resolution 320 --out reports/conv_sampler_smoothness_r8/rss_local_v1
"""
import argparse
import json
import math
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch import nn

import _path  # noqa: F401
from activemri.reconstructors.UNet import LOUPEUNet
from activemri.samplers.conv import LOCATION_BIAS_SCALE, scores_to_prob
from activemri.training.builders import build_dataloaders
from activemri.training.trainer import build_model
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img

REPO = Path(__file__).resolve().parents[2]
CONV_RUN = REPO / 'checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput'
LOUPE_RUN = REPO / 'checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05'
OUT = REPO / 'reports/conv_sampler_smoothness_r8'
HIGH_FREQUENCY = 1 / 8  # cycles per pixel: structure finer than 8 px
FOURIER_FREQUENCIES = (1, 2, 4, 8, 16, 32, 64, 128)  # cycles per grid side
DIRECTION_SIGMAS = (0, 1, 2, 4, 8)  # px; 0 = white noise
PIXEL_BAND_SIGMA = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--part', choices=('decompose', 'filter', 'fit'), required=True)
    parser.add_argument('--slices', type=int, default=8, help='per anatomy')
    parser.add_argument('--fit-steps', type=int, default=400)
    parser.add_argument('--fit-arm', choices=('plain', 'fourier', 'bias'), default=None, help='fit: one arm only')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--conv-run', type=Path, default=CONV_RUN)
    parser.add_argument('--loupe-run', type=Path, default=LOUPE_RUN)
    parser.add_argument('--dataset', default='brain_knee_balanced')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--input', choices=tuple(SAMPLER_INPUTS), default='legacy',
                        help="legacy: the pre-conj sum_c S_c IFFT(k_c) the default checkpoint trained on; "
                             "model: the run's own input, built by propose_mask")
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def load(run, device):
    checkpoint = torch.load(run / 'last_model.pt', map_location='cpu', weights_only=False)
    options = checkpoint['options']
    options.sens_precision = 'fp32'
    options.recon.unet_precision = 'fp32'
    model = build_model(options)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval().requires_grad_(False)


def val_slices(per_anatomy, dataset_name, resolution):
    with initialize_config_dir(config_dir=str(REPO / 'configs'), version_base=None):
        cfg = compose('config', overrides=[f'datasets={dataset_name}', f'misc.resolution=[{resolution},{resolution}]',
                                           'datasets.num_workers=0'])
    _, val_loader = build_dataloaders(cfg)
    dataset, picks, offset = val_loader.dataset, [], 0
    for part in dataset.datasets:
        picks += [offset + int(i) for i in torch.linspace(0, len(part) - 1, per_anatomy).long()]
        offset += len(part)
    return val_loader.collate_fn([dataset[i] for i in picks])


@torch.no_grad()
def legacy_sampler_input(model, kspace):
    """The conv sampler's [B, 5, H, W] input as the checkpoint trained on it: the old combination sum_c S_c IFFT(k_c)."""
    sense_maps, _ = model.sensetivity_net(torch.view_as_complex(kspace.contiguous()), mask_method=model.sense_mask_method)
    old_mask, valid = model._initial_masks(kspace)
    observed = torch.view_as_complex((kspace * old_mask.unsqueeze(-1)).contiguous())
    combined = torch.view_as_real(fft_2d_img((ifft_2d_img(observed) * sense_maps).sum(dim=1)))
    stacked = torch.cat([combined, combined * old_mask.permute(0, 2, 3, 1)], dim=-1)
    if model.sampler.image_domain_input:
        stacked = model.sampler._to_image_domain(stacked)
    return torch.cat([stacked.permute(0, 3, 1, 2), old_mask], dim=1), old_mask, valid


@torch.no_grad()
def model_sampler_input(model, kspace):
    """The U-Net's input exactly as the run's training built it (propose_mask)."""
    captured = []
    hook = model.sampler.mask_net.unet.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    proposal = model.propose_mask(kspace)
    hook.remove()
    return captured[0], proposal['old_mask'], proposal['valid_mask']


SAMPLER_INPUTS = {'legacy': legacy_sampler_input, 'model': model_sampler_input}


class FourierUNet(nn.Module):
    """The sampler's U-Net with sin/cos coordinate channels of the k-space position appended to its input."""

    def __init__(self, shape, chans=64, depth=4):
        super().__init__()
        coordinates = [torch.arange(n, dtype=torch.float64) / n for n in shape]
        channels = []
        for f in FOURIER_FREQUENCIES:
            for axis, c in enumerate(coordinates):
                for wave in (torch.sin, torch.cos):
                    values = wave(2 * math.pi * f * c)
                    channels.append(values.view(-1, 1).expand(shape) if axis == 0 else values.view(1, -1).expand(shape))
        self.register_buffer('coordinates', torch.stack(channels).unsqueeze(0).float())
        self.unet = LOUPEUNet(5 + len(channels), 1, chans, depth, 0, upsample_mode='bilinear')

    def forward(self, x):
        return self.unet(torch.cat([x, self.coordinates.to(x.dtype).expand(len(x), -1, -1, -1)], dim=1))


class BiasHeadUNet(nn.Module):
    """The sampler's U-Net plus head=bias: a free logit per location, scaled as LocationHead does."""

    def __init__(self, shape, chans=64, depth=4):
        super().__init__()
        self.unet = LOUPEUNet(5, 1, chans, depth, 0, upsample_mode='bilinear')
        self.bias = nn.Parameter(torch.zeros(1, 1, *shape))

    def forward(self, x):
        return self.unet(x) + LOCATION_BIAS_SCALE * self.bias


def plain_unet(chans=64, depth=4):
    return LOUPEUNet(5, 1, chans, depth, 0, upsample_mode='bilinear')


def high_frequency_fraction(maps, cutoff=HIGH_FREQUENCY):
    """Fraction of the mean-removed spectral power of [N, H, W] maps above `cutoff` cycles/px (white noise: ~0.95)."""
    maps = maps - maps.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.fft2(maps.double()).abs().pow(2)
    fy = torch.fft.fftfreq(maps.shape[-2], dtype=torch.float64, device=maps.device)
    fx = torch.fft.fftfreq(maps.shape[-1], dtype=torch.float64, device=maps.device)
    radius = torch.hypot(fy[:, None], fx[None, :])
    return float((power * (radius > cutoff)).sum() / power.sum())


def roughness(prob, region=None):
    """compare_conv_loupe_masks_r8.roughness: mean |4-neighbour difference| over the map's mean, averaged over maps.

    With a `region`, only neighbour pairs inside it count, so the edge of a zero-filled extent is not roughness."""
    region = torch.ones_like(prob) if region is None else region.expand_as(prob).to(prob.dtype)
    terms = []
    for dim in (-1, -2):
        inside = region.narrow(dim, 1, prob.shape[dim] - 1) * region.narrow(dim, 0, prob.shape[dim] - 1)
        terms.append((prob.diff(dim=dim).abs() * inside).sum(dim=(-2, -1)) / inside.sum(dim=(-2, -1)))
    mean = (prob * region).sum(dim=(-2, -1)) / region.sum(dim=(-2, -1))
    return float(((terms[0] + terms[1]) / 2 / mean).mean())


def gaussian_blur(maps, sigma):
    if sigma == 0:
        return maps
    radius = int(math.ceil(3 * sigma))
    taps = torch.exp(-torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device) ** 2 / (2 * sigma ** 2))
    taps = taps / taps.sum()
    flat = maps.reshape(-1, 1, *maps.shape[-2:])
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode='circular'), taps.view(1, 1, 1, -1))
    flat = F.conv2d(F.pad(flat, (0, 0, radius, radius), mode='circular'), taps.view(1, 1, -1, 1))
    return flat.view_as(maps)


def correlation(a, b, region):
    a, b = a[region > 0], b[region > 0]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def effective_probability(model, probability, old_mask, valid):
    budget = model.mask_budget(old_mask, valid)
    rescaled, _ = model.sampler.select_locations(probability * (1 - old_mask), old_mask, budget, valid)
    return (rescaled + old_mask)[:, 0]


def decompose_group(conv, loupe, x, old_mask, valid):
    unet = conv.sampler.mask_net.unet
    features = []
    hook = unet.final_conv.register_forward_hook(lambda module, inputs, output: features.append(inputs[0]))
    logits = conv.sampler.mask_net._located(unet(x))
    hook.remove()
    network_term = unet.final_conv(features[0])
    residual_term = x[:, :1]
    weights = unet.final_conv.weight.abs().flatten()
    per_channel = torch.tensor([high_frequency_fraction(features[0][:, c]) for c in range(features[0].shape[1])],
                               device=weights.device)
    loupe_logits = loupe.sampler.mask_net.raw_logits().unsqueeze(1) * loupe.sampler.mask_net.slope
    total = logits.var()
    return {
        'logit_variance_share': {'residual_input_term': float(residual_term.var() / total),
                                 'network_term': float(network_term.var() / total),
                                 'corr_residual_vs_logits': correlation(residual_term, logits, torch.ones_like(logits))},
        'high_frequency_fraction': {
            'white_noise_reference': high_frequency_fraction(torch.randn(4, *logits.shape[-2:])),
            'input_image_real': high_frequency_fraction(x[:, 0]),
            'last_features_weighted_by_final_conv': float((per_channel * weights).sum() / weights.sum()),
            'last_features_max_channel': float(per_channel.max()),
            'network_term': high_frequency_fraction(network_term[:, 0]),
            'logits': high_frequency_fraction(logits[:, 0]),
            'loupe_logits': high_frequency_fraction(loupe_logits[:, 0]),
        },
        'roughness_effective_probability_in_acquired': {
            'conv': roughness(effective_probability(conv, scores_to_prob(logits), old_mask, valid), valid[:, 0]),
            'loupe': roughness(effective_probability(loupe, torch.sigmoid(loupe_logits).expand_as(logits), old_mask, valid),
                               valid[:, 0]),
        },
        'logit_corr_between_slices_in_acquired': mean_pairwise_correlation(logits[:, 0], (valid * (1 - old_mask))[:, 0]),
    }


def mean_pairwise_correlation(maps, region):
    """Mean correlation between every pair of slices' maps over the locations inside both regions (1 = static)."""
    pairs = [correlation(maps[i], maps[j], region[i] * region[j])
             for i in range(len(maps)) for j in range(i + 1, len(maps))]
    return sum(pairs) / len(pairs)


def decompose(args, conv, loupe, batch, observation):
    x, old_mask, valid = observation
    groups = {'all': list(range(len(x)))}
    for index, anatomy in enumerate(batch['anatomy']):
        groups.setdefault(anatomy, []).append(index)
    return {name: decompose_group(conv, loupe, x[rows], old_mask[rows], valid[rows]) for name, rows in groups.items()}


def realised_change(z_fn, params, x, direction, region):
    """Logit change of one linearised gradient step towards `direction` (finite difference, float64)."""
    z = z_fn(x)
    grads = torch.autograd.grad((z * direction * region).sum(), params)
    norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    eps = 1e-3 / float(norm)
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(eps * g)
        plus = z_fn(x)
        for p, g in zip(params, grads):
            p.sub_(2 * eps * g)
        minus = z_fn(x)
        for p, g in zip(params, grads):
            p.add_(eps * g)
    return (plus - minus) / (2 * eps)


def filter_part(args, conv, loupe, batch, observation):
    x, old_mask, valid = observation
    x, region = x[:2].double(), ((1 - old_mask[:2]) * valid[:2]).double()
    torch.manual_seed(0)
    trained = conv.sampler.mask_net.double().requires_grad_(True)
    arms = {
        'trained_all_weights': (lambda v: trained._located(trained.unet(v)), list(trained.parameters())),
        'trained_final_1x1_conv_only': (lambda v: trained._located(trained.unet(v)), list(trained.unet.final_conv.parameters())),
    }
    random_plain = plain_unet().double().to(x.device)
    arms['random_init_all_weights'] = (random_plain, list(random_plain.parameters()))
    random_fourier = FourierUNet(tuple(x.shape[-2:])).double().to(x.device)
    arms['random_init_fourier_features'] = (random_fourier, list(random_fourier.parameters()))
    loupe_mask = loupe.sampler.mask_net.double().requires_grad_(True)
    arms['loupe'] = (lambda v: loupe_mask.slope * loupe_mask.raw_logits().unsqueeze(1).expand(len(v), -1, -1, -1),
                     [loupe_mask.mask])
    generator = torch.Generator().manual_seed(1)
    white = torch.randn(x.shape[0], 1, *x.shape[-2:], generator=generator, dtype=torch.float64).to(x.device)
    result = {}
    for sigma in DIRECTION_SIGMAS:
        direction = gaussian_blur(white, sigma)
        direction = direction / direction.std()
        row = {'direction_high_frequency_fraction': high_frequency_fraction((direction * region)[:, 0])}
        for name, (z_fn, params) in arms.items():
            change = realised_change(z_fn, params, x, direction, region)
            row[name] = {'corr_with_requested': correlation(change, direction, region.expand_as(change)),
                         'high_frequency_fraction': high_frequency_fraction((change * region)[:, 0])}
        result[f'sigma_{sigma}px'] = row
        print(sigma, json.dumps(row), flush=True)
    return result


def fit_part(args, conv, loupe, batch, observation):
    x, old_mask, valid = observation
    x, region = x[:4], (1 - old_mask[:4]) * valid[:4]
    target = (loupe.sampler.mask_net.slope * loupe.sampler.mask_net.raw_logits()).clamp(-6, 6)
    target = target.unsqueeze(1).expand(len(x), -1, -1, -1)
    target_pixel_band = target - gaussian_blur(target, PIXEL_BAND_SIGMA)
    makers = {'plain': plain_unet, 'fourier': lambda: FourierUNet(tuple(x.shape[-2:])), 'bias': lambda: BiasHeadUNet(tuple(x.shape[-2:]))}
    result = {}
    for name in [args.fit_arm] if args.fit_arm else list(makers):
        torch.manual_seed(0)
        net = makers[name]().to(x.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        trace = {}
        for step in range(args.fit_steps + 1):
            z = net(x)
            loss = ((z - target) ** 2 * region).sum() / region.sum()
            if step % 50 == 0 or step == args.fit_steps:
                with torch.no_grad():
                    fitted_band = z - gaussian_blur(z, PIXEL_BAND_SIGMA)
                    r2 = 1 - float(((z - target) ** 2 * region).sum() / (((target - (target * region).sum() / region.sum()) ** 2) * region).sum())
                    r2_band = 1 - float(((fitted_band - target_pixel_band) ** 2 * region).sum() / (target_pixel_band ** 2 * region).sum())
                trace[step] = {'r2': r2, 'r2_pixel_band': r2_band, 'loss': float(loss)}
                print(name, step, trace[step], flush=True)
            if step < args.fit_steps:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        result[name] = trace
    return result


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    conv, loupe = load(args.conv_run, args.device), load(args.loupe_run, args.device)
    batch = val_slices(args.slices, args.dataset, args.resolution)
    observation = SAMPLER_INPUTS[args.input](conv, batch['kspace'].to(args.device))
    part = {'decompose': decompose, 'filter': filter_part, 'fit': fit_part}[args.part]
    result = part(args, conv, loupe, batch, observation)
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.part + (f'_{args.fit_arm}' if args.fit_arm else '')
    (args.out / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
