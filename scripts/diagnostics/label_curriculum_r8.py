"""Can any network fit the per-slice oracle labels of oracle_labels_r8.py? Follow-up to label_predictor_gonogo_r8.py,
whose fine-tuned sampler U-Net reached a deviation correlation of only ~0.1-0.15 even on its own training rows.

Changes against that predictor:
  - a fresh U-Net predicts only the deviation from the anatomy-mean label: logit = logit(anatomy mean) + delta,
    with delta zero at initialisation, so the shared pattern is given rather than learned;
  - a scale-free loss on that deviation: per slice, sum(available * (p - t)^2) / sum(available * (r - t)^2),
    with r the anatomy mean and t the target (1 at initialisation, 0 for a perfect fit);
  - normalised inputs (see INPUTS), from the observed 10x10 centre or (`--full`, privileged) the fully sampled
    SENSE-combined k-space;
  - a coarse-to-fine curriculum: the target deviation is Gaussian-blurred in logit space, with sigma stepping
    through `--sigmas` (`--epochs-per-stage` each).

After each stage it reports, per sigma-blurred target and for the unblurred label, the deviation correlation
of label_predictor_gonogo_r8.py on the training rows it fitted and on the validation labels, and the retest
ceiling at that blur (two seeds' labels of the same validation slices). `--memorise n` fits only n training
rows: a network that cannot drive their correlation towards the ceiling points at the pipeline, not the objective.

    python scripts/diagnostics/label_curriculum_r8.py --input acs_image --memorise 64
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

import _path  # noqa: F401
from activemri.reconstructors.UNet import LOUPEUNet
from activemri.training.builders import build_dataloaders
from activemri.utils.image_processing import ifft_2d_img
from label_predictor_gonogo_r8 import (ANATOMIES, anatomy_reference, available_of, deviation_correlation,
                                       full_sampler_input, label_probability, load_shards, logit_of)
from oracle_labels_r8 import OUT_DIR, RUN, load_frozen

INPUTS = {
    'kspace': 'the stored SENSE-combined k-space (real, imag), unnormalised, as the sampler sees it',
    'kspace_log': 'log k-space magnitude standardised per slice, with the phase as (cos, sin)',
    'acs_image': 'the inverse FFT of that k-space (the low-resolution coil-combined ACS image with --full off), '
                 'scaled by its per-slice peak magnitude, as (real, imag, magnitude)',
}
INPUT_CHANNELS = {'kspace': 2, 'kspace_log': 3, 'acs_image': 3}
OUT = Path('analysis/label_gonogo/curriculum')
RETEST_DIR = Path('/tmp/bkadota_oracle_labels_r8_retest')
CHECK_ROWS = 128


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=None, help=f'default {OUT}/<run name>.json')
    parser.add_argument('--input', choices=tuple(INPUTS), default='acs_image')
    parser.add_argument('--full', action='store_true', help='privileged: the fully sampled k-space instead of the centre')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[8, 4, 2, 1, 0],
                        help='target blur (pixels) of each curriculum stage; a single 0 is no curriculum')
    parser.add_argument('--epochs-per-stage', type=int, default=8)
    parser.add_argument('--memorise', type=int, default=None, help='fit only this many training rows')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def run_name(args):
    stages = '-'.join(f'{s:g}' for s in args.sigmas)
    memorise = f'_mem{args.memorise}' if args.memorise else ''
    return f"{args.input}{'_full' if args.full else ''}_s{stages}{memorise}"


def network_input(kspace, input_kind):
    """[N, H, W, 2] SENSE-combined k-space -> [N, C, H, W] network channels (see INPUTS)."""
    kspace = kspace.float()
    if input_kind == 'kspace':
        return kspace.permute(0, 3, 1, 2)
    complex_kspace = torch.view_as_complex(kspace.contiguous())
    if input_kind == 'kspace_log':
        log_magnitude = complex_kspace.abs().clamp(min=1e-8).log()
        mean = log_magnitude.mean(dim=(1, 2), keepdim=True)
        std = log_magnitude.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        phase = complex_kspace.angle()
        return torch.stack([(log_magnitude - mean) / std, phase.cos(), phase.sin()], dim=1)
    image = ifft_2d_img(complex_kspace, axes=[1, 2])
    image = image / image.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    return torch.stack([image.real, image.imag, image.abs()], dim=1)


def gaussian_blur(field, sigma):
    """Separable Gaussian blur of [N, 1, H, W] with reflect padding; sigma 0 is the identity."""
    if sigma <= 0:
        return field
    radius = int(np.ceil(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    field = F.conv2d(F.pad(field, (radius, radius, 0, 0), mode='reflect'), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(field, (0, 0, radius, radius), mode='reflect'), kernel.view(1, 1, -1, 1))


def blurred_target(label_logit, reference_logit, sigma):
    return torch.sigmoid(reference_logit + gaussian_blur(label_logit - reference_logit, sigma))


class DeviationPredictor(torch.nn.Module):
    def __init__(self, in_chans, chans, depth):
        super().__init__()
        self.unet = LOUPEUNet(in_chans=in_chans + 1, out_chans=1, chans=chans, depth=depth, drop_prob=0,
                              upsample_mode='bilinear')
        torch.nn.init.zeros_(self.unet.final_conv.weight)
        torch.nn.init.zeros_(self.unet.final_conv.bias)

    def forward(self, channels, old_mask, reference_logit):
        return torch.sigmoid(reference_logit + self.unet(torch.cat([channels, old_mask], dim=1)))


def deviation_loss(probability, target, reference, available):
    unexplained = (available * (probability - target) ** 2).flatten(1).sum(1)
    total = (available * (reference - target) ** 2).flatten(1).sum(1).clamp(min=1e-12)
    return (unexplained / total).mean()


class Rows:
    """One split's labels on the CPU, with everything the predictor needs per row."""

    def __init__(self, labels, channels, anatomy_mean):
        self.channels = channels
        self.old_mask = labels['old_mask'].float()
        self.available = available_of(labels)
        self.label_logit = logit_of(label_probability(labels))
        self.reference = anatomy_reference(anatomy_mean, labels['anatomy'])
        self.reference_logit = logit_of(self.reference)

    def batch(self, rows, device):
        return [x[rows].to(device) for x in (self.channels, self.old_mask, self.available, self.label_logit,
                                             self.reference, self.reference_logit)]


@torch.no_grad()
def predict(net, data, rows, device):
    net.eval()
    out = []
    for part in torch.as_tensor(rows).split(32):
        channels, old_mask, _, _, _, reference_logit = data.batch(part, device)
        out.append(net(channels, old_mask, reference_logit).cpu())
    return torch.cat(out)


def blurred_targets(data, rows, sigma):
    return torch.cat([blurred_target(data.label_logit[part], data.reference_logit[part], sigma)
                      for part in torch.as_tensor(rows).split(64)])


def report(net, data, rows, sigma, device):
    """Deviation correlation and loss of the predictions on `rows`, against the sigma-blurred target and the label."""
    rows = np.asarray(rows)
    probability = predict(net, data, rows, device)
    reference, available = data.reference[rows], data.available[rows]
    result = {}
    for name, blur in (('blurred', sigma), ('label', 0)):
        target = blurred_targets(data, rows, blur)
        result[name] = {'correlation': deviation_correlation(probability, reference, target, available),
                        'loss': float(deviation_loss(probability, target, reference, available))}
    return result


def retest_ceiling(retest, val, sigma):
    """Deviation correlation between the sigma-blurred labels of two seeds on the same validation slices."""
    row_of = {int(index): row for row, index in enumerate(val['indices'])}
    rows = np.array([row_of[int(index)] for index in retest['indices']])
    reference_logit = val['data'].reference_logit[rows]
    retest_logit = logit_of(label_probability(retest))
    first = blurred_target(val['data'].label_logit[rows], reference_logit, sigma)
    second = blurred_target(retest_logit, reference_logit, sigma)
    return deviation_correlation(second, val['data'].reference[rows], first, val['data'].available[rows])


def split_inputs(args, labels, loader, model):
    kspace = (full_sampler_input(model, loader, labels['indices'].numpy(), args.device, args.chunk)
              if args.full else labels['kspace_in'])
    return torch.cat([network_input(part, args.input) for part in kspace.split(64)])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    out = args.out or OUT / f'{run_name(args)}.json'
    train, val = load_shards(args.label_dir, 'train'), load_shards(args.label_dir, 'val')
    model = loaders = None
    if args.full:
        model, _ = load_frozen(args.device)
        cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
        cfg.datasets.num_workers = 4
        loaders = build_dataloaders(cfg)
    train_probability = label_probability(train)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    train_data = Rows(train, split_inputs(args, train, loaders and loaders[0], model), anatomy_mean)
    val_data = Rows(val, split_inputs(args, val, loaders and loaders[1], model), anatomy_mean)
    del model
    retest = load_shards(RETEST_DIR, 'val') if RETEST_DIR.exists() else None

    order = rng.permutation(len(train['label_logits']))
    fit_rows = order[:args.memorise] if args.memorise else order
    check_rows = fit_rows[:CHECK_ROWS]
    val_rows = np.arange(len(val['label_logits']))
    print(f"{run_name(args)}: fitting {len(fit_rows)} training rows, input channels "
          f"{tuple(train_data.channels.shape[1:])}", flush=True)

    net = DeviationPredictor(INPUT_CHANNELS[args.input], args.chans, args.depth).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    stages = []
    for sigma in args.sigmas:
        trace = []
        for epoch in range(args.epochs_per_stage):
            net.train()
            losses = []
            for rows in torch.as_tensor(rng.permutation(fit_rows)).split(args.batch_size):
                channels, old_mask, available, label_logit, reference, reference_logit = train_data.batch(rows, args.device)
                target = blurred_target(label_logit, reference_logit, sigma)
                loss = deviation_loss(net(channels, old_mask, reference_logit), target, reference, available)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
            trace.append(float(np.mean(losses)))
            print(f"sigma {sigma:g} epoch {epoch + 1}: train loss {trace[-1]:.4f}", flush=True)
        stage = {'sigma': sigma, 'train_loss': trace,
                 'fit_rows': report(net, train_data, check_rows, sigma, args.device),
                 'val': report(net, val_data, val_rows, sigma, args.device),
                 'retest_ceiling': retest_ceiling(retest, {'indices': val['indices'], 'data': val_data}, sigma)
                 if retest is not None else None}
        print('stage', json.dumps({k: v for k, v in stage.items() if k != 'train_loss'}), flush=True)
        stages.append(stage)

    out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out.write_text(json.dumps({'stages': stages, 'config': config}, indent=2) + '\n')


if __name__ == '__main__':
    main()
