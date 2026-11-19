"""seqmri_2d: SeqMRI's 2D sampler (Yin et al. 2021, arXiv 2105.06460) on multi-coil data.

The original is this repo's root commit cf503ae, activemri/baselines/sequential_sampling_codes/sampler2d.py. One U-Net,
shared by every step, reads five channels: the real and imaginary parts of the k-space the latest in-loop reconstruction
predicts, then of the k-space measured so far, then the mask. Its output goes through a softplus, is divided by its
per-slice maximum and is zeroed where the mask has sampled. The sampler then rescales it to the step's budget and draws
it with the straight-through threshold every sampler uses, as SeqMRI does (RescaleProbMap, ThresholdRandomMaskSigmoidV1).

SeqMRI's data is single-coil. Here the coil k-space enters as one coil: the first of `acs_compression`'s virtual coils,
the ACS's top coil eigenvector phase-referenced at the ACS k-space peak. Measured locations keep their values and the
rest stay 0, as in SeqMRI's input. Both k-spaces are divided by the ACS k-space peak; SeqMRI divided its data by one
training-set constant.

Two fscr1 options (reports/multistep_adaptivity/addendum_2026-11-16_fscr1.md):
- `input_scale='signed_log'` passes each real and imaginary part v through sign(v) log1p(|v| / f) / log1p(1 / f), with
  the log-magnitude channels' floor f = 1e-6. Over the ACS peak, k-space beyond the centre is ~1e-5, so with the raw
  values ('raw', kscr3) the U-Net read only the mask channel. 0 stays 0 and the peak maps to +/-1.
- `data_blind` is the twin: a fixed tensor replaces the four data channels and the mask channel is kept, so spacing
  between steps stays open to it. It reads no reconstruction, so the loop runs none before its steps.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from activemri.samplers.seq_adapt import FIXED_DATA_SEED, LOG_MAGNITUDE_FLOOR, StepContext

DATA_CHANNELS = 4  # predicted k-space (real, imaginary), measured k-space (real, imaginary)
INPUT_CHANNELS = DATA_CHANNELS + 1  # and the mask
VIRTUAL_COILS = 1
INPUT_SCALES = ('raw', 'signed_log')


def signed_log(values, floor=LOG_MAGNITUDE_FLOOR):
    """sign(v) log1p(|v| / floor) / log1p(1 / floor): 0 at 0, +/-1 at +/-1, odd and increasing."""
    return torch.sign(values) * torch.log1p(values.abs() / floor) / math.log1p(1 / floor)


class _ConvBlock(nn.Sequential):
    """SeqMRI's ConvBlock without its (zero) dropout: twice a 3x3 convolution, instance norm and ReLU."""

    def __init__(self, in_chans, out_chans):
        super().__init__(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1), nn.InstanceNorm2d(out_chans), nn.ReLU(),
                         nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1), nn.InstanceNorm2d(out_chans),
                         nn.ReLU())


def virtual_coil(coil_kspace, compression):
    """[B, H, W] complex: coil k-space [B, C, H, W] mixed by the first virtual coil's weights `compression` [B, C, V]."""
    return torch.einsum('bchw,bc->bhw', coil_kspace, compression[..., 0].conj())


class SeqMRISampler2D(nn.Module):
    """SeqMRI's Sampler2D: one step's per-location scores in [0, 1], 0 where the mask has sampled."""

    reads_reconstruction = False

    def __init__(self, shape, chans=64, num_pool_layers=4, checkpoint_network=True, input_scale='raw', data_blind=False):
        """`checkpoint_network` recomputes the U-Net in the backward pass instead of keeping every step's activations.
        `input_scale` and `data_blind` are the fscr1 options in the module docstring. The twin's fixed tensor comes from
        its own generator, so an arm and its twin draw the same U-Net from the global random stream."""
        super().__init__()
        if input_scale not in INPUT_SCALES:
            raise ValueError(f"input_scale must be one of {INPUT_SCALES}, got {input_scale!r}")
        self.checkpoint_network = checkpoint_network
        self.input_scale = input_scale
        self.data_blind = data_blind
        self.complex_coils = 0 if data_blind else VIRTUAL_COILS
        if data_blind:
            generator = torch.Generator().manual_seed(FIXED_DATA_SEED)
            self.register_buffer('fixed_data', torch.randn(1, DATA_CHANNELS, *shape, generator=generator))
        self.down_sample_layers = nn.ModuleList([_ConvBlock(INPUT_CHANNELS, chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(_ConvBlock(ch, ch * 2))
            ch *= 2
        self.conv = _ConvBlock(ch, ch)
        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_sample_layers.append(_ConvBlock(ch * 2, ch // 2))
            ch //= 2
        self.up_sample_layers.append(_ConvBlock(ch * 2, ch))
        self.conv2 = nn.Sequential(nn.Conv2d(ch, ch // 2, kernel_size=1), nn.Conv2d(ch // 2, 1, kernel_size=1),
                                   nn.Conv2d(1, 1, kernel_size=1))

    def forward(self, old_mask, context: StepContext, anatomy_labels=None, details=None):  # noqa: ARG002
        """Scores [B, 1, H, W]: softplus of the network's output over its per-slice maximum, times 1 - M_i."""
        inputs = self.inputs(old_mask, context)
        if self.checkpoint_network and torch.is_grad_enabled():
            raw = checkpoint(self.network, inputs, use_reentrant=False)
        else:
            raw = self.network(inputs)
        scores = F.softplus(raw)
        scores = scores / scores.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        return scores * (1 - old_mask)

    def network(self, inputs):
        stack, output = [], inputs
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)
        output = self.conv(output)
        for layer in self.up_sample_layers:
            skip = stack.pop()
            output = F.interpolate(output, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            output = layer(torch.cat([output, skip], dim=1))
        return self.conv2(output)

    def inputs(self, old_mask, context: StepContext):
        """[B, 5, H, W]: the predicted, then the measured virtual-coil k-space over the ACS k-space peak (the twin's fixed
        tensor instead), then M_i."""
        if self.data_blind:
            return torch.cat([self.fixed_data.expand(len(old_mask), -1, -1, -1).to(old_mask.dtype), old_mask], dim=1)
        if context.measured is None or context.compression is None:
            raise ValueError("seqmri_2d reads the measured coil k-space and its virtual-coil weights; the context has neither")
        if context.predicted is None:
            raise ValueError("seqmri_2d reads the latest in-loop reconstruction's k-space; the context carries none")
        peak = context.calibration_peaks[1].view(-1, 1, 1)
        measured = virtual_coil(context.measured, context.compression) / peak
        predicted = virtual_coil(context.predicted, context.compression) / peak
        channels = torch.stack([predicted.real, predicted.imag, measured.real, measured.imag], dim=1)
        if self.input_scale == 'signed_log':
            channels = signed_log(channels)
        return torch.cat([channels.to(old_mask.dtype), old_mask], dim=1)
