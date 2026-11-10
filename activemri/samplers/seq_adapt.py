"""seq_adapt_2d: a static map per condition plus a bounded residual read from the k-space measured so far.

At step i of a K-step acquisition the logits are slope * L_c + r(z_i):

    L_c   one free logit map per condition, as in loupe_2d (stored as logits / slope, so loupe_2d checkpoints load)
    r     a small U-Net whose final convolution starts at zero, bounded to +/- limit by limit * tanh(. / limit)
    z_i   data      the RSS image of the measured coil k-space y * M_i over its ACS peak, and the coil-RSS magnitude
                    of y * M_i over its ACS peak, log-scaled to [0, 1] (0 where nothing is measured)
          mask      M_i, every location acquired before step i
          header    the acquired extent, i / K, the active coil count over the coil dimension, a one-hot condition

`data_blind` is the twin: its data channels are a fixed tensor and it is never handed the measured k-space, so it
keeps M_i, the step index and the header. Spacing (filling the gaps between earlier draws) and header routing are then
open to both arms and are not credited to adaptivity. The U-Net and the fixed tensor are built off the global random
stream, so an adaptive arm and its twin start bitwise identical, and with r = 0 at the start they draw the same masks.

Two ablations keep that start. `data_channels` names the data channels an adaptive arm reads ('image', 'kspace'); any
other channel gets the twin's fixed values. `reads_reconstruction` adds a data channel, a frozen reconstructor's
magnitude image of y * M_i over the ACS image peak (SeqMRI-style); the residual's first convolution gains that input,
and every other weight is as without it.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from activemri.reconstructors.UNet import LOUPEUNet
from activemri.samplers.loupe import _select_mask
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.mask_binarization import floored_grad_sigmoid

DATA_CHANNEL_NAMES = ('image', 'kspace')
DATA_CHANNELS = len(DATA_CHANNEL_NAMES)
MASK_AND_HEADER_CHANNELS = 4  # M_i, acquired extent, i / K, active coil fraction
LOG_MAGNITUDE_FLOOR = 1e-6
FIXED_DATA_SEED = 0
RECONSTRUCTION_CHANNEL_SEED = 1
PEAK_FLOOR = 1e-12


@dataclass
class StepContext:
    """What one step's sampler may read besides its mask.

    `measured` is the complex coil k-space y * M_i [B, C, H, W] and `calibration_peaks` the (image, k-space) ACS peaks
    [B] that scale it; both are None for a data-blind sampler. `reconstruction` [B, H, W] is the frozen reconstructor's
    image of y * M_i over the ACS image peak, for a sampler that reads one.
    """
    step: int
    num_steps: int
    extent: torch.Tensor
    coil_fraction: torch.Tensor
    measured: Optional[torch.Tensor] = None
    calibration_peaks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    reconstruction: Optional[torch.Tensor] = None


def _coil_rss(coil_values):
    return coil_values.abs().pow(2).sum(dim=1).sqrt()


def calibration_peaks(calibration_kspace):
    """Peaks [B] of the ACS's RSS image and of its coil-RSS k-space magnitude, from the coil k-space y * M_0."""
    image_peak = _coil_rss(ifft_2d_img(calibration_kspace)).flatten(1).amax(dim=1)
    kspace_peak = _coil_rss(calibration_kspace).flatten(1).amax(dim=1)
    return image_peak.clamp(min=PEAK_FLOOR), kspace_peak.clamp(min=PEAK_FLOOR)


def measured_channels(measured, peaks):
    """The data channels [B, 2, H, W] of measured coil k-space [B, C, H, W]: its RSS image and its log-scaled coil-RSS
    magnitude, each over its ACS peak."""
    image_peak, kspace_peak = peaks
    image = _coil_rss(ifft_2d_img(measured)) / image_peak.view(-1, 1, 1)
    magnitude = _coil_rss(measured) / kspace_peak.view(-1, 1, 1)
    log_floor = math.log(LOG_MAGNITUDE_FLOOR)
    log_magnitude = (torch.log(magnitude + LOG_MAGNITUDE_FLOOR) - log_floor) / -log_floor
    return torch.stack([image, log_magnitude], dim=1).float()


def _widen_first_conv(unet, extra_channels):
    """Give the U-Net's first convolution `extra_channels` more inputs, keeping its weights for the existing ones. The new
    weights come from their own generator, at the bound PyTorch's default init uses for the existing inputs."""
    name, conv = next((name, m) for name, m in unet.down_sample_layers[0].named_modules() if isinstance(m, nn.Conv2d))
    parent_name, _, attribute = name.rpartition('.')
    parent = unet.down_sample_layers[0].get_submodule(parent_name)
    wider = nn.Conv2d(conv.in_channels + extra_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      bias=conv.bias is not None)
    bound = 1 / math.sqrt(conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1])
    generator = torch.Generator().manual_seed(RECONSTRUCTION_CHANNEL_SEED)
    extra = (2 * torch.rand(conv.out_channels, extra_channels, *conv.kernel_size, generator=generator) - 1) * bound
    with torch.no_grad():
        wider.weight.copy_(torch.cat([conv.weight, extra], dim=1))
        if conv.bias is not None:
            wider.bias.copy_(conv.bias)
    setattr(parent, attribute, wider)
    unet.in_chans += extra_channels


def _data_channels(names, data_blind):
    """The data channels an adaptive arm reads, in DATA_CHANNEL_NAMES order; None means all of them."""
    if names is None:
        return DATA_CHANNEL_NAMES
    unknown = set(names) - set(DATA_CHANNEL_NAMES)
    if unknown or not names:
        raise ValueError(f"data_channels must be a non-empty subset of {DATA_CHANNEL_NAMES}, got {list(names)}")
    if data_blind and set(names) != set(DATA_CHANNEL_NAMES):
        raise ValueError("a data-blind twin reads no data channel; leave data_channels at its default")
    return tuple(name for name in DATA_CHANNEL_NAMES if name in names)


class SequentialAdaptiveMask(nn.Module):
    """Per-location sampling probabilities sigmoid(slope * L_c + r(z_i)) for one step of a K-step acquisition."""

    def __init__(self, shape, slope, num_conditioning_classes=0, chans=16, depth=4, limit=6.0, data_blind=False,
                 grad_floor=0.0, eps=1e-2, checkpoint_residual=True, data_channels=None, reads_reconstruction=False):
        """`checkpoint_residual` recomputes the U-Net in the backward pass instead of keeping every step's activations:
        at K=16 and batch 16 those alone fill a 48 GB GPU."""
        super().__init__()
        self.shape = tuple(shape)
        self.checkpoint_residual = checkpoint_residual
        self.slope = slope
        self.limit = limit
        self.grad_floor = grad_floor
        self.data_blind = data_blind
        self.data_channels = _data_channels(data_channels, data_blind)
        if data_blind and reads_reconstruction:
            raise ValueError("a data-blind twin cannot read a reconstruction of the data")
        self.reads_reconstruction = reads_reconstruction
        self.num_conditioning_classes = num_conditioning_classes
        uniform = [torch.zeros(*self.shape).uniform_(eps, 1 - eps) for _ in range(max(num_conditioning_classes, 1))]
        self.mask = nn.Parameter(torch.stack([-torch.log(1. / u - 1.) / slope for u in uniform]).unsqueeze(-1))
        with torch.random.fork_rng(devices=[]):
            self.residual = LOUPEUNet(DATA_CHANNELS + MASK_AND_HEADER_CHANNELS + num_conditioning_classes, 1, chans,
                                      depth, drop_prob=0, upsample_mode='bilinear', input_skip=False)
            if reads_reconstruction:
                _widen_first_conv(self.residual, 1)
        nn.init.zeros_(self.residual.final_conv.weight)
        nn.init.zeros_(self.residual.final_conv.bias)
        generator = torch.Generator().manual_seed(FIXED_DATA_SEED)
        self.register_buffer('fixed_data', torch.randn(1, DATA_CHANNELS, *self.shape, generator=generator),
                             persistent=False)
        reads = torch.tensor([name in self.data_channels for name in DATA_CHANNEL_NAMES]).view(1, -1, 1, 1)
        self.register_buffer('reads_channel', reads, persistent=False)

    def forward(self, old_mask, context: StepContext, anatomy_labels=None, details=None):
        """Probabilities [B, 1, H, W], before the budget rescale; a `details` dict receives the residual."""
        static = _select_mask(self.mask, anatomy_labels, self.num_conditioning_classes).permute(0, 3, 1, 2)
        residual = self.residual_logits(old_mask, context, anatomy_labels)
        if details is not None:
            details['residual'] = residual
        return floored_grad_sigmoid(static + residual / self.slope, self.slope, self.grad_floor)

    def residual_logits(self, old_mask, context: StepContext, anatomy_labels=None):
        """The bounded residual r(z_i) [B, 1, H, W], in logit units."""
        inputs = self.inputs(old_mask, context, anatomy_labels)
        if self.checkpoint_residual and torch.is_grad_enabled():
            raw = checkpoint(self.residual, inputs, use_reentrant=False)
        else:
            raw = self.residual(inputs)
        return self.limit * torch.tanh(raw / self.limit)

    def inputs(self, old_mask, context: StepContext, anatomy_labels=None):
        """z_i [B, channels, H, W]."""
        batch = old_mask.shape[0]
        if self.data_blind:
            if context.measured is not None:
                raise ValueError("a data-blind sampler must not be handed the measured k-space")
            data = self.fixed_data.expand(batch, -1, -1, -1)
        else:
            data = measured_channels(context.measured, context.calibration_peaks)
            if len(self.data_channels) < DATA_CHANNELS:
                data = torch.where(self.reads_channel, data, self.fixed_data)
            if self.reads_reconstruction:
                if context.reconstruction is None:
                    raise ValueError("this sampler reads a reconstruction of y * M_i, and the context carries none")
                data = torch.cat([data, context.reconstruction.unsqueeze(1).to(data.dtype)], dim=1)
        step_fraction = old_mask.new_full((batch, 1, *self.shape), context.step / context.num_steps)
        coil_fraction = context.coil_fraction.to(old_mask.dtype).view(batch, 1, 1, 1).expand(-1, -1, *self.shape)
        channels = [data.to(old_mask.dtype), old_mask, context.extent.to(old_mask.dtype), step_fraction, coil_fraction]
        if self.num_conditioning_classes > 0:
            channels.append(self._condition_channels(anatomy_labels, old_mask))
        return torch.cat(channels, dim=1)

    def _condition_channels(self, anatomy_labels, like):
        batch = like.shape[0]
        if anatomy_labels is None:
            return like.new_zeros(batch, self.num_conditioning_classes, *self.shape)
        one_hot = F.one_hot(anatomy_labels.to(device=like.device, dtype=torch.long), self.num_conditioning_classes)
        return one_hot.to(like.dtype).view(batch, -1, 1, 1).expand(-1, -1, *self.shape)
