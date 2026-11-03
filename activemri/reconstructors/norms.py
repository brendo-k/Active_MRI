"""Swappable normalisation layers for the U-Nets (VarNet cascades and the sensitivity net), which are built with
InstanceNorm2d(affine=False) after every convolution.

    instance        InstanceNorm2d as built: per sample and channel, over H x W, no affine
    layer           GroupNorm(1, C): per sample over C x H x W, with a per-channel affine
    channel_layer   LayerNorm over the channels of each pixel (ConvNeXt / NAFNet / Restormer style), per-channel affine
    batch           BatchNorm2d: batch statistics in training, running statistics in eval
    none            no normalisation

None of them draws random numbers when built, so swapping one in leaves every other weight's initial value as it was.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_LAYERS = ('instance', 'layer', 'channel_layer', 'batch', 'none')


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an [N, C, H, W] tensor, at every pixel."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), self.weight.shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == 'instance':
        return nn.InstanceNorm2d(channels, affine=False)
    if kind == 'layer':
        return nn.GroupNorm(1, channels)
    if kind == 'channel_layer':
        return ChannelLayerNorm(channels)
    if kind == 'batch':
        return nn.BatchNorm2d(channels)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f"norm must be one of {NORM_LAYERS}, got {kind!r}")


def replace_instance_norms(module: nn.Module, kind: str) -> nn.Module:
    """Replace every InstanceNorm2d inside `module` by `kind` (in place); 'instance' leaves the module as built."""
    if kind not in NORM_LAYERS:
        raise ValueError(f"norm must be one of {NORM_LAYERS}, got {kind!r}")
    if kind == 'instance':
        return module
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm2d):
            setattr(module, name, make_norm(kind, child.num_features))
        else:
            replace_instance_norms(child, kind)
    return module
