"""Full-resolution, same-formula sensitivity normalization candidates for profiling."""
import torch
from torch import nn


class SpatialMomentsNorm(nn.Module):
    """Per-sample, per-channel normalization with FP32 statistics and biased variance.

    Same formula/epsilon as stateless InstanceNorm2d. Floating-point reductions
    and backward rounding differ; this is not a claim of bitwise equivalence.
    The pointwise output can retain NHWC strides instead of forcing NCHW.
    """
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        v, m = torch.var_mean(x.float(), dim=(-2, -1), correction=0, keepdim=True)
        return ((x.float() - m) * torch.rsqrt(v + self.eps)).to(x.dtype)


def replace_sensitivity_norms(module, kind):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.InstanceNorm2d):
            if child.affine or child.track_running_stats:
                raise ValueError('Candidate requires stateless, non-affine InstanceNorm2d')
            replacement = (nn.GroupNorm(child.num_features, child.num_features, eps=child.eps, affine=False)
                           if kind == 'group' else SpatialMomentsNorm(child.eps))
            setattr(module, name, replacement)
        else:
            replace_sensitivity_norms(child, kind)


class ChannelsLastBody(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        # A layout conversion creates a tensor and does not propagate Dynamo's
        # dynamic-dimension annotation from the sensitivity model's input.
        torch._dynamo.maybe_mark_dynamic(x, 0)
        return self.inner(x)
