"""Shared machinery for turning an unnormalized sampling score into a binary mask.

Both the LOUPE and conv sampler families produce an unnormalized probability map;
these two steps are what make it a usable, differentiable acquisition mask.
"""
import torch
from torch.autograd import Function


def RescaleProbMap(batch_x, sparsity, available=None):
    """Rescale a probability map so its mean equals the target sparsity.

    If mean(x) > sparsity the rescaling is direct: x' = x * sparsity / mean(x).
    Otherwise the same is applied to (1-x) and the result inverted, which keeps
    the output inside [0, 1] in both regimes.

    `available` (1 = may still be sampled) confines the rescaling to those
    locations: the rest stay at 0, instead of the inverted branch lifting them
    to 1 - beta, and the target mean over the available ones rises so the
    expected count is still sparsity * numel.
    """
    if available is None:
        available = torch.ones_like(batch_x)
    available = available.to(batch_x.dtype).expand_as(batch_x)

    ret = []
    for i in range(len(batch_x)):
        x = batch_x[i:i + 1]
        a = available[i:i + 1]
        num_available = a.sum()
        xbar = (x * a).sum() / num_available
        target = sparsity * x.numel() / num_available
        r = target / xbar
        beta = (1 - target) / (1 - xbar)

        le = torch.le(r, 1).float()
        ret.append(a * (le * x * r + (1 - le) * (1 - (1 - x) * beta)))

    return torch.cat(ret, dim=0)


STE_SLOPE = 5
SIGMOID_PEAK_DERIVATIVE = 0.25


def floored_sigmoid_derivative(sigmoid_output, slope, grad_floor):
    """d/dx sigmoid(slope * x), floored at `grad_floor` times its peak (slope / 4).

    The floor keeps saturated sigmoids from silencing the gradient entirely;
    grad_floor=0 is the exact derivative.
    """
    local = torch.clamp(sigmoid_output * (1 - sigmoid_output), min=grad_floor * SIGMOID_PEAK_DERIVATIVE)
    return slope * local


class FlooredGradSigmoid(Function):
    """sigmoid(slope * x) in the forward pass; backward uses `floored_sigmoid_derivative`."""

    @staticmethod
    def forward(ctx, x, slope, grad_floor):
        output = torch.sigmoid(slope * x)
        ctx.save_for_backward(output)
        ctx.slope = slope
        ctx.grad_floor = grad_floor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * floored_sigmoid_derivative(output, ctx.slope, ctx.grad_floor), None, None


def floored_grad_sigmoid(x, slope, grad_floor=0.0):
    if grad_floor <= 0:
        return torch.sigmoid(slope * x)
    return FlooredGradSigmoid.apply(x, slope, grad_floor)


class ThresholdRandomMaskSigmoidV1(Function):
    """Straight-through estimator for stochastic binarization.

    Forward thresholds each probability p against a single u ~ U(0, 1), so
    every location is an independent Bernoulli(p) draw and the sampled count
    matches the budget only in expectation. That is the same draw as
    logit(p) + Logistic(0, 1) > 0, i.e. hard Gumbel-sigmoid noise. Backward,
    though, is not the binary-Concrete relaxation (a tempered sigmoid in logit
    space): it substitutes the derivative of sigmoid(STE_SLOPE * (p - u)) in
    probability space for the non-differentiable `>`, floored at `grad_floor`
    of its peak.
    """

    @staticmethod
    def forward(ctx, input, grad_floor=0.0):
        threshold = torch.rand_like(input)
        ctx.save_for_backward(input, threshold)
        ctx.grad_floor = grad_floor
        return (input > threshold).to(input.dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        input, threshold = ctx.saved_tensors

        surrogate = torch.sigmoid(STE_SLOPE * (input - threshold))
        current_grad = floored_sigmoid_derivative(surrogate, STE_SLOPE, ctx.grad_floor)

        return current_grad * grad_output, None


def threshold_random_mask(probabilities, grad_floor=0.0):
    return ThresholdRandomMaskSigmoidV1.apply(probabilities, grad_floor)
