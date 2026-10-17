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
    def forward(ctx, input, grad_floor=0.0, threshold=None):
        if threshold is None:
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

        return current_grad * grad_output, None, None


def threshold_random_mask(probabilities, grad_floor=0.0, threshold=None):
    """Bernoulli(p) mask with a straight-through gradient; `threshold` (same shape, U(0, 1)) replaces the fresh draw."""
    return ThresholdRandomMaskSigmoidV1.apply(probabilities, grad_floor, threshold)


def stratified_thresholds(like, draws):
    """Thresholds for `draws` masks of the same slices, laid out draw-major: [draws * B, ...] from [draws * B, ...] `like`.

    One U(0, 1) noise map per slice is rotated by k / draws (mod 1) for the k-th draw. Every draw is
    still marginally uniform, so each mask is an exact Bernoulli(p) sample and its gradient is unbiased;
    the draws are negatively correlated (for draws=2 the antithetic pair u, u + 1/2), so their average
    has lower variance than independent draws, and a location of probability p is covered by
    ~p * draws of them instead of a binomial scatter.
    """
    total = like.shape[0]
    if total % draws:
        raise ValueError(f"a batch of {total} is not a whole number of {draws} draws per slice")
    base = torch.rand_like(like[:total // draws])
    return torch.cat([(base + k / draws) % 1.0 for k in range(draws)], dim=0)


def topk_straight_through(probabilities, k, available=None):
    """Exact-budget binarization: the k highest probabilities per example become 1.

    Backward is the identity on `probabilities`, the usual straight-through
    surrogate for a hard threshold. `available` (1 = may be selected) confines
    the choice; if fewer than k are available only those are selected.
    """
    scores = probabilities.detach().flatten(1)
    if available is not None:
        scores = scores.masked_fill(available.expand_as(probabilities).flatten(1) <= 0, float('-inf'))
    chosen = torch.zeros_like(scores).scatter_(1, scores.topk(k, dim=1).indices, 1.0).view_as(probabilities)
    if available is not None:
        chosen = chosen * available.to(chosen.dtype)
    return chosen + (probabilities - probabilities.detach())
