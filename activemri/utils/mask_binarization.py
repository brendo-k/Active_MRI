"""Shared machinery for turning an unnormalized sampling score into a binary mask.

Both the LOUPE and conv sampler families produce an unnormalized probability map;
these two steps are what make it a usable, differentiable acquisition mask.
"""
import torch
from torch.autograd import Function


def RescaleProbMap(batch_x, sparsity):
    """Rescale a probability map so its mean equals the target sparsity.

    If mean(x) > sparsity the rescaling is direct: x' = x * sparsity / mean(x).
    Otherwise the same is applied to (1-x) and the result inverted, which keeps
    the output inside [0, 1] in both regimes.
    """
    ret = []
    for i in range(len(batch_x)):
        x = batch_x[i:i + 1]
        xbar = torch.mean(x)
        r = sparsity / xbar
        beta = (1 - sparsity) / (1 - xbar)

        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1 - le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)


class ThresholdRandomMaskSigmoidV1(Function):
    """Straight-through estimator for stochastic binarization.

    Forward draws a uniform sample and thresholds against it, retrying until the
    realized density matches the requested one. Backward substitutes the
    derivative of a sigmoid (slope=5) for the non-differentiable `>`, trading
    some gradient bias for lower variance versus a steeper slope.
    """

    @staticmethod
    def forward(ctx, input):
        probs = []
        results = []

        for i in range(len(input)):
            x = input[i:i + 1]

            count = 0
            while True:
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()

                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):
                    break

                count += 1
                if count > 1000:
                    raise RuntimeError(
                        f"Binarization failed to match target density after 1000 draws: "
                        f"target={torch.mean(x).item():.6f} realized={torch.mean(result).item():.6f}"
                    )

            probs.append(prob)
            results.append(result)

        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)
        ctx.save_for_backward(input, probs)

        return results

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]

        slope = 5
        input, prob = ctx.saved_tensors

        current_grad = (
            slope
            * torch.exp(-slope * (input - prob))
            / (torch.exp(-slope * (input - prob)) + 1).pow(2)
        )

        return current_grad * grad_output
