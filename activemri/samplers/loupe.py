"""LOUPE samplers: a learned mask that is the same for every input.

The sampling pattern is a free parameter of the model rather than a function of
the observed k-space, so these ignore their input and simply expose their
weights through a sigmoid. From https://arxiv.org/abs/1907.11374.

    ProbMask                  loupe_2d - a probability per k-space location
    LineConstrainedProbMask   loupe_1d - a probability per k-space column

Both return NHWC, matching the LOUPE convention; `Sampler` permutes to NCHW.
"""
import torch
from torch import nn


class ProbMask(nn.Module):
    """A learned probability for every k-space location.

    Args:
        shape: Shape of the reconstructed image.
        slope: Sigmoid slope. Larger slopes drive the mask to a deterministic
            state faster.
    """

    def __init__(self, shape=(320, 320), slope=5):
        super().__init__()

        self.slope = slope
        self.mask = nn.Parameter(self._init_mask_weights(shape))

    def forward(self, input):
        """Args: input NHWC. Returns: probabilities [1, H, W, 1]."""
        logits = self.mask.view(1, input.shape[1], input.shape[2], 1)
        return torch.sigmoid(self.slope * logits)

    def _init_mask_weights(self, shape, eps=1e-2):
        """Logits whose sigmoid is uniform on (eps, 1-eps)."""
        temp = torch.zeros(*shape).uniform_(eps, 1 - eps)
        logits = -torch.log(1. / temp - 1.) / self.slope
        return logits.reshape(1, shape[0], shape[1], 1)


class LineConstrainedProbMask(nn.Module):
    """A learned probability per k-space column, constrained to whole readout lines.

    Args:
        shape: Shape of the reconstructed image; only shape[0] is used.
        slope: Sigmoid slope, as in ProbMask.
    """

    def __init__(self, shape=(320,), slope=5):
        super().__init__()

        self.slope = slope
        self.mask = nn.Parameter(self._init_mask_weights(shape[0]))

    def forward(self, input):  # noqa: ARG002
        """Args: input NHWC. Returns: probabilities [1, 1, W, 1], broadcast over rows."""
        return torch.sigmoid(self.slope * self.mask).view(1, 1, self.mask.shape[0], 1)

    def _init_mask_weights(self, length, eps=1e-2):
        """Logits whose sigmoid is uniform on (eps, 1-eps)."""
        temp = torch.zeros(length).uniform_(eps, 1 - eps)
        return -torch.log(1. / temp - 1.) / self.slope
