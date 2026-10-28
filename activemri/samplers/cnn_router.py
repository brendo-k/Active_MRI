"""ACS CNN -> mask-bank index, with hard routing and a straight-through training gradient."""

import torch
from torch import nn
from torch.nn import functional as F

from activemri.samplers.smooth_adapt import SmoothAdaptiveMask
from activemri.utils.mask_binarization import floored_grad_sigmoid


class ACSRouter(nn.Module):
    """Map a normalized ACS magnitude patch [B, 1, C, C] to mask indices [B].

    ``logits`` exposes class scores for training; ``forward`` returns the
    deterministic index. No anatomy labels or reconstruction features are used.
    """

    def __init__(self, bank_size, hidden=32):
        super().__init__()
        if bank_size < 1 or hidden < 1:
            raise ValueError("bank_size and hidden must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden, bank_size),
        )

    def logits(self, acs):
        return self.network(acs)

    def forward(self, acs):
        return self.logits(acs).argmax(dim=-1)


class CNNRoutedMaskBank(nn.Module):
    """Select one learned LOUPE probability map using only the central ACS.

    Input is the sampler's NHWC observation (real/imaginary in channels 0:2).
    The CNN sees mean-centered log magnitude of the ACS. Training uses hard
    Gumbel-softmax routing, so exactly one bank member is used in the forward
    pass, with a soft surrogate gradient to the CNN. Evaluation uses argmax.
    The existing sampler then rescales and binarizes the selected probability
    map. Mask indices do not denote anatomy classes.
    """

    calibration_features = SmoothAdaptiveMask.calibration_features

    def __init__(self, shape, slope, calibration_size, bank_size=8, hidden=32,
                 grad_floor=0.0, eps=1e-2):
        super().__init__()
        if not 1 <= calibration_size <= min(shape):
            raise ValueError("calibration_size must fit inside the k-space grid")
        self.shape = tuple(shape)
        self.calibration_size = calibration_size
        self.slope = slope
        self.grad_floor = grad_floor
        self.bank_size = bank_size
        self.router = ACSRouter(bank_size, hidden)
        uniform = torch.empty(bank_size, *shape).uniform_(eps, 1 - eps)
        self.mask = nn.Parameter(torch.logit(uniform) / slope)

    def acs_patch(self, kspace):
        return self.calibration_features(kspace).reshape(
            -1, 1, self.calibration_size, self.calibration_size)

    def indices(self, kspace):
        """Deterministic mask indices [B], independent of train/eval mode."""
        return self.router(self.acs_patch(kspace))

    def weights(self, kspace):
        logits = self.router.logits(self.acs_patch(kspace))
        if self.training:
            return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        return F.one_hot(logits.argmax(dim=-1), self.bank_size).to(logits.dtype)

    def forward(self, kspace, anatomy_labels=None):  # noqa: ARG002
        bank = floored_grad_sigmoid(self.mask, self.slope, self.grad_floor)
        return torch.einsum('nk,khw->nhw', self.weights(kspace), bank).unsqueeze(1)
