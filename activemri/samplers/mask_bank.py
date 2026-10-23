"""`bank_adapt_2d`: a bank of learned LOUPE masks, blended per slice by a gate that reads the calibration block.

The bank holds `bank_size` free logit maps, each started from LOUPE's uniform-noise initialisation so they differ.
A small MLP reads the log-magnitude of the centred `calibration_size` x `calibration_size` always-acquired block
(the same phase-invariant feature as `smooth_adapt_2d`) and returns softmax weights over the bank; the slice's
probability map is the weighted mean of the bank's sigmoids. The gate's output layer starts at zero, so every slice
begins with the plain average of the bank and learns to route from the calibration alone.
"""
import torch
from torch import nn

from activemri.samplers.smooth_adapt import SmoothAdaptiveMask
from activemri.utils.mask_binarization import floored_grad_sigmoid


class MaskBank(nn.Module):
    """Args:
        shape: k-space grid (H, W).
        slope: sigmoid slope, as in ProbMask.
        calibration_size: side of the centred calibration block the gate reads.
        bank_size: number of masks in the bank.
        hidden: width of the gate's two hidden layers.
        grad_floor: floor on the sigmoid's backward derivative, as in ProbMask.
        gate_temperature: the gate's logits are divided by this before the softmax (>1 flattens the weights).
        gate_logit_limit: if > 0 the gate's logits are bounded to +/- this by `limit * tanh(logit / limit)` before the
            temperature. Unbounded, the gate drives most of the bank's weights to ~e^-40 and those masks stop
            receiving gradient for good; the bound keeps every mask's weight above a floor
            (>= exp(-2 limit / temperature) relative to the largest).
    """

    calibration_features = SmoothAdaptiveMask.calibration_features

    def __init__(self, shape, slope, calibration_size, bank_size=8, hidden=64, grad_floor=0.0, eps=1e-2,
                 gate_temperature=1.0, gate_logit_limit=0.0):
        super().__init__()
        self.shape = tuple(shape)
        self.slope = slope
        self.calibration_size = calibration_size
        self.bank_size = bank_size
        self.grad_floor = grad_floor
        self.gate_temperature = gate_temperature
        self.gate_logit_limit = gate_logit_limit
        uniform = torch.zeros(bank_size, *self.shape).uniform_(eps, 1 - eps)
        self.mask = nn.Parameter(-torch.log(1.0 / uniform - 1.0) / slope)
        width = calibration_size ** 2
        self.gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.gate_head = nn.Linear(hidden, bank_size)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def weights(self, kspace):
        """[N, bank_size] softmax weights from the calibration block of NHWC k-space."""
        logits = self.gate_head(self.gate(self.calibration_features(kspace)))
        if self.gate_logit_limit > 0:
            logits = self.gate_logit_limit * torch.tanh(logits / self.gate_logit_limit)
        return torch.softmax(logits / self.gate_temperature, dim=-1)

    def forward(self, kspace, anatomy_labels=None):  # noqa: ARG002
        """Args: kspace NHWC. Returns: probabilities [N, 1, H, W]."""
        bank = floored_grad_sigmoid(self.mask, self.slope, self.grad_floor)  # [K, H, W]
        return torch.einsum('nk,khw->nhw', self.weights(kspace), bank).unsqueeze(1)
