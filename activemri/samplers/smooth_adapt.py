"""`smooth_adapt_2d`: a LOUPE mask plus a smooth offset predicted from the calibration block.

The base is a free logit per k-space location (one map, or one per anatomy class), bounded to +/- `logit_limit`
through a tanh so every location stays within reach of the offset. The offset is a coarse `grid` x `grid` map of
logits predicted from the log-magnitude of the acquired calibration k-space (phase-invariant, normalised per slice),
bounded the same way and upsampled bicubically to the full grid, so a slice can reshape its density only smoothly.
Unbounded, the offset grows without limit under Adam (each step about lr whatever the gradient's size) and the masks
saturate within an epoch. The predictor's output
layer has no bias and starts at zero: at initialisation the sampler is exactly its base mask, and any offset it learns
depends on the input. `grid=0` drops the predictor, leaving a bounded LOUPE mask (the static control).
"""
import torch
import torch.nn.functional as F
from torch import nn

from activemri.utils.mask_binarization import floored_grad_sigmoid

LOG_EPS = 1e-8


class SmoothAdaptiveMask(nn.Module):
    """Args:
        shape: k-space grid (H, W).
        slope: base logit = logit_limit * tanh(slope * w / logit_limit), w the free parameter (LOUPE's convention).
        calibration_size: side of the centred, always-acquired calibration block the predictor reads.
        grid: side of the coarse offset grid; 0 disables the offset.
        hidden: width of the predictor's two hidden layers.
        num_conditioning_classes: if > 0, one base map per class, selected by the anatomy label.
        grad_floor: floor on the sigmoid's backward derivative, as in ProbMask.
        logit_limit: bound on the base logits.
    """

    def __init__(self, shape, slope, calibration_size, grid=16, hidden=64, num_conditioning_classes=0,
                 grad_floor=0.0, logit_limit=6.0):
        super().__init__()
        self.shape = tuple(shape)
        self.slope = slope
        self.calibration_size = calibration_size
        self.grid = grid
        self.grad_floor = grad_floor
        self.logit_limit = logit_limit
        self.num_conditioning_classes = num_conditioning_classes
        self.mask = nn.Parameter(torch.zeros(max(num_conditioning_classes, 1), *self.shape))
        if grid > 0:
            width = calibration_size ** 2
            self.predictor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden), nn.ReLU())
            self.offset_head = nn.Linear(hidden, grid * grid, bias=False)
            nn.init.zeros_(self.offset_head.weight)

    def base_logits(self, anatomy_labels=None):
        """[N or 1, H, W] bounded base logits."""
        bounded = self.logit_limit * torch.tanh(self.slope * self.mask / self.logit_limit)
        if self.num_conditioning_classes > 0:
            if anatomy_labels is None:
                raise ValueError("a conditioned smooth_adapt_2d mask needs anatomy labels")
            return bounded[anatomy_labels]
        return bounded

    @torch.no_grad()
    def set_base_logits(self, logits):
        """Make every class's base equal `logits` [H, W] (clipped just inside the bound)."""
        limit = self.logit_limit
        clipped = logits.clamp(-0.999 * limit, 0.999 * limit)
        self.mask.copy_((limit / self.slope * torch.atanh(clipped / limit)).expand_as(self.mask))

    def calibration_features(self, kspace):
        """Log-magnitude of the centred calibration block of NHWC k-space, minus its per-slice mean: [N, c * c]."""
        height, width = self.shape
        size = self.calibration_size
        top, left = height // 2 - size // 2, width // 2 - size // 2
        block = kspace[:, top:top + size, left:left + size, :2]
        log_magnitude = torch.log(torch.linalg.vector_norm(block, dim=-1).flatten(1) + LOG_EPS)
        return log_magnitude - log_magnitude.mean(dim=1, keepdim=True)

    def offset_logits(self, kspace):
        """[N, H, W] smooth offset (zeros when grid == 0)."""
        if self.grid == 0:
            return kspace.new_zeros(kspace.shape[0], *self.shape)
        raw = self.offset_head(self.predictor(self.calibration_features(kspace)))
        coarse = (self.logit_limit * torch.tanh(raw / self.logit_limit)).view(-1, 1, self.grid, self.grid)
        return F.interpolate(coarse, size=self.shape, mode='bicubic', align_corners=False)[:, 0]

    def forward(self, kspace, anatomy_labels=None):
        """Args: kspace NHWC (the sampler's observation; real/imag in the first two channels).
        Returns: probabilities [N, 1, H, W]."""
        logits = self.base_logits(anatomy_labels) + self.offset_logits(kspace)
        return floored_grad_sigmoid(logits, 1.0, self.grad_floor).unsqueeze(1)
