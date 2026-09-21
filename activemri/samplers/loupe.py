"""LOUPE samplers: a learned mask that is the same for every input.

The sampling pattern is a free parameter of the model rather than a function of
the observed k-space, so these ignore their input and simply expose their
weights through a sigmoid. From https://arxiv.org/abs/1907.11374.

With anatomy conditioning the free parameter gains a leading class dimension:
each anatomy owns a mask, selected per sample by its label.

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
        num_conditioning_classes: If > 0, learn one mask per anatomy instead of
            one shared mask.
    """

    def __init__(self, shape=(320, 320), slope=10, num_conditioning_classes=0):
        super().__init__()

        self.slope = slope
        self.num_conditioning_classes = num_conditioning_classes
        num_masks = max(num_conditioning_classes, 1)
        self.mask = nn.Parameter(torch.cat([self._init_mask_weights(shape) for _ in range(num_masks)]))

    def forward(self, input, anatomy_labels=None):
        """Args: input NHWC. Returns: probabilities [N or 1, H, W, 1]."""
        logits = _select_mask(self.mask, anatomy_labels, self.num_conditioning_classes)
        logits = logits.view(-1, input.shape[1], input.shape[2], 1)
        return torch.sigmoid(self.slope * logits)

    def raw_logits(self, anatomy_labels=None):
        """The free parameter itself (pre-sigmoid), [N or 1, H, W] -- the score
        field a REINFORCE policy samples from, as opposed to :meth:`forward`'s
        sigmoid probabilities used by the straight-through path."""
        logits = _select_mask(self.mask, anatomy_labels, self.num_conditioning_classes)
        return logits.squeeze(-1)

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
        num_conditioning_classes: If > 0, learn one line pattern per anatomy
            instead of one shared pattern.
    """

    def __init__(self, shape=(320,), slope=10, num_conditioning_classes=0):
        super().__init__()

        self.slope = slope
        self.num_conditioning_classes = num_conditioning_classes
        weights = [self._init_mask_weights(shape[0]) for _ in range(max(num_conditioning_classes, 1))]
        # Unconditioned, the parameter keeps its original 1-D shape so existing
        # checkpoints still load.
        self.mask = nn.Parameter(torch.stack(weights) if num_conditioning_classes > 0 else weights[0])

    def forward(self, input, anatomy_labels=None):  # noqa: ARG002
        """Args: input NHWC. Returns: probabilities [N or 1, 1, W, 1], broadcast over rows."""
        logits = _select_mask(self.mask, anatomy_labels, self.num_conditioning_classes)
        return torch.sigmoid(self.slope * logits).view(-1, 1, self.mask.shape[-1], 1)

    def raw_logits(self, anatomy_labels=None):
        """The free parameter itself (pre-sigmoid), [N or 1, W] -- the score
        field a REINFORCE policy samples columns from."""
        logits = _select_mask(self.mask, anatomy_labels, self.num_conditioning_classes)
        return logits.view(-1, self.mask.shape[-1])

    def _init_mask_weights(self, length, eps=1e-2):
        """Logits whose sigmoid is uniform on (eps, 1-eps)."""
        temp = torch.zeros(length).uniform_(eps, 1 - eps)
        return -torch.log(1. / temp - 1.) / self.slope


def _select_mask(mask, anatomy_labels, num_conditioning_classes):
    """Pick the per-anatomy mask for each sample, or the single shared one.

    Falls back to the mean over anatomies when conditioning is on but no labels
    are supplied, so the model still runs on data of unknown anatomy.
    """
    if num_conditioning_classes <= 0:
        return mask
    if anatomy_labels is None:
        return mask.mean(dim=0, keepdim=True)
    return mask.index_select(0, anatomy_labels.to(device=mask.device, dtype=torch.long))
