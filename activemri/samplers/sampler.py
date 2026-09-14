"""Selects and runs one of the four sampling families.

`Sampler` owns the choice of family and the shared post-processing that turns a
raw score map into a binary acquisition mask.
"""
import torch
from torch import nn

from activemri.samplers.conv import KspaceLineConstrainedSampler, SampProb2D
from activemri.samplers.loupe import LineConstrainedProbMask, ProbMask
from activemri.utils.mask_binarization import RescaleProbMap, ThresholdRandomMaskSigmoidV1


SAMPLER_TYPES = ("conv_2d", "conv_1d", "loupe_2d", "loupe_1d")


def is_line_constrained(sampler_type):
    """Whether a sampler selects whole k-space columns rather than single locations."""
    return sampler_type.endswith("_1d")


class Sampler(nn.Module):
    """Generates sampling masks for sequential k-space acquisition.

    Four sampler families are available, selected by `sampler_type`:

        conv_2d   U-Net over k-space, one score per location   (input-adaptive)
        conv_1d   MLP over k-space, one score per column       (input-adaptive)
        loupe_2d  learned mask, one weight per location        (input-independent)
        loupe_1d  learned mask, one weight per column          (input-independent)

    The LOUPE families ignore the observed k-space; their mask is a free
    parameter shared across the dataset. All four are put through the same
    rescale-to-sparsity and straight-through binarization steps.
    """

    def __init__(
        self,
        sampler_type="conv_2d",
        shape=(320, 320),
        slope=5,
        with_uncertainty=False,
        detach_kspace=False,
        fixed_input=False,
        with_recon=False,
        with_gfactor=False,
        uncertainty_mode='none',
        num_anatomy_classes=0,
        ):
        super().__init__()

        if sampler_type not in SAMPLER_TYPES:
            raise ValueError(f"Unknown sampler_type {sampler_type!r}; expected one of {SAMPLER_TYPES}")

        self.sampler_type = sampler_type
        self.line_constrained = is_line_constrained(sampler_type)
        self.is_loupe = sampler_type.startswith("loupe")
        self.shape = shape
        self.detach_kspace = detach_kspace
        self.with_uncertainty = with_uncertainty
        self.with_recon = with_recon
        self.uncertainty_mode = uncertainty_mode
        self.with_gfactor = with_gfactor
        self.num_anatomy_classes = num_anatomy_classes

        if sampler_type == "conv_2d":
            self.mask_net = SampProb2D(
                fixed_input=fixed_input,
                with_recon=with_recon,
                with_gfactor=with_gfactor,
                uncertainty_mode=uncertainty_mode,
                num_anatomy_classes=num_anatomy_classes,
            )
        elif sampler_type == "conv_1d":
            if with_uncertainty:
                raise NotImplementedError("conv_1d does not support with_uncertainty")
            self.mask_net = KspaceLineConstrainedSampler(
                shape[0], fixed_input=fixed_input,
                num_anatomy_classes=num_anatomy_classes,
            )
        elif sampler_type == "loupe_2d":
            self.mask_net = ProbMask(shape, slope)
        else:
            self.mask_net = LineConstrainedProbMask(shape, slope)

        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply

    def _predict_probabilities(self, input_kspace, old_mask, recon_complex, gfactor):
        """Run the mask network, returning (scores NCHW, uncertainty or None, anatomy_logits or None)."""
        if self.is_loupe:
            prob = self.mask_net(input_kspace).permute(0, 3, 1, 2)
            prob = prob.expand(input_kspace.shape[0], -1, -1, -1)
            already = (old_mask.sum(dim=-2, keepdim=True) == self.shape[0]).to(prob.dtype) \
                if self.line_constrained else old_mask
            return prob * (1 - already), None, None

        if self.line_constrained:
            prob, anatomy_logits = self.mask_net(input_kspace, old_mask)
            return prob, None, anatomy_logits

        prob, uncertainty, anatomy_logits = self.mask_net(
            input_kspace, old_mask, recon=recon_complex, gfactor=gfactor,
        )
        return prob, uncertainty, anatomy_logits

    def forward(self, observed_kspace, old_mask, budget, uncertainty_mask=None, valid_mask=None, recon_complex=None, gfactor=None):  # noqa: ARG002
        """
        Returns:
            binary_mask: Updated sampling mask (includes previous locations), shape NCHW
            data_to_vis_sampler: Dict of intermediate results for visualization
            anatomy_logits: [N, num_classes] or None
        """
        if self.detach_kspace:
            observed_kspace = observed_kspace.detach()

        input_kspace = torch.cat([observed_kspace, observed_kspace * old_mask.permute(0, 2, 3, 1)], dim=-1)

        samp_prob_unscaled, sampler_uncertainty, anatomy_logits = self._predict_probabilities(
            input_kspace, old_mask, recon_complex, gfactor
        )

        if self.line_constrained:
            sparsity = budget / self.shape[0]
        else:
            sparsity = budget / (self.shape[0] * self.shape[1])

        samp_prob = self.rescale(samp_prob_unscaled, sparsity)
        binary_mask = self.binarize(samp_prob)

        if valid_mask is not None:
            if self.line_constrained:
                valid_mask = (valid_mask.sum(dim=2, keepdim=True) > 0).to(binary_mask.dtype)
            binary_mask = binary_mask * valid_mask

        binary_mask = torch.clamp(old_mask + binary_mask, min=0, max=1)

        data_to_vis_sampler = {
            'prob_mask': samp_prob_unscaled[0, 0, :, :].detach(),
            'rescaled_mask': samp_prob[0, 0, :, :].detach(),
            'binarized_mask': binary_mask[0, 0, :, :].detach(),
        }

        if sampler_uncertainty is not None:
            data_to_vis_sampler['sampler_uncertainty'] = sampler_uncertainty[0, 0].detach()

        return binary_mask, data_to_vis_sampler, anatomy_logits
