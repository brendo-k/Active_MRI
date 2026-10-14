"""Selects and runs one of the sampling families.

`Sampler` owns the choice of family and the shared post-processing that turns a
raw score map into a binary acquisition mask.
"""
from functools import partial

import torch
from torch import nn

from activemri.samplers.autosamp import AutoSampCoordinates
from activemri.samplers.caipi import caipi_mask
from activemri.samplers.conv import KspaceLineConstrainedSampler, SampProb2D
from activemri.samplers.loupe import LineConstrainedProbMask, ProbMask
from activemri.samplers.oracle import plackett_luce_sample_and_logprob
from activemri.utils.image_processing import ifft_2d_img
from activemri.utils.mask_binarization import RescaleProbMap, threshold_random_mask


SAMPLER_TYPES = ("conv_2d", "conv_1d", "loupe_2d", "loupe_1d", "caipi_2d", "autosamp_2d")
MASK_TRAIN_MODES = ("straight_through", "reinforce", "random")
REINFORCE_EVAL_MODES = ("sample", "topk")


def is_line_constrained(sampler_type):
    """Whether a sampler selects whole k-space columns rather than single locations."""
    return sampler_type.endswith("_1d")


class Sampler(nn.Module):
    """Generates sampling masks for sequential k-space acquisition.

    Six sampler families are available, selected by `sampler_type`:

        conv_2d      U-Net over k-space, one score per location   (input-adaptive)
        conv_1d      MLP over k-space, one score per column       (input-adaptive)
        loupe_2d     learned mask, one weight per location        (input-independent)
        loupe_1d     learned mask, one weight per column          (input-independent)
        caipi_2d     fixed 2x2 CAIPI lattice, no parameters       (input-independent)
        autosamp_2d  learned coordinates snapped to the grid      (input-independent)

    `autosamp_2d` (see :mod:`activemri.samplers.autosamp`) learns a fixed number of
    continuous k-space positions and snaps each to its nearest free grid cell. The
    count is fixed at construction (`autosamp_num_points`), so it is a single-step
    sampler and ignores the per-call `budget`.

    The LOUPE families ignore the observed k-space; their mask is a free
    parameter shared across the dataset. By default all four are put through
    the same rescale-to-sparsity and straight-through binarization steps
    (`mask_train_mode='straight_through'`); LOUPE families can instead be
    trained with `mask_train_mode='reinforce'`, which samples an exact-budget
    binary mask via Gumbel-top-k (Plackett-Luce) from the mask's raw logits
    and hands back its log-probability instead of a biased gradient estimator
    -- see :func:`activemri.samplers.oracle.plackett_luce_sample_and_logprob`
    and the oracle-only experiments this was validated against
    (`scripts/loupe_vs_greedy_oracle.py`, `scripts/reinforce2d_vs_greedy_loupe.py`).
    The conv families are input-adaptive and don't fit this framing (their
    score depends on a partial reconstruction that would itself need to be
    resampled per Monte-Carlo draw), so `reinforce` is LOUPE-only for now.

    With `num_conditioning_classes > 0` every family is additionally given the
    anatomy of each sample: the conv families take a learned embedding as extra
    network input, and the LOUPE families keep one free mask per anatomy. Without
    it a mixed-anatomy run collapses to a single dataset-average pattern.
    """

    def __init__(
        self,
        sampler_type="conv_2d",
        shape=(320, 320),
        slope=10,
        with_uncertainty=False,
        detach_kspace=False,
        fixed_input=False,
        with_recon=False,
        with_gfactor=False,
        uncertainty_mode='none',
        num_anatomy_classes=0,
        num_conditioning_classes=0,
        mask_train_mode='straight_through',
        sampler_chans=64,
        sampler_depth=4,
        image_domain_input=False,
        reinforce_vd_power=0.0,
        reinforce_eval='sample',
        caipi_keep_fraction=1.0,
        autosamp_num_points=0,
        autosamp_init='uniform',
        mask_grad_floor=0.0,
        ste_grad_floor=0.0,
        ):
        super().__init__()

        if sampler_type not in SAMPLER_TYPES:
            raise ValueError(f"Unknown sampler_type {sampler_type!r}; expected one of {SAMPLER_TYPES}")
        if mask_train_mode not in MASK_TRAIN_MODES:
            raise ValueError(f"Unknown mask_train_mode {mask_train_mode!r}; expected one of {MASK_TRAIN_MODES}")
        if reinforce_eval not in REINFORCE_EVAL_MODES:
            raise ValueError(f"Unknown reinforce_eval {reinforce_eval!r}; expected one of {REINFORCE_EVAL_MODES}")

        self.sampler_type = sampler_type
        self.line_constrained = is_line_constrained(sampler_type)
        self.is_loupe = sampler_type.startswith("loupe")
        if image_domain_input and sampler_type != 'conv_2d':
            raise ValueError("image_domain_input is only implemented for conv_2d")
        self.image_domain_input = image_domain_input
        self.mask_train_mode = mask_train_mode
        if mask_train_mode in ('reinforce', 'random') and not self.is_loupe:
            raise ValueError(f"mask_train_mode={mask_train_mode!r} is only implemented for loupe_2d/loupe_1d")
        self.shape = shape
        self.detach_kspace = detach_kspace
        self.with_uncertainty = with_uncertainty
        self.with_recon = with_recon
        self.uncertainty_mode = uncertainty_mode
        self.with_gfactor = with_gfactor
        self.num_anatomy_classes = num_anatomy_classes
        self.num_conditioning_classes = num_conditioning_classes
        # How `reinforce` deploys the policy in eval: 'sample' draws a mask from it, as in
        # training; 'topk' takes the highest logits, which for a smooth policy is a
        # coherent low-pass disc the reconstructor never trained on.
        self.reinforce_eval = reinforce_eval
        # Set by the trainer and reseeded every val epoch, so eval draws repeat exactly.
        self.eval_generator: torch.Generator | None = None

        if sampler_type == "caipi_2d":
            self.register_buffer("caipi_pattern", caipi_mask(shape, keep_fraction=caipi_keep_fraction)[None, None])
        elif sampler_type == "autosamp_2d":
            self.mask_net = AutoSampCoordinates(shape, autosamp_num_points, init=autosamp_init)
        elif sampler_type == "conv_2d":
            self.mask_net = SampProb2D(
                fixed_input=fixed_input,
                with_recon=with_recon,
                with_gfactor=with_gfactor,
                uncertainty_mode=uncertainty_mode,
                num_anatomy_classes=num_anatomy_classes,
                shape=shape,
                num_conditioning_classes=num_conditioning_classes,
                chans=sampler_chans,
                depth=sampler_depth,
            )
        elif sampler_type == "conv_1d":
            if with_uncertainty:
                raise NotImplementedError("conv_1d does not support with_uncertainty")
            self.mask_net = KspaceLineConstrainedSampler(
                shape[0], fixed_input=fixed_input,
                num_anatomy_classes=num_anatomy_classes,
                num_conditioning_classes=num_conditioning_classes,
            )
        elif sampler_type == "loupe_2d":
            self.mask_net = ProbMask(shape, slope, num_conditioning_classes=num_conditioning_classes,
                                     grad_floor=mask_grad_floor)
        else:
            self.mask_net = LineConstrainedProbMask(shape, slope, num_conditioning_classes=num_conditioning_classes,
                                                    grad_floor=mask_grad_floor)

        if mask_train_mode == 'reinforce' and reinforce_vd_power > 0:
            self.mask_net.init_variable_density(reinforce_vd_power)

        self.rescale = RescaleProbMap
        self.binarize = partial(threshold_random_mask, grad_floor=ste_grad_floor)

    @staticmethod
    def _to_image_domain(kspace_channels):
        """IFFT each (real, imag) pair of a NHWC k-space stack, keeping the layout."""
        pairs = kspace_channels.reshape(*kspace_channels.shape[:-1], -1, 2)
        images = ifft_2d_img(torch.view_as_complex(pairs.contiguous()), axes=[1, 2])
        return torch.view_as_real(images).reshape(kspace_channels.shape)

    def _predict_probabilities(self, input_kspace, old_mask, recon_complex, gfactor, anatomy_labels):
        """Run the mask network, returning (scores NCHW, uncertainty or None, anatomy_logits or None)."""
        if self.is_loupe:
            prob = self.mask_net(input_kspace, anatomy_labels).permute(0, 3, 1, 2)
            prob = prob.expand(input_kspace.shape[0], -1, -1, -1)
            already = (old_mask.sum(dim=-2, keepdim=True) == self.shape[0]).to(prob.dtype) \
                if self.line_constrained else old_mask
            return prob * (1 - already), None, None

        if self.line_constrained:
            prob, anatomy_logits = self.mask_net(input_kspace, old_mask, anatomy_labels=anatomy_labels)
            return prob, None, anatomy_logits

        prob, uncertainty, anatomy_logits = self.mask_net(
            input_kspace, old_mask, recon=recon_complex, gfactor=gfactor,
            anatomy_labels=anatomy_labels,
        )
        return prob, uncertainty, anatomy_logits

    def _reinforce_forward(self, old_mask, budget, valid_mask, anatomy_labels, uniform=False):
        """Exact-budget Plackett-Luce sampling from the LOUPE mask's raw logits.

        Args:
            old_mask: [B, 1, H, W] locations already sampled in earlier steps.
            budget: Target number of *new* locations/columns this step.
            valid_mask: Optional [B, 1, H, W] mask of in-bounds k-space.
            anatomy_labels: Optional [B] long tensor.
            uniform: If True, sample uniformly at random instead of from the
                mask's learned logits, and don't touch mask_net's parameters
                (used by `mask_train_mode='random'` to pretrain a reconstructor
                against an arbitrary mask distribution, not the policy's own).

        Returns:
            binary_mask: [B, 1, H, W], old_mask union this step's draw.
            data_to_vis_sampler: dict, for visualization only.
            logp: [B] log-probability of this step's draw, differentiable in
                the mask logits -- what the REINFORCE loss term needs. None
                when `uniform=True` (no policy to credit) and in eval mode,
                where the mask is the deterministic top-k of the logits.
        """
        B = old_mask.shape[0]
        if uniform:
            flat = self.shape[1] if self.line_constrained else self.shape[0] * self.shape[1]
            raw = torch.zeros(B, flat, device=old_mask.device)
        else:
            raw = self.mask_net.raw_logits(anatomy_labels)  # [1 or B, ...]
            if raw.shape[0] == 1:
                raw = raw.expand(B, *raw.shape[1:])

        if self.line_constrained:
            logits = raw.reshape(B, self.shape[1])
            taken = (old_mask.sum(dim=-2, keepdim=True) == self.shape[0]).view(B, -1)
            if valid_mask is not None:
                taken = taken | ((valid_mask.sum(dim=2, keepdim=True) <= 0).view(B, -1))
        else:
            logits = raw.reshape(B, -1)
            taken = old_mask.reshape(B, -1) > 0
            if valid_mask is not None:
                taken = taken | (valid_mask.reshape(B, -1) <= 0)

        logits = logits.masked_fill(taken, float('-inf'))

        # Entropy of the first-pick categorical over remaining positions --
        # an exploration signal for `mask_train_mode='reinforce'`: high when
        # the policy still spreads mass broadly, collapsing toward 0 as it
        # commits to one pattern. Differentiable in the logits, so an
        # `entropy_weight` term in the loss can discourage premature collapse.
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # A single Plackett-Luce call needs one shared k across the batch; if
        # per-example valid regions differ (e.g. padded mixed-anatomy data),
        # cap k at the tightest example rather than crash or silently pick a
        # different budget per row.
        k = min(int(round(float(budget))), int((~taken).sum(dim=1).min().item()))
        # In eval there is nothing to credit, so no log-prob is returned -- which also
        # keeps the REINFORCE term out of the validation loss that picks the best
        # checkpoint. The mask is a draw from the policy (reinforce_eval='sample'),
        # from `eval_generator` so it repeats every epoch, or its top-k ('topk').
        evaluating = not self.training and not uniform
        if evaluating and self.reinforce_eval == 'topk':
            order, logp = torch.topk(logits, k, dim=1).indices, None
        else:
            order, logp = plackett_luce_sample_and_logprob(
                logits, k, generator=self.eval_generator if evaluating else None)

        delta = torch.zeros_like(logits)
        delta.scatter_(1, order, 1.0)  # delta has no grad history -- safe in-place

        if self.line_constrained:
            delta_mask = delta.view(B, 1, 1, self.shape[1]).expand(-1, -1, self.shape[0], -1)
        else:
            delta_mask = delta.view(B, 1, self.shape[0], self.shape[1])

        binary_mask = torch.clamp(old_mask + delta_mask.detach(), min=0, max=1)

        vis_logits = logits[0].detach().clone()
        vis_logits[torch.isinf(vis_logits)] = 0.0
        data_to_vis_sampler = {
            'prob_mask': vis_logits.view(1, -1) if self.line_constrained else vis_logits.view(*self.shape),
            'rescaled_mask': delta_mask[0, 0].detach(),
            'binarized_mask': binary_mask[0, 0].detach(),
        }
        draws_policy = not (uniform or evaluating)
        return binary_mask, data_to_vis_sampler, (logp if draws_policy else None), (entropy if draws_policy else None)

    def forward(self, observed_kspace, old_mask, budget, uncertainty_mask=None, valid_mask=None, recon_complex=None, gfactor=None, anatomy_labels=None):  # noqa: ARG002
        """
        Returns:
            binary_mask: Updated sampling mask (includes previous locations), shape NCHW
            data_to_vis_sampler: Dict of intermediate results for visualization
            anatomy_logits: [N, num_classes] or None
            logp: [B] Plackett-Luce log-probability of this step's draw, or
                None outside `mask_train_mode='reinforce'`.
            entropy: [B] entropy of this step's draw distribution, or None
                outside `mask_train_mode='reinforce'`.
        """
        if self.detach_kspace:
            observed_kspace = observed_kspace.detach()

        if self.sampler_type == "caipi_2d":
            binary_mask = torch.clamp(old_mask + self.caipi_pattern.to(old_mask.dtype), max=1)
            data_to_vis_sampler = {
                'prob_mask': self.caipi_pattern[0, 0],
                'rescaled_mask': self.caipi_pattern[0, 0],
                'binarized_mask': binary_mask[0, 0].detach(),
            }
            return binary_mask, data_to_vis_sampler, None, None, None

        if self.sampler_type == "autosamp_2d":
            new_mask, soft_mask = self.mask_net(old_mask, valid_mask)
            binary_mask = torch.clamp(old_mask + new_mask, min=0, max=1)
            data_to_vis_sampler = {
                'prob_mask': soft_mask,
                'rescaled_mask': new_mask[0, 0].detach(),
                'binarized_mask': binary_mask[0, 0].detach(),
            }
            return binary_mask, data_to_vis_sampler, None, None, None

        if self.mask_train_mode == 'reinforce':
            binary_mask, data_to_vis_sampler, logp, entropy = self._reinforce_forward(
                old_mask, budget, valid_mask, anatomy_labels)
            return binary_mask, data_to_vis_sampler, None, logp, entropy

        if self.mask_train_mode == 'random':
            binary_mask, data_to_vis_sampler, _, _ = self._reinforce_forward(
                old_mask, budget, valid_mask, anatomy_labels, uniform=True)
            return binary_mask, data_to_vis_sampler, None, None, None

        sampler_input = torch.cat([observed_kspace, observed_kspace * old_mask.permute(0, 2, 3, 1)], dim=-1)
        if self.image_domain_input:
            sampler_input = self._to_image_domain(sampler_input)

        samp_prob_unscaled, sampler_uncertainty, anatomy_logits = self._predict_probabilities(
            sampler_input, old_mask, recon_complex, gfactor, anatomy_labels
        )

        if self.line_constrained:
            sparsity = budget / self.shape[0]
        else:
            sparsity = budget / (self.shape[0] * self.shape[1])

        if self.line_constrained:
            available = (old_mask.sum(dim=2, keepdim=True) < self.shape[0]).to(old_mask.dtype)
            if valid_mask is not None:
                valid_mask = (valid_mask.sum(dim=2, keepdim=True) > 0).to(old_mask.dtype)
        else:
            available = 1 - old_mask
        if valid_mask is not None:
            available = available * valid_mask

        samp_prob = self.rescale(samp_prob_unscaled, sparsity, available)
        binary_mask = self.binarize(samp_prob)

        if valid_mask is not None:
            binary_mask = binary_mask * valid_mask

        binary_mask = torch.clamp(old_mask + binary_mask, min=0, max=1)

        data_to_vis_sampler = {
            'prob_mask': samp_prob_unscaled[0, 0, :, :].detach(),
            'rescaled_mask': samp_prob[0, 0, :, :].detach(),
            'binarized_mask': binary_mask[0, 0, :, :].detach(),
        }

        if sampler_uncertainty is not None:
            data_to_vis_sampler['sampler_uncertainty'] = sampler_uncertainty[0, 0].detach()

        return binary_mask, data_to_vis_sampler, anatomy_logits, None, None
