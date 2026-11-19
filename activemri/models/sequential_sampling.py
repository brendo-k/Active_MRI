"""The sequential sampling model: acquire, reconstruct, repeat.

Composes a `Sampler` with an injected reconstructor. The reconstructor is a
constructor argument, so this model is independent of which one is used.
"""
import contextlib
import copy
import os

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from activemri.reconstructors.CG_Sense import compute_gfactor
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.samplers.fixed_mask import FileMaskPolicy
from activemri.samplers.ppo import PPOMaskPolicy
from activemri.samplers.sampler import Sampler, is_line_constrained
from activemri.samplers.seq_adapt import StepContext, acs_compression, calibration_peaks
from activemri.utils.image_processing import fft_2d_img as torch_fft2
from activemri.utils.image_processing import ifft_2d_img as torch_ifft2
from activemri.utils.util import compute_gaussian_nll_loss, compute_psnr_torch, compute_ssim_torch


RSS_EPSILON = 1e-12
BUDGET_REGIONS = ('grid', 'acquired')
SAMPLER_COIL_COMBINES = ('sense', 'rss')
SAMPLER_INPUTS = ('measured', 'recon', 'full', 'seqmri')
IN_LOOP_RECON_STEPS = (6, 9, 12, 14)


def reinforce_group_size(batch_size, configured):
    """Mask draws per slice in a REINFORCE batch: `configured`, or the whole batch if 0.

    Capped at the batch size so a short final batch still forms one group.
    """
    return min(configured or batch_size, batch_size)


class SequentialSampling(nn.Module):
    """Sequential MRI sampling and reconstruction model.

    Iteratively selects k-space measurements and reconstructs images. At each step,
    a sampler network chooses new acquisition locations based on the current
    reconstruction, and a reconstructor produces an updated image estimate.

    Pre-selects center k-space lines/region before adaptive sampling begins.
    """

    def __init__(
            self,
            reconstructor: nn.Module,
            shape=(320, 320),
            sparsity=0.25,
            sampler_type='conv_2d',
            slope=10,
            num_step=4,
            preselect=True,
            preselect_num=2,
            with_uncertainty=False,
            detach_kspace=False,
            fixed_input=False,
            recon_type='unet',
            sense_tikhonov_weight=0.0,
            sense_mask_method='pre_network',
            with_recon=False,
            with_gfactor=False,
            gfactor_replicas=16,
            sens_upsample_method='conv',
            sens_conv_after_upsample=False,
            sens_resolution=None,
            sens_precision='fp32',
            uncertainty_mode='none',
            num_anatomy_classes=0,
            num_conditioning_classes=0,
            condition_reconstructor=False,
            mask_train_mode='straight_through',
            reinforce_weight=1.0,
            reinforce_entropy_weight=0.0,
            reinforce_normalize_advantages=False,
            reinforce_group_size=0,
            reinforce_vd_power=0.0,
            reinforce_eval='sample',
            sampler_chans=64,
            sampler_depth=4,
            sampler_head='none',
            sampler_image_input=False,
            caipi_keep_fraction=1.0,
            caipi_accel_y=2,
            caipi_accel_z=2,
            autosamp_init='uniform',
            mask_grad_floor=0.0,
            ste_grad_floor=0.0,
            mask_draws=1,
            logit_limit=6.0,
            logit_leak=0.01,
            adapt_grid=16,
            adapt_hidden=64,
            bank_size=8,
            gate_temperature=1.0,
            gate_logit_limit=0.0,
            selection='bernoulli',
            dcm_vd_power=4.0,
            dcm_tau=5.0,
            ppo_policy=None,
            fixed_mask_path=None,
            loupe_vd_init_power=0.0,
            loupe_same_start=False,
            acquired_only=False,
            budget_over='grid',
            restrict_recon_to_extent=False,
            sampler_coil_combine='sense',
            residual_chans=16,
            residual_depth=4,
            residual_limit=6.0,
            data_blind=False,
            residual_checkpoint=True,
            data_channels=None,
            sampler_input='measured',
            recon_steps=IN_LOOP_RECON_STEPS,
            virtual_coils=4,
            seqmri_input_scale='raw',
            ):
        """`budget_over='acquired'` makes `sparsity` a fraction of each slice's acquired k-space (`mask_budget`);
        'grid' (the default, as older runs trained) a fraction of the whole grid, all of it spent inside the acquired
        region. `restrict_recon_to_extent` confines a VarNet's regularizer to the acquired extent when the data
        supplies no `kspace_support`. `sampler_coil_combine` is how the sampler's input image is formed from the
        observed coil k-space (`_sampler_image`). `residual_*` and `data_blind` configure seq_adapt_2d's residual
        network, its data-blind twin and the data channels it reads (`activemri.samplers.seq_adapt`). `sampler_input='recon'`
        also feeds that residual a frozen copy of the reconstructor's image of y * M_i at every step; the copy's weights
        are loaded by the trainer (`input_recon_checkpoint`) and never train.

        Two kscr2 inputs of the adaptive residual (addendum_2026-11-15.md):
        - 'full', the clairvoyant ceiling: the data channels are made from the fully sampled k-space at every step.
        - 'seqmri': before each step in `recon_steps` the trained reconstructor reconstructs y * M_i, and the residual
          also reads `virtual_coils` virtual coils of y * M_i and of the coil k-space FFT(S x) predicted from the latest of
          those reconstructions. The reconstruction sees M_i detached, so the final loss trains its weights through the
          sampler's input but adds no gradient path to earlier draws.

        `sampler_type='seqmri_2d'` is SeqMRI's loop (activemri.samplers.seqmri, kscr3): the trained reconstructor
        reconstructs before every step, the centre included, and the gradient takes SeqMRI's routes (`_sample_steps`).
        `seqmri_input_scale` scales its input (fscr1: 'signed_log'). Its `data_blind` twin reads no reconstruction, so
        none runs before its steps; it keeps SeqMRI's routes, so its earlier draws are credited only through the weights
        every step shares.

        At num_step > 1 the steps share the budget equally, each reads the k-space measured so far, and only the final
        mask is reconstructed (plus the 'seqmri' in-loop reconstructions), so the sampler inputs that need an
        intermediate reconstruction of the old kind are refused."""
        super().__init__()
        if budget_over not in BUDGET_REGIONS:
            raise ValueError(f"budget_over must be one of {BUDGET_REGIONS}, got {budget_over!r}")
        if sampler_coil_combine not in SAMPLER_COIL_COMBINES:
            raise ValueError(f"sampler_coil_combine must be one of {SAMPLER_COIL_COMBINES}, got {sampler_coil_combine!r}")
        if sampler_input not in SAMPLER_INPUTS:
            raise ValueError(f"sampler_input must be one of {SAMPLER_INPUTS}, got {sampler_input!r}")
        if sampler_input in ('full', 'seqmri') and (sampler_type != 'seq_adapt_2d' or data_blind):
            raise ValueError(f"sampler_input={sampler_input!r} is an input of seq_adapt_2d's adaptive residual; "
                             "a data-blind twin cannot read a reconstruction of the data")
        self.recon_steps = ()
        if sampler_input == 'seqmri':
            self.recon_steps = tuple(sorted({int(step) for step in recon_steps}))
            if num_step < 2 or not self.recon_steps or self.recon_steps[0] < 0 or self.recon_steps[-1] >= num_step:
                raise ValueError(f"in-loop reconstructions need steps in [0, {num_step}), got {list(recon_steps)}")
            if virtual_coils < 1:
                raise ValueError(f"sampler_input='seqmri' reads at least one virtual coil, got {virtual_coils}")
        self.seqmri_routes = sampler_type == 'seqmri_2d'
        if self.seqmri_routes:
            if sampler_input != 'measured':
                raise ValueError("seqmri_2d reads SeqMRI's own input (the predicted and measured k-space and the mask); "
                                 f"leave sampler_input at 'measured', got {sampler_input!r}")
            self.recon_steps = () if data_blind else tuple(range(num_step))
        self.sampler_input = sampler_input
        self.sampler_coil_combine = sampler_coil_combine
        self.sparsity = sparsity
        self.budget_over = budget_over
        self.restrict_recon_to_extent = restrict_recon_to_extent

        self.recon_type = recon_type
        self.sense_mask_method = sense_mask_method
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.preselect_num_one_side = preselect_num // 2
        self.sampler_type = sampler_type
        self.line_constrained = is_line_constrained(sampler_type)
        self.with_uncertainty = with_uncertainty
        self.with_gfactor = with_gfactor
        self.gfactor_replicas = gfactor_replicas
        self.uncertainty_mode = uncertainty_mode
        self.sense_tikhonov_weight = sense_tikhonov_weight
        self.num_anatomy_classes = num_anatomy_classes
        self.num_conditioning_classes = num_conditioning_classes
        self.condition_reconstructor = condition_reconstructor
        self.mask_train_mode = mask_train_mode
        self.reinforce_weight = reinforce_weight
        self.reinforce_entropy_weight = reinforce_entropy_weight
        self.reinforce_normalize_advantages = reinforce_normalize_advantages
        self.reinforce_group_size = reinforce_group_size
        self.acquired_only = acquired_only
        self.zero_unacquired = False  # inference-only switch: fixed masks and the predicted k-space are zeroed where the scan acquired nothing (padded columns)
        self.data_for_vis: list[dict] = [dict() for _ in range(num_step)]
        self.store_visuals = True  # copy the first slice's images to data_for_vis on every forward (a host sync); the trainer turns it off between logged batches
        self.step_recorder = None  # an activemri.utils.multistep.StepRecorder that every step of the next forwards reports to

        self.reconstructor = reconstructor

        # Compute per-step sampling budget
        if not self.line_constrained:
            effective_sparsity = sparsity - (preselect_num ** 2) / (shape[0] * shape[1]) if preselect else sparsity
            self.budget = effective_sparsity * shape[0] * shape[1]
        else:
            effective_sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity
            self.budget = effective_sparsity * shape[0]

        if sampler_type == 'autosamp_2d' and num_step != 1:
            raise ValueError(f"autosamp_2d learns one fixed set of positions, so it needs num_step=1, got {num_step}")
        if num_step > 1 and (with_recon or with_gfactor):
            raise ValueError("a multi-step run reconstructs only its final mask, so there is no intermediate "
                             f"reconstruction to feed the sampler (with_recon={with_recon}, with_gfactor={with_gfactor})")

        # Built before the sampler, whose random draws depend on its mask count, so arms that differ only in their
        # sampler start from the same sensitivity net.
        self.sensetivity_net = SensetivityModel_mc(
            2, 2, 8, conv_after_upsample=sens_conv_after_upsample, upsample_method=sens_upsample_method,
            resolution=sens_resolution, precision=sens_precision)

        self.sampler = Sampler(
            sampler_type=sampler_type,
            shape=shape,
            slope=slope,
            with_uncertainty=with_uncertainty,
            detach_kspace=detach_kspace,
            fixed_input=fixed_input,
            with_recon=with_recon,
            with_gfactor=with_gfactor,
            uncertainty_mode=uncertainty_mode,
            num_anatomy_classes=num_anatomy_classes,
            num_conditioning_classes=num_conditioning_classes,
            mask_train_mode=mask_train_mode,
            sampler_chans=sampler_chans,
            sampler_depth=sampler_depth,
            sampler_head=sampler_head,
            image_domain_input=sampler_image_input,
            reinforce_vd_power=reinforce_vd_power,
            reinforce_eval=reinforce_eval,
            caipi_keep_fraction=caipi_keep_fraction,
            caipi_accel_y=caipi_accel_y,
            caipi_accel_z=caipi_accel_z,
            autosamp_num_points=round(self.budget),
            poisson_acceleration=1 / sparsity,
            autosamp_init=autosamp_init,
            mask_grad_floor=mask_grad_floor,
            ste_grad_floor=ste_grad_floor,
            mask_draws=mask_draws,
            logit_limit=logit_limit,
            logit_leak=logit_leak,
            adapt_grid=adapt_grid,
            adapt_hidden=adapt_hidden,
            bank_size=bank_size,
            gate_temperature=gate_temperature,
            gate_logit_limit=gate_logit_limit,
            calibration_size=preselect_num,
            selection=selection,
            dcm_vd_power=dcm_vd_power,
            dcm_tau=dcm_tau,
            residual_chans=residual_chans,
            residual_depth=residual_depth,
            residual_limit=residual_limit,
            data_blind=data_blind,
            residual_checkpoint=residual_checkpoint,
            data_channels=data_channels,
            reads_reconstruction=sampler_input == 'recon',
            complex_coils=virtual_coils if sampler_input == 'seqmri' else 0,
            seqmri_input_scale=seqmri_input_scale,
        )

        if loupe_same_start:
            self.sampler.start_masks_from_first()
        if loupe_vd_init_power > 0:
            self.sampler.init_loupe_variable_density(
                loupe_vd_init_power, 1 - self._init_mask(torch.zeros(1))[0, 0], self.budget)

        # A frozen PPO policy or a saved mask replaces the sampler's mask (the sampler stays only as a placeholder).
        if ppo_policy and fixed_mask_path:
            raise ValueError("ppo_policy and fixed_mask_path both replace the sampler's mask; set one")
        if ppo_policy:
            self.mask_policy = PPOMaskPolicy(ppo_policy, sense_mask_method)
        elif fixed_mask_path:
            self.mask_policy = FileMaskPolicy(fixed_mask_path, shape, centre=self._init_mask(torch.zeros(1))[0, 0])
        else:
            self.mask_policy = None
        if self.mask_policy is not None and num_step != 1:
            raise ValueError(f"a mask policy produces the whole mask at once, so it needs num_step=1, got {num_step}")

        # Copied after the sampler so they draw nothing from the random stream; build_model gives them the same norms.
        frozen = sampler_input == 'recon'
        self.input_reconstructor = copy.deepcopy(self.reconstructor).requires_grad_(False).eval() if frozen else None
        self.input_sensetivity_net = copy.deepcopy(self.sensetivity_net).requires_grad_(False).eval() if frozen else None

        print(f"Sampler: {sampler_type}")
        print(f"Number of sequential steps: {num_step}")
        print(f"Sparsity: {sparsity}")

    def _valid_kspace_mask(self, full_kspace):
        """Locations where the original k-space carries signal across any coil.

        Zero-filled regions (e.g. padded readout/phase encodes in mixed-anatomy
        data) return 0 so the sampler never spends budget acquiring them.

        Args:
            full_kspace: Ground truth multicoil kspace, shape [B, C, H, W, 2]

        Returns:
            Mask tensor, shape [B, 1, H, W]
        """
        coil_energy = torch.view_as_complex(full_kspace.contiguous()).abs().sum(dim=1)
        return (coil_energy > 0).to(full_kspace.dtype).unsqueeze(1)

    def _kspace_extent(self, full_kspace):
        """The rectangle of k-space rows and columns that hold data in any coil, shape [B, 1, H, W].

        For fully sampled data this is the acquired k-space, and outside it the grid was zero-filled. Unlike
        `_valid_kspace_mask` it stays the whole envelope when points inside it are missing (e.g. a holdout split),
        unless an entire row or column is missing.
        """
        valid = self._valid_kspace_mask(full_kspace)
        return valid.amax(dim=-1, keepdim=True) * valid.amax(dim=-2, keepdim=True)

    def mask_budget(self, old_mask, valid_mask):
        """New locations (columns, for a line-constrained sampler) the whole acquisition adds to the pre-selected centre.

        With budget_over='acquired' it is one per slice, [B]: the mask, centre included, then holds `sparsity` of the
        slice's acquired k-space (`valid_mask`). With 'grid' it is the scalar `self.budget`, `sparsity` of the grid.
        """
        if self.budget_over == 'grid':
            return self.budget
        if self.line_constrained:
            old_mask, valid_mask = old_mask.amax(dim=-2), valid_mask.amax(dim=-2)
        acquired = valid_mask.flatten(1).sum(dim=1).double()
        preselected = (old_mask * valid_mask).flatten(1).sum(dim=1).double()
        return self.sparsity * acquired - preselected

    @staticmethod
    def _coil_combine(coil_kspace, sense_maps):
        """SENSE coil combination sum_c conj(S_c) IFFT(k_c) [B, H, W] of complex coil k-space [B, C, H, W]."""
        return (torch_ifft2(coil_kspace) * sense_maps.conj()).sum(dim=1)

    def _sampler_image(self, coil_kspace, sense_maps):
        """The complex image [B, H, W] the sampler's input is formed from.

        'sense' is the SENSE combination. Its phase is the maps' common phase, which an RSS magnitude loss leaves
        free, so a jointly trained sensitivity net makes it pixel-to-pixel random. 'rss' is the RSS of the coil
        images (real, nonnegative), which needs no maps and has no phase to scramble.
        """
        if self.sampler_coil_combine == 'rss':
            return torch_ifft2(coil_kspace).abs().pow(2).sum(dim=1).sqrt().to(coil_kspace.dtype)
        return self._coil_combine(coil_kspace, sense_maps)

    @staticmethod
    def _coil_rss(image, sense_maps, kspace_extent):
        """RSS magnitude [B, 1, H, W] of the coil images S_c x, each coil's k-space zeroed outside `kspace_extent`, as
        the target is the RSS of the zero-filled coil k-space."""
        coil_kspace = torch_fft2(image * sense_maps) * kspace_extent
        coil_power = torch_ifft2(coil_kspace).abs().pow(2).sum(dim=1, keepdim=True)
        # The epsilon keeps sqrt's gradient finite at pixels that are exactly zero in every
        # coil (masked-out background); there sqrt' is inf and pow(2)' is 0, giving NaN.
        return (coil_power + RSS_EPSILON).sqrt()

    def step_forward(self, full_kspace, pred_kspace, old_mask, target, step,
                     sense_maps, sense_mask, uncertainty_map=None, valid_mask=None,
                     prev_gfactor=None, anatomy_labels=None, fixed_mask=None, kspace_support=None, budget=None,
                     context=None):
        """Execute one sampling + reconstruction step (`_sample`, then `_reconstruct`).

        Args:
            full_kspace: Ground truth multicoil kspace, shape [B, C, H, W, 2]
            pred_kspace: The k-space the sampler reads, shape [B, C, H, W, 2]: the measured full_kspace * old_mask
            old_mask: Previous sampling mask, shape [B, 1, H, W]
            target: Ground truth image, shape [B, 1, H, W]
            step: Current iteration index
            sense_maps: Estimated sensitivity maps, shape [B, C, H, W] complex
            sense_mask: Body mask from sensitivity estimation, shape [B, 1, H, W]
            uncertainty_map: Optional uncertainty estimate
            prev_gfactor: G-factor map from previous step [B, 1, H, W] for
                noise-aware sampling guidance.
            anatomy_labels: Optional long tensor [B] of anatomy indices, used to
                condition the sampler (and the reconstructor, if it supports it).
            fixed_mask: Optional mask [B, 1, H, W] used instead of the sampler's; it is cut to `valid_mask`.
            budget: New locations this step, a number or one per slice; None = an equal share of `self.budget`.
            context: The StepContext a `reads_context` sampler (seq_adapt_2d) reads.

        Returns:
            Dict with keys: output, mask, zero_recon, uncertainty_map, gfactor, anatomy_logits, reinforce_logp,
            reinforce_entropy
        """
        sampled = self._sample(full_kspace, pred_kspace, old_mask, step, sense_maps, uncertainty_map, valid_mask,
                               prev_gfactor, anatomy_labels, fixed_mask, budget, context)
        reconstructed = self._reconstruct(full_kspace, sampled['mask'], target, step, sense_maps, sense_mask,
                                          uncertainty_map, anatomy_labels, kspace_support)
        return {**reconstructed, **sampled}

    def _sample(self, full_kspace, observed_kspace, old_mask, step, sense_maps, uncertainty_map=None, valid_mask=None,
                prev_gfactor=None, anatomy_labels=None, fixed_mask=None, budget=None, context=None, threshold=None):
        """This step's mask, old_mask union its new locations: the sampler's, or `fixed_mask` cut to `valid_mask`.

        `observed_kspace` [B, C, H, W, 2] is the k-space the sampler reads; a `reads_context` sampler reads `context`
        instead. `threshold` [B, 1, H, W] of U(0, 1) values replaces the Bernoulli draw's fresh noise, so runs can share
        their draws. Returns a dict with the mask, the sampler's anatomy logits and REINFORCE terms, and `details`: the
        step's unscaled and rescaled probabilities and, for seq_adapt_2d, its residual (empty for a fixed mask).
        """
        if budget is None:
            budget = self.budget / self.num_step

        recon_complex = None
        if fixed_mask is not None or not self.sampler.reads_input:
            sampler_kspace = observed_kspace[:, 0]  # the right shape, which is all the sampler takes from it
        else:
            sampler_image = self._sampler_image(torch.view_as_complex(observed_kspace), sense_maps)
            sampler_kspace = torch.view_as_real(torch_fft2(sampler_image))

            if torch.isnan(sampler_kspace).any():
                print(f"NaN in the sampler's input at step {step}")

            if self.sampler.with_recon:
                recon_2ch = torch.stack([sampler_image.real, sampler_image.imag], dim=1)
                recon_complex = recon_2ch.detach()

        details = {}
        if fixed_mask is not None:
            mask = fixed_mask if valid_mask is None else fixed_mask * valid_mask
            data_to_vis_sampler = dict.fromkeys(('prob_mask', 'rescaled_mask', 'binarized_mask'), fixed_mask[0, 0])
            anatomy_logits = reinforce_logp = reinforce_entropy = None
        else:
            mask, data_to_vis_sampler, anatomy_logits, reinforce_logp, reinforce_entropy = self.sampler(
                sampler_kspace, old_mask, budget, uncertainty_map, valid_mask,
                recon_complex, gfactor=prev_gfactor, anatomy_labels=anatomy_labels, context=context,
                threshold=threshold, details=details,
            )
        if self.zero_unacquired and valid_mask is not None:
            mask = mask * valid_mask
        if self.store_visuals:
            self.data_for_vis[step].update(data_to_vis_sampler)
        if self.step_recorder is not None:
            self.step_recorder.add(step, old_mask, mask, details)
        return {'mask': mask, 'anatomy_logits': anatomy_logits, 'reinforce_logp': reinforce_logp,
                'reinforce_entropy': reinforce_entropy, 'details': details}

    def _reconstruct(self, full_kspace, mask, target, step, sense_maps, sense_mask, uncertainty_map=None,
                     anatomy_labels=None, kspace_support=None):
        """Reconstruct from the k-space `mask` acquires, storing the visuals as step `step`'s.

        Returns:
            Dict with keys: output, zero_recon, uncertainty_map, gfactor
        """
        recon, zero_filled_complex, zero_filled_mag = self._run_reconstructor(
            full_kspace, mask, sense_maps, anatomy_labels, kspace_support,
            zero_filled=self.store_visuals or not self.training)

        # Compute g-factor map for SENSE-based reconstructors (detached — not differentiable)
        gfactor = None
        if self.with_gfactor and self.recon_type in ('cg_sense', 'lls'):
            with torch.no_grad():
                gfactor = compute_gfactor(sense_maps, mask, num_replicas=self.gfactor_replicas)

        if self.store_visuals:
            self._store_visuals(step, target, full_kspace, zero_filled_complex, zero_filled_mag, recon, uncertainty_map,
                                sense_mask, gfactor)

        return {'output': recon, 'zero_recon': zero_filled_mag, 'uncertainty_map': uncertainty_map, 'gfactor': gfactor}

    def _run_reconstructor(self, full_kspace, mask, sense_maps, anatomy_labels=None, kspace_support=None,
                           zero_filled=False, reconstructor=None):
        """(complex image [B, 1, H, W], zero-filled SENSE image or None, its magnitude or None) from the k-space `mask`
        acquires. The zero-filled image is formed when `zero_filled`, and always for an image-domain reconstructor.
        `reconstructor` defaults to the trained one."""
        reconstructor = self.reconstructor if reconstructor is None else reconstructor
        masked_kspace = torch.view_as_real(mask * torch.view_as_complex(full_kspace))

        # In `straight_through` mode the mask is differentiable, so gradients must
        # flow through the reconstructor's *operations* to reach it, even if the
        # reconstructor's own weights are frozen. In `reinforce`/`random` mode the
        # mask draw is non-differentiable (credit flows through the REINFORCE
        # log-prob term instead), so tracking this forward pass is only needed if
        # the reconstructor's own weights are still being trained -- skipping it
        # otherwise avoids building a multi-cascade autograd graph nothing uses.
        recon_needs_grad = torch.is_grad_enabled() and (
            self.mask_train_mode == 'straight_through'
            or any(p.requires_grad for p in reconstructor.parameters())
        )
        with torch.set_grad_enabled(recon_needs_grad):
            kspace_recon = self.recon_type in ('lls', 'cg_sense', 'varnet', 'sdum')
            # Zero-filled reconstruction (coil-combined). In training it reaches no loss term, only the logged baseline.
            zero_filled_complex = zero_filled_mag = None
            if not kspace_recon or zero_filled:
                zero_filled_complex = (torch_ifft2(torch.view_as_complex(masked_kspace)) * sense_maps.conj()).sum(dim=1, keepdim=True)
                zero_filled_mag = zero_filled_complex.abs()

            # Full reconstruction
            if kspace_recon:
                masked_kspace_complex = torch.view_as_complex(masked_kspace.contiguous())
                recon_kwargs = {'anatomy_labels': anatomy_labels} \
                    if self.condition_reconstructor or getattr(reconstructor, 'iters_by_label', None) else {}
                if getattr(reconstructor, 'dc_mode', None) == 'multiply':
                    recon_kwargs['full_kspace'] = torch.view_as_complex(full_kspace.contiguous())
                if kspace_support is not None and self.recon_type == 'varnet':
                    recon_kwargs['kspace_support'] = kspace_support
                recon_mag = reconstructor(masked_kspace_complex, sense_maps, mask, **recon_kwargs)
                recon = recon_mag.to(torch.complex64)
            else:
                zero_filled_2ch = torch.view_as_real(zero_filled_complex)
                zero_filled_2ch = einops.rearrange(zero_filled_2ch, 'b c h w two -> b (c two) h w')
                recon_2ch = reconstructor(zero_filled_2ch)
                recon = torch.view_as_complex(
                    einops.rearrange(recon_2ch, 'b (c two) h w -> b c h w two', two=2).contiguous()
                )
        return recon, zero_filled_complex, zero_filled_mag

    def _output_image(self, recon, sense_maps, output_extent, valid_mask):
        """The magnitude image [B, 1, H, W] the model outputs for its complex reconstruction `recon`."""
        image = self._coil_rss(recon, sense_maps, output_extent)
        if self.zero_unacquired:
            # Mask the output magnitude image in k-space where the scan acquired nothing. The complex network image carries a
            # phase that moves most of its spectrum out of the acquired columns, so only the magnitude image is masked.
            image = torch_ifft2(torch_fft2(image.to(torch.complex64)) * valid_mask).abs()
        return image

    def sense_maps_for(self, full_kspace):
        """(sensitivity maps [B, C, H, W], body mask [B, 1, H, W]) the forward estimates for `full_kspace`.

        The maps depend only on the k-space, never the mask, so their forward is tracked only when the sensitivity
        network's own weights are training, and never inside a caller's no_grad (e.g. validation).
        """
        sens_needs_grad = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.sensetivity_net.parameters()
        )
        with torch.set_grad_enabled(sens_needs_grad):
            return self.sensetivity_net(torch.view_as_complex(full_kspace.contiguous()),
                                        mask_method=self.sense_mask_method)

    def reconstruct_from_mask(self, full_kspace, mask, sense_maps, anatomy_labels=None, kspace_support=None,
                              reconstructor=None):
        """The output image [B, 1, H, W] for the complete mask `mask` [B, 1, H, W] (centre included), by the forward's
        own final reconstruction: a forward whose steps end in `mask` outputs exactly this. Any num_step.
        `reconstructor` defaults to the trained one."""
        valid_mask = self._valid_kspace_mask(full_kspace)
        if self.zero_unacquired:
            mask = mask * valid_mask
        output_extent = self._kspace_extent(full_kspace) if kspace_support is None else kspace_support
        recon, _, _ = self._run_reconstructor(full_kspace.contiguous(), mask, sense_maps, anatomy_labels,
                                              self._recon_support(kspace_support, output_extent),
                                              reconstructor=reconstructor)
        return self._output_image(recon, sense_maps, output_extent, valid_mask)

    @torch.no_grad()
    def _input_reconstruction(self, full_kspace, mask, image_peak, anatomy_labels=None):
        """The recon input [B, H, W]: the frozen copies' magnitude image of the k-space `mask` acquires, over the ACS
        image peak [B], as the data channels are scaled."""
        maps, _ = self.input_sensetivity_net(torch.view_as_complex(full_kspace.contiguous()),
                                             mask_method=self.sense_mask_method)
        image = self.reconstruct_from_mask(full_kspace, mask, maps, anatomy_labels, reconstructor=self.input_reconstructor)
        return (image[:, 0] / image_peak.view(-1, 1, 1)).float()

    def train(self, mode=True):
        """The frozen input copies stay in eval mode."""
        super().train(mode)
        for frozen in (self.input_reconstructor, self.input_sensetivity_net):
            if frozen is not None:
                frozen.eval()
        return self

    @contextlib.contextmanager
    def _quietly(self):
        """No visuals stored and no steps recorded, for the forwards a diagnostic runs besides the model's own."""
        stored = self.store_visuals, self.step_recorder
        self.store_visuals, self.step_recorder = False, None
        try:
            yield
        finally:
            self.store_visuals, self.step_recorder = stored

    @torch.no_grad()
    def sample_trajectory(self, full_kspace, anatomy_labels=None, thresholds=None, sense_maps=None):
        """The steps' draws for `full_kspace`, with no reconstruction: (masks after each step [M_1, ..., M_K], each
        [B, 1, H, W] with the centre included, and each step's details). `thresholds` as in forward."""
        if sense_maps is None and (self.sampler.reads_input or self.recon_steps):
            sense_maps, _ = self.sense_maps_for(full_kspace)
        old_mask, valid_mask = self._initial_masks(full_kspace)
        step_budget = self.mask_budget(old_mask, valid_mask) / self.num_step
        with self._quietly():
            steps = self._sample_steps(full_kspace, old_mask, valid_mask, sense_maps, step_budget, anatomy_labels,
                                       thresholds=thresholds, uncertainty_map=torch.zeros_like(old_mask))
        return steps['masks'], steps['details']

    def latest_recon_step(self, step):
        """The last in-loop reconstruction step at or before `step`, or None (always None unless sampler_input='seqmri')."""
        return max((s for s in self.recon_steps if s <= step), default=None)

    def _in_loop_prediction(self, full_kspace, mask, sense_maps, extent, anatomy_labels=None):
        """Coil k-space [B, C, H, W] predicted by the trained reconstructor from the k-space `mask` acquires: FFT(S x),
        zero outside the acquired `extent`. For the 'seqmri' input `mask` arrives detached, so only the reconstructor's
        weights (and the maps') get gradient through it; seqmri_2d passes its draws' straight-through gradient too."""
        recon, _, _ = self._run_reconstructor(full_kspace, mask, sense_maps, anatomy_labels,
                                              self._recon_support(None, extent))
        return torch_fft2(recon * sense_maps) * extent

    @torch.no_grad()
    def counterfactual_step(self, full_kspace, mask, step, data_kspace, threshold=None, anatomy_labels=None,
                            recon_mask=None, data_sense_maps=None):
        """Step `step`'s draw from the mask `mask` [B, 1, H, W] when the sampler reads `data_kspace` (another slice's coil
        k-space, [B, C, H, W, 2], scaled by its own ACS) in place of the slice's own data. The mask, header, acquired
        region, budget and `threshold` stay the slice's own, so data_kspace = full_kspace gives the step itself.

        A 'full' sampler reads all of `data_kspace`. A 'seqmri' sampler past its first in-loop reconstruction also needs
        `recon_mask`, the slice's own mask at `latest_recon_step(step)`: the reconstruction is redone from `data_kspace`
        on that mask, with `data_kspace`'s own virtual coils and sensitivity maps (`data_sense_maps`, else estimated as
        `sample_trajectory` estimates them).

        Returns the step's details plus `new`, the locations it adds [B, 1, H, W]. seq_adapt_2d only.
        """
        if not self.sampler.reads_context:
            raise ValueError(f"a counterfactual data swap needs a sampler that reads a StepContext, not {self.sampler_type}")
        old_mask, valid_mask = self._initial_masks(full_kspace)
        budget = self.mask_budget(old_mask, valid_mask) / self.num_step
        extent, coil_fraction, _, _ = self._step_header(full_kspace, old_mask)
        context = StepContext(step, self.num_step, extent, coil_fraction)
        if not self.sampler.data_blind:
            data = torch.view_as_complex(data_kspace.contiguous())
            measured = data_kspace if self.sampler_input == 'full' else data_kspace * mask.unsqueeze(-1)
            context.measured = torch.view_as_complex(measured.contiguous())
            context.calibration_peaks = calibration_peaks(data * old_mask)
            if self.sampler.reads_reconstruction:
                context.reconstruction = self._input_reconstruction(data_kspace, mask, context.calibration_peaks[0],
                                                                    anatomy_labels)
            if self.sampler.complex_coils:
                context.compression = acs_compression(data * old_mask, self.sampler.complex_coils)
                if self.latest_recon_step(step) is not None:
                    if recon_mask is None:
                        raise ValueError(f"step {step} reads the in-loop reconstruction of step "
                                         f"{self.latest_recon_step(step)}; pass that step's own mask as recon_mask")
                    maps = self.sense_maps_for(data_kspace)[0] if data_sense_maps is None else data_sense_maps
                    context.predicted = self._in_loop_prediction(data_kspace, recon_mask, maps, extent, anatomy_labels)
        with self._quietly():
            sampled = self._sample(full_kspace, full_kspace * mask.unsqueeze(-1), mask, step, None,
                                   valid_mask=valid_mask, anatomy_labels=anatomy_labels, budget=budget,
                                   context=context, threshold=threshold)
        return {**sampled['details'], 'new': sampled['mask'] - mask}

    @torch.no_grad()
    def _store_sampling_visuals(self, step, target, full_kspace, mask, sense_maps, sense_mask, uncertainty_map):
        """Visuals of a step that is not reconstructed: its zero-filled image, and a blank reconstruction panel."""
        zero_filled_complex = (torch_ifft2(mask * torch.view_as_complex(full_kspace.contiguous()))
                               * sense_maps.conj()).sum(dim=1, keepdim=True)
        self._store_visuals(step, target, full_kspace, zero_filled_complex, zero_filled_complex.abs(), None,
                            uncertainty_map, sense_mask, None)

    def _store_visuals(self, step, target, full_kspace, zero_filled_complex, zero_filled_mag, recon, uncertainty_map,
                       sense_mask, gfactor):
        self.data_for_vis[step].update({
            'input': target[0].cpu().detach().numpy() if target.dim() == 3 else target[0, 0].cpu().detach().numpy(),
            'kspace': torch.view_as_complex(full_kspace[0, :, :, :]).cpu().detach().numpy(),
            'masked_kspace': zero_filled_complex[0].cpu().detach().numpy(),
            'zero_filled_recon': zero_filled_mag[0, 0, :, :].cpu().detach().numpy(),
            'recon': recon[0, 0, :, :].abs().cpu().detach().numpy() if recon is not None else np.zeros(self.shape),
            'uncertainty_map': uncertainty_map[0, 0].cpu().detach().numpy() if uncertainty_map is not None else np.zeros(self.shape),
            'sense_mask': sense_mask[0, 0, :, :].cpu().detach().numpy() if sense_mask is not None else np.zeros(self.shape),
            'gfactor': gfactor[0, 0].cpu().detach().numpy() if gfactor is not None else None,
        })

    @torch.no_grad()
    def reconstruct_acquired(self, kspace, sense_maps, kspace_support=None):
        """Magnitude image [B, H, W] reconstructed from every nonzero k-space point: the estimate of the fully sampled image.

        For data whose k-space is already scanner-undersampled, this is the best image the acquired set supports.
        """
        k = torch.view_as_complex(kspace.contiguous())
        mask = self._valid_kspace_mask(kspace)
        output_extent = self._kspace_extent(kspace) if kspace_support is None else kspace_support
        recon_support = self._recon_support(kspace_support, output_extent)
        kwargs = {'kspace_support': recon_support} if recon_support is not None and self.recon_type == 'varnet' else {}
        image = self.reconstructor(k, sense_maps, mask, **kwargs)
        return self._coil_rss(image, sense_maps, output_extent)[:, 0].float()

    def _recon_support(self, kspace_support, output_extent):
        """The k-space region a VarNet's regularizer may write to: the data's support, else the acquired extent when
        `restrict_recon_to_extent`, else None (anywhere)."""
        if kspace_support is not None or not self.restrict_recon_to_extent:
            return kspace_support
        return output_extent

    def _init_mask(self, data):
        """Initialize mask with pre-selected center k-space region.

        For 2D: selects center NxN region (shifted to corners in non-fftshifted space).
        For line-constrained: selects center vertical lines.

        Args:
            data: Input kspace tensor, shape [B, C, H, W, 2]

        Returns:
            Mask tensor, shape [B, 1, H, W] (fft-shifted)
        """
        a = torch.zeros([data.shape[0], 1, self.shape[0], self.shape[1]], device=data.device)

        if self.preselect:
            n = self.preselect_num_one_side
            if not self.line_constrained:
                a[:, 0, :n, :n] = 1
                a[:, 0, :n, -n:] = 1
                a[:, 0, -n:, :n] = 1
                a[:, 0, -n:, -n:] = 1
            else:
                a[:, :, :, :n] = 1
                a[:, :, :, -n:] = 1

        return torch.fft.fftshift(a, dim=(-1, -2))

    def _initial_masks(self, full_kspace):
        """(pre-selected centre mask, valid k-space mask), both [B, 1, H, W]."""
        old_mask = self._init_mask(full_kspace)
        valid_mask = self._valid_kspace_mask(full_kspace)
        if self.acquired_only:
            # Pre-selected points the scanner never acquired would count as measured zeros in data consistency.
            old_mask = old_mask * valid_mask
        return old_mask, valid_mask

    def propose_mask(self, full_kspace, anatomy_labels=None):
        """The single-step sampler's mask for `full_kspace`, with no reconstruction.

        Differentiable in the sampler's weights, so it also serves training the sampler on labels.

        Returns:
            Dict with `old_mask` (the pre-selected centre), `valid_mask`, `unscaled` and `rescaled`
            probabilities, and the binary `mask` (centre included), all [B, 1, H, W].
        """
        if self.num_step != 1:
            raise ValueError(f"propose_mask is the single-step mask, got num_step={self.num_step}")
        with torch.no_grad():
            sense_maps, _ = self.sensetivity_net(torch.view_as_complex(full_kspace.contiguous()),
                                                 mask_method=self.sense_mask_method)
        old_mask, valid_mask = self._initial_masks(full_kspace)
        observed_kspace = full_kspace * old_mask.unsqueeze(-1)
        combined = self._sampler_image(torch.view_as_complex(observed_kspace.contiguous()), sense_maps)
        header = self._step_header(full_kspace, old_mask) if self.sampler.reads_context else None
        unscaled, _, _ = self.sampler.unscaled_probabilities(
            torch.view_as_real(torch_fft2(combined)), old_mask, anatomy_labels=anatomy_labels,
            context=self._step_context(header, observed_kspace, 0, full_kspace, old_mask, anatomy_labels))
        rescaled, mask = self.sampler.select_locations(unscaled, old_mask, self.mask_budget(old_mask, valid_mask),
                                                       valid_mask)
        return {'old_mask': old_mask, 'valid_mask': valid_mask, 'unscaled': unscaled, 'rescaled': rescaled, 'mask': mask}

    def forward(self, target, full_kspace, idx=None, anatomy_labels=None, sense_maps=None, kspace_support=None, fixed_mask=None,  # noqa: ARG002
                thresholds=None):
        """Run full sequential sampling pipeline.

        Args:
            target: Ground truth image, shape [B, 1, H, W]
            full_kspace: Multicoil kspace, shape [B, C, H, W, 2]
            anatomy_labels: Optional long tensor [B] of anatomy indices
            sense_maps: Optional precomputed sensitivity maps [B, C, H, W] complex, used instead of
                the sensitivity network (for data with a separate calibration scan)
            kspace_support: Optional [B, 1, H, W] k-space region the VarNet's regularizer may write to and the output
                keeps. None leaves the regularizer free (trained VarNets use the zero-filled region between cascades)
                and keeps the extent of `full_kspace` (`_kspace_extent`) in the output
            fixed_mask: Optional [B, 1, H, W] complete mask (pre-selected centre included) that replaces
                the sampler's. Single-step only; it stays differentiable, so a caller can optimise it.
            thresholds: Optional sequence of num_step [B, 1, H, W] U(0, 1) maps, one per step, that replace the
                Bernoulli draws' fresh noise (activemri.utils.multistep.step_thresholds), so runs share their draws

        Returns:
            Dict with final reconstruction, intermediate results, masks, and sense maps.
        """
        if sense_maps is None:
            sense_maps, sense_mask = self.sense_maps_for(full_kspace)
        else:
            sense_mask = (sense_maps.abs().pow(2).sum(dim=1, keepdim=True) > 0.5).to(full_kspace.dtype)
        old_mask, valid_mask = self._initial_masks(full_kspace)
        output_extent = self._kspace_extent(full_kspace) if kspace_support is None else kspace_support
        recon_support = self._recon_support(kspace_support, output_extent)
        step_budget = self.mask_budget(old_mask, valid_mask) / self.num_step
        uncertainty_map = torch.zeros_like(old_mask)

        if fixed_mask is not None and self.num_step != 1:
            raise ValueError(f"a fixed_mask is the whole mask, so it needs num_step=1, got {self.num_step}")
        if fixed_mask is None and self.mask_policy is not None:
            fixed_mask = self.mask_policy(full_kspace)

        steps = self._sample_steps(full_kspace, old_mask, valid_mask, sense_maps, step_budget, anatomy_labels,
                                   fixed_mask, thresholds, uncertainty_map,
                                   visuals=(target, sense_mask) if self.store_visuals else None)
        reconstructed = self._reconstruct(full_kspace, steps['mask'], target, self.num_step - 1, sense_maps, sense_mask,
                                          uncertainty_map, anatomy_labels, recon_support)
        final_complex = reconstructed['output']
        final_recon = self._output_image(final_complex, sense_maps, output_extent, valid_mask)

        return {
            'output': final_recon,
            'recon_complex': final_complex,
            'intermediate_recons': [final_complex.detach()],
            'mask': steps['mask'].detach(),
            'masks': steps['masks'],
            'zero_filled_recon': [reconstructed['zero_recon']],
            'uncertainty_maps': [uncertainty_map] * self.num_step,
            'sense_maps': sense_maps,
            'reinforce_logp': steps['reinforce_logp'],
            'reinforce_entropy': steps['reinforce_entropy'],
            'anatomy_logits': steps['anatomy_logits'],
        }

    def _sample_steps(self, full_kspace, old_mask, valid_mask, sense_maps, step_budget, anatomy_labels=None,
                      fixed_mask=None, thresholds=None, uncertainty_map=None, visuals=None):
        """Every step's draw, from the pre-selected centre `old_mask` to the final mask.

        Each step reads only the measured k-space y * M_i and sees M_i detached, as its input and to find the locations
        still available. The union keeps every step's straight-through increment, so a loss on the final mask reaches
        each step's choices. seqmri_2d takes SeqMRI's routes instead: the union is detached between steps and each in-loop
        reconstruction is not, so an earlier step's draws reach the loss only through the reconstructions later steps
        read, and only the last step's reach it through the final mask. `visuals` (target, body mask) stores the panels
        of the steps that are not reconstructed.

        Returns a dict: `mask` (the final mask, differentiable), `masks` (detached, after each step), `details` (each
        step's), the sampler's mean anatomy logits and its REINFORCE terms summed over the steps.
        """
        header = self._step_header(full_kspace, old_mask) if self.sampler.reads_context else None
        mask = old_mask
        masks, details, anatomy_logits = [], [], []
        reinforce_logp = None  # [B], sum of every step's Plackett-Luce log-prob
        reinforce_entropy = None  # [B], sum of every step's draw entropy
        predicted = None  # 'seqmri': the coil k-space predicted by the latest in-loop reconstruction
        for i in range(self.num_step):
            observed = full_kspace * mask.detach().unsqueeze(-1)
            if i in self.recon_steps:
                predicted = self._in_loop_prediction(full_kspace, mask if self.seqmri_routes else mask.detach(),
                                                     sense_maps, header[0], anatomy_labels)
            sampled = self._sample(
                full_kspace, observed, mask.detach(), i, sense_maps, uncertainty_map, valid_mask,
                anatomy_labels=anatomy_labels, fixed_mask=fixed_mask, budget=step_budget,
                context=self._step_context(header, observed, i, full_kspace, mask.detach(), anatomy_labels, predicted),
                threshold=None if thresholds is None else thresholds[i])
            mask = sampled['mask'] if i == 0 or self.seqmri_routes else mask + (sampled['mask'] - mask.detach())
            masks.append(mask.detach())
            details.append(sampled['details'])
            if sampled['anatomy_logits'] is not None:
                anatomy_logits.append(sampled['anatomy_logits'])
            if sampled['reinforce_logp'] is not None:
                step_logp = sampled['reinforce_logp']
                reinforce_logp = step_logp if reinforce_logp is None else reinforce_logp + step_logp
            if sampled['reinforce_entropy'] is not None:
                step_entropy = sampled['reinforce_entropy']
                reinforce_entropy = step_entropy if reinforce_entropy is None else reinforce_entropy + step_entropy
            if visuals is not None and i + 1 < self.num_step:
                self._store_sampling_visuals(i, visuals[0], full_kspace, mask.detach(), sense_maps, visuals[1],
                                             uncertainty_map)
        return {'mask': mask, 'masks': masks, 'details': details,
                'anatomy_logits': torch.stack(anatomy_logits).mean(dim=0) if anatomy_logits else None,
                'reinforce_logp': reinforce_logp, 'reinforce_entropy': reinforce_entropy}

    def _step_header(self, full_kspace, calibration_mask):
        """What every step of a `reads_context` sampler reads besides its mask and data: (acquired extent [B, 1, H, W],
        active coil fraction [B], ACS peaks scaling the data, None for a data-blind sampler, and the 'seqmri' virtual-coil
        weights [B, C, V] from the ACS, else None)."""
        coil_kspace = torch.view_as_complex(full_kspace.contiguous())
        active_coils = (coil_kspace.abs().flatten(2).sum(dim=-1) > 0).sum(dim=1)
        coil_fraction = active_coils.to(full_kspace.dtype) / coil_kspace.shape[1]
        peaks = None if self.sampler.data_blind else calibration_peaks(coil_kspace * calibration_mask)
        compression = (acs_compression(coil_kspace * calibration_mask, self.sampler.complex_coils)
                       if self.sampler.complex_coils else None)
        return self._kspace_extent(full_kspace), coil_fraction, peaks, compression

    def _step_context(self, header, observed_kspace, step, full_kspace=None, mask=None, anatomy_labels=None,
                      predicted=None):
        """The StepContext of step `step` (None for a sampler that reads none); a data-blind one gets no data. A sampler
        with a recon input also gets the frozen reconstruction of the k-space `mask` [B, 1, H, W] acquires; a 'full'
        sampler reads all of `full_kspace`; a 'seqmri' one also gets the latest in-loop prediction `predicted`."""
        if header is None:
            return None
        extent, coil_fraction, peaks, compression = header
        if self.sampler.data_blind:
            measured = None
        elif self.sampler_input == 'full':
            if full_kspace is None:
                raise ValueError("a clairvoyant sampler's context needs the slice's fully sampled k-space")
            measured = torch.view_as_complex(full_kspace.contiguous())
        else:
            measured = torch.view_as_complex(observed_kspace.contiguous())
        reconstruction = None
        if self.sampler.reads_reconstruction:
            if full_kspace is None or mask is None:
                raise ValueError("a recon-input sampler's context needs the slice's k-space and the mask so far")
            reconstruction = self._input_reconstruction(full_kspace, mask, peaks[0], anatomy_labels)
        return StepContext(step, self.num_step, extent, coil_fraction, measured, peaks, reconstruction, predicted,
                           compression)

    def loss(self, pred_dict, target_dict, meta, loss_type):
        """Compute training loss.

        Args:
            pred_dict: Model outputs (output, zero_filled_recon, uncertainty_maps, sense_maps)
            target_dict: Ground truth (target, label, kspace)
            meta: Loss weights (recon_weight, kspace_weight, uncertainty_weight)
            loss_type: One of 'l1', 'ssim', 'nmse', 'psnr', 'xentropy'

        Returns:
            (loss, log_dict) tuple
        """
        target = target_dict['target']
        pred = pred_dict['output']
        if pred.dim() != target.dim():
            target = target.unsqueeze(1)
        zero_filled = pred_dict['zero_filled_recon'][-1]
        gt_kspace = target_dict['kspace']

        def zero_filled_baseline(error):
            """`error` of the zero-filled image, NaN when the forward did not form one."""
            return pred.new_tensor(float('nan')) if zero_filled is None else error(zero_filled)

        self_supervised = 'holdout_mask' in target_dict
        per_example_ssdu = None

        # Uncertainty loss (NLL)
        nll_loss = torch.tensor(0.0, device=pred.device)
        if self.with_uncertainty:
            for step_output, step_uncertainty in zip(pred_dict['output'], pred_dict['uncertainty_maps']):
                nll_loss = nll_loss + torch.mean(
                    compute_gaussian_nll_loss(step_output, target, step_uncertainty)
                )

        # Reconstruction loss
        if self_supervised:
            per_example_ssdu = self.ssdu_loss(pred_dict, target_dict)
            reconstruction_loss = per_example_ssdu.mean()
            zero_loss = torch.tensor(0.0, device=pred.device)
        elif loss_type == 'l1':
            reconstruction_loss = F.l1_loss(pred, target, size_average=True)
            zero_loss = zero_filled_baseline(lambda image: F.l1_loss(image, target, size_average=True))
        elif loss_type == 'ssim':
            reconstruction_loss = -torch.mean(compute_ssim_torch(pred, target))
            zero_loss = zero_filled_baseline(lambda image: -torch.mean(compute_ssim_torch(image, target)))
        elif loss_type == 'nmse':
            reconstruction_loss = torch.norm(pred - target) ** 2 / torch.norm(target) ** 2
            zero_loss = zero_filled_baseline(lambda image: torch.norm(image - target) ** 2 / torch.norm(target) ** 2)
        elif loss_type == 'psnr':
            reconstruction_loss = -torch.mean(compute_psnr_torch(pred, target))
            zero_loss = zero_filled_baseline(lambda image: -torch.mean(compute_psnr_torch(image, target)))
        elif loss_type == 'xentropy':
            label = target_dict['label']
            reconstruction_loss = nn.CrossEntropyLoss()(pred, label)
            zero_loss = torch.tensor(0.0, device=pred.device)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        # K-space loss
        if meta['kspace_weight'] > 0 and not self_supervised:
            image = torch.cat([pred, torch.zeros_like(pred)], dim=1).permute(0, 2, 3, 1)
            image_complex = torch.view_as_complex(image.contiguous())
            pred_kspace_c = torch.view_as_real(torch.fft.fft2(torch.fft.ifftshift(image_complex, dim=(1, 2))))
            pred_kspace_c = torch.norm(pred_kspace_c, dim=-1, keepdim=True).permute(0, 3, 1, 2)
            gt_kspace_c = torch.norm(gt_kspace, dim=-1, keepdim=True)
            if gt_kspace_c.dim() == 5:  # multicoil [B, C, H, W, 1]
                gt_kspace_c = gt_kspace_c.mean(dim=1)
            gt_kspace_c = gt_kspace_c.permute(0, 3, 1, 2)
            kspace_loss = -torch.mean(compute_ssim_torch(pred_kspace_c, gt_kspace_c))
        else:
            kspace_loss = torch.tensor(0.0, device=pred.device)

        loss = reconstruction_loss * meta['recon_weight'] + kspace_loss * meta['kspace_weight']

        # Tikhonov regularization on sensitivity maps
        sense_maps = pred_dict.get('sense_maps')
        if sense_maps is not None and self.sense_tikhonov_weight > 0:
            sm = sense_maps.abs()
            grad_h = sm[:, :, 1:, :] - sm[:, :, :-1, :]
            grad_w = sm[:, :, :, 1:] - sm[:, :, :, :-1]
            tikhonov_loss = grad_h.pow(2).mean() + grad_w.pow(2).mean()
            loss = loss + self.sense_tikhonov_weight * tikhonov_loss
        else:
            tikhonov_loss = torch.tensor(0.0, device=pred.device)

        anatomy_loss = torch.tensor(0.0, device=pred.device)
        anatomy_logits = pred_dict.get('anatomy_logits')
        anatomy_labels = target_dict.get('anatomy_labels')
        if anatomy_logits is not None and anatomy_labels is not None:
            anatomy_loss = nn.CrossEntropyLoss()(anatomy_logits, anatomy_labels)
            anatomy_weight = meta.get('anatomy_weight', 0.0)
            loss = loss + anatomy_weight * anatomy_loss

        # REINFORCE term for mask_train_mode='reinforce': the mask itself carries
        # no gradient (sampling it is non-differentiable, unlike the straight-through
        # path), so this is the only path any gradient reaches self.sampler.mask_net
        # through. Reward is a clean per-example NMSE independent of loss_type, so the
        # policy always optimizes the same quantity regardless of what the reconstructor
        # is trained against. Each batch item is one Monte Carlo draw of the (input-
        # independent) LOUPE mask -- exactly the role a training minibatch already
        # plays, so this needs no extra forward passes -- and the mean reward over the
        # draws sharing a slice is the REINFORCE baseline (the whole batch by default;
        # `reinforce_group_size` splits it into several slices' worth of groups, each
        # with its own baseline), following scripts/loupe_vs_greedy_oracle.py and
        # scripts/reinforce2d_vs_greedy_loupe.py, where the same estimator was
        # validated against an oracle reconstructor.
        reinforce_loss = torch.tensor(0.0, device=pred.device)
        reinforce_reward = torch.tensor(0.0, device=pred.device)
        reinforce_logp = pred_dict.get('reinforce_logp')
        pg_grad_norm = entropy_grad_norm = None
        if reinforce_logp is not None:
            if pred.shape[0] < 2:
                raise RuntimeError(
                    "mask_train_mode='reinforce' needs batch_size >= 2: each batch item is "
                    "one Monte Carlo draw of the mask, and the REINFORCE baseline is their "
                    "mean reward -- with one item there is nothing to compare it to.")
            if self_supervised:
                per_example_error = per_example_ssdu
            else:
                per_example_error = (
                    (pred - target).pow(2).flatten(1).sum(dim=1)
                    / target.pow(2).flatten(1).sum(dim=1).clamp(min=1e-12)
                )
            reward = -per_example_error.detach()
            group = reinforce_group_size(reward.shape[0], self.reinforce_group_size)
            if group < 2 or reward.shape[0] % group:
                raise RuntimeError(
                    f"batch of {reward.shape[0]} must be a multiple of reinforce_group_size={group} "
                    "and each group needs >= 2 draws to have a baseline")
            grouped_reward = reward.view(-1, group)
            advantage = grouped_reward - grouped_reward.mean(dim=1, keepdim=True)
            if self.reinforce_normalize_advantages:
                advantage = (advantage - advantage.mean(dim=1, keepdim=True)) / (
                    advantage.std(dim=1, keepdim=True) + 1e-8)
            reinforce_loss = -(advantage.reshape(-1) * reinforce_logp).mean()
            reinforce_reward = reward.mean()
            loss = loss + self.reinforce_weight * reinforce_loss
            pg_grad_norm = self._policy_grad_norm(reinforce_loss)

            # Entropy bonus: encourages the policy to keep spreading probability
            # mass across positions instead of collapsing to one pattern before
            # it has sampled enough to know that pattern is actually good.
            reinforce_entropy = pred_dict.get('reinforce_entropy')
            if reinforce_entropy is not None:
                entropy_grad_norm = self._policy_grad_norm(reinforce_entropy.mean())
                if self.reinforce_entropy_weight > 0:
                    loss = loss - self.reinforce_entropy_weight * reinforce_entropy.mean()

        log_dict = {
            'Total Loss': loss.item(),
            'Zero Filled Loss': zero_loss.item(),
            'K-space Loss': kspace_loss.item(),
            'Recon loss': reconstruction_loss.item(),
            'Tikhonov Loss': tikhonov_loss.item(),
            'Anatomy Loss': anatomy_loss.item(),
            'Reinforce Loss': reinforce_loss.item(),
            'Reinforce Reward': reinforce_reward.item(),
        }
        if reinforce_logp is not None and reinforce_entropy is not None:
            log_dict['Reinforce Entropy'] = reinforce_entropy.mean().item()
        if pg_grad_norm is not None:
            # Norms on the mask logits alone, of each term as it enters the update. The
            # entropy norm is per unit weight: multiply by reinforce_entropy_weight for
            # what it actually contributes, or divide the PG norm by this to read off
            # the weight at which the two terms are comparable.
            log_dict['Reinforce PG Grad Norm'] = pg_grad_norm * self.reinforce_weight
            if entropy_grad_norm is not None:
                log_dict['Reinforce Entropy Grad Norm'] = entropy_grad_norm

        if self.with_uncertainty:
            loss = loss + nll_loss * meta['uncertainty_weight']
            log_dict['Uncertainty Loss'] = nll_loss.item()

        return loss, log_dict

    @staticmethod
    def ssdu_loss(pred_dict, target_dict):
        """Per-example normalized L1 + L2 error of the predicted k-space on the held-out set Lambda (Yaman et al., SSDU).

        The hold-out k-space was never an input, so this is a loss on measured data the model has not
        seen. It also scores the sampler's mask: a mask whose pool points predict Lambda better scores lower.
        """
        predicted = torch_fft2(pred_dict['recon_complex'] * pred_dict['sense_maps'])
        measured = torch.view_as_complex(target_dict['holdout_kspace'].contiguous())
        error = ((predicted - measured) * target_dict['holdout_mask']).abs().flatten(1)
        reference = measured.abs().flatten(1)
        relative_l2 = error.pow(2).sum(dim=1).sqrt() / reference.pow(2).sum(dim=1).sqrt().clamp(min=1e-12)
        relative_l1 = error.sum(dim=1) / reference.sum(dim=1).clamp(min=1e-12)
        return relative_l2 + relative_l1

    def _policy_grad_norm(self, term):
        """Gradient norm of `term` w.r.t. the mask logits, or None when nothing is being differentiated.

        The REINFORCE log-prob and entropy graphs run only through the mask
        logits (the drawn mask is detached), so this is a cheap extra backward.
        """
        if not term.requires_grad:
            return None
        params = [p for p in self.sampler.mask_net.parameters() if p.requires_grad]
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None)).item()

    def visualize_and_save(self, options, epoch, data_for_vis_name):
        """Save visualization plots for each sampling step."""
        for step in range(self.num_step):
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18, 6])
            cmap = "viridis" if options.resolution[1] == 32 else "gray"

            # Top row: images
            ax[0, 0].imshow(self.data_for_vis[step]['input'], cmap=cmap)
            ax[0, 1].imshow(np.log(np.abs(self.data_for_vis[step]['kspace'][0])))
            masked_k = self.data_for_vis[step]['masked_kspace'].copy()
            masked_k[masked_k == 0] = 1
            ax[0, 2].imshow(np.log(np.abs(masked_k[0])))
            ax[0, 3].imshow(self.data_for_vis[step]['zero_filled_recon'], cmap=cmap)
            ax[0, 4].imshow(self.data_for_vis[step]['recon'], cmap=cmap)

            # Bottom row: masks and uncertainty
            ax[1, 0].imshow(self.data_for_vis[step]['prob_mask'].cpu().detach(), aspect='auto')
            ax[1, 1].imshow(self.data_for_vis[step]['rescaled_mask'].cpu().detach(), aspect='auto')
            ax[1, 2].imshow(self.data_for_vis[step]['binarized_mask'].cpu().detach(), aspect='auto')
            ax[1, 3].imshow(self.data_for_vis[step]['sense_mask'], cmap='gray', vmin=0, vmax=1)
            ax[1, 4].imshow(np.exp(self.data_for_vis[step]['uncertainty_map']), cmap=cmap)

            titles_top = ['Input image', 'Log k-space', 'Undersampled log k-space', 'Zero-filled', 'Reconstruction']
            titles_bot = ['Probability mask', 'Rescaled mask', 'Binary mask', 'Sense body mask', 'Uncertainty']
            for i, t in enumerate(titles_top):
                ax[0, i].set_title(t)
            for i, t in enumerate(titles_bot):
                ax[1, i].set_title(t)

            for a in ax.flat:
                fig.colorbar(a.images[0], ax=a)

            plt.suptitle(f'Epoch [{1 + epoch}/{options.num_epochs}] Step [{step + 1}/{options.num_step}]', fontsize=20)
            os.makedirs(options.visualization_dir, exist_ok=True)

            plt.savefig(os.path.join(options.visualization_dir, f'{data_for_vis_name}_step{step}.png'))
            plt.close()
