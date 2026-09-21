"""The sequential sampling model: acquire, reconstruct, repeat.

Composes a `Sampler` with an injected reconstructor. The reconstructor is a
constructor argument, so this model is independent of which one is used.
"""
import os
import pickle

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from activemri.reconstructors.CG_Sense import compute_gfactor
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.samplers.sampler import Sampler, is_line_constrained
from activemri.utils.image_processing import fft_2d_img as torch_fft2
from activemri.utils.image_processing import ifft_2d_img as torch_ifft2
from activemri.utils.util import compute_gaussian_nll_loss, compute_psnr_torch, compute_ssim_torch


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
            uncertainty_mode='none',
            num_anatomy_classes=0,
            num_conditioning_classes=0,
            condition_reconstructor=False,
            mask_train_mode='straight_through',
            reinforce_weight=1.0,
            ):
        super().__init__()

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
        self.data_for_vis: list[dict] = [dict() for _ in range(num_step)]

        self.reconstructor = reconstructor

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
        )

        # Sensitivity estimation
        self.sensetivity_net = SensetivityModel_mc(2, 2, 8)

        # Compute per-step sampling budget
        if not self.line_constrained:
            effective_sparsity = sparsity - (preselect_num ** 2) / (shape[0] * shape[1]) if preselect else sparsity
            self.budget = effective_sparsity * shape[0] * shape[1]
        else:
            effective_sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity
            self.budget = effective_sparsity * shape[0]

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

    def step_forward(self, full_kspace, pred_kspace, old_mask, target, step,
                     sense_maps, sense_mask, uncertainty_map=None, valid_mask=None,
                     prev_gfactor=None, anatomy_labels=None):
        """Execute one sampling + reconstruction step.

        Args:
            full_kspace: Ground truth multicoil kspace, shape [B, C, H, W, 2]
            pred_kspace: Current predicted kspace, shape [B, C, H, W, 2]
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

        Returns:
            Dict with keys: output, mask, zero_recon, uncertainty_map, gfactor
        """
        budget = self.budget / self.num_step

        # Combine coils to single-coil equivalent for sampler input
        pred_single_coil_img = (torch_ifft2(torch.view_as_complex(pred_kspace)) * sense_maps).sum(dim=1)
        pred_single_coil_k = torch.view_as_real(torch_fft2(pred_single_coil_img))

        if torch.isnan(pred_single_coil_k).any():
            print(f"NaN in pred_single_coil_k at step {step}")

        # Prepare complex recon image for sampler conditioning
        recon_complex = None
        if self.sampler.with_recon:
            recon_2ch = torch.stack([pred_single_coil_img.real, pred_single_coil_img.imag], dim=1)
            recon_complex = recon_2ch.detach()

        mask, data_to_vis_sampler, anatomy_logits, reinforce_logp = self.sampler(
            pred_single_coil_k, old_mask, budget, uncertainty_map, valid_mask,
            recon_complex, gfactor=prev_gfactor, anatomy_labels=anatomy_labels,
        )
        masked_kspace = torch.view_as_real(mask * torch.view_as_complex(full_kspace))
        self.data_for_vis[step].update(data_to_vis_sampler)

        # Zero-filled reconstruction (coil-combined)
        zero_filled_complex = (torch_ifft2(torch.view_as_complex(masked_kspace)) * sense_maps.conj()).sum(dim=1, keepdim=True)
        zero_filled_mag = zero_filled_complex.abs()

        # Full reconstruction
        if self.recon_type in ('lls', 'cg_sense', 'varnet'):
            masked_kspace_complex = torch.view_as_complex(masked_kspace.contiguous())
            recon_kwargs = {'anatomy_labels': anatomy_labels} if self.condition_reconstructor else {}
            recon_mag = self.reconstructor(masked_kspace_complex, sense_maps, mask, **recon_kwargs)
            recon = recon_mag.to(torch.complex64)
        else:
            zero_filled_2ch = torch.view_as_real(zero_filled_complex)
            zero_filled_2ch = einops.rearrange(zero_filled_2ch, 'b c h w two -> b (c two) h w')
            recon_2ch = self.reconstructor(zero_filled_2ch)
            recon = torch.view_as_complex(
                einops.rearrange(recon_2ch, 'b (c two) h w -> b c h w two', two=2).contiguous()
            )

        # Compute g-factor map for SENSE-based reconstructors (detached — not differentiable)
        gfactor = None
        if self.with_gfactor and self.recon_type in ('cg_sense', 'lls'):
            with torch.no_grad():
                gfactor = compute_gfactor(sense_maps, mask, num_replicas=self.gfactor_replicas)

        # Store visualization data
        self.data_for_vis[step].update({
            'input': target[0].cpu().detach().numpy() if target.dim() == 3 else target[0, 0].cpu().detach().numpy(),
            'kspace': torch.view_as_complex(full_kspace[0, :, :, :]).cpu().detach().numpy(),
            'masked_kspace': zero_filled_complex[0].cpu().detach().numpy(),
            'zero_filled_recon': zero_filled_mag[0, 0, :, :].cpu().detach().numpy(),
            'recon': recon[0, 0, :, :].abs().cpu().detach().numpy(),
            'uncertainty_map': uncertainty_map[0, 0].cpu().detach().numpy() if uncertainty_map is not None else np.zeros(self.shape),
            'sense_mask': sense_mask[0, 0, :, :].cpu().detach().numpy() if sense_mask is not None else np.zeros(self.shape),
            'gfactor': gfactor[0, 0].cpu().detach().numpy() if gfactor is not None else None,
        })

        return {
            'output': recon, 'mask': mask, 'zero_recon': zero_filled_mag,
            'uncertainty_map': uncertainty_map, 'gfactor': gfactor,
            'anatomy_logits': anatomy_logits, 'reinforce_logp': reinforce_logp,
        }

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

    def forward(self, target, full_kspace, idx=None, anatomy_labels=None):  # noqa: ARG002
        """Run full sequential sampling pipeline.

        Args:
            target: Ground truth image, shape [B, 1, H, W]
            full_kspace: Multicoil kspace, shape [B, C, H, W, 2]
            anatomy_labels: Optional long tensor [B] of anatomy indices

        Returns:
            Dict with final reconstruction, intermediate results, masks, and sense maps.
        """
        full_kspace_complex = torch.view_as_complex(full_kspace.contiguous())

        sense_maps, sense_mask = self.sensetivity_net(full_kspace_complex, mask_method=self.sense_mask_method)
        old_mask = self._init_mask(full_kspace)
        valid_mask = self._valid_kspace_mask(full_kspace)
        pred_kspace = full_kspace * old_mask.unsqueeze(-1)

        uncertainty_map = torch.zeros_like(old_mask)
        gfactor = None

        reconstructions = []
        zero_filled = []
        uncertainty_maps = []
        masks = []
        anatomy_logits_list = []
        reinforce_logp = None  # [B], sum of every step's Plackett-Luce log-prob

        for i in range(self.num_step):
            pred_dict = self.step_forward(
                full_kspace, pred_kspace, old_mask, target, i,
                sense_maps, sense_mask, uncertainty_map, valid_mask,
                prev_gfactor=gfactor, anatomy_labels=anatomy_labels,
            )

            new_img = pred_dict['output']
            old_mask = pred_dict['mask'].detach()
            uncertainty_map = pred_dict['uncertainty_map']
            gfactor = pred_dict.get('gfactor')

            reconstructions.append(new_img.detach())
            zero_filled.append(pred_dict['zero_recon'])
            uncertainty_maps.append(uncertainty_map)
            masks.append(old_mask)
            if pred_dict['anatomy_logits'] is not None:
                anatomy_logits_list.append(pred_dict['anatomy_logits'])
            if pred_dict.get('reinforce_logp') is not None:
                step_logp = pred_dict['reinforce_logp']
                reinforce_logp = step_logp if reinforce_logp is None else reinforce_logp + step_logp

            image_for_kspace = new_img * sense_maps.conj()
            pred_kspace = torch.view_as_real(torch_fft2(image_for_kspace))

        final_recon = torch_ifft2(torch.view_as_complex(pred_kspace)).abs().pow(2).sum(dim=1, keepdim=True).sqrt()

        anatomy_logits = torch.stack(anatomy_logits_list).mean(dim=0) if anatomy_logits_list else None

        return {
            'output': final_recon,
            'intermediate_recons': reconstructions,
            'mask': old_mask,
            'masks': masks,
            'zero_filled_recon': zero_filled,
            'uncertainty_maps': uncertainty_maps,
            'sense_maps': sense_maps,
            'reinforce_logp': reinforce_logp,
            'anatomy_logits': anatomy_logits,
        }

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

        # Uncertainty loss (NLL)
        nll_loss = torch.tensor(0.0, device=pred.device)
        if self.with_uncertainty:
            for step_output, step_uncertainty in zip(pred_dict['output'], pred_dict['uncertainty_maps']):
                nll_loss = nll_loss + torch.mean(
                    compute_gaussian_nll_loss(step_output, target, step_uncertainty)
                )

        # Reconstruction loss
        if loss_type == 'l1':
            reconstruction_loss = F.l1_loss(pred, target, size_average=True)
            zero_loss = F.l1_loss(zero_filled, target, size_average=True)
        elif loss_type == 'ssim':
            reconstruction_loss = -torch.mean(compute_ssim_torch(pred, target))
            zero_loss = -torch.mean(compute_ssim_torch(zero_filled, target))
        elif loss_type == 'nmse':
            reconstruction_loss = torch.norm(pred - target) ** 2 / torch.norm(target) ** 2
            zero_loss = torch.norm(zero_filled - target) ** 2 / torch.norm(target) ** 2
        elif loss_type == 'psnr':
            reconstruction_loss = -torch.mean(compute_psnr_torch(pred, target))
            zero_loss = -torch.mean(compute_psnr_torch(zero_filled, target))
        elif loss_type == 'xentropy':
            label = target_dict['label']
            reconstruction_loss = nn.CrossEntropyLoss()(pred, label)
            zero_loss = torch.tensor(0.0, device=pred.device)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        # K-space loss
        if meta['kspace_weight'] > 0:
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
        # plays, so this needs no extra forward passes -- and the batch-mean reward is
        # the REINFORCE baseline, following scripts/loupe_vs_greedy_oracle.py and
        # scripts/reinforce2d_vs_greedy_loupe.py, where the same estimator was
        # validated against an oracle reconstructor.
        reinforce_loss = torch.tensor(0.0, device=pred.device)
        reinforce_reward = torch.tensor(0.0, device=pred.device)
        reinforce_logp = pred_dict.get('reinforce_logp')
        if reinforce_logp is not None:
            if pred.shape[0] < 2:
                raise RuntimeError(
                    "mask_train_mode='reinforce' needs batch_size >= 2: each batch item is "
                    "one Monte Carlo draw of the mask, and the REINFORCE baseline is their "
                    "mean reward -- with one item there is nothing to compare it to.")
            per_example_nmse = (
                (pred - target).pow(2).flatten(1).sum(dim=1)
                / target.pow(2).flatten(1).sum(dim=1).clamp(min=1e-12)
            )
            reward = -per_example_nmse.detach()
            baseline = reward.mean().detach()
            advantage = reward - baseline
            reinforce_loss = -(advantage * reinforce_logp).mean()
            reinforce_reward = reward.mean()
            loss = loss + self.reinforce_weight * reinforce_loss

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

        if self.with_uncertainty:
            loss = loss + nll_loss * meta['uncertainty_weight']
            log_dict['Uncertainty Loss'] = nll_loss.item()

        return loss, log_dict

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

            pkl_path = os.path.join(options.visualization_dir, f'{data_for_vis_name}_step{step}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.data_for_vis[step], f)
