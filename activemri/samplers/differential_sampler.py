import os
import pickle

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.samplers.conv_sampler import BiLineConstrainedSampler, KspaceLineConstrainedSampler
from activemri.samplers.ConvProb2D import SampProb2D
from activemri.samplers.LearnedProb2D import RescaleProbMap, ThresholdRandomMaskSigmoidV1
from activemri.utils.image_processing import fft_2d_img as torch_fft2
from activemri.utils.image_processing import ifft_2d_img as torch_ifft2
from activemri.utils.util import compute_gaussian_nll_loss, compute_psnr_torch, compute_ssim_torch


class Sampler(nn.Module):
    """Generates sampling masks for sequential k-space acquisition.

    Components:
        mask_net: Predicts sampling probabilities from current observations
        rescale: Normalizes probabilities to match target sparsity
        binarize: Stochastic binarization via straight-through estimator
    """

    def __init__(
        self, 
        shape=(320, 320), 
        line_constrained=False, 
        bi_direction=False, 
        binary_sampler=False, 
        with_uncertainty=False, 
        clamp=100, 
        detach_kspace=False, 
        fixed_input=False
        ):
        super().__init__()
        self.shape = shape
        self.line_constrained = line_constrained
        self.bidirection = bi_direction
        self.detach_kspace = detach_kspace
        self.with_uncertainty = with_uncertainty

        if line_constrained:
            if bi_direction:
                if binary_sampler:
                    raise NotImplementedError("Binary sampler not supported with bi-direction")
                self.mask_net = BiLineConstrainedSampler(
                    shape[0], clamp=clamp, with_uncertainty=with_uncertainty, fixed_input=fixed_input
                )
            else:
                self.mask_net = KspaceLineConstrainedSampler(
                    shape[0], shape[0], with_uncertainty=with_uncertainty, fixed_input=fixed_input
                )
        else:
            self.mask_net = SampProb2D(fixed_input=fixed_input)

        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply

    def forward(self, observed_kspace, old_mask, budget, uncertainty_mask=None):
        """
        Args:
            observed_kspace: Already sampled measurements, shape NHWC
            old_mask: Previous sampling trajectories, shape NCHW
            budget: Number of sampling lines or pixels to acquire
            uncertainty_mask: Optional uncertainty map for guided sampling

        Returns:
            binary_mask: Updated sampling mask (includes previous locations), shape NCHW
            data_to_vis_sampler: Dict of intermediate results for visualization
        """
        if self.detach_kspace:
            observed_kspace = observed_kspace.detach()

        input_kspace = torch.cat([observed_kspace, observed_kspace * old_mask.permute(0, 2, 3, 1)], dim=-1)
        if self.with_uncertainty:
            samp_prob_unscaled = self.mask_net(input_kspace, old_mask, uncertainty_mask)
        else:
            samp_prob_unscaled = self.mask_net(input_kspace, old_mask)

        sparsity = budget / self.shape[0] if self.line_constrained else budget / (self.shape[0] * self.shape[1])
        batch_size = samp_prob_unscaled.shape[0]

        if self.bidirection:
            samp_prob = self.rescale(samp_prob_unscaled, sparsity / 2)
            binary_mask = self.binarize(samp_prob)

            vert_mask, horiz_mask = torch.chunk(binary_mask, dim=-1, chunks=2)
            vert_mask = vert_mask.reshape(batch_size, 1, 1, -1)
            horiz_mask = horiz_mask.reshape(batch_size, 1, -1, 1)
            binary_mask = torch.clamp(vert_mask + horiz_mask, max=1, min=0)

            if self.line_constrained:
                # Reshape for visualization
                vert_raw, horiz_raw = torch.chunk(samp_prob_unscaled, dim=-1, chunks=2)
                samp_prob_unscaled = vert_raw.reshape(batch_size, 1, 1, -1) + horiz_raw.reshape(batch_size, 1, -1, 1)

                vert_scaled, horiz_scaled = torch.chunk(samp_prob, dim=-1, chunks=2)
                samp_prob = vert_scaled.reshape(batch_size, 1, 1, -1) + horiz_scaled.reshape(batch_size, 1, -1, 1)
        else:
            samp_prob = self.rescale(samp_prob_unscaled, sparsity)
            binary_mask = self.binarize(samp_prob)

        binary_mask = torch.clamp(old_mask + binary_mask, min=0, max=1)

        data_to_vis_sampler = {
            'prob_mask': samp_prob_unscaled[0, 0, :, :].detach(),
            'rescaled_mask': samp_prob[0, 0, :, :].detach(),
            'binarized_mask': binary_mask[0, 0, :, :].detach(),
        }

        return binary_mask, data_to_vis_sampler


class SequentialUnet(nn.Module):
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
            line_constrained=False,
            num_step=4,
            preselect=True,
            preselect_num=2,
            bi_direction=False,
            binary_sampler=False,
            clamp=100,
            with_uncertainty=False,
            detach_kspace=False,
            fixed_input=False,
            recon_type='unet',
            sense_tikhonov_weight=0.0,
            ):
        super().__init__()

        self.recon_type = recon_type
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.preselect_num_one_side = preselect_num // 2
        self.bidirection = bi_direction
        self.line_constrained = line_constrained
        self.with_uncertainty = with_uncertainty
        self.sense_tikhonov_weight = sense_tikhonov_weight
        self.data_for_vis: list[dict] = [dict() for _ in range(num_step)]

        self.reconstructor = reconstructor

        # Sampler
        self.sampler = Sampler(
            shape, line_constrained, bi_direction=bi_direction,
            binary_sampler=binary_sampler, clamp=clamp,
            with_uncertainty=with_uncertainty, detach_kspace=detach_kspace,
            fixed_input=fixed_input,
        )

        # Sensitivity estimation
        self.sensetivity_net = SensetivityModel_mc(2, 2, 8)

        # Compute per-step sampling budget
        if not line_constrained:
            effective_sparsity = sparsity - (preselect_num ** 2) / (shape[0] * shape[1]) if preselect else sparsity
            self.budget = effective_sparsity * shape[0] * shape[1]
        else:
            effective_sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity
            self.budget = effective_sparsity * shape[0]

        print(f"Number of sequential steps: {num_step}")
        print(f"Sparsity: {sparsity}")

    def step_forward(self, full_kspace, pred_kspace, old_mask, target, step,
                     sense_maps, sense_mask, uncertainty_map=None):
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

        Returns:
            Dict with keys: output, mask, zero_recon, uncertainty_map
        """
        budget = self.budget / self.num_step

        # Combine coils to single-coil equivalent for sampler input
        pred_single_coil_img = (torch_ifft2(torch.view_as_complex(pred_kspace)) * sense_maps).sum(dim=1)
        pred_single_coil_k = torch.view_as_real(torch_fft2(pred_single_coil_img))

        if torch.isnan(pred_single_coil_k).any():
            print(f"NaN in pred_single_coil_k at step {step}")

        # Generate new sampling mask
        mask, data_to_vis_sampler = self.sampler(pred_single_coil_k, old_mask, budget, uncertainty_map)
        masked_kspace = torch.view_as_real(mask * torch.view_as_complex(full_kspace))
        self.data_for_vis[step].update(data_to_vis_sampler)

        # Zero-filled reconstruction (coil-combined)
        zero_filled_complex = (torch_ifft2(torch.view_as_complex(masked_kspace)) * sense_maps.conj()).sum(dim=1, keepdim=True)
        zero_filled_mag = zero_filled_complex.abs()

        # Full reconstruction
        if self.recon_type in ('lls', 'cg_sense', 'varnet'):
            masked_kspace_complex = torch.view_as_complex(masked_kspace.contiguous())
            recon_mag = self.reconstructor(masked_kspace_complex, sense_maps, mask)
            recon = recon_mag.to(torch.complex64)
        else:
            zero_filled_2ch = torch.view_as_real(zero_filled_complex)
            zero_filled_2ch = einops.rearrange(zero_filled_2ch, 'b c h w two -> b (c two) h w')
            recon_2ch = self.reconstructor(zero_filled_2ch)
            recon = torch.view_as_complex(
                einops.rearrange(recon_2ch, 'b (c two) h w -> b c h w two', two=2).contiguous()
            )

        # Store visualization data
        self.data_for_vis[step].update({
            'input': target[0].cpu().detach().numpy() if target.dim() == 3 else target[0, 0].cpu().detach().numpy(),
            'kspace': torch.view_as_complex(full_kspace[0, :, :, :]).cpu().detach().numpy(),
            'masked_kspace': zero_filled_complex[0].cpu().detach().numpy(),
            'zero_filled_recon': zero_filled_mag[0, 0, :, :].cpu().detach().numpy(),
            'recon': recon[0, 0, :, :].abs().cpu().detach().numpy(),
            'uncertainty_map': uncertainty_map[0, 0].cpu().detach().numpy() if uncertainty_map is not None else np.zeros(self.shape),
            'sense_mask': sense_mask[0, 0, :, :].cpu().detach().numpy(),
        })

        return {'output': recon, 'mask': mask, 'zero_recon': zero_filled_mag, 'uncertainty_map': uncertainty_map}

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

    def forward(self, target, full_kspace, idx=None):  # noqa: ARG002
        """Run full sequential sampling pipeline.

        Args:
            target: Ground truth image, shape [B, 1, H, W]
            full_kspace: Multicoil kspace, shape [B, C, H, W, 2]

        Returns:
            Dict with final reconstruction, intermediate results, masks, and sense maps.
        """
        full_kspace_complex = torch.view_as_complex(full_kspace.contiguous())

        sense_maps, sense_mask = self.sensetivity_net(full_kspace_complex, mask_method='pre_network')
        old_mask = self._init_mask(full_kspace)
        pred_kspace = full_kspace * old_mask.unsqueeze(-1)

        uncertainty_map = torch.zeros_like(old_mask)

        reconstructions = []
        zero_filled = []
        uncertainty_maps = []
        masks = []

        for i in range(self.num_step):
            pred_dict = self.step_forward(
                full_kspace, pred_kspace, old_mask, target, i,
                sense_maps, sense_mask, uncertainty_map,
            )

            new_img = pred_dict['output']
            old_mask = pred_dict['mask'].detach()
            uncertainty_map = pred_dict['uncertainty_map']

            reconstructions.append(new_img.detach())
            zero_filled.append(pred_dict['zero_recon'])
            uncertainty_maps.append(uncertainty_map)
            masks.append(old_mask)

            # Predict next k-space from current reconstruction
            image_for_kspace = new_img * sense_maps.conj()
            pred_kspace = torch.view_as_real(torch_fft2(image_for_kspace))

        # Final RSS reconstruction
        final_recon = torch_ifft2(torch.view_as_complex(pred_kspace)).abs().pow(2).sum(dim=1, keepdim=True).sqrt()

        return {
            'output': final_recon,
            'intermediate_recons': reconstructions,
            'mask': old_mask,
            'masks': masks,
            'zero_filled_recon': zero_filled,
            'uncertainty_maps': uncertainty_maps,
            'sense_maps': sense_maps,
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

        log_dict = {
            'Total Loss': loss.item(),
            'Zero Filled Loss': zero_loss.item(),
            'K-space Loss': kspace_loss.item(),
            'Recon loss': reconstruction_loss.item(),
            'Tikhonov Loss': tikhonov_loss.item(),
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
