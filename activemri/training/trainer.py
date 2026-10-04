from collections import defaultdict
from typing import List, Optional
import logging
import os
import random
import warnings

import torch
import torch.nn.functional as F
import wandb
import numpy as np

import time
import matplotlib.pyplot as plt

from activemri.utils.evaluate import build_metrics
from activemri.models.sequential_sampling import SequentialSampling, reinforce_group_size
from activemri.utils.util import compute_ssim_torch, compute_psnr_torch
from activemri.config import TrainConfig
from activemri.reconstructors.config import UNetReconConfig, CGSenseReconConfig, VarNetReconConfig
from activemri.reconstructors.CG_Sense import CGSENSEReconstructor
from activemri.reconstructors.UNet import LOUPEUNet
from activemri.reconstructors.VarNet import VarNet


class Sequential_Trainer:
    def __init__(
        self,
        options: TrainConfig,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self.options = options
        self.device = device
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.anatomy_to_idx = {name: i for i, name in enumerate(options.label_classes)}
        self.model = build_model(self.options).to(device)

        if options.pretrained_recon is not None:
            self._load_pretrained_recon(options.pretrained_recon)
        if options.freeze_recon:
            self._freeze_recon()

        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = build_optimizer(options.lr, options.weight_decay, self._param_groups())
        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader is not None else train_loader
        self.visual_examples = select_visual_examples(self.val_loader.dataset)

        self.scheduler = build_lr_scheduler(options, self.optimizer)
        self.best_val_loss = np.inf
        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        exp_dir = options.out_dir[0]
        exp_dir.mkdir(parents=True, exist_ok=True)

        vis_dir = exp_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir
        self.vis_dir = vis_dir

        run_name = self._build_run_name(options)
        wandb.init(project="SeqMRI", name=run_name, config=vars(options), dir=str(exp_dir))

        self.logger.info(self.options)
        self.logger.info(self.model)

    def _load_pretrained_recon(self, checkpoint_path: str):
        """Load only the `reconstructor`/`sensetivity_net` weights from a prior
        run's checkpoint, leaving the sampler at its fresh initialization."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
        recon_state = {
            k: v for k, v in state_dict.items()
            if k.startswith('reconstructor.') or k.startswith('sensetivity_net.')
        }
        missing, unexpected = self.model.load_state_dict(recon_state, strict=False)
        missing = [k for k in missing if not k.startswith('sampler.')]
        if missing or unexpected:
            raise RuntimeError(
                f"Pretrained recon checkpoint {checkpoint_path} did not match the model's "
                f"reconstructor/sensetivity_net: missing={missing}, unexpected={unexpected}"
            )
        self.logger.info(f"Loaded pretrained reconstructor weights from {checkpoint_path}")

    def _param_groups(self):
        """Trainable parameters, with the mask logits split out when REINFORCE has its own lr.

        Under Adam each logit moves about `lr` per step whatever its gradient's
        scale, so the policy's pace is set by this lr, not by the reconstructor's.
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        policy_lr = self.options.reinforce_lr
        if self.options.mask_train_mode != 'reinforce' or policy_lr is None:
            return trainable
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        policy = [p for p in model.sampler.mask_net.parameters() if p.requires_grad]
        policy_ids = {id(p) for p in policy}
        rest = [p for p in trainable if id(p) not in policy_ids]
        return [group for group in ({'params': rest}, {'params': policy, 'lr': policy_lr})
                if group['params']]

    @staticmethod
    def _replicate_for_reinforce(batch, group_size=0):
        """Replace a batch of distinct slices with `group_size` copies of each of its first few slices.

        REINFORCE's baseline is the mean reward over the draws that share a
        slice, and every batch item draws its own independent mask. If a group
        held different slices, the advantage (reward - baseline) would be
        dominated by per-slice difficulty rather than mask quality -- the policy
        gradient drowns in noise it can't act on. Replicating a slice across its
        group, as `train_single_slice_loupe.py` already does, makes every item
        in it an independent Monte Carlo draw of the *same* reconstruction
        target, so the baseline actually isolates mask quality.

        With `group_size=0` the whole batch is one group (a single slice per
        step). A smaller group spends the same batch on several slices, so each
        policy update averages over slices instead of chasing one.
        """
        n = batch['kspace'].shape[0]
        group = reinforce_group_size(n, group_size)
        slices = n // group
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value[:slices].repeat_interleave(group, dim=0)
            elif isinstance(value, list):
                out[key] = [item for item in value[:slices] for _ in range(group)]
            else:
                out[key] = value
        return out

    def _recon_modules(self):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        return model.reconstructor, model.sensetivity_net

    def _freeze_recon(self):
        for module in self._recon_modules():
            module.requires_grad_(False)
            module.eval()
        self.logger.info("Froze reconstructor and sensetivity_net; training sampler only")

    @staticmethod
    def _build_run_name(options: TrainConfig) -> str:
        parts = []

        if options.model and options.model != 'SequentialSampling':
            parts.append(options.model)

        recon_type = options.recon.recon_type
        parts.append(recon_type)

        if options.loss_type != 'l1':
            parts.append(f"loss-{options.loss_type}")

        if options.anatomy_classes:
            parts.append("-".join(options.anatomy_classes))

        if options.train_anatomies:
            parts.append("train-" + "-".join(options.train_anatomies))

        if options.sampler_type != 'conv_2d':
            parts.append(f"sampler-{options.sampler_type}")

        if getattr(options, 'caipi_keep_fraction', 1.0) != 1.0:
            parts.append(f"pf{options.caipi_keep_fraction:g}")

        if (options.sampler_chans, options.sampler_depth) != (64, 4):
            parts.append(f"samp-c{options.sampler_chans}d{options.sampler_depth}")

        if getattr(options, 'sampler_image_input', False):
            parts.append("samp-img")

        if options.uncertainty_loss:
            parts.append("unc")

        if options.anatomy_weight > 0:
            parts.append(f"anat{options.anatomy_weight:.1f}")

        if options.anatomy_conditioning:
            parts.append("cond-anat")

        if options.mask_train_mode != 'straight_through':
            parts.append(f"mask-{options.mask_train_mode}")

        if options.freeze_recon:
            parts.append("frozen-recon")

        res = f"{options.resolution[0]}x{options.resolution[1]}"
        parts.append(res)

        accel = 'x'.join(str(int(a)) if a == int(a) else str(a) for a in options.accelerations)
        parts.append(f"{accel}acc")

        is_loupe = 'loupe' in (options.model or '').lower() or 'loupe' in options.sampler_type.lower()
        is_static = options.sampler_type == 'oracle'

        if is_static:
            parts.append("static")
        elif is_loupe:
            parts.append("zero_step")
        else:
            parts.append(f"{options.num_step}step")

        parts.append(f"bs{options.batch_size}")

        return "_".join(parts)

    def __call__(self):
        if self.options.checkpoint is not None:
            self.load_checkpoint()
        return self._train_loop()

    def _train_loop(self):
        history = defaultdict(list)

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.scheduler.step(epoch)

            train_loss, train_time, train_metrics = self.train_epoch()
            val_loss, mean_sparsity, val_time, val_metrics = self.val_epoch()

            for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
                for key, val in metrics.items():
                    history[f'{prefix}_{key.lower()}'].append(
                        val.item() if hasattr(val, 'item') else val
                    )
            self._plot_metrics(history)

            is_new_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.save_model(is_new_best)

            self.logger.info(
                f'Epoch = [{1 + self.epoch:4d}/{self.options.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} MeanSparsity = {mean_sparsity:.4g} '
                f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )

        wandb.finish()

    def train_epoch(self):
        self.model.train()
        if self.options.freeze_recon:
            for module in self._recon_modules():
                module.eval()

        losses = []
        metrics = build_metrics(device=self.options.device)
        anatomy_stats = _AnatomyStats(self.options.device)
        logged_train_anatomies = set()
        anatomy_images = {}

        epoch_start_time = start_iter = time.perf_counter()
        global_step_offset = self.epoch * len(self.train_loader)

        max_batches = max(1, int(self.options.limit_batches * len(self.train_loader)))
        for iter_idx, batch in enumerate(self.train_loader):
            if iter_idx >= max_batches:
                break

            if self.options.mask_train_mode == 'reinforce':
                batch = self._replicate_for_reinforce(batch, self.options.reinforce_group_size)

            kspace = batch['kspace'].to(self.device)
            target = batch['target'].to(self.device)
            anatomies = list(batch['anatomy'])

            try:
                pred_dict = self.model(target, kspace, anatomy_labels=self._anatomy_labels(batch))
            except RuntimeError as err:
                if 'Binarization failed to match target density' not in str(err):
                    raise
                self.logger.warning(
                    "Skipping batch with NaN-driven binarization failure "
                    f"(fnames={batch['fname']}, slice_ids={batch['slice_id'].tolist()}): {err}"
                )
                self.optimizer.zero_grad()
                continue

            if iter_idx == 0:
                data_for_vis_name = f'train_epoch={self.epoch + 1}_iter=1'
                self._visualize_and_save(self.epoch, data_for_vis_name)

            for anatomy in dict.fromkeys(anatomies):
                if anatomy in logged_train_anatomies:
                    continue
                logged_train_anatomies.add(anatomy)
                index = anatomies.index(anatomy)
                anatomy_images.update(_build_image_log(
                    pred_dict, target, f'train/{anatomy}',
                    self.model.data_for_vis if index == 0 else None,
                    index=index,
                ))

            output = pred_dict['output']
            target_dict = {'target': target, 'kspace': kspace}
            anatomy_labels = self._anatomy_labels(batch)
            if anatomy_labels is not None:
                target_dict['anatomy_labels'] = anatomy_labels
            meta = {
                'recon_weight': 1.0, 'kspace_weight': 0.0,
                'uncertainty_weight': self.options.uncertainty_weight,
                'anatomy_weight': self.options.anatomy_weight,
            }
            loss, log_dict = self.model.loss(pred_dict, target_dict, meta, self.options.loss_type)

            self.optimizer.zero_grad()
            loss.backward()
            grad_log = gradient_stats(self.model, GRAD_CLIP_NORM)
            bad_params = nonfinite_gradient_params(self.model) if grad_log['grad/nonfinite'] else []
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            grad_log['data/zero_coil_frac'] = zero_coil_fraction(kspace)
            grad_log['train/lr'] = self.optimizer.param_groups[0]['lr']
            if self.options.mask_train_mode == 'reinforce':
                grad_log['train/policy_lr'] = self.optimizer.param_groups[-1]['lr']
            if grad_log['grad/nonfinite']:
                self.logger.warning(
                    "Skipping optimizer step with non-finite gradient norm "
                    f"(loss={loss.item():.4g}, fnames={batch['fname']}, slice_ids={batch['slice_id'].tolist()}, "
                    f"non-finite params={bad_params})"
                )
                wandb.log({**grad_log, 'train/skipped_step': 1.0, 'train/step': global_step_offset + iter_idx})
                self.optimizer.zero_grad()
                continue
            self.optimizer.step()

            losses.append(loss.item())

            target_for_metrics = target.detach()
            if output.dim() != target_for_metrics.dim():
                target_for_metrics = target_for_metrics.unsqueeze(1)
            data_range = target_for_metrics.max()
            metrics["PSNR"].data_range = data_range
            metrics["SSIM"].data_range = data_range
            masked_output, masked_target = _apply_signal_mask(output.detach(), target_for_metrics)
            metrics.update(masked_output, masked_target)
            anatomy_stats.update(anatomies, masked_output, masked_target, pred_dict['mask'].detach())

            step_log = {'train/loss': loss.item(), 'train/step': global_step_offset + iter_idx, **grad_log}
            if self.options.mask_train_mode == 'reinforce':
                step_log['train/reinforce_reward'] = log_dict.get('Reinforce Reward', 0.0)
                step_log['train/reinforce_loss'] = log_dict.get('Reinforce Loss', 0.0)
                for key, name in (('Reinforce Entropy', 'reinforce_entropy'),
                                  ('Reinforce PG Grad Norm', 'reinforce_pg_grad_norm'),
                                  ('Reinforce Entropy Grad Norm', 'reinforce_entropy_grad_norm')):
                    if key in log_dict:
                        step_log[f'train/{name}'] = log_dict[key]
            anatomy_logits = pred_dict.get('anatomy_logits')
            if anatomy_logits is not None and 'anatomy_labels' in target_dict:
                preds = anatomy_logits.argmax(dim=1)
                acc = (preds == target_dict['anatomy_labels']).float().mean()
                step_log['train/anatomy_acc'] = acc.item()
                step_log['train/anatomy_loss'] = log_dict.get('Anatomy Loss', 0.0)

            start_iter = time.perf_counter()
            wandb.log(step_log)

        metric_means = metrics.compute()
        epoch_log = {f'train/{name}': value.item() for name, value in metric_means.items()}
        epoch_log['train/loss'] = np.mean(losses)
        epoch_log['epoch'] = self.epoch
        epoch_log.update(anatomy_images)
        epoch_log.update(anatomy_stats.compute('train'))
        wandb.log(epoch_log)

        return np.mean(losses), time.perf_counter() - epoch_start_time, metric_means

    def val_epoch(self):
        self.model.eval()
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)
        anatomy_stats = _AnatomyStats(self.options.device)
        start = time.perf_counter()
        logged_anatomies = set()
        anatomy_images = {}

        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_loader):
                kspace = batch['kspace'].to(self.device)
                target = batch['target'].to(self.device)
                anatomies = list(batch['anatomy'])

                try:
                    pred_dict = self.model(target, kspace, anatomy_labels=self._anatomy_labels(batch))
                except RuntimeError as err:
                    if 'Binarization failed to match target density' not in str(err):
                        raise
                    self.logger.warning(
                        "Skipping val batch with NaN-driven binarization failure "
                        f"(fnames={batch['fname']}, slice_ids={batch['slice_id'].tolist()}): {err}"
                    )
                    continue

                if iter_idx == 0:
                    data_for_vis_name = f'val_epoch={self.epoch + 1}'
                    self._visualize_and_save(self.epoch, data_for_vis_name)

                for anatomy in dict.fromkeys(anatomies):
                    if anatomy in logged_anatomies or anatomy in self.visual_examples:
                        continue
                    logged_anatomies.add(anatomy)
                    index = anatomies.index(anatomy)
                    anatomy_images.update(_build_image_log(
                        pred_dict, target, f'val/{anatomy}',
                        self.model.data_for_vis if index == 0 else None,
                        index=index,
                    ))

                output = pred_dict['output']
                if isinstance(output, list):
                    output = output[-1]

                target_dict = {'target': target, 'kspace': kspace}
                anatomy_labels = self._anatomy_labels(batch)
                if anatomy_labels is not None:
                    target_dict['anatomy_labels'] = anatomy_labels
                meta = {
                    'recon_weight': 1.0, 'kspace_weight': 0.0,
                    'uncertainty_weight': self.options.uncertainty_weight,
                    'anatomy_weight': self.options.anatomy_weight,
                }
                loss, _ = self.model.loss(pred_dict, target_dict, meta, self.options.loss_type)

                mask = pred_dict['mask']
                sparsity.append(torch.mean(mask).item())
                losses.append(loss.item())

                target_for_metrics = target
                if output.dim() != target_for_metrics.dim():
                    target_for_metrics = target_for_metrics.unsqueeze(1)
                data_range = target_for_metrics.max()
                metrics["PSNR"].data_range = data_range
                metrics["SSIM"].data_range = data_range
                masked_output, masked_target = _apply_signal_mask(output, target_for_metrics)
                metrics.update(masked_output, masked_target)
                anatomy_stats.update(anatomies, masked_output, masked_target, mask)

            anatomy_images.update(self._fixed_example_images())

            metric_means = metrics.compute()
            epoch_log = {f'val/{name}': value.item() for name, value in metric_means.items()}
            epoch_log['val/loss'] = np.mean(losses)
            epoch_log['epoch'] = self.epoch
            epoch_log.update(anatomy_images)
            epoch_log.update(anatomy_stats.compute('val'))
            wandb.log(epoch_log)

            self.logger.info(anatomy_stats.summary())

        return np.mean(losses), np.mean(sparsity), time.perf_counter() - start, metric_means

    def _fixed_example_images(self):
        """Image logs for the same mid-volume val slice of each anatomy every epoch."""
        images = {}
        dataset = self.val_loader.dataset
        for anatomy, index in self.visual_examples.items():
            batch = self.val_loader.collate_fn([dataset[index]])
            kspace = batch['kspace'].to(self.device)
            target = batch['target'].to(self.device)
            try:
                pred_dict = self.model(target, kspace, anatomy_labels=self._anatomy_labels(batch))
            except RuntimeError as err:
                if 'Binarization failed to match target density' not in str(err):
                    raise
                self.logger.warning(f"Skipping fixed {anatomy} val example: {err}")
                continue
            images.update(_build_image_log(pred_dict, target, f'val/{anatomy}', self.model.data_for_vis))
        return images

    def _anatomy_labels(self, batch):
        """Conditioning-class index tensor for a batch, or None when the run has no classes."""
        if not self.anatomy_to_idx:
            return None
        return torch.tensor([self.anatomy_to_idx[n] for n in batch[self.options.label_key]], device=self.device)

    def _plot_metrics(self, history):
        epochs = list(range(1, len(history['train_ssim']) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, metric in zip(axes, ['ssim', 'nmse', 'psnr']):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            if train_key in history:
                ax.plot(epochs, history[train_key], label='Train')
            if val_key in history:
                ax.plot(epochs, history[val_key], label='Val')
            ax.set_xlabel('Epoch')
            ax.set_title(metric.upper())
            ax.legend()

        fig.tight_layout()
        fig.savefig(self.vis_dir / 'metrics.png', dpi=150)
        plt.close(fig)

    def loss(self, pred_dict, target, loss_type):
        pred = pred_dict['output']
        energy = pred_dict['energy']

        if loss_type == 'l1':
            reconstruction_loss = F.l1_loss(pred, target)
        elif loss_type == 'ssim':
            reconstruction_loss = -torch.mean(compute_ssim_torch(pred, target))
        elif loss_type == 'psnr':
            reconstruction_loss = -torch.mean(compute_psnr_torch(pred, target))
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        entropy_loss = torch.mean(energy)
        loss = reconstruction_loss + self.options.uncertainty_weight * entropy_loss

        log_dict = {
            'Total Loss': loss.item(),
            'Entropy': entropy_loss.item(),
            'Reconstruction': reconstruction_loss.item(),
        }
        return loss, log_dict

    def save_model(self, is_new_best):
        checkpoint = {
            'epoch': self.epoch,
            'options': self.options,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if not is_new_best:
            return

        best_path = self.exp_dir / 'best_model.pt'
        torch.save(checkpoint, f=best_path)
        for old_checkpoint in self.exp_dir.glob('model_epoch_*.pt'):
            old_checkpoint.unlink()

    def load_checkpoint(self):
        checkpoint = torch.load(self.options.checkpoint, weights_only=False)
        self.logger.info(
            f"Loaded checkpoint epoch {checkpoint['epoch']} with loss {checkpoint['best_val_loss']}"
        )
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.start_epoch = checkpoint['epoch'] + 1


    def _visualize_and_save(self, epoch, data_for_vis_name):
        data_for_vis = self.model.data_for_vis
        assert data_for_vis is not None
        for step in range(self.options.num_step):
            has_gfactor = data_for_vis[step].get('gfactor') is not None
            has_sampler_unc = data_for_vis[step].get('sampler_uncertainty') is not None
            ncols = 5 + int(has_gfactor) + int(has_sampler_unc)

            fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=[3.6 * ncols, 6])
            cmap = "viridis" if self.options.resolution[1] == 32 else "gray"

            ax[0, 0].imshow(data_for_vis[step]['input'], cmap=cmap)
            ax[0, 1].imshow(np.log(np.abs(data_for_vis[step]['kspace'][0])))
            masked_k = data_for_vis[step]['masked_kspace'].copy()
            masked_k[masked_k == 0] = 1
            ax[0, 2].imshow(np.log(np.abs(masked_k[0])))
            ax[0, 3].imshow(data_for_vis[step]['zero_filled_recon'], cmap=cmap)
            ax[0, 4].imshow(data_for_vis[step]['recon'], cmap=cmap)
            ax[1, 0].imshow(data_for_vis[step]['prob_mask'].cpu().detach(), cmap='inferno', aspect='auto')
            ax[1, 1].imshow(data_for_vis[step]['rescaled_mask'].cpu().detach(), cmap='inferno', aspect='auto')
            ax[1, 2].imshow(data_for_vis[step]['binarized_mask'].cpu().detach(), aspect='auto')
            ax[1, 3].imshow(data_for_vis[step]['sense_mask'], cmap='gray', vmin=0, vmax=1)
            ax[1, 4].imshow(np.exp(data_for_vis[step]['uncertainty_map']), cmap=cmap)

            ax[0, 0].title.set_text('Input image')
            ax[0, 1].title.set_text('Log k-space of the input')
            ax[0, 2].title.set_text('Undersampled log k-space')
            ax[0, 4].title.set_text('Reconstruction')
            ax[1, 0].title.set_text('Probabilistic mask')
            ax[1, 1].title.set_text('Rescaled mask')
            ax[1, 2].title.set_text('Binary mask')
            ax[1, 3].title.set_text('Sense body mask')
            ax[1, 4].title.set_text('Uncertainty of the reconstruction')

            col = 5
            if has_gfactor:
                gf = data_for_vis[step]['gfactor']
                im = ax[0, col].imshow(gf, cmap='inferno', vmin=0)
                ax[0, col].title.set_text(f'G-factor (max={gf.max():.1f})')
                fig.colorbar(im, ax=ax[0, col], fraction=0.046)
                ax[1, col].imshow(data_for_vis[step]['prob_mask'].cpu().detach(), cmap='inferno', aspect='auto')
                ax[1, col].title.set_text('Prob mask (inferno)')
                col += 1

            if has_sampler_unc:
                unc = data_for_vis[step]['sampler_uncertainty']
                unc_np = unc.cpu().numpy() if hasattr(unc, 'cpu') else unc
                im = ax[0, col].imshow(unc_np, cmap='inferno')
                ax[0, col].title.set_text('Sampler uncertainty')
                fig.colorbar(im, ax=ax[0, col], fraction=0.046)
                ax[1, col].axis('off')
                col += 1

            for a in ax.flat:
                a.axis('off')

            plt.suptitle(
                f'Epoch = [{1 + epoch}/{self.options.num_epochs}] Step = [{step + 1}/{self.options.num_step}]',
                fontsize=20,
            )
            plt.savefig(self.vis_dir / f'{data_for_vis_name}_step{step}.png')
            plt.close()


class NonRLTester:
    def __init__(
        self,
        val_loader: torch.utils.data.DataLoader,
        checkpoint_path: str,
        device: torch.device,
    ):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.options: TrainConfig = checkpoint['options']
        self.device = device

        random.seed(self.options.seed)
        np.random.seed(self.options.seed)
        torch.manual_seed(self.options.seed)

        self.model = build_model(self.options).to(device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.anatomy_classes = self.options.label_classes

        self.val_loader = val_loader

        self.exp_dir = self.options.out_dir[0]
        self.vis_dir = self.exp_dir / 'visualizations'
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        run_name = Sequential_Trainer._build_run_name(self.options)
        wandb.init(project="SeqMRI", name=f"test_{run_name}", config=vars(self.options), dir=str(self.exp_dir))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)

    def _visualize_and_save(self, iter_idx, data_for_vis_name):
        data_for_vis = self.model.data_for_vis
        for step in range(self.options.num_step):
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18, 6])
            cmap = "viridis" if self.options.resolution[1] == 32 else "gray"

            ax[0, 0].imshow(data_for_vis[step]['input'], cmap=cmap)
            ax[0, 1].imshow(np.log(np.abs(data_for_vis[step]['kspace'][0])))
            masked_k = data_for_vis[step]['masked_kspace']
            masked_k[masked_k == 0] = 1
            ax[0, 2].imshow(np.log(np.abs(masked_k[0])))
            ax[0, 3].imshow(data_for_vis[step]['zero_filled_recon'], cmap=cmap)
            ax[0, 4].imshow(data_for_vis[step]['recon'], cmap=cmap)
            ax[1, 0].imshow(data_for_vis[step]['prob_mask'].cpu().detach(), aspect='auto')
            ax[1, 1].imshow(data_for_vis[step]['rescaled_mask'].cpu().detach(), aspect='auto')
            ax[1, 2].imshow(data_for_vis[step]['binarized_mask'].cpu().detach(), aspect='auto')
            ax[1, 3].imshow(data_for_vis[step]['sense_mask'], cmap='gray', vmin=0, vmax=1)
            ax[1, 4].imshow(np.exp(data_for_vis[step]['uncertainty_map']), cmap=cmap)

            ax[0, 0].title.set_text('Input image')
            ax[0, 1].title.set_text('Log k-space of the input')
            ax[0, 2].title.set_text('Undersampled log k-space')
            ax[0, 4].title.set_text('Reconstruction')
            ax[1, 0].title.set_text('Probabilistic mask')
            ax[1, 1].title.set_text('Rescaled mask')
            ax[1, 2].title.set_text('Binary mask')
            ax[1, 3].title.set_text('Sense body mask')
            ax[1, 4].title.set_text('Uncertainty of the reconstruction')

            plt.suptitle(f'Test iter = {iter_idx + 1} Step = [{step + 1}/{self.options.num_step}]', fontsize=20)
            plt.savefig(self.vis_dir / f'{data_for_vis_name}_step{step}.png')
            plt.close()

    def evaluate(self):
        self.model.eval()
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)

        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_loader):
                kspace = batch['kspace'].to(self.device)
                target = batch['target'].to(self.device)

                anatomy_labels = anatomy_labels_from_batch(batch, self.anatomy_classes, self.device, self.options.label_key)

                pred_dict = self.model(target, kspace, anatomy_labels=anatomy_labels)

                if iter_idx % 100 == 0:
                    data_for_vis_name = f'test_iter={iter_idx + 1}'
                    self._visualize_and_save(iter_idx, data_for_vis_name)
                    wandb.log(_build_image_log(pred_dict, target, 'test', self.model.data_for_vis))

                output = pred_dict['output']
                if isinstance(output, list):
                    output = output[-1]

                loss = compute_ssim_torch(output, target)

                mask = pred_dict['mask']
                sparsity.append(torch.mean(mask).item())
                losses.append(loss.item())

                data_range = target.max()
                metrics["PSNR"].data_range = data_range
                metrics["SSIM"].data_range = data_range
                metrics.update(output, target)

        return losses, np.mean(sparsity), metrics.compute()

    def __call__(self):
        val_loss, sparsity, metric_results = self.evaluate()

        self.logger.info(f"L1 Loss {np.mean(val_loss):.4g} +/- {np.std(val_loss):.4g} Sparsity {sparsity:.4g}")

        metrics_str = ' '.join(f'{k} = {v:.4g}' for k, v in metric_results.items())
        with open(self.exp_dir / 'statistics.txt', 'w') as f:
            f.write(f'L1 Loss {np.mean(val_loss)} +- {np.std(val_loss)}\n')
            f.write(metrics_str)

        self.logger.info(metrics_str)
        torch.save(val_loss, self.exp_dir / 'loss.pt')
        wandb.finish()


def select_visual_examples(dataset):
    """Dataset index of a mid-volume slice for each anatomy of a ConcatDataset of
    FastMRIDatasets, so the logged val images are the same representative slice
    every epoch instead of whatever slice happens to lead the first batch.
    Returns {} for any other dataset type."""
    if not isinstance(dataset, torch.utils.data.ConcatDataset):
        return {}
    offsets = [0] + list(dataset.cumulative_sizes[:-1])
    examples = {}
    for offset, sub in zip(offsets, dataset.datasets):
        if not getattr(sub, 'examples', None):
            continue
        volumes = list(dict.fromkeys(fname for fname, _ in sub.examples))
        middle_volume = volumes[len(volumes) // 2]
        slices = [i for i, (fname, _) in enumerate(sub.examples) if fname == middle_volume]
        examples[sub.anatomy] = offset + slices[len(slices) // 2]
    return examples


def anatomy_labels_from_batch(batch, anatomy_classes, device, key='anatomy'):
    """Index tensor for a batch's anatomies, for conditioning at evaluation time.

    Returns None — which makes a conditioned model fall back to the mean
    embedding — when the run has no anatomy classes, the batch carries no
    anatomy, or the data holds an anatomy the model never saw.
    """
    if not anatomy_classes or key not in batch:
        return None

    index = {name: i for i, name in enumerate(anatomy_classes)}
    unknown = {a for a in batch[key] if a not in index}
    if unknown:
        warnings.warn(
            f"Anatomies {sorted(unknown)} are not in the checkpoint's classes {list(anatomy_classes)}; "
            "evaluating with the mean anatomy embedding instead."
        )
        return None

    return torch.tensor([index[a] for a in batch[key]], device=device)


def build_optimizer(lr, weight_decay, model_parameters, optim_type='adam'):
    if optim_type == 'adam':
        return torch.optim.Adam(model_parameters, lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        return torch.optim.SGD(model_parameters, lr, weight_decay=weight_decay)
    raise NotImplementedError(f"Unknown optimizer type: {optim_type}")


def build_lr_scheduler(options: TrainConfig, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, options.lr_step_size, options.lr_gamma)


def _sampler_type(options: TrainConfig) -> str:
    """Sampler family for these options, falling back to the pre-sampler_type
    `line_constrained` flag so older checkpoints still load."""
    sampler_type = getattr(options, 'sampler_type', None)
    if sampler_type is not None:
        return sampler_type
    return 'conv_1d' if getattr(options, 'line_constrained', False) else 'conv_2d'


GRAD_CLIP_NORM = 1.0
GRADIENT_GROUPS = ('sampler', 'reconstructor', 'sensetivity_net')


def _unwrapped(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def gradient_stats(model, max_norm):
    """Per-step gradient diagnostics, keyed for wandb. Call before clipping.

    Reports the global norm, whether clipping will fire, whether any gradient is
    non-finite, and for each of the sampler, reconstructor and sensitivity net:
    the group's gradient norm, its largest absolute entry, and how many
    parameter tensors hold a non-finite gradient. Together these show which
    component blows up first.
    """
    model = _unwrapped(model)
    all_grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.stack(torch._foreach_norm(all_grads)).norm()
    names = ['grad/total_norm', 'grad/clipped', 'grad/nonfinite']
    values = [total_norm, (total_norm > max_norm).float(), (~torch.isfinite(total_norm)).float()]
    for group in GRADIENT_GROUPS:
        grads = [p.grad for p in getattr(model, group).parameters() if p.grad is not None]
        if not grads:
            continue
        per_tensor_norm = torch.stack(torch._foreach_norm(grads))
        names += [f'grad/{group}_norm', f'grad/{group}_max_abs', f'grad/{group}_nonfinite_tensors']
        values += [
            per_tensor_norm.norm(),
            torch.stack([g.abs().max() for g in grads]).max(),
            (~torch.isfinite(per_tensor_norm)).sum().float(),
        ]
    return dict(zip(names, torch.stack([v.float() for v in values]).tolist()))


def nonfinite_gradient_params(model, limit=5):
    """Names of the first few parameters whose gradient is not finite."""
    bad = [
        name for name, p in _unwrapped(model).named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    return bad[:limit] + ([f'... +{len(bad) - limit} more'] if len(bad) > limit else [])


def zero_coil_fraction(kspace):
    """Fraction of coil channels in the batch that are entirely zero (padding)."""
    return (kspace.abs().sum(dim=(2, 3, 4)) == 0).float().mean().item()


def build_model(options: TrainConfig):
    recon_cfg = options.recon

    anatomy_classes = options.label_classes
    num_anatomy_classes = len(anatomy_classes) if options.anatomy_weight > 0 else 0
    num_sampler_conditioning_classes = (
        len(anatomy_classes) if getattr(options, 'anatomy_conditioning', False) else 0
    )
    num_recon_conditioning_classes = (
        len(anatomy_classes)
        if getattr(recon_cfg, 'anatomy_conditioning', False) or getattr(recon_cfg, 'film_conditioning', False)
        else 0
    )

    if isinstance(recon_cfg, UNetReconConfig):
        reconstructor = LOUPEUNet(
            recon_cfg.in_chans,
            recon_cfg.out_chans,
            recon_cfg.num_chans,
            recon_cfg.num_pools,
            recon_cfg.drop_prob,
            with_uncertainty=options.uncertainty_loss,
        )
    elif isinstance(recon_cfg, VarNetReconConfig):
        reconstructor = VarNet(
            num_cascades=recon_cfg.num_cascades,
            chans=recon_cfg.chans,
            pools=recon_cfg.pools,
            num_conditioning_classes=num_recon_conditioning_classes,
            film=getattr(recon_cfg, 'film_conditioning', False),
        )
    elif isinstance(recon_cfg, CGSenseReconConfig):
        reconstructor = CGSENSEReconstructor(
            num_iters=recon_cfg.cg_iters,
            lamda=recon_cfg.cg_lamda,
        )
    else:
        raise ValueError(f"Unknown recon config type: {type(recon_cfg)}")

    model = SequentialSampling(
        reconstructor=reconstructor,
        sparsity=1.0 / options.accelerations[0],
        shape=options.resolution,
        num_step=options.num_step,
        preselect_num=options.preselect_num,
        sampler_type=_sampler_type(options),
        slope=getattr(options, 'slope', 10.0),
        with_uncertainty=options.uncertainty_loss,
        fixed_input=options.fixed_input,
        recon_type=recon_cfg.recon_type,
        sense_mask_method=options.sense_mask_method,
        with_recon=getattr(options, 'with_recon', False),
        with_gfactor=getattr(options, 'with_gfactor', False),
        gfactor_replicas=getattr(options, 'gfactor_replicas', 16),
        uncertainty_mode=getattr(options, 'uncertainty_mode', 'none'),
        num_anatomy_classes=num_anatomy_classes,
        num_conditioning_classes=num_sampler_conditioning_classes,
        condition_reconstructor=num_recon_conditioning_classes > 0,
        mask_train_mode=getattr(options, 'mask_train_mode', 'straight_through'),
        reinforce_weight=getattr(options, 'reinforce_weight', 1.0),
        reinforce_entropy_weight=getattr(options, 'reinforce_entropy_weight', 0.0),
        reinforce_normalize_advantages=getattr(options, 'reinforce_normalize_advantages', False),
        reinforce_group_size=getattr(options, 'reinforce_group_size', 0),
        reinforce_vd_power=getattr(options, 'reinforce_vd_power', 0.0),
        sampler_chans=getattr(options, 'sampler_chans', 64),
        sampler_depth=getattr(options, 'sampler_depth', 4),
        sampler_image_input=getattr(options, 'sampler_image_input', False),
        caipi_keep_fraction=getattr(options, 'caipi_keep_fraction', 1.0),
    )

    return model


class _AnatomyStats:
    """Per-anatomy metrics and mean sampling mask, accumulated over mixed batches.

    Batches from a ConcatDataset over several anatomies are split by the per-sample
    `anatomy` field so knee and brain patterns can be compared within one run.
    """

    def __init__(self, device):
        self.device = device
        self.metrics = {}
        self.mask_sums = {}
        self.counts = {}

    def update(self, anatomies, output, target, mask=None):
        for anatomy in dict.fromkeys(anatomies):
            indices = torch.tensor(
                [i for i, name in enumerate(anatomies) if name == anatomy],
                device=output.device,
            )
            output_a = output.index_select(0, indices)
            target_a = target.index_select(0, indices)

            if anatomy not in self.metrics:
                self.metrics[anatomy] = build_metrics(device=self.device)
            metrics = self.metrics[anatomy]
            data_range = target_a.max()
            metrics["PSNR"].data_range = data_range
            metrics["SSIM"].data_range = data_range
            metrics.update(output_a, target_a)

            if mask is not None:
                mask = mask.detach().float()
                if mask.shape[0] == 1:
                    mask_sum = mask[0] * len(indices)
                else:
                    mask_sum = mask.index_select(0, indices).sum(dim=0)
                if anatomy in self.mask_sums:
                    self.mask_sums[anatomy] = self.mask_sums[anatomy] + mask_sum
                else:
                    self.mask_sums[anatomy] = mask_sum
                self.counts[anatomy] = self.counts.get(anatomy, 0) + len(indices)

    def mean_masks(self):
        return {
            anatomy: (mask_sum / self.counts[anatomy]).cpu()
            for anatomy, mask_sum in self.mask_sums.items()
        }

    def compute(self, prefix):
        log = {}
        for anatomy, metrics in self.metrics.items():
            for name, value in metrics.compute().items():
                log[f'{prefix}/{anatomy}/{name}'] = value.item()
        for anatomy, mean_mask in self.mean_masks().items():
            log[f'{prefix}/{anatomy}/sparsity'] = mean_mask.mean().item()
            log[f'{prefix}/{anatomy}/mean_mask'] = wandb.Image(_normalize_for_wandb(mean_mask))
        return log

    def summary(self):
        parts = []
        for anatomy, metrics in self.metrics.items():
            values = ' '.join(f'{k}={v.item():.4g}' for k, v in metrics.compute().items())
            sparsity = self.mask_sums[anatomy].mean().item() / self.counts[anatomy] if anatomy in self.mask_sums else float('nan')
            parts.append(f'[{anatomy}] {values} Sparsity={sparsity:.4g} (n={self.counts.get(anatomy, 0)})')
        return ' | '.join(parts)


def _signal_mask(target, threshold=0.1):
    """Body mask from the target RSS, zeroing background so metrics are computed
    over the signal region only. Per-sample threshold as a fraction of max."""
    flat = target.flatten(1)
    max_val = flat.max(dim=1).values.view(-1, *([1] * (target.dim() - 1)))
    return (target > threshold * max_val).to(target.dtype)


def _apply_signal_mask(output, target):
    mask = _signal_mask(target)
    return output * mask, target * mask


def _normalize_for_wandb(tensor):
    t = tensor.float()
    if not torch.isfinite(t).all():
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t = t - t.min()
    denom = t.max()
    if denom > 0:
        t = t / denom
    return t * 255  # wandb.Image casts tensors straight to uint8, so [0, 1] would truncate to black


def _build_image_log(pred_dict, target, tag_prefix, data_for_vis=None, index=0):
    """Log images for sample `index` of the batch, so a mixed-anatomy batch can be
    logged once per anatomy. `data_for_vis` is only valid for index 0."""
    images = {}
    target_img = target[index] if target.dim() == 4 else target[index].unsqueeze(0)
    images[f'{tag_prefix}/target'] = wandb.Image(_normalize_for_wandb(target_img).cpu())

    def _pick(tensor):
        return tensor[index] if tensor.shape[0] > index else tensor[0]

    step_masks = pred_dict.get('masks', None)
    if step_masks is not None:
        for i, mask in enumerate(step_masks):
            images[f'{tag_prefix}/mask_step_{i}'] = wandb.Image(_normalize_for_wandb(_pick(mask)).cpu())

    output = pred_dict.get('intermediate_recons', pred_dict['output'])
    if isinstance(output, list):
        for i, recon in enumerate(output):
            images[f'{tag_prefix}/intermediate_recon_{i}'] = wandb.Image(_normalize_for_wandb(_pick(recon).abs()).cpu())
    else:
        images[f'{tag_prefix}/reconstruction'] = wandb.Image(_normalize_for_wandb(_pick(output)).cpu())
        if step_masks is None:
            images[f'{tag_prefix}/mask'] = wandb.Image(_normalize_for_wandb(_pick(pred_dict['mask'])).cpu())

    images[f'{tag_prefix}/output'] = wandb.Image(_normalize_for_wandb(_pick(pred_dict['output']).abs()).cpu())

    if data_for_vis is not None:
        step_data = data_for_vis[0]
        gfactor = step_data.get('gfactor')
        if gfactor is not None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(gfactor, cmap='inferno', vmin=0)
            ax.axis('off')
            ax.set_title(f'G-factor (max={gfactor.max():.1f})')
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            images[f'{tag_prefix}/gfactor'] = wandb.Image(fig)
            plt.close(fig)

        sampler_unc = step_data.get('sampler_uncertainty')
        if sampler_unc is not None:
            unc_np = sampler_unc.cpu().numpy() if hasattr(sampler_unc, 'cpu') else sampler_unc
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(unc_np, cmap='inferno')
            ax.axis('off')
            ax.set_title('Sampler Uncertainty')
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            images[f'{tag_prefix}/sampler_uncertainty'] = wandb.Image(fig)
            plt.close(fig)

        prob = step_data.get('prob_mask')
        if prob is not None:
            prob_np = prob.cpu().numpy() if hasattr(prob, 'cpu') else prob
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(prob_np, cmap='inferno')
            ax.axis('off')
            ax.set_title('Sampling Probability')
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            images[f'{tag_prefix}/prob_mask'] = wandb.Image(fig)
            plt.close(fig)

    return images
