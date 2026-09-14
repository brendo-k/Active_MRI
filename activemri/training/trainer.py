from collections import defaultdict
from pathlib import Path
from typing import List, Optional
import logging
import os
import random

import torch
import torch.nn.functional as F
import wandb
import numpy as np

import time
import matplotlib.pyplot as plt

from activemri.utils.evaluate import build_metrics
from activemri.models.sequential_sampling import SequentialSampling
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
        self.anatomy_to_idx = {name: i for i, name in enumerate(options.anatomy_classes)} if options.anatomy_classes else {}
        self.model = build_model(self.options).to(device)
        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = build_optimizer(options.lr, options.weight_decay, self.model.parameters())
        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader is not None else train_loader

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

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    @staticmethod
    def _build_run_name(options: TrainConfig) -> str:
        recon_type = options.recon.recon_type
        res = f"{options.resolution[0]}x{options.resolution[1]}"
        accel = 'x'.join(str(int(a)) if a == int(a) else str(a) for a in options.accelerations)
        steps = options.num_step
        bs = options.batch_size
        return f"{recon_type}_{res}_{accel}acc_step{steps}_bs{bs}"

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

            kspace = batch['kspace'].to(self.device)
            target = batch['target'].to(self.device)

            pred_dict = self.model(target, kspace)

            if iter_idx == 0:
                data_for_vis_name = f'train_epoch={self.epoch + 1}_iter=1'
                self._visualize_and_save(self.epoch, data_for_vis_name)

            anatomies = list(batch['anatomy'])
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
            if self.anatomy_to_idx:
                target_dict['anatomy_labels'] = torch.tensor(
                    [self.anatomy_to_idx[a] for a in anatomies], device=self.device,
                )
            meta = {
                'recon_weight': 1.0, 'kspace_weight': 0.0,
                'uncertainty_weight': self.options.uncertainty_weight,
                'anatomy_weight': self.options.anatomy_weight,
            }
            loss, log_dict = self.model.loss(pred_dict, target_dict, meta, self.options.loss_type)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
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

            step_log = {'train/loss': loss.item(), 'train/step': global_step_offset + iter_idx}
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

                pred_dict = self.model(target, kspace)

                if iter_idx == 0:
                    data_for_vis_name = f'val_epoch={self.epoch + 1}'
                    self._visualize_and_save(self.epoch, data_for_vis_name)

                anatomies = list(batch['anatomy'])
                for anatomy in dict.fromkeys(anatomies):
                    if anatomy in logged_anatomies:
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
                if self.anatomy_to_idx:
                    target_dict['anatomy_labels'] = torch.tensor(
                        [self.anatomy_to_idx[a] for a in anatomies], device=self.device,
                    )
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

            metric_means = metrics.compute()
            epoch_log = {f'val/{name}': value.item() for name, value in metric_means.items()}
            epoch_log['val/loss'] = np.mean(losses)
            epoch_log['epoch'] = self.epoch
            epoch_log.update(anatomy_images)
            epoch_log.update(anatomy_stats.compute('val'))
            wandb.log(epoch_log)

            anatomy_stats.save_mean_masks(self.vis_dir, f'val_epoch={self.epoch + 1}')
            self.logger.info(anatomy_stats.summary())

        return np.mean(losses), np.mean(sparsity), time.perf_counter() - start, metric_means

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
        epoch_path = self.exp_dir / f'model_epoch_{self.epoch:03d}.pt'
        torch.save(checkpoint, f=epoch_path)
        if is_new_best:
            best_path = self.exp_dir / 'best_model.pt'
            torch.save(checkpoint, f=best_path)

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
        import pickle
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

            with open(self.vis_dir / f'{data_for_vis_name}_step{step}.pkl', 'wb') as f:
                pickle.dump(data_for_vis[step], f)


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
        import pickle
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

            with open(self.vis_dir / f'{data_for_vis_name}_step{step}.pkl', 'wb') as f:
                pickle.dump(data_for_vis[step], f)

    def evaluate(self):
        self.model.eval()
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)

        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_loader):
                kspace = batch['kspace'].to(self.device)
                target = batch['target'].to(self.device)

                pred_dict = self.model(target, kspace)

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


def build_model(options: TrainConfig):
    recon_cfg = options.recon

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
        )
    elif isinstance(recon_cfg, CGSenseReconConfig):
        reconstructor = CGSENSEReconstructor(
            num_iters=recon_cfg.cg_iters,
            lamda=recon_cfg.cg_lamda,
        )
    else:
        raise ValueError(f"Unknown recon config type: {type(recon_cfg)}")

    num_anatomy_classes = len(options.anatomy_classes) if options.anatomy_classes else 0

    model = SequentialSampling(
        reconstructor=reconstructor,
        sparsity=1.0 / options.accelerations[0],
        shape=options.resolution,
        num_step=options.num_step,
        preselect_num=options.preselect_num,
        sampler_type=_sampler_type(options),
        slope=getattr(options, 'slope', 5.0),
        with_uncertainty=options.uncertainty_loss,
        fixed_input=options.fixed_input,
        recon_type=recon_cfg.recon_type,
        sense_mask_method=options.sense_mask_method,
        with_recon=getattr(options, 'with_recon', False),
        with_gfactor=getattr(options, 'with_gfactor', False),
        gfactor_replicas=getattr(options, 'gfactor_replicas', 16),
        uncertainty_mode=getattr(options, 'uncertainty_mode', 'none'),
        num_anatomy_classes=num_anatomy_classes,
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
            log[f'{prefix}/{anatomy}/mean_mask'] = wandb.Image(_normalize_for_tb(mean_mask))
        return log

    def save_mean_masks(self, out_dir, name):
        for anatomy, mean_mask in self.mean_masks().items():
            np.save(Path(out_dir) / f'{name}_{anatomy}_mean_mask.npy', mean_mask.numpy())

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


def _normalize_for_tb(tensor):
    t = tensor.float()
    t = t - t.min()
    denom = t.max()
    if denom > 0:
        t = t / denom
    return t


def _build_image_log(pred_dict, target, tag_prefix, data_for_vis=None, index=0):
    """Log images for sample `index` of the batch, so a mixed-anatomy batch can be
    logged once per anatomy. `data_for_vis` is only valid for index 0."""
    images = {}
    target_img = target[index] if target.dim() == 4 else target[index].unsqueeze(0)
    images[f'{tag_prefix}/target'] = wandb.Image(_normalize_for_tb(target_img).cpu())

    def _pick(tensor):
        return tensor[index] if tensor.shape[0] > index else tensor[0]

    step_masks = pred_dict.get('masks', None)
    if step_masks is not None:
        for i, mask in enumerate(step_masks):
            images[f'{tag_prefix}/mask_step_{i}'] = wandb.Image(_normalize_for_tb(_pick(mask)).cpu())

    output = pred_dict.get('intermediate_recons', pred_dict['output'])
    if isinstance(output, list):
        for i, recon in enumerate(output):
            images[f'{tag_prefix}/intermediate_recon_{i}'] = wandb.Image(_normalize_for_tb(_pick(recon).abs()).cpu())
    else:
        images[f'{tag_prefix}/reconstruction'] = wandb.Image(_normalize_for_tb(_pick(output)).cpu())
        if step_masks is None:
            images[f'{tag_prefix}/mask'] = wandb.Image(_normalize_for_tb(_pick(pred_dict['mask'])).cpu())

    images[f'{tag_prefix}/output'] = wandb.Image(_normalize_for_tb(_pick(pred_dict['output']).abs()).cpu())

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
