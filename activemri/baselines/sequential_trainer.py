from collections import defaultdict
from pathlib import Path
from typing import List, Optional
import logging
import os
import random

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np

import time
import matplotlib.pyplot as plt

from activemri.utils.evaluate import build_metrics
from activemri.samplers.differential_sampler import SequentialUnet
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
        self.writer = SummaryWriter(log_dir=str(exp_dir / run_name))

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

        self.writer.close()

    def train_epoch(self):
        self.model.train()

        losses = []
        metrics = build_metrics(device=self.options.device)
        logged_train_anatomies = set()

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

            anatomy = batch['anatomy'][0]
            if anatomy not in logged_train_anatomies:
                logged_train_anatomies.add(anatomy)
                _log_images(self.writer, pred_dict, target, f'train/{anatomy}', global_step_offset)

            output = pred_dict['output']
            target_dict = {'target': target, 'kspace': kspace}
            meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': self.options.uncertainty_weight}
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
            metrics.update(output.detach(), target_for_metrics)

            start_iter = time.perf_counter()
            self.writer.add_scalar('Train_Loss', loss.item(), global_step_offset + iter_idx)

        metric_means = metrics.compute()
        for name, value in metric_means.items():
            self.writer.add_scalar(f'Train_{name}', value.item(), self.epoch)

        return np.mean(losses), time.perf_counter() - epoch_start_time, metric_means

    def val_epoch(self):
        self.model.eval()
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)
        start = time.perf_counter()
        logged_anatomies = set()

        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_loader):
                kspace = batch['kspace'].to(self.device)
                target = batch['target'].to(self.device)

                pred_dict = self.model(target, kspace)

                if iter_idx == 0:
                    data_for_vis_name = f'val_epoch={self.epoch + 1}'
                    self._visualize_and_save(self.epoch, data_for_vis_name)

                anatomy = batch['anatomy'][0]
                if anatomy not in logged_anatomies:
                    logged_anatomies.add(anatomy)
                    _log_images(self.writer, pred_dict, target, f'val/{anatomy}', self.epoch)

                output = pred_dict['output']
                if isinstance(output, list):
                    output = output[-1]

                target_dict = {'target': target, 'kspace': kspace}
                meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': self.options.uncertainty_weight}
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
                metrics.update(output, target_for_metrics)

            metric_means = metrics.compute()
            for name, value in metric_means.items():
                self.writer.add_scalar(f'Val_{name}', value.item(), self.epoch)
            self.writer.add_scalar('Val_Loss', np.mean(losses), self.epoch)

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

        self.writer = SummaryWriter(log_dir=str(self.exp_dir / self._build_run_name(self.options)))

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
                    _log_images(self.writer, pred_dict, target, 'test', iter_idx)

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
        self.writer.close()


def build_optimizer(lr, weight_decay, model_parameters, optim_type='adam'):
    if optim_type == 'adam':
        return torch.optim.Adam(model_parameters, lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        return torch.optim.SGD(model_parameters, lr, weight_decay=weight_decay)
    raise NotImplementedError(f"Unknown optimizer type: {optim_type}")


def build_lr_scheduler(options: TrainConfig, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, options.lr_step_size, options.lr_gamma)


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

    model = SequentialUnet(
        reconstructor=reconstructor,
        sparsity=1.0 / options.accelerations[0],
        shape=options.resolution,
        num_step=options.num_step,
        preselect_num=options.preselect_num,
        line_constrained=options.line_constrained,
        with_uncertainty=options.uncertainty_loss,
        fixed_input=options.fixed_input,
        recon_type=recon_cfg.recon_type,
    )

    return model


def _normalize_for_tb(tensor):
    t = tensor.float()
    t = t - t.min()
    denom = t.max()
    if denom > 0:
        t = t / denom
    return t


def _log_images(writer, pred_dict, target, tag_prefix, global_step):
    target_img = target[0] if target.dim() == 4 else target[0].unsqueeze(0)
    writer.add_image(f'{tag_prefix}/target', _normalize_for_tb(target_img), global_step)

    step_masks = pred_dict.get('masks', None)
    if step_masks is not None:
        for i, mask in enumerate(step_masks):
            writer.add_image(f'{tag_prefix}/mask_step_{i}', _normalize_for_tb(mask[0]), global_step)

    output = pred_dict.get('intermediate_recons', pred_dict['output'])
    if isinstance(output, list):
        for i, recon in enumerate(output):
            writer.add_image(f'{tag_prefix}/intermediate_recon_{i}', _normalize_for_tb(recon[0].abs()), global_step)
    else:
        writer.add_image(f'{tag_prefix}/reconstruction', _normalize_for_tb(output[0]), global_step)
        if step_masks is None:
            writer.add_image(f'{tag_prefix}/mask', _normalize_for_tb(pred_dict['mask'][0]), global_step)

    writer.add_image(f'{tag_prefix}/output', _normalize_for_tb(pred_dict['output'][0].abs()), global_step)
