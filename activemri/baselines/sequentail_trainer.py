
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
import activemri.envs.loupe_envs as loupe_envs
from activemri.samplers.differential_sampler import SequentialUnet
from .loupe import LOUPE
from ..envs.util import compute_gaussian_nll_loss, compute_ssim_torch, compute_psnr_torch
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.config import TrainConfig
from activemri.reconstructors.CG_Sense import CGSENSEReconstructor
from activemri.reconstructors.UNet import LOUPEUNet

class Sequential_Trainer :
    """Differentiabl Sampler Trainer for active MRI acquisition.

    Configuration for the trainer is provided by argument ``options``. Must contain the
    following fields:

    """
    def __init__(
        self,
        options: TrainConfig,
        env: loupe_envs.BuildDataLoaders,
        device: torch.device
    ):
        self.options = options
        self.env = env
        self.device = device
        self.model = build_model(self.options)
        if options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())

        self.train_loader, self.val_loader, self.display_loader, _ = self.env._setup_data_handlers()

        self.scheduler = build_lr_scheduler(self.options, self.optimizer)
        self.best_val_loss = np.inf
        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        # setup saving, writer, and logging
        options.exp_dir.mkdir(parents=True, exist_ok=True)
        options.visualization_dir.mkdir(parents=True, exist_ok=True)
        torch.save(options.__dict__, options.exp_dir / 'args.pt')  # save options as a dictionary for loading in non-rl tester

        self.writer = SummaryWriter(log_dir=options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def _visualize_and_save(self, epoch, data_for_vis_name):
        import pickle
        data_for_vis = self.model.data_for_vis
        for step in range(self.options.num_step):
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18, 6])
            cmap = "viridis" if self.options.resolution[1] == 32 else "gray"

            sp1 = ax[0, 0].imshow(data_for_vis[step]['input'], cmap=cmap)
            sp2 = ax[0, 1].imshow(np.log(np.abs(data_for_vis[step]['kspace'][0])))
            masked_k = data_for_vis[step]['masked_kspace']
            masked_k[masked_k == 0] = 1
            sp3 = ax[0, 2].imshow(np.log(np.abs(masked_k[0])))
            sp4 = ax[0, 3].imshow(data_for_vis[step]['zero_filled_recon'], cmap=cmap)
            sp5 = ax[0, 4].imshow(data_for_vis[step]['recon'], cmap=cmap)
            sp6 = ax[1, 0].imshow(data_for_vis[step]['prob_mask'].cpu().detach(), aspect='auto')
            sp7 = ax[1, 1].imshow(data_for_vis[step]['rescaled_mask'].cpu().detach(), aspect='auto')
            sp8 = ax[1, 2].imshow(data_for_vis[step]['binarized_mask'].cpu().detach(), aspect='auto')
            sp9 = ax[1, 3].imshow(data_for_vis[step]['sense_mask'], cmap='gray', vmin=0, vmax=1)
            sp10 = ax[1, 4].imshow(np.exp(data_for_vis[step]['uncertainty_map']), cmap=cmap)

            ax[0, 0].title.set_text('Input image')
            ax[0, 1].title.set_text('Log k-space of the input')
            ax[0, 2].title.set_text('Undersampled log k-space')
            ax[0, 4].title.set_text('Reconstruction')
            ax[1, 0].title.set_text('Probabilistic mask')
            ax[1, 1].title.set_text('Rescaled mask')
            ax[1, 2].title.set_text('Binary mask')
            ax[1, 3].title.set_text('Sense body mask')
            ax[1, 4].title.set_text('Uncertainty of the reconstruction')

            fig.colorbar(sp1, ax=ax[0, 0])
            fig.colorbar(sp2, ax=ax[0, 1])
            fig.colorbar(sp3, ax=ax[0, 2])
            fig.colorbar(sp4, ax=ax[0, 3])
            fig.colorbar(sp5, ax=ax[0, 4])
            fig.colorbar(sp6, ax=ax[1, 0])
            fig.colorbar(sp7, ax=ax[1, 1])
            fig.colorbar(sp8, ax=ax[1, 2])
            fig.colorbar(sp9, ax=ax[1, 3])
            fig.colorbar(sp10, ax=ax[1, 4])

            plt.suptitle(f'Epoch = [{1 + epoch}/{self.options.num_epochs}] Step = [{step + 1}/{self.options.num_step}]', fontsize=20)
            self.options.visualization_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(self.options.visualization_dir) + '/' + data_for_vis_name + f'_step{step}.png')
            plt.close()

            path = str(self.options.visualization_dir) + '/' + data_for_vis_name + f'_step{step}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(data_for_vis[step], f)

    def __call__(self):
        if self.options.resume:
            self.load_checkpoint()
        return self._train_loupe()
        
    def _train_loupe(self):
        history = defaultdict(list)

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.scheduler.step(epoch)

            # train, val loop
            train_loss, train_time, train_metrics = self.train_epoch()
            val_loss, mean_sparsity, val_time, val_metrics = self.val_epoch()

            # add metrics to history and log
            for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
                for key, val in metrics.items():
                    history[f'{prefix}_{key.lower()}'].append(val.item() if hasattr(val, 'item') else val)
            self._plot_metrics(history)

            # save checkpoint
            is_new_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if self.options.save_model:
                self.save_model(is_new_best)

            # print log
            self.logger.info(
                f'Epoch = [{1 + self.epoch:4d}/{self.options.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} MeanSparsity = {mean_sparsity:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )
        
        self.writer.close()

    def train_epoch(self):
        self.model.train()

        losses = []
        metrics = build_metrics(device=self.options.device)

        epoch_start_time = start_iter = time.perf_counter()

        global_step_offset = self.epoch * len(self.train_loader)

        max_batches = max(1, int(getattr(self.options, 'limit_batches', 1.0) * len(self.train_loader))) # limit batches for faster training if specified
        for iter, data in enumerate(self.train_loader):
            if iter >= max_batches:
                break

            kspace, _, target, label, *ignored = data

            # adapt data to loupe
            target = target.to(self.options.device)
            kspace = kspace.to(self.options.device)


            pred_dict = self.model(target, kspace)

            if (self.epoch == 0 or (self.epoch+1) % 1 == 0) and iter ==  0:
                data_for_vis_name = 'train_epoch={}_iter={}'.format(str(self.epoch+1), str(iter+1))
                self._visualize_and_save(self.epoch, data_for_vis_name)
                _log_images(self.writer, pred_dict, target, 'train', global_step_offset)

            output = pred_dict['output']
            target_dict = {'target': target, 'label': label, 'kspace': kspace}
            meta = {'entropy_weight': self.options.entropy_weight, 'recon_weight': self.options.recon_weight,
                'kspace_weight': self.options.kspace_weight,
                'uncertainty_weight': self.options.uncertainty_weight if 'uncertainty_weight' in self.options.__dict__ else 0}

            loss_fn = self.model.loss if hasattr(self.model, 'loss') else self.loss
            loss, log_dict = loss_fn(pred_dict, target_dict, meta, self.options.loss_type)

            # training loop
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

            if iter % self.options.report_interval == 0:
                self.logger.info(
                    f'Epoch = [{1 + self.epoch:3d}/{self.options.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(self.train_loader):4d}] '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
                for key, val in log_dict.items():
                    print('{} = {}'.format(key, val))

            start_iter = time.perf_counter()
            self.writer.add_scalar('Train_Loss', loss.item(), global_step_offset + iter)

        metric_means = metrics.compute()
        for name, value in metric_means.items():
            self.writer.add_scalar(f'Train_{name}', value.item(), self.epoch)
        print(' '.join(f'{k} = {v:.4g}' for k, v in metric_means.items()))

        return np.mean(np.array(losses)), time.perf_counter() - epoch_start_time, metric_means


    def val_epoch(self):
        self.model.eval()
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)
        start = time.perf_counter()

        with torch.no_grad():
            for iter, data in enumerate(self.val_loader):
                kspace, _, input, label, *ignored = data

                # adapt data to loupe
                target = input.float().unsqueeze(1)

                input = input.to(self.options.device)
                target = target.to(self.options.device)
                kspace = kspace.to(self.options.device)

                pred_dict = self.model(target, kspace)

                if (self.epoch == 0 or (self.epoch+1) % 1 == 0) and iter == 0:
                    data_for_vis_name = 'eval_epoch=' + str(self.epoch+1)
                    self._visualize_and_save(self.epoch, data_for_vis_name)
                    _log_images(self.writer, pred_dict, target, 'eval', self.epoch)

                output = pred_dict['output']
                if isinstance(output, list):
                    output = output[-1]

                target_dict = {'target': target, 'label': label, 'kspace':kspace}
                meta = {'entropy_weight': self.options.entropy_weight, 'recon_weight': self.options.recon_weight,
                'uncertainty_weight': 0, 'kspace_weight': self.options.kspace_weight}

                loss_fn = self.model.loss if hasattr(self.model, 'loss') else self.loss
                loss, log_dict = loss_fn(pred_dict, target_dict, meta, self.options.loss_type)

                mask = pred_dict['mask']
                sparsity.append(torch.mean(mask).item())
                losses.append(loss.item())

                data_range = target.max()
                metrics["PSNR"].data_range = data_range
                metrics["SSIM"].data_range = data_range
                metrics.update(output, target)

            metric_means = metrics.compute()
            for name, value in metric_means.items():
                self.writer.add_scalar(f'Val_{name}', value.item(), self.epoch)
            self.writer.add_scalar('Val_Loss', np.mean(losses), self.epoch)
            print(' '.join(f'{k} = {v:.4g}' for k, v in metric_means.items()))

        return np.mean(losses), np.mean(sparsity), time.perf_counter() - start, metric_means

    def _plot_metrics(self, history):
        epochs = list(range(1, len(history['train_ssim']) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, metric in zip(axes, ['ssim', 'nmse', 'psnr']):
            ax.plot(epochs, history[f'train_{metric}'], label='Train')
            ax.plot(epochs, history[f'val_{metric}'], label='Val')
            ax.set_xlabel('Epoch')
            ax.set_title(metric.upper())
            ax.legend()

        fig.tight_layout()
        fig.savefig(self.options.visualization_dir / 'metrics.png', dpi=150)
        plt.close(fig)


    def loss(self, pred_dict, target_dict, meta, loss_type):
        """
        Args:
            pred_dict:
                output: reconstructed image from downsampled kspace measurement
                energy: negative entropy of the probability mask
                mask: the binazried sampling mask (used for visualization)

            target_dict:
                target: original fully sampled image

            meta:
                recon_weight: weight of reconstruction loss
                entropy_weight: weight of the entropy loss (to encourage exploration)
        """
        target = target_dict['target']
        pred = pred_dict['output']
        energy = pred_dict['energy']
        
        if loss_type == 'l1':
            reconstruction_loss = F.l1_loss(pred, target, size_average=True) 
        elif loss_type == 'ssim':
            reconstruction_loss = -torch.mean(compute_ssim_torch(pred, target))
        elif loss_type == 'psnr':
            reconstruction_loss = - torch.mean(compute_psnr_torch(pred, target))
        else:
            raise NotImplementedError

        entropy_loss = torch.mean(energy)

        loss = entropy_loss * meta['entropy_weight'] + reconstruction_loss * meta['recon_weight']

        log_dict = {'Total Loss': loss.item(), 'Entropy': entropy_loss.item(), 'Reconstruction': reconstruction_loss.item()}

        return loss, log_dict


    def save_model(self, is_new_best):
        exp_dir: Path = self.options.exp_dir
        checkpoint = {
            'epoch': self.epoch,
            'options': self.options,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'exp_dir': exp_dir
        }
        epoch_path = exp_dir / f'model_epoch_{self.epoch:03d}.pt'
        exp_dir.glob('model_epoch_*.pt')  # remove old checkpoints
        torch.save(checkpoint, f=epoch_path)
        if is_new_best:
            epoch_path.rename(exp_dir / 'best_model.pt')


    def load_checkpoint(self):
        self.model = build_model(self.options)
        if self.options.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint1 = torch.load(self.options.checkpoint1)
        print("Load checkpoint {} with loss {}".format(checkpoint1['epoch'], checkpoint1['best_val_loss']))

        self.model.load_state_dict(checkpoint1['model'])

        self.optimizer = build_optimizer(self.options.lr, self.options.weight_decay, self.model.parameters())
        self.optimizer.load_state_dict(checkpoint1['optimizer'])

        self.best_val_loss = checkpoint1['best_val_loss']
        self.start_epoch = checkpoint1['epoch'] + 1


class NonRLTester:
    def __init__(
        self,
        env: loupe_envs.BuildDataLoaders,
        exp_dir: Path,
        options: TrainConfig,
        label_range: Optional[List[int]]
    ):
        self.env = env
        # load options and model
        checkpoint_file = options.checkpoint1 if options.checkpoint1 else os.path.join(exp_dir, 'best_model.pt')
        self.load(checkpoint_file)
        self.options.label_range = label_range
        self.options.exp_dir = exp_dir
        self.options.test_visual_frequency = options.test_visual_frequency
        self.options.visualization_dir = options.visualization_dir
        self.options.batch_size = 1
        _, _, self.val_loader, _  = self.env._setup_data_handlers()

        # setup saving and logging
        self.options.exp_dir.mkdir(parents=True, exist_ok=True)
        self.options.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.options.exp_dir / 'summary')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.options)
        self.logger.info(self.model)

    def _visualize_and_save(self, epoch, data_for_vis_name):
        import pickle
        data_for_vis = self.model.data_for_vis
        for step in range(self.options.num_step):
            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18, 6])
            cmap = "viridis" if self.options.resolution[1] == 32 else "gray"

            sp1 = ax[0, 0].imshow(data_for_vis[step]['input'], cmap=cmap)
            sp2 = ax[0, 1].imshow(np.log(np.abs(data_for_vis[step]['kspace'][0])))
            masked_k = data_for_vis[step]['masked_kspace']
            masked_k[masked_k == 0] = 1
            sp3 = ax[0, 2].imshow(np.log(np.abs(masked_k[0])))
            sp4 = ax[0, 3].imshow(data_for_vis[step]['zero_filled_recon'], cmap=cmap)
            sp5 = ax[0, 4].imshow(data_for_vis[step]['recon'], cmap=cmap)
            sp6 = ax[1, 0].imshow(data_for_vis[step]['prob_mask'].cpu().detach(), aspect='auto')
            sp7 = ax[1, 1].imshow(data_for_vis[step]['rescaled_mask'].cpu().detach(), aspect='auto')
            sp8 = ax[1, 2].imshow(data_for_vis[step]['binarized_mask'].cpu().detach(), aspect='auto')
            sp9 = ax[1, 3].imshow(data_for_vis[step]['sense_mask'], cmap='gray', vmin=0, vmax=1)
            sp10 = ax[1, 4].imshow(np.exp(data_for_vis[step]['uncertainty_map']), cmap=cmap)

            ax[0, 0].title.set_text('Input image')
            ax[0, 1].title.set_text('Log k-space of the input')
            ax[0, 2].title.set_text('Undersampled log k-space')
            ax[0, 4].title.set_text('Reconstruction')
            ax[1, 0].title.set_text('Probabilistic mask')
            ax[1, 1].title.set_text('Rescaled mask')
            ax[1, 2].title.set_text('Binary mask')
            ax[1, 3].title.set_text('Sense body mask')
            ax[1, 4].title.set_text('Uncertainty of the reconstruction')

            fig.colorbar(sp1, ax=ax[0, 0])
            fig.colorbar(sp2, ax=ax[0, 1])
            fig.colorbar(sp3, ax=ax[0, 2])
            fig.colorbar(sp4, ax=ax[0, 3])
            fig.colorbar(sp5, ax=ax[0, 4])
            fig.colorbar(sp6, ax=ax[1, 0])
            fig.colorbar(sp7, ax=ax[1, 1])
            fig.colorbar(sp8, ax=ax[1, 2])
            fig.colorbar(sp9, ax=ax[1, 3])
            fig.colorbar(sp10, ax=ax[1, 4])

            plt.suptitle(f'Epoch = [{1 + epoch}/{self.options.num_epochs}] Step = [{step + 1}/{self.options.num_step}]', fontsize=20)
            self.options.visualization_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(self.options.visualization_dir) + '/' + data_for_vis_name + f'_step{step}.png')
            plt.close()

            path = str(self.options.visualization_dir) + '/' + data_for_vis_name + f'_step{step}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(data_for_vis[step], f)

    def evaluate(self):
        self.model.eval()
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)

        with torch.no_grad():
            for iter, data in enumerate(self.val_loader):
                kspace, _, input, label, *ignored = data

                # adapt data to loupe
                target = input.float().unsqueeze(1)

                input = input.to(self.options.device)
                target = target.to(self.options.device)
                kspace = kspace.to(self.options.device)

                pred_dict = self.model(target, kspace)

                if iter % self.options.test_visual_frequency == 0:
                    data_for_vis_name = 'test_iter=' + str(iter+1)
                    print("visualize {}".format(data_for_vis_name))
                    self._visualize_and_save(iter, data_for_vis_name)
                    _log_images(self.writer, pred_dict, target, 'test', iter)

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

        print("Done Prediction")

        return losses, np.mean(sparsity), metrics.compute()

    def __call__(self):
        val_loss, sparsity, metric_results = self.evaluate()

        print("L1 Loss {} STD {} Sparsity {}".format(np.mean(val_loss), np.std(val_loss), sparsity))

        metrics_str = ' '.join(f'{k} = {v:.4g}' for k, v in metric_results.items())
        with open(os.path.join(self.options.exp_dir, 'statistics.txt'), 'w') as f:
            f.write('L1 Loss {} +- {}\n'.format(np.mean(val_loss), np.std(val_loss)))
            f.write(metrics_str)

        print(metrics_str)

        torch.save(val_loss, os.path.join(self.options.exp_dir, 'loss.pt'))
        self.writer.close()

    def load(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        print("Load checkpoint {} with loss {}".format(checkpoint['epoch'], checkpoint['best_val_loss']))
        self.options = checkpoint['options']
        random.seed(self.options.seed)
        np.random.seed(self.options.seed)
        torch.manual_seed(self.options.seed)


        if 'bi_dir' not in self.options.__dict__:
            self.options.bi_dir = False

        if 'clamp' not in self.options.__dict__:
            self.options.clamp = 100

        if 'fixed_input' not in self.options.__dict__:
            self.options.fixed_input = False

        print('clamp value {}'.format(self.options.clamp))

        self.model = build_model(self.options)
        self.model.eval()
        self.model.load_state_dict(checkpoint['model'])

        torch.save({'model':self.model.reconstructor.state_dict()}, os.path.join(*checkpoint_file.split('/')[:-1], 'best_recon.pt'))

        # checkpoint = torch.load('checkpoints/Reconstructors/dicom_ckpt/PG_DICOM_Knee_Reconstructor_UNet/best_model.pt', map_location='cpu')
        # self.model.reconstructor.load_state_dict(checkpoint['model'])

def build_optimizer(lr, weight_decay, model_parameters, type='adam'):
    if type == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr, weight_decay=weight_decay)
    elif type == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()

    return optimizer

def build_lr_scheduler(options, optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.lr_step_size, options.lr_gamma)
    return scheduler

def build_model(options):
    if options.model == 'LOUPE':
        model = LOUPE(
            in_chans=options.input_chans,
            out_chans=options.output_chans,
            chans=options.num_chans,
            num_pool_layers=options.num_pools,
            drop_prob=options.drop_prob,
            sparsity=1.0/options.accelerations[0],
            shape=options.resolution,
            conjugate_mask=options.conjugate_mask,
            line_constrained=options.line_constrained,
            bi_dir=options.bi_dir,
            preselect_num=options.preselect_num,
            preselect=options.preselect,
            random=options.random if 'random' in options.__dict__ else False,
            poisson=options.poisson if 'poisson' in options.__dict__ else False,
            spectrum=options.spectrum if 'spectrum' in options.__dict__ else False,
            equispaced=options.equispaced if 'equispaced' in options.__dict__ else False,).to(options.device)
    elif options.model == 'SequentialSampling':
        recon_type = options.recon_type if 'recon_type' in options.__dict__ else 'unet'
        if recon_type == 'unet':
            reconstructor = LOUPEUNet(
                options.input_chans, options.output_chans, options.num_chans,
                options.num_pools, options.drop_prob,
                with_uncertainty=options.uncertainty_loss,
            )
        elif recon_type == 'lls':
            cg_iters = options.cg_iters if 'cg_iters' in options.__dict__ else 10
            cg_lamda = options.cg_lamda if 'cg_lamda' in options.__dict__ else 0.0
            reconstructor = CGSENSEReconstructor(num_iters=cg_iters, lamda=cg_lamda)
        else:
            raise ValueError(f"Unknown recon_type '{recon_type}'. Choose 'unet' or 'lls'.")

        pretrained_recon = options.pretrained_recon if 'pretrained_recon' in options.__dict__ else False
        if pretrained_recon:
            sparsity = 1.0 / options.accelerations[0]
            if sparsity != 0.25:
                raise NotImplementedError("Pretrained recon only available for sparsity=0.25")
            path = 'reconstructors/4x_random_recon.pt'
            print(f"Use Pretrained Reconstructor from {path} and freeze Reconstructor Weights")
            reconstructor.load_state_dict(torch.load(path, map_location='cpu')['model'])
            reconstructor.requires_grad_(False)
            reconstructor.eval()

        model = SequentialUnet(
            reconstructor=reconstructor,
            sparsity=1.0/options.accelerations[0],
            shape=options.resolution,
            num_step=options.num_step,
            preselect=options.preselect,
            bi_direction=options.bi_dir,
            preselect_num=options.preselect_num,
            binary_sampler=options.binary_sampler,
            clamp=options.clamp,
            line_constrained=options.line_constrained,
            with_uncertainty=options.uncertainty_loss,
            detach_kspace=options.detach_kspace,
            fixed_input=options.fixed_input,
            recon_type=recon_type,
            sense_tikhonov_weight=options.sense_tikhonov_weight if 'sense_tikhonov_weight' in options.__dict__ else 0.0,
        ).to(options.device)
    else:
        raise NotImplementedError()

    return model

def get_mask_stats(masks):
    masks = np.array(masks)

    return np.mean(masks, axis=0), np.std(masks, axis=0)


def _normalize_for_tb(tensor):
    """Normalize a CHW tensor to [0, 1] for tensorboard."""
    t = tensor.float()
    t = t - t.min()
    denom = t.max()
    if denom > 0:
        t = t / denom
    return t


def _log_images(writer, pred_dict, target, tag_prefix, global_step):
    """Log ground truth, per-step reconstructions, and masks to tensorboard."""
    output = pred_dict['intermediate_recons'] if 'intermediate_recons' in pred_dict else pred_dict['output']
    target_img = target[0] if target.dim() == 4 else target[0].unsqueeze(0)
    writer.add_image(f'{tag_prefix}/target', _normalize_for_tb(target_img), global_step)

    step_masks = pred_dict.get('masks', None)
    if step_masks is not None:
        for i, mask in enumerate(step_masks):
            writer.add_image(f'{tag_prefix}/mask_step_{i}', _normalize_for_tb(mask[0]), global_step)

    if isinstance(output, list):
        for i, recon in enumerate(output):
            writer.add_image(f'{tag_prefix}/intermediate_recon_{i}', _normalize_for_tb(recon[0].abs()), global_step)

    else:
        writer.add_image(f'{tag_prefix}/reconstruction', _normalize_for_tb(output[0]), global_step)
        if step_masks is None:
            writer.add_image(f'{tag_prefix}/mask', _normalize_for_tb(pred_dict['mask'][0]), global_step)

    writer.add_image(f'{tag_prefix}/output', _normalize_for_tb(pred_dict['output'][0].abs()), global_step)
