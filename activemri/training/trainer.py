from collections import defaultdict
from typing import Dict, List, Optional
import gc
import logging
import math
import os
import pathlib
import random
import sys
import warnings

import torch
import torch._inductor.config
import torch.nn.functional as F
import wandb
import numpy as np

import time
import matplotlib.pyplot as plt

from activemri.utils.compilers import allow_recompiles, use_system_compilers_if_missing
from activemri.utils.evaluate import build_metrics
from activemri.utils.metrics import MaskedMetricAccumulator
from activemri.utils.phase_scramble import scramble_phase_real
from activemri.utils.metric_mask import metric_mask_from_real_kspace
from activemri.utils.multistep import StepRecorder, step_grid_figure, step_panels
from activemri.samplers.seq_adapt import SequentialAdaptiveMask
from activemri.models.sequential_sampling import SequentialSampling, reinforce_group_size
from activemri.samplers.autosamp import annealed_kernel_sigma
from activemri.samplers.loupe import LineConstrainedProbMask, ProbMask
from activemri.utils.util import compute_ssim_torch, compute_psnr_torch
from activemri.config import TrainConfig
from activemri.training.run_naming import run_group, run_name, run_tags
from activemri.training.ema import ParameterEMA

# Checkpoints pickled under Python 3.13 name `pathlib._local`, which Python 3.14 no longer has.
try:
    import pathlib._local  # noqa: F401
except ModuleNotFoundError:
    sys.modules['pathlib._local'] = pathlib
from activemri.training.muon import MuonAdam, muon_parameters
from activemri.reconstructors.norms import replace_instance_norms
from activemri.reconstructors.config import UNetReconConfig, CGSenseReconConfig, SDUMReconConfig, VarNetReconConfig
from activemri.reconstructors.CG_Sense import CGSENSEReconstructor
from activemri.reconstructors.UNet import LOUPEUNet
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.reconstructors.VarNet import NormUnet, VarNet
from activemri.reconstructors.sdum import SDUMRecon

PHASE_SCRAMBLE_VAL_SEED = 0
REINFORCE_VAL_MASK_SEED = 0
HEAD_MLP_PARAMETERS = ('.location_head.mlp.', '.encoder_head.')  # trained at sampler_head_mlp_lr


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
        self.val_phase_generator = torch.Generator(device=device)
        self.model = build_model(self.options).to(device)
        self.val_mask_generator = torch.Generator(device=device)
        self.model.sampler.eval_generator = self.val_mask_generator
        self.metric_mask_cache = {}  # (anatomy, fname, slice_id) -> [1, H, W] bool, filled by _metric_mask

        if options.pretrained_recon is not None:
            self._load_pretrained_recon(options.pretrained_recon)
        if getattr(options, 'sampler_input', 'measured') == 'recon':
            self._load_input_reconstructor(getattr(options, 'input_recon_checkpoint', None))
        if options.freeze_recon:
            self._freeze_recon()
        if options.fixed_mask_checkpoints:
            self._load_fixed_masks(options.fixed_mask_checkpoints)
        if getattr(options, 'init_mask_checkpoint', None):
            if getattr(options, 'loupe_vd_init_power', 0.0) > 0:
                raise ValueError("init_mask_checkpoint and loupe_vd_init_power both set the starting mask; set one")
            self._init_mask_from(options.init_mask_checkpoint)

        if options.compile_unets:
            channels_last = getattr(options, 'channels_last_unets', False)
            if channels_last is True:
                torch._inductor.config.force_layout_optimization = True
            compile_unets(self.model, channels_last)

        if options.data_parallel:
            if options.sampler_type == 'autosamp_2d':
                raise ValueError("autosamp_2d projects its positions in place during forward, which DataParallel's replicas can't do")
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = build_optimizer(options, self._param_groups(), _unwrapped(self.model))
        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader is not None else train_loader
        self.visual_examples = select_visual_examples(self.val_loader.dataset)

        self.scheduler = build_lr_scheduler(options, self.optimizer)
        ema_decay = getattr(options, 'ema_decay', 0.0)
        self.ema = None if ema_decay <= 0 else ParameterEMA(
            [p for module in self._recon_modules() for p in module.parameters() if p.requires_grad], ema_decay)
        self.best_val_loss = np.inf
        self.validations_since_best = 0
        self.previous_mask_logits = None
        self.epoch = 0
        self.start_epoch = 0
        self.end_epoch = options.num_epochs

        exp_dir = options.out_dir[0]
        exp_dir.mkdir(parents=True, exist_ok=True)

        vis_dir = exp_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir
        self.vis_dir = vis_dir

        wandb.init(project="SeqMRI", name=run_name(options), group=run_group(options), tags=run_tags(options),
                   notes=str(exp_dir), job_type='train', config=vars(options), dir=str(exp_dir))

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
        missing = [k for k in missing if not k.startswith(('sampler.', 'input_reconstructor.', 'input_sensetivity_net.'))]
        if missing or unexpected:
            raise RuntimeError(
                f"Pretrained recon checkpoint {checkpoint_path} did not match the model's "
                f"reconstructor/sensetivity_net: missing={missing}, unexpected={unexpected}"
            )
        self.logger.info(f"Loaded pretrained reconstructor weights from {checkpoint_path}")

    def _load_input_reconstructor(self, checkpoint_path: Optional[str]):
        """Load a prior run's reconstructor and sensitivity net into the frozen copies that make seq_adapt_2d's recon
        input (sampler.input=recon)."""
        if checkpoint_path is None:
            raise ValueError("sampler.input=recon needs sampler.input_recon_checkpoint, the run whose reconstructor "
                             "makes the sampler's input")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
        frozen_state = {f'input_{k}': v for k, v in state_dict.items()
                        if k.startswith('reconstructor.') or k.startswith('sensetivity_net.')}
        model = _unwrapped(self.model)
        missing, unexpected = model.load_state_dict(frozen_state, strict=False)
        missing = [k for k in missing if k.startswith(('input_reconstructor.', 'input_sensetivity_net.'))]
        if missing or unexpected:
            raise RuntimeError(f"{checkpoint_path} does not fit the frozen input reconstructor: missing={missing}, "
                               f"unexpected={unexpected}")
        self.logger.info(f"Loaded the frozen input reconstructor from {checkpoint_path}")

    def _load_fixed_masks(self, checkpoints: Dict[str, str]):
        """Copy each anatomy's learned LOUPE mask from its own checkpoint into the
        conditioned sampler and freeze the sampler, so only the reconstructor trains."""
        options = self.options
        if options.sampler_type != 'loupe_2d' or not options.anatomy_conditioning:
            raise ValueError("fixed_mask_checkpoints needs sampler_type=loupe_2d with anatomy_conditioning=true")
        if set(checkpoints) != set(options.label_classes):
            raise ValueError(f"fixed_mask_checkpoints keys {sorted(checkpoints)} must match anatomies {options.label_classes}")
        mask = self.model.sampler.mask_net.mask
        for anatomy, path in checkpoints.items():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['options'].slope != options.slope:
                raise ValueError(f"{path} was trained with slope {checkpoint['options'].slope}, not {options.slope}")
            logits = checkpoint['model']['sampler.mask_net.mask']
            if logits.shape[0] != 1:
                raise ValueError(f"{path} holds {logits.shape[0]} masks; expected one learned mask")
            with torch.no_grad():
                mask[self.anatomy_to_idx[anatomy]] = logits[0].to(mask)
            self.logger.info(f"Fixed the {anatomy} mask to the one learned in {path}")
        mask.requires_grad_(False)

    def _init_mask_from(self, path: str):
        """Start every class's mask from a loupe_2d run's learned mask, left trainable (unlike fixed_mask_checkpoints)."""
        options = self.options
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        source = checkpoint['options']
        if source.sampler_type != 'loupe_2d':
            raise ValueError(f"{path} is a {source.sampler_type} run; init_mask_checkpoint needs a loupe_2d run")
        weights = checkpoint['model']['sampler.mask_net.mask']
        if weights.shape[0] != 1:
            raise ValueError(f"{path} holds {weights.shape[0]} masks; expected one learned mask")
        logits = source.slope * weights[0, ..., 0]
        mask_net = _unwrapped(self.model).sampler.mask_net
        if options.sampler_type == 'smooth_adapt_2d':
            mask_net.set_base_logits(logits.to(mask_net.mask))
        elif options.sampler_type in ('loupe_2d', 'seq_adapt_2d'):
            with torch.no_grad():
                mask_net.mask.copy_((logits / options.slope).view(1, *logits.shape, 1).expand_as(mask_net.mask))
        else:
            raise ValueError("init_mask_checkpoint is for loupe_2d, smooth_adapt_2d or seq_adapt_2d, "
                             f"not {options.sampler_type}")
        self.logger.info(f"Started the sampler's mask from {path}")

    def _param_groups(self):
        """Trainable parameters, with the mask parameters split out when they have their own lr.

        Under Adam each logit (or autosamp position, in grid cells) moves about
        `lr` per step whatever its gradient's scale, so the sampler's pace is set
        by this lr, not by the reconstructor's.
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        head_mlp = [p for name, p in _unwrapped(self.model).sampler.named_parameters()
                    if any(key in f'.{name}' for key in HEAD_MLP_PARAMETERS) and p.requires_grad]
        if head_mlp:
            head_ids = {id(p) for p in head_mlp}
            rest = [p for p in trainable if id(p) not in head_ids]
            return [{'params': rest}, {'params': head_mlp, 'lr': getattr(self.options, 'sampler_head_mlp_lr', 1e-4)}]
        if self.options.sampler_type == 'seq_adapt_2d':
            return self._residual_param_groups(trainable)
        gate_lr = getattr(self.options, 'gate_lr', None)
        if self.options.sampler_type == 'bank_adapt_2d' and gate_lr is not None:
            gate = [p for name, p in _unwrapped(self.model).sampler.mask_net.named_parameters() if name != 'mask' and p.requires_grad]
            gate_ids = {id(p) for p in gate}
            return [{'params': [p for p in trainable if id(p) not in gate_ids]}, {'params': gate, 'lr': gate_lr}]
        if self.options.sampler_type == 'autosamp_2d':
            policy_lr = self.options.autosamp_lr
        elif self.options.mask_train_mode == 'reinforce':
            policy_lr = self.options.reinforce_lr
        else:
            policy_lr = getattr(self.options, 'sampler_lr', None)
        if policy_lr is None:
            return trainable
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        policy = [p for p in model.sampler.mask_net.parameters() if p.requires_grad]
        policy_ids = {id(p) for p in policy}
        rest = [p for p in trainable if id(p) not in policy_ids]
        if self.options.sampler_type == 'smooth_adapt_2d':
            head = [p for name, p in model.sampler.mask_net.named_parameters() if name != 'mask' and p.requires_grad]
            head_ids = {id(p) for p in head}
            groups = ({'params': rest}, {'params': [p for p in policy if id(p) not in head_ids], 'lr': policy_lr},
                      {'params': head, 'lr': self.options.adapt_head_lr})
            return [group for group in groups if group['params']]
        policy_group = {'params': policy, 'lr': policy_lr}
        if self.options.sampler_type == 'autosamp_2d':
            policy_group['weight_decay'] = 0.0  # decay would drag every sample towards DC
            policy_group['eps'] = POSITION_ADAM_EPS
        return [group for group in ({'params': rest}, policy_group) if group['params']]

    def _residual_param_groups(self, trainable):
        """seq_adapt_2d: its residual U-Net in its own last Adam group at residual_lr (build_optimizer keeps the sampler
        out of Muon), and its static maps in a group at sampler_lr when that is set."""
        mask_net = _unwrapped(self.model).sampler.mask_net
        residual = [p for p in mask_net.residual.parameters() if p.requires_grad]
        sampler_lr = getattr(self.options, 'sampler_lr', None)
        static = [mask_net.mask] if mask_net.mask.requires_grad and sampler_lr is not None else []
        separate = {id(p) for p in residual + static}
        groups = ({'params': [p for p in trainable if id(p) not in separate]},
                  {'params': static, 'lr': sampler_lr},
                  {'params': residual, 'lr': getattr(self.options, 'residual_lr', 1e-4)})
        return [group for group in groups if group['params']]

    @staticmethod
    def _tile_for_mask_draws(batch, kspace, target, draws):
        """Repeat the whole batch `draws` times, draw-major, so the sampler's stratified thresholds line up.

        The k-space and target are tiled after any phase scrambling, so every draw of a slice sees the
        same data and the draws differ only in their mask. The loss then averages over the draws.
        """
        def tile(value):
            return value.repeat(draws, *[1] * (value.dim() - 1))

        tiled = {key: tile(value) if torch.is_tensor(value) else value * draws if isinstance(value, list) else value
                 for key, value in batch.items()}
        return tiled, tile(kspace), tile(target)

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

    def _anneal_autosamp_kernel(self, progress):
        """Set the autosamp soft-mask width for `progress` in [0, 1] through training; a no-op without a soft forward.

        Once the anneal ends the positions are frozen, so the rest of training fine-tunes
        the reconstructor on a fixed mask. A narrow kernel's gradient would otherwise
        again favour the empty neighbour over the sampled cell and drift points onto the
        snap boundaries.
        """
        if self.options.sampler_type != 'autosamp_2d' or self.options.autosamp_kernel_sigma_start <= 0:
            return
        mask_net = _unwrapped(self.model).sampler.mask_net
        annealing = progress < self.options.autosamp_anneal_fraction
        mask_net.positions.requires_grad_(annealing)
        mask_net.kernel_sigma = annealed_kernel_sigma(
            self.options.autosamp_kernel_sigma_start, self.options.autosamp_kernel_sigma_end,
            self.options.autosamp_anneal_fraction, progress,
        ) if annealing else 0.0

    def _recon_modules(self):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        return model.reconstructor, model.sensetivity_net

    def _freeze_recon(self):
        for module in self._recon_modules():
            module.requires_grad_(False)
            module.eval()
        self.logger.info("Froze reconstructor and sensetivity_net; training sampler only")

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
            if self._is_every(getattr(self.options, 'val_every', 1)):
                val_loss, mean_sparsity, val_time, val_metrics = self.val_epoch()
            else:
                val_loss, mean_sparsity, val_time, val_metrics = float('nan'), float('nan'), 0.0, {}

            for prefix, metrics in [('train', train_metrics), ('val', val_metrics)]:
                if metrics:
                    history[f'{prefix}_epoch'].append(epoch + 1)
                for key, val in metrics.items():
                    history[f'{prefix}_{key.lower()}'].append(
                        val.item() if hasattr(val, 'item') else val
                    )
            self._plot_metrics(history)

            is_new_best = val_loss < self.best_val_loss
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if not np.isnan(val_loss):
                self.validations_since_best = 0 if is_new_best else self.validations_since_best + 1
            self.save_model(is_new_best)
            if epoch + 1 in getattr(self.options, 'save_epochs', []):
                self.save_epoch_snapshot()
            if epoch == self.start_epoch:
                # Compiled graphs and hydra's config stay alive all run; left in the collector, every full collection walks them.
                gc.collect()
                gc.freeze()

            self.logger.info(
                f'Epoch = [{1 + self.epoch:4d}/{self.options.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} MeanSparsity = {mean_sparsity:.4g} '
                f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s PeakMemory = {self._peak_memory_gib():.2f}GiB',
            )
            patience = getattr(self.options, 'early_stop_patience', 0)
            if patience > 0 and self.validations_since_best >= patience:
                self.logger.info(f'Early stop after epoch {1 + self.epoch}: no new best val loss in {patience} '
                                 f'validations (best {self.best_val_loss:.4g})')
                break

        self.save_ema_model()
        wandb.finish()

    def _peak_memory_gib(self) -> float:
        """Peak memory held by tensors on the training GPU so far, not the allocator's reserve that nvidia-smi shows."""
        return torch.cuda.max_memory_allocated(self.device) / 2**30 if torch.cuda.is_available() else float('nan')

    def _is_every(self, every: int) -> bool:
        """Whether this epoch is one of every `every`-th epochs; the last epoch always is."""
        return (self.epoch + 1) % max(1, every) == 0 or self.epoch + 1 == self.end_epoch

    def _set_store_visuals(self, store: bool):
        _unwrapped(self.model).store_visuals = store

    def train_epoch(self):
        self.model.train()
        visualize = self._is_every(getattr(self.options, 'visualize_every', 1))
        metrics_every = max(1, getattr(self.options, 'train_metrics_every', 1))
        if self.options.freeze_recon:
            for module in self._recon_modules():
                module.eval()

        losses = []
        metrics = build_metrics(device=self.options.device)
        masked_metrics = MaskedMetricAccumulator(device=self.options.device)
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
            self._anneal_autosamp_kernel((self.epoch + iter_idx / max_batches) / self.options.num_epochs)

            kspace = self._load_kspace(batch)
            target = self._target(batch, kspace)
            if self.options.mask_draws > 1:
                batch, kspace, target = self._tile_for_mask_draws(batch, kspace, target, self.options.mask_draws)
            anatomies = list(batch['anatomy'])
            log_images = visualize and not set(anatomies) <= logged_train_anatomies
            self._set_store_visuals(log_images)
            step_figure_slices = {anatomy: anatomies.index(anatomy) for anatomy in dict.fromkeys(anatomies)
                                  if anatomy not in logged_train_anatomies} if log_images else {}
            recorder = self._record_steps(step_figure_slices.values()) \
                if log_images or iter_idx % metrics_every == 0 else None

            try:
                pred_dict = self._train_forward(target, kspace, batch)
            except RuntimeError as err:
                if 'Binarization failed to match target density' not in str(err):
                    raise
                self.logger.warning(
                    "Skipping batch with NaN-driven binarization failure "
                    f"(fnames={batch['fname']}, slice_ids={batch['slice_id'].tolist()}): {err}"
                )
                self.optimizer.zero_grad()
                continue
            finally:
                _unwrapped(self.model).step_recorder = None

            if iter_idx == 0 and visualize:
                data_for_vis_name = f'train_epoch={self.epoch + 1}_iter=1'
                self._visualize_and_save(self.epoch, data_for_vis_name)

            for anatomy in dict.fromkeys(anatomies) if log_images else ():
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
            target_dict = self._target_dict(batch, target, kspace)
            meta = {
                'recon_weight': 1.0, 'kspace_weight': 0.0,
                'uncertainty_weight': self.options.uncertainty_weight,
                'anatomy_weight': self.options.anatomy_weight,
            }
            loss, log_dict = self.model.loss(pred_dict, target_dict, meta, self.options.loss_type)

            grad_log = self._optimize(loss, batch, kspace, target, global_step_offset + iter_idx)
            if grad_log is None:
                continue

            losses.append(loss.item())
            if recorder is not None:
                grad_log.update(recorder.scalars('train'))
            if step_figure_slices:
                anatomy_images.update(self._step_figures(pred_dict, kspace, target, batch, step_figure_slices,
                                                         recorder, 'train'))

            if iter_idx % metrics_every == 0:
                target_for_metrics = target.detach()
                if output.dim() != target_for_metrics.dim():
                    target_for_metrics = target_for_metrics.unsqueeze(1)
                data_range = target_for_metrics.max()
                metrics["PSNR"].data_range = data_range
                metrics["SSIM"].data_range = data_range
                signal_mask = self._metric_mask(batch)
                masked_output, masked_target = _apply_signal_mask(output.detach(), target_for_metrics, signal_mask)
                metrics.update(masked_output, masked_target)
                if iter_idx % self.options.train_masked_metrics_every == 0:
                    masked_metrics.update(output.detach(), target_for_metrics, signal_mask)
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
        masked_metric_means = masked_metrics.compute()
        epoch_log = {f'train/{name}': value.item() for name, value in metric_means.items()}
        epoch_log.update({f'train/{name}': value.item() for name, value in masked_metric_means.items()})
        self._set_store_visuals(True)
        epoch_log['train/loss'] = np.mean(losses)
        epoch_log['epoch'] = self.epoch
        epoch_log['system/peak_memory_gib'] = self._peak_memory_gib()
        epoch_log['system/train_epoch_seconds'] = time.perf_counter() - epoch_start_time
        epoch_log.update(anatomy_images)
        epoch_log.update(anatomy_stats.compute('train'))
        mask_net = getattr(_unwrapped(self.model).sampler, 'mask_net', None)  # caipi_2d has none
        if isinstance(mask_net, (ProbMask, LineConstrainedProbMask)):
            logit_log, self.previous_mask_logits = mask_logit_stats(mask_net, self.previous_mask_logits)
            epoch_log.update(logit_log)
        wandb.log(epoch_log)

        return np.mean(losses), time.perf_counter() - epoch_start_time, metric_means

    def _train_forward(self, target, kspace, batch):
        """The forward pass whose output the step's loss, metrics and logged images come from."""
        return self.model(target, kspace, **self._model_inputs(batch))

    def _optimize(self, loss, batch, kspace, target, step):  # noqa: ARG002
        """Back-propagate `loss` and take an optimizer step; returns the step's gradient log, or None if the step was skipped."""
        self.optimizer.zero_grad()
        loss.backward()
        grad_log = gradient_stats(self.model, self.options.grad_clip_norm)
        bad_params = nonfinite_gradient_params(self.model) if grad_log['grad/nonfinite'] else []
        clip_gradients_per_group(self.model, self.options.grad_clip_norm)
        grad_log['data/zero_coil_frac'] = zero_coil_fraction(kspace)
        grad_log['train/lr'] = self.optimizer.param_groups[0]['lr']
        if self.options.mask_train_mode == 'reinforce':
            grad_log['train/policy_lr'] = self.optimizer.param_groups[-1]['lr']
        if self.options.sampler_type == 'autosamp_2d':
            grad_log['train/autosamp_lr'] = self.optimizer.param_groups[-1]['lr']
            mask_net = _unwrapped(self.model).sampler.mask_net
            grad_log['train/autosamp_cell_change_frac'] = mask_net.cell_change_fraction
            if mask_net.kernel_sigma > 0:
                grad_log['train/autosamp_kernel_sigma'] = mask_net.kernel_sigma
        if grad_log['grad/nonfinite']:
            self.logger.warning(
                "Skipping optimizer step with non-finite gradient norm "
                f"(loss={loss.item():.4g}, fnames={batch['fname']}, slice_ids={batch['slice_id'].tolist()}, "
                f"non-finite params={bad_params})"
            )
            wandb.log({**grad_log, 'train/skipped_step': 1.0, 'train/step': step})
            self.optimizer.zero_grad()
            return None
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update()
        return grad_log

    def val_epoch(self):
        self.model.eval()
        visualize = self._is_every(getattr(self.options, 'visualize_every', 1))
        self.val_phase_generator.manual_seed(PHASE_SCRAMBLE_VAL_SEED)
        self.val_mask_generator.manual_seed(REINFORCE_VAL_MASK_SEED)
        losses = []
        sparsity = []
        metrics = build_metrics(device=self.options.device)
        masked_metrics = MaskedMetricAccumulator(device=self.options.device)
        anatomy_stats = _AnatomyStats(self.options.device)
        start = time.perf_counter()
        logged_anatomies = set()
        anatomy_images = {}
        step_scalars = []

        with torch.no_grad():
            for iter_idx, batch in enumerate(self.val_loader):
                kspace = self._load_kspace(batch, self.val_phase_generator)
                target = self._target(batch, kspace)
                anatomies = list(batch['anatomy'])
                log_images = visualize and (
                    iter_idx == 0 or not set(anatomies) <= logged_anatomies | set(self.visual_examples))
                self._set_store_visuals(log_images)
                recorder = self._record_steps(())

                try:
                    pred_dict = self.model(target, kspace, **self._model_inputs(batch))
                except RuntimeError as err:
                    if 'Binarization failed to match target density' not in str(err):
                        raise
                    self.logger.warning(
                        "Skipping val batch with NaN-driven binarization failure "
                        f"(fnames={batch['fname']}, slice_ids={batch['slice_id'].tolist()}): {err}"
                    )
                    continue
                finally:
                    _unwrapped(self.model).step_recorder = None
                step_scalars.append(recorder.scalars('val'))

                if iter_idx == 0 and visualize:
                    data_for_vis_name = f'val_epoch={self.epoch + 1}'
                    self._visualize_and_save(self.epoch, data_for_vis_name)

                for anatomy in dict.fromkeys(anatomies) if log_images else ():
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

                target_dict = self._target_dict(batch, target, kspace)
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
                signal_mask = self._metric_mask(batch)
                masked_output, masked_target = _apply_signal_mask(output, target_for_metrics, signal_mask)
                metrics.update(masked_output, masked_target)
                masked_metrics.update(output, target_for_metrics, signal_mask)
                anatomy_stats.update(anatomies, masked_output, masked_target, mask)

            self._set_store_visuals(True)
            if visualize:
                anatomy_images.update(self._fixed_example_images())

            metric_means = metrics.compute()
            masked_metric_means = masked_metrics.compute()
            epoch_log = {f'val/{name}': value.item() for name, value in metric_means.items()}
            epoch_log.update({f'val/{name}': value.item() for name, value in masked_metric_means.items()})
            epoch_log['val/loss'] = np.mean(losses)
            epoch_log['epoch'] = self.epoch
            epoch_log.update(anatomy_images)
            epoch_log.update(anatomy_stats.compute('val'))
            epoch_log.update(_mean_scalars(step_scalars))
            wandb.log(epoch_log)

            self.logger.info(anatomy_stats.summary())

        return np.mean(losses), np.mean(sparsity), time.perf_counter() - start, metric_means

    def _fixed_example_images(self):
        """Image logs for the same mid-volume val slice of each anatomy every epoch."""
        images = {}
        dataset = self.val_loader.dataset
        for anatomy, index in self.visual_examples.items():
            batch = self.val_loader.collate_fn([dataset[index]])
            kspace = self._load_kspace(batch, self.val_phase_generator)
            target = batch['target'].to(self.device)
            recorder = self._record_steps([0])
            try:
                pred_dict = self.model(target, kspace, **self._model_inputs(batch))
            except RuntimeError as err:
                if 'Binarization failed to match target density' not in str(err):
                    raise
                self.logger.warning(f"Skipping fixed {anatomy} val example: {err}")
                continue
            finally:
                _unwrapped(self.model).step_recorder = None
            images.update(_build_image_log(pred_dict, target, f'val/{anatomy}', self.model.data_for_vis))
            images.update(self._step_figures(pred_dict, kspace, target, batch, {anatomy: 0}, recorder, 'val'))
        return images

    def _record_steps(self, slices):
        """A StepRecorder that the model's next forward reports every step to; the caller clears it afterwards."""
        recorder = StepRecorder(slices=slices)
        _unwrapped(self.model).step_recorder = recorder
        return recorder

    def _step_figures(self, pred_dict, kspace, target, batch, slices, recorder, prefix):
        """{f'{prefix}/{anatomy}/steps': image} of one row per sampling step for each `slices` index, also saved as
        PNGs in the visualization folder (activemri.utils.multistep.step_grid_figure)."""
        model = _unwrapped(self.model)
        labels, metric_mask = self._anatomy_labels(batch), self._metric_mask(batch)
        images = {}
        for anatomy, index in slices.items():
            panels = step_panels(model, kspace, target, pred_dict['sense_maps'].detach(), pred_dict['masks'], index,
                                 labels, metric_mask, recorder)
            source = 'the loss gradient is this batch\'s, the reconstructions use the weights after its update; ' \
                if prefix == 'train' else ''
            title = (f"{prefix} epoch {self.epoch + 1}: {anatomy} {batch['fname'][index]} slice "
                     f"{int(batch['slice_id'][index])}, K={model.num_step}; {source}rows before the last reconstruct "
                     "masks the reconstructor never trained on")
            figure = step_grid_figure(panels, title)
            figure.savefig(self.vis_dir / f'{prefix}_epoch={self.epoch + 1}_{anatomy}_steps.png')
            images[f'{prefix}/{anatomy}/steps'] = wandb.Image(figure)
            plt.close(figure)
        return images

    def _metric_mask(self, batch):
        """The run-independent metric support mask of the batch's slices, from their fully sampled k-space.

        A slice's mask depends on that slice alone, so it is computed the first time the slice is seen and kept on the
        CPU. Self-supervised batches are not cached: their k-space is a fresh holdout split at every training draw.
        """
        if 'holdout_mask' in batch:
            return metric_mask_from_real_kspace(batch['kspace'].to(self.device), batch['anatomy'])
        keys = list(zip(batch['anatomy'], batch['fname'], batch['slice_id'].tolist()))
        unseen = [i for i, key in enumerate(keys) if key not in self.metric_mask_cache]
        if unseen:
            masks = metric_mask_from_real_kspace(batch['kspace'][unseen].to(self.device),
                                                 [batch['anatomy'][i] for i in unseen])
            self.metric_mask_cache.update(zip([keys[i] for i in unseen], masks.bool().cpu()))
        return torch.stack([self.metric_mask_cache[key] for key in keys]).to(self.device).float()

    def _load_kspace(self, batch, generator=None):
        """The batch's k-space on the device, phase-scrambled when the run asks for it."""
        kspace = batch['kspace'].to(self.device)
        if self.options.phase_scramble_std > 0:
            kspace = scramble_phase_real(
                kspace, self.options.phase_scramble_std, self.options.phase_scramble_band, generator)
        return kspace

    def _target(self, batch, kspace):
        """The image the loss and metrics compare against: the batch's, or the frozen reconstructor's estimate from all acquired data."""
        if not self.options.pseudo_target:
            return batch['target'].to(self.device)
        inputs = self._model_inputs(batch)
        return _unwrapped(self.model).reconstruct_acquired(kspace, inputs['sense_maps'], inputs.get('kspace_support'))

    def _model_inputs(self, batch):
        """Keyword inputs the model takes beyond the k-space: anatomy labels, and the sense maps and k-space support if the data ships them."""
        inputs = {'anatomy_labels': self._anatomy_labels(batch)}
        if 'sense_maps' in batch:
            inputs['sense_maps'] = batch['sense_maps'].to(self.device)
        if 'kspace_support' in batch:
            inputs['kspace_support'] = batch['kspace_support'].to(self.device)
        return inputs

    def _target_dict(self, batch, target, kspace):
        """What the loss is scored against: the image target, plus the held-out k-space for self-supervised data."""
        target_dict = {'target': target, 'kspace': kspace}
        anatomy_labels = self._anatomy_labels(batch)
        if anatomy_labels is not None:
            target_dict['anatomy_labels'] = anatomy_labels
        if 'holdout_mask' in batch and not self.options.pseudo_target:
            target_dict['holdout_mask'] = batch['holdout_mask'].to(self.device)
            target_dict['holdout_kspace'] = batch['holdout_kspace'].to(self.device)
        return target_dict

    def _anatomy_labels(self, batch):
        """Conditioning-class index tensor for a batch, or None when the run has no classes."""
        if not self.anatomy_to_idx:
            return None
        return torch.tensor([self.anatomy_to_idx[n] for n in batch[self.options.label_key]], device=self.device)

    def _plot_metrics(self, history):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, metric in zip(axes, ['ssim', 'nmse', 'psnr']):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            if train_key in history:
                ax.plot(history['train_epoch'], history[train_key], label='Train')
            if val_key in history:
                ax.plot(history['val_epoch'], history[val_key], label='Val')
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
        # Write then rename, so a job killed mid-save leaves the previous checkpoint intact for the resume.
        last_tmp = self.exp_dir / 'last_model.pt.tmp'
        torch.save(checkpoint, f=last_tmp)
        last_tmp.replace(self.exp_dir / 'last_model.pt')
        if not is_new_best:
            return

        best_path = self.exp_dir / 'best_model.pt'
        best_tmp = self.exp_dir / 'best_model.pt.tmp'
        torch.save(checkpoint, f=best_tmp)
        best_tmp.replace(best_path)
        for old_checkpoint in self.exp_dir.glob('model_epoch_*.pt'):
            old_checkpoint.unlink()

    def save_epoch_snapshot(self):
        """`epoch_NNN.pt` (1-indexed, weights only), kept for an eval of a fixed epoch, e.g. the pilot gate's epoch 10.

        Not `model_epoch_*.pt`: save_model deletes those whenever a new best is saved.
        """
        torch.save({'epoch': self.epoch, 'options': self.options, 'model': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss}, self.exp_dir / f'epoch_{self.epoch + 1:03d}.pt')

    def save_ema_model(self):
        """`ema_model.pt`: a `last_model.pt` with the averaged weights, for the same eval scripts."""
        if self.ema is None:
            return
        with self.ema.applied():
            torch.save({'epoch': self.epoch, 'options': self.options, 'model': self.model.state_dict(),
                        'best_val_loss': self.best_val_loss}, self.exp_dir / 'ema_model.pt')

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

        wandb.init(project="SeqMRI", name=f"test_{run_name(self.options)}", group=run_group(self.options),
                   tags=run_tags(self.options), notes=str(self.exp_dir), job_type='test',
                   config=vars(self.options), dir=str(self.exp_dir))

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


OPTIMIZERS = ('adam', 'muon')


def build_optimizer(options: TrainConfig, param_groups, model):
    """Adam over `param_groups` (a parameter list or Adam group dicts), or with `options.optimizer='muon'` the same
    groups with the U-Nets' hidden convolutions moved into one Muon group (activemri/training/muon.py)."""
    kind = getattr(options, 'optimizer', 'adam')
    fused = getattr(options, 'fused_adam', False) or None
    if kind == 'adam':
        return torch.optim.Adam(param_groups, options.lr, weight_decay=options.weight_decay, fused=fused)
    if kind != 'muon':
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {kind!r}")
    groups = [dict(group) for group in param_groups] if isinstance(param_groups[0], dict) else [{'params': list(param_groups)}]
    trainable = {id(p) for group in groups for p in group['params']}
    muon = [p for p in muon_parameters(model, exclude=[model.sampler]) if id(p) in trainable]
    muon_ids = {id(p) for p in muon}
    for group in groups:
        group['params'] = [p for p in group['params'] if id(p) not in muon_ids]
    groups = [group for group in groups if group['params']]
    if muon:  # slot 1, so the last group stays the sampler's: the trainer logs train/policy_lr and train/autosamp_lr from it
        groups.insert(1, {'params': muon, 'use_muon': True, 'lr': options.muon_lr or options.lr,
                          'momentum': options.muon_momentum, 'weight_decay': options.weight_decay})
    return MuonAdam(groups, options.lr, weight_decay=options.weight_decay, fused=fused)


COSINE_FLOOR = 0.01  # fraction of the base lr the cosine schedules decay to
WARMUP_START = 0.1  # fraction of the base lr at epoch 0 of warmup_cosine


def build_lr_scheduler(options: TrainConfig, optimizer):
    """Per-epoch schedule. The trainer calls `step(epoch)`, so every schedule is a closed form of the epoch."""
    kind = options.lr_scheduler
    if kind == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, options.lr_step_size, options.lr_gamma)

    total, warmup = options.num_epochs, options.lr_warmup_epochs

    def cosine(epoch):
        progress = min(epoch / total, 1.0)
        return COSINE_FLOOR + (1 - COSINE_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))

    if kind == 'constant':
        factor = lambda epoch: 1.0
    elif kind == 'cosine':
        factor = cosine
    elif kind == 'warmup_cosine':
        factor = lambda epoch: WARMUP_START + (1 - WARMUP_START) * epoch / warmup if epoch < warmup else cosine(epoch)
    else:
        raise ValueError(f"Unknown lr_scheduler {kind!r}; expected step, constant, cosine or warmup_cosine")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _sampler_type(options: TrainConfig) -> str:
    """Sampler family for these options, falling back to the pre-sampler_type
    `line_constrained` flag so older checkpoints still load."""
    sampler_type = getattr(options, 'sampler_type', None)
    if sampler_type is not None:
        return sampler_type
    return 'conv_1d' if getattr(options, 'line_constrained', False) else 'conv_2d'


# Positions' gradients scale with the k-space amplitude and are then shrunk by the global
# clip, so they can fall below Adam's default eps (1e-8) and stall; this keeps Adam's
# per-step move at ~lr whatever their scale.
POSITION_ADAM_EPS = 1e-16
GRADIENT_GROUPS = ('sampler', 'reconstructor', 'sensetivity_net')
SATURATED_LOGIT = math.log(99)  # |slope * w| beyond this puts the mask probability outside [0.01, 0.99]


def _unwrapped(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def mask_logit_stats(mask_net, previous_logits):
    """Per-epoch LOUPE logit diagnostics, keyed for wandb, plus the logits to pass next epoch.

    Drift shows as saturated_abs_mean and abs_max climbing steadily while
    saturated_moving_outward_frac stays near 1: saturated logits that keep
    being pushed further out by a floored gradient instead of settling.
    """
    logits = (mask_net.slope * mask_net.mask.detach()).flatten().float()
    magnitude = logits.abs()
    saturated = magnitude > SATURATED_LOGIT
    stats = {
        'mask/logit_abs_max': magnitude.max(),
        'mask/logit_abs_p99': torch.quantile(magnitude, 0.99),
        'mask/saturated_frac': saturated.float().mean(),
        'mask/saturated_abs_mean': magnitude[saturated].mean() if saturated.any() else magnitude.new_zeros(()),
    }
    if previous_logits is not None and saturated.any():
        moved_outward = magnitude > previous_logits.abs()
        stats['mask/saturated_moving_outward_frac'] = moved_outward[saturated].float().mean()
    log = {name: value.item() for name, value in stats.items()}
    log['mask/logit_hist'] = wandb.Histogram(logits.cpu().numpy())
    return log, logits


def gradient_stats(model, max_norm):
    """Per-step gradient diagnostics, keyed for wandb. Call before clipping.

    Reports the global norm, whether any gradient is non-finite, whether each of
    `clip_gradients_per_group`'s clips will fire (`grad/clipped` for the
    non-sampler parameters, `grad/sampler_clipped`), and for each of the
    sampler, reconstructor and sensitivity net:
    the group's gradient norm, its largest absolute entry, and how many
    parameter tensors hold a non-finite gradient. Together these show which
    component blows up first.
    """
    model = _unwrapped(model)
    all_grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.stack(torch._foreach_norm(all_grads)).norm()
    sampler_grads = [p.grad for p in model.sampler.parameters() if p.grad is not None]
    sampler_norm = torch.stack(torch._foreach_norm(sampler_grads)).norm() if sampler_grads else total_norm.new_zeros(())
    rest_norm = (total_norm ** 2 - sampler_norm ** 2).clamp_min(0).sqrt()
    names = ['grad/total_norm', 'grad/clipped', 'grad/sampler_clipped', 'grad/nonfinite']
    values = [total_norm, (rest_norm > max_norm).float(), (sampler_norm > max_norm).float(),
              (~torch.isfinite(total_norm)).float()]
    for group in GRADIENT_GROUPS:
        grads = [p.grad for p in getattr(model, group).parameters() if p.grad is not None]
        if not grads:
            continue
        per_tensor_norm = torch.stack(torch._foreach_norm(grads))
        names += [f'grad/{group}_norm', f'grad/{group}_max_abs', f'grad/{group}_nonfinite_tensors']
        values += [
            per_tensor_norm.norm(),
            torch.stack(torch._foreach_norm(grads, math.inf)).max(),
            (~torch.isfinite(per_tensor_norm)).sum().float(),
        ]
    mask_net = getattr(model.sampler, 'mask_net', None)
    if isinstance(mask_net, SequentialAdaptiveMask):
        for name, params in (('sampler_residual', list(mask_net.residual.parameters())), ('sampler_static', [mask_net.mask])):
            grads = [p.grad for p in params if p.grad is not None]
            if grads:
                names.append(f'grad/{name}_norm')
                values.append(torch.stack(torch._foreach_norm(grads)).norm())
    return dict(zip(names, torch.stack([v.float() for v in values]).tolist()))


def _mean_scalars(logs):
    """Key-wise mean of {name: number} dicts, each key over the dicts that hold it."""
    collected = defaultdict(list)
    for log in logs:
        for key, value in log.items():
            collected[key].append(value)
    return {key: float(np.mean(values)) for key, values in collected.items()}


def clip_gradients_per_group(model, max_norm):
    """Clip the sampler's gradient and the rest of the model's separately, each to `max_norm`.

    One global clip lets a sampler gradient blow-up shrink the reconstructor's step by the same factor,
    freezing it while the sampler diverges.
    """
    model = _unwrapped(model)
    sampler_ids = {id(p) for p in model.sampler.parameters()}
    sampler = [p for p in model.sampler.parameters() if p.grad is not None]
    rest = [p for p in model.parameters() if p.grad is not None and id(p) not in sampler_ids]
    for group in (sampler, rest):
        if group:
            torch.nn.utils.clip_grad_norm_(group, max_norm)


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


def _cg_iters_by_label(iters_by_anatomy, label_classes):
    """CG iterations per label index from {anatomy: iters}; a class such as knee-PD takes its anatomy's count."""
    if not iters_by_anatomy:
        return None
    missing = [c for c in label_classes if c.split('-')[0] not in iters_by_anatomy]
    if missing:
        raise ValueError(f"cg_iters_by_anatomy {dict(iters_by_anatomy)} has no entry for classes {missing}")
    return [int(iters_by_anatomy[c.split('-')[0]]) for c in label_classes]


def compile_unets(model: torch.nn.Module, channels_last=False) -> None:
    """torch.compile every U-Net body in place (the VarNet cascades' and the sensitivity net's), so state_dict keys
    stay as they are. Shapes are static: each new batch shape (the smaller last batch, validation) compiles once, except
    the sensitivity net's batch of live coils, which it marks dynamic.

    `channels_last` True lays out every U-Net channels-last; 'sens' only the sensitivity net, whose channel counts are
    multiples of 8. cuDNN pads the cascades' 18 and 36 channels in that layout, which costs more than it saves."""
    use_system_compilers_if_missing()
    allow_recompiles()
    for module in list(model.modules()):
        if isinstance(module, NormUnet):
            module.unet.compile(dynamic=False)
        elif isinstance(module, SensetivityModel_mc):
            module.model.compile(dynamic=False, options={'force_layout_optimization': True} if channels_last == 'sens' else None)


def build_model(options: TrainConfig):
    recon_cfg = options.recon

    anatomy_classes = options.label_classes
    num_anatomy_classes = len(anatomy_classes) if options.anatomy_weight > 0 else 0
    num_sampler_conditioning_classes = (
        len(anatomy_classes) if getattr(options, 'anatomy_conditioning', False) else 0
    )
    num_recon_conditioning_classes = (
        len(anatomy_classes)
        if any(getattr(recon_cfg, key, False) for key in ('anatomy_conditioning', 'film_conditioning', 'prompt_conditioning'))
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
            prompt=getattr(recon_cfg, 'prompt_conditioning', False),
            label_blind=getattr(recon_cfg, 'label_blind', False),
            dc_mode=getattr(recon_cfg, 'dc_mode', 'where'),
            dc_weight_shape=tuple(options.resolution) if getattr(recon_cfg, 'dc_weight_map', False) else None,
            adaptive_input=getattr(recon_cfg, 'adaptive_input', False),
            momentum_history=getattr(recon_cfg, 'momentum_history', 0),
            checkpoint_cascades=getattr(recon_cfg, 'checkpoint_cascades', False),
            checkpoint_sense=getattr(recon_cfg, 'checkpoint_sense', True),
            backbone=getattr(recon_cfg, 'backbone', 'unet'),
            csme=getattr(recon_cfg, 'csme', False),
            unet_precision=getattr(recon_cfg, 'unet_precision', 'fp32'),
            fused_coil_ops=getattr(recon_cfg, 'fused_coil_ops', False),
        )
    elif isinstance(recon_cfg, SDUMReconConfig):
        reconstructor = SDUMRecon(
            recon_cfg.repo_path,
            class_acq_types=recon_cfg.class_acq_types,
            anatomy_conditioning=recon_cfg.anatomy_conditioning,
            config_name=recon_cfg.config_name,
            checkpoint=recon_cfg.checkpoint,
            pretrained=recon_cfg.pretrained,
            mask_type=recon_cfg.mask_type,
            acq_type=recon_cfg.acq_type,
            output_scale=recon_cfg.output_scale,
        )
    elif isinstance(recon_cfg, CGSenseReconConfig):
        reconstructor = CGSENSEReconstructor(
            num_iters=recon_cfg.cg_iters,
            lamda=recon_cfg.cg_lamda,
            iters_by_label=_cg_iters_by_label(recon_cfg.cg_iters_by_anatomy, anatomy_classes),
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
        sens_upsample_method=getattr(options, 'sens_upsample_method', 'conv'),
        sens_conv_after_upsample=getattr(options, 'sens_conv_after_upsample', False),
        sens_resolution=getattr(options, 'sens_resolution', None),
        sens_precision=getattr(options, 'sens_precision', 'fp32'),
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
        reinforce_eval=getattr(options, 'reinforce_eval', 'sample'),
        sampler_chans=getattr(options, 'sampler_chans', 64),
        sampler_depth=getattr(options, 'sampler_depth', 4),
        sampler_head=getattr(options, 'sampler_head', 'none'),
        sampler_image_input=getattr(options, 'sampler_image_input', False),
        caipi_keep_fraction=getattr(options, 'caipi_keep_fraction', 1.0),
        caipi_accel_y=getattr(options, 'caipi_accel_y', 2),
        caipi_accel_z=getattr(options, 'caipi_accel_z', 2),
        autosamp_init=getattr(options, 'autosamp_init', 'uniform'),
        mask_grad_floor=getattr(options, 'mask_grad_floor', 0.0),
        ste_grad_floor=getattr(options, 'ste_grad_floor', 0.0),
        mask_draws=getattr(options, 'mask_draws', 1),
        logit_limit=getattr(options, 'logit_limit', 6.0),
        logit_leak=getattr(options, 'logit_leak', 0.01),
        adapt_grid=getattr(options, 'adapt_grid', 16),
        adapt_hidden=getattr(options, 'adapt_hidden', 64),
        bank_size=getattr(options, 'bank_size', 8),
        gate_temperature=getattr(options, 'gate_temperature', 1.0),
        gate_logit_limit=getattr(options, 'gate_logit_limit', 0.0),
        selection=getattr(options, 'selection', 'bernoulli'),
        dcm_vd_power=getattr(options, 'dcm_vd_power', 4.0),
        dcm_tau=getattr(options, 'dcm_tau', 5.0),
        ppo_policy=getattr(options, 'ppo_policy', None),
        fixed_mask_path=getattr(options, 'fixed_mask_path', None),
        loupe_vd_init_power=getattr(options, 'loupe_vd_init_power', 0.0),
        loupe_same_start=getattr(options, 'loupe_same_start', False),
        acquired_only=getattr(options, 'ssdu', False),
        budget_over=getattr(options, 'budget_over', 'grid'),
        sampler_coil_combine=getattr(options, 'sampler_coil_combine', 'sense'),
        restrict_recon_to_extent=getattr(recon_cfg, 'restrict_to_extent', False),
        residual_chans=getattr(options, 'residual_chans', 16),
        residual_depth=getattr(options, 'residual_depth', 4),
        residual_limit=getattr(options, 'residual_limit', 6.0),
        data_blind=getattr(options, 'data_blind', False),
        residual_checkpoint=getattr(options, 'residual_checkpoint', True),
        data_channels=getattr(options, 'data_channels', None),
        sampler_input=getattr(options, 'sampler_input', 'measured'),
        recon_steps=getattr(options, 'recon_steps', None) or [6, 9, 12, 14],
        virtual_coils=getattr(options, 'virtual_coils', 4),
        seqmri_input_scale=getattr(options, 'seqmri_input_scale', 'raw'),
        loupe_point_symmetric=getattr(options, 'loupe_point_symmetric', False),
    )
    replace_instance_norms(model.reconstructor, getattr(recon_cfg, 'norm', 'instance'))
    replace_instance_norms(model.sensetivity_net, getattr(options, 'sens_norm', 'instance'))
    if model.input_reconstructor is not None:
        replace_instance_norms(model.input_reconstructor, getattr(recon_cfg, 'norm', 'instance'))
        replace_instance_norms(model.input_sensetivity_net, getattr(options, 'sens_norm', 'instance'))
        model.input_reconstructor.requires_grad_(False)
        model.input_sensetivity_net.requires_grad_(False)

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
    """Legacy target-threshold mask, kept only for scripts not yet moved to activemri.utils.metric_mask."""
    flat = target.flatten(1)
    max_val = flat.max(dim=1).values.view(-1, *([1] * (target.dim() - 1)))
    return (target > threshold * max_val).to(target.dtype)


def _apply_signal_mask(output, target, mask=None):
    """Zero both images outside the metric support `mask` (see activemri.utils.metric_mask).

    Without a mask this falls back to the legacy target-threshold mask, which no longer matches the trainer's metrics.
    """
    if mask is None:
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
