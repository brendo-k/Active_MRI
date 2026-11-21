"""Config and dataloader builders for Sequential_Trainer (the loupe / conv / reinforce samplers)."""
from pathlib import Path

from omegaconf import DictConfig

from activemri.config import TrainConfig
from activemri.reconstructors.config import UNetReconConfig, CGSenseReconConfig, SDUMReconConfig, VarNetReconConfig
from activemri.datasets.fastmri_data import build_fastmri_dataloaders, build_manifest_dataloaders
from activemri.datasets.siemens_space_data import build_siemens_space_dataloaders
from activemri.samplers.sampler import CONV_2D_TYPES

RECON_CONFIGS = {
    'unet': UNetReconConfig,
    'varnet': VarNetReconConfig,
    'sdum': SDUMReconConfig,
    'cg_sense': CGSenseReconConfig,
}

def build_recon_config(cfg: DictConfig):
    for config_name, config_class in RECON_CONFIGS.items():
        if cfg.recon.recon_type == config_name:
            return config_class(**cfg.recon)
    raise ValueError(f"Unknown recon_type: {cfg.recon.recon_type}")


def _sampler_image_input(sampler_cfg) -> bool:
    """Whether the sampler sees the zero-filled image; unset means yes for the conv samplers, which alone support it."""
    image_input = sampler_cfg.get('image_input', None)
    return sampler_cfg.sampler_type in CONV_2D_TYPES if image_input is None else bool(image_input)


def build_train_config(cfg: DictConfig) -> TrainConfig:
    config = TrainConfig(
        model=cfg.sampler.model,
        num_step=cfg.training.num_step,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        lr=cfg.training.lr,
        lr_step_size=cfg.training.lr_step_size,
        lr_gamma=cfg.training.lr_gamma,
        lr_scheduler=cfg.training.get('lr_scheduler', 'step'),
        lr_warmup_epochs=cfg.training.get('lr_warmup_epochs', 2),
        weight_decay=cfg.training.weight_decay,
        optimizer=cfg.training.get('optimizer', 'adam'),
        muon_lr=cfg.training.get('muon_lr', None),
        muon_momentum=cfg.training.get('muon_momentum', 0.95),
        grad_clip_norm=cfg.training.get('grad_clip_norm', 1.0),
        limit_batches=cfg.training.limit_batches,
        data_parallel=cfg.training.data_parallel,
        compile_unets=cfg.training.get('compile_unets', False),
        channels_last_unets=cfg.training.get('channels_last_unets', False),
        cudnn_benchmark=cfg.training.get('cudnn_benchmark', True),
        train_masked_metrics_every=cfg.training.get('train_masked_metrics_every', 1),
        train_metrics_every=cfg.training.get('train_metrics_every', 1),
        val_every=cfg.training.get('val_every', 1),
        early_stop_patience=cfg.training.get('early_stop_patience', 0),
        ema_decay=cfg.training.get('ema_decay', 0.0),
        fused_adam=cfg.training.get('fused_adam', False),
        visualize_every=cfg.training.get('visualize_every', 1),
        loss_type=cfg.loss.loss_type,
        uncertainty_loss=cfg.loss.uncertainty_loss,
        uncertainty_weight=cfg.loss.uncertainty_weight,
        anatomy_weight=cfg.loss.get('anatomy_weight', 0.0),
        anatomy_classes=list(cfg.datasets.anatomy),
        condition_classes=list(cfg.datasets.get('condition_classes', [])),
        train_anatomies=list(cfg.datasets.get('train_anatomies', [])),
        anatomy_conditioning=cfg.sampler.get('anatomy_conditioning', False),
        out_dir=[Path(cfg.output.out_dir)],
        run_name=cfg.output.get('run_name', None),
        checkpoint=cfg.output.checkpoint,
        pretrained_recon=cfg.output.get('pretrained_recon', None),
        accelerations=list(cfg.sampler.accelerations),
        sampler_type=cfg.sampler.sampler_type,
        slope=cfg.sampler.get('slope', 10.0),
        mask_grad_floor=cfg.sampler.get('mask_grad_floor', 0.0),
        ste_grad_floor=cfg.sampler.get('ste_grad_floor', 0.0),
        mask_draws=cfg.sampler.get('mask_draws', 1),
        selection=cfg.sampler.get('selection', 'bernoulli'),
        dcm_vd_power=cfg.sampler.get('dcm_vd_power', 4.0),
        dcm_tau=cfg.sampler.get('dcm_tau', 5.0),
        logit_limit=cfg.sampler.get('logit_limit', 6.0),
        logit_leak=cfg.sampler.get('logit_leak', 0.01),
        loupe_logit_limit=cfg.sampler.get('loupe_logit_limit', 0.0),
        slope_start=cfg.sampler.get('slope_start', None),
        slope_anneal_epochs=cfg.sampler.get('slope_anneal_epochs', 20.0),
        flat_mask_fraction=cfg.sampler.get('flat_mask_fraction', 0.0),
        flat_mask_decay_epochs=cfg.sampler.get('flat_mask_decay_epochs', None),
        adapt_grid=cfg.sampler.get('adapt_grid', 16),
        adapt_hidden=cfg.sampler.get('adapt_hidden', 64),
        bank_size=cfg.sampler.get('bank_size', 8),
        gate_temperature=cfg.sampler.get('gate_temperature', 1.0),
        gate_logit_limit=cfg.sampler.get('gate_logit_limit', 0.0),
        gate_lr=cfg.sampler.get('gate_lr', None),
        init_mask_checkpoint=cfg.sampler.get('init_mask_checkpoint', None),
        loupe_vd_init_power=cfg.sampler.get('loupe_vd_init_power', 0.0),
        loupe_same_start=cfg.sampler.get('loupe_same_start', False),
        sampler_lr=cfg.sampler.get('lr', None),
        adapt_head_lr=cfg.sampler.get('adapt_head_lr', 1e-4),
        sampler_chans=cfg.sampler.get('chans', 64),
        sampler_depth=cfg.sampler.get('depth', 4),
        sampler_head=cfg.sampler.get('head', 'none'),
        sampler_head_mlp_lr=cfg.sampler.get('head_mlp_lr', 1e-4),
        sampler_image_input=_sampler_image_input(cfg.sampler),
        caipi_keep_fraction=cfg.sampler.get('caipi_keep_fraction', 1.0),
        caipi_accel_y=cfg.sampler.get('caipi_accel_y', 2),
        caipi_accel_z=cfg.sampler.get('caipi_accel_z', 2),
        preselect_num=cfg.sampler.preselect_num,
        mask_train_mode=cfg.sampler.get('mask_train_mode', 'straight_through'),
        reinforce_weight=cfg.sampler.get('reinforce_weight', 1.0),
        reinforce_entropy_weight=cfg.sampler.get('reinforce_entropy_weight', 0.0),
        reinforce_normalize_advantages=cfg.sampler.get('reinforce_normalize_advantages', False),
        reinforce_group_size=cfg.sampler.get('reinforce_group_size', 0),
        reinforce_vd_power=cfg.sampler.get('reinforce_vd_power', 0.0),
        reinforce_eval=cfg.sampler.get('reinforce_eval', 'sample'),
        reinforce_lr=cfg.sampler.get('reinforce_lr', None),
        autosamp_init=cfg.sampler.get('autosamp_init', 'uniform'),
        autosamp_lr=cfg.sampler.get('autosamp_lr', 0.01),
        autosamp_kernel_sigma_start=cfg.sampler.get('autosamp_kernel_sigma_start', 0.0),
        autosamp_kernel_sigma_end=cfg.sampler.get('autosamp_kernel_sigma_end', 1.0),
        autosamp_anneal_fraction=cfg.sampler.get('autosamp_anneal_fraction', 0.8),
        ppo_policy=cfg.sampler.get('ppo_policy', None),
        fixed_mask_path=cfg.sampler.get('fixed_mask_path', None),
        mnet_accept=cfg.sampler.get('mnet_accept', 'q_check'),
        mnet_compare_random=cfg.sampler.get('mnet_compare_random', True),
        mnet_rollback=cfg.sampler.get('mnet_rollback', True),
        mnet_degeneracy_fallback=cfg.sampler.get('mnet_degeneracy_fallback', True),
        mnet_label_binarize=cfg.sampler.get('mnet_label_binarize', 'threshold'),
        mnet_random_retry=cfg.sampler.get('mnet_random_retry', False),
        mnet_label_steps=cfg.sampler.get('mnet_label_steps', 20),
        mnet_sampler_steps=cfg.sampler.get('mnet_sampler_steps', 40),
        mnet_sparsity_weight=cfg.sampler.get('mnet_sparsity_weight', 2e-5),
        mnet_consistency_weight=cfg.sampler.get('mnet_consistency_weight', 0.05),
        mnet_init_margin=cfg.sampler.get('mnet_init_margin', 0.1),
        mnet_label_lr=cfg.sampler.get('mnet_label_lr', None),
        freeze_recon=cfg.sampler.get('freeze_recon', False),
        fixed_mask_checkpoints=dict(cfg.sampler.fixed_mask_checkpoints) if cfg.sampler.get('fixed_mask_checkpoints') else None,
        budget_over=cfg.sampler.get('budget_over', 'grid'),
        sampler_coil_combine=cfg.sampler.get('coil_combine', 'sense'),
        sense_mask_method=cfg.sampler.get('sense_mask_method', 'pre_network'),
        with_recon=cfg.sampler.get('with_recon', False),
        with_gfactor=cfg.sampler.get('with_gfactor', False),
        gfactor_replicas=cfg.sampler.get('gfactor_replicas', 16),
        sens_upsample_method=cfg.sampler.get('sens_upsample_method', 'conv'),
        sens_conv_after_upsample=cfg.sampler.get('sens_conv_after_upsample', False),
        sens_resolution=cfg.sampler.get('sens_resolution', None),
        sens_precision=cfg.sampler.get('sens_precision', 'fp32'),
        sens_norm=cfg.sampler.get('sens_norm', 'instance'),
        uncertainty_mode=cfg.sampler.get('uncertainty_mode', 'none'),
        min_energy_fraction=cfg.datasets.get('min_energy_fraction', 0.2),
        ssdu=cfg.datasets.get('ssdu', False),
        pseudo_target=cfg.datasets.get('pseudo_target', False),
        recon=build_recon_config(cfg),
        resolution=list(cfg.misc.resolution),
        random_rotate=cfg.misc.random_rotate,
        phase_scramble_std=cfg.misc.get('phase_scramble_std', 0.0),
        phase_scramble_band=list(cfg.misc.get('phase_scramble_band', [6.0, 16.0])),
        seed=cfg.training.get('seed', None),
    )

    if config.num_step == 0:
        config.fixed_input = True
        config.num_step = 1

    return config


def build_siemens_space_loaders(cfg: DictConfig):
    cache = Path(cfg.datasets.data_dir).expanduser()
    volumes = cfg.datasets.get('volumes') or sorted(path.stem for path in cache.glob('*.h5'))
    return build_siemens_space_dataloaders(
        [cache / f'{volume}.h5' for volume in volumes],
        image_shape=tuple(cfg.misc.resolution),
        batch_size=cfg.training.batch_size,
        holdout_fraction=cfg.datasets.holdout_fraction,
        num_workers=cfg.datasets.num_workers,
        val_volumes=list(cfg.datasets.get('val_volumes') or []),
    )


def uses_manifest(cfg: DictConfig) -> bool:
    return cfg.datasets.get('loader', 'fastmri') == 'fastmri_manifest'


def build_manifest_loaders(cfg: DictConfig, image_shape: list, shuffle_seed=None):
    return build_manifest_dataloaders(
        manifest=cfg.datasets.manifest,
        roots=dict(cfg.datasets.roots),
        classes=list(cfg.datasets.classes),
        image_shape=image_shape,
        batch_size=cfg.training.batch_size,
        num_coils=cfg.datasets.num_coils,
        train_tier=cfg.datasets.get('train_tier', 'S'),
        num_workers=cfg.datasets.num_workers,
        min_energy_fraction=cfg.datasets.get('min_energy_fraction', 0.2),
        shuffle_seed=shuffle_seed,
        train_anatomies=list(cfg.datasets.get('train_anatomies', [])),
        limit_volumes=cfg.datasets.get('limit_volumes', None),
        val_anatomies=list(cfg.datasets.get('val_anatomies', None) or []),
        val_limit_volumes=cfg.datasets.get('val_limit_volumes', None),
    )


def build_dataloaders(cfg: DictConfig):
    if cfg.datasets.get('loader', 'fastmri') == 'siemens_space':
        return build_siemens_space_loaders(cfg)
    if uses_manifest(cfg):
        return build_manifest_loaders(cfg, list(cfg.misc.resolution), shuffle_seed=cfg.training.seed)
    return build_fastmri_dataloaders(
        data_root=Path(cfg.datasets.data_root),
        anatomies=list(cfg.datasets.anatomy),
        image_shape=list(cfg.misc.resolution),
        batch_size=cfg.training.batch_size,
        num_coils=cfg.datasets.num_coils,
        num_workers=cfg.datasets.num_workers,
        contrast=cfg.datasets.get('contrast', None),
        data_dir=cfg.datasets.get('data_dir', None),
        min_energy_fraction=cfg.datasets.get('min_energy_fraction', 0.2),
        balance=cfg.datasets.get('balance', False),
        train_anatomies=list(cfg.datasets.get('train_anatomies', [])),
        train_slices=cfg.datasets.get('train_slices', None),
        sense_cache_dir=cfg.datasets.get('sense_cache_dir', None),
        balance_by=cfg.datasets.get('balance_by', 'anatomy'),
        test_fraction=cfg.datasets.get('test_fraction', 0.0),
        trim_to_smallest=cfg.datasets.get('trim_to_smallest', True),
        shuffle_seed=cfg.training.seed,
    )
