"""Config and dataloader builders for Sequential_Trainer (the loupe / conv / reinforce samplers)."""
from pathlib import Path

from omegaconf import DictConfig

from activemri.config import TrainConfig
from activemri.reconstructors.config import UNetReconConfig, CGSenseReconConfig, VarNetReconConfig
from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.datasets.siemens_space_data import build_siemens_space_dataloaders

RECON_CONFIGS = {
    'unet': UNetReconConfig,
    'varnet': VarNetReconConfig,
    'cg_sense': CGSenseReconConfig,
}

def build_recon_config(cfg: DictConfig):
    for config_name, config_class in RECON_CONFIGS.items():
        if cfg.recon.recon_type == config_name:
            return config_class(**cfg.recon)
    raise ValueError(f"Unknown recon_type: {cfg.recon.recon_type}")


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
        limit_batches=cfg.training.limit_batches,
        data_parallel=cfg.training.data_parallel,
        loss_type=cfg.loss.loss_type,
        uncertainty_loss=cfg.loss.uncertainty_loss,
        uncertainty_weight=cfg.loss.uncertainty_weight,
        anatomy_weight=cfg.loss.get('anatomy_weight', 0.0),
        anatomy_classes=list(cfg.datasets.anatomy),
        condition_classes=list(cfg.datasets.get('condition_classes', [])),
        train_anatomies=list(cfg.datasets.get('train_anatomies', [])),
        anatomy_conditioning=cfg.sampler.get('anatomy_conditioning', False),
        out_dir=[Path(cfg.output.out_dir)],
        checkpoint=cfg.output.checkpoint,
        pretrained_recon=cfg.output.get('pretrained_recon', None),
        accelerations=list(cfg.sampler.accelerations),
        sampler_type=cfg.sampler.sampler_type,
        slope=cfg.sampler.get('slope', 10.0),
        mask_grad_floor=cfg.sampler.get('mask_grad_floor', 0.0),
        ste_grad_floor=cfg.sampler.get('ste_grad_floor', 0.0),
        sampler_chans=cfg.sampler.get('chans', 64),
        sampler_depth=cfg.sampler.get('depth', 4),
        sampler_image_input=cfg.sampler.get('image_input', False),
        caipi_keep_fraction=cfg.sampler.get('caipi_keep_fraction', 1.0),
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
        freeze_recon=cfg.sampler.get('freeze_recon', False),
        fixed_mask_checkpoints=dict(cfg.sampler.fixed_mask_checkpoints) if cfg.sampler.get('fixed_mask_checkpoints') else None,
        sense_mask_method=cfg.sampler.get('sense_mask_method', 'pre_network'),
        with_recon=cfg.sampler.get('with_recon', False),
        with_gfactor=cfg.sampler.get('with_gfactor', False),
        gfactor_replicas=cfg.sampler.get('gfactor_replicas', 16),
        uncertainty_mode=cfg.sampler.get('uncertainty_mode', 'none'),
        min_energy_fraction=cfg.datasets.get('min_energy_fraction', 0.2),
        ssdu=cfg.datasets.get('ssdu', False),
        pseudo_target=cfg.datasets.get('pseudo_target', False),
        recon=build_recon_config(cfg),
        resolution=list(cfg.misc.resolution),
        random_rotate=cfg.misc.random_rotate,
        phase_scramble_std=cfg.misc.get('phase_scramble_std', 0.0),
        phase_scramble_band=list(cfg.misc.get('phase_scramble_band', [6.0, 16.0])),
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


def build_dataloaders(cfg: DictConfig):
    if cfg.datasets.get('loader', 'fastmri') == 'siemens_space':
        return build_siemens_space_loaders(cfg)
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
    )
