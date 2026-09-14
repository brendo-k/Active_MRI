import os
import sys
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from activemri.training.trainer import Sequential_Trainer
from activemri.config import TrainConfig
from activemri.reconstructors.config import UNetReconConfig, CGSenseReconConfig, VarNetReconConfig
from activemri.datasets.fastmri_data import build_fastmri_dataloaders

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
        weight_decay=cfg.training.weight_decay,
        limit_batches=cfg.training.limit_batches,
        data_parallel=cfg.training.data_parallel,
        loss_type=cfg.loss.loss_type,
        uncertainty_loss=cfg.loss.uncertainty_loss,
        uncertainty_weight=cfg.loss.uncertainty_weight,
        anatomy_weight=cfg.loss.get('anatomy_weight', 0.0),
        anatomy_classes=list(cfg.datasets.anatomy) if cfg.loss.get('anatomy_weight', 0.0) > 0 else [],
        out_dir=[Path(cfg.output.out_dir)],
        checkpoint=cfg.output.checkpoint,
        accelerations=list(cfg.sampler.accelerations),
        sampler_type=cfg.sampler.sampler_type,
        slope=cfg.sampler.get('slope', 5.0),
        preselect_num=cfg.sampler.preselect_num,
        sense_mask_method=cfg.sampler.get('sense_mask_method', 'pre_network'),
        with_recon=cfg.sampler.get('with_recon', False),
        with_gfactor=cfg.sampler.get('with_gfactor', False),
        gfactor_replicas=cfg.sampler.get('gfactor_replicas', 16),
        uncertainty_mode=cfg.sampler.get('uncertainty_mode', 'none'),
        min_energy_fraction=cfg.datasets.get('min_energy_fraction', 0.2),
        recon=build_recon_config(cfg),
        resolution=list(cfg.misc.resolution),
        random_rotate=cfg.misc.random_rotate,
    )

    if config.num_step == 0:
        config.fixed_input = True
        config.num_step = 1

    return config


def build_dataloaders(cfg: DictConfig):
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
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    config = build_train_config(cfg)

    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    train_loader, val_loader = build_dataloaders(cfg)
    trainer = Sequential_Trainer(config, train_loader, torch.device(config.device), val_loader=val_loader)
    trainer()


if __name__ == "__main__":
    main()
