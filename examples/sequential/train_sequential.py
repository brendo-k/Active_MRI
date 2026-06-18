import os
import sys
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from activemri.baselines.sequential_trainer import Sequential_Trainer
from activemri.config import TrainConfig
from activemri.reconstructors.config import UNetReconConfig, CGSenseReconConfig
from activemri.data.fastmri_data import build_fastmri_dataloaders

def build_recon_config(cfg: DictConfig):
    if cfg.recon.recon_type == 'unet':
        return UNetReconConfig(
            num_pools=cfg.recon.num_pools,
            num_chans=cfg.recon.num_chans,
            drop_prob=cfg.recon.drop_prob,
            in_chans=cfg.recon.in_chans,
            out_chans=cfg.recon.out_chans,
        )
    elif cfg.recon.recon_type == 'cg_sense':
        return CGSenseReconConfig(
            cg_iters=cfg.recon.cg_iters,
            cg_lamda=cfg.recon.cg_lamda,
            sense_tikhonov_weight=cfg.recon.sense_tikhonov_weight,
        )
    else:
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
        out_dir=[Path(cfg.output.out_dir)],
        checkpoint=cfg.output.checkpoint,
        accelerations=list(cfg.sampler.accelerations),
        line_constrained=cfg.sampler.line_constrained,
        preselect_num=cfg.sampler.preselect_num,
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
        data_root=Path(cfg.data.data_root),
        anatomies=list(cfg.data.anatomy),
        image_shape=list(cfg.misc.resolution),
        batch_size=cfg.training.batch_size,
        num_coils=cfg.data.num_coils,
        num_workers=cfg.data.num_workers,
        contrast=cfg.data.get('contrast', None),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
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
