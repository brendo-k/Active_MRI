import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import _path  # noqa: F401
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.envs.mri_env import MRIEnv
from activemri.training.dqn_trainer import DQNTrainer, DQNTrainConfig


def build_envs(cfg: DictConfig) -> tuple[MRIEnv, MRIEnv]:
    train_loader, val_loader = build_fastmri_dataloaders(
        data_root=Path(cfg.datasets.data_root),
        anatomies=list(cfg.datasets.anatomy),
        image_shape=list(cfg.env.resolution),
        batch_size=cfg.training.batch_size,
        num_coils=cfg.datasets.num_coils,
        num_workers=cfg.datasets.num_workers,
        contrast=cfg.datasets.get('contrast', None),
        data_dir=cfg.datasets.get('data_dir', None),
    )

    budget = int(round(cfg.env.resolution[1] / cfg.env.acceleration)) - cfg.env.preselect_num

    def _make_env(loader):
        return MRIEnv(
            loader=loader,
            resolution=list(cfg.env.resolution),
            budget=budget,
            preselect_num=cfg.env.preselect_num,
            recon_checkpoint=cfg.dqn.pretrained_checkpoint,
            num_cascades=cfg.recon.num_cascades,
            chans=cfg.recon.chans,
            pools=cfg.recon.pools,
            sense_mask_method=cfg.env.sense_mask_method,
            device=cfg.device,
        )

    return _make_env(train_loader), _make_env(val_loader)


def build_dqn_config(cfg: DictConfig) -> DQNTrainConfig:
    return DQNTrainConfig(
        num_train_steps=cfg.dqn.num_train_steps,
        mem_capacity=cfg.dqn.mem_capacity,
        dqn_batch_size=cfg.dqn.dqn_batch_size,
        dqn_burn_in=cfg.dqn.dqn_burn_in,
        dqn_normalize=cfg.dqn.dqn_normalize,
        dqn_learning_rate=cfg.dqn.dqn_learning_rate,
        gamma=cfg.dqn.gamma,
        target_net_update_freq=cfg.dqn.target_net_update_freq,
        epsilon_start=cfg.dqn.epsilon_start,
        epsilon_end=cfg.dqn.epsilon_end,
        epsilon_decay=cfg.dqn.epsilon_decay,
        val_episode_freq=cfg.dqn.val_episode_freq,
        num_val_episodes=cfg.dqn.num_val_episodes,
        checkpoint_freq=cfg.dqn.checkpoint_freq,
        chans=cfg.dqn.chans,
        num_pools=cfg.dqn.num_pools,
        backbone_cascade_idx=cfg.dqn.backbone_cascade_idx,
        pretrained_checkpoint=cfg.dqn.pretrained_checkpoint,
        out_dir=cfg.output.out_dir,
        resume=cfg.dqn.resume,
        seed=cfg.training.seed,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="train_dqn")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    train_env, val_env = build_envs(cfg)
    dqn_config = build_dqn_config(cfg)
    trainer = DQNTrainer(dqn_config, train_env, val_env, torch.device(cfg.device))
    trainer()


if __name__ == "__main__":
    main()
