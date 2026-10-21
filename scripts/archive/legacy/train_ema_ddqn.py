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
from activemri.training.joint_trainer import JointTrainer, JointTrainConfig


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
            recon_checkpoint=cfg.ema_dqn.pretrained_checkpoint,
            num_cascades=cfg.recon.num_cascades,
            chans=cfg.recon.chans,
            pools=cfg.recon.pools,
            sense_mask_method=cfg.env.sense_mask_method,
            device=cfg.device,
        )

    return _make_env(train_loader), _make_env(val_loader)


def build_joint_config(cfg: DictConfig) -> JointTrainConfig:
    return JointTrainConfig(
        num_train_steps=cfg.ema_dqn.num_train_steps,
        mem_capacity=cfg.ema_dqn.mem_capacity,
        dqn_batch_size=cfg.ema_dqn.dqn_batch_size,
        dqn_burn_in=cfg.ema_dqn.dqn_burn_in,
        dqn_normalize=cfg.ema_dqn.dqn_normalize,
        dqn_learning_rate=cfg.ema_dqn.dqn_learning_rate,
        gamma=cfg.ema_dqn.gamma,
        target_net_update_freq=cfg.ema_dqn.target_net_update_freq,
        epsilon_start=cfg.ema_dqn.epsilon_start,
        epsilon_end=cfg.ema_dqn.epsilon_end,
        epsilon_decay=cfg.ema_dqn.epsilon_decay,
        val_episode_freq=cfg.ema_dqn.val_episode_freq,
        num_val_episodes=cfg.ema_dqn.num_val_episodes,
        checkpoint_freq=cfg.ema_dqn.checkpoint_freq,
        chans=cfg.ema_dqn.chans,
        num_pools=cfg.ema_dqn.num_pools,
        backbone_cascade_idx=cfg.ema_dqn.backbone_cascade_idx,
        pretrained_checkpoint=cfg.ema_dqn.pretrained_checkpoint,
        recon_learning_rate=cfg.ema_dqn.recon_learning_rate,
        recon_ema_decay=cfg.ema_dqn.recon_ema_decay,
        recon_updates_per_step=cfg.ema_dqn.recon_updates_per_step,
        out_dir=cfg.output.out_dir,
        resume=cfg.ema_dqn.resume,
        seed=cfg.training.seed,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="train_ema_ddqn")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    train_env, val_env = build_envs(cfg)
    joint_config = build_joint_config(cfg)
    trainer = JointTrainer(joint_config, train_env, val_env, torch.device(cfg.device))
    trainer()


if __name__ == "__main__":
    main()
