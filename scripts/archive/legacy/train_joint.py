import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import _path  # noqa: F401
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from activemri.training.joint_trainer import JointTrainer, JointTrainConfig
from activemri.training.rl_builders import build_dqn_envs as build_envs


def build_joint_config(cfg: DictConfig) -> JointTrainConfig:
    return JointTrainConfig(
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
        recon_learning_rate=cfg.joint.recon_learning_rate,
        recon_ema_decay=cfg.joint.recon_ema_decay,
        recon_updates_per_step=cfg.joint.recon_updates_per_step,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="train_joint")
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
