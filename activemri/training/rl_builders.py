"""Environment and config builders for the PPO and DQN trainers (an MRIEnv episode is many steps)."""
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from activemri.datasets.fastmri_data import build_fastmri_dataloaders
from activemri.envs.mri_env import MRIEnv
from activemri.training.dqn_trainer import DQNTrainConfig
from activemri.training.ppo_trainer import PPOTrainConfig


def episode_steps(cfg: DictConfig) -> int:
    """Steps needed to go from the preselected region to `1 / acceleration` of k-space."""
    height, width = cfg.env.resolution
    if cfg.env.action_space == "point":
        total, preselected = height * width, cfg.env.center_block**2
    else:
        total, preselected = width, cfg.env.preselect_num
    return int(round((total / cfg.env.acceleration - preselected) / cfg.env.points_per_step))


def evenly_spaced_val_loader(loader: DataLoader, num_slices: int) -> DataLoader:
    """A fixed, shuffle-free val loader over `num_slices` slices spread across the whole val set.

    The val set is brain slices followed by knee slices, so taking the first batches would only score brain.
    """
    indices = np.linspace(0, len(loader.dataset) - 1, num_slices).round().astype(int).tolist()
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=min(loader.num_workers, 4),
        collate_fn=loader.collate_fn,
    )


def build_ppo_envs(cfg: DictConfig) -> tuple[MRIEnv, MRIEnv]:
    train_loader, val_loader = build_fastmri_dataloaders(
        data_root=Path(cfg.datasets.data_root),
        anatomies=list(cfg.datasets.anatomy),
        image_shape=list(cfg.env.resolution),
        batch_size=cfg.training.batch_size,
        num_coils=cfg.datasets.num_coils,
        num_workers=cfg.datasets.num_workers,
        contrast=cfg.datasets.get('contrast', None),
        data_dir=cfg.datasets.get('data_dir', None),
        min_energy_fraction=cfg.datasets.get('min_energy_fraction', 0.2),
        balance=cfg.datasets.get('balance', False),
        train_anatomies=list(cfg.datasets.get('train_anatomies', [])),
    )
    val_loader = evenly_spaced_val_loader(val_loader, cfg.ppo.num_val_episodes * cfg.training.batch_size)

    budget = episode_steps(cfg)

    def _make_env(loader):
        return MRIEnv(
            loader=loader,
            resolution=list(cfg.env.resolution),
            budget=budget,
            preselect_num=cfg.env.preselect_num,
            recon_checkpoint=cfg.ppo.pretrained_checkpoint,
            num_cascades=cfg.recon.num_cascades,
            chans=cfg.recon.chans,
            pools=cfg.recon.pools,
            sense_mask_method=cfg.env.sense_mask_method,
            device=cfg.device,
            reward_mode=cfg.env.reward_mode,
            action_space=cfg.env.action_space,
            points_per_step=cfg.env.points_per_step,
            center_block=cfg.env.center_block,
            observation=cfg.env.get('observation', 'kspace'),
        )

    return _make_env(train_loader), _make_env(val_loader)


def build_ppo_config(cfg: DictConfig) -> PPOTrainConfig:
    ppo = OmegaConf.to_container(cfg.ppo, resolve=True)
    return PPOTrainConfig(**ppo, out_dir=cfg.output.out_dir, seed=cfg.training.seed)


def build_dqn_envs(cfg: DictConfig) -> tuple[MRIEnv, MRIEnv]:
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
