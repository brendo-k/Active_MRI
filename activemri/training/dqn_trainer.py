"""Double-DQN trainer for the VarNet-backboned line sampler.

Port of `active-mri-acquisition`'s `activemri.baselines.ddqn.DDQNTrainer`
training loop (epsilon schedule, replay buffer, target-network sync) onto
`activemri.envs.mri_env.MRIEnv` + `activemri.samplers.dqn.DQNPolicy`. Unlike
`Sequential_Trainer`, an episode here is many environment steps (one action
per step) rather than one differentiable forward+backward per batch.
"""
import copy
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import wandb

from activemri.envs.mri_env import MRIEnv
from activemri.samplers.dqn import DQNPolicy
from activemri.samplers.replay_buffer import ReplayMemory


@dataclass
class DQNTrainConfig:
    num_train_steps: int = 200_000
    mem_capacity: int = 100_000
    dqn_batch_size: int = 32
    dqn_burn_in: int = 1_000
    dqn_normalize: bool = False
    dqn_learning_rate: float = 1e-4
    gamma: float = 0.99
    target_net_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 50_000.0
    val_episode_freq: int = 50
    num_val_episodes: int = 20
    checkpoint_freq: int = 50
    chans: int = 18
    num_pools: int = 4
    backbone_cascade_idx: int = -1
    pretrained_checkpoint: Optional[str] = None
    out_dir: str = "checkpoints/dqn"
    resume: bool = False
    seed: Optional[int] = None


def _epsilon(step: int, cfg: DQNTrainConfig) -> float:
    return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-step / cfg.epsilon_decay)


class DQNTrainer:
    def __init__(self, options: DQNTrainConfig, env: MRIEnv, val_env: Optional[MRIEnv], device: torch.device):
        self.options = options
        self.env = env
        self.val_env = val_env if val_env is not None else env
        self.device = device
        self.steps = 0
        self.episode = 0
        self.best_val_score = -np.inf

        os.makedirs(options.out_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        obs_shape = (5, *env.shape)
        self.replay_memory = ReplayMemory(
            options.mem_capacity, obs_shape, options.dqn_batch_size, options.dqn_burn_in,
            use_normalization=options.dqn_normalize,
        )
        self.policy = DQNPolicy(
            env.num_lines, device, memory=self.replay_memory,
            chans=options.chans, num_pools=options.num_pools,
            gamma=options.gamma, dqn_learning_rate=options.dqn_learning_rate,
            pretrained_checkpoint=options.pretrained_checkpoint,
            backbone_cascade_idx=options.backbone_cascade_idx,
        )
        self.target_net = DQNPolicy(
            env.num_lines, device, memory=None,
            chans=options.chans, num_pools=options.num_pools,
        )
        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()

        wandb.init(project="SeqMRI-DQN", config=vars(options), dir=options.out_dir)

        if options.resume:
            self._load_if_present()

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.options.out_dir, name)

    def _load_if_present(self):
        path = self._checkpoint_path("latest.pt")
        if not os.path.isfile(path):
            self.logger.info("No DQN checkpoint found at %s; starting from scratch.", path)
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.steps = checkpoint["steps"]
        self.episode = checkpoint["episode"]
        self.best_val_score = checkpoint["best_val_score"]
        self.logger.info("Resumed DQN training from %s at episode %d.", path, self.episode)

    def save(self, name: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target_net.state_dict(),
                "steps": self.steps,
                "episode": self.episode,
                "best_val_score": self.best_val_score,
                "options": self.options,
            },
            self._checkpoint_path(name),
        )

    @torch.no_grad()
    def evaluate(self, num_episodes: int) -> float:
        self.policy.model.eval()
        scores = []
        for _ in range(num_episodes):
            obs, _ = self.val_env.reset()
            done = False
            while not done:
                actions = self.policy.get_action(obs, eps_threshold=0.0)
                obs, _, dones, meta = self.val_env.step(actions)
                done = all(dones)
            scores.append(float(np.mean(meta["current_score"]["nmse"])))
        return -float(np.mean(scores))  # higher is better (lower NMSE)

    def __call__(self) -> float:
        self.logger.info("Starting DQN training at step %d/%d.", self.steps, self.options.num_train_steps)

        while self.steps < self.options.num_train_steps:
            if self.options.val_episode_freq and self.episode % self.options.val_episode_freq == 0:
                val_score = self.evaluate(self.options.num_val_episodes)
                wandb.log({"val/score": val_score, "episode": self.episode}, step=self.steps)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.save("best.pt")
                    self.logger.info("New best DQN val score %.6f at episode %d.", val_score, self.episode)

            if self.episode % self.options.checkpoint_freq == 0:
                self.save("latest.pt")

            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                epsilon = _epsilon(self.steps, self.options)
                actions = self.policy.get_action(obs, eps_threshold=epsilon)
                next_obs, reward, dones, _ = self.env.step(actions)
                done = all(dones)
                self.steps += 1

                for i in range(obs.shape[0]):
                    self.policy.add_experience(
                        obs[i].cpu(), actions[i], next_obs[i].cpu(), reward[i].item(), dones[i],
                    )

                update_result = self.policy.update_parameters(self.target_net)
                if self.steps % self.options.target_net_update_freq == 0:
                    self.target_net.load_state_dict(self.policy.state_dict())

                if update_result is not None and self.steps % 250 == 0:
                    wandb.log({"train/epsilon": epsilon, **{f"train/{k}": v for k, v in update_result.items()}}, step=self.steps)

                episode_reward += float(reward.mean().item())
                obs = next_obs

            wandb.log({"train/episode_reward": episode_reward, "episode": self.episode}, step=self.steps)
            self.episode += 1

        self.save("latest.pt")
        wandb.finish()
        return self.best_val_score
