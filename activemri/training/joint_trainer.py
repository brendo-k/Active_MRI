"""Joint DDQN + reconstructor training via an EMA-lagged reward network.

`DQNTrainer` treats `MRIEnv.reconstructor` as fixed -- it's loaded once from
a checkpoint and never updated, so the DQN reward (an NMSE delta computed by
that reconstructor, `mri_env._nmse`) is a stationary function of the mask
throughout training. That's also its limitation: the reconstructor never
adapts to the mask *distribution* the policy actually learns to produce.

`JointTrainer` removes that limitation while keeping the reward
(approximately) stationary from the replay buffer's point of view:

- An `online_reconstructor` (a second `VarNet`, same architecture) is fit by
  ordinary supervised gradient descent every env step, on exactly the mask
  the policy just chose (`env.state["mask"]`) against the ground-truth
  target -- so it tracks the policy's actual sampling distribution instead
  of whatever distribution the original checkpoint was pretrained on.
- `env.reconstructor` -- the copy `MRIEnv.step` actually scores reward
  against -- is *not* trained directly. Instead it's nudged towards
  `online_reconstructor` by a parameter-wise EMA after every gradient step
  (`activemri.training.ema.EMA`). With `recon_ema_decay` close to 1, its
  weights move slowly enough that transitions sampled from the replay
  buffer (up to `mem_capacity` steps old) still have approximately correct
  rewards, so the buffer never needs to be cleared and the two networks
  train in a single interleaved loop instead of alternating phases.

The sensitivity net is left frozen throughout (mirroring `MRIEnv`'s
existing assumption) -- only the VarNet cascades are jointly trained.
"""
import copy
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from activemri.envs.mri_env import MRIEnv
from activemri.samplers.dqn import DQNPolicy
from activemri.samplers.replay_buffer import ReplayMemory
from activemri.training.dqn_trainer import DQNTrainConfig, _epsilon
from activemri.training.ema import EMA


@dataclass
class JointTrainConfig(DQNTrainConfig):
    recon_learning_rate: float = 1e-4
    recon_ema_decay: float = 0.999
    # Gradient steps on the online reconstructor per env step. >1 lets the
    # reconstructor keep pace with a fast-moving policy early in training.
    recon_updates_per_step: int = 1


class JointTrainer:
    def __init__(self, options: JointTrainConfig, env: MRIEnv, val_env: Optional[MRIEnv], device: torch.device):
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

        # Online reconstructor: same architecture/init as env.reconstructor
        # (which already holds the pretrained weights), but trainable.
        # env.reconstructor itself becomes the EMA/"target" copy used for
        # reward -- its params are updated in place by `self.ema`, never by
        # its own optimizer.
        self.online_reconstructor = copy.deepcopy(env.reconstructor).to(device)
        self.online_reconstructor.load_state_dict(env.reconstructor.state_dict())
        for p in self.online_reconstructor.parameters():
            p.requires_grad_(True)
        self.online_reconstructor.train()
        self.recon_optimizer = torch.optim.Adam(
            self.online_reconstructor.parameters(), lr=options.recon_learning_rate,
        )
        self.ema = EMA(options.recon_ema_decay)

        # Ensure env.reconstructor is on the correct device and in eval mode
        # (it won't be trained directly, only updated via EMA)
        env.reconstructor.to(device)
        env.reconstructor.eval()
        for p in env.reconstructor.parameters():
            p.requires_grad_(False)

        wandb.init(project="SeqMRI-Joint", config=vars(options), dir=options.out_dir)

        if options.resume:
            self._load_if_present()

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.options.out_dir, name)

    def _load_if_present(self):
        path = self._checkpoint_path("latest.pt")
        if not os.path.isfile(path):
            self.logger.info("No joint checkpoint found at %s; starting from scratch.", path)
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.online_reconstructor.load_state_dict(checkpoint["online_reconstructor"])
        self.recon_optimizer.load_state_dict(checkpoint["recon_optimizer"])
        self.env.load_reconstructor_state(checkpoint["env_reconstructor"], self.env.sensetivity_net.state_dict())
        self.steps = checkpoint["steps"]
        self.episode = checkpoint["episode"]
        self.best_val_score = checkpoint["best_val_score"]
        self.logger.info("Resumed joint training from %s at episode %d.", path, self.episode)

    def save(self, name: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target_net.state_dict(),
                "online_reconstructor": self.online_reconstructor.state_dict(),
                "recon_optimizer": self.recon_optimizer.state_dict(),
                "env_reconstructor": self.env.reconstructor.state_dict(),
                "steps": self.steps,
                "episode": self.episode,
                "best_val_score": self.best_val_score,
                "options": self.options,
            },
            self._checkpoint_path(name),
        )

    def _recon_train_step(self) -> float:
        """One supervised gradient step for `online_reconstructor` on the
        mask the policy just chose, then EMAs the result into
        `env.reconstructor`. Returns the L1 loss."""
        state = self.env.state
        mask, full_kspace, sense_maps, target = (
            state["mask"], state["full_kspace"], state["sense_maps"], state["target"],
        )

        mask = mask.to(self.device).float()
        full_kspace = full_kspace.to(self.device)
        sense_maps = sense_maps.to(self.device)
        target = target.to(self.device)

        masked_kspace = torch.view_as_complex((mask.unsqueeze(-1) * full_kspace).contiguous())

        self.online_reconstructor.train()
        recon = self.online_reconstructor(masked_kspace, sense_maps, mask)
        target = target.squeeze(1) if target.dim() == 4 else target
        loss = (recon.squeeze(1) - target).abs().mean()

        self.recon_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_reconstructor.parameters(), 1)
        self.recon_optimizer.step()

        self.ema.update(self.env.reconstructor, self.online_reconstructor)
        return loss.item()

    @torch.no_grad()
    def evaluate(self, num_episodes: int) -> float:
        """Scored against the env's current EMA reconstructor -- i.e. the
        same reward network the policy is training against right now."""
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
        return -float(np.mean(scores))

    def __call__(self) -> float:
        self.logger.info("Starting joint training at step %d/%d.", self.steps, self.options.num_train_steps)

        while self.steps < self.options.num_train_steps:
            if self.options.val_episode_freq and self.episode % self.options.val_episode_freq == 0:
                val_score = self.evaluate(self.options.num_val_episodes)
                wandb.log({"val/score": val_score, "episode": self.episode}, step=self.steps)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.save("best.pt")
                    self.logger.info("New best joint val score %.6f at episode %d.", val_score, self.episode)

            if self.episode % self.options.checkpoint_freq == 0:
                self.save("latest.pt")

            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            recon_losses = []
            while not done:
                epsilon = _epsilon(self.steps, self.options)
                actions = self.policy.get_action(obs, eps_threshold=epsilon)
                next_obs, reward, dones, _ = self.env.step(actions)
                done = all(dones)
                self.steps += 1

                for _ in range(self.options.recon_updates_per_step):
                    recon_losses.append(self._recon_train_step())

                for i in range(obs.shape[0]):
                    self.policy.add_experience(
                        obs[i].cpu(), actions[i], next_obs[i].cpu(), reward[i].item(), dones[i],
                    )

                update_result = self.policy.update_parameters(self.target_net)
                if self.steps % self.options.target_net_update_freq == 0:
                    self.target_net.load_state_dict(self.policy.state_dict())

                if self.steps % 250 == 0:
                    log = {"train/epsilon": epsilon, "train/recon_loss": np.mean(recon_losses[-250:])}
                    if update_result is not None:
                        log.update({f"train/{k}": v for k, v in update_result.items()})
                    wandb.log(log, step=self.steps)

                episode_reward += float(reward.mean().item())
                obs = next_obs

            wandb.log(
                {"train/episode_reward": episode_reward, "train/episode_recon_loss": np.mean(recon_losses),
                 "episode": self.episode},
                step=self.steps,
            )
            self.episode += 1

        self.save("latest.pt")
        wandb.finish()
        return self.best_val_score
