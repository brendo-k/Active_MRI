"""PPO trainer for the VarNet-backboned k-space sampler, alternating with reconstructor training.

On-policy counterpart to `DQNTrainer`, on the same `MRIEnv`. Training
alternates, as in L2SR (https://arxiv.org/abs/2212.02190), between:

- a **PPO phase**: roll out `episodes_per_update` full episodes with the
  stochastic policy, compute GAE advantages (episodes always run to the
  budget, so the bootstrap value after the last step is 0), then run several
  epochs of clipped-surrogate minibatch updates of the coupled actor-critic
  (`activemri.samplers.ppo.PPOPolicy`), against a *frozen* reconstructor;
- a **reconstructor phase**: fit a copy of the reconstructor with supervised
  L1 loss on the masks the current policy produces, then swap it into the
  train and val envs.

The first round trains the policy longest; later rounds shrink the step
count, policy learning rate and reconstructor learning rate. Each swap
changes the reward function, so the running reward scale is reset (the critic
is kept as a warm start).

Intended for `MRIEnv(reward_mode="sparse")` (L2SR's terminal-reward POMDP),
but works with `"dense"` too. `recon_steps_per_round=0` disables the
reconstructor phase (policy-only PPO against the checkpoint's reconstructor).
"""
import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from activemri.envs.mri_env import MRIEnv
from activemri.samplers.ppo import PPOPolicy


@dataclass
class PPOTrainConfig:
    # Alternation
    num_rounds: int = 3
    steps_per_round: int = 100_000  # env.step() calls in round 0
    later_round_step_fraction: float = 0.2  # rounds >= 1 use this fraction of steps_per_round
    round_lr_decay: float = 0.5  # per-round multiplier on the policy and reconstructor learning rates
    # PPO
    episodes_per_update: int = 2
    ppo_epochs: int = 4
    minibatch_size: int = 32
    clip_range: float = 0.2
    # With K points drawn per step the step's log-prob is a sum over K terms, so the
    # raw probability ratio swings ~K times harder than for a single action. True
    # applies the clip to the per-point (geometric-mean) ratio instead.
    per_point_ratio: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03  # stop an update's epochs early once mean approx-KL exceeds 1.5x this
    gamma: float = 1.0
    gae_lambda: float = 0.95
    learning_rate: float = 1e-4
    normalize_rewards: bool = True  # divide rewards by the running std of episode returns
    # Reconstructor phase
    recon_steps_per_round: int = 500  # gradient steps (each on a fresh batch of slices)
    recon_learning_rate: float = 1e-4
    recon_max_grad_norm: float = 1.0
    recon_policy_deterministic: bool = True  # masks from the greedy policy (as L2SR), not sampled
    # Evaluation / checkpointing
    val_update_freq: int = 10
    num_val_episodes: int = 10
    checkpoint_freq: int = 10
    # Network
    chans: int = 18
    num_pools: int = 4
    critic_hidden: int = 256
    backbone_cascade_idx: int = -1
    pretrained_checkpoint: Optional[str] = None
    out_dir: str = "checkpoints/ppo"
    resume: bool = False
    seed: Optional[int] = None

    def round_steps(self, round_idx: int) -> int:
        fraction = 1.0 if round_idx == 0 else self.later_round_step_fraction
        return int(self.steps_per_round * fraction)

    def round_lr_scale(self, round_idx: int) -> float:
        return self.round_lr_decay**round_idx


class RunningStd:
    """Running standard deviation (Welford/Chan merge) of a stream of scalars."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: torch.Tensor):
        n = values.numel()
        batch_mean = values.mean().item()
        batch_m2 = ((values - batch_mean) ** 2).sum().item()
        total = self.count + n
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / self.count)) if self.count > 1 else 1.0

    def state_dict(self) -> Dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: Dict[str, float]):
        self.count, self.mean, self.m2 = state["count"], state["mean"], state["m2"]


class Rollout:
    """Flat tensors for `N = episodes * steps * batch` transitions."""

    def __init__(self, obs, actions, log_probs, advantages, returns, values):
        self.obs, self.actions, self.log_probs = obs, actions, log_probs
        self.advantages, self.returns, self.values = advantages, returns, values

    def __len__(self) -> int:
        return self.actions.shape[0]


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over one finished episode. `rewards`, `values`: `[T, B]`; the value after step `T` is 0."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        next_value = values[t + 1] if t + 1 < rewards.shape[0] else torch.zeros_like(values[0])
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * gae_lambda * running
        advantages[t] = running
    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, options: PPOTrainConfig, env: MRIEnv, val_env: Optional[MRIEnv], device: torch.device):
        self.options = options
        self.env = env
        self.val_env = val_env if val_env is not None else env
        self.device = device
        self.steps = 0
        self.update_idx = 0
        self.round_idx = 0
        self.round_steps_done = 0
        self.best_val_score = -np.inf
        self.return_std = RunningStd()

        os.makedirs(options.out_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.policy = PPOPolicy(
            env.action_space, env.points_per_step, env.target_fraction, device,
            chans=options.chans, num_pools=options.num_pools, critic_hidden=options.critic_hidden,
            pretrained_checkpoint=options.pretrained_checkpoint,
            backbone_cascade_idx=options.backbone_cascade_idx,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=options.learning_rate)

        wandb.init(project="SeqMRI-PPO", config=vars(options), dir=options.out_dir)

        if options.resume:
            self._load_if_present()

    def _envs(self) -> List[MRIEnv]:
        return [self.env] if self.val_env is self.env else [self.env, self.val_env]

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.options.out_dir, name)

    def _load_if_present(self):
        path = self._checkpoint_path("latest.pt")
        if not os.path.isfile(path):
            self.logger.info("No PPO checkpoint found at %s; starting from scratch.", path)
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.return_std.load_state_dict(checkpoint["return_std"])
        for env in self._envs():
            env.load_reconstructor_state(checkpoint["reconstructor"], env.sensetivity_net.state_dict())
        self.steps = checkpoint["steps"]
        self.update_idx = checkpoint["update"]
        self.round_idx = checkpoint["round"]
        self.round_steps_done = checkpoint["round_steps_done"]
        self.best_val_score = checkpoint["best_val_score"]
        self.logger.info(
            "Resumed PPO training from %s at round %d, update %d.", path, self.round_idx, self.update_idx
        )

    def save(self, name: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "return_std": self.return_std.state_dict(),
                "reconstructor": self.env.reconstructor.state_dict(),
                "steps": self.steps,
                "update": self.update_idx,
                "round": self.round_idx,
                "round_steps_done": self.round_steps_done,
                "best_val_score": self.best_val_score,
                "options": self.options,
            },
            self._checkpoint_path(name),
        )

    @torch.no_grad()
    def evaluate(self, num_episodes: int) -> float:
        self.policy.eval()
        scores = []
        for _ in range(num_episodes):
            obs, _ = self.val_env.reset()
            done = False
            while not done:
                actions, _, _ = self.policy.act(obs, deterministic=True)
                obs, _, dones, meta = self.val_env.step(actions)
                done = all(dones)
            scores.append(float(np.mean(meta["current_score"]["nmse"])))
        return -float(np.mean(scores))  # higher is better (lower NMSE)

    def _validate(self):
        val_score = self.evaluate(self.options.num_val_episodes)
        wandb.log({"val/score": val_score, "update": self.update_idx, "round": self.round_idx}, step=self.steps)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.save("best.pt")
            self.logger.info("New best PPO val score %.6f at update %d.", val_score, self.update_idx)
        return val_score

    @torch.no_grad()
    def collect_rollout(self) -> tuple[Rollout, Dict[str, float]]:
        """Runs `episodes_per_update` episodes with the stochastic policy."""
        opts = self.options
        self.policy.eval()
        episodes = []
        final_nmse: List[float] = []
        for _ in range(opts.episodes_per_update):
            obs, _ = self.env.reset()
            step_obs, step_actions, step_log_probs, step_values, step_rewards = [], [], [], [], []
            done = False
            while not done:
                actions, log_probs, values = self.policy.act(obs)
                next_obs, reward, dones, meta = self.env.step(actions)
                step_obs.append(obs)
                step_actions.append(actions)
                step_log_probs.append(log_probs)
                step_values.append(values)
                step_rewards.append(reward.to(self.device))
                obs = next_obs
                done = all(dones)
                self.steps += 1
                self.round_steps_done += 1
            final_nmse.extend(meta["current_score"]["nmse"].tolist())
            episodes.append(tuple(torch.stack(x) for x in (
                step_obs, step_actions, step_log_probs, step_values, step_rewards,
            )))

        if opts.normalize_rewards:
            self.return_std.update(torch.cat([self._discounted_return(rewards) for *_, rewards in episodes]))
        reward_scale = 1.0 / max(self.return_std.std, 1e-8) if opts.normalize_rewards else 1.0

        batches = {k: [] for k in ("obs", "actions", "log_probs", "advantages", "returns", "values")}
        raw_returns = []
        for obs_t, actions_t, log_probs_t, values_t, rewards_t in episodes:
            advantages, returns = compute_gae(rewards_t * reward_scale, values_t, opts.gamma, opts.gae_lambda)
            raw_returns.append(rewards_t.sum(dim=0))
            for key, tensor in zip(batches, (obs_t, actions_t, log_probs_t, advantages, returns, values_t)):
                batches[key].append(tensor.flatten(0, 1))
        rollout = Rollout(**{k: torch.cat(v) for k, v in batches.items()})

        stats = {
            "train/episode_reward": torch.cat(raw_returns).mean().item(),
            "train/final_nmse": float(np.mean(final_nmse)),
            "train/reward_scale": reward_scale,
        }
        return rollout, stats

    def _discounted_return(self, rewards: torch.Tensor) -> torch.Tensor:
        """Per-episode discounted return from step 0: `[T, B] -> [B]`."""
        discounts = self.options.gamma ** torch.arange(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
        return (rewards * discounts[:, None]).sum(dim=0)

    def update_policy(self, rollout: Rollout) -> Dict[str, float]:
        opts = self.options
        self.policy.train()
        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio_scale = 1.0 / self.policy.points_per_step if opts.per_point_ratio else 1.0

        n = len(rollout)
        stats: Dict[str, List[float]] = {
            k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        }
        stopped_early = False
        for _ in range(opts.ppo_epochs):
            epoch_kl = []
            for idx in torch.randperm(n, device=self.device).split(opts.minibatch_size):
                log_probs, entropy, values = self.policy.evaluate(rollout.obs[idx], rollout.actions[idx])
                log_ratio = (log_probs - rollout.log_probs[idx]) * ratio_scale
                ratio = log_ratio.exp()
                adv = advantages[idx]
                policy_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - opts.clip_range, 1 + opts.clip_range)
                ).mean()
                value_loss = F.mse_loss(values, rollout.returns[idx])
                loss = policy_loss + opts.vf_coef * value_loss - opts.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), opts.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    epoch_kl.append(approx_kl)
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append(approx_kl)
                    stats["clip_fraction"].append(((ratio - 1).abs() > opts.clip_range).float().mean().item())
            if opts.target_kl is not None and np.mean(epoch_kl) > 1.5 * opts.target_kl:
                stopped_early = True
                break

        explained_var = 1 - (rollout.returns - rollout.values).var() / (rollout.returns.var() + 1e-8)
        result = {f"train/{k}": float(np.mean(v)) for k, v in stats.items()}
        result["train/explained_variance"] = explained_var.item()
        result["train/stopped_early"] = float(stopped_early)
        return result

    def _ppo_phase(self):
        opts = self.options
        lr = opts.learning_rate * opts.round_lr_scale(self.round_idx)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        target_steps = opts.round_steps(self.round_idx)
        self.logger.info(
            "Round %d PPO phase: %d/%d steps, lr %.2e.", self.round_idx, self.round_steps_done, target_steps, lr
        )

        while self.round_steps_done < target_steps:
            if opts.val_update_freq and self.update_idx % opts.val_update_freq == 0:
                self._validate()
            if self.update_idx % opts.checkpoint_freq == 0:
                self.save("latest.pt")

            rollout, rollout_stats = self.collect_rollout()
            update_stats = self.update_policy(rollout)
            wandb.log(
                {**rollout_stats, **update_stats, "update": self.update_idx, "round": self.round_idx},
                step=self.steps,
            )
            self.update_idx += 1
            del rollout

    def _recon_phase(self):
        """Fits a copy of the reconstructor on masks from the current policy, then swaps it into the envs."""
        opts = self.options
        online = copy.deepcopy(self.env.reconstructor)
        for p in online.parameters():
            p.requires_grad_(True)
        online.train()
        lr = opts.recon_learning_rate * opts.round_lr_scale(self.round_idx)
        optimizer = torch.optim.Adam(online.parameters(), lr=lr)
        self.logger.info("Round %d reconstructor phase: %d steps, lr %.2e.", self.round_idx, opts.recon_steps_per_round, lr)

        score_before = self.evaluate(opts.num_val_episodes)
        losses = []
        self.policy.eval()
        for step in range(opts.recon_steps_per_round):
            obs, _ = self.env.reset()
            done = False
            with torch.no_grad():
                while not done:
                    actions, _, _ = self.policy.act(obs, deterministic=opts.recon_policy_deterministic)
                    obs, _, dones, _ = self.env.step(actions)
                    done = all(dones)
            losses.append(self._recon_train_step(online, optimizer))
            if (step + 1) % 25 == 0:
                wandb.log({"recon/loss": float(np.mean(losses[-25:])), "round": self.round_idx}, step=self.steps)

        online.eval()
        for p in online.parameters():
            p.requires_grad_(False)
        for env in self._envs():
            env.load_reconstructor_state(online.state_dict(), env.sensetivity_net.state_dict())
        del online, optimizer

        # The reward function just changed: stale scale/returns would mislead the critic's targets.
        self.return_std = RunningStd()
        score_after = self._validate()
        wandb.log(
            {"val/score_before_recon_phase": score_before, "val/score_after_recon_phase": score_after,
             "round": self.round_idx},
            step=self.steps,
        )
        self.logger.info("Round %d recon phase: val score %.6f -> %.6f.", self.round_idx, score_before, score_after)

    def _recon_train_step(self, online: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """One supervised step on the episode the env just finished (its final mask, ground truth)."""
        state = self.env.state
        mask = state["mask"]
        masked_kspace = torch.view_as_complex((mask.unsqueeze(-1) * state["full_kspace"]).contiguous())
        recon = online(masked_kspace, state["sense_maps"], mask)
        target = state["target"]
        target = target.squeeze(1) if target.dim() == 4 else target
        loss = (recon.squeeze(1).abs() - target).abs().mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), self.options.recon_max_grad_norm)
        optimizer.step()
        return loss.item()

    def __call__(self) -> float:
        opts = self.options
        self.logger.info("Starting PPO training at round %d/%d.", self.round_idx, opts.num_rounds)

        while self.round_idx < opts.num_rounds:
            self._ppo_phase()
            if opts.recon_steps_per_round > 0:
                self._recon_phase()
            self.round_idx += 1
            self.round_steps_done = 0
            self.save("latest.pt")

        wandb.finish()
        return self.best_val_score
