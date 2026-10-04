"""Line-constrained DQN sampler, backboned by VarNet's cascade U-Net.

Unlike the differentiable samplers in `activemri.samplers.conv`/`loupe`, this
picks one discrete k-space column per environment step and is trained by
Q-learning (`activemri.training.dqn_trainer.DQNTrainer`), not by backprop
through `Sampler.forward`. Its algorithm (epsilon-greedy action selection,
Double-DQN target, replay buffer) is a near-verbatim port of
`active-mri-acquisition`'s `activemri.baselines.ddqn.DDQN`, but the value
network is new: `active-mri-acquisition`'s DDQN used a from-scratch CNN
(`cvpr19_evaluator.EvaluatorNetwork`) with no relation to any reconstructor.
Here the value network's body *is* one cascade of `activemri.reconstructors
.VarNet.NormUnet`'s `fastmri.models.Unet`, so a pretrained joint knee+brain
VarNet checkpoint (e.g. `checkpoints/mt_varnet_256_1s/best_model.pt`) can
initialize it directly via `load_pretrained_backbone`.
"""
import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fastmri.models import Unet
from torch import nn, optim

from activemri.samplers.replay_buffer import ReplayMemory

logger = logging.getLogger(__name__)


class VarNetQNetwork(nn.Module):
    """Predicts one Q-value per k-space column from the current observation.

    The observation is a single packed tensor `[B, in_chans, H, W]`: the
    already-sampled k-space (real+imag, full and zero-filled: 4 channels) plus
    the previous mask (1 channel), matching `Sampler`'s
    `cat([observed_kspace, observed_kspace * old_mask], dim=-1)` convention
    (`activemri/samplers/sampler.py:211`) so the same batches used elsewhere
    in this repo need no reshaping. The mask is always the last channel.

    Args:
        num_lines: Number of k-space columns, i.e. the image width -- also
            the number of actions.
        chans: U-Net base channel count. Must match the pretrained VarNet
            checkpoint's `chans` (18 for `mt_varnet_256_1s`) for
            `load_pretrained_backbone` to transfer any weights.
        num_pools: U-Net pooling depth. Must match the checkpoint's `pools`.
        in_chans: Channels in the packed observation (4 k-space + 1 mask by
            default).
    """

    def __init__(self, num_lines: int, chans: int = 18, num_pools: int = 4, in_chans: int = 5):
        super().__init__()
        self.num_lines = num_lines
        self.in_chans = in_chans
        self.unet = Unet(in_chans=in_chans, out_chans=1, chans=chans, num_pool_layers=num_pools, drop_prob=0.0)

    @staticmethod
    def _pad(x: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def _unpad(x: torch.Tensor, h_pad, w_pad, h_mult, w_mult) -> torch.Tensor:
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predicts action (Q-)values.

        Args:
            obs: Packed observation, `[B, in_chans, H, W]`, mask as the last channel.

        Returns:
            Q-values for all actions, `[B, num_lines]`. Already-sampled
            columns are forced to `-1e10` so they're never selected again.
        """
        mask = obs[:, -1:, :, :]
        x, pad_sizes = self._pad(obs)
        scores = self.unet(x)
        scores = self._unpad(scores, *pad_sizes)
        q_per_line = scores.mean(dim=2).squeeze(1)  # [B, W]: average score over height

        already_sampled = (mask.sum(dim=2) == mask.shape[2]).float().view(mask.shape[0], -1)
        return q_per_line - 1e10 * already_sampled


def load_pretrained_backbone(
    model: VarNetQNetwork, checkpoint_path: str, cascade_idx: int = -1
) -> Tuple[List[str], List[str]]:
    """Initializes `model.unet` from one cascade of a VarNet checkpoint.

    Only layers whose shape matches are copied -- because `VarNetQNetwork`
    uses a different `in_chans`/`out_chans` than the pretrained `NormUnet`
    (packed 5-channel observation + 1-channel Q-map vs. the reconstructor's
    2-channel complex image), the first and last conv layers are skipped and
    left at their random init; every interior encoder/decoder layer (same
    `chans`/`num_pools`) transfers.

    Args:
        model: Freshly constructed `VarNetQNetwork` to initialize in place.
        checkpoint_path: Path to a `Sequential_Trainer` checkpoint whose
            `options.recon` is a `VarNetReconConfig` (e.g.
            `checkpoints/mt_varnet_256_1s/best_model.pt`).
        cascade_idx: Which cascade's `NormUnet` to copy from; negative
            indexes from the end (default: the last/most-refined cascade).

    Returns:
        (loaded_keys, skipped_keys): the `model.unet` state_dict keys that
        were and weren't copied, for logging/verification.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    cascade_indices = {
        int(k.split(".")[2]) for k in state_dict if k.startswith("reconstructor.cascades.")
    }
    if not cascade_indices:
        raise ValueError(f"No 'reconstructor.cascades.*' keys found in {checkpoint_path}")
    num_cascades = max(cascade_indices) + 1
    idx = cascade_idx if cascade_idx >= 0 else num_cascades + cascade_idx
    if idx not in cascade_indices:
        raise ValueError(f"Cascade {idx} not found in {checkpoint_path} ({num_cascades} cascades)")

    prefix = f"reconstructor.cascades.{idx}.model.unet."
    source = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    dst_state = model.unet.state_dict()
    loaded, skipped = [], []
    for key, value in source.items():
        if key in dst_state and dst_state[key].shape == value.shape:
            dst_state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.unet.load_state_dict(dst_state)

    logger.info(
        "Loaded %d/%d tensors for VarNetQNetwork.unet from cascade %d of %s (skipped: %s)",
        len(loaded), len(source), idx, checkpoint_path, skipped,
    )
    return loaded, skipped


class DQNPolicy(nn.Module):
    """Double-DQN policy over a `VarNetQNetwork`.

    Port of `active-mri-acquisition`'s `activemri.baselines.ddqn.DDQN`
    (epsilon-greedy `get_action`, Double-DQN Bellman update in
    `update_parameters`), adapted to the packed-tensor observation used here
    instead of that repo's `_encode_obs_dict`/`_decode_obs_tensor`.
    """

    def __init__(
        self,
        num_lines: int,
        device: torch.device,
        memory: Optional[ReplayMemory] = None,
        chans: int = 18,
        num_pools: int = 4,
        gamma: float = 0.99,
        dqn_learning_rate: float = 1e-4,
        pretrained_checkpoint: Optional[str] = None,
        backbone_cascade_idx: int = -1,
    ):
        super().__init__()
        self.model = VarNetQNetwork(num_lines, chans=chans, num_pools=num_pools)
        if pretrained_checkpoint is not None:
            load_pretrained_backbone(self.model, pretrained_checkpoint, backbone_cascade_idx)
        self.num_lines = num_lines
        self.memory = memory
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=dqn_learning_rate)
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    def add_experience(self, observation, action, next_observation, reward, done):
        self.memory.push(observation, action, next_observation, reward, done)

    def get_action(self, obs: torch.Tensor, eps_threshold: float = 0.0) -> List[int]:
        """Epsilon-greedy action selection: a random unsampled column with
        probability `eps_threshold`, otherwise the column with the highest
        predicted Q-value."""
        obs = obs.to(self.device)
        mask = obs[:, -1:, :, :]
        already_sampled = (mask.sum(dim=2) == mask.shape[2]).view(mask.shape[0], -1)

        with torch.no_grad():
            self.model.eval()
            q_values = self(obs)

        actions = torch.argmax(q_values, dim=1)
        if eps_threshold > 0:
            for i in range(obs.shape[0]):
                if random.random() < eps_threshold:
                    choices = (~already_sampled[i]).nonzero(as_tuple=True)[0]
                    actions[i] = choices[torch.randint(len(choices), (1,)).item()]
        return actions.tolist()

    def update_parameters(self, target_net: "DQNPolicy") -> Optional[Dict[str, float]]:
        self.model.train()
        batch = self.memory.sample()
        if batch is None:
            return None

        observations = batch["observations"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        not_done_mask = batch["dones"].to(self.device).logical_not()

        q_values = self(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self(next_observations).max(1)[1]
            target_q_next = target_net(next_observations).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_values = rewards.clone()
            target_values[not_done_mask] = (
                rewards[not_done_mask] + self.gamma * target_q_next[not_done_mask]
            )

        loss = F.smooth_l1_loss(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "q_values_mean": q_values.detach().mean().item(),
            "q_values_std": q_values.detach().std().item(),
        }
