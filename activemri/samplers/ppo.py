"""Coupled actor-critic for PPO, backboned by VarNet's cascade U-Net.

Same observation as `activemri.samplers.dqn` -- a packed `[B, 5, H, W]` tensor
with the mask as the last channel -- but trained on-policy by `activemri
.training.ppo_trainer.PPOTrainer` instead of by Q-learning.

Policy and value function share one U-Net trunk (as in L2SR's shared
encoder): the policy reads the decoder's output map, the value head reads the
globally pooled bottleneck. The U-Net subclasses `fastmri.models.Unet`, so its
state-dict keys match one cascade of `activemri.reconstructors.VarNet
.NormUnet` and `activemri.samplers.dqn.load_pretrained_backbone` initializes it
directly.

Each step acquires `points_per_step` distinct actions (k-space columns or 2D
points), drawn without replacement from one softmax over the not-yet-acquired
actions, so the policy's log-prob of a step is that of a Plackett-Luce
ranking, computed in closed form (`sequence_log_prob`).
"""
from typing import Tuple

import torch
import torch.nn.functional as F
from fastmri.models import Unet
from torch import nn

from activemri.samplers.dqn import VarNetQNetwork, load_pretrained_backbone

MASKED_LOGIT = -1e10  # large finite value rather than -inf: keeps `p * log p` NaN-free


class _SharedUnet(Unet):
    """`fastmri.models.Unet` whose forward also returns the pooled bottleneck features.

    Returns `(output [B, out_chans, H, W], bottleneck [B, chans * 2**num_pools])`.
    """

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stack = []
        output = image
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2)
        output = self.conv(output)
        bottleneck = output.mean(dim=(2, 3))

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            skip = stack.pop()
            output = transpose_conv(output)
            output = torch.cat([output, skip], dim=1)
            output = conv(output)
        return output, bottleneck


class ActorCriticNetwork(nn.Module):
    """`obs -> (action logits [B, num_actions], value [B])`.

    Args:
        action_space: `"line"` (logits per column: decoder map averaged over
            height) or `"point"` (logits per 2D k-space point: the decoder map
            itself, flattened row-major).
        target_fraction: Fraction of k-space acquired at episode end
            (`MRIEnv.target_fraction`). The value head is given the fraction
            of that target already acquired, since with a finite horizon the
            value depends on how many acquisitions remain.
        chans, num_pools, in_chans: As in `VarNetQNetwork`. `chans` and
            `num_pools` must match the pretrained VarNet for weights to transfer.
        hidden: Width of the value head's hidden layer.

    Already-acquired actions get logit `MASKED_LOGIT`.
    """

    def __init__(
        self,
        action_space: str,
        target_fraction: float,
        chans: int = 18,
        num_pools: int = 4,
        in_chans: int = 5,
        hidden: int = 256,
    ):
        super().__init__()
        self.action_space = action_space
        self.target_fraction = target_fraction
        self.unet = _SharedUnet(in_chans, 1, chans=chans, num_pool_layers=num_pools, drop_prob=0.0)
        self.value_head = nn.Sequential(
            nn.Linear(chans * 2**num_pools + 1, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = obs[:, -1:]  # [B, 1, H, W]
        x, pad_sizes = VarNetQNetwork._pad(obs)
        score_map, bottleneck = self.unet(x)
        score_map = VarNetQNetwork._unpad(score_map, *pad_sizes)  # [B, 1, H, W]

        acquired_fraction = mask.mean(dim=(1, 2, 3)) / self.target_fraction
        value = self.value_head(torch.cat([bottleneck, acquired_fraction[:, None]], dim=1)).squeeze(1)

        if self.action_space == "point":
            logits = score_map.flatten(1)
            acquired = mask.flatten(1) > 0.5
        else:
            logits = score_map.mean(dim=2).squeeze(1)
            acquired = (mask.sum(dim=2) == mask.shape[2]).squeeze(1)
        return logits.masked_fill(acquired, MASKED_LOGIT), value


def sequence_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Log-prob of `actions` `[B, K]` drawn in order without replacement from `softmax(logits)`.

    Step `i` picks `a_i` with probability `p[a_i] / (1 - sum_{j<i} p[a_j])`, so
    the log-prob is `sum_i log p[a_i] - sum_i log(1 - sum_{j<i} p[a_j])`.
    The running sum is accumulated in float64 so the remaining mass stays
    accurate when the policy is peaked.
    """
    log_p = F.log_softmax(logits.float(), dim=1).gather(1, actions)  # [B, K]
    p = log_p.exp().double()
    remaining = (1.0 - (p.cumsum(dim=1) - p)).clamp(min=1e-12)
    return (log_p.double() - remaining.log()).sum(dim=1).float()


def sample_without_replacement(logits: torch.Tensor, k: int, deterministic: bool = False) -> torch.Tensor:
    """`[B, k]` distinct actions, in selection order.

    Taking the top-`k` of `logits + Gumbel noise` is exactly sequential
    sampling without replacement from `softmax(logits)`.
    """
    if deterministic:
        return logits.topk(k, dim=1).indices
    uniform = torch.rand_like(logits).clamp(min=1e-20)
    gumbel = -torch.log((-torch.log(uniform)).clamp(min=1e-20))
    return (logits + gumbel).topk(k, dim=1).indices


class PPOPolicy(nn.Module):
    """`ActorCriticNetwork` plus the action-selection/evaluation API PPO needs."""

    def __init__(
        self,
        action_space: str,
        points_per_step: int,
        target_fraction: float,
        device: torch.device,
        chans: int = 18,
        num_pools: int = 4,
        critic_hidden: int = 256,
        pretrained_checkpoint: str | None = None,
        backbone_cascade_idx: int = -1,
    ):
        super().__init__()
        self.points_per_step = points_per_step
        self.net = ActorCriticNetwork(
            action_space, target_fraction, chans=chans, num_pools=num_pools, hidden=critic_hidden
        )
        if pretrained_checkpoint is not None:
            load_pretrained_backbone(self.net, pretrained_checkpoint, backbone_cascade_idx)
        self.device = device
        self.to(device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Picks `points_per_step` actions per batch element.

        Returns:
            (actions `[B, K]`, sequence log-probs `[B]`, values `[B]`).
        """
        logits, value = self.net(obs.to(self.device))
        actions = sample_without_replacement(logits, self.points_per_step, deterministic)
        return actions, sequence_log_prob(logits, actions), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable (log-probs `[B]`, entropy `[B]`, values `[B]`) for `actions` `[B, K]`.

        Entropy is that of the per-step softmax over not-yet-acquired actions.
        """
        logits, value = self.net(obs)
        log_p = F.log_softmax(logits.float(), dim=1)
        entropy = -(log_p.exp() * log_p).sum(dim=1)
        return sequence_log_prob(logits, actions), entropy, value
