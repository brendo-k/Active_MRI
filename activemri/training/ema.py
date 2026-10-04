"""Parameter-wise EMA between two identically-shaped modules.

Used by `JointTrainer` to keep `MRIEnv`'s reconstructor a slowly-moving
average of the online reconstructor being fit by gradient descent, so the
DQN reward (an NMSE delta computed by the env's reconstructor) stays close
to stationary from one replay-buffer transition to the next even while the
reconstructor itself keeps learning.
"""
import torch
from torch import nn


class EMA:
    def __init__(self, decay: float):
        assert 0.0 <= decay < 1.0
        self.decay = decay

    @torch.no_grad()
    def update(self, ema_module: nn.Module, online_module: nn.Module):
        ema_params = dict(ema_module.named_parameters())
        for name, param in online_module.named_parameters():
            ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
        ema_buffers = dict(ema_module.named_buffers())
        for name, buf in online_module.named_buffers():
            ema_buffers[name].copy_(buf)

    @torch.no_grad()
    def copy(self, ema_module: nn.Module, online_module: nn.Module):
        """Hard sync, for initialization (decay=0 update)."""
        ema_module.load_state_dict(online_module.state_dict())
