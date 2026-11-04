"""Exponential moving averages of weights.

`EMA` averages between two identically-shaped modules. `JointTrainer` uses it to keep `MRIEnv`'s reconstructor a
slowly-moving average of the online reconstructor being fit by gradient descent, so the DQN reward (an NMSE delta
computed by the env's reconstructor) stays close to stationary from one replay-buffer transition to the next even while
the reconstructor itself keeps learning.

`ParameterEMA` keeps the average in a shadow copy beside a list of parameters and never writes to them, so it leaves the
training trajectory as it was. `Sequential_Trainer` uses it to save an averaged checkpoint.
"""
from contextlib import contextmanager
from typing import Iterable

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


class ParameterEMA:
    """Moving average of `parameters`: shadow = decay * shadow + (1 - decay) * parameter, starting from their values."""

    def __init__(self, parameters: Iterable[nn.Parameter], decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.parameters = list(parameters)
        self.shadow = [parameter.detach().clone() for parameter in self.parameters]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.shadow, self.parameters, 1 - self.decay)

    @contextmanager
    @torch.no_grad()
    def applied(self):
        """The parameters hold the average inside the block and their own values again after it."""
        online = [parameter.detach().clone() for parameter in self.parameters]
        torch._foreach_copy_(self.parameters, self.shadow)
        try:
            yield
        finally:
            torch._foreach_copy_(self.parameters, online)
