"""Muon (MomentUm Orthogonalized by Newton-Schulz; Jordan et al. 2024) for the U-Nets' hidden convolutions, with Adam
for every other parameter.

A Muon parameter's update is its Nesterov momentum, viewed as a [out, in * kh * kw] matrix and replaced by the nearest
semi-orthogonal matrix (5 quintic Newton-Schulz iterations in bfloat16), scaled by 0.2 * sqrt(max(rows, cols)) so its
RMS matches a typical Adam update (Liu et al. 2025, "Muon is scalable for LLM training"). The same lr then suits both
kinds of parameter, and lr schedulers scale every group alike.

Muon is for the 3x3 and transposed convolutions inside the U-Nets: the input and output convolutions (2 channels
on one side), biases, norm affines, data-consistency weights and the sampler's mask logits stay on Adam, exactly as
torch.optim.Adam updates them.
"""
from collections import defaultdict
from typing import Iterable, List

import torch
import torch.nn as nn

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ADAM_RMS = 0.2
MIN_MUON_CHANNELS = 3  # convolutions with 2 or fewer input or output channels map the complex image in or out


def muon_parameters(module: nn.Module, exclude: Iterable[nn.Module] = ()) -> List[nn.Parameter]:
    """Weights of the trainable convolutions in `module` with at least MIN_MUON_CHANNELS input and output channels,
    leaving out those inside any module of `exclude`."""
    excluded = {id(p) for m in exclude for p in m.parameters()}
    weights = []
    for layer in module.modules():
        if not isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)) or not layer.weight.requires_grad:
            continue
        if min(layer.in_channels, layer.out_channels) < MIN_MUON_CHANNELS or id(layer.weight) in excluded:
            continue
        weights.append(layer.weight)
    return weights


def orthogonalize(matrices: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Nearest semi-orthogonal matrices (U V^T of each SVD) of a [batch, rows, cols] stack, by Newton-Schulz."""
    a, b, c = NS_COEFFICIENTS
    x = matrices.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


class MuonAdam(torch.optim.Adam):
    """torch.optim.Adam for groups without `use_muon`, Muon for groups with `use_muon=True`.

    Muon groups take `momentum` (default 0.95), `nesterov` (True) and `ns_steps` (5); their `lr` is scheduled like any
    other group's. Weight decay on a Muon group is decoupled (p *= 1 - lr * weight_decay).
    """

    def __init__(self, param_groups, lr: float, **adam_kwargs):
        for group in param_groups:
            if group.get('use_muon'):
                group.setdefault('momentum', 0.95)
                group.setdefault('nesterov', True)
                group.setdefault('ns_steps', 5)
        super().__init__(param_groups, lr, **adam_kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        all_groups = self.param_groups
        self.param_groups = [g for g in all_groups if not g.get('use_muon')]
        try:
            super().step()
        finally:
            self.param_groups = all_groups
        for group in all_groups:
            if group.get('use_muon'):
                self._muon_step(group)
        return loss

    def _muon_step(self, group):
        params = [p for p in group['params'] if p.grad is not None]
        grads = [p.grad for p in params]
        buffers = []
        for p in params:
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)
            buffers.append(state['momentum_buffer'])
        torch._foreach_mul_(buffers, group['momentum'])
        torch._foreach_add_(buffers, grads)
        updates = torch._foreach_add(grads, buffers, alpha=group['momentum']) if group['nesterov'] else buffers
        if group['weight_decay']:
            torch._foreach_mul_(params, 1 - group['lr'] * group['weight_decay'])
        by_shape = defaultdict(list)
        for p, update in zip(params, updates):
            by_shape[update.shape].append((p, update))
        for shape, items in by_shape.items():
            rows = shape[0]
            cols = shape.numel() // rows
            matrices = torch.stack([update.reshape(rows, cols) for _, update in items])
            directions = orthogonalize(matrices, group['ns_steps']).to(items[0][0].dtype)
            torch._foreach_add_([p for p, _ in items], [direction.view(shape) for direction in directions.unbind(0)],
                                alpha=-group['lr'] * ADAM_RMS * max(rows, cols) ** 0.5)
