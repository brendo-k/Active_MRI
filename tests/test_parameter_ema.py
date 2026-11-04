import torch
from torch import nn

from activemri.training.ema import ParameterEMA


def test_update_follows_the_recurrence():
    layer = nn.Linear(2, 2)
    start = layer.weight.detach().clone()
    ema = ParameterEMA(layer.parameters(), decay=0.9)
    with torch.no_grad():
        layer.weight.add_(1.0)
    ema.update()
    expected = 0.9 * start + 0.1 * layer.weight.detach()
    assert torch.allclose(ema.shadow[0], expected)


def test_update_leaves_the_parameters_alone():
    layer = nn.Linear(2, 2)
    ema = ParameterEMA(layer.parameters(), decay=0.9)
    before = layer.weight.detach().clone()
    ema.update()
    assert torch.equal(layer.weight, before)


def test_applied_swaps_the_average_in_and_restores_the_parameters():
    layer = nn.Linear(2, 2)
    ema = ParameterEMA(layer.parameters(), decay=0.9)
    with torch.no_grad():
        layer.weight.add_(1.0)
    ema.update()
    online = layer.weight.detach().clone()
    with ema.applied():
        assert torch.equal(layer.weight, ema.shadow[0])
    assert torch.equal(layer.weight, online)
