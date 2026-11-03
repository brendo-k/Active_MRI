import copy

import pytest
import torch
import torch.nn as nn
from fastmri.models import Unet

from activemri.reconstructors.norms import NORM_LAYERS, replace_instance_norms
from activemri.training.muon import MuonAdam, muon_parameters, orthogonalize


def test_orthogonalize_gives_near_unit_singular_values():
    torch.manual_seed(0)
    for rows, cols in ((18, 162), (144, 72 * 9), (36, 18)):
        singular = torch.linalg.svdvals(orthogonalize(torch.randn(3, rows, cols)).float())
        assert singular.min() > 0.5 and singular.max() < 1.3


def test_muon_parameters_are_the_hidden_convolutions():
    unet = Unet(2, 2, 18, 4)
    chosen = {id(p) for p in muon_parameters(unet)}
    assert id(unet.down_sample_layers[0].layers[0].weight) not in chosen  # 2 -> 18 input conv
    assert id(unet.up_conv[-1][-1].weight) not in chosen  # 18 -> 2 output conv
    assert id(unet.conv.layers[0].weight) in chosen
    assert id(unet.up_transpose_conv[0].layers[0].weight) in chosen
    assert len(chosen) == 21


def test_muon_adam_updates_adam_groups_exactly_like_adam():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(4, 8, 3), nn.Conv2d(8, 8, 3), nn.Conv2d(8, 4, 1))
    reference = copy.deepcopy(model)
    muon = [model[1].weight]
    rest = [p for p in model.parameters() if p is not model[1].weight]
    optimizer = MuonAdam([{'params': rest}, {'params': muon, 'use_muon': True}], lr=1e-3)
    adam = torch.optim.Adam(reference.parameters(), lr=1e-3)
    x = torch.randn(2, 4, 12, 12)
    for net, opt in ((model, optimizer), (reference, adam)):
        opt.zero_grad()
        net(x).square().mean().backward()
        opt.step()
    for name in ('0.weight', '0.bias', '2.weight', '2.bias'):
        assert torch.equal(dict(model.named_parameters())[name], dict(reference.named_parameters())[name])
    assert not torch.allclose(model[1].weight, reference[1].weight)


def test_muon_step_is_rms_matched_to_adam():
    torch.manual_seed(0)
    weight = nn.Parameter(torch.zeros(144, 72, 3, 3))
    weight.grad = torch.randn_like(weight)
    MuonAdam([{'params': [weight], 'use_muon': True, 'nesterov': False}], lr=1.0).step()
    assert weight.detach().square().mean().sqrt().item() == pytest.approx(0.2, rel=0.2)


@pytest.mark.parametrize('kind', NORM_LAYERS)
def test_norm_swap_keeps_other_weights_and_runs(kind):
    torch.manual_seed(0)
    reference = Unet(2, 2, 18, 4)
    torch.manual_seed(0)
    swapped = replace_instance_norms(Unet(2, 2, 18, 4), kind)
    conv_weights = {n: p for n, p in reference.named_parameters()}
    for name, p in swapped.named_parameters():
        if name in conv_weights:
            assert torch.equal(p, conv_weights[name])
    has_instance = any(isinstance(m, nn.InstanceNorm2d) for m in swapped.modules())
    assert has_instance == (kind == 'instance')
    assert swapped(torch.randn(2, 2, 32, 32)).shape == (2, 2, 32, 32)
