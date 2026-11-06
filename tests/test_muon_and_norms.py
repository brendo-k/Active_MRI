import copy

import pytest
import torch
import torch.nn as nn
from fastmri.models import Unet

from activemri.config import TrainConfig
from activemri.reconstructors.norms import NORM_LAYERS, replace_instance_norms
from activemri.training.muon import MuonAdam, muon_parameters, orthogonalize
from activemri.training.trainer import build_optimizer


class SamplerAndUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sampler = nn.Linear(4, 4)
        self.unet = nn.Sequential(nn.Conv2d(2, 8, 3), nn.Conv2d(8, 8, 3), nn.Conv2d(8, 2, 3))


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


def test_a_step_with_no_gradient_on_the_muon_group_updates_only_adam():
    adam_weight = nn.Parameter(torch.ones(3))
    muon_weight = nn.Parameter(torch.ones(8, 8, 3, 3))
    optimizer = MuonAdam([{'params': [adam_weight]}, {'params': [muon_weight], 'use_muon': True}], lr=1e-3)
    adam_weight.grad = torch.ones(3)
    optimizer.step()
    assert not torch.equal(adam_weight, torch.ones(3))
    assert torch.equal(muon_weight, torch.ones(8, 8, 3, 3))


def test_frozen_unets_leave_no_muon_group():
    model = SamplerAndUnet()
    model.unet.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(TrainConfig(optimizer='muon'), trainable, model)
    assert not any(group.get('use_muon') for group in optimizer.param_groups)
    model.sampler.weight.grad = torch.ones_like(model.sampler.weight)
    optimizer.step()


def test_the_samplers_group_stays_last_under_muon():
    model = SamplerAndUnet()
    groups = [{'params': list(model.unet.parameters())}, {'params': list(model.sampler.parameters()), 'lr': 0.05}]
    optimizer = build_optimizer(TrainConfig(optimizer='muon', lr=1e-3), groups, model)
    assert optimizer.param_groups[0]['lr'] == 1e-3
    assert optimizer.param_groups[1].get('use_muon')
    assert optimizer.param_groups[-1]['lr'] == 0.05


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
