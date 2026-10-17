import torch

from activemri.samplers.sampler import Sampler
from activemri.samplers.smooth_adapt import SmoothAdaptiveMask

SHAPE = (32, 32)


def kspace(batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, *SHAPE, 4, generator=generator)


def test_starts_at_its_base_mask_for_every_input():
    net = SmoothAdaptiveMask(SHAPE, slope=5, calibration_size=4, grid=4)
    base = torch.linspace(-8, 8, SHAPE[0] * SHAPE[1]).view(SHAPE)
    net.set_base_logits(base)
    probability = net(kspace(3))
    expected = torch.sigmoid(base.clamp(-0.999 * 6, 0.999 * 6))
    assert torch.allclose(probability[:, 0], expected.expand(3, -1, -1), atol=1e-5)


def test_base_logits_stay_bounded():
    net = SmoothAdaptiveMask(SHAPE, slope=5, calibration_size=4, grid=0)
    with torch.no_grad():
        net.mask.fill_(100.0)
    assert net.base_logits().abs().max() <= 6.0


def test_offset_learns_from_the_input_only_through_the_predictor():
    net = SmoothAdaptiveMask(SHAPE, slope=5, calibration_size=4, grid=4)
    observation = kspace(2)
    (net(observation)[:, 0, :8, :8].sum()).backward()
    assert net.offset_head.weight.grad.abs().sum() > 0
    assert not hasattr(net.offset_head, 'bias') or net.offset_head.bias is None


def test_features_ignore_phase_and_global_scale():
    net = SmoothAdaptiveMask(SHAPE, slope=5, calibration_size=4, grid=4)
    observation = kspace(1)
    complex_k = torch.view_as_complex(observation[..., :2].contiguous())
    rotated = torch.view_as_real(complex_k * torch.exp(torch.tensor(1j * 0.7)) * 3.0)
    changed = torch.cat([rotated, observation[..., 2:]], dim=-1)
    assert torch.allclose(net.calibration_features(observation), net.calibration_features(changed), atol=1e-4)


def test_conditioned_base_selects_by_label():
    net = SmoothAdaptiveMask(SHAPE, slope=5, calibration_size=4, grid=0, num_conditioning_classes=2)
    with torch.no_grad():
        net.mask[1].fill_(0.5)
    probability = net(kspace(2), torch.tensor([0, 1]))
    assert probability[0].mean() < probability[1].mean()


def test_sampler_keeps_the_budget():
    sampler = Sampler(sampler_type='smooth_adapt_2d', shape=SHAPE, slope=5, adapt_grid=4, calibration_size=4)
    old_mask = torch.zeros(2, 1, *SHAPE)
    unscaled, _, _ = sampler.unscaled_probabilities(kspace(2)[..., :2], old_mask)
    rescaled, mask = sampler.select_locations(unscaled, old_mask, budget=128)
    assert torch.allclose(rescaled.flatten(1).sum(1), torch.full((2,), 128.0), rtol=1e-4)
    assert mask.shape == old_mask.shape
