import pytest
import torch

from activemri.samplers.loupe import LineConstrainedProbMask, ProbMask
from activemri.samplers.sampler import Sampler
from activemri.utils.mask_binarization import (
    SIGMOID_PEAK_DERIVATIVE,
    STE_SLOPE,
    RescaleProbMap,
    ThresholdRandomMaskSigmoidV1,
    floored_grad_sigmoid,
    floored_sigmoid_derivative,
)

SLOPE = 5.0
LOGITS = torch.linspace(-4, 4, 101, dtype=torch.float64)


def _grad(fn, x):
    x = x.clone().requires_grad_(True)
    fn(x).sum().backward()
    return x.grad


def test_zero_floor_matches_torch_sigmoid():
    expected = _grad(lambda x: torch.sigmoid(SLOPE * x), LOGITS)
    assert torch.equal(_grad(lambda x: floored_grad_sigmoid(x, SLOPE, 0.0), LOGITS), expected)


@pytest.mark.parametrize("grad_floor", [0.01, 0.05, 0.2])
def test_floor_bounds_derivative_and_keeps_forward(grad_floor):
    x = LOGITS.clone().requires_grad_(True)
    output = floored_grad_sigmoid(x, SLOPE, grad_floor)
    output.sum().backward()

    assert torch.equal(output.detach(), torch.sigmoid(SLOPE * LOGITS))
    peak = SLOPE * SIGMOID_PEAK_DERIVATIVE
    assert (x.grad >= grad_floor * peak - 1e-12).all()
    exact = _grad(lambda x: torch.sigmoid(SLOPE * x), LOGITS)
    unsaturated = exact > grad_floor * peak
    assert torch.allclose(x.grad[unsaturated], exact[unsaturated])


def _ste_grad(probabilities, grad_floor):
    torch.manual_seed(0)
    x = probabilities.clone().requires_grad_(True)
    ThresholdRandomMaskSigmoidV1.apply(x, grad_floor).sum().backward()
    return x.grad


def test_ste_zero_floor_matches_original_surrogate():
    margin = torch.linspace(-1, 1, 201, dtype=torch.float64)  # p - u
    original = STE_SLOPE * torch.exp(-STE_SLOPE * margin) / (torch.exp(-STE_SLOPE * margin) + 1).pow(2)
    floored = floored_sigmoid_derivative(torch.sigmoid(STE_SLOPE * margin), STE_SLOPE, 0.0)
    assert torch.allclose(floored, original)


def test_binarization_is_a_single_independent_draw():
    probabilities = torch.rand(3, 1, 32, 32)
    torch.manual_seed(1)
    mask = ThresholdRandomMaskSigmoidV1.apply(probabilities, 0.0)
    torch.manual_seed(1)
    assert torch.equal(mask, (probabilities > torch.rand_like(probabilities)).float())


def test_ste_floor_bounds_derivative():
    grad = _ste_grad(torch.full((1, 1, 64, 64), 0.05), grad_floor=0.2)
    assert (grad >= 0.2 * STE_SLOPE * SIGMOID_PEAK_DERIVATIVE - 1e-6).all()


@pytest.mark.parametrize("mask_cls,shape", [(ProbMask, (8, 8)), (LineConstrainedProbMask, (8,))])
def test_saturated_loupe_logits_still_get_gradient(mask_cls, shape):
    mask = mask_cls(shape, slope=SLOPE, grad_floor=0.05)
    with torch.no_grad():
        mask.mask.fill_(-10.0)  # sigmoid(-50): exact derivative ~1e-21
    mask(torch.zeros(1, 8, 8, 2)).sum().backward()
    assert (mask.mask.grad >= 0.05 * SLOPE * SIGMOID_PEAK_DERIVATIVE - 1e-6).all()


@pytest.mark.parametrize("mask_grad_floor,ste_grad_floor", [(0.05, 0.0), (0.0, 0.2), (0.05, 0.2)])
def test_sampler_routes_each_floor_to_its_own_sigmoid(mask_grad_floor, ste_grad_floor):
    sampler = Sampler(sampler_type="loupe_2d", shape=(16, 16), slope=SLOPE,
                      mask_grad_floor=mask_grad_floor, ste_grad_floor=ste_grad_floor)
    assert sampler.mask_net.grad_floor == mask_grad_floor
    assert sampler.binarize.keywords["grad_floor"] == ste_grad_floor


def test_mask_floor_keeps_saturated_logits_trainable_through_sampler():
    sampler = Sampler(sampler_type="loupe_2d", shape=(16, 16), slope=SLOPE, mask_grad_floor=0.05)
    with torch.no_grad():
        sampler.mask_net.mask[..., :8, :] = -10.0
    old_mask = torch.zeros(2, 1, 16, 16)
    mask, *_ = sampler(torch.randn(2, 16, 16, 2), old_mask, budget=64)
    (mask * torch.randn_like(mask)).sum().backward()
    assert (sampler.mask_net.mask.grad[..., :8, :] != 0).all()


def test_rescale_without_available_matches_original():
    x = torch.rand(3, 1, 16, 16, dtype=torch.float64) * 0.05  # mean < sparsity: inverted branch
    original = []
    for i in range(len(x)):
        xi = x[i:i + 1]
        r, beta = 0.25 / xi.mean(), 0.75 / (1 - xi.mean())
        le = (r <= 1).double()
        original.append(le * xi * r + (1 - le) * (1 - (1 - xi) * beta))
    assert torch.allclose(RescaleProbMap(x, 0.25), torch.cat(original))


@pytest.mark.parametrize("scale", [0.05, 0.9])  # inverted and direct branches
def test_rescale_keeps_unavailable_at_zero_and_preserves_count(scale):
    available = torch.ones(2, 1, 16, 16, dtype=torch.float64)
    available[:, :, :4, :4] = 0
    x = torch.rand(2, 1, 16, 16, dtype=torch.float64) * scale * available
    rescaled = RescaleProbMap(x, 0.25, available)
    assert (rescaled[available == 0] == 0).all()
    assert torch.allclose(rescaled.sum(dim=(1, 2, 3)), torch.full((2,), 0.25 * 256, dtype=torch.float64))
    assert (rescaled <= 1).all() and (rescaled >= 0).all()


def test_mask_logit_stats_flags_outward_drift():
    from activemri.training.trainer import mask_logit_stats
    mask = ProbMask((8, 8), slope=SLOPE)
    with torch.no_grad():
        mask.mask.fill_(-2.0)  # slope * w = -10: saturated
    _, previous = mask_logit_stats(mask, None)
    with torch.no_grad():
        mask.mask.sub_(0.5)
    log, _ = mask_logit_stats(mask, previous)
    assert log['mask/saturated_frac'] == 1.0
    assert log['mask/saturated_moving_outward_frac'] == 1.0
    assert log['mask/saturated_abs_mean'] == pytest.approx(12.5)
