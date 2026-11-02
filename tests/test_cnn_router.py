import torch

from activemri.samplers.cnn_router import ACSRouter, CNNRoutedMaskBank
from activemri.samplers.sampler import Sampler


def test_router_outputs_indices_and_responds_to_acs():
    router = ACSRouter(bank_size=2, hidden=1)
    with torch.no_grad():
        for layer in (router.network[0], router.network[2]):
            layer.weight.zero_()
            layer.weight[0, 0, 1, 1] = 1
            layer.bias.zero_()
        router.network[-1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        router.network[-1].bias.copy_(torch.tensor([0.1, -0.1]))
    acs = torch.zeros(2, 1, 8, 8)
    acs[1, :, :4] = 1
    acs[1, :, 4:] = -1
    assert torch.equal(router(acs), torch.tensor([0, 1]))


def test_eval_selects_one_mask_and_ignores_unobserved_data_and_labels():
    torch.manual_seed(5)
    bank = CNNRoutedMaskBank((32, 32), 5, 8, bank_size=3).eval()
    observation = torch.randn(2, 32, 32, 4)
    changed = torch.randn_like(observation) * 100
    changed[:, 12:20, 12:20, :2] = observation[:, 12:20, 12:20, :2]
    assert torch.equal(bank.indices(observation), bank.indices(changed))
    expected = torch.sigmoid(bank.slope * bank.mask)[bank.indices(observation)].unsqueeze(1)
    torch.testing.assert_close(bank(observation), expected)
    torch.testing.assert_close(bank(changed, torch.tensor([0, 1])), expected)


def test_training_routes_to_one_mask_and_backpropagates_to_cnn_and_bank():
    torch.manual_seed(7)
    bank = CNNRoutedMaskBank((16, 16), 5, 8, bank_size=3)
    observation = torch.randn(4, 16, 16, 2)
    output = bank(observation)
    candidates = torch.sigmoid(bank.slope * bank.mask)
    distance = (output[:, 0, None] - candidates[None]).abs().flatten(2).amax(2)
    assert (distance.min(dim=1).values < 1e-6).all()
    (output * torch.randn_like(output)).sum().backward()
    for parameter in (bank.mask, bank.router.network[0].weight, bank.router.network[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_sampler_preserves_acs_and_expected_budget():
    sampler = Sampler(sampler_type='cnn_router_2d', shape=(32, 32),
                      calibration_size=8, bank_size=2).eval()
    observation = torch.randn(2, 32, 32, 2)
    old_mask = torch.zeros(2, 1, 32, 32)
    old_mask[:, :, 12:20, 12:20] = 1
    observation = observation * old_mask.permute(0, 2, 3, 1)
    probabilities, _, _ = sampler.unscaled_probabilities(observation, old_mask)
    assert (probabilities[:, :, 12:20, 12:20] == 0).all()
    rescaled, mask = sampler.select_locations(probabilities, old_mask, budget=64)
    torch.testing.assert_close(rescaled.flatten(1).sum(1), torch.full((2,), 64.0))
    assert (mask[:, :, 12:20, 12:20] == 1).all()
