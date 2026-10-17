import torch

from activemri.samplers.sampler import Sampler
from activemri.training.mnet_trainer import accept_label, distinct_rows, is_degenerate, random_new_locations
from activemri.utils.mask_binarization import topk_straight_through

SHAPE = (32, 32)
BUDGET = 40


def test_topk_selects_exact_count_among_available_only():
    probabilities = torch.rand(3, 1, *SHAPE)
    available = torch.ones(3, 1, *SHAPE)
    available[:, :, :4] = 0
    chosen = topk_straight_through(probabilities, BUDGET, available)
    assert (chosen.flatten(1).sum(dim=1) == BUDGET).all()
    assert (chosen * (1 - available)).sum() == 0
    top = probabilities.masked_fill(available == 0, -1).flatten(1).topk(BUDGET, dim=1).indices
    assert torch.equal(chosen.flatten(1).gather(1, top), torch.ones(3, BUDGET))


def test_topk_backward_is_identity():
    probabilities = torch.rand(2, 1, *SHAPE, requires_grad=True)
    weights = torch.randn(2, 1, *SHAPE)
    (topk_straight_through(probabilities, BUDGET) * weights).sum().backward()
    assert torch.equal(probabilities.grad, weights)


def test_mnet_sampler_adds_exactly_the_budget_and_is_deterministic():
    sampler = Sampler(sampler_type='mnet_2d', shape=SHAPE, sampler_chans=4, sampler_depth=2)
    kspace = torch.randn(2, *SHAPE, 2)
    old_mask = torch.zeros(2, 1, *SHAPE)
    old_mask[:, :, :2, :2] = 1
    first = sampler(kspace, old_mask, BUDGET)[0]
    second = sampler(kspace, old_mask, BUDGET)[0]
    assert torch.equal(first, second)
    assert ((first - old_mask).flatten(1).sum(dim=1) == BUDGET).all()
    assert (first >= old_mask).all()


def test_accept_requires_beating_both_the_sampler_mask_and_random():
    better, worse = torch.tensor([-0.1, -0.1]), torch.tensor([-0.2, -0.2])
    assert accept_label(better, worse, worse, 'q_check', True)
    assert not accept_label(better, better, worse, 'q_check', True)  # a tie is a reject
    assert not accept_label(better, worse, better, 'q_check', True)
    assert accept_label(better, worse, better + 1, 'q_check', False)  # random ignored
    assert accept_label(worse, better, better, 'always', True)


def test_degeneracy_is_more_than_half_identical():
    masks = torch.rand(6, 1, 8, 8) > 0.5
    assert distinct_rows(masks.float())[0] == 6
    assert not is_degenerate(masks.float())
    masks[:4] = masks[0]
    assert is_degenerate(masks.float())
    masks[3] = ~masks[0]
    assert not is_degenerate(masks.float())  # 3 of 6 is not more than half


def test_random_locations_have_exact_count_inside_available():
    available = torch.ones(4, 1, *SHAPE)
    available[:, :, :, :8] = 0
    chosen = random_new_locations(available, BUDGET)
    assert (chosen.flatten(1).sum(dim=1) == BUDGET).all()
    assert (chosen * (1 - available)).sum() == 0
