import pytest
import torch

from activemri.samplers.sampler import Sampler
from activemri.training.trainer import Sequential_Trainer
from activemri.utils.mask_binarization import stratified_thresholds, threshold_random_mask


def test_thresholds_are_rotated_copies_laid_out_draw_major():
    like = torch.zeros(6, 1, 4, 4)
    thresholds = stratified_thresholds(like, 3)
    first, second, third = thresholds.chunk(3)
    assert torch.allclose(second, (first + 1 / 3) % 1)
    assert torch.allclose(third, (first + 2 / 3) % 1)
    assert thresholds.min() >= 0 and thresholds.max() < 1


def test_batch_must_hold_whole_draw_groups():
    with pytest.raises(ValueError):
        stratified_thresholds(torch.zeros(5, 1, 4, 4), 2)


def test_every_draw_is_a_bernoulli_sample_and_draws_are_negatively_correlated():
    torch.manual_seed(0)
    probability = torch.full((2 * 4000, 1, 4, 4), 0.2)
    masks = threshold_random_mask(probability, threshold=stratified_thresholds(probability, 2))
    first, second = masks.chunk(2)
    assert first.mean().item() == pytest.approx(0.2, abs=0.01)
    assert second.mean().item() == pytest.approx(0.2, abs=0.01)
    assert (first * second).mean().item() < 0.2 * 0.2 - 0.02  # p = 0.2 < 0.5: the antithetic pair never overlaps


def test_averaging_stratified_draws_has_lower_variance_than_independent_draws():
    torch.manual_seed(0)
    probability = torch.full((2 * 4000, 1, 4, 4), 0.3)
    stratified = threshold_random_mask(probability, threshold=stratified_thresholds(probability, 2)).chunk(2)
    independent = threshold_random_mask(probability).chunk(2)
    assert (sum(stratified) / 2).var() < 0.7 * (sum(independent) / 2).var()


def test_explicit_threshold_gives_straight_through_gradient():
    probability = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    mask = threshold_random_mask(probability, threshold=torch.tensor([[[[0.1, 0.9], [0.4, 0.6]]]]))
    assert mask.flatten().tolist() == [1.0, 0.0, 1.0, 0.0]
    mask.sum().backward()
    assert (probability.grad > 0).all()


def make_sampler(**kwargs):
    return Sampler(sampler_type="loupe_2d", shape=(16, 16), slope=5, **kwargs)


def test_training_copies_of_a_slice_share_probabilities_but_draw_different_masks():
    sampler = make_sampler(mask_draws=2).train()
    kspace = torch.randn(2, 16, 16, 2).repeat(2, 1, 1, 1)
    mask, *_ = sampler(kspace, torch.zeros(4, 1, 16, 16), budget=32)
    assert not torch.equal(mask[:2], mask[2:])


def test_eval_draws_once_per_slice_whatever_the_batch_size():
    sampler = make_sampler(mask_draws=2).eval()
    mask, *_ = sampler(torch.randn(3, 16, 16, 2), torch.zeros(3, 1, 16, 16), budget=32)
    assert mask.shape[0] == 3


def test_draws_need_a_thresholding_straight_through_sampler():
    with pytest.raises(ValueError):
        make_sampler(mask_draws=2, mask_train_mode="reinforce")
    with pytest.raises(ValueError):
        Sampler(sampler_type="caipi_2d", shape=(16, 16), mask_draws=2)


def test_trainer_tiles_batch_draw_major_after_loading():
    batch = {'kspace': torch.arange(2.0).view(2, 1), 'anatomy': ['brain', 'knee'], 'epoch': 3}
    tiled, kspace, target = Sequential_Trainer._tile_for_mask_draws(batch, torch.arange(2.0).view(2, 1), torch.arange(2.0), 2)
    assert tiled['anatomy'] == ['brain', 'knee', 'brain', 'knee']
    assert tiled['kspace'].flatten().tolist() == [0, 1, 0, 1]
    assert kspace.flatten().tolist() == [0, 1, 0, 1]
    assert target.tolist() == [0, 1, 0, 1]
    assert tiled['epoch'] == 3


def test_logit_limit_and_leak_set_the_probability_floor_and_the_gradient_beyond_it():
    from activemri.samplers.conv import scores_to_prob
    logits = torch.tensor([-20.0, 20.0], requires_grad=True)
    narrow = scores_to_prob(logits)
    wide = scores_to_prob(logits, logit_limit=10.0, logit_leak=0.05)
    assert narrow[0].item() > wide[0].item()
    assert wide[0].item() == pytest.approx(torch.sigmoid(torch.tensor(-10.0 - 0.05 * 10.0)).item(), rel=1e-4)
    wide[0].backward()
    assert logits.grad[0].item() > 0
