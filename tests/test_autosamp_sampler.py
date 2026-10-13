"""AutoSamp sampler: coordinates snapped to the Cartesian grid by L2 distance."""
import pytest
import torch

from activemri.samplers.autosamp import AutoSampCoordinates, annealed_kernel_sigma
from activemri.samplers.sampler import Sampler


SHAPE = (32, 32)


def _sampler(num_points, positions=None, **kwargs):
    sampler = AutoSampCoordinates(SHAPE, num_points, **kwargs)
    if positions is not None:
        with torch.no_grad():
            sampler.positions.copy_(torch.tensor(positions, dtype=torch.float32))
    return sampler


def _empty_mask(batch=1):
    return torch.zeros(batch, 1, *SHAPE)


def _sampled_cells(new_mask):
    return {tuple(cell) for cell in new_mask[0, 0].nonzero().tolist()}


@pytest.mark.parametrize("init", ["uniform", "normal"])
def test_initial_points_are_distinct_cells(init):
    sampler = AutoSampCoordinates(SHAPE, 200, init=init)

    new_mask, _ = sampler(_empty_mask())

    assert new_mask.sum().item() == 200


def test_point_snaps_to_nearest_cell():
    sampler = _sampler(3, [[5.4, 7.6], [10.0, 10.0], [20.49, 3.51]])

    new_mask, _ = sampler(_empty_mask())

    assert _sampled_cells(new_mask) == {(5, 8), (10, 10), (20, 4)}


def test_colliding_points_still_spend_the_whole_budget():
    sampler = _sampler(40, [[16.0, 16.0]] * 40)

    new_mask, _ = sampler(_empty_mask())

    assert new_mask.sum().item() == 40


def test_loser_of_a_collision_takes_the_nearest_free_cell():
    sampler = _sampler(2, [[8.0, 8.0], [8.2, 8.0]])  # both round to (8, 8); point 0 is nearer

    new_mask, _ = sampler(_empty_mask())

    assert _sampled_cells(new_mask) == {(8, 8), (9, 8)}


def test_nearest_claimant_keeps_the_contested_cell():
    sampler = _sampler(2, [[8.4, 8.0], [8.1, 8.0]])

    sampler(_empty_mask())

    assert torch.allclose(sampler.positions[1].detach(), torch.tensor([8.1, 8.0]))  # nearer, untouched
    assert torch.allclose(sampler.positions[0].detach(), torch.tensor([9.0, 8.0]))  # displaced to the nearest free cell


def test_points_avoid_already_acquired_cells():
    old_mask = _empty_mask()
    old_mask[:, :, 14:18, 14:18] = 1
    sampler = _sampler(30, [[16.0, 16.0]] * 30)

    new_mask, _ = sampler(old_mask)

    assert new_mask.sum().item() == 30
    assert (new_mask * old_mask).sum().item() == 0


def test_training_writes_displaced_points_back_but_leaves_the_rest_continuous():
    sampler = _sampler(3, [[5.3, 5.3], [12.0, 12.0], [12.1, 12.0]])

    sampler(_empty_mask())

    assert torch.allclose(sampler.positions[0].detach(), torch.tensor([5.3, 5.3]))
    assert sampler.positions.detach().round().unique(dim=0).shape[0] == 3


def test_eval_leaves_parameters_untouched():
    sampler = _sampler(40, [[16.0, 16.0]] * 40).eval()
    before = sampler.positions.detach().clone()

    new_mask, _ = sampler(_empty_mask())

    assert torch.equal(sampler.positions.detach(), before)
    assert new_mask.sum().item() == 40


def test_positions_wrap_around_the_edge():
    sampler = _sampler(2, [[-0.4, 31.6], [32.2, -1.2]])

    new_mask, _ = sampler(_empty_mask())

    assert _sampled_cells(new_mask) == {(0, 0), (0, 31)}
    assert sampler.positions.detach().min() >= -0.5
    assert sampler.positions.detach().max() < SHAPE[0] - 0.5


def test_gradient_follows_the_loss_slope_over_k_space():
    sampler = _sampler(1, [[10.3, 10.6]])
    ramp = torch.arange(SHAPE[1], dtype=torch.float32).expand(*SHAPE)

    new_mask, _ = sampler(_empty_mask())
    (new_mask[0, 0] * ramp).sum().backward()

    assert sampler.positions.grad[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert sampler.positions.grad[0, 1].item() == pytest.approx(1.0, abs=1e-5)


def test_gradient_ignores_acquired_cells():
    old_mask = _empty_mask()
    old_mask[:, :, 10, 11] = 1
    sampler = _sampler(1, [[10.0, 10.4]])

    new_mask, _ = sampler(old_mask)
    (new_mask[0, 0, 10, 11] * 1e6).backward()

    assert sampler.positions.grad.abs().sum().item() == 0.0


@pytest.mark.parametrize("kernel_sigma", [0.0, 0.3, 1.0, 3.0])
def test_training_mask_is_binary_at_any_kernel_width(kernel_sigma):
    # Noiseless k-space: any fractional weight on a cell would reveal its whole value.
    sampler = _sampler(3, [[5.4, 7.6], [10.0, 10.0], [20.3, 3.7]])
    sampler.kernel_sigma = kernel_sigma

    new_mask, _ = sampler(_empty_mask())

    assert ((new_mask == 0) | (new_mask == 1)).all()
    assert _sampled_cells(new_mask) == {(5, 8), (10, 10), (20, 4)}


def test_gaussian_surrogate_sees_cells_beyond_the_adjacent_ones():
    reward = torch.zeros(*SHAPE)
    reward[10, 12] = 1.0  # two cells away: outside the bilinear splat's reach
    bilinear = _sampler(1, [[10.0, 10.0]])
    gaussian = _sampler(1, [[10.0, 10.0]])
    gaussian.kernel_sigma = 1.0

    for sampler in (bilinear, gaussian):
        new_mask, _ = sampler(_empty_mask())
        (-(new_mask[0, 0] * reward).sum()).backward()

    assert bilinear.positions.grad.abs().sum().item() == 0.0
    assert gaussian.positions.grad[0, 1].item() < 0  # descent moves the point towards column 12
    assert gaussian.positions.grad[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_gaussian_surrogate_ignores_acquired_cells():
    old_mask = _empty_mask()
    old_mask[:, :, 10, 11] = 1
    sampler = _sampler(1, [[10.0, 10.0]])
    sampler.kernel_sigma = 1.0

    new_mask, _ = sampler(old_mask)
    (new_mask[0, 0, 10, 11] * 1e6).backward()

    assert sampler.positions.grad.abs().sum().item() == 0.0


def test_kernel_width_anneals_geometrically_then_holds():
    assert annealed_kernel_sigma(1.0, 0.1, 0.8, 0.0) == pytest.approx(1.0)
    assert annealed_kernel_sigma(1.0, 0.1, 0.8, 0.4) == pytest.approx(0.1 ** 0.5)
    assert annealed_kernel_sigma(1.0, 0.1, 0.8, 0.8) == pytest.approx(0.1)
    assert annealed_kernel_sigma(1.0, 0.1, 0.8, 1.0) == pytest.approx(0.1)


def test_samples_outside_the_valid_region_are_dropped():
    valid = torch.ones(1, 1, *SHAPE)
    valid[:, :, :, 16:] = 0
    sampler = _sampler(2, [[4.0, 4.0], [4.0, 20.0]])

    new_mask, _ = sampler(_empty_mask(), valid)

    assert _sampled_cells(new_mask) == {(4, 4)}


def test_cell_change_fraction_tracks_points_crossing_cells():
    sampler = _sampler(2, [[5.0, 5.0], [9.0, 9.0]])
    sampler(_empty_mask())
    assert sampler.cell_change_fraction == 0.0

    with torch.no_grad():
        sampler.positions[0] += 1.0
    sampler(_empty_mask())

    assert sampler.cell_change_fraction == 0.5


def test_too_many_points_for_the_free_cells_is_rejected():
    old_mask = _empty_mask()
    old_mask[:, :, :, :] = 1
    old_mask[0, 0, 0, 0] = 0

    with pytest.raises(ValueError, match="free"):
        AutoSampCoordinates(SHAPE, 2)(old_mask)


def test_sampler_adds_exactly_the_budget_on_top_of_the_preselected_region():
    sampler = Sampler(sampler_type="autosamp_2d", shape=SHAPE, autosamp_num_points=64)
    old_mask = _empty_mask(batch=3)
    old_mask[:, :, 14:18, 14:18] = 1

    mask, vis, anatomy_logits, logp, entropy = sampler(None, old_mask, budget=64)

    assert mask.shape == old_mask.shape
    assert mask.sum(dim=(1, 2, 3)).tolist() == [16 + 64] * 3
    assert {"prob_mask", "rescaled_mask", "binarized_mask"} <= vis.keys()
    assert (anatomy_logits, logp, entropy) == (None, None, None)


def test_sampler_trains_positions_through_the_mask():
    sampler = Sampler(sampler_type="autosamp_2d", shape=SHAPE, autosamp_num_points=64)
    kspace = torch.randn(2, 1, *SHAPE)

    mask, *_ = sampler(None, _empty_mask(batch=2), budget=64)
    (mask * kspace).pow(2).sum().backward()

    assert sampler.mask_net.positions.grad.abs().sum().item() > 0


def test_sampler_rejects_reinforce_mode():
    with pytest.raises(ValueError, match="loupe"):
        Sampler(sampler_type="autosamp_2d", shape=SHAPE, autosamp_num_points=8, mask_train_mode="reinforce")


class _ZeroFilledRecon(torch.nn.Module):
    """Coil-combined zero-filled magnitude: differentiable in the mask, no weights."""

    def forward(self, masked_kspace, sense_maps, mask):  # noqa: ARG002
        images = torch.fft.ifft2(torch.fft.ifftshift(masked_kspace, dim=(-2, -1)))
        return (images * sense_maps.conj()).sum(dim=1, keepdim=True).abs()


def _sequential_model(num_step=1):
    from activemri.models.sequential_sampling import SequentialSampling

    return SequentialSampling(
        _ZeroFilledRecon(), shape=(64, 64), sparsity=0.25, sampler_type="autosamp_2d",
        num_step=num_step, preselect_num=8, recon_type="cg_sense",
    )


def test_sequential_sampling_spends_the_sparsity_budget_and_trains_positions():
    model = _sequential_model()
    target, full_kspace = torch.rand(2, 1, 64, 64), torch.randn(2, 4, 64, 64, 2)

    pred = model(target, full_kspace)
    loss, _ = model.loss(pred, {"target": target, "kspace": full_kspace}, {"recon_weight": 1.0, "kspace_weight": 0.0}, "l1")
    loss.backward()

    assert pred["mask"].sum(dim=(1, 2, 3)).tolist() == [0.25 * 64 * 64] * 2
    positions = model.sampler.mask_net.positions
    assert torch.isfinite(positions.grad).all() and positions.grad.abs().sum() > 0


def test_sequential_sampling_rejects_multiple_steps():
    with pytest.raises(ValueError, match="num_step=1"):
        _sequential_model(num_step=2)


def test_trainer_gives_the_positions_their_own_learning_rate():
    from types import SimpleNamespace

    from activemri.training.trainer import Sequential_Trainer

    model = _sequential_model()
    options = SimpleNamespace(sampler_type="autosamp_2d", autosamp_lr=0.5, mask_train_mode="straight_through", reinforce_lr=None)

    groups = Sequential_Trainer._param_groups(SimpleNamespace(model=model, options=options))

    assert groups[-1]["lr"] == 0.5
    assert groups[-1]["weight_decay"] == 0.0
    assert groups[-1]["eps"] < 1e-12
    assert groups[-1]["params"] == [model.sampler.mask_net.positions]
    assert all(model.sampler.mask_net.positions is not p for p in groups[0]["params"])


def test_trainer_anneals_the_kernel_with_training_progress():
    from types import SimpleNamespace

    from activemri.training.trainer import Sequential_Trainer

    model = _sequential_model()
    options = SimpleNamespace(
        sampler_type="autosamp_2d", autosamp_kernel_sigma_start=1.0,
        autosamp_kernel_sigma_end=0.1, autosamp_anneal_fraction=0.5,
    )
    trainer = SimpleNamespace(model=model, options=options)

    mask_net = model.sampler.mask_net

    Sequential_Trainer._anneal_autosamp_kernel(trainer, 0)
    assert mask_net.kernel_sigma == pytest.approx(1.0)
    Sequential_Trainer._anneal_autosamp_kernel(trainer, 0.25)
    assert mask_net.kernel_sigma == pytest.approx(0.1 ** 0.5)
    assert mask_net.positions.requires_grad


def test_trainer_freezes_positions_on_the_hard_mask_after_the_anneal():
    from types import SimpleNamespace

    from activemri.training.trainer import Sequential_Trainer

    model = _sequential_model()
    options = SimpleNamespace(
        sampler_type="autosamp_2d", autosamp_kernel_sigma_start=1.0,
        autosamp_kernel_sigma_end=0.1, autosamp_anneal_fraction=0.5,
    )
    Sequential_Trainer._anneal_autosamp_kernel(SimpleNamespace(model=model, options=options), 0.9)
    target, full_kspace = torch.rand(2, 1, 64, 64), torch.randn(2, 4, 64, 64, 2)

    pred = model(target, full_kspace)
    loss, _ = model.loss(pred, {"target": target, "kspace": full_kspace}, {"recon_weight": 1.0, "kspace_weight": 0.0}, "l1")
    loss.backward()

    assert model.sampler.mask_net.kernel_sigma == 0.0
    assert model.sampler.mask_net.positions.grad is None
    assert pred["mask"].sum(dim=(1, 2, 3)).tolist() == [0.25 * 64 * 64] * 2
