"""seqmri_2d (activemri.samplers.seqmri): SeqMRI's sampler and loop on multi-coil data, its fscr1 signed-log input and
data-blind twin, and the train slice stride.

Run with ``.venv/bin/python -m pytest tests/test_seqmri_sampler.py`` from the repo root.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.datasets.fastmri_data import _every_nth_slice  # noqa: E402
from activemri.samplers.seq_adapt import StepContext, acs_compression, calibration_peaks  # noqa: E402
from activemri.samplers.seqmri import SeqMRISampler2D, signed_log  # noqa: E402
from activemri.utils.multistep import StepRecorder, step_thresholds  # noqa: E402
from tests.test_multistep_sampling import LABELS, SIZE, data, model  # noqa: E402

NUM_STEP = 4


def seqmri_model(seed=0, **kwargs):
    return model(sampler_type='seqmri_2d', num_step=NUM_STEP, seed=seed, sampler_chans=4, sampler_depth=2, **kwargs)


def thresholds():
    return step_thresholds(11, (0, 1), NUM_STEP, (SIZE, SIZE))


def test_seqmri_2d_reconstructs_on_the_centre_and_after_every_step_and_its_trajectory_repeats_the_forward():
    sampling_model = seqmri_model()
    sampling_model.eval()
    kspace, maps, target = data()
    with torch.no_grad():
        sampling_model.reconstructor.calls = 0
        forward = sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())
        assert sampling_model.reconstructor.calls == NUM_STEP + 1
        masks, _ = sampling_model.sample_trajectory(kspace, LABELS, thresholds(), sense_maps=maps)
    assert all(torch.equal(a, b) for a, b in zip(masks, forward['masks']))
    assert all((after >= before).all() and (after > before).any() for before, after in zip(masks, masks[1:]))


def test_seqmri_2d_scores_are_softplus_over_their_maximum_zero_where_sampled_and_read_measured_values_only():
    torch.manual_seed(0)
    kspace, _, _ = data()
    coil = torch.view_as_complex(kspace.contiguous())
    old_mask = (torch.rand(2, 1, SIZE, SIZE) < 0.3).float()
    centre = torch.zeros_like(old_mask)
    centre[..., :2, :2] = centre[..., :2, -2:] = centre[..., -2:, :2] = centre[..., -2:, -2:] = 1
    old_mask = torch.clamp(old_mask + centre, max=1)
    context = StepContext(1, NUM_STEP, torch.ones_like(old_mask), torch.ones(2), measured=coil * old_mask,
                          calibration_peaks=calibration_peaks(coil * centre), predicted=coil,
                          compression=acs_compression(coil * centre, 1))
    sampler = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2)
    with torch.no_grad():
        scores = sampler(old_mask, context)
        inputs = sampler.inputs(old_mask, context)
    assert scores.min() >= 0 and scores.flatten(1).amax(dim=1).le(1 + 1e-6).all()
    assert scores[old_mask.bool()].eq(0).all()
    unmeasured = (old_mask == 0).expand(-1, 2, -1, -1)
    assert inputs[:, 2:4][unmeasured].eq(0).all() and inputs[:, 0:2][unmeasured].ne(0).any()
    assert torch.equal(inputs[:, 4:], old_mask)


def step_gradient_norms(detach_predictions, **kwargs):
    sampling_model = seqmri_model(**kwargs)
    if detach_predictions:
        predict = sampling_model._in_loop_prediction
        sampling_model._in_loop_prediction = lambda *args, **kwargs: predict(*args, **kwargs).detach()
    kspace, maps, target = data()
    recorder = StepRecorder()
    sampling_model.step_recorder = recorder
    forward = sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())
    forward['output'].abs().mean().backward()
    return [step['grad_norm'] for step in recorder.steps]


def test_seqmri_2d_credits_earlier_draws_only_through_the_reconstructions_later_steps_read():
    routed, cut = step_gradient_norms(detach_predictions=False), step_gradient_norms(detach_predictions=True)
    assert all(norm is not None and norm.gt(0).all() for norm in routed)
    assert all(norm is None or norm.eq(0).all() for norm in cut[:-1])
    assert cut[-1] is not None and cut[-1].gt(0).all()


def test_signed_log_is_odd_increasing_and_maps_zero_and_the_peak_to_themselves():
    values = torch.tensor([-1.0, -1e-3, -1e-5, 0.0, 1e-5, 1e-3, 1.0])
    scaled = signed_log(values)
    assert torch.equal(scaled[3], torch.tensor(0.0)) and torch.allclose(scaled[[0, -1]], torch.tensor([-1.0, 1.0]))
    assert torch.equal(signed_log(-values), -scaled) and (scaled.diff() > 0).all()
    assert 0.15 < scaled[4] < 0.2, "a 1e-5 value, typical beyond the centre, must reach the network"


def step_context(seed=0):
    torch.manual_seed(seed)
    kspace, _, _ = data(seed)
    coil = torch.view_as_complex(kspace.contiguous()) * 1e-3
    centre = torch.zeros(2, 1, SIZE, SIZE)
    centre[..., SIZE // 2 - 2:SIZE // 2 + 2, SIZE // 2 - 2:SIZE // 2 + 2] = 1
    coil[..., SIZE // 2, SIZE // 2] = 1
    old_mask = torch.clamp((torch.rand(2, 1, SIZE, SIZE) < 0.3).float() + centre, max=1)
    return old_mask, StepContext(1, NUM_STEP, torch.ones_like(old_mask), torch.ones(2), measured=coil * old_mask,
                                 calibration_peaks=calibration_peaks(coil * centre), predicted=coil,
                                 compression=acs_compression(coil * centre, 1))


def test_signed_log_input_is_the_raw_input_through_signed_log_with_the_mask_kept():
    old_mask, context = step_context()
    raw = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2)
    logged = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2, input_scale='signed_log')
    with torch.no_grad():
        raw_inputs, logged_inputs = raw.inputs(old_mask, context), logged.inputs(old_mask, context)
    assert torch.allclose(logged_inputs[:, :4], signed_log(raw_inputs[:, :4]))
    assert torch.equal(logged_inputs[:, 4:], old_mask)
    typical = raw_inputs[:, 2:4].abs().ge(1e-5) & raw_inputs[:, 2:4].abs().lt(1e-2)
    assert typical.any() and logged_inputs[:, 2:4][typical].abs().gt(0.15).all()


def test_the_twin_reads_a_fixed_tensor_and_the_mask_and_starts_from_the_arms_network():
    old_mask, context = step_context()
    _, other_context = step_context(seed=1)
    torch.manual_seed(3)
    arm = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2, input_scale='signed_log')
    torch.manual_seed(3)
    twin = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2, input_scale='signed_log', data_blind=True)
    shared = {k: v for k, v in twin.state_dict().items() if k != 'fixed_data'}
    assert shared.keys() == arm.state_dict().keys()
    assert all(torch.equal(v, arm.state_dict()[k]) for k, v in shared.items())
    assert twin.complex_coils == 0 and arm.complex_coils == 1
    with torch.no_grad():
        inputs = twin.inputs(old_mask, context)
        assert torch.equal(inputs, twin.inputs(old_mask, other_context))
        assert torch.equal(inputs, twin.inputs(old_mask, StepContext(1, NUM_STEP, context.extent, context.coil_fraction)))
    assert torch.equal(inputs[:, :4], twin.fixed_data.expand(2, -1, -1, -1)) and torch.equal(inputs[:, 4:], old_mask)
    blind_again = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2, data_blind=True)
    assert torch.equal(blind_again.fixed_data, twin.fixed_data)


def test_the_twin_runs_no_in_loop_reconstruction_and_its_draws_ignore_the_data():
    twin = seqmri_model(data_blind=True, seqmri_input_scale='signed_log')
    assert twin.recon_steps == () and twin.sampler.data_blind
    twin.eval()
    kspace, maps, target = data()
    with torch.no_grad():
        twin.reconstructor.calls = 0
        forward = twin(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())
        assert twin.reconstructor.calls == 1
        other, _, _ = data(seed=7)
        other = other * (kspace.abs().sum(dim=(1, 4), keepdim=True) > 0)
        swapped = twin(target, other, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())
    assert all(torch.equal(a, b) for a, b in zip(forward['masks'], swapped['masks']))


def test_the_signed_log_arm_reads_its_data_and_credits_earlier_draws_through_the_reconstructions():
    arm = seqmri_model(seqmri_input_scale='signed_log')
    arm.eval()
    kspace, maps, target = data()
    other, _, _ = data(seed=7)
    other = other * (kspace.abs().sum(dim=(1, 4), keepdim=True) > 0)
    with torch.no_grad():
        own = arm(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())['masks']
        swapped = arm(target, other, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())['masks']
    assert not torch.equal(own[-1], swapped[-1])
    routed = step_gradient_norms(detach_predictions=False, seqmri_input_scale='signed_log')
    assert all(norm is not None and norm.gt(0).all() for norm in routed)


def test_every_nth_slice_keeps_the_middle_of_each_run_of_each_volume():
    dataset = SimpleNamespace(examples=[('a', s) for s in range(7)] + [('b', s) for s in range(10, 13)])
    _every_nth_slice(dataset, 3)
    assert dataset.examples == [('a', 1), ('a', 4), ('b', 11)]
