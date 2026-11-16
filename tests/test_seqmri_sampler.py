"""seqmri_2d (activemri.samplers.seqmri): SeqMRI's sampler and loop on multi-coil data, and the train slice stride.

Run with ``.venv/bin/python -m pytest tests/test_seqmri_sampler.py`` from the repo root.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.datasets.fastmri_data import _every_nth_slice  # noqa: E402
from activemri.samplers.seq_adapt import StepContext, acs_compression, calibration_peaks  # noqa: E402
from activemri.samplers.seqmri import SeqMRISampler2D  # noqa: E402
from activemri.utils.multistep import StepRecorder, step_thresholds  # noqa: E402
from tests.test_multistep_sampling import LABELS, SIZE, data, model  # noqa: E402

NUM_STEP = 4


def seqmri_model(seed=0):
    return model(sampler_type='seqmri_2d', num_step=NUM_STEP, seed=seed, sampler_chans=4, sampler_depth=2)


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
    sampler = SeqMRISampler2D(chans=4, num_pool_layers=2)
    with torch.no_grad():
        scores = sampler(old_mask, context)
        inputs = sampler.inputs(old_mask, context)
    assert scores.min() >= 0 and scores.flatten(1).amax(dim=1).le(1 + 1e-6).all()
    assert scores[old_mask.bool()].eq(0).all()
    unmeasured = (old_mask == 0).expand(-1, 2, -1, -1)
    assert inputs[:, 2:4][unmeasured].eq(0).all() and inputs[:, 0:2][unmeasured].ne(0).any()
    assert torch.equal(inputs[:, 4:], old_mask)


def step_gradient_norms(detach_predictions):
    sampling_model = seqmri_model()
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


def test_every_nth_slice_keeps_the_middle_of_each_run_of_each_volume():
    dataset = SimpleNamespace(examples=[('a', s) for s in range(7)] + [('b', s) for s in range(10, 13)])
    _every_nth_slice(dataset, 3)
    assert dataset.examples == [('a', 1), ('a', 4), ('b', 11)]
