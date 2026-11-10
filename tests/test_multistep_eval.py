"""The K-step eval's building blocks (activemri.utils.multistep, SequentialSampling's diagnostic entry points).

Run with ``.venv/bin/python -m pytest tests/test_multistep_eval.py`` from the repo root.

The eval's nulls are only as good as these: shared draws, a reconstruction identical to the forward's, a twin that draws
the same trajectory on any header-matched slice, and a data swap that reads the donor's data and nothing else. Each
null has a positive control, so a broken function cannot pass by reading 0.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.utils.metrics import NMSE, MaskedMetricAccumulator  # noqa: E402
from activemri.utils.multistep import (StepRecorder, cluster_bootstrap, header_signature, masked_slice_metrics,  # noqa: E402
                                       match_donors, step_grid_figure, step_panels, step_thresholds)
from tests.test_multistep_sampling import LABELS, SIZE, data, model, randomise_residual_output  # noqa: E402


def thresholds_for(num_step, slice_ids=(0, 1), seed=11):
    return step_thresholds(seed, slice_ids, num_step, (SIZE, SIZE))


def header_matched_donor(kspace, seed=9):
    """New values everywhere the slices hold data, zeros kept, so the acquired region and coil count match."""
    fresh = torch.randn(kspace.shape, generator=torch.Generator().manual_seed(seed))
    return torch.where(kspace != 0, fresh, torch.zeros_like(kspace))


def test_thresholds_are_keyed_by_seed_slice_and_step():
    together = step_thresholds(3, [5, 9], 2, (8, 8))
    alone = step_thresholds(3, [9], 2, (8, 8))
    assert torch.equal(together[1][1], alone[1][0])
    assert not torch.equal(together[0][0], together[1][0])
    assert not torch.equal(together[0][0], step_thresholds(4, [5], 1, (8, 8))[0][0])


@pytest.mark.parametrize('num_step', [1, 4])
def test_reconstruct_from_mask_is_the_forward_output_and_trajectory_its_masks(num_step):
    sampling_model = model(num_step=num_step)
    randomise_residual_output(sampling_model)
    sampling_model.eval()
    kspace, maps, target = data()
    thresholds = thresholds_for(num_step)
    with torch.no_grad():
        forward = sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds)
        masks, details = sampling_model.sample_trajectory(kspace, LABELS, thresholds, sense_maps=maps)
        rebuilt = sampling_model.reconstruct_from_mask(kspace, forward['mask'], maps, LABELS)
    assert all(torch.equal(a, b) for a, b in zip(masks, forward['masks']))
    assert len(details) == num_step and all('residual' in d for d in details)
    assert torch.equal(rebuilt, forward['output'])


def test_shared_thresholds_repeat_the_draws_and_differ_from_fresh_noise():
    sampling_model = model(num_step=4)
    sampling_model.eval()
    kspace, maps, _ = data()
    with torch.no_grad():
        first, _ = sampling_model.sample_trajectory(kspace, LABELS, thresholds_for(4), sense_maps=maps)
        torch.manual_seed(123)
        second, _ = sampling_model.sample_trajectory(kspace, LABELS, thresholds_for(4), sense_maps=maps)
        fresh, _ = sampling_model.sample_trajectory(kspace, LABELS, None, sense_maps=maps)
    assert torch.equal(first[-1], second[-1])
    assert not torch.equal(first[-1], fresh[-1])


@pytest.mark.parametrize('data_blind', [True, False])
def test_header_matched_donor_trajectory(data_blind):
    """The twin draws its own final mask on a header-matched donor (the swap null); the adaptive arm does not."""
    sampling_model = model(num_step=4, data_blind=data_blind)
    randomise_residual_output(sampling_model)
    sampling_model.eval()
    kspace, maps, _ = data()
    donor = header_matched_donor(kspace)
    assert header_signature(kspace) == header_signature(donor)
    with torch.no_grad():
        own, _ = sampling_model.sample_trajectory(kspace, LABELS, thresholds_for(4), sense_maps=maps)
        swapped, _ = sampling_model.sample_trajectory(donor, LABELS, thresholds_for(4), sense_maps=maps)
    assert torch.equal(own[-1], swapped[-1]) == data_blind


@pytest.mark.parametrize('data_blind', [True, False])
def test_counterfactual_step_reads_the_donors_data_and_nothing_else(data_blind):
    sampling_model = model(num_step=4, data_blind=data_blind)
    randomise_residual_output(sampling_model)
    sampling_model.eval()
    kspace, maps, _ = data()
    thresholds = thresholds_for(4)
    donor = header_matched_donor(kspace)
    with torch.no_grad():
        masks, details = sampling_model.sample_trajectory(kspace, LABELS, thresholds, sense_maps=maps)
        calibration, _ = sampling_model._initial_masks(kspace)
        previous = [calibration] + masks[:-1]
        for step in range(4):
            own_new = masks[step] - previous[step]
            itself = sampling_model.counterfactual_step(kspace, previous[step], step, kspace, thresholds[step], LABELS)
            assert torch.equal(itself['new'], own_new)
            assert torch.equal(itself['residual'], details[step]['residual'])
            swapped = sampling_model.counterfactual_step(kspace, previous[step], step, donor, thresholds[step], LABELS)
            assert torch.equal(swapped['new'], own_new) == data_blind


def test_counterfactual_step_reads_the_donors_data_through_the_recon_input_too():
    """For A_K^rec the swap function's own-data call reproduces each step, and a donor's data moves samples."""
    sampling_model = model(num_step=4, sampler_input='recon')
    randomise_residual_output(sampling_model)
    sampling_model.eval()
    kspace, maps, _ = data()
    thresholds = thresholds_for(4)
    donor = header_matched_donor(kspace)
    with torch.no_grad():
        masks, details = sampling_model.sample_trajectory(kspace, LABELS, thresholds, sense_maps=maps)
        calibration, _ = sampling_model._initial_masks(kspace)
        previous = [calibration] + masks[:-1]
        moved = []
        for step in range(4):
            itself = sampling_model.counterfactual_step(kspace, previous[step], step, kspace, thresholds[step], LABELS)
            assert torch.equal(itself['new'], masks[step] - previous[step])
            assert torch.equal(itself['residual'], details[step]['residual'])
            swapped = sampling_model.counterfactual_step(kspace, previous[step], step, donor, thresholds[step], LABELS)
            moved.append(not torch.equal(swapped['new'], itself['new']))
    assert any(moved)


def test_cluster_bootstrap_weights_slices_and_resamples_volumes_within_conditions():
    values = np.array([1.0, 1.0, 1.0, 3.0, np.nan, 10.0, 10.0])
    volumes = np.array(['a', 'a', 'a', 'b', 'b', 'c', 'd'])
    strata = np.array(['x', 'x', 'x', 'x', 'x', 'y', 'y'])
    result = cluster_bootstrap(values, volumes, strata, draws=500, seed=1)
    assert result['mean'] == pytest.approx(26.0 / 6)
    assert result['n_left_out'] == 1 and result['n_slices'] == 6 and result['n_volumes'] == 4
    low, high = result['ci95']
    assert low <= result['mean'] <= high
    assert cluster_bootstrap(values, volumes, strata, draws=500, seed=1) == result
    constant = cluster_bootstrap(np.full(6, 2.5), volumes[:6], strata[:6], draws=200)
    assert constant['ci95'] == [2.5, 2.5]


def test_masked_slice_metrics_are_the_trainers():
    generator = torch.Generator().manual_seed(4)
    target = torch.rand(3, 1, 32, 32, generator=generator)
    output = target + 0.05 * torch.randn(3, 1, 32, 32, generator=generator)
    mask = (torch.rand(3, 1, 32, 32, generator=generator) > 0.3).float()
    ours = masked_slice_metrics(output, target, mask)
    accumulator = MaskedMetricAccumulator()
    accumulator.update(output, target, mask)
    torch.testing.assert_close(ours['psnr'], torch.tensor(accumulator.psnr_values, dtype=torch.float64))
    torch.testing.assert_close(ours['ssim'], torch.tensor(accumulator.ssim_values, dtype=torch.float64))
    nmse = NMSE()
    nmse.update(output * mask, target * mask)
    torch.testing.assert_close(ours['nmse'].mean(), nmse.compute().double(), rtol=1e-5, atol=1e-7)


def test_donors_share_condition_and_signature_and_come_from_another_volume():
    conditions = np.array(['knee', 'knee', 'knee', 'brain', 'brain'])
    volumes = np.array(['v1', 'v1', 'v2', 'v3', 'v4'])
    signatures = [('s', 15), ('s', 15), ('s', 15), ('t', 16), ('u', 16)]
    donors = match_donors(conditions, volumes, signatures)
    assert donors[0] == 2 and donors[1] == 2 and donors[2] in (0, 1)
    assert donors[3] == -1 and donors[4] == -1


def test_header_signature_sees_the_acquired_region_and_coils():
    kspace, _, _ = data()
    narrower, fewer_coils = kspace.clone(), kspace.clone()
    narrower[0, :, :, 20] = 0
    fewer_coils[0, 1] = 0
    base = header_signature(kspace)
    assert header_signature(narrower)[0] != base[0] and header_signature(narrower)[1] == base[1]
    assert header_signature(fewer_coils)[0][1] == base[0][1] - 1


def test_recorder_gets_every_steps_gradient_and_the_figure_one_row_per_step():
    sampling_model = model(num_step=4)
    randomise_residual_output(sampling_model)
    sampling_model.train()
    kspace, maps, target = data()
    recorder = StepRecorder(slices=[1])
    sampling_model.step_recorder = recorder
    torch.manual_seed(5)
    output = sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps)
    sampling_model.step_recorder = None
    (output['output'] - target).abs().mean().backward()
    assert len(recorder.steps) == 4
    assert all(r['grad_norm'] is not None and float(r['grad_norm'].sum()) > 0 for r in recorder.steps)
    assert all({'grad', 'rescaled', 'residual', 'new', 'old_mask'} <= set(r['maps'][1]) for r in recorder.steps)
    scalars = recorder.scalars('train')
    assert {f'train/steps/grad_norm_{i}' for i in range(4)} <= set(scalars)
    assert 0 < scalars['train/steps/last_grad_share'] < 1

    panels = step_panels(sampling_model, kspace, target, maps, output['masks'], 1, LABELS,
                         torch.ones(2, 1, SIZE, SIZE), recorder)
    figure = step_grid_figure(panels, 'test')
    assert len(panels['rows']) == 4 and all('psnr' in row and 'grad' in row for row in panels['rows'])
    assert figure.axes and len(figure.axes) == 4 * 9
