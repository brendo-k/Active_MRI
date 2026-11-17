"""fscr1's scorer additions (scripts/eval/eval_multistep_adaptivity.py): the (A-B)-(C-D) contrast and the zero-data probe.

Run with ``.venv/bin/python -m pytest tests/test_fscr1_eval.py`` from the repo root.
"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'eval')]

from tests.test_seqmri_sampler import NUM_STEP, seqmri_model, step_context, thresholds  # noqa: E402
from tests.test_multistep_sampling import LABELS, data  # noqa: E402

scorer = importlib.import_module('eval_multistep_adaptivity')


def test_contrast_terms_read_a_pair_and_a_difference_of_two_pairs():
    assert scorer.contrast_terms('G=A_4_s42-T_4_s42') == ('G', [(1, 'A_4_s42'), (-1, 'T_4_s42')])
    assert scorer.contrast_terms('D=(F_4_tF-L_1_tF)-(F_4_tS5-L_1_tS5)') == (
        'D', [(1, 'F_4_tF'), (-1, 'L_1_tF'), (-1, 'F_4_tS5'), (1, 'L_1_tS5')])
    for broken in ('G=A', 'G=A-B-C', 'D=(A-B)-C', 'D=(A-B)-(C)', '=A-B'):
        with pytest.raises(SystemExit):
            scorer.contrast_terms(broken)


def test_zeroed_data_channels_keep_the_mask_and_restore_the_sampler():
    arm = seqmri_model(seqmri_input_scale='signed_log').sampler.mask_net
    old_mask, context = step_context()
    with torch.no_grad():
        real = arm.inputs(old_mask, context)
        with scorer.zeroed_data_channels(arm):
            zeroed = arm.inputs(old_mask, context)
        restored = arm.inputs(old_mask, context)
    assert zeroed[:, :4].eq(0).all() and torch.equal(zeroed[:, 4:], old_mask)
    assert real[:, :4].ne(0).any() and torch.equal(restored, real)


def test_the_zero_data_probe_moves_draws_of_an_arm_that_reads_its_data():
    arm = seqmri_model(seqmri_input_scale='signed_log')
    arm.eval()
    kspace, _, _ = data()
    with torch.no_grad():
        masks, _ = arm.sample_trajectory(kspace, LABELS, thresholds())
        calibration, _ = arm._initial_masks(kspace)
        previous = [calibration] + masks[:-1]
        moved = []
        for step in range(NUM_STEP):
            recon_mask = previous[arm.latest_recon_step(step)]
            with scorer.zeroed_data_channels(arm.sampler.mask_net):
                zeroed = arm.counterfactual_step(kspace, previous[step], step, kspace, thresholds()[step], LABELS,
                                                 recon_mask=recon_mask)
            moved.append(scorer.moved_fraction((masks[step] - previous[step]) > 0, zeroed['new']))
    assert torch.stack(moved).nan_to_num(0).gt(0).any()
