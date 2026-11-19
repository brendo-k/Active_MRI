"""seqmri_lr_bias_r8: seqmri_2d's per-step logit maps (`sampler.seqmri_step_bias`), the optimizer groups of the arms,
and the learning rate a resumed run continues at.

Run with ``.venv/bin/python -m pytest tests/test_seqmri_step_bias.py`` from the repo root.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.config import TrainConfig  # noqa: E402
from activemri.samplers.sampler import Sampler  # noqa: E402
from activemri.samplers.seqmri import SeqMRISampler2D  # noqa: E402
from activemri.training.trainer import Sequential_Trainer, build_lr_scheduler, build_optimizer  # noqa: E402
from tests.test_multistep_sampling import LABELS, SIZE, data  # noqa: E402
from tests.test_seqmri_sampler import NUM_STEP, seqmri_model, step_context, thresholds  # noqa: E402


def sampler_pair(**kwargs):
    torch.manual_seed(3)
    plain = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2, input_scale='signed_log', **kwargs)
    torch.manual_seed(3)
    biased = SeqMRISampler2D((SIZE, SIZE), chans=4, num_pool_layers=2, input_scale='signed_log', step_bias_steps=NUM_STEP,
                             **kwargs)
    return plain, biased


@pytest.mark.parametrize('data_blind', [False, True])
def test_zero_maps_change_no_score_and_draw_nothing_from_the_random_stream(data_blind):
    plain, biased = sampler_pair(data_blind=data_blind)
    shared = {k: v for k, v in biased.state_dict().items() if k != 'step_bias'}
    assert shared.keys() == plain.state_dict().keys()
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in shared.items())
    assert biased.step_bias.shape == (NUM_STEP, 1, SIZE, SIZE) and biased.step_bias.eq(0).all()
    old_mask, context = step_context()
    with torch.no_grad():
        assert torch.equal(plain(old_mask, context), biased(old_mask, context))


@pytest.mark.parametrize('data_blind', [False, True])
def test_each_step_reads_its_own_map(data_blind):
    plain, biased = sampler_pair(data_blind=data_blind)
    old_mask, context = step_context()
    with torch.no_grad():
        biased.step_bias[context.step + 1].normal_()
        assert torch.equal(plain(old_mask, context), biased(old_mask, context))
        biased.step_bias[context.step].normal_()
        assert not torch.equal(plain(old_mask, context), biased(old_mask, context))


def test_a_model_with_zero_maps_draws_the_masks_of_one_without():
    plain, biased = seqmri_model(seqmri_input_scale='signed_log'), seqmri_model(seqmri_input_scale='signed_log',
                                                                                seqmri_step_bias=True)
    assert plain.sampler.mask_net.step_bias is None
    assert biased.sampler.mask_net.step_bias.shape == (NUM_STEP, 1, SIZE, SIZE)
    plain.eval()
    biased.eval()
    kspace, maps, target = data()
    with torch.no_grad():
        first = plain(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())
        second = biased(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds())
    assert all(torch.equal(a, b) for a, b in zip(first['masks'], second['masks']))
    assert torch.equal(first['output'], second['output'])


def test_the_maps_belong_to_seqmri_2d_only():
    with pytest.raises(ValueError, match='seqmri_step_bias'):
        Sampler(sampler_type='seq_adapt_2d', shape=(SIZE, SIZE), seqmri_step_bias_steps=NUM_STEP)


def trainer_groups(sampler_lr=None, step_bias=False, optimizer='adam'):
    trainer = Sequential_Trainer.__new__(Sequential_Trainer)
    trainer.model = seqmri_model(seqmri_input_scale='signed_log', seqmri_step_bias=step_bias)
    trainer.options = TrainConfig(sampler_type='seqmri_2d', sampler_lr=sampler_lr, lr=1e-3, optimizer=optimizer,
                                  seqmri_step_bias=step_bias)
    return trainer, trainer._param_groups()


def ids(parameters):
    return {id(p) for p in parameters}


def test_the_controls_groups_are_unchanged_by_the_new_code():
    trainer, groups = trainer_groups()
    assert [id(p) for p in groups] == [id(p) for p in trainer.model.parameters() if p.requires_grad]


def test_a_sampler_lr_puts_the_whole_sampler_in_its_own_last_group():
    trainer, groups = trainer_groups(sampler_lr=1e-4)
    sampler = list(trainer.model.sampler.mask_net.parameters())
    assert len(groups) == 2 and groups[-1]['lr'] == 1e-4 and ids(groups[-1]['params']) == ids(sampler)
    assert not ids(groups[0]['params']) & ids(sampler)


@pytest.mark.parametrize('sampler_lr', [None, 1e-4])
def test_the_maps_get_their_own_last_group_at_the_samplers_rate(sampler_lr):
    trainer, groups = trainer_groups(sampler_lr=sampler_lr, step_bias=True)
    mask_net = trainer.model.sampler.mask_net
    network = [p for name, p in mask_net.named_parameters() if name != 'step_bias']
    assert ids(groups[-1]['params']) == {id(mask_net.step_bias)}
    assert groups[-1]['lr'] == (sampler_lr if sampler_lr is not None else 1e-3)
    if sampler_lr is None:
        assert len(groups) == 2 and ids(network) <= ids(groups[0]['params'])
    else:
        assert len(groups) == 3 and ids(groups[1]['params']) == ids(network) and groups[1]['lr'] == sampler_lr
    covered = [id(p) for group in groups for p in group['params']]
    assert sorted(covered) == sorted(id(p) for p in trainer.model.parameters() if p.requires_grad)


def test_under_muon_the_maps_stay_last_and_out_of_muon():
    trainer, groups = trainer_groups(step_bias=True, optimizer='muon')
    optimizer = build_optimizer(trainer.options, groups, trainer.model)
    assert ids(optimizer.param_groups[-1]['params']) == {id(trainer.model.sampler.mask_net.step_bias)}
    muon = [group for group in optimizer.param_groups if group.get('use_muon')]
    assert all(not ids(group['params']) & ids(trainer.model.sampler.parameters()) for group in muon)


def test_the_step_schedule_scales_every_group_and_a_resume_continues_at_the_stepped_rate():
    options = TrainConfig(lr=1e-3, lr_scheduler='step', lr_step_size=90, lr_gamma=0.1, num_epochs=150)
    weights = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    first = torch.optim.Adam([{'params': [weights[0]]}, {'params': [weights[1]], 'lr': 1e-4}], options.lr)
    scheduler = build_lr_scheduler(options, first)
    scheduler.step(99)
    assert [group['lr'] for group in first.param_groups] == pytest.approx([1e-4, 1e-5])
    saved = first.state_dict()
    second = torch.optim.Adam([{'params': [weights[0]]}, {'params': [weights[1]], 'lr': 1e-4}], options.lr)
    resumed = build_lr_scheduler(options, second)
    second.load_state_dict(saved)
    for epoch in (100, 149):
        resumed.step(epoch)
        assert [group['lr'] for group in second.param_groups] == pytest.approx([1e-4, 1e-5])
