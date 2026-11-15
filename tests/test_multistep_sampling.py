"""Multi-step sampling (num_step = K > 1) and seq_adapt_2d, the experiment `multistep_adaptivity` runs on.

Run with ``.venv/bin/python -m pytest tests/test_multistep_sampling.py`` from the repo root.

What the K-step loop must do, each pinned by a test here:
- the final loss reaches every step's probabilities, not only the last step's;
- every step reads only the k-space measured so far, y * M_i, never a prediction or unacquired k-space;
- one reconstruction per forward, of the final mask;
- the data-blind twin reads its mask, the step and the header, but no k-space values, and starts bitwise identical
  to its adaptive arm;
- each step spends an equal share of the acquired budget.
K=1 is checked separately against the main checkout, bitwise (reports/multistep_adaptivity/).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.config import TrainConfig  # noqa: E402
from activemri.models.sequential_sampling import SequentialSampling  # noqa: E402
from activemri.samplers.seq_adapt import StepContext, acs_compression, calibration_peaks, virtual_coil_channels  # noqa: E402
from activemri.training.run_naming import run_name  # noqa: E402
from activemri.training.trainer import Sequential_Trainer, build_optimizer  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402

SIZE = 64
COILS = 4
PRESELECT = 4
SPARSITY = 0.25
LABELS = torch.tensor([0, 2])
SEQMRI = {'sampler_input': 'seqmri', 'recon_steps': (1, 3), 'virtual_coils': 2}  # the kscr2 S arm, scaled to K=4


class CountingReconstructor(nn.Module):
    """The zero-filled SENSE image times a learned scale, counting its calls."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, masked_kspace, sense_maps, mask, **kwargs):  # noqa: ARG002
        self.calls += 1
        return self.scale * (ifft_2d_img(masked_kspace) * sense_maps.conj()).sum(dim=1, keepdim=True)


def model(sampler_type='seq_adapt_2d', num_step=4, data_blind=False, seed=0, residual_chans=4, **kwargs):
    torch.manual_seed(seed)
    return SequentialSampling(CountingReconstructor(), shape=(SIZE, SIZE), sparsity=SPARSITY, sampler_type=sampler_type,
                              num_step=num_step, preselect_num=PRESELECT, recon_type='varnet', budget_over='acquired',
                              num_conditioning_classes=3, data_blind=data_blind, residual_chans=residual_chans,
                              residual_depth=2, **kwargs)


def data(seed=0):
    """(k-space [2, C, H, W, 2], unit-norm sense maps, target): slice 1 acquires a FLAIR-like 75% of the columns."""
    generator = torch.Generator().manual_seed(seed)
    kspace = torch.randn(2, COILS, SIZE, SIZE, 2, generator=generator)
    kspace[1, :, :, :8] = 0
    kspace[1, :, :, -8:] = 0
    maps = torch.randn(2, COILS, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    maps = maps / maps.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    return kspace, maps, torch.rand(2, 1, SIZE, SIZE, generator=generator)


def run(sampling_model, kspace, maps, target, seed=5):
    torch.manual_seed(seed)
    return sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps)


def record_steps(sampling_model):
    """Each step's (unscaled probabilities, rescaled probabilities, old mask), appended as the sampler runs."""
    steps = []
    select = sampling_model.sampler.select_locations

    def recording(unscaled, old_mask, budget, valid_mask=None, threshold=None):
        if unscaled.requires_grad:
            unscaled.retain_grad()
        rescaled, mask = select(unscaled, old_mask, budget, valid_mask, threshold=threshold)
        steps.append(SimpleNamespace(unscaled=unscaled, rescaled=rescaled, old_mask=old_mask))
        return rescaled, mask

    sampling_model.sampler.select_locations = recording
    return steps


def randomise_residual_output(sampling_model, seed=1):
    """Give the zero-initialised residual a nonzero output layer, so its dependence on its inputs can be seen."""
    generator = torch.Generator().manual_seed(seed)
    final = sampling_model.sampler.mask_net.residual.final_conv
    with torch.no_grad():
        final.weight.copy_(torch.randn(final.weight.shape, generator=generator))
        final.bias.copy_(torch.randn(final.bias.shape, generator=generator))


def rescale_values_keeping_zeros(kspace, where, seed=3):
    """`kspace` with each value under `where` multiplied by a random factor in [0.5, 2.5): zeros stay zero, so the
    acquired region, extent and coil count it implies are unchanged."""
    factor = 0.5 + 2 * torch.rand(kspace.shape, generator=torch.Generator().manual_seed(seed))
    return torch.where(where, kspace * factor, kspace)


@pytest.mark.parametrize('sampler_type', ['loupe_2d', 'seq_adapt_2d'])
def test_final_loss_reaches_every_steps_probabilities(sampler_type):
    sampling_model = model(sampler_type, num_step=4)
    sampling_model.train()
    steps = record_steps(sampling_model)
    kspace, maps, target = data()
    output = run(sampling_model, kspace, maps, target)['output']
    (output - target).abs().mean().backward()
    assert len(steps) == 4
    for index, step in enumerate(steps):
        assert step.unscaled.grad is not None and step.unscaled.grad.abs().sum() > 0, f"step {index} gets no gradient"


@pytest.mark.parametrize('num_step', [1, 4, 16])
def test_one_reconstruction_per_forward(num_step):
    sampling_model = model(num_step=num_step)
    sampling_model.train()
    run(sampling_model, *data())
    assert sampling_model.reconstructor.calls == 1


def test_steps_read_only_the_kspace_measured_so_far():
    sampling_model = model(num_step=4)
    randomise_residual_output(sampling_model)
    sampling_model.eval()
    kspace, maps, target = data()
    steps = record_steps(sampling_model)
    with torch.no_grad():
        final_mask = run(sampling_model, kspace, maps, target)['mask']
        unacquired = (final_mask == 0).unsqueeze(-1).expand_as(kspace)
        run(sampling_model, rescale_values_keeping_zeros(kspace, unacquired), maps, target)
        calibration = (steps[0].old_mask > 0).unsqueeze(-1).expand_as(kspace)
        run(sampling_model, rescale_values_keeping_zeros(kspace, calibration), maps, target)
    original, unacquired_changed, calibration_changed = steps[:4], steps[4:8], steps[8:]
    for before, after in zip(original, unacquired_changed):
        assert torch.equal(before.unscaled, after.unscaled)
    assert not torch.equal(original[0].unscaled, calibration_changed[0].unscaled), "the adaptive arm ignores its data"


def test_twin_ignores_kspace_values_but_reads_its_mask():
    twin = model(num_step=4, data_blind=True)
    randomise_residual_output(twin)
    twin.eval()
    kspace, maps, target = data()
    steps = record_steps(twin)
    with torch.no_grad():
        first = run(twin, kspace, maps, target)
        second = run(twin, rescale_values_keeping_zeros(kspace, kspace != 0), maps, target)
    for before, after in zip(steps[:4], steps[4:]):
        assert torch.equal(before.unscaled, after.unscaled)
    assert torch.equal(first['mask'], second['mask'])

    calibration = steps[0].old_mask
    header = twin._step_header(kspace, calibration)
    context = twin._step_context(header, kspace * calibration.unsqueeze(-1), 1)
    assert context.measured is None
    mask_net = twin.sampler.mask_net
    with torch.no_grad():
        residual_at_calibration = mask_net.residual_logits(calibration, context, LABELS)
        residual_later = mask_net.residual_logits(steps[1].old_mask, context, LABELS)
    assert not torch.equal(residual_at_calibration, residual_later), "the twin's residual ignores its mask"
    with pytest.raises(ValueError, match='must not be handed'):
        mask_net.inputs(calibration, StepContext(1, 4, header[0], header[1], torch.view_as_complex(kspace), None))


def test_adaptive_arm_and_twin_start_bitwise_identical():
    adaptive = model(num_step=4, data_blind=False, seed=3)
    after_adaptive = torch.rand(3)
    twin = model(num_step=4, data_blind=True, seed=3)
    after_twin = torch.rand(3)
    assert torch.equal(after_adaptive, after_twin)
    adaptive_state, twin_state = adaptive.state_dict(), twin.state_dict()
    assert adaptive_state.keys() == twin_state.keys()
    assert all(torch.equal(adaptive_state[key], twin_state[key]) for key in adaptive_state)

    adaptive.train()
    twin.train()
    kspace, maps, target = data()
    adaptive_output, twin_output = run(adaptive, kspace, maps, target), run(twin, kspace, maps, target)
    assert all(torch.equal(a, t) for a, t in zip(adaptive_output['masks'], twin_output['masks']))
    assert torch.equal(adaptive_output['output'], twin_output['output'])


@pytest.mark.parametrize('num_step', [1, 4, 16])
def test_each_step_spends_an_equal_share_of_the_acquired_budget(num_step):
    sampling_model = model(num_step=num_step)
    sampling_model.eval()
    kspace, maps, target = data()
    steps = record_steps(sampling_model)
    with torch.no_grad():
        final_mask = run(sampling_model, kspace, maps, target)['mask']
    calibration, valid = sampling_model._initial_masks(kspace)
    budget = sampling_model.mask_budget(calibration, valid)
    for step in steps:
        torch.testing.assert_close(step.rescaled.flatten(1).sum(dim=1).double(), budget / num_step, rtol=1e-4, atol=1e-2)
    new_locations = ((final_mask - calibration) * valid).flatten(1).sum(dim=1).double()
    assert ((new_locations - budget).abs() <= 5 * budget.sqrt()).all(), (new_locations, budget)
    assert (final_mask * (1 - valid)).sum() == 0


def test_residual_trains_in_its_own_adam_group_outside_muon():
    sampling_model = model(num_step=4, residual_chans=16)
    options = SimpleNamespace(sampler_type='seq_adapt_2d', residual_lr=3e-4, sampler_lr=None, optimizer='muon', lr=1e-3,
                              weight_decay=0.0, fused_adam=False, muon_lr=None, muon_momentum=0.95)
    trainer = Sequential_Trainer.__new__(Sequential_Trainer)
    trainer.model, trainer.options = sampling_model, options
    optimizer = build_optimizer(options, trainer._param_groups(), sampling_model)
    residual = {id(p) for p in sampling_model.sampler.mask_net.residual.parameters()}
    residual_groups = [group for group in optimizer.param_groups if residual & {id(p) for p in group['params']}]
    assert len(residual_groups) == 1 and {id(p) for p in residual_groups[0]['params']} == residual
    assert residual_groups[0]['lr'] == 3e-4 and not residual_groups[0].get('use_muon', False)
    assert optimizer.param_groups[-1] is residual_groups[0]
    muon_groups = [group for group in optimizer.param_groups if group.get('use_muon')]
    assert muon_groups, "the sensitivity net's convolutions should train under Muon"
    sampler = {id(p) for p in sampling_model.sampler.parameters()}
    assert not sampler & {id(p) for group in muon_groups for p in group['params']}


def test_residual_checkpointing_keeps_the_gradients():
    gradients = []
    for checkpointed in (False, True):
        sampling_model = model(num_step=4, residual_checkpoint=checkpointed)
        randomise_residual_output(sampling_model)
        sampling_model.train()
        kspace, maps, target = data()
        output = run(sampling_model, kspace, maps, target)['output']
        (output - target).abs().mean().backward()
        gradients.append({key: p.grad for key, p in sampling_model.named_parameters() if p.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    assert any(key.startswith('sampler.mask_net.residual.') for key in gradients[0])
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key])


def test_multi_step_refuses_inputs_that_need_an_intermediate_reconstruction():
    with pytest.raises(ValueError, match='reconstructs only its final mask'):
        model('conv_2d', num_step=4, with_recon=True)


def test_variable_density_start_matches_loupe():
    loupe = model('loupe_2d', num_step=1, loupe_vd_init_power=4.0)
    sequential = model('seq_adapt_2d', num_step=1, loupe_vd_init_power=4.0)
    assert torch.equal(loupe.sampler.mask_net.mask, sequential.sampler.mask_net.mask)


def test_single_step_proposal_reads_the_calibration():
    sampling_model = model(num_step=1)
    randomise_residual_output(sampling_model)
    kspace, _, _ = data()
    proposal = sampling_model.propose_mask(kspace, anatomy_labels=LABELS)
    assert proposal['mask'].shape == (2, 1, SIZE, SIZE)
    assert (proposal['unscaled'] * proposal['old_mask']).sum() == 0


def test_run_names_tell_the_twin_from_its_arm():
    adaptive = TrainConfig(sampler_type='seq_adapt_2d', num_step=4)
    twin = TrainConfig(sampler_type='seq_adapt_2d', num_step=4, data_blind=True)
    image_only = TrainConfig(sampler_type='seq_adapt_2d', num_step=4, data_channels=['image'])
    assert 'seqadapt2d-4step' in run_name(adaptive) and 'data-blind' not in run_name(adaptive)
    assert 'seqadapt2d-4step' in run_name(twin) and 'data-blind' in run_name(twin)
    assert 'data-image' in run_name(image_only) and 'data-image' not in run_name(adaptive)


def test_channel_ablation_reads_the_image_channel_alone_and_starts_as_its_arm():
    """A_K^img: the k-space channel gets the twin's fixed values, the image channel the slice's data."""
    full = model(num_step=4, seed=3)
    image_only = model(num_step=4, seed=3, data_channels=['image'])
    twin = model(num_step=4, seed=3, data_blind=True)
    full_state, image_state = full.state_dict(), image_only.state_dict()
    assert full_state.keys() == image_state.keys()
    assert all(torch.equal(full_state[key], image_state[key]) for key in full_state)

    full.train()
    image_only.train()
    twin.train()
    kspace, maps, target = data()
    outputs = [run(arm, kspace, maps, target) for arm in (full, image_only, twin)]
    for other in outputs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(outputs[0]['masks'], other['masks']))
        assert torch.equal(outputs[0]['output'], other['output'])

    calibration, _ = image_only._initial_masks(kspace)
    header = image_only._step_header(kspace, calibration)
    context = image_only._step_context(header, kspace * calibration.unsqueeze(-1), 1)
    mask_net = image_only.sampler.mask_net
    with torch.no_grad():
        image_inputs = mask_net.inputs(calibration, context, LABELS)
        full_inputs = full.sampler.mask_net.inputs(calibration, context, LABELS)
    assert torch.equal(image_inputs[:, 0], full_inputs[:, 0])
    assert torch.equal(image_inputs[:, 1], mask_net.fixed_data[:, 1].expand_as(image_inputs[:, 1]))
    assert not torch.equal(full_inputs[:, 1], image_inputs[:, 1])

    randomise_residual_output(image_only)
    image_only.eval()
    steps = record_steps(image_only)
    with torch.no_grad():
        run(image_only, kspace, maps, target)
        run(image_only, rescale_values_keeping_zeros(kspace, calibration.unsqueeze(-1).bool().expand_as(kspace)),
            maps, target)
    assert not torch.equal(steps[0].unscaled, steps[4].unscaled), "the image-only arm ignores its data"


def test_recon_input_arm_shares_every_weight_but_one_input_and_starts_as_its_arm():
    """A_K^rec: the residual's first convolution gains the reconstruction channel; nothing else differs from A_K."""
    adaptive = model(num_step=4, seed=3)
    after_adaptive = torch.rand(3)
    with_recon = model(num_step=4, seed=3, sampler_input='recon')
    after_recon = torch.rand(3)
    assert torch.equal(after_adaptive, after_recon)
    adaptive_state, recon_state = adaptive.state_dict(), with_recon.state_dict()
    frozen = {key for key in recon_state if key.startswith(('input_reconstructor.', 'input_sensetivity_net.'))}
    assert frozen and set(recon_state) - frozen == set(adaptive_state)
    widened = [key for key in adaptive_state if recon_state[key].shape != adaptive_state[key].shape]
    assert len(widened) == 1 and recon_state[widened[0]].shape[1] == adaptive_state[widened[0]].shape[1] + 1
    assert torch.equal(recon_state[widened[0]][:, :-1], adaptive_state[widened[0]])
    assert all(torch.equal(recon_state[key], adaptive_state[key]) for key in adaptive_state if key not in widened)
    assert not any(p.requires_grad for p in with_recon.input_reconstructor.parameters())
    assert not any(p.requires_grad for p in with_recon.input_sensetivity_net.parameters())

    adaptive.train()
    with_recon.train()
    assert not with_recon.input_reconstructor.training and not with_recon.input_sensetivity_net.training
    kspace, maps, target = data()
    adaptive_output, recon_output = run(adaptive, kspace, maps, target), run(with_recon, kspace, maps, target)
    assert all(torch.equal(a, r) for a, r in zip(adaptive_output['masks'], recon_output['masks']))
    assert torch.equal(adaptive_output['output'], recon_output['output'])
    (recon_output['output'] - target).abs().mean().backward()
    assert all(p.grad is None for p in with_recon.input_reconstructor.parameters())
    assert with_recon.reconstructor.scale.grad is not None

    calibration, _ = with_recon._initial_masks(kspace)
    header = with_recon._step_header(kspace, calibration)
    context = with_recon._step_context(header, kspace * calibration.unsqueeze(-1), 1, kspace, calibration, LABELS)
    with torch.no_grad():
        frozen_maps, _ = with_recon.input_sensetivity_net(torch.view_as_complex(kspace.contiguous()),
                                                          mask_method=with_recon.sense_mask_method)
        expected = with_recon.reconstruct_from_mask(kspace, calibration, frozen_maps, LABELS,
                                                    reconstructor=with_recon.input_reconstructor)
        inputs = with_recon.sampler.mask_net.inputs(calibration, context, LABELS)
    torch.testing.assert_close(context.reconstruction, expected[:, 0] / header[2][0].view(-1, 1, 1))
    assert torch.equal(inputs[:, 2], context.reconstruction)


def test_recon_input_arm_loads_the_base_into_both_reconstructors_and_trains_only_one(tmp_path):
    """The fine-tune start of A_K^rec: the base's reconstructor goes into the trained one and into the frozen copies, and
    the optimizer never sees the copies."""
    base = model(sampler_type='loupe_2d', num_step=1, seed=7)
    with torch.no_grad():
        base.reconstructor.scale.fill_(1.7)
    torch.save({'model': base.state_dict()}, tmp_path / 'base.pt')
    with_recon = model(num_step=4, seed=3, sampler_input='recon')
    trainer = Sequential_Trainer.__new__(Sequential_Trainer)
    trainer.model, trainer.device = with_recon, torch.device('cpu')
    trainer.logger = SimpleNamespace(info=lambda message: None)
    trainer._load_pretrained_recon(str(tmp_path / 'base.pt'))
    trainer._load_input_reconstructor(str(tmp_path / 'base.pt'))
    assert with_recon.reconstructor.scale.item() == pytest.approx(1.7)
    assert with_recon.input_reconstructor.scale.item() == pytest.approx(1.7)
    base_sens = base.sensetivity_net.state_dict()
    assert all(torch.equal(value, with_recon.input_sensetivity_net.state_dict()[key]) for key, value in base_sens.items())
    assert not with_recon.input_reconstructor.scale.requires_grad

    trainer.options = SimpleNamespace(sampler_type='seq_adapt_2d', residual_lr=1e-4, sampler_lr=None, optimizer='muon',
                                      lr=1e-3, weight_decay=0.0, fused_adam=False, muon_lr=None, muon_momentum=0.95)
    optimizer = build_optimizer(trainer.options, trainer._param_groups(), with_recon)
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    frozen = {id(p) for p in [*with_recon.input_reconstructor.parameters(), *with_recon.input_sensetivity_net.parameters()]}
    assert not optimized & frozen and id(with_recon.reconstructor.scale) in optimized
    with pytest.raises(ValueError, match='input_recon_checkpoint'):
        trainer._load_input_reconstructor(None)


def test_recon_input_refuses_the_twin_and_other_samplers():
    with pytest.raises(ValueError, match='cannot read a reconstruction'):
        model(num_step=4, data_blind=True, sampler_input='recon')
    with pytest.raises(ValueError, match="seq_adapt_2d's"):
        model(sampler_type='loupe_2d', num_step=4, sampler_input='recon')
    with pytest.raises(ValueError, match='sampler_input must be one of'):
        model(num_step=4, sampler_input='prediction')


@pytest.mark.parametrize('arm', [{'sampler_input': 'measured'}, {'sampler_input': 'full'}, SEQMRI],
                         ids=['measured', 'clairvoyant', 'seqmri'])
def test_kscr2_arms_start_from_the_twins_weights_masks_and_random_stream(arm):
    """A, C and S share every weight the twin has (S's first conv only gains inputs, appended last) and draw the twin's
    masks at the start; S's in-loop reconstructions draw nothing from the random stream."""
    twin = model(num_step=4, data_blind=True, seed=3)
    after_twin = torch.rand(3)
    adaptive = model(num_step=4, seed=3, **arm)
    after_adaptive = torch.rand(3)
    assert torch.equal(after_twin, after_adaptive)
    twin_state, adaptive_state = twin.state_dict(), adaptive.state_dict()
    assert twin_state.keys() == adaptive_state.keys()
    for key, value in twin_state.items():
        shared = adaptive_state[key] if adaptive_state[key].shape == value.shape else adaptive_state[key][:, :value.shape[1]]
        assert torch.equal(shared, value), key

    twin.train()
    adaptive.train()
    kspace, maps, target = data()
    twin_output = run(twin, kspace, maps, target)
    twin_stream = torch.get_rng_state()
    adaptive_output = run(adaptive, kspace, maps, target)
    assert torch.equal(torch.get_rng_state(), twin_stream)
    assert all(torch.equal(a, t) for a, t in zip(adaptive_output['masks'], twin_output['masks']))
    assert torch.equal(adaptive_output['output'], twin_output['output'])


def test_clairvoyant_arm_reads_unacquired_kspace():
    clairvoyant = model(num_step=4, sampler_input='full')
    randomise_residual_output(clairvoyant)
    clairvoyant.eval()
    kspace, maps, target = data()
    steps = record_steps(clairvoyant)
    with torch.no_grad():
        final_mask = run(clairvoyant, kspace, maps, target)['mask']
        unacquired = (final_mask == 0).unsqueeze(-1).expand_as(kspace)
        run(clairvoyant, rescale_values_keeping_zeros(kspace, unacquired), maps, target)
    assert not torch.equal(steps[0].unscaled, steps[4].unscaled), "the clairvoyant arm ignores unacquired k-space"


def test_seqmri_arm_reconstructs_before_its_steps_and_trains_the_reconstructor_through_the_sampler():
    """Before steps 1 and 3 the trained reconstructor reconstructs y * M_i with M_i detached; its prediction carries the
    gradient into the reconstructor's weights, and the final reconstruction is still the forward's one more call."""
    seqmri = model(num_step=4, **SEQMRI)
    randomise_residual_output(seqmri)
    seqmri.train()
    seen = []
    predict = seqmri._in_loop_prediction

    def recording(full_kspace, mask, sense_maps, extent, anatomy_labels=None):
        predicted = predict(full_kspace, mask, sense_maps, extent, anatomy_labels)
        seen.append(SimpleNamespace(mask=mask, predicted=predicted))
        return predicted

    seqmri._in_loop_prediction = recording
    kspace, maps, target = data()
    output = run(seqmri, kspace, maps, target)
    assert seqmri.reconstructor.calls == 1 + len(SEQMRI['recon_steps'])
    assert len(seen) == 2
    assert torch.equal(seen[0].mask, output['masks'][0]) and torch.equal(seen[1].mask, output['masks'][2])
    assert not any(s.mask.requires_grad for s in seen) and all(s.predicted.requires_grad for s in seen)
    (output['output'] - target).abs().mean().backward()
    with_in_loop_path = seqmri.reconstructor.scale.grad.clone()

    seqmri.zero_grad(set_to_none=True)
    seqmri._in_loop_prediction = lambda *args, **kwargs: predict(*args, **kwargs).detach()
    (run(seqmri, kspace, maps, target)['output'] - target).abs().mean().backward()
    assert not torch.equal(seqmri.reconstructor.scale.grad, with_in_loop_path), "the in-loop path sends no gradient"


def test_seqmri_steps_read_the_prediction_from_the_first_in_loop_reconstruction_on():
    seqmri = model(num_step=4, **SEQMRI)
    randomise_residual_output(seqmri)
    seqmri.eval()
    kspace, maps, target = data()
    steps = record_steps(seqmri)
    with torch.no_grad():
        run(seqmri, kspace, maps, target)
        seqmri.reconstructor.scale.fill_(3.0)
        run(seqmri, kspace, maps, target)
    before, after = steps[:4], steps[4:]
    assert torch.equal(before[0].unscaled, after[0].unscaled), "step 0 comes before any in-loop reconstruction"
    assert not torch.equal(before[1].unscaled, after[1].unscaled), "step 1 ignores its in-loop reconstruction"


def test_virtual_coils_are_orthonormal_phase_referenced_and_blind_to_per_coil_phase():
    kspace, _, _ = data()
    coils = torch.view_as_complex(kspace.contiguous())
    calibration = model(num_step=4, **SEQMRI)._initial_masks(kspace)[0]
    weights = acs_compression(coils * calibration, 2)
    identity = torch.eye(2, dtype=weights.dtype).expand(2, -1, -1)
    torch.testing.assert_close(weights.conj().transpose(1, 2) @ weights, identity, atol=1e-5, rtol=0)
    flat = (coils * calibration).flatten(2)
    peak = flat.abs().pow(2).sum(dim=1).argmax(dim=1)
    at_peak = torch.einsum('bcv,bc->bv', weights.conj(), flat[torch.arange(2), :, peak])
    torch.testing.assert_close(at_peak.imag, torch.zeros_like(at_peak.imag), atol=1e-4, rtol=0)
    assert (at_peak.real > 0).all()

    phases = torch.exp(2j * torch.pi * torch.rand(2, COILS, 1, 1, generator=torch.Generator().manual_seed(4)))
    rotated = coils * phases
    kspace_peak = calibration_peaks(coils * calibration)[1]
    channels = virtual_coil_channels(coils, weights, kspace_peak)
    rotated_channels = virtual_coil_channels(rotated, acs_compression(rotated * calibration, 2), kspace_peak)
    assert channels.shape == (2, 3 * 2, SIZE, SIZE)
    torch.testing.assert_close(channels, rotated_channels, atol=1e-3, rtol=0)


def test_run_names_tell_the_kscr2_arms_apart():
    names = {sampler_input: run_name(TrainConfig(sampler_type='seq_adapt_2d', num_step=16, sampler_input=sampler_input))
             for sampler_input in ('measured', 'full', 'seqmri')}
    assert len(set(names.values())) == 3
    assert 'clairvoyant' in names['full'] and 'seqmri-input' in names['seqmri']


def test_kscr2_inputs_refuse_the_twin_bad_steps_and_other_samplers():
    for arm in ({'sampler_input': 'full'}, SEQMRI):
        with pytest.raises(ValueError, match='cannot read a reconstruction'):
            model(num_step=4, data_blind=True, **arm)
        with pytest.raises(ValueError, match="seq_adapt_2d's"):
            model(sampler_type='loupe_2d', num_step=4, **arm)
    with pytest.raises(ValueError, match='in-loop reconstructions need steps'):
        model(num_step=4, sampler_input='seqmri', recon_steps=(1, 4))
    with pytest.raises(ValueError, match='in-loop reconstructions need steps'):
        model(num_step=1, sampler_input='seqmri', recon_steps=(0,))


def test_channel_ablation_refuses_unknown_or_blind_channel_sets():
    with pytest.raises(ValueError, match='non-empty subset'):
        model(num_step=4, data_channels=['phase'])
    with pytest.raises(ValueError, match='non-empty subset'):
        model(num_step=4, data_channels=[])
    with pytest.raises(ValueError, match='data-blind twin reads no data channel'):
        model(num_step=4, data_blind=True, data_channels=['image'])
    with pytest.raises(ValueError, match='seq_adapt_2d'):
        model(sampler_type='loupe_2d', num_step=4, data_channels=['image'])
