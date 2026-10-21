"""PromptMR+ additions to the VarNet: the momentum layer, the anatomy prompt and the label-blind control.

Run with ``.venv/bin/python -m pytest tests/test_varnet_promptmr.py`` from the repo root.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.VarNet import MomentumLayer, VarNet  # noqa: E402

COILS, HEIGHT, WIDTH = 2, 32, 32


def conditioned_varnet(seed=0, **kwargs):
    torch.manual_seed(seed)
    return VarNet(num_cascades=3, chans=4, pools=2, num_conditioning_classes=3, prompt=True,
                  adaptive_input=True, momentum_history=2, **kwargs).eval()


def inputs(batch=2):
    torch.manual_seed(1)
    kspace = torch.randn(batch, COILS, HEIGHT, WIDTH, dtype=torch.complex64)
    sense = torch.randn(batch, COILS, HEIGHT, WIDTH, dtype=torch.complex64)
    mask = (torch.rand(batch, 1, HEIGHT, WIDTH) < 0.3).float()
    return kspace * mask, sense, mask


def test_label_blind_control_has_the_conditioned_models_parameters_and_init():
    conditioned, blind = conditioned_varnet().state_dict(), conditioned_varnet(label_blind=True).state_dict()

    assert conditioned.keys() == blind.keys()
    assert all(torch.equal(conditioned[key], blind[key]) for key in conditioned)


def test_label_blind_control_ignores_the_label_and_the_conditioned_model_uses_it():
    blind, conditioned = conditioned_varnet(label_blind=True), conditioned_varnet()
    for model in (blind, conditioned):
        torch.nn.init.normal_(model.anatomy_embedding.embedding.weight)

    with torch.no_grad():
        blind_outputs = [blind(*inputs(), anatomy_labels=torch.tensor(labels)) for labels in ([0, 2], [1, 1])]
        conditioned_outputs = [conditioned(*inputs(), anatomy_labels=torch.tensor(labels)) for labels in ([0, 2], [1, 1])]

    assert torch.equal(*blind_outputs)
    assert not torch.allclose(*conditioned_outputs)


def test_prompt_conditioned_model_refuses_to_run_without_labels():
    with pytest.raises(ValueError, match='anatomy_labels'):
        conditioned_varnet()(*inputs())


def test_momentum_history_is_newest_first_and_starts_from_the_first_cascades_features():
    layer = MomentumLayer(chans=4, history=3)
    first, second = torch.full((1, 4, 2, 2), 1.0), torch.full((1, 4, 2, 2), 2.0)

    _, history = layer(first)
    _, history = layer(second, history)

    assert history[0, ::4, 0, 0].tolist() == [2.0, 1.0, 1.0]


def test_momentum_and_prompt_get_gradient_from_the_output():
    model = conditioned_varnet().train()
    output = model(*inputs(), anatomy_labels=torch.tensor([0, 2]))
    output.abs().mean().backward()
    first_cascade = model.cascades[0].model

    assert first_cascade.momentum[0].fuse[0].weight.grad.abs().sum() > 0
    assert first_cascade.prompts[0].anatomy_weights.weight.grad.abs().sum() > 0
    assert model.anatomy_embedding.embedding.weight.grad[1].abs().sum() == 0


def promptmr_varnet(**kwargs):
    torch.manual_seed(0)
    return VarNet(num_cascades=3, chans=8, pools=3, backbone='promptmr', adaptive_input=True, momentum_history=2,
                  **kwargs).eval()


def test_promptmr_backbone_runs_without_labels_and_gives_prompt_gradient():
    model = promptmr_varnet().train()
    output = model(*inputs())
    output.abs().mean().backward()

    assert output.shape == (2, 1, HEIGHT, WIDTH)
    assert model.cascades[0].model.unet.prompts[0].feature_weights.weight.grad.abs().sum() > 0
    assert model.cascades[0].model.unet.momentum[0].fuse[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('injection', ['film', 'prompt', 'channels'])
def test_promptmr_backbone_uses_the_anatomy_label(injection):
    flags = {'film': injection == 'film', 'prompt': injection == 'prompt'}
    model = promptmr_varnet(num_conditioning_classes=3, **flags)
    torch.nn.init.normal_(model.anatomy_embedding.embedding.weight)
    for parameter in model.parameters():
        if parameter.dim() > 0 and parameter.abs().sum() == 0:
            torch.nn.init.normal_(parameter, std=0.1)

    with torch.no_grad():
        first, second = (model(*inputs(), anatomy_labels=torch.tensor(labels)) for labels in ([0, 0], [2, 2]))

    assert not torch.allclose(first, second)


def test_promptmr_backbone_at_48_chans_matches_promptmr_plus_parameter_count():
    from activemri.reconstructors.VarNet import PromptUnet
    parameters = sum(p.numel() for p in PromptUnet(6, 6, chans=48).parameters())

    # PromptMR+'s PromptUnet for 3 adjacent slices, measured from its repository; it shares one PReLU across its CABs.
    assert abs(parameters - 6_632_081) < 100


def line_mask(batch=2, centre_lines=4, stride=4):
    mask = torch.zeros(batch, 1, HEIGHT, WIDTH)
    mask[..., ::stride] = 1
    mask[..., WIDTH // 2 - centre_lines // 2:WIDTH // 2 + centre_lines // 2] = 1
    return mask


def test_acs_region_is_the_central_block_of_acquired_columns():
    from activemri.reconstructors.csme import acs_region_mask
    region = acs_region_mask(line_mask(centre_lines=6, stride=8))

    assert region[0, 0].sum() == HEIGHT * 6
    assert region[0, 0, :, WIDTH // 2 - 3:WIDTH // 2 + 3].all()


def test_acs_region_refuses_a_mask_without_the_centre():
    from activemri.reconstructors.csme import acs_region_mask
    mask = line_mask()
    mask[..., WIDTH // 2] = 0

    with pytest.raises(ValueError, match='centre'):
        acs_region_mask(mask)


@pytest.mark.parametrize('adaptive_input', [False, True])
def test_estimated_sense_ignores_external_maps_and_trains_every_cascades_estimator(adaptive_input):
    torch.manual_seed(0)
    model = VarNet(num_cascades=2, chans=4, pools=2, estimate_sense=True, adaptive_input=adaptive_input).train()
    kspace, sense, _ = inputs()
    mask = line_mask()
    other_sense = torch.randn_like(sense)

    output = model(kspace * mask, sense, mask)
    output.abs().mean().backward()

    assert torch.equal(output, model(kspace * mask, other_sense, mask))
    assert all(cascade.sense_model.unet.conv_0.conv_0.conv.weight.grad.abs().sum() > 0 for cascade in model.cascades)


def test_external_sense_is_the_default_and_has_no_estimator():
    assert all(cascade.sense_model is None for cascade in VarNet(num_cascades=2, chans=4, pools=2).cascades)
