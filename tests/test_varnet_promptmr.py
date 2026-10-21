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
