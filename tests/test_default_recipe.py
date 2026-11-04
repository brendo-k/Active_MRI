"""The YAML defaults are Muon + LayerNorm; checkpoints keep what they trained with.

The dataclass defaults stay Adam + InstanceNorm on purpose. A pickled TrainConfig holds only the fields that existed when
it was saved, and `getattr` on it falls back to the class attribute, so changing those would rebuild old InstanceNorm
checkpoints with GroupNorm and fail every eval's strict load.
"""
from pathlib import Path

import pytest
import torch.nn as nn
from hydra import compose, initialize_config_dir

from activemri.config import TrainConfig
from activemri.reconstructors.config import VarNetReconConfig
from activemri.training.builders import build_train_config
from activemri.training.muon import MuonAdam
from activemri.training.trainer import build_model, build_optimizer

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
SMALL_LOUPE_VARNET = ['recon=varnet', 'recon.num_cascades=1', 'recon.chans=8', 'sampler.sampler_type=loupe_2d', 'misc.resolution=[64,64]',
                      'sampler.accelerations=[8.0]', 'training.num_step=1']
EARLIER_DEFAULT = ['training.optimizer=adam', 'recon.norm=instance', 'sampler.sens_norm=instance', 'training.channels_last_unets=false']


def options_from_yaml(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return build_train_config(compose('config', overrides=[*SMALL_LOUPE_VARNET, *overrides]))


def norms(module):
    return {type(layer) for layer in module.modules() if isinstance(layer, (nn.InstanceNorm2d, nn.GroupNorm))}


def test_yaml_defaults_are_the_muon_layernorm_recipe():
    options = options_from_yaml()
    assert (options.optimizer, options.recon.norm, options.sens_norm) == ('muon', 'layer', 'layer')
    assert options.fused_adam is True
    assert options.channels_last_unets == 'sens'


def test_the_default_model_trains_layernorm_unets_under_muon():
    options = options_from_yaml()
    model = build_model(options)
    assert norms(model.reconstructor) == {nn.GroupNorm}
    assert norms(model.sensetivity_net) == {nn.GroupNorm}
    optimizer = build_optimizer(options, list(model.parameters()), model)
    assert isinstance(optimizer, MuonAdam)
    muon_groups = [group for group in optimizer.param_groups if group.get('use_muon')]
    assert len(muon_groups) == 1 and muon_groups[0]['params']


def test_the_earlier_default_is_one_override_away():
    options = options_from_yaml(*EARLIER_DEFAULT)
    model = build_model(options)
    assert norms(model.reconstructor) == {nn.InstanceNorm2d}
    assert norms(model.sensetivity_net) == {nn.InstanceNorm2d}
    assert not isinstance(build_optimizer(options, list(model.parameters()), model), MuonAdam)


@pytest.mark.parametrize('config, field, value', [(TrainConfig, 'optimizer', 'adam'), (TrainConfig, 'sens_norm', 'instance'),
                                                  (VarNetReconConfig, 'norm', 'instance')])
def test_options_saved_before_a_field_existed_keep_the_old_behaviour(config, field, value):
    saved_without_the_field = object.__new__(config)
    assert getattr(saved_without_the_field, field, None) == value
