from dataclasses import replace

from activemri.config import TrainConfig
from activemri.reconstructors.config import VarNetReconConfig
from activemri.training.run_naming import run_name


def varnet_tokens(norm='instance', **fields):
    return run_name(replace(TrainConfig(recon=VarNetReconConfig(norm=norm)), **fields)).split('_')


def test_adam_and_instance_norm_add_no_tokens():
    assert not [token for token in varnet_tokens() if 'muon' in token or 'norm' in token]


def test_muon_with_layer_norm_in_both_unets_names_the_run():
    assert {'muon', 'norm-layer'} <= set(varnet_tokens(norm='layer', optimizer='muon', sens_norm='layer'))


def test_a_norm_in_one_unet_names_that_unet():
    assert 'cascade-norm-layer' in varnet_tokens(norm='layer')
    assert 'sens-norm-channel-layer' in varnet_tokens(sens_norm='channel_layer')


def test_muon_settings_name_the_run_only_when_set():
    assert {'muon-lr0.02', 'muon-mom0.9'} <= set(varnet_tokens(optimizer='muon', muon_lr=0.02, muon_momentum=0.9))
    assert not [token for token in varnet_tokens(optimizer='muon') if token.startswith('muon-')]
