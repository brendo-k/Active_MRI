"""The loss_domain__loupe_loss_partial_fourier launcher (scripts/launch/hpc/loss_domain__loupe_loss_partial_fourier/
common.sh): the RSS control is fscr1's tier-F LOUPE recipe, and every other arm is the control at its seed with only the
loss and/or the map's symmetry changed, as train.sbatch builds them on CoreHPC.

Run with ``.venv/bin/python -m pytest tests/test_loupe_loss_pf_launch.py`` from the repo root.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.test_multistep_launch import composed, corehpc_arguments  # noqa: E402

LAUNCH = REPO / 'scripts' / 'launch' / 'hpc' / 'loss_domain__loupe_loss_partial_fourier'
NAMING = {'output.run_name', 'output.out_dir'}
CHANGES = {'Lrss': {}, 'Lcplx': {'loss.loss_type': 'complex_l1'}, 'Lrsssym': {'sampler.loupe_point_symmetric': True},
           'Lcplxsym': {'loss.loss_type': 'complex_l1', 'sampler.loupe_point_symmetric': True}}


def arguments(arm, exp_dir):
    env = {key: value for key, value in os.environ.items() if key != 'LAB_SERVER_TEST'}
    env.update(GPU_NAME='test', EXP_DIR=str(exp_dir))
    script = f'source "{LAUNCH}/common.sh" && loss_pf_arm_args "$1" && printf "%s\\0" "${{ARM_ARGS[@]}}" "${{RECIPE[@]}}"'
    built = subprocess.run(['bash', '-c', script, 'bash', arm], env=env, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return built.stdout.split('\0')[:-1] + [f'output.out_dir={exp_dir}/runs/{arm}']


def leaves(cfg, prefix=''):
    flat = {}
    for key, value in cfg.items():
        name = f'{prefix}{key}'
        flat.update(leaves(value, f'{name}.') if isinstance(value, dict) else {name: value})
    return flat


def config(args):
    return leaves(OmegaConf.to_container(composed(args), resolve=True))


def differing(first, second):
    return {key for key in first.keys() | second.keys() if first.get(key) != second.get(key)} - NAMING


def test_the_rss_control_is_fscr1s_loupe_recipe(tmp_path):
    assert differing(config(arguments('Lrss_1_s42_tF', tmp_path)), config(corehpc_arguments('L_1_s42_fscr1_tF', tmp_path))) == set()


@pytest.mark.parametrize('seed', [42, 43])
@pytest.mark.parametrize('kind', sorted(CHANGES))
def test_each_arm_is_the_control_at_its_seed_with_only_its_changes(kind, seed, tmp_path):
    arm, control = config(arguments(f'{kind}_1_s{seed}_tF', tmp_path)), config(arguments(f'Lrss_1_s{seed}_tF', tmp_path))
    assert differing(arm, control) == set(CHANGES[kind])
    assert all(arm[key] == value for key, value in CHANGES[kind].items())
    assert arm['training.seed'] == seed and arm['output.run_name'] == f'{kind}_1_s{seed}_tF'


@pytest.mark.parametrize('arm', ['Lrss_1_s44_tF', 'Lcplx_4_s42_tF', 'Lsym_1_s42_tF', 'Lrss_1_s42_tS5'])
def test_other_names_are_refused(arm, tmp_path):
    with pytest.raises(AssertionError, match='unknown loss_domain__loupe_loss_partial_fourier arm'):
        arguments(arm, tmp_path)
