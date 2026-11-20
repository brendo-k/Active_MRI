"""The multistep_adaptivity__seqmri_lr_bias_r8 launcher (scripts/launch/hpc/multistep_adaptivity__seqmri_lr_bias_r8/
common.sh): each arm composes to its fscr1 control's config with exactly the one change the proposal names, as
train.sbatch builds them on CoreHPC.

Run with ``.venv/bin/python -m pytest tests/test_seqmri_lr_bias_launch.py`` from the repo root.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from activemri.training.builders import build_train_config  # noqa: E402
from tests.test_multistep_launch import composed, corehpc_arguments  # noqa: E402

LAUNCH = REPO / 'scripts' / 'launch' / 'hpc' / 'multistep_adaptivity__seqmri_lr_bias_r8'
FSCR1_RUNS = '/fscr1/runs'
NAMING = {'output.run_name', 'output.out_dir'}
ARMS = {
    'F_4_s42_fscr1_tF_r150': ('F_4_s42_fscr1_tF', {'output.checkpoint': f'{FSCR1_RUNS}/F_4_s42_fscr1_tF/last_model.pt',
                                                   'training.num_epochs': 150}),
    'Fb_4_s42_fscr1_tF_r150': ('Fb_4_s42_fscr1_tF', {'output.checkpoint': f'{FSCR1_RUNS}/Fb_4_s42_fscr1_tF/last_model.pt',
                                                     'training.num_epochs': 150}),
    'L_1_s42_fscr1_tF_r150': ('L_1_s42_fscr1_tF', {'output.checkpoint': f'{FSCR1_RUNS}/L_1_s42_fscr1_tF/last_model.pt',
                                                   'training.num_epochs': 150}),
    'F_4_s42_lrs_tF': ('F_4_s42_fscr1_tF', {'sampler.lr': 1e-4}),
    'Fb_4_s42_lrs_tF': ('Fb_4_s42_fscr1_tF', {'sampler.lr': 1e-4}),
    'F_4_s42_route_tF': ('F_4_s42_fscr1_tF', {'sampler.seqmri_union_route': True}),
    'Fb_4_s42_route_tF': ('Fb_4_s42_fscr1_tF', {'sampler.seqmri_union_route': True}),
    'F_4_s42_routebias_tF': ('F_4_s42_fscr1_tF', {'sampler.seqmri_union_route': True, 'sampler.seqmri_step_bias': True}),
    'Fb_4_s42_routebias_tF': ('Fb_4_s42_fscr1_tF', {'sampler.seqmri_union_route': True,
                                                    'sampler.seqmri_step_bias': True}),
}


def lr_bias_arguments(arm, exp_dir):
    env = {key: value for key, value in os.environ.items() if key != 'LAB_SERVER_TEST'}
    env.update(GPU_NAME='test', EXP_DIR=str(exp_dir), FSCR1_RUNS=FSCR1_RUNS)
    script = (f'source "{LAUNCH}/common.sh" && lr_bias_arm_args "$1" && '
              'printf "%s\\0" "${ARM_ARGS[@]}" "${RECIPE[@]}"')
    built = subprocess.run(['bash', '-c', script, 'bash', arm], env=env, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return built.stdout.split('\0')[:-1] + [f'output.out_dir={exp_dir}/runs/{arm}']


def leaves(cfg, prefix=''):
    flat = {}
    for key, value in cfg.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(leaves(value, f'{name}.'))
        else:
            flat[name] = value
    return flat


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_each_arm_is_its_fscr1_control_with_one_change(arm, tmp_path):
    control, change = ARMS[arm]
    new = leaves(OmegaConf.to_container(composed(lr_bias_arguments(arm, tmp_path)), resolve=True))
    old = leaves(OmegaConf.to_container(composed(corehpc_arguments(control, tmp_path)), resolve=True))
    differing = {key for key in new.keys() | old.keys() if new.get(key) != old.get(key)} - NAMING
    assert differing == set(change), differing
    assert all(new[key] == value for key, value in change.items())
    assert new['output.run_name'] == arm


@pytest.mark.parametrize('arm', ['F_4_s42_fscr1_tS5_r150', 'L_1_s42_lrs_tF', 'F_4_s42_bias_tF', 'F_4_s43_route_tF',
                                 'F_4_s42_fscr1_tF'])
def test_other_names_are_refused(arm, tmp_path):
    with pytest.raises(AssertionError, match='unknown multistep_adaptivity__seqmri_lr_bias_r8 arm'):
        lr_bias_arguments(arm, tmp_path)


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_the_trainer_options_carry_the_change(arm, tmp_path):
    """The change reaches the TrainConfig train.py builds: a resume arm's checkpoint (the trainer resumes only when
    options.checkpoint is set) and 150 epochs, the sampler's lr, the per-step maps."""
    options = build_train_config(composed(lr_bias_arguments(arm, tmp_path)))
    control = ARMS[arm][0]
    resumed = arm.endswith('_r150')
    assert options.checkpoint == (f'{FSCR1_RUNS}/{control}/last_model.pt' if resumed else None)
    assert options.num_epochs == (150 if resumed else 100)
    assert options.sampler_lr == (1e-4 if '_lrs_' in arm else None)
    assert options.seqmri_step_bias == ('_routebias_' in arm)
    assert options.seqmri_union_route == ('_route' in arm)
