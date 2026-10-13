"""PPOMaskPolicy must reproduce the greedy episode PPOTrainer's MRIEnv plays."""
from types import SimpleNamespace

import pytest
import torch

from activemri.envs.mri_env import MRIEnv
from activemri.samplers.ppo import PPOMaskPolicy, PPOPolicy

SHAPE = [32, 32]
CENTER_BLOCK = 4
POINTS_PER_STEP = 16
BUDGET = 3
CHANS, POOLS = 4, 2


@pytest.fixture
def env_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    kspace = torch.randn(2, 4, *SHAPE, 2)
    loader = [{"kspace": kspace, "target": torch.rand(2, 1, *SHAPE)}]
    env = MRIEnv(
        loader, SHAPE, BUDGET, num_cascades=1, chans=CHANS, pools=POOLS, device="cpu",
        reward_mode="sparse", action_space="point", points_per_step=POINTS_PER_STEP, center_block=CENTER_BLOCK,
    )
    policy = PPOPolicy(
        "point", POINTS_PER_STEP, env.target_fraction, torch.device("cpu"), chans=CHANS, num_pools=POOLS
    )
    path = tmp_path / "ppo.pt"
    torch.save(
        {
            "policy": policy.state_dict(),
            "sensetivity_net": env.sensetivity_net.state_dict(),
            "options": SimpleNamespace(chans=CHANS, num_pools=POOLS, critic_hidden=256),
            "env_config": {
                "action_space": "point", "points_per_step": POINTS_PER_STEP, "budget": BUDGET,
                "center_block": CENTER_BLOCK, "target_fraction": env.target_fraction, "resolution": SHAPE,
            },
        },
        path,
    )
    return env, policy, kspace, str(path)


def test_mask_matches_env_greedy_episode(env_and_checkpoint):
    env, policy, kspace, path = env_and_checkpoint
    obs, _ = env.reset()
    for _ in range(BUDGET):
        actions, _, _ = policy.act(obs, deterministic=True)
        obs, *_ = env.step(actions)

    mask = PPOMaskPolicy(path)(kspace)

    assert torch.equal(mask, env.state["mask"])


def test_mask_acquires_the_target_fraction(env_and_checkpoint):
    env, _, kspace, path = env_and_checkpoint

    mask = PPOMaskPolicy(path)(kspace)

    assert mask.mean().item() == pytest.approx(env.target_fraction)


def test_policy_is_frozen(env_and_checkpoint):
    *_, path = env_and_checkpoint

    policy = PPOMaskPolicy(path).train()

    assert not any(p.requires_grad for p in policy.parameters())
    assert not policy.training
