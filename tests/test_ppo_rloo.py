"""Leave-one-out (RLOO) advantages and the same-batch replay they rely on."""
import pytest
import torch

from activemri.envs.mri_env import MRIEnv
from activemri.training.ppo_trainer import PPOTrainConfig, PPOTrainer, leave_one_out_advantages, rewards_to_go

SHAPE = [32, 32]
BUDGET = 3


def _env(num_batches: int = 2) -> MRIEnv:
    torch.manual_seed(0)
    loader = [
        {"kspace": torch.randn(2, 4, *SHAPE, 2), "target": torch.rand(2, 1, *SHAPE)} for _ in range(num_batches)
    ]
    return MRIEnv(
        loader, SHAPE, BUDGET, num_cascades=1, chans=4, pools=2, device="cpu",
        reward_mode="sparse", action_space="point", points_per_step=16, center_block=4,
    )


def test_rewards_to_go_sums_the_remaining_discounted_rewards():
    rewards = torch.tensor([[1.0], [2.0], [4.0]])

    assert torch.allclose(rewards_to_go(rewards, 0.5), torch.tensor([[3.0], [4.0], [4.0]]))


def test_leave_one_out_subtracts_the_other_draws_mean():
    terminal = torch.tensor([[1.0, 10.0], [2.0, 20.0], [6.0, 30.0]])  # [M=3, B=2]
    group_rewards = torch.zeros(3, 2, 2)
    group_rewards[:, -1] = terminal

    advantages, returns = leave_one_out_advantages(group_rewards, gamma=1.0)

    expected = torch.tensor([[1 - 4.0, 10 - 25.0], [2 - 3.5, 20 - 20.0], [6 - 1.5, 30 - 15.0]])
    assert torch.allclose(advantages, expected[:, None].expand(-1, 2, -1))
    assert torch.allclose(returns, terminal[:, None].expand(-1, 2, -1))


def test_leave_one_out_ignores_per_slice_offsets():
    group_rewards = torch.randn(4, 3, 2)
    shifted = group_rewards.clone()
    shifted[:, -1] += torch.tensor([5.0, -7.0])

    assert torch.allclose(
        leave_one_out_advantages(group_rewards, 1.0)[0], leave_one_out_advantages(shifted, 1.0)[0], atol=1e-6
    )


def test_same_batch_reset_replays_the_previous_slices():
    env = _env()
    first_obs, _ = env.reset()
    target = env.state["target"]
    env.step(torch.zeros(2, 16, dtype=torch.long))

    replay_obs, _ = env.reset(same_batch=True)

    assert torch.equal(replay_obs, first_obs)
    assert env.state["target"] is target
    assert env.state["steps_taken"] == 0
    assert env.state["mask"].sum() == 2 * 4 * 4


def test_plain_reset_moves_to_the_next_batch():
    env = _env()
    env.reset()
    target = env.state["target"]

    env.reset()

    assert not torch.equal(env.state["target"], target)


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    options = PPOTrainConfig(
        episodes_per_update=4, group_size=2, advantage_estimator="rloo", chans=4, num_pools=2,
        out_dir=str(tmp_path),
    )
    env = _env()
    return PPOTrainer(options, env, env, torch.device("cpu"))


def test_rloo_rollout_advantages_sum_to_zero_per_group(trainer):
    rollout, stats = trainer.collect_rollout()

    steps, batch = BUDGET, 2
    advantages = rollout.advantages.view(2, 2, steps, batch)  # [groups, M, T, B]
    assert torch.allclose(advantages.sum(dim=1), torch.zeros(2, steps, batch), atol=1e-4)
    assert advantages.abs().max() > 1e-2
    assert "train/within_slice_return_std" in stats


def test_rloo_needs_at_least_two_draws(tmp_path):
    options = PPOTrainConfig(advantage_estimator="rloo", group_size=1, out_dir=str(tmp_path))

    with pytest.raises(ValueError, match="group_size"):
        PPOTrainer(options, _env(), None, torch.device("cpu"))
