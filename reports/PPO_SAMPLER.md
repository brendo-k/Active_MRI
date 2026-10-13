# PPO k-space sampler with alternating reconstructor training

Status: implemented and smoke-tested; **no real training run yet** (see [Open items](#open-items)).

## Summary

An on-policy PPO option for the sequential k-space sampler, next to the existing DQN and straight-through/REINFORCE samplers. It follows the ideas of L2SR (Learning to Sample and Reconstruct, <https://arxiv.org/abs/2212.02190>, code at `yangpuPKU/L2SR-Learning-to-Sample-and-Reconstruct`):

- a sparse terminal reward,
- masking of already-acquired actions,
- alternating sampler and reconstructor training.

Differences from the reference are in [Differences from L2SR](#differences-from-l2sr).

| Aspect | Choice |
|---|---|
| Algorithm | Clipped-surrogate PPO with GAE |
| Networks | Policy and value coupled on one shared U-Net trunk, `vf_coef` weighting the value loss |
| Reconstructor | The existing VarNet + sensitivity net, a separate module, trained in alternating phases |
| Action space | 2D k-space point (65536 actions), 128 points acquired per step |
| Preselected region | 16x16 fully sampled center block |
| Horizon at R=4 | (256*256/4 - 16*16) / 128 = **126 steps** |
| Reward | Sparse: 0 until the last step, then the episode's NMSE improvement |

## What L2SR actually does

Two things worth knowing about the reference, since the repo name and some identifiers say PPO:

1. Training uses Stable-Baselines3 **`A2C`**, not PPO (`train_sparse.py`, `train_alter.py`). We implement real clipped PPO as requested. `ppo_epochs=1` with a very large clip range approximates A2C.
2. The policy and value function **share one U-Net encoder** (`feature_extractor/`); the value is a linear head on shared features and the loss is `policy + vf_coef * value + ent_coef * entropy`. This is why `vf_coef` exists in the original and why it matters here (see [Coupled actor-critic](#coupled-actor-critic)).

L2SR's hyperparameters (`gamma=0.5`, RMSprop, 128x128 images) come with a budget of 16 steps (`resolution/accel - resolution/low_freq_ratio`). They do not transfer to a 126-step horizon unchanged.

## Design

### Files

| File | Role |
|---|---|
| `activemri/samplers/ppo.py` | `ActorCriticNetwork`, `PPOPolicy`, `sequence_log_prob`, `sample_without_replacement` |
| `activemri/training/ppo_trainer.py` | `PPOTrainConfig`, `PPOTrainer` (rollouts, GAE, PPO update, reconstructor phase, checkpoints) |
| `activemri/envs/mri_env.py` | `MRIEnv` gains `reward_mode`, `action_space`, `points_per_step`, `center_block` |
| `scripts/train_ppo.py`, `configs/train_ppo.yaml` | Hydra entry point and config |

### Environment (`MRIEnv`)

- **`reward_mode="sparse"`**: the reconstructor runs only at `reset()` and at the terminal step. Intermediate rewards are 0; the terminal reward is `NMSE_initial - NMSE_final`, equal to the sum of the dense per-step rewards. The observation's second channel pair is the zero-filled image instead of the current reconstruction. `"dense"` (the default) is unchanged.
- **`action_space="point"`**: an action is a flat `row * W + col` index into the H x W k-space. `step()` accepts `[B, K]` long tensors (K points per element), or `List[int]` as before. The preselected region is a `center_block` square.
- **`action_space="line"`** (default): one column per action, K columns per step if `points_per_step > 1`. DQN scripts use this and are unaffected.
- `target_fraction` exposes the fraction of k-space acquired at episode end; the value head uses it.
- Observation is unchanged in shape: `[B, 5, H, W]` = acquired k-space (re/im), second image-or-k-space pair (re/im), mask.

### Coupled actor-critic

`ActorCriticNetwork` is a single `fastmri.models.Unet` subclass whose forward also returns the pooled bottleneck:

- **Policy head**: the decoder's `[B, 1, H, W]` output map, flattened to 65536 logits (`line` mode: mean over height). Acquired actions get logit `-1e10` (large finite rather than `-inf`, so `p * log p` stays NaN-free).
- **Value head**: pooled bottleneck features concatenated with `acquired_fraction / target_fraction`, then an MLP. With a finite horizon and a terminal reward the value depends on steps remaining, which the mask only encodes implicitly.
- The U-Net keeps `fastmri.models.Unet`'s state-dict keys, so `load_pretrained_backbone` copies every layer of a VarNet cascade whose shape matches (all interior layers; the first and last convs differ in channel count and stay at random init).
- **Why shared**: matches L2SR, costs one U-Net pass per step instead of two, and the encoder features useful for choosing where to sample are also informative about remaining reward. The coupling makes `vf_coef` meaningful: it sets whether value-regression or policy gradients dominate the shared features. Gradient clipping is joint (`max_grad_norm=0.5`).

### Multi-point actions

Each step draws K = 128 distinct points **without replacement** from one softmax over the not-yet-acquired points.

- **Sampling**: top-K of `logits + Gumbel noise`, which is exactly sequential sampling without replacement.
- **Log-prob** of the ordered K-tuple (Plackett-Luce), in closed form:

  `log p = sum_i log p[a_i] - sum_i log(1 - sum_{j<i} p[a_j])`

  with the running sum in float64 so the remaining mass stays accurate when the policy is peaked. Memory is O(B*K), versus O(B*K*A) for a K-step Python loop with autograd.
- **Entropy**: of the per-step softmax over not-yet-acquired points (an approximation to the sequence entropy, used only as a regulariser).
- **Ratio clipping**: the log-prob of a step is a sum over K terms, so the raw probability ratio swings about K times harder than for one action and the 0.2 clip would block almost all movement. `per_point_ratio: true` (default) clips the geometric-mean per-point ratio `exp((logp - old_logp) / K)` instead. This is a heuristic, not a strict trust region on the joint policy. Set it to `false` for the strict ratio.

### PPO update

- Each update rolls out `episodes_per_update` full episodes (each `training.batch_size` slices in lockstep) with the stochastic policy. All episodes run to the budget, so the value after the last step is 0 and no mid-episode bootstrapping is needed.
- GAE per episode, then flatten; advantages normalised per update.
- Rewards divided by the running std of per-episode discounted returns (raw NMSE improvements are ~1e-2, which would otherwise make the value loss negligible). The std state is saved in checkpoints.
- `ppo_epochs=4` over minibatches of 32; early stop when mean approx-KL exceeds `1.5 * target_kl`.
- Observations are kept on the GPU for the rollout: about 1.3 MB per observation, so roughly 1.3 GB for 2 episodes x 126 steps x batch 4.

### Alternating training

```
for round in range(num_rounds):
    PPO phase           # policy + value trained against the *frozen* reconstructor
    reconstructor phase # reconstructor trained on masks from the current policy
```

- **PPO phase**: `steps_per_round` env steps in round 0, `later_round_step_fraction` of that afterwards. Policy learning rate is scaled by `round_lr_decay**round`.
- **Reconstructor phase**: a trainable copy of the reconstructor is fit for `recon_steps_per_round` gradient steps. Each step resets the env (fresh slices), rolls the policy to the end (greedy by default, as L2SR does), then does one supervised step with L1 loss between `|recon|` and the target on that final mask. The sensitivity net stays frozen. The result is copied into both the train and val envs. Validation scores before and after are logged (`val/score_before_recon_phase`, `val/score_after_recon_phase`); the new weights are always accepted.
- After each swap the reward function has changed, so the running reward scale is reset. The critic and optimizer state are kept as a warm start.
- Setting `recon_steps_per_round: 0` disables the reconstructor phase (policy-only PPO against the checkpoint's reconstructor).
- Checkpoints (`latest.pt`, `best.pt`) contain the policy, optimizer, reconstructor weights, running reward std, round and step counters. `best.pt` pairs the best-validation policy with the reconstructor that scored it. `ppo.resume=true` resumes at the saved round; an interrupted reconstructor phase restarts from its beginning.

## Bug found and fixed in `MRIEnv`

`MRIEnv` computed NMSE between the **complex** reconstruction and the **real** magnitude target, so the score was dominated by phase: a full-mask reconstruction scored NMSE about 2.0 where the magnitude comparison gives about 0.017. `_nmse_to_target` now compares `|recon|`.

**This changes the reward the DQN and EMA-DDQN trainers see.** Any previously logged DQN rewards or checkpoints trained against the old reward are not comparable.

## Recommended settings and reasoning

| Setting | Default | Notes |
|---|---|---|
| `gamma` | 1.0 | The objective is the final NMSE improvement of a fixed-length episode, which is exactly the undiscounted return. The dense rewards telescope to the sparse one, so both modes optimise the same thing. With `gamma < 1` the credit for an action at step `t` is scaled by `gamma^(T-t)`: 0.99^125 is about 0.28, 0.5^125 is about 2e-38. |
| `gae_lambda` | 0.95 | Only the last step pays, so try 0.98 to 1.0 once the critic fits well (`train/explained_variance`). If unstable, lower `gamma` to 0.99 before changing `lambda`. |
| `vf_coef` | 0.5 | Standard; revisit if `value_loss` dominates or `explained_variance` stays near 0. |
| `learning_rate` | 1e-4 | Same as DQN; untested for PPO. |
| `recon_steps_per_round` | 500 | Each step is a full VarNet forward+backward on a fresh batch. |
| `num_rounds` | 3 | Round 0 uses `steps_per_round`, later rounds 0.2x that. |

`reward_mode=dense` is available and gives lower-variance credit assignment at the cost of one reconstruction per step (about 126 per episode).

## Verification performed

| Check | Result |
|---|---|
| Sequence log-prob vs brute force (K=3 of 6, one masked action) | Closed form 0.018547397 vs brute force 0.018547393; probabilities over all ordered tuples sum to 0.999997 |
| Sampler vs exact distribution (200k draws) | Max frequency error 1.2e-3 (within sampling noise); a masked action was never sampled |
| Point-mode geometry | Initial mask 256 points with the center block fully set; 126 steps of 128 distinct points; final fraction exactly 0.25; no acquired point re-chosen |
| `act` vs `evaluate` log-probs / values | Agree to about 1e-5 (so the first-epoch ratio is 1) |
| GAE | Matches hand calculation (`gamma=lambda=1`, terminal reward gives advantage `R - V_t`) |
| Sparse reward | Reward is 0 before the last step; terminal reward equals the sum of dense rewards; the reconstructor is called exactly twice per episode (dense: once per step plus reset) |
| Line mode / DQN path | `step([int, int])` works as before; line mode with K=4 acquires 22 columns = target |
| End to end | 2 rounds with the reconstructor phase, wandb offline, then resume into round 2 from `latest.pt`, all completed |

The smoke runs used batch 2, a handful of updates and 1 validation episode, so their val scores are noise and say nothing about learning.

## Open items

- **No real training run.** Whether PPO learns, and how it compares with DQN, greedy oracle and REINFORCE, is unmeasured. Watch `train/explained_variance`, `train/final_nmse`, `train/approx_kl`, `train/clip_fraction` and `val/score`.
- **`per_point_ratio` is untuned.** If `approx_kl` stays near 0 and `clip_fraction` is 0, the policy may be moving too slowly; if it frequently stops early, too fast.
- **Reconstructor phase is unconditional.** New reconstructor weights are always accepted; L2SR instead keeps the best epoch on validation.
- **Initial reconstructor** is the CAIPI-R4-trained `varnet_caipi2d_r4_brain_knee`, not trained on policy masks (the reconstructor phase addresses this).
- **Cost.** In the smoke runs a PPO update of 252 env steps took about 25 s on an A6000 and 3 reconstructor steps about 10 s. The default 100k steps in round 0 is about 400 updates.
- **Possible follow-ups:** per-episode random acceleration (needs the budget fed to the network), best-of-phase reconstructor selection, and comparing against the EMA joint trainer, which interleaves the two networks instead of alternating.

## Usage

```bash
source .venv/bin/activate
python scripts/train_ppo.py                              # defaults: point actions, sparse reward, 3 rounds
python scripts/train_ppo.py env.reward_mode=dense        # per-step rewards
python scripts/train_ppo.py ppo.recon_steps_per_round=0  # policy-only, frozen reconstructor
python scripts/train_ppo.py env.action_space=line env.points_per_step=1  # one column per step
python scripts/train_ppo.py ppo.num_rounds=1 ppo.steps_per_round=2520    # quick sanity run
```

Outputs go to `checkpoints/ppo` (`output.out_dir`); logs go to the wandb project `SeqMRI-PPO`.
