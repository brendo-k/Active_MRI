# Proposal: brain-only VarNet with a frozen shared mask vs a learned LOUPE mask (R=8)

**Status: approved by the user 2026-11-10 ("it's good to submit"), with three arms and the VD start dropped.**

- Topic / branch: `brain_static_vs_loupe`, branch `exp/brain_static_vs_loupe`,
  worktree `../SeqMRI-worktrees/exp-brain_static_vs_loupe`, report folder `reports/brain_static_vs_loupe/`.
- Roadmap: step 1 (anatomy / header routing).

## Question

On local v1, a mask per condition beat one shared mask by only +0.07 / +0.12 dB on brain (seeds 42 / 43), although
the masks differed by 29–50% of the budget.

Either the mask shape barely matters for brain, or the per-condition gain is held back by joint training: one VarNet
has to reconstruct from several different masks.

This experiment removes the joint training. The VarNet trains on brain only, and the only thing that changes is
whether the mask may keep learning on brain:

- **static**: the shared mask is frozen. Its probabilities are still Bernoulli-sampled every step, as in LOUPE.
- **loupe_shared**: the mask starts from the same shared mask and is trainable (LOUPE straight-through).
- **loupe_uniform**: the mask starts from uniform noise (`loupe_vd_init_power=0`, main's default) and is trainable.

The VD start is dropped at the user's request (2026-11-10). It has repeatedly converged worse than the uniform start.

How to read the result:

- **loupe ≈ static**: for brain, moving away from the shared mask buys nothing, even without joint training. The
  shared-vs-split null is then about mask shape, not about the VarNet.
- **loupe > static** by a clear margin: a brain-specific mask is worth something once the VarNet is brain-only. So the
  small gain in the joint per-condition run points at the joint setup. This is consistent with the bimodal-reconstructor
  hypothesis but does not prove it: this design never trains a joint VarNet. The direct test is the next experiment
  (frozen masks × {blind, FiLM} VarNet, 2×2).
- **static > loupe**: learning the mask on a small brain-only set hurts (it overfits, or the mask and VarNet
  co-adapt into a worse place). Report this; do not read it as "sampling doesn't matter".

## Shared mask source (reused, not retrained)

`~/data/seqmri_checkpoints/loss_domain/runs_v2/loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt`

- Commit 676675b. It includes the brain-flip fix b25d2c5 (checked with `git merge-base --is-ancestor`), so brain DC is
  on row 160. Older shared checkpoints (muon_layernorm_100ep, local_v1) predate the fix and are not used.
- `loupe_2d`, one mask `[1, 320, 320, 1]`, slope 10, VD p=4 start, `budget_over=acquired`, no anatomy conditioning.
- Trained jointly on full local v1 (brain FLAIR + knee PD/PDFS), RSS L1 loss, 100 epochs, Adam + InstanceNorm.
- It was the `rss` (= main-like) arm of loss_domain round 2. Its optimizer and norm do not matter here, because only its
  mask is used.
- Before launch, check that the mask's slope matches (`_load_fixed_masks` checks this).

## Common config (both arms)

- `recon=varnet` (12 cascades), with main defaults: Muon + LayerNorm, `sense_mask_method=null`, fast path.
- `datasets=fastmri_local_v1_flair_knee`, `misc.resolution=[320,320]`, `datasets.train_anatomies=[brain]`.
- **Quarter data**: `datasets.limit_volumes=0.25`. That is 20 of the 80 brain training volumes (about 300 slices),
  drawn with a fixed seed (1234), so both arms and all seeds get the same files.
  - `limit_volumes` also cuts val, to about 10 brain volumes. Final scoring therefore re-evaluates the best checkpoints
    on the **full brain val set (40 volumes, about 605 slices)** with the trainer's metric path. If the eval script
    cannot override `limit_volumes`, I will add that flag.
- `sampler.sampler_type=loupe_2d`, `mask_train_mode=straight_through`, R=8, `budget_over=acquired`, slope 10.
- `sampler.anatomy_conditioning=true` with the three condition classes, in both arms. This gives identical parameter
  shapes, so a given seed starts from an identical VarNet. Only the brain slot gets data.
- Batch 16 and lr 1e-3. The step schedule is scaled to the epoch count: `lr_step_size` = 0.9 × `num_epochs`.
- Seeds 42, 43 and 44, paired across arms.

## Control vs experiment (the single difference)

| Arm | Mask setting | Mask trainable |
|---|---|---|
| static (control) | `sampler.fixed_mask_checkpoints={brain: S, knee-PD: S, knee-PDFS: S}` | no |
| loupe_shared (experiment 1) | `sampler.init_mask_checkpoint=S` | yes |
| loupe_uniform (experiment 2) | `sampler.loupe_vd_init_power=0` (uniform noise start) | yes |

S is the shared checkpoint above. Check before launch: at step 0 both arms must hold bit-identical brain-slot logits.
`_load_fixed_masks` copies raw weights, while `_init_mask_from` rescales by the slope ratio, which is 1 here.

loupe_shared differs from static only in whether the mask learns. loupe_uniform also differs in where the mask starts,
so it answers a slightly different question: can a brain-only LOUPE trained from scratch beat the shared mask? The two
LOUPE arms together show how much the starting point matters (loss-landscape traversal, an open question). Caveat: the
shared mask S was itself learned from a VD p=4 start.

## Noise floor

- Three seeds per arm, compared by seed (static_s vs loupe_s) and by slice on the full brain val set.
- Seed spread is the spread of the static arm across its three seeds.
- Seed-swap check: (loupe_s42 − static_s43) etc. A real effect has to be larger than the gaps between unpaired seeds.

## Mask movement check (needed before any conclusion)

The quarter-data budget gives about 19 steps per epoch, roughly 1/13 of the steps in a full-data run. If the LOUPE mask
never moves, "loupe ≈ static" means nothing.

Measure: the fraction of the sampling budget relocated, ½ Σ|p_final − p_init| / Σ p_init, over the brain slot.
Full-data runs moved 42–57%. This applies to loupe_shared. loupe_uniform always moves a lot from its noise start, so
it gets no movement gate.

## Expected outcomes

| Metric (full brain val, apodized metric mask) | Prediction (loupe_shared − static) |
|---|---|
| PSNR | about 0 to +0.1 dB (prior: per-condition brain gain +0.07/+0.12 at full data, joint recon) |
| NMSE | follows PSNR |
| SSIM | difference < 0.002. Both arms at least 0.88 (healthy). Below 0.80 in any arm means a bug: debug before reporting |

With 1/4 of the data and 1/13 of the steps, the mask may move little. The prediction is a small effect or none.

loupe_uniform − static: with few steps, the mask may not get past its noise start. Prediction: −0.3 to +0.1 dB.
Below −0.3 dB means under-convergence, not "the shared mask is better".

## Success criteria (frozen at the proposal commit)

1. **Validity gate**: brain SSIM ≥ 0.80 in all 9 runs, and every run finishes. Crashed runs are rerun in full.
Criteria 2–6 compare loupe_shared with static (primary). The same thresholds 3–6 are applied to loupe_uniform − static
(secondary), and to loupe_shared − loupe_uniform (the effect of the start).

2. **Mask moved**: budget relocated in loupe_shared ≥ 10% in every seed. Otherwise the result is **inconclusive (mask
   did not move)**, whatever the PSNR.
3. **Loupe wins**: mean paired PSNR (loupe_shared − static) ≥ +0.10 dB, the same sign in 3 of 3 seeds, and larger than the
   static arm's seed spread (max − min).
4. **No effect** ("mask shape doesn't matter for brain"): |mean paired PSNR| < 0.05 dB with criterion 2 met.
5. **Static wins**: mean paired PSNR ≤ −0.10 dB with the same sign in 3 of 3 seeds.
6. Anything else (0.05–0.10 dB, or mixed signs) is **inconclusive**.

## Compute plan

- GPUs 0–1 only. The other two stay free.
- Smoke test first: 2 epochs per arm, to measure seconds per epoch (training on about 300 slices plus val on the
  quarter val set of all classes), check the identical step-0 mask, and check the frozen mask stays bit-identical.
- Estimate: about 25 s training + about 20 s val per epoch, so about 45 s per epoch.
  - 9 runs on 2 GPUs. To fit about 4 h (8 GPU-hours), that is about 50 min per run, roughly 65 epochs.
  - The epoch count is set from the smoke-test time and is equal for all arms. The smoke-test figures are in the
    launch commit.
  - Run order is seed-major (static, loupe_shared, loupe_uniform for s42, then s43, then s44). An overrun then still
    leaves complete seed triplets.
- Queue: one worker per GPU, wandb group `brain_static_vs_loupe`, runs named `bsl_<arm>_s<seed>`, checkpoints in
  `~/data/seqmri_checkpoints/brain_static_vs_loupe/`.
- Final eval of the 9 best checkpoints on full brain val: about 5 min each.

## Notes

- `git fetch origin` failed (publickey), so the `main...origin/main = 0 0` check was against a stale `origin/main`.
- The older pre-local-v1 study (`reports/LOUPE_SINGLE_VS_SHARED_ANATOMY_R8.md`) is a hypothesis only: different
  data, legacy metric, pre-flip. It is not used as a control.
