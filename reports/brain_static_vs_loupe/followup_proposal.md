# Follow-up: does the learned brain mask beat S once a fresh VarNet trains on it frozen? (2026-11-10)

Frozen before launch. The criteria of `proposal.md` are unchanged; this file adds runs and criteria for a follow-up on
the same branch, data, budget and scoring.

## Why
The mask-swap eval (`scripts/eval/brain_static_vs_loupe_swap.py`, full brain val, 605 slices) found two things.
- Every one of the 9 VarNets scored higher with loupe_shared's learned brain mask M than with S. The 6 VarNets that never
  trained on M gained +0.05 to +0.13 dB.
- On M, the static VarNet beat loupe_shared's own VarNet by +0.09, +0.27 and +0.25 dB (s42, s43, s44). On the 301 training
  slices, loupe_shared's VarNet instead won by +0.27 and +0.28 in s43 and s44, so it generalised worse.

So loupe_shared may tie static because a better mask is offset by a worse VarNet. Two candidate causes of the worse VarNet:
- (a) training against a moving mask (joint training);
- (b) the mask itself: M is less random than S (mean per-pixel entropy 0.18-0.20 vs 0.22 bits), so it gives less
  mask augmentation.

Freezing M and training a fresh VarNet on it separates the two.

## Runs
New runs go in the same queue, `~/data/seqmri_checkpoints/brain_static_vs_loupe/`, with the same job line as static
except as listed.

| Run | Brain slot of `fixed_mask_checkpoints` | Purpose |
|---|---|---|
| `frozen_learned_s{42,43,44}` | `loupe_shared_s{seed}/last_model.pt` (its brain row); knee slots S as in static | the experiment |
| `static_repeat_s42` | S (static_s42's job line again) | repeat-run noise at a fixed seed; checks the loader change leaves static unchanged |

- A seed gives the same VarNet initialisation, data order and parameter groups as static_s{seed}, so only the brain mask
  differs.
- Controls are reused: static_s{42,43,44} and loupe_shared_s{42,43,44}, trained at `f218677`. Their training code
  differs from this follow-up's only in `_load_fixed_masks`, which now also reads a per-anatomy checkpoint's row.

## Scoring
- Same as the swap eval: `last_model.pt`, the full brain val set (40 volumes, 605 slices), common per-slice Bernoulli
  draws (seeds 1234, 4321) and the trainer's apodized metric mask.
- Each new VarNet is scored with its own mask and with the same seed's other masks.
- The same is done on the 301 training slices.

## Criteria (PSNR, paired per slice, per seed)
**Primary: frozen_learned − static**, each with its own mask.
- Learned mask wins: mean ≥ +0.10 dB, positive in 3/3 seeds, and paired t over seeds ≥ 2.92 (one-sided 95%, 2 df).
- No effect: |mean| < 0.05 dB.
- S wins: mean ≤ −0.10 dB and negative in 3/3 seeds.
- Otherwise inconclusive.
- `proposal.md` also required the mean to exceed the static seed spread (0.44 dB). Here the paired t replaces that
  clause: this contrast is paired at a matched initialisation and data order, whereas the spread compares different
  seeds.

**Secondary: frozen_learned − loupe_shared**, each with its own mask (the same end mask M). Same thresholds. This
measures the cost of training against a moving mask.

**Mechanism.** Use s43 and s44 only, where loupe_shared's train − val gap exceeded static's by 0.30-0.46 dB.
G = (own-mask PSNR on train − on val) of frozen_learned, minus the same for static.
- G < 0.15 dB in both seeds: (a), the moving mask.
- G > 0.30 dB in both seeds: (b), the mask.
- Otherwise unresolved.

**Reported without a threshold:**
- Mask effect of the (frozen_learned, static) pair: ½[(frozen_learned VarNet on M − on S) + (static VarNet on M − on
  S)]. Both VarNets score both masks, so seed noise in the VarNets cancels.
- Repeat-run noise: static_repeat_s42 − static_s42.

**Validity:**
- All 4 runs finish.
- Brain SSIM ≥ 0.80.
- Each frozen brain slot is bit-identical to loupe_shared_s{seed}'s brain row, and static_repeat_s42's to S.

## Expected outcomes
- **Under (a):**
  - Primary +0.05 to +0.13 dB if a fresh VarNet gains what the six non-own VarNets gained in the swap. Up to about
    +0.25 dB if it also adapts to M.
  - Secondary positive, +0.1 to +0.3 dB.
  - G near 0.
- **Under (b):** frozen_learned's VarNet repeats loupe_shared's generalisation gap (G > 0.3 dB), the primary is near
  0 or negative, and the secondary is near 0.
- **SSIM and NMSE** follow PSNR. Paired SSIM differences stay within ±0.002, as in the main experiment.

## Compute
4 runs of about 36 min on GPUs 0-1 (two rounds, about 75 min), then about 20 min of swap eval.
