# Does a radial-profile REINFORCE policy beat the per-pixel one at R=8? (2026-10-06)

## Question
`LOUPE_VS_REINFORCE_R8.md` found that the per-pixel REINFORCE policy has no detectable per-pixel gradient
signal, but a clear signal summed over k-space rings. Its proposed fix was a radially factored policy.
This run tests that fix: a policy with only 64 parameters, logit(ky, kx) = R(r) + A(r)·kx/r. R and A are
piecewise linear over 32 radial knots. A is a kx-odd asymmetry, so A < 0 favours kx < 0.
Did it learn a better mask, and why does its logged validation score come out about 2 dB *below* the per-pixel run?

Conventions:
- Radius r is measured from the fft-shifted centre, in units of the k-space half-extent. It is 1 at the edge
  along the axes and about 1.41 in the corners.
- kx is the phase-encode (column) axis.
- Budget is 8092 picks plus a 100-location preselected centre, out of 65536 (256×256).

## Runs compared
Both runs use `train_sequential.py` on brain+knee balanced data with a 12-cascade VarNet:
- batch 16, 100 epochs, lr 3e-4 stepped ×0.1 at epoch 90
- REINFORCE policy lr 0.02, 4 slices × 4 draws, entropy weight 0.005, normalized advantages
- variable-density init `reinforce_vd_power=2.5`

The overrides are identical except `sampler.radial_bins=32`.

| Run | Path | wandb | Policy | Status | Best val epoch | Brain PSNR / SSIM | Knee PSNR / SSIM |
|---|---|---|---|---|---|---|---|
| **radial32** | `checkpoints/reinforce2d_radial32_varnet_r8_brain_knee_vd2.5_100ep` | `onalzjt5` | 32-knot R(r) + A(r) | running, epoch 94/100 when written | 36 | 34.12 / 0.9169 | 32.66 / 0.9323 |
| per-pixel (control) | `checkpoints/reinforce2d_varnet_r8_brain_knee_vd2.5_100ep` | `oztalq2t` | 65k per-pixel logits | done | 100 | 36.40 / 0.9315 | 34.24 / 0.9524 |
| per-pixel, policy lr 0.002 | `checkpoints/reinforce2d_varnet_r8_brain_knee_vd2.5_plr0.002_100ep` | – | per-pixel, 10× lower policy lr | done | 99 | 34.35 / 0.9211 | 32.65 / 0.9349 |
| LOUPE (reference) | `checkpoints/loupe2d_varnet_r8_brain_knee` | `azd6r64k` | straight-through | done, 50 epochs | 50 | 36.08 / 0.9299 | 33.90 / 0.9493 |

The metrics are the trainer's background-masked validation metrics, n=344 slices per anatomy. They are
logged with the trainer's deployed mask, which for REINFORCE is the **deterministic top-k** of the policy
logits. Every SSIM is above 0.90, so no model is broken. The radial numbers are interim, but its best
epoch (36) was 24 epochs before the lr drop and is unlikely to change.

## Methods
1. **Training curves** (`scripts/summarize_train_logs.py ... --at-epoch 36 --window 61 90`). This parses
   `train_sequential.log` for best, last and matched-epoch validation metrics, plus brain-PSNR
   mean / sd / min / max over epochs 61–90. Output is in `reports/radial_reinforce_r8/train_log_summary.json`.
   Epochs are given 1-indexed, as logged. `static.json`/`eval.json` store the checkpoint's 0-indexed epoch
   (35, 99).
2. **Policy parameters and mask structure** (`scripts/analyze_radial_reinforce_r8.py static`):
   - Reads the learned R(r) and A(r) from radial32's `best_model.pt` (epoch 36) and compares them with the
     vd2.5 init, −2.5·log(r + 0.05).
   - For both runs, reports the per-ring density and budget share of (a) the top-k mask and (b) Plackett-Luce
     (PL) draws, the masks the reconstructor was trained on. The draw statistics use 64 Monte Carlo draws.
   - Ring bins match `LOUPE_VS_REINFORCE_R8.md`.
   - Outputs: `static.json` and `radial_policy.png`.
3. **Deployment-rule test** (`scripts/analyze_radial_reinforce_r8.py eval`, GPU 1). Each run's best checkpoint
   is scored on the full validation set with its own policy, deployed two ways: top-k, and PL draws (seeds 0
   and 1). Scoring uses the trainer's masked metric path (`_apply_signal_mask`, `_AnatomyStats`). Output is in
   `eval.json`. The top-k rows reproduce the logged best-epoch numbers to 0.01 dB.

## Results

### 1. The logged gap is mostly the deployment rule, not the policy
| Policy | Deployed as | Brain PSNR / SSIM | Knee PSNR / SSIM |
|---|---|---|---|
| radial32 (epoch 36) | top-k (what the trainer logs) | 34.12 / 0.9169 | 32.66 / 0.9323 |
| radial32 (epoch 36) | PL draws, seed 0 / 1 | **35.95 / 35.94** / 0.9291 | **33.77 / 33.77** / 0.9474 |
| per-pixel (epoch 100) | top-k | 36.40 / 0.9315 | 34.24 / 0.9524 |
| per-pixel (epoch 100) | PL draws, seed 0 / 1 | 36.36 / 36.36 / 0.9310 | 34.22 / 34.23 / 0.9522 |

- Scored on the draws it was trained on, radial32 gains **+1.82 dB on brain and +1.11 dB on knee** over its
  own top-k mask.
- For the per-pixel policy, top-k and draws differ by only 0.04 dB.
- The draw-to-draw noise floor is 0.003 dB.

### 2. Why top-k hurts only the radial policy
Per-ring density, as a fraction of the ring sampled:

| Ring (r) | n px | radial32 top-k | radial32 PL draws | per-pixel top-k | LOUPE (from earlier report) |
|---|---|---|---|---|---|
| 0.00–0.05 | 34 | 1.00 | 1.00 | 1.00 | 0.82 |
| 0.05–0.10 | 375 | 1.00 | 1.00 | 0.96 | 0.78 |
| 0.10–0.20 | 1552 | **1.00** | 0.77 | 0.58 | 0.61 |
| 0.20–0.30 | 2576 | **0.95** | 0.41 | 0.34 | 0.44 |
| 0.30–0.50 | 8212 | 0.24 | 0.27 | 0.21 | 0.28 |
| 0.50–0.70 | 12384 | **0.06** | 0.16 | 0.13 | 0.16 |
| 0.70–1.00 | 26196 | 0.04 | 0.05 | 0.08 | 0.05 |
| >1.00 (corners) | 14107 | 0.00 | 0.00 | 0.03 | 0.002 |

- **The radial policy's draws recover LOUPE's radial profile almost exactly.**
  - Draw density is within 0.03 of LOUPE's in every ring from r = 0.2 outward.
  - The policy also zeroed the corners. The per-pixel policy never managed this; it still leaves 3% density
    there, about 5% of its budget.
- **Top-k converts that profile into a solid low-pass disc.** A radial (plus kx-odd) logit field ranks pixels
  by radius alone, so the k highest logits form a filled disc with no incoherent sampling.
  - The disc fills 100% of the r < 0.2 rings and 95% of 0.2–0.3, and samples 6% of 0.5–0.7 (draws: 16%).
  - It puts 54% of the budget inside r < 0.3. The draws put 33% there.
  - The remainder sits in a few thin crescents at the knots where R(r) has local bumps (`radial_policy.png`).
- The per-pixel policy's logits carry pixel-level noise, so its top-k mask is still an incoherent
  variable-density pattern. That is why it shows no top-k/draw gap.
- The policy-lr 0.002 per-pixel run points the same way. Its logits barely leave the radial vd2.5 init, so its
  top-k mask is also a compact low-pass blob (`.../plr0.002_100ep/visualizations/val_epoch=99_step0.png`).
  It also lands near 34.3 dB (34.35 brain, 32.65 knee). This is consistent with the mechanism but was not
  separately tested by scoring that run with draws.

### 3. Learned parameters (radial32, epoch 36)
| | Value |
|---|---|
| R(r) max \|drift\| from init | 5.98 logits |
| Sign changes in the slope of R across 32 knots | 13 (the init is monotone, 0) |
| R at r = 0.73 → 0.78 → 0.96 | +1.70 → −0.03 → −6.00: a cliff that cuts off r > 0.78 |
| A(r) over r = 0.50–0.87 | negative throughout, reaching −4.2 at r = 0.87 (favours kx < 0) |
| Share of top-k budget in kx < 0 | 71% (per-pixel: 54%) |
| Uncommitted positions in draws, inclusion in (0.1, 0.9) | 37% (per-pixel: 18%) |

The profile is non-monotone, and these ripples become the rings visible in the draw-inclusion map. The
asymmetry is partial-Fourier-like over mid-to-high radii. The crescent is already present in the epoch-1
validation figure, so A moves within the first epoch.

### 4. Training trajectory: the radial run peaks early and decays
| Run | Brain PSNR at epoch 36 (top-k) | Brain, epochs 61–90: mean ± sd (min–max) | Last logged |
|---|---|---|---|
| radial32 | 34.12 (peak) | 33.29 ± 0.26 (32.64–33.58) | 32.87 (epoch 94) |
| per-pixel | 35.82 | 36.27 ± 0.06 (36.14–36.37) | 36.40 (epoch 100) |
| per-pixel, plr 0.002 | 33.91 | 34.17 ± 0.11 (33.93–34.38) | 34.34 (epoch 100) |

- radial32's top-k score falls by about 0.8 dB after epoch 36 and fluctuates 4× more from epoch to epoch than
  the control's.
- At matched training length (epoch 36), radial32's PL-draw score of 35.95 / 33.77 is **+0.13 dB on brain and
  −0.06 dB on knee** compared with the control's 35.82 / 33.83. The control figure is top-k, but its
  top-k/draw gap is only 0.04 dB.
- Only `best_model.pt` (epoch 36) is saved, so the draw score of radial32's later epochs is unknown.

### 5. How the validation mask differs from the training masks
Both masks come from the same policy logits, and both start from the same preselected 10×10 centre
(100 locations). Both then pick exactly 8092 more locations. The only difference is noise:

| | Training (`model.train()`) | Validation (`model.eval()`) |
|---|---|---|
| Rule | Plackett-Luce draw via Gumbel-top-k: add i.i.d. Gumbel noise to every logit, take the top 8092 (`plackett_luce_sample_and_logprob`, `activemri/samplers/oracle.py`) | Top 8092 logits, no noise (`torch.topk` in `Sampler._reinforce_forward`, `activemri/samplers/sampler.py`) |
| Randomness | Fresh mask every draw: 4 slices × 4 draws per batch | One fixed mask for every slice and epoch |
| Purpose | Sample the policy, so REINFORCE can credit each draw (reward = −NMSE) | The deterministic mask that would be deployed. It has no log-prob, so no REINFORCE term enters the validation loss |
| Checkpointing | – | `best_model.pt` = lowest validation loss **on this top-k mask** |

The Gumbel noise has a standard deviation of about 1.28 logits. That noise decides which positions win, and
removing it affects the two policies very differently:
- **Per-pixel policy.** Each pixel has its own logit, which picked up pixel-level noise (an Adam random walk)
  during training. Removing the Gumbel noise still leaves a scattered variable-density mask, so validation
  and training masks look alike (0.04 dB apart).
- **Radial policy.** Every pixel in a ring shares one logit (R(r) plus the A(r)·kx/r tilt), so within a ring
  *only* the Gumbel noise separates pixels. Without it the logits sort pixels purely by radius, and the top
  8092 is a filled low-pass disc. Wherever R(r) has a local bump, the disc gains a fully sampled thin crescent.
  The reconstructor trained only on scattered draws, and the disc is a coherent mask far outside that
  distribution. (Tables 1–2: 1.00 vs 0.77 density at r = 0.1–0.2, 0.06 vs 0.16 at r = 0.5–0.7.)

### 6. Why the logged radial score was ~2 dB below the per-pixel run
| Piece | Brain dB | Status |
|---|---|---|
| Validation scores the top-k disc, not the draws the model trained on (34.12 → 35.95) | ~1.8 | Measured (Table 1) |
| Chosen checkpoint has fewer epochs (36 vs the control's 100) | ~0.4 | Likely: the control gained 0.58 dB from epoch 36 to 100 (35.82 → 36.40) |
| Radial prior can't vary density within a ring | ≤ ~0.2 | Bounded, not measured directly (see below) |

1. **Top-k deployment (the bulk).** Section 5 explains the mechanism.
   - The disc samples 100% of k-space to r ≈ 0.2 and 95% of 0.2–0.3, then about 6% at r = 0.5–0.7.
   - Undersampling then shows up as blur, not as the noise-like aliasing the VarNet has learned to remove.
2. **Early stopping on the wrong criterion.** `best_model.pt` was selected on the top-k score, which peaked at
   epoch 36. That is too early for the reconstructor to catch up with the 100-epoch control. At epoch 36 the
   two methods tie (+0.13 dB brain, −0.06 dB knee; Table 4).
3. **Why the logged score fell after epoch 36** (hypothesis, untested).
   - Policy lr 0.02 was tuned for 65k noisy per-pixel logits. Here each of the 64 parameters receives a
     strong pooled gradient, and Adam moves every one of them a roughly lr-sized step on every update.
   - The learned R(r) is jagged: 13 slope reversals, up to 6 logits of drift from the init.
   - Top-k is very sensitive to those bumps, since each becomes a fully sampled ring. So the validation mask
     kept changing shape, consistent with the 4× larger epoch-to-epoch swings (sd 0.26 vs 0.06 dB).
4. **The restrictive prior itself (small).** `LOUPE_VS_REINFORCE_R8.md` §3 replaced masks with ring-constant
   ones of equal per-ring density. That cost LOUPE 0.18 dB and per-pixel REINFORCE about 0.20 dB.
   - The radial policy's draws score 35.95 dB, almost exactly the 35.89 dB of ring-constant LOUPE.
   - So the radial prior reached about the ceiling a ring-constant mask allows. That ceiling sits about
     0.2 dB below unconstrained masks.
   - Those earlier numbers are for a co-adapted fixed pattern, so they bound, rather than measure, the cost
     of the prior here.

**Implication.** The case against the radial prior is the ~0.2 dB ceiling and an unusable deterministic mask,
not the 2 dB. A rerun with policy lr ≈ 0.002, scored and checkpointed on draws, would be expected to land
within about 0.2–0.3 dB of the per-pixel run (untested). The radial policy was shelved on 2026-10-07 (code in
a git stash), and the analysis script `scripts/analyze_radial_reinforce_r8.py` was deleted. Its outputs remain
in `reports/radial_reinforce_r8/`.

### 7. Validation now samples from the policy: re-validated per-pixel runs (2026-10-07)
**Code change.** REINFORCE validation now draws its mask from the policy by default (`sampler.reinforce_eval:
sample`), using the same Gumbel-top-k (Plackett-Luce) sampling as training. Top-k stays available with
`reinforce_eval: topk`.
- The draws use a dedicated `torch.Generator` that the trainer reseeds at the start of every validation epoch
  (`REINFORCE_VAL_MASK_SEED`). Validation masks therefore repeat exactly from epoch to epoch, and checkpoint
  selection is not driven by mask noise.
- Training draws still use the global RNG.
- No log-prob is returned in eval, so the REINFORCE term stays out of the validation loss.
- Changed files: `samplers/oracle.py` (optional `generator`), `samplers/sampler.py`,
  `models/sequential_sampling.py`, `training/trainer.py`, `config.py`, `configs/config.yaml`,
  `scripts/train_sequential.py`.
- `compare_loupe_reinforce_r8.py` now requests top-k explicitly for its "topk" condition.

**Re-validation.** `scripts/revalidate_reinforce.py` builds a `Sequential_Trainer` from each run's hydra config
(wandb disabled), loads `best_model.pt` and calls the trainer's own `val_epoch()` once per mode. Outputs:
`reports/reinforce_sampled_val/*.json`.

| Run (best epoch) | Top-k brain / knee PSNR | **Sampled** brain / knee PSNR | Δ (sampled − top-k) | Sampled SSIM brain / knee |
|---|---|---|---|---|
| v1, default policy lr (50) | 33.50 / 32.74 | 33.24 / 32.46 | −0.27 / −0.28 | 0.9070 / 0.9351 |
| v2, policy lr 0.02 (48) | 35.69 / 33.84 | 35.59 / 33.80 | −0.09 / −0.04 | 0.9253 / 0.9478 |
| v2, 100 epochs (99) | 36.35 / 34.24 | 36.28 / 34.22 | −0.07 / −0.02 | 0.9305 / 0.9522 |
| vd2.5, 100 epochs (100) | 36.40 / 34.24 | 36.36 / 34.22 | −0.04 / −0.02 | 0.9310 / 0.9522 |
| vd2.5, group size 8 (99) | 36.30 / 34.20 | 36.25 / 34.19 | −0.05 / −0.01 | 0.9302 / 0.9514 |
| **vd2.5, policy lr 0.002 (99)** | 34.35 / 32.65 | **36.38 / 34.14** | **+2.03 / +1.48** | 0.9315 / 0.9507 |

- **The top-k disc effect is confirmed on a second, independent run.** The policy-lr 0.002 run barely moved off
  the radial VD init, so its top-k mask was a compact low-pass blob. Scored on draws, it goes from 34.35 to
  36.38 dB brain, level with the best per-pixel run (36.36–36.40). Its "failure" was entirely the evaluation.
- For policies with pixel-level noise in their logits, sampling costs 0.01–0.09 dB against top-k, so the switch
  barely changes their numbers.
- **Observation:** a policy that stayed essentially at the vd2.5 prior (plr 0.002) scores the same on draws as
  one trained at 10× the policy lr (36.38 vs 36.36 brain; knee 34.14 vs 34.22). On this evidence, REINFORCE
  adds little over a well-chosen variable-density prior at R=8. *Hypothesis, untested*, because each is a
  single seed and the reconstructors differ.
- The top-k column reproduces the logged best-epoch numbers to 0.01 dB for every run except v1 (logged 33.21
  brain at epoch 50, top-k now 33.50). v1's logged validation loss is negative, so it was trained under older
  validation code that included the REINFORCE term and apparently sampled. Its sampled score (33.24) matches
  its log.
- All checkpoints were still *selected* on top-k (or older validation code), so the sampled numbers may
  understate what draw-based checkpoint selection would give, most likely for plr 0.002.
- radial32 could not be re-validated, because its code is in the git stash. Section 1's draw-scored numbers
  (35.95 / 33.77) used the same sampling with a different seed path.

### 8. Learned patterns of the per-pixel runs, and a left/right (kx) asymmetry (2026-10-07)
**Patterns** (`scripts/plot_reinforce_patterns.py` → `reports/reinforce_patterns_r8/patterns.png`, `summary.json`).
For each run the figure shows:
- the policy's inclusion probability: 64 Plackett-Luce draws for REINFORCE, the rescaled Bernoulli probability
  for LOUPE
- one sampled mask
- the top-k mask
- the inclusion minus its own ring average (σ = 3 px smoothing)
- a radial-profile overlay for all runs

| Run | Budget at r > 1 (corners) | ky/kx extent | Max \|smoothed residual\| |
|---|---|---|---|
| LOUPE, 50 ep | 0.4% | 0.90 | 0.18 |
| REINFORCE uniform init, 100 ep | 11.6% | 0.98 | 0.08 |
| REINFORCE vd2.5, 100 ep | 6.5% | 0.98 | 0.09 |
| REINFORCE vd2.5, policy lr 0.002 | 4.4% | 0.99 | 0.04 |

- **LOUPE has the most structure.**
  - Density drops to zero beyond r ≈ 0.9, so the corners are unused.
  - The support is wider along kx than ky.
  - It has a strong lean to the kx < 0 side and a thin spike along the kx axis at kx > 0.
  - It spends more of its budget in the mid band (r ≈ 0.3–0.7).
- **The lr-0.02 REINFORCE runs are diffuse variable-density patterns.**
  - Their pixel-level texture is mostly noise.
  - They put 6–12% of the budget in the corners and keep a heavier tail past r ≈ 0.8.
- **The lr-0.002 run stays essentially at the vd2.5 prior.** It has a dense core to r ≈ 0.25, almost no
  residual structure, and a top-k mask that is a solid disc (section 7).

**kx asymmetry test** (`scripts/kx_asymmetry_test.py` → `reports/reinforce_patterns_r8/kx_asymmetry.json`).
- Statistic: the share of inclusion mass in the kx < 0 half, overall and per ring (r ≥ 0.05).
- Null: 500 shuffles of the run's own inclusion values within 64 radial rings. This keeps the radial profile and
  destroys any direction preference.
- z = (observed − null mean) / null sd.
- The ky < 0 (top/bottom) share is the control axis.

| Run | kx < 0, overall | r 0.3–0.5 | r 0.5–0.7 | r 0.7–1.0 | ky < 0 control, overall |
|---|---|---|---|---|---|
| LOUPE | 61.7% (z 50) | 53.2% (z 7.1) | 70.5% (z 38) | **84.4%** (z 54) | 48.2% (z −5.5) |
| REINFORCE uniform init | 52.8% (z 7.9) | 51.6% (z 2.1) | 53.9% (z 4.8) | 54.5% (z 7.0) | 50.4% (z 1.5) |
| REINFORCE vd2.5 | 53.5% (z 10.4) | 52.4% (z 3.2) | 55.2% (z 6.3) | 56.1% (z 8.1) | 50.0% (z 0.8) |
| REINFORCE vd2.5, lr 0.002 | 50.1% (z 6.6) | 50.4% (z 4.2) | 50.4% (z 3.3) | 50.8% (z 5.3) | 49.6% (z −0.4) |

- **Every run leans toward kx < 0, and REINFORCE learns it too.** The two lr-0.02 REINFORCE runs reach z 8–10
  overall. Their control axis shows nothing (|z| ≤ 1.5).
- **The lean grows with radius.** Every run is symmetric inside r < 0.3 (|z| ≤ 2). The lean appears from
  r ≈ 0.3 and is strongest at r 0.7–1.0.
- **It is consistent across runs.** Four runs with different methods, inits and policy lrs all lean the same
  way, including the lr-0.002 run that barely left its prior (+0.8 percentage points at r 0.7–1.0). This points
  to a signal in the reward (data plus reconstructor), not chance. The stashed radial policy also put 71% of its
  top-k budget at kx < 0 (section 3).
- **REINFORCE learns the lean at about 1/6 of LOUPE's strength.** At r 0.7–1.0 it moves 4–6 percentage points
  toward kx < 0; LOUPE moves 34. This matches the gradient-SNR finding in `LOUPE_VS_REINFORCE_R8.md`: no
  per-pixel signal is detectable, but summed over a large region (a ring there, a half-plane here) the signal
  is clear, and REINFORCE follows it slowly.
- **Caveat:** the null treats pixels within a ring as exchangeable. That holds for REINFORCE's
  pixel-independent logit noise. LOUPE's spatially smooth structure makes its z-scores overstated, although
  its asymmetry is large either way. Inclusion maps come from 64 Monte Carlo draws, single seed per run.
- **Hypothesis, untested:** the data rewards high-|kx| samples on one side because of a phase or
  partial-Fourier-like property of this k-space (conjugate symmetry makes one half partly redundant). Testing
  it means mirroring the k-space or the learned mask in kx and re-scoring, or retraining on
  phase-scrambled data.

## Conclusions
- **The radial parameterization did what it was meant to.** It pooled the ring-level REINFORCE signal and
  learned LOUPE's radial density profile from r = 0.2 outward, corners included. The per-pixel policy failed to
  do this in 100 epochs. (Table 2.)
- **The headline "radial is 2.3 dB worse" is an artefact of evaluating with top-k.** For a radially symmetric
  policy, top-k is a filled low-pass disc that the reconstructor never trained on. Scored on its training
  draws, radial32 loses only 0.42 dB (brain) and 0.46 dB (knee) to the 100-epoch control, at 36 versus 100
  epochs. (Table 1.)
- **At equal training length the two policies tie** (+0.13 / −0.06 dB). There is no evidence yet that the
  radial policy produces a *better* (mask, reconstructor) pair. (Table 4.)
- **Hypothesis, untested:** the post-epoch-36 decline and the 4× larger epoch-to-epoch swings come from the
  policy lr. lr 0.02 was tuned for 65k noisy per-pixel logits. With 64 parameters that each get a strong
  pooled signal, Adam's lr-sized steps may overshoot, producing the zig-zag R(r) and the drifting top-k disc.
  Neither the draw-scored trajectory nor a lower-lr radial run has been measured.
- **The remaining 0.4 dB gap to the 100-epoch control is consistent with 64 fewer epochs.** The control
  itself gained +0.58 dB on brain between epoch 36 and epoch 100 (35.82 → 36.40). PL draws are already
  random within each ring, so the radial policy is not short of incoherence. *Hypothesis, untested:* a
  learned within-ring pattern, worth about 0.2 dB to co-adapted masks in the earlier report, is a
  secondary candidate.

## Caveats
- **Single seed per run.** No seed-to-seed range is available for any row.
- **The radial run is still training** (epoch 94/100 at the time of writing). The radial checkpoint is
  epoch 36, the control's is epoch 100, so Table 1 is not length-matched. Table 4 gives the matched
  comparison, but only from the control's logged top-k numbers.
- **Checkpoint selection was done with top-k.** `best_model.pt` is the lowest top-k val loss, which is the
  wrong criterion for a draw-deployed radial policy. A later epoch could have a better draw score.
- **PL draws are random masks.** Deploying them means a different mask per scan, or one fixed draw. A single
  fixed draw was not scored. The draws from a policy rather than a fixed mask may be exactly what keeps the
  reconstructor robust.
- The plr0.002 explanation (section 2) rests on its validation figure and score, not on a draw-scored eval.
- `reports/radial_r8.log` / `reports/loupe_vs_reinforce_r8/radial.json` are an earlier, different
  experiment: ring-constant masks for the old LOUPE and REINFORCE checkpoints. They are not this run.

## Figures
- `reports/radial_reinforce_r8/radial_policy.png`: learned R(r) against the init, A(r), radial32's top-k mask,
  radial32's PL inclusion map, and the per-pixel control's top-k mask.
- `checkpoints/reinforce2d_radial32_varnet_r8_brain_knee_vd2.5_100ep/visualizations/val_epoch={1,36,92}_step0.png`:
  the logged validation masks (disc plus crescents) at init, at the best epoch and late in training.
- `checkpoints/reinforce2d_varnet_r8_brain_knee_vd2.5_plr0.002_100ep/visualizations/val_epoch=99_step0.png`:
  the low-lr per-pixel run's similarly compact top-k mask.
- `reports/reinforce_patterns_r8/patterns.png`: learned patterns of LOUPE and three per-pixel REINFORCE runs.
  Per run: inclusion probability, one sampled mask, top-k mask and smoothed residual vs ring mean, plus a
  radial-profile overlay (section 8).

## Next steps
1. ~~**Evaluate (and select checkpoints) with draws for smooth policies.**~~ Done 2026-10-07: validation now
   samples by default (section 7). The radial run itself was not retrained.
2. **Lower the radial policy lr** (e.g. 0.002 and 0.005) to test whether the late decline and the zig-zag
   R(r) come from step size.
3. **Hybrid policy:** per-ring logits plus a small per-pixel residual, so top-k keeps incoherence and the
   ring-level credit sharing is kept.
4. **Rescore radial32's final epoch** with draws once the run finishes. Save `last_model.pt`, or re-run with
   per-epoch checkpoints, to get the draw-scored trajectory.
5. **Test the kx asymmetry** (section 8): mirror the learned masks in kx (or the k-space) and re-score with the
   trainer's validation. A drop would confirm that the preferred half matters for these reconstructors.
