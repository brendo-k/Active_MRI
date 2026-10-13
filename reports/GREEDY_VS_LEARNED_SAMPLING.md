# Greedy vs. LOUPE vs. REINFORCE: closing the gap in k-space sampling search

## Motivation

`oracle_sampling_search.py` already showed that greedy line selection (searched
directly against a closed-form CG-SENSE oracle, see `activemri/samplers/oracle.py`)
beats equispaced/random/low-pass baselines by a wide margin. The open question
this investigation answers: **how close does a *learned* sampler (LOUPE-style
gradient relaxation, or a REINFORCE policy) get to that greedy reference, and
what closes the gap?**

Every comparison below routes through the same oracle reconstructor
(closed-form CG-SENSE in 1D, actual CG-SENSE solves in 2D) that greedy uses,
so results isolate the *mask-search algorithm* from confounds like an
undertrained or capacity-limited neural reconstructor.

## 1D (phase-encode lines): `scripts/loupe_vs_greedy_oracle.py`

Brain, R=4, 16-line budget, W=64, 4 search + 4 held-out slices.

| method | train NMSE | gap to greedy | held-out NMSE | gap to greedy |
|---|---|---|---|---|
| **greedy(oracle)** | 0.2096 | — | 0.2080 | — |
| loupe(oracle), from scratch | 0.3034 | +44.7% | 0.3148 | +51.4% |
| **loupe + greedy-init** | 0.2096 | **+0.0%** | 0.2080 | **+0.0%** |
| **reinforce(oracle)**, 300 episodes | 0.2043 | **−2.5%** | 0.2028 | **−2.5%** |
| random+ACS | 0.3562 | +69.9% | 0.3705 | +78.2% |
| equispaced+ACS | 0.3600 | +71.7% | 0.3805 | +83.0% |

**Key finding:** LOUPE-style gradient descent on a sigmoid-relaxed mask,
started from a uniform initialization, has a large real gap to greedy
(45–51%). But warm-starting the *identical* local search from the greedy mask
holds it with **zero drift** — the relaxed landscape doesn't repel from the
greedy optimum, it just can't find it from a naive start. The gap is an
**initialization/exploration problem, not a representational limit**.

REINFORCE (Plackett-Luce policy gradient, sampling whole budget-constrained
masks via Gumbel-top-k, trained directly against oracle reward) matches and
slightly *beats* greedy from scratch, given enough training (300 episodes;
150 was undertrained and scored worse than LOUPE).

Two real bugs were found and fixed while validating this:
- `_projected_gradient_search`'s backtracking line search declared false
  convergence after one step because genuine score improvements were the same
  order of magnitude as float32 rounding noise. Fixed with a relative
  tolerance and patience-based stopping (see `activemri/samplers/oracle.py`).
- A misplaced `clamp_min` in the Gumbel-noise formula produced NaN noise on
  every draw, so the first REINFORCE run's policy never explored (its score
  was frozen at one value for 150 episodes). Fixed in
  `plackett_luce_sample_and_logprob`.

## 2D (arbitrary k-space points): `scripts/reinforce2d_vs_greedy_loupe.py`

Brain, R=4, 576-point budget, 48×48, 1 search + 1 held-out slice (2D reward
needs an actual CG solve, so no closed-form population oracle like 1D's
`MeanLineOracle` — results are more prone to overfitting the single search
slice; the held-out column is the more meaningful comparison).

There is no closed-form 2D oracle, so `greedy_2d`, `loupe_style_design_2d`,
and `reinforce_2d` all route through actual (differentiable, for LOUPE)
CG-SENSE solves — the same reconstructor, no confound from a neural network.

| method | search NMSE | gap to greedy | held-out NMSE | gap to greedy |
|---|---|---|---|---|
| **greedy_2d** | 0.01405 | — | 0.01384 | — |
| loupe_2d(oracle), from scratch | 0.01630 | +16.0% | 0.01592 | +15.0% |
| **loupe_2d + greedy-init** | 0.01405 | **+0.0%** | 0.01384 | **+0.0%** |
| reinforce_2d, 25 updates | 0.01719 | +22.4% | 0.01398 | +1.0% |
| reinforce_2d, 150 updates | 0.01276 | −9.2% | 0.01174 | −15.2% |
| **reinforce_2d, 500 updates** | **0.01164** | **−28.6%** | **0.00975** | **−29.5%** |
| reinforce_2d, 150 upd., residual-init | 0.02089 | +48.7% | 0.02061 | +49.0% |
| random+ACS | 0.02190 | +55.9% | 0.01997 | +44.3% |
| low-pass | 0.11386 | +710% | 0.10259 | +641% |

**The 1D pattern replicates exactly:** `loupe_2d+greedy-init` matches greedy
to the last decimal again, confirming the initialization-not-representation
diagnosis isn't a 1D fluke.

**REINFORCE needed far more training in 2D than 1D**, and kept improving well
past the point where it already beat greedy: 25 updates was clearly
undertrained (+22% gap), 150 updates crossed over to beating greedy (−9%),
and 500 updates pushed the lead to −29% on held-out — **still not converged**
(the last 15-update chunk of the 500-update run showed its steepest recent
drop yet, −3.3%; the tail oscillates between about −3% and +1% per chunk
rather than flattening to 0%). This tracks the higher per-sample cost in 2D:
each reward needs a real CG solve (vs. a closed-form Cholesky in 1D), so
"enough training" is proportionally larger.

**A static single-round prior actively hurts.** Initializing `reinforce_2d`'s
logits from `kspace_residual_scores` against the preselect-only
reconstruction (the same quantity `greedy_2d`'s own first round ranks by,
standardized and temperature-scaled) made the final deterministic mask
*worse* than zero-init (+48.7% vs. +22.4% at matched training budget), even
though both variants' *stochastic* training rewards converged to nearly the
same value. The likely reason: the prior only reflects round-1 information
and is never refreshed, while `greedy_2d` recomputes its residual after every
committed batch — a stale round-dependent prior is the wrong tool for a
round-dependent decision. More training time, not a smarter initialization,
is the right lever for 2D REINFORCE.

## What actually closes the gap

1. **Warm-start LOUPE's mask parameter from a cheap greedy pass** (1D: exact
   closed-form; 2D: `greedy_2d`), rather than the uniform/constant init
   `train_single_slice_loupe.py` currently uses. Validated to hold the greedy
   optimum exactly in both 1D and 2D.
2. **Train the mask with REINFORCE instead of (or alongside) the
   straight-through sigmoid estimator.** Given enough samples, it matches or
   beats greedy in both regimes, and unlike LOUPE it needs no warm start to
   get there from scratch — it just needs more training budget, which is the
   correct thing to spend more compute on (not a smarter prior).
3. Do **not** try to save training time in 2D with a single-shot prior
   (leverage or residual score) for REINFORCE's mask logits — it measurably
   hurts here. If a round-dependent prior is wanted, it would need to be
   *recomputed* each round, which is most of the way to just running
   `greedy_2d` itself.

## Wired into the real pipeline

REINFORCE mask training (item 2 above) is now implemented for production
use, not just the oracle testbed:

- `activemri/samplers/loupe.py`: `raw_logits()` on `ProbMask` /
  `LineConstrainedProbMask`, exposing the pre-sigmoid free parameter as the
  Plackett-Luce score field.
- `activemri/samplers/sampler.py`: `Sampler(mask_train_mode='reinforce')` —
  exact-budget Gumbel-top-k sampling from the raw logits in place of
  rescale + straight-through binarization, returning the draw's
  log-probability. LOUPE-only (`loupe_2d`/`loupe_1d`); the conv samplers are
  input-adaptive and don't fit this framing without per-draw resampling of a
  partial reconstruction.
- `activemri/models/sequential_sampling.py`: accumulates each step's
  log-probability; `loss()` adds a REINFORCE term using **per-example NMSE as
  reward** and the **batch mean as baseline** — every batch item is one Monte
  Carlo draw of the (input-independent) mask, so no extra forward passes are
  needed. Raises clearly if `batch_size < 2` (no baseline possible with one
  sample) instead of silently training a degenerate policy.
- `activemri/config.py` / `training/trainer.py`: `mask_train_mode` and
  `reinforce_weight` plumbed through `TrainConfig` → `build_model`.
- `scripts/train_single_slice_loupe.py`: `--mask-train-mode reinforce
  --reinforce-batch N` replicates the one slice N times across the batch
  (LOUPE ignores its input anyway) to get N independent Monte Carlo draws per
  step.

Verified, not just written: the straight-through path was regression-checked
unchanged; the REINFORCE path was checked to produce a differentiable,
per-batch-item-distinct `logp`, and after `.backward()` the mask parameter's
gradient was confirmed nonzero on exactly the non-preselected positions
(87.5% = 28/32 lines, matching a 4-line preselect region exactly). A
150-iteration training run on real brain data with the actual CG-SENSE
reconstructor was stable (no NaNs, loss 0.096→0.030, PSNR 17.8→28.9).

## Caveats

- The oracle testbeds are small (1D: W=64, 8 slices total; 2D: 48×48, 2
  slices total) and CPU-only when these numbers were collected, chosen for
  fast turnaround rather than matching the paper's actual scale (128×128 or
  256×256, larger slice populations). The qualitative story (large
  from-scratch gap; warm-start closes it exactly; REINFORCE needs more
  samples but gets there) should generalize, but exact percentages should be
  re-checked at full resolution before being treated as final numbers.
- 2D greedy here used a single search slice (2D reward is too expensive to
  average over a population the way 1D's `MeanLineOracle` does), so it may
  itself be somewhat overfit to that slice — part of why REINFORCE's
  held-out numbers look even better relative to greedy than its search-slice
  numbers do.
- Neither the 150- nor the 500-update `reinforce_2d` run had converged when
  stopped; the true ceiling of this approach at this budget is still
  unknown.
- GPU is now available in this environment (`torch==2.5.1+cu121`, downgraded
  from `2.13.0` to match the installed driver, CUDA 12.2 max) but this broke
  `torchvision` (no Python-3.13-compatible build exists for cu121) and
  several other shared-env tools' declared version floors (`vllm` removed;
  `instanttensor`, `humming-kernels`, `compressed-tensors` now have unmet
  `torch>=` requirements). `activemri` itself was patched to not need
  torchvision at import time and is fully functional on GPU.
