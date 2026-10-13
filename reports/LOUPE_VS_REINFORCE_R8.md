# LOUPE vs REINFORCE at R=8 (brain + knee, 12-cascade VarNet)

Runs: `checkpoints/loupe2d_varnet_r8_brain_knee` (wandb `azd6r64k`) vs
`checkpoints/reinforce2d_varnet_r8_brain_knee_v2` (wandb `mj4759it`, policy lr 0.02,
4 slices x 4 draws, entropy 0.005, normalized advantages). Same data, batch 16, 50 epochs,
lr 3e-4 stepped x0.1 at epoch 40; only the mask-training mode differs.
Analysis: `scripts/compare_loupe_reinforce_r8.py` (`static`, `crossval`, `radial`); raw output
in `reports/loupe_vs_reinforce_r8/`. The cross-eval reproduces both runs' logged best-epoch
numbers to 0.01 dB using the trainer's masked metric path.

## Headline

| val (n=344 each) | brain PSNR / SSIM | knee PSNR / SSIM |
|---|---|---|
| LOUPE | **36.07** / 0.9299 | **33.90** / 0.9493 |
| REINFORCE v2 | 35.69 / 0.9263 | 33.84 / 0.9484 |
| REINFORCE v1 (policy lr default — discounted, did not learn) | 33.27 / 0.9077 | 32.48 / 0.9358 |

Paired per slice: LOUPE is better on **91% of brain slices** (mean +0.46 dB, and the worst
7% of slices do not drive it). It is better on 58% of knee slices (mean +0.11 dB).
The brain gap is real. Knee is nearly a tie.
LOUPE's draw-to-draw noise floor (Bernoulli reseed) is 0.002 dB.

### Update: 100-epoch REINFORCE (`..._v2_100ep`, policy lr held at 0.02 until epoch 90)

| | brain PSNR / SSIM | knee PSNR / SSIM |
|---|---|---|
| REINFORCE, 100 epochs (best epoch 98) | **36.36** / 0.9311 | **34.24** / 0.9525 |
| LOUPE, 50 epochs (above) | 36.07 / 0.9299 | 33.90 / 0.9493 |

REINFORCE with twice the training beats 50-epoch LOUPE, by +0.29 dB on brain and +0.34 dB
on knee. This is **not a like-for-like comparison**, because LOUPE never got 100 epochs.

The mask barely changed in the extra training:
- 38.7% of the budget is still at r > 0.7 (it was 41%), and 9.8% is still in the corners.
- The innermost ring rose from 0.76 to 0.85.
- The policy is somewhat sharper: 21% of positions uncommitted, and two draws overlap with
  Dice 0.61.

So most of the gain came from longer training (a better reconstructor and/or a better
pair), not from pulling budget out of the periphery. That is consistent with explanation 2
below, the reconstructor being data- and step-starved. It also means sections 1–3 describe
the 50-epoch pair only.

### Why the REINFORCE pattern barely moves: policy-gradient SNR

`scripts/reinforce_gradient_snr.py` replays the training gradient of the 100-epoch policy:
PL draws, reward −NMSE, advantages normalized in groups of 4, 16 draws per step. It uses 24
val slices × 64 draws (1536 total) and compares against a within-group advantage-permutation
null.

| ring | per-pixel \|z\|>3 (real / null) | per-pixel SNR per step (est.) | steps for drift > diffusion, per pixel | ring-sum z (null rms) | ring direction |
|---|---|---|---|---|---|
| 0.05–0.1 | 0.5% / 0.2% | 0.026 | ~1,500 | **+2.3** (0.66) | raise |
| 0.1–0.2 | 0.0% / 0.15% | below detection | – | **+3.6** (1.04) | raise |
| 0.2–0.3 | 0.27% / 0.20% | below detection | – | +0.8 (0.80) | – |
| 0.3–0.5 | 0.12% / 0.17% | 0.009 | ~11,000 | −0.6 (1.17) | – |
| 0.5–0.7 | 0.15% / 0.15% | 0.007 | ~21,000 | −0.3 (0.83) | – |
| 0.7–1.0 | 0.16% / 0.14% | below detection | – | −1.7 (1.13) | lower |
| >1.0 | 0.13% / 0.14% | 0.006 | ~32,000 | **−1.4** (0.63) | lower |

- **No individual position has a detectable signal.** The share of positions with
  \|z\| > 3 matches the null in every ring. The per-step SNR per logit is about 0.01, so
  Adam's per-coordinate steps (≈ lr each, whatever the gradient size) are mostly noise.
  Drift only beats diffusion after roughly 10⁴–3·10⁴ steps. The 100-epoch run had ~15k
  steps at full lr.
- **Summed over a ring, the signal is clear and points the same way as LOUPE's profile:**
  raise 0.05–0.2 and lower the periphery and corners. At the ring level drift wins in only
  ~10–50 steps.
- So the gradient knows *which radial profile* is better, but the per-pixel parameterization
  divides that knowledge across 14k–26k independent logits, each of which only sees noise.
  That is why the pattern barely changed in 50 extra epochs.
- Fixes: put the radial profile in the init (variable-density init; `reinforce_vd_power≈2.5`
  matches LOUPE's profile under PL top-k), or parameterize the policy as per-ring logits plus
  a per-pixel residual.

## 1. Cross-evaluation: reconstructor × mask (PSNR brain / knee)

| reconstructor \ mask | LOUPE Bernoulli (deployed) | LOUPE top-k | REINFORCE top-k (deployed) | REINFORCE PL draws (trained on) | uniform random |
|---|---|---|---|---|---|
| LOUPE recon | **36.07 / 33.90** | 35.99 / 33.78 | 35.44 / 33.72 | – | 28.32 / 29.65 |
| REINFORCE recon | 35.46 / 33.59 | 35.23 / 33.33 | **35.69 / 33.84** | 35.59 / 33.80 | 29.82 / 30.49 |

- **Each reconstructor does best on its own mask.** Handing REINFORCE's reconstructor
  LOUPE's mask makes it *worse* (−0.23 brain, −0.25 knee). So there is no evidence that the
  REINFORCE policy is suboptimal *for the reconstructor it co-trained with*. The gap belongs
  to the (mask, recon) pair, not to the mask alone.
- REINFORCE's reconstructor is a generalist: it is +1.5 dB better than LOUPE's on random
  masks, and it loses less when its mask is swapped. That is the expected result of training
  on a broad mask distribution.
- The train/eval mismatch in REINFORCE (sampled masks in training, top-k at eval) costs
  nothing. Top-k beats the training draws by +0.10 dB on brain.

## 2. Mask structure (budget 8092 + 100 preselect, of 65536)

Densities below are for the **deployed** masks: LOUPE's rescaled Bernoulli probability and
REINFORCE's top-k. The seed-42 init was recovered by replaying `train_sequential.main` and is
identical for both runs. The uniform init density is 0.12.

| ring (r / k_max) | LOUPE density | REINFORCE density | LOUPE logit corr. w/ init | REINFORCE corr. w/ init | REINFORCE abs drift |
|---|---|---|---|---|---|
| 0.00–0.05 (34 px outside preselect) | 0.82 | 0.77 | 0.94 | −0.15 | 3.84 |
| 0.05–0.10 | 0.78 | 0.64 | 0.84 | 0.21 | 2.90 |
| 0.10–0.20 | 0.61 | 0.44 | 0.75 | 0.14 | 2.06 |
| 0.20–0.30 | 0.44 | 0.30 | 0.77 | 0.14 | 1.63 |
| 0.30–0.50 | 0.28 | 0.18 | 0.78 | 0.08 | 1.36 |
| 0.50–0.70 | 0.16 | 0.13 | 0.76 | 0.10 | 1.24 |
| 0.70–1.00 | 0.05 | 0.09 | 0.53 | 0.11 | 1.19 |
| >1.00 (corners) | 0.002 | 0.07 | 0.92 | 0.11 | 1.17 |

- **The main difference is radial allocation.** REINFORCE puts 41% of its budget at r > 0.7,
  including 12.6% in the corners. LOUPE puts 18% there and 0.4% in the corners. In exchange,
  REINFORCE undersamples the 0.05–0.3 band (0.64 / 0.44 / 0.30 vs 0.78 / 0.61 / 0.44).
  Neither mask fully samples the lowest frequencies outside the preselect.
- **Both deployed distributions are similarly uncommitted.** No LOUPE position exceeds
  probability 0.87, and 26.1% sit in (0.1, 0.9). REINFORCE's policy is 26.5% uncommitted;
  two of its draws overlap with Dice 0.47. Its first-pick entropy fell steadily from 11.06
  to 8.97 and was still falling linearly when the shared StepLR cut the policy lr 10x at
  epoch 40, which froze it. So "commitment" does not separate the two methods.
- **REINFORCE's peripheral logits look like diffusion.** Their drift (~1.2) matches the
  size of an Adam random walk at this lr (≈ 0.02·√6840 ≈ 1.65). They are uncorrelated with
  init. The resulting density (0.07–0.09) has barely moved off the uniform 0.12.
- **LOUPE stays anchored to its init.** Its logits correlate 0.75–0.94 with the random init
  in every ring. In the 0.3–0.5 ring it kept 85% of the init's top choices. LOUPE learned a
  radial profile by shifting whole rings; its within-ring pattern is the original random init.

## 3. Does the fine structure matter? (ring-constant density, Bernoulli draws)

Each mask is replaced by one with the same per-ring density but a uniform pattern within
each ring. The reconstructor is unchanged, and each result is compared with that model's own
stochastic mask.

| | brain | knee |
|---|---|---|
| LOUPE recon: own Bernoulli → radial-only | 36.07 → 35.89 (−0.18) | 33.90 → 33.82 (−0.08) |
| REINFORCE recon: own PL draws → radial-only (inclusion density) | 35.59 → 35.39 (−0.20) | 33.80 → 33.69 (−0.11) |
| REINFORCE recon: own top-k → radial-only (top-k density) | 35.69 → 35.51 (−0.18) | 33.84 → 33.74 (−0.10) |
| **Gap, radial-only LOUPE vs radial-only REINFORCE (top-k)** | **0.38** | 0.08 |

- Both reconstructors get ~0.2 dB from their mask's within-ring pattern. For LOUPE that
  pattern is mostly the random init, so the 0.18 dB is co-adaptation value: the
  reconstructor learned to exploit whichever pattern it was given. It says nothing about
  whether a *different* fine pattern would be better.
- **The LOUPE–REINFORCE gap is unchanged (0.38 dB brain) when both masks lose their fine
  structure.** So the gap is in the radial profile plus the reconstructor, not in the
  fine pattern either method found.

## Why REINFORCE loses here

What the evidence establishes:
- The gap is a joint (mask, recon) effect. REINFORCE's reconstructor does not prefer
  LOUPE's mask.
- The gap survives removing fine structure. It lives in the radial allocation (REINFORCE
  leaks ~2x the budget to r > 0.7 and the corners) and/or in reconstructor quality.

There are two competing explanations. Neither has been isolated causally yet.

1. **The policy is misallocated or under-trained.** One scalar reward per draw has to credit
   8092 picks over 65k positions, with a 4-draw baseline.
   - Normalized advantages + Adam convert noise into lr-sized steps. That is consistent with
     the random-walk drift in the periphery, where density never fell from its uniform init
     toward LOUPE's ~0.
   - The oracle testbed where REINFORCE beat greedy had 576 picks over 2304 positions. It
     needed 500+ updates and was still improving. This problem is ~28x larger per draw.
   - The policy was still learning when its lr was cut 10x at epoch 40.
2. **The reconstructor saw ~4x fewer distinct slices.** `_replicate_for_reinforce` keeps 4
   of each batch's 16 slices, each copied 4 times. So per epoch the REINFORCE recon trains on
   ~25% of the unique data LOUPE's does, in the same number of steps. This alone could
   explain a weaker recon that then co-adapts to whatever mask it gets.

Contributing factors:
- **Non-stationary reward.** The reward is NMSE through a reconstructor that is still
  training.
- **Training on a broad mask distribution.** The recon becomes a generalist (+1.5 dB on
  random masks) rather than a specialist.
- **Inert or small effects.**
  - Global grad clipping (norm 1.0) fires on *every* step, because normalized advantages pin
    the policy grad norm at ~19.5. That scales all recon gradients by ~1/20, but Adam is
    nearly invariant to a constant scale.
  - The entropy bonus is negligible (0.005 × 0.08 vs ~19.5).

**On local optima:**
- LOUPE is anchored to its init, but its init-locked fine structure is not what makes it win.
  Whether a better fine structure exists is untested. The oracle testbed suggests a
  greedy warm start would find one.
- REINFORCE has not converged into any optimum. Its policy is still diffuse and was still
  moving.
- **"Not exploring a large enough mask space" does not fit REINFORCE.** It explores plenty.
  The problem is that its gradient signal is too weak to pull budget out of the periphery
  within 40 epochs at full lr.

## Experiments that would separate the explanations

- **The 100-epoch run** (`checkpoints/reinforce2d_varnet_r8_brain_knee_v2_100ep`, in progress)
  keeps policy lr 0.02 until epoch 90. Check whether the r > 0.7 budget share falls from 41%
  and whether brain PSNR closes on 36.07.
- **Data-matched LOUPE**: rerun LOUPE with the same 4x4 slice replication to size
  explanation 2.
- **Frozen-recon policy training**: warm-start REINFORCE on LOUPE's reconstructor
  (`PRETRAINED_RECON=... sampler.freeze_recon=true`). This removes non-stationarity and the
  data confound, and is a direct test of explanation 1.
- **Better priors or estimators for the policy**:
  - a variable-density init (`reinforce_vd_power > 0`)
  - a radially factored policy (per-ring logit + residual) so credit is shared
  - 8–16 draws per slice with a leave-one-out baseline
  - no advantage normalization
  - a policy lr schedule decoupled from the recon's StepLR
- **Warm-start LOUPE from greedy** to test whether LOUPE's init-locked fine structure leaves
  performance on the table.
