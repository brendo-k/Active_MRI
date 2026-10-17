# Why do the learned R=8 samplers end up near-static? Signal, sampler transport and reconstructor co-adaptation (2026-10-16)

## Question
Every input-adaptive sampler trained so far at R=8 on brain + knee ends with masks that barely change across slices.
Across the conv_2d variants (k-space input, image input, bias head, two draws, logit clamp), a slice's map correlates
0.987-0.993 with its anatomy's mean map, and the brain and knee mean maps correlate 0.988-0.999
(`reports/conv_vs_loupe_r8/adaptivity.json`). None of them beats the input-independent LOUPE mask. LOUPE learns one
map for all inputs, so its population mask is built in and needs no explanation. The question is why a sampler that
*can* condition on its input does not. Three explanations were tested on small sub-problems:

- **H0, signal:** there is little per-slice structure that the sampler's input can predict.
- **H1, transport:** the way gradients reach the sampler's parameters keeps the population part and averages the
  per-slice part away.
- **H2, reconstructor:** the reconstructor, trained jointly on the masks the sampler makes, makes those masks
  self-reinforcing and so biases the sampler toward in-distribution, static masks.

**How to read the numbers.** During training each slice gets a random Bernoulli mask draw, and the sampler learns
from the gradient of the reconstruction loss with respect to its scores. That gradient is split into four parts:

| Part | Meaning |
|---|---|
| Population | What every slice wants changed in the mask, the same for all slices |
| Anatomy offset | What brain slices want differently from knee slices (the "anatomy signal") |
| Slice-specific | What one slice wants differently from the other slices of its anatomy |
| Draw noise | What changes from one random mask draw to the next for the same slice |

The SNR (signal-to-noise ratio) of a part is its energy (squared size, with the noise removed) divided by the energy of
the draw noise in one draw. An anatomy SNR of 0.0045 means one training step's gradient carries 0.0045 units of "brain
and knee want different masks" per unit of random draw noise. Separating the two takes on the order of 1 / 0.0045, or
about 220, draws of the same anatomy, before the sampler can even tell which way to move. Higher is better; the
comparison that matters is between reconstructors at the same masks.

Conventions. R=8 (12.5% of a 256 x 256 k-space, point-wise), 10 x 10 centre preselected. z is the score map (logit)
the sampler emits (LOUPE: slope x w); m is the binary mask the reconstructor sees; p is the effective sampling
probability after budget rescaling. "Neighbour" is the adjacent slice of the same volume. Gradient energies are
noise-free estimates: the inner product of two means over independent halves of the Bernoulli draws. "Disattenuated"
correlations divide by the same noise-free self-energies. PSNR is the trainer's background-masked metric.

## Runs compared
All real-data models: VarNet (12 cascades, 18 channels, `dc_mode=where`), `brain_knee_balanced` (1350 train slices per
anatomy), R=8, batch 16, lr 1e-3 stepped to 1e-4 at epoch 40, 50 epochs, seed 42, one training seed each.

| Name | Checkpoint | wandb | Sampler, training masks | Val brain PSNR / SSIM | Val knee PSNR / SSIM | Base PSNR, tolerance subset |
|---|---|---|---|---|---|---|
| conv ep50 | `checkpoints/conv2d_varnet_r8_brain_knee_floor0.05/last_model.pt` | `ziuji11f` | conv_2d on the SENSE-combined 10 x 10 centre, floors 0.05 | 36.28 / 0.931 | 34.07 / 0.951 | 38.45 |
| conv ep4 | same run, `resume_from_epoch4.pt` (after 5 epochs) | (resumed run) | as above, early state | val L1 0.0572 | | 36.29 |
| LOUPE | `checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05/last_model.pt` | `uo8d68n0` | loupe_2d, slope 5, floors 0.05 | 36.55 / 0.933 | 34.26 / 0.953 | 39.06 |
| uniform-random-mask | `checkpoints/random_mask_varnet_r8_brain_knee/last_model.pt` | `fkg5xsd6` | none: uniform random R=8 masks | 34.85 / 0.922 | 33.48 / 0.947 | 34.74 (on conv ep50 masks) |

Val metrics are the runs' final logged values. The last column is the trainer metric on the 86 validation slices
of the tolerance test (every 8th), with each reconstructor's own sampler, except the uniform-random-mask
reconstructor, which is scored on conv ep50 masks. All SSIMs are above 0.92.

The toy (method 8) is a separate 32 x 32 problem with two image classes whose best masks are known.

## Methods
1. **Gradient anatomy** (`scripts/diagnostics/mask_gradient_anatomy_r8.py`). 24 adjacent-slice pairs (12 training
   volumes per anatomy, one random pair per volume), 32 Bernoulli draws per slice. For each draw: the gradient of the
   slice's L1 loss with respect to z, through the sampler run's estimator (clamp, leak, sigmoid and straight-through
   floors, budget rescale), and with respect to m outside the centre. Split as
   g = population + anatomy offset + slice-specific + draw noise. Reported:
   - energy shares;
   - per-draw SNR (part energy over draw-noise energy);
   - split-half reliability of the slice part;
   - its neighbour and other-volume correlation;
   - the share kept after a Gaussian blur of sigma px.

   Operating points: each run with its own reconstructor; the conv ep50 masks through the LOUPE, conv ep4 and
   uniform-random-mask reconstructors; and the LOUPE masks through the conv ep50 reconstructor.
2. **Transport** (same script, conv runs only). How a sampler-network update moves every slice's scores: Jacobian
   products with the checkpoint's Adam preconditioner 1 / (sqrt(v) + eps), or plain SGD; forward-mode JVPs.
   - *Full update* from all 48 slices' gradients: alignment (correlation) of the common, anatomy-offset and
     slice-specific parts of the score change with the matching parts of an independent half of the gradient. A free
     per-slice map would have slice alignment equal to the split-half reliability.
   - *Single-slice push*: an update from one slice's slice-specific gradient alone, its cosine with that gradient
     ("self cosine"), and the response of other slices. Controls push the population and anatomy directions the same
     way.
   - *Anatomy push*: an update from the brain (or knee) offset; its response on the other anatomy.
3. **Pull** (`scripts/diagnostics/summarize_adaptivity_mechanisms_r8.py`). Correlation, outside the centre, of a
   reconstructor's population descent direction on m with p_other - p_own, between the conv ep50 and LOUPE masks.
4. **Tolerance** (`scripts/diagnostics/recon_mask_tolerance_r8.py tolerance`). Each reconstructor on its own sampler's
   masks, perturbed two ways at matched budget, with common random numbers across reconstructors and 2 draws:
   - *smooth per-slice reshape*: z + sigma x f, with f a unit-variance Gaussian random field of length 8 px;
     reported with the correlation r of the perturbed p with the base p;
   - *ring permutation*: a fraction of the drawn non-centre samples moved to empty locations at the same radius.

   95% CIs bootstrap volumes.
5. **Fine-tune control** (`recon_mask_tolerance_r8.py finetune --sigma 1.0`). Two copies of the conv ep50
   reconstructor, each trained 1000 steps (batch 8, lr 1e-4) on training slices: one on reshaped masks, one on
   unperturbed masks. Mismatch = (frozen penalty) - (penalty after adapting): the part of the reshape penalty that
   training on the new masks recovers.
6. **Free-mask transfer** (`scripts/diagnostics/recon_bias_slice_masks_r8.py`). Same 48 slices. Through a frozen
   reconstructor, from the conv run's population prior, fit (120 Adam steps, lr 0.05, 2 draws, the conv estimator):
   - one free map per anatomy (shared);
   - one free map per slice.

   Each slice map is scored on its own slice, its neighbour and two other volumes, as a gain over the shared map.
7. **Label predictability** (`scripts/diagnostics/oracle_labels_r8.py`, `label_predictor_gonogo_r8.py`, chain in
   `analysis/label_gonogo/`). Per-slice oracle masks fitted through the frozen conv ep50 reconstructor for 1200
   training and 128 validation slices, a 32-slice retest with another seed, and the conv sampler's U-Net fine-tuned
   to predict them.
8. **Toy** (`scripts/diagnostics/toy_adaptive_sampling.py`, grid in
   `analysis/adaptivity_mechanisms/run_toy.sh`). 32 x 32 real images: a smooth blob plus five bars, vertical for
   class A (energy along kx) and horizontal for class B. R=8 with the 4 x 4 centre preselected, and the conv sampler's
   estimator (`scores_to_prob` clamp, rescale, floored Bernoulli straight-through). Reconstructor: 3 unrolled CNN
   cascades with differentiable data consistency.

   Samplers:
   - static;
   - table: one map per class, given the class;
   - net: an MLP on the *acquired* 4 x 4 centre, given as raw real/imag parts, as magnitudes, or as the class
     one-hot (the MLP's output bias is a per-location map).

   Reconstructors: frozen generalist (pretrained 6000 steps on random row-band, column-band and isotropic densities),
   joint from scratch, or the jointly co-adapted reconstructor of the static run. Adaptivity is read as:
   - the correlation of the class-mean probability maps (1 = static);
   - the swap cost: PSNR with the image's own mask minus with a mask made for an image of the other class.

   Two seeds per arm. Also:
   - a decodability probe of the class from the inputs;
   - a direct regression of the net onto the table's maps;
   - the class probe of the net's hidden features;
   - a gradient split at the static mask;
   - a quality control: the reconstructor of the joint table run, well fitted but co-adapted to class-specific masks
     (`analysis/adaptivity_mechanisms/run_toy_quality_control.sh`).

## Results

### 0. Adaptivity performance so far (wandb project `SeqMRI`)
Final logged validation metrics (R=8, brain + knee, 50 epochs unless noted). Adaptivity of the conv samplers is
recomputed on the validation set (`scripts/eval/compare_conv_adaptivity_r8.py`,
`reports/conv_vs_loupe_r8/adaptivity.json`): slice-vs-anatomy-mean map correlation (brain / knee) and brain-vs-knee
mean-map correlation, 1 = static.

| Run | wandb | Sampler | Val brain PSNR / SSIM | Val knee PSNR / SSIM | Slice vs mean corr | Brain vs knee corr |
|---|---|---|---|---|---|---|
| LOUPE, floor 0.05 | `uo8d68n0` | loupe_2d (static) | 36.55 / 0.933 | 34.26 / 0.953 | 1 by construction | 1 |
| LOUPE, no floor | `ijhnsmnm` | loupe_2d (static) | 36.51 / 0.932 | 34.25 / 0.953 | 1 | 1 |
| REINFORCE, 300 epochs | `wqk8bxvy` | loupe_2d Plackett-Luce (static) | 36.53 / 0.933 | 34.32 / 0.954 | 1 | 1 |
| conv, k-space input | `ziuji11f` | conv_2d | 36.28 / 0.931 | 34.07 / 0.951 | 0.990 / 0.987 | 0.988 |
| conv, image input | `dyihq1h2` | conv_2d | 36.47 / 0.932 | 34.20 / 0.952 | 0.990 / 0.991 | 0.999 |
| conv, image input + bias head | `lj113kv4` | conv_2d | 36.51 / 0.933 | 34.20 / 0.952 | 0.992 / 0.993 | 0.999 |
| conv, 2 draws, batch 8 | `d4hb98to` | conv_2d | 36.49 / 0.926 | 34.21 / 0.949 | 0.988 / 0.983 | 0.989 |
| conv, logit clamp 10, leak 0.05 | `ewfi3v76` | conv_2d | 36.44 / 0.932 | 34.20 / 0.952 | 0.992 / 0.987 | 0.994 |
| conv, image input + MLP head (LayerNorm, split clip) | `069x1apw` | conv_2d | 36.40 / 0.932 | 34.15 / 0.952 | 1.000 / 1.000 | 1.000 |
| conv, MLP head, global clip | `ypmzc42n` | conv_2d | 27.96 / 0.881 (diverged) | 27.16 / 0.899 | | |
| MNet alternating | `35y4evjr` | mnet_2d | 33.23 / 0.914 (crashed, epoch 3) | 32.07 / 0.929 | | |
| uniform random masks | `fkg5xsd6` | none (reconstructor only) | 34.85 / 0.922 | 33.48 / 0.947 | | |

No adaptive sampler beats the static LOUPE or REINFORCE masks. The newest variant, the MLP location head, converged
to an exactly input-independent mask (across-slice std about 1e-6; roughness 1.40, LOUPE-like pixel structure 1.18).
Given a per-location parameter it learned a LOUPE mask and ignored its input.


### 1. The per-slice signal exists, is reliable, and is mostly unpredictable (H0)
Score-map (z) gradient through each reconstructor, 48 training slices, 32 draws each:

| Masks | Reconstructor | L1 loss | Population share | Anatomy share | Slice share | Slice SNR / draw | Anatomy SNR / draw | Split-half reliability (16 draws) | Neighbour corr | Other-volume corr |
|---|---|---|---|---|---|---|---|---|---|---|
| conv ep50 | conv ep50 (co-adapted) | 0.0471 | 0.026 | 0.025 | 0.949 | 0.17 | 0.0045 | 0.76 | 0.12 | -0.05 |
| conv ep4 | conv ep4 (co-adapted) | 0.0505 | 0.043 | 0.044 | 0.913 | 0.33 | 0.0159 | 0.87 | 0.21 | -0.06 |
| LOUPE | LOUPE (co-adapted) | 0.0465 | 0.030 | 0.028 | 0.942 | 0.18 | 0.0054 | 0.77 | 0.10 | -0.05 |

At a converged operating point the population gradient is near zero by construction, so the large slice share is
expected and not by itself evidence. The informative columns are the last three:
- The per-slice signal is a stable property of the slice (reliability 0.76-0.87).
- Only 10-21% of its variance is shared with the adjacent slice of the same volume, and nothing with other volumes
  (the -0.05 is the -1/23 that centring on the anatomy mean induces).

Share of the slice-specific energy kept at spatial scale >= sigma, and the neighbour correlation of what is kept:

| Masks, reconstructor | 1 px | 2 px | 4 px | 8 px |
|---|---|---|---|---|
| conv ep50, co-adapted | 0.32 / 0.26 | 0.16 / 0.40 | 0.07 / 0.54 | 0.03 / 0.64 |
| conv ep4, co-adapted | 0.39 / 0.41 | 0.24 / 0.60 | 0.14 / 0.77 | 0.09 / 0.83 |
| LOUPE, co-adapted | 0.22 / 0.21 | 0.10 / 0.36 | 0.05 / 0.52 | 0.02 / 0.66 |

About 85% of the converged per-slice signal is at sub-2-px scale: which exact neighbouring locations to sample. That
part is idiosyncratic. The smooth part (7% of the energy at >= 4 px) is shared with the neighbour (0.54) but not across
volumes. This smooth, volume-specific density change is the actual adaptive opportunity.

Per-slice oracle labels (method 7), as the correlation of a slice's label deviation from the anatomy-mean label:

| Comparison | Correlation |
|---|---|
| Same slice, two fitting seeds (ceiling) | 0.97 |
| Fine-tuned U-Net predictor, validation | 0.06 |
| Same predictor, its own training slices | 0.055 |
| k nearest training slices (centre k-space) | 0.02 |
| Native conv sampler | 0.025 |

The oracle masks are reproducible but neither predictable from the sampler's input nor even fittable by the U-Net on
its training slices. Scored on the whole validation set (688 slices, frozen conv ep50 reconstructor, common draws;
`reports/conv_vs_loupe_r8/label_gonogo.json`):

| Comparison (PSNR, dB) | Difference |
|---|---|
| Oracle labels vs best non-adaptive map (fitted per-anatomy map), 128 labelled val slices | +0.91 [+0.80, +1.03] |
| Predictor vs fitted per-anatomy map | -0.011 [-0.029, +0.008] |
| kNN vs fitted per-anatomy map | -0.020 [-0.039, -0.001] |
| Native conv sampler vs fitted per-anatomy map | -0.004 (38.451 vs 38.456 dB) |

No-go. About 0.9 dB of per-slice headroom exists through this reconstructor, and none of it is recovered from the
sampler's input.

### 2. The sampler network does not turn per-slice gradients into per-slice changes (H1)
Transport through the conv sampler network (method 2):

| Sampler | Update | Full update: common vs population | Full update: anatomy offset | Full update: slice-specific | Free per-slice map | Single push self cosine: slice | Control: population | Control: anatomy | Other-volume response / self | Brain push moves knee / brain |
|---|---|---|---|---|---|---|---|---|---|---|
| conv ep4 | Adam | 0.168 | 0.030 | -0.004 | 0.87 | 0.044 | 0.082 | 0.073 | 0.72 | 1.00 |
| conv ep4 | SGD | 0.067 | 0.087 | -0.014 | 0.87 | 0.115 | 0.126 | 0.260 | 0.95 | 1.02 |
| conv ep50 | Adam | 0.027 | -0.010 | -0.002 | 0.76 | 0.008 | 0.031 | 0.022 | 0.28 | 1.10 |
| conv ep50 | SGD | 0.048 | 0.043 | -0.002 | 0.76 | 0.015 | 0.036 | 0.034 | 0.35 | 0.66 |

- **Full update.** In one batch update the score change shared by all slices follows the population gradient (0.17
  with Adam at epoch 4, before equilibrium). The part that differs between slices is uncorrelated with what each
  slice wants (-0.014 to -0.002). A free per-slice map would reach 0.76-0.87.
- **Single-slice steps.** These align poorly with any target: the population and anatomy controls are only 1.1-4x the
  slice cosine. So the single-step numbers do not single out per-slice directions. The batch-level contrast does.
- **Early training.** At epoch 4 the network is close to a pure population map. A step for one slice moves other
  volumes' scores 72-95% as much along that slice's direction. A step along the brain offset moves the knee masks
  equally (ratio 1.00-1.02).

The earlier fit-capacity test agrees (`reports/conv_vs_loupe_r8/sampler_fit_capacity.json`, 32 training slices,
300 full-batch steps, VarNet frozen):

| Arm | Train PSNR |
|---|---|
| Native start | 38.57 dB |
| Free per-slice maps | 39.47 dB |
| Shared free map | 38.89 dB |
| Conv sampler weights fine-tuned (lr 1e-3 / 5e-3) | 38.51 / 38.53 dB |

### 3. A co-adapted reconstructor suppresses the adaptive signal it sends back (H2)
Same masks, different reconstructors:

| Masks | Reconstructor | L1 loss | Anatomy SNR / draw | Slice SNR / draw | Neighbour corr of slice part | Slice energy kept at >= 4 px |
|---|---|---|---|---|---|---|
| conv ep50 | conv ep50 (co-adapted) | 0.0471 | 0.0045 | 0.17 | 0.12 | 0.07 |
| conv ep50 | LOUPE | 0.0470 | 0.0156 | 0.19 | 0.18 | 0.14 |
| conv ep50 | conv ep4 | 0.0520 | 0.0436 | 0.26 | 0.43 | 0.29 |
| conv ep50 | uniform-random-mask | 0.0602 | 0.0405 | 0.46 | 0.65 | 0.37 |
| LOUPE | LOUPE (co-adapted) | 0.0465 | 0.0054 | 0.18 | 0.10 | 0.05 |
| LOUPE | conv ep50 | 0.0474 | 0.0120 | 0.22 | 0.32 | 0.11 |

At its own masks the co-adapted reconstructor returns the weakest anatomy-level signal in both directions:
- conv masks: 3.5x below the quality-matched LOUPE reconstructor;
- LOUPE masks: 2.2x below the conv reconstructor.

Its per-slice signal is also finer-scale and less shared. The less-fitted reconstructors (conv ep4,
uniform-random-mask) show this more strongly, but their loss is higher too. That part of the ladder fits two readings:
the reconstructor penalises unfamiliar masks (bias), or it has already absorbed what adaptation would add
(substitution).

Pull toward the other sampler's masks (method 3):

| Own masks | Reconstructor | corr(descent, p_other - p_own) |
|---|---|---|
| conv ep50 | conv ep50 | 0.025 |
| conv ep50 | LOUPE | 0.203 |
| LOUPE | LOUPE | 0.161 |
| LOUPE | conv ep50 | 0.097 |

The LOUPE reconstructor pulls conv masks toward LOUPE's. The conv reconstructor does not pull LOUPE masks toward the
conv masks more than LOUPE's own reconstructor does. The location-level pull is therefore one-sided, not a general
"each reconstructor wants its own masks" effect.

Tolerance to mask deviations (method 4), PSNR change vs each reconstructor's base masks:

| Reconstructor | Reshape sigma 0.5 | Sigma 1 | Sigma 1.5 | Ring 0.25 | Ring 1 |
|---|---|---|---|---|---|
| conv ep50 | -0.24 [-0.28, -0.19] (r 0.95) | -0.93 [-1.12, -0.79] (r 0.84) | -1.83 [-2.23, -1.54] (r 0.75) | -0.12 [-0.17, -0.10] | -1.01 [-1.30, -0.87] |
| conv ep4 | -0.15 [-0.19, -0.13] (r 0.97) | -0.58 [-0.72, -0.52] (r 0.91) | -1.17 [-1.46, -1.06] (r 0.84) | -0.06 [-0.10, -0.04] | -0.81 [-1.04, -0.72] |
| LOUPE | -0.13 [-0.16, -0.10] (r 0.97) | -0.59 [-0.74, -0.49] (r 0.91) | -1.41 [-1.80, -1.21] (r 0.82) | -0.09 [-0.13, -0.08] | -0.41 [-0.54, -0.37] |
| uniform-random-mask (conv ep50 masks) | -0.06 [-0.13, +0.03] | -0.16 [-0.27, -0.05] | -0.30 [-0.45, -0.18] | +0.16 [+0.12, +0.21] | +0.17 [+0.12, +0.26] |

All three jointly trained reconstructors lose 0.13-1.4 dB for small per-slice reshapes (r 0.82-0.97). At matched map
correlation, conv ep50 is not less tolerant than conv ep4 or LOUPE (`tolerance.png`), so no steepening during
training is shown. The reconstructor trained on diverse masks barely reacts, but its base PSNR is 3.7 dB lower and
variable-density masks are themselves out of its distribution.

Fine-tune control (method 5), reshape sigma 1, 86 val slices, dB [95% CI]:

| Quantity | conv ep50 (r 0.84) | LOUPE (r 0.91) |
|---|---|---|
| Penalty with the frozen co-adapted reconstructor | 0.93 [0.78, 1.11] | 0.59 [0.50, 0.74] |
| Penalty after training on reshaped masks (1000 steps) vs on unperturbed masks | 0.72 [0.60, 0.84] | 0.44 [0.37, 0.53] |
| Mismatch: penalty recovered by adapting | 0.21 [0.17, 0.30] | 0.14 [0.11, 0.23] |
| Cost of the reshape-trained copy on unperturbed masks | 0.09 [0.07, 0.12] | 0.04 [0.02, 0.06] |
| Control: unperturbed fine-tune, change on unperturbed masks | +0.03 | -0.00 |

For both co-adapted reconstructors about a quarter of the deviation penalty is mismatch (23-24%). Because 1000 steps
may not finish adapting, these are lower bounds. The co-adapted reconstructor charges this on any per-slice change, informative or not, and it is the size of
the anatomy-specific mask effects seen before (0.1-0.26 dB in the R=4 per-anatomy LOUPE runs). The other 0.72 dB is what a random
smooth reshape costs in information; an adaptive sampler's changes would be chosen, not random.

Per-slice free masks through each reconstructor (method 6), gain of a slice's own fitted map over the
reconstructor's per-anatomy refit, when scored on:

| Reconstructor | Own slice | Adjacent slice | Two other volumes | Per-anatomy refit vs prior |
|---|---|---|---|---|
| conv ep50 (co-adapted) | +0.31 [+0.25, +0.37] | -0.30 [-0.36, -0.25] | -0.37 [-0.40, -0.33] | +0.25 [+0.20, +0.31] |
| LOUPE | +0.24 [+0.19, +0.29] | -0.34 [-0.41, -0.28] | -0.40 [-0.45, -0.35] | +0.32 [+0.26, +0.38] |
| conv ep4 | +0.31 [+0.26, +0.36] | -0.33 [-0.41, -0.26] | -0.57 [-0.71, -0.46] | +0.94 [+0.72, +1.15] |
| uniform-random-mask | -0.04 [-0.51, +0.45] | -0.40 [-0.88, +0.12] | -0.97 [-1.50, -0.46] | +1.04 [+0.70, +1.43] |

Under the three jointly trained VarNets (conv ep50, conv ep4, LOUPE; two of them not co-adapted to these masks), a
slice's fitted mask gains 0.24-0.31 dB on its own slice and loses 0.30-0.34 dB on the adjacent slice, which is about
as similar an input as exists. So non-transfer is not specific to the reconstructor co-adapted to these masks. Through
conv ep4 the adjacent slice fares better than other volumes (-0.33 vs -0.57), matching its larger shared, smooth
gradient component, but it is still below the per-anatomy map.

The uniform-random-mask row tests nothing. Its per-slice fits did not improve their own slice (-0.04 [-0.51, +0.45]);
the 120 steps went into adapting the conv prior to that reconstructor (refit vs prior +1.04 dB). A reconstructor
trained on diverse masks therefore gave no usable per-slice transfer result.

### 4. Toy: where the adaptive optimum is known
Class decodability from the acquired 4 x 4 centre (held-out accuracy, purity 1.0):

| Input | Linear probe | MLP probe |
|---|---|---|
| Raw real/imag parts | 0.49 | 0.99 |
| Magnitude | 0.98 | 0.98 |

Final adaptivity, both seeds (class-mean corr, 1 = static / swap cost in dB):

| Sampler | Frozen generalist, from scratch | Joint, from scratch | Frozen generalist, from static mask | Frozen co-adapted, from static mask | Joint, from co-adapted + static mask |
|---|---|---|---|---|---|
| static | 1.00 / 0 (31.8 dB) | 1.00 / 0 (40.0-40.7 dB) | | | |
| table | 0.65 / 6.1 (32.7 dB) | 0.64 / 13.1-13.9 | 0.66 / 6.5-6.6 | 0.85 / 7.0-7.5 | 0.77-0.78 / 14.0-14.3 |
| net, label input | 0.65 / 6.7-7.0 (33.5 dB) | 0.78-0.82 / 12.8-12.9 | | | |
| net, magnitude input | 0.73-0.75 / 5.0-5.3 | 0.975 / 2.35; 1.00 / 0.10 | | | |
| net, raw input | 0.995 / 0.39; 1.00 / 0.07 | 1.00 / 0.0 | 0.77-0.79 / 4.7-4.8 | 1.00 / 0.0 | 1.00 / 0.0 |

Readings:
- **Adapting is valuable and possible.** Under a frozen generalist reconstructor, the table gains about 0.9 dB over
  static, and the net can represent the table's maps (direct regression: relative error 0.08 for raw input,
  84% of the class difference). Cross-sampler PSNR is otherwise confounded by how fast each parameterisation's masks
  become deterministic, so adaptivity is read from correlation and swap cost.
- **A free per-class table always adapts:** under every reconstructor, from every start.
- **The raw-input net, the analogue of the conv sampler, adapts only in one condition:** a frozen generalist
  reconstructor, starting from the population optimum (static mask). There its hidden features learn the class
  (linear probe 0.97-0.98, from 0.70 at initialisation).
  - With the same start mask and a frozen *jointly trained* reconstructor it stays static. That holds both for the
    reconstructor co-adapted to the static mask (features 0.68-0.77) and for one co-adapted to class-specific masks
    (from the joint table run; corr 1.000, swap 0.0). Both are well fitted (about 39-40 dB at the static mask against
    32 dB for the generalist), so the toy cannot separate co-adaptation from fit quality.
  - From scratch it stays static; the population direction is the likely reason (hypothesis).
  - Under joint training it stays static.
- **The magnitude-input net adapts with the frozen generalist but mostly not under joint training.**

Toy score-map gradient at the static mask (2 seeds):

| Reconstructor | Class share | Class energy | Draw noise / draw | Class SNR / draw | Class offset vs ideal move |
|---|---|---|---|---|---|
| co-adapted static | 0.085-0.117 | 2.1-4.4e-11 | 2.2-5.0e-8 | 0.0009 | 0.54-0.59 |
| jointly trained with the table (class-specific masks) | 0.140-0.155 | 0.6-1.1e-10 | 3.3-5.9e-8 | 0.0017-0.0018 | 0.60-0.61 |
| generalist | 0.102-0.103 | 5.0-5.1e-11 | 4.4-5.1e-9 | 0.010-0.011 | 0.60-0.63 |

The raw-input net from the static mask through the frozen table-trained reconstructor stays static (both seeds: corr
1.000, swap -0.01 / +0.01 dB, 42.2 / 43.4 dB).

The jointly trained reconstructors still point the class signal the right way (0.54-0.61 alignment with the ideal
move), but at 1/6 to 1/11 of the generalist's SNR. Their draw noise is 5-13x higher: a well-fitted reconstructor
reacts strongly to which exact locations each draw takes. A table integrates the weak signal anyway; a network that has
to learn class features from it does not.

## Conclusions
- **LOUPE.** LOUPE converges to a population mask by construction: one parameter map receives every slice's gradient.
  Nothing more is needed to explain it.
- **H0, signal: the main cause at the slice level.** Per-slice signal exists and is reproducible:
  - gradient reliability 0.76-0.87;
  - oracle-label retest 0.97;
  - +0.9 dB of per-slice oracle headroom through the frozen conv reconstructor.

  But it is idiosyncratic:
  - 10-21% of its variance is shared with the adjacent slice, none across volumes;
  - about 85% is at sub-2-px scale;
  - per-slice optima fitted through three jointly trained VarNets gain 0.24-0.31 dB on their own slice and lose
    0.30-0.34 dB on the adjacent one;
  - a U-Net predictor trained on 1200 oracle labels recovers none of the headroom (no-go, -0.011 dB vs the per-anatomy
    map).

  The only shareable part is a smooth (>= 4 px), volume-specific density change. It holds about 7% of the converged
  per-slice energy and correlates 0.54 with the adjacent slice.
- **H1, transport: confirmed at the batch level.**
  - In one update of the conv sampler network at epoch 4, the score change shared by all slices follows the population
    gradient (alignment 0.17 with Adam). The part that differs between slices is uncorrelated with what each slice
    wants (-0.014 to -0.004, against 0.87 for free per-slice maps). At epoch 50 the population gradient is near
    equilibrium and the common alignment is only 0.03-0.05.
  - The cancellation is visible at epoch 4. A step for one slice alone moves that slice's scores along its own
    gradient (self cosine 0.04 Adam, 0.12 SGD). The same per-slice components summed over a batch leave a slice-specific
    change with alignment of about 0 (-0.004, -0.014).
  - Early in training the network is effectively a population map: a step for one slice moves other volumes 72-95% as
    much, and a step along the brain offset moves knee masks equally.
  - Single-step self cosines are low for every direction (0.01-0.26), so the evidence is the batch cancellation, not
    an inability to follow per-slice directions in isolation.
  - In the toy, the same mechanism stops a sampler network from learning an adaptation worth about 0.9 dB that is
    decodable from its input (MLP probe 0.99) and representable (regression error 0.08), even with a frozen generalist
    reconstructor. From scratch it stays static (class-mean corr 0.995-1.000). Started at the population optimum, the
    same net under the same reconstructor learns it (0.77-0.79), and its hidden features pick up the class (probe
    0.69-0.70 to 0.97-0.98). This fits the population direction dominating early updates, as the epoch-4 transport
    numbers show on real data. The toy does not test that link directly.
- **H2, reconstructor: confirmed as a strong amplifier, not as the cause of non-transfer.** A reconstructor co-adapted
  to the current masks does three things:
  - it returns 2.2-3.5x less anatomy-level signal at its own masks than a quality-matched reconstructor trained on the
    other sampler's masks (both directions), and 9-10x less than the epoch-4 or uniform-random-mask reconstructors;
  - it charges a mismatch tax on per-slice reshapes, about a quarter of the penalty: 0.21 [0.17, 0.30] of 0.93 dB
    (conv ep50, r 0.84) and 0.14 [0.11, 0.23] of 0.59 dB (LOUPE, r 0.91), both lower bounds;
  - in the toy, a frozen well-fitted, jointly trained reconstructor is enough to block a network sampler. From the
    same static mask the raw-input net learns class masks with the frozen generalist (corr 0.77-0.79) but not with
    either jointly trained one (1.00), where the class SNR per draw is 6-11x lower while its direction is still right.
    The generalist is also much less accurate (32 vs 39-40 dB), so this toy result shows fit quality or co-adaptation,
    not co-adaptation alone. The real-data anatomy comparison against the quality-matched LOUPE reconstructor is the
    controlled evidence.

  Non-transfer of per-slice optima is not specific to it: they also fail to transfer under the LOUPE and conv ep4
  VarNets. The diverse-mask reconstructor gave no usable per-slice fit, so a reconstructor without any
  variable-density co-adaptation was not tested here.
- **Combined picture.** The conv sampler sees mostly unpredictable per-slice signal. Its updates cancel the per-slice
  part across a batch, and the population part saturates first. Meanwhile the reconstructor co-adapts to the
  resulting static mask, which shrinks the shared (anatomy or volume level) signal by 2-10x and taxes deviations. In the
  toy, free per-class tables always adapt and a net given the class label adapts too. A net given phase-invariant
  magnitudes adapts under a frozen generalist reconstructor but mostly not under joint training. A net that must learn
  the class from raw complex k-space, like the conv sampler, adapts only from a static start under a frozen generalist
  reconstructor.
- **Hypothesis, untested.** The one adaptive opportunity that survives, smooth volume-specific density, needs a
  low-dimensional sampler head trained against a reconstructor that has seen diverse masks.

## Caveats
- **Seeds and sample size.** One training seed for every real model. The gradient anatomy uses 48 training slices
  (24 pairs, 32 draws each); neighbour correlations are medians over 24 pairs.
- **Neighbour correlation as a predictability ceiling.** It is only approximate: adjacent slices have similar, not
  identical, sampler inputs.
- **Reconstructor ladder.** The less-fitted reconstructors (conv ep4, uniform-random-mask) also have higher loss on
  these masks. Variable-density masks are out of distribution for the uniform-random-mask reconstructor, so it is a
  different bias, not no bias. Only the LOUPE reconstructor is quality-matched to conv ep50 (L1 0.0470 vs 0.0471).
- **Transport** is a linearised one-step analysis with the checkpoint's Adam second moments and without momentum.
- **Tolerance perturbations** are random, so their penalties mix information loss and mismatch. The fine-tune split
  covers one level (sigma 1), two reconstructors and 1000 steps, so the mismatch is a lower bound.
- **Pull test.** The location-level pull toward a reconstructor's own training masks was one-sided (LOUPE recon 0.20,
  conv recon 0.10 vs 0.16 for LOUPE's own) and is not claimed.
- **Toy.**
  - It is a 32 x 32 synthetic two-class problem with an MLP sampler, not a U-Net over k-space.
  - PSNR across sampler types is confounded by how quickly each parameterisation's masks become deterministic
    (joint net arms have 7-9% stochastic locations vs 12-18% for joint tables), so adaptivity is read from correlation
    and swap cost.
  - A first toy grid was discarded for two reasons: the net read unacquired k-space (the mask could leak it to the
    reconstructor), and its logits had no clamp and grew to about 10^6. The reported grid uses the acquired 4 x 4
    centre only and the conv sampler's clamped `scores_to_prob`.
  - Two seeds per arm, one for some variants.
- **Prior results.** The step-0 slice-adaptivity outputs (`scripts/diagnostics/slice_adaptivity_gap*.py`) are no
  longer on disk, so their numbers are not used here.

## Figures
- `reports/adaptivity_mechanisms_r8/gradient_scales.png`: share of the slice-specific gradient energy kept vs blur
  scale, and its neighbour correlation, for every operating point and reconstructor.
- `reports/adaptivity_mechanisms_r8/gradient_radial.png`: conv ep50 score-gradient energy per location by k-space
  radius (population, anatomy, slice, one-draw noise).
- `reports/adaptivity_mechanisms_r8/tolerance.png`: PSNR change vs perturbed-map correlation (smooth reshape) and vs
  the fraction of samples moved within their ring.
- `reports/adaptivity_mechanisms_r8/toy.png`: class-mean mask correlation over training, table vs raw-input net, per
  reconstructor condition, two seeds.
- `reports/adaptivity_mechanisms_r8/summary.md`: every generated table; `summary.json`: the numbers.

## Next steps
1. Two-stage training, the condition under which the toy net adapted: fit the population mask first, then train an
   adaptive head against a reconstructor trained on diverse masks and held frozen, before any joint fine-tune. Test
   on R=8 with the anatomy label withheld; read adaptivity from brain-vs-knee correlation and label-swap cost.
2. Give the sampler only what is shareable: a low-dimensional, smooth head (a few density-profile parameters, or a
   blur-limited per-location map) predicted from phase-invariant calibration features (magnitude or power spectrum of
   the 10 x 10 centre), not a per-pixel U-Net on raw complex k-space. Under a frozen generalist reconstructor the toy
   magnitude-input net adapted where the raw-input one did not; under joint training it mostly did not.
3. Raise the per-draw SNR of the shared signal (anatomy SNR 0.0045 per draw at convergence): more draws per slice
   (`mask_draws`), or per-anatomy averaging of the sampler gradient before it reaches the head.
4. Repeat the fine-tune split with a reconstructor pre-trained on a broad variable-density family. That gives an
   in-distribution reference for the reshape penalty, and a second seed for the conv ep50 numbers.
