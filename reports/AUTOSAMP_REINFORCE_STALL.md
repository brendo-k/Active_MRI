# Why AutoSamp and REINFORCE barely move from init (R=8, brain + knee VarNet)

Runs: `autosamp2d_varnet_r8_brain_knee` (uniform init, lr 0.01), `..._lr0.05`, `..._normalinit`,
`reinforce2d_varnet_r8_brain_knee_v2`, `reinforce2d_varnet_r8_brain_knee_vd2.5_100ep`
(the vd2.5 run is still in progress; best checkpoint epoch 58, logged through epoch 68 of 100).
Scripts: `scripts/diagnose_mask_drift.py <run>...` (seed-42 init recovered by replaying
`train_sequential`; displacement and per-ring density) and
`scripts/diagnose_autosamp_restoring_force.py` (preselect check, STE gradient direction on the
trained checkpoint, Plackett-Luce inclusion density).

## Results (best logged brain epoch)

| | brain PSNR / SSIM | knee PSNR / SSIM |
|---|---|---|
| AutoSamp, uniform init (lr 0.01) | 35.10 / 0.9227 | 33.47 / 0.9476 |
| AutoSamp, uniform init (lr 0.05) | 35.11 / 0.9229 | 33.44 / 0.9476 |
| AutoSamp, normal (VD) init | 36.17 / 0.9302 | 34.07 / 0.9521 |
| REINFORCE vd2.5 (in progress) | 36.29 / 0.9305 | 34.12 / 0.9510 |
| LOUPE 50 ep (prior report) | 36.07 / 0.9299 | 33.90 / 0.9493 |
| REINFORCE v2 100 ep (prior report) | 36.36 / 0.9311 | 34.24 / 0.9525 |

The init sets the result. For AutoSamp, the VD init is worth +1.07 dB on brain, and a 5x lr changes
nothing. There is no frozen-init control, so the data do not show whether the learned updates
help or hurt. They add no measurable radial structure.

## AutoSamp: the points are stuck oscillating on cell boundaries

| | uniform lr 0.01 | uniform lr 0.05 | normal init |
|---|---|---|---|
| median / p99 displacement from init (cells) | 2.1 / 7.4 | 2.1 / 8.7 | 2.5 / 8.5 |
| coords within 0.02 of the x.5 snap boundary (uniform ≈ 0.04) | **0.53** | **0.83** | **0.44** |
| points changing cell per step (late training, wandb) | ~20% | ~25% | ~20% |
| STE gradient points back toward the boundary (null 0.5) | 0.83–0.85 | – | 0.78–0.79 |

- Per-ring density is unchanged from init in every ring for all three runs.
  - Uniform init stays at 0.12 everywhere.
  - Normal init keeps its truncated Gaussian.
  - The only exception is the innermost ring. There, 7 and 25 init points (uniform and normal)
    landed inside the 10×10 preselect and were evicted outward on the first forward pass. That
    is an artifact, not learning.
- Adam with eps 1e-16 moves each coordinate about lr per step, which allows 80–400 cells of
  travel over ~8.4k steps. Instead the points stay within ~2 cells, and raising the lr only packs
  more of them onto the boundary. That is a bounded oscillation, not slow learning.

**Mechanism.**
- The forward pass uses the hard mask of the nearest cell. The backward pass splats the point
  bilinearly onto `floor` and `floor + 1` (`autosamp.py:_splat`). For a point in `[n, n+1)`, the
  gradient along that axis is `g[n+1] − g[n]`, where `g = dL/dmask`.
- If the point snaps to `n`, cell `n` is sampled and `n+1` is not. To first order, adding a sample
  at the empty cell always looks better than keeping the existing one, so the step moves the
  point toward `n+1`.
- Once the point crosses `n + 0.5`, the roles swap and the gradient flips. The point parks on the
  boundary, its cell flickers every few steps, and it never reaches the integer where the splat
  would start comparing it with farther cells.
- **Each point is confined to roughly one cell per axis.** Even without the oscillation, local
  moves at lr 0.01 cannot shift density ~100 cells from the periphery toward DC.

**Fix options (none tested):**
1. Train with a continuous forward. Use the bilinear splat (or a sinc/Kaiser kernel) as the
   actual mask weights during training, anneal the kernel toward a hard nearest-cell assignment,
   and snap only at eval. This is the closest analogue of AutoSamp's NUFFT, where the
   measurement really is continuous in position.
2. Drop coordinates for Cartesian sampling. Exact-budget per-cell logits (LOUPE or REINFORCE)
   already cover this space, and coordinates only add the exact budget, which top-k also gives.

## REINFORCE: per-pixel noise dominates, and with a VD prior it erodes the prior

Inclusion density under the Plackett-Luce policy that is actually sampled in training (64 draws,
k = 8092, preselect excluded):

| ring | v2 init | v2 final | vd2.5 init | vd2.5 final | vd2.5 budget share init → final |
|---|---|---|---|---|---|
| 0.00–0.05 | 0.12 | 0.74 | 1.00 | 1.00 | 0.004 → 0.004 |
| 0.05–0.10 | 0.12 | 0.57 | 1.00 | 0.94 | 0.046 → 0.044 |
| 0.10–0.20 | 0.13 | 0.37 | 0.85 | **0.58** | 0.164 → 0.111 |
| 0.20–0.30 | 0.12 | 0.25 | 0.53 | **0.35** | 0.170 → 0.110 |
| 0.30–0.50 | 0.12 | 0.17 | 0.25 | 0.22 | 0.250 → 0.219 |
| 0.50–0.70 | 0.12 | 0.12 | 0.11 | 0.13 | 0.163 → 0.204 |
| 0.70–1.00 | 0.12 | 0.10 | 0.05 | **0.08** | 0.158 → **0.251** |
| >1.00 | 0.12 | 0.09 | 0.03 | 0.03 | 0.045 → 0.057 |

**v2 (uniform init).** The diagnosis in `LOUPE_VS_REINFORCE_R8.md` stands.
- The per-logit gradient SNR is about 0.01 per step, so drift only beats diffusion after
  10⁴–3·10⁴ steps. (`reports/gradient_snr.log` is a stale traceback from a failed attempt; the
  JSON and report were written afterwards.)
- The policy learns the centre ring by ring (0.12 → 0.74).
- The periphery barely moves (0.12 → 0.09–0.10), so 47% of the budget is still at r > 0.7.

**vd2.5.** The logits moved, but the movement is diffusion.
- Mean |drift| is 1.42–1.60 in every ring outside the preselect. A pure Adam random walk
  predicts ≈ 0.8 · lr · √steps = 0.8 · 0.02 · √9.8k ≈ 1.6. The size of the drift is set by
  noise.
- The ring means moved in the direction the SNR analysis predicts: 0.05–0.10 rose by 1.2 logits,
  and the corners fell by 0.3.
- The noise is the same size as the prior's radial slope (~1 logit per ring step), so it flattens
  the prior. The budget share at r > 0.5 rose from 37% to 51%, and the 0.1–0.3 band lost a third of
  its density.
- The deployed top-k mask changes even more (the init top-k is a hard disc, so the 0.2–0.3 ring
  falls from 1.0 to 0.38), but the policy never trained on that disc.

**Fix options (none tested):**
1. Factor the policy as per-ring logits (full policy lr) plus a per-pixel residual with ~10x
   lower lr, or an L2 penalty toward zero. Ring-summed signal beats noise within ~10–50 steps
   (prior SNR table), and per-pixel signal does not.
2. Lower the per-pixel policy lr with the VD init. At lr 0.002 the random walk is ~0.16 logits
   over the same steps, which preserves the prior.
3. Use 8–16 draws per slice with a leave-one-out baseline to raise per-step SNR.

## Update: Gaussian-surrogate gradient unsticks AutoSamp

Implemented in `activemri/samplers/autosamp.py` (`autosamp_kernel_sigma_start/end`,
`autosamp_anneal_fraction`). The forward mask stays hard. Only the positions' gradient comes
from a Gaussian splat of width σ, annealed 4 → 1 over the first 80% of training, after which
the positions freeze.

- **A soft (fractional) forward is invalid here.** Simulated k-space is noiseless, so a weight w
  on a cell reveals its whole value. At σ = 1, 73% of cells carried weight > 0.05. The first
  launch (`*_soft1.0*_leaky`) trained on near-full k-space: val brain SSIM 0.79 after epoch 1.
- **Trap vs kernel width** (uniform-init checkpoint; share of coords pushed back to the x.5
  boundary, then share of descent steps pointing toward DC in rings r ≥ 0.3; null 0.5 for both):
  - bilinear: 0.84; 0.47–0.51
  - σ = 0.5: 0.79; 0.48–0.51
  - σ = 1: 0.61; 0.50–0.55
  - σ = 2: 0.52; 0.57–0.66
  - σ = 4: 0.49; 0.73–0.86
- Anneal floor: σ below ~1 brings the trap back. That is why the anneal stops at σ = 1 and the
  positions then freeze.

Runs: `autosamp2d_varnet_r8_brain_knee_gauss4-1` and `..._gauss4-1_lr0.05`. Both use uniform
init and 50 epochs. Best checkpoint epoch 47.

| | brain PSNR / SSIM | knee PSNR / SSIM | median / p99 travel (cells) | corner density (init 0.12) |
|---|---|---|---|---|
| straight-through, uniform init | 35.10 / 0.9227 | 33.47 / 0.9476 | 2.1 / 7.4 | 0.12 |
| Gaussian 4→1, lr 0.01 | 36.24 / 0.9306 | 34.03 / 0.9522 | 22.6 / 33.5 | 0.045 |
| Gaussian 4→1, lr 0.05 | **36.39 / 0.9318** | **34.05 / 0.9519** | 36.1 / 61.7 | 0.006 |
| straight-through, VD init | 36.18 / 0.9303 | 34.07 / 0.9521 | – | 0.03 (init) |
| LOUPE, 50 ep | 36.07 / 0.9299 | 33.90 / 0.9493 | – | 0.002 |

**Learned profile.** Final per-ring density, lr 0.05:

| r | 0.05–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.5 | 0.5–0.7 | 0.7–1.0 | > 1.0 |
|---|---|---|---|---|---|---|---|
| density | 0.64 | 0.51 | 0.36 | 0.24 | 0.16 | 0.08 | 0.006 |

- Starting from a flat 0.12, the positions learned a variable-density profile. It is close to
  LOUPE's (0.78 / 0.61 / 0.44 / 0.28 / 0.16 / 0.05 / 0.002), slightly flatter in the centre.
- The lr now matters: 5x the lr gives ~1.6x the travel and a more complete profile. The old
  straight-through runs showed no lr dependence. Together these are the signature of the trap
  being gone.
- **The lr 0.01 run is travel-limited.** Its p99 travel (33 cells) is near the ~40-cell cap that
  lr × the high-σ steps allows, and 4.5% density is left in the corners.
- **Boundary parking at the end** (67% / 28% of coords within 0.02 of x.5) comes from the σ ≈ 1
  phase before the freeze. It does not affect the deployed hard mask.

**Caveats.**
- One seed per arm.
- The LOUPE and REINFORCE comparisons are 50-epoch runs on the same data. REINFORCE at
  100 epochs reaches 36.36 / 34.24.
- The brain margin over the VD-init straight-through run is small (+0.21 dB at lr 0.05). The
  clear result is mechanistic: uniform init now learns the profile on its own.
