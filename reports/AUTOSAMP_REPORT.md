# AutoSamp on a Cartesian grid: implementation and R=8 brain+knee results (2026-10-05)

## Summary
- Added `autosamp_2d`, a sampler that learns a fixed set of continuous k-space positions (after
  [AutoSamp](https://arxiv.org/abs/2306.02888)) and snaps each to the nearest free Cartesian cell by L2 distance.
  The sample count equals the budget exactly.
- Trained it at R=8 on balanced brain + knee with a VarNet, matching the LOUPE and REINFORCE baselines.
- **The learned positions barely move.** After 50 epochs the median point has moved ~2 cells on a 256-cell grid,
  from either a uniform or a variable-density start, and a 5x learning rate changes nothing. The pattern is
  therefore its initialization, and the results track the initialization:
  - Uniform start: ~1 dB PSNR below LOUPE on brain (a random mask with a trained VarNet).
  - Variable-density (`normal`) start: level with LOUPE on brain and slightly ahead on knee, but this comes from the
    starting layout, not from learning. There is no frozen-init control, so this run cannot show whether learning
    helped or hurt.
- Cause (verified on the checkpoints): the straight-through gradient makes each point oscillate across its snap
  boundary instead of travelling. See [Why the positions do not learn](#why-the-positions-do-not-learn).
- None of these samplers is anatomy-conditioned, so brain and knee get the identical pattern (difference is exactly 0).

## What was built
`autosamp_2d` (`activemri/samplers/autosamp.py`, wired through `Sampler`).

| Piece | Behaviour |
|---|---|
| Parameters | `K = budget` positions `[K, 2]`, in grid-cell units on the fft-shifted array (DC at `[H//2, W//2]`), wrapping at the edges as AutoSamp wraps its coordinates modulo the field of view. |
| Init | `uniform` (distinct random cells) or `normal` (truncated Gaussian centred on DC, std = a quarter of the extent, AutoSamp's variable-density initializer). |
| Snap | Each point claims its nearest free cell by L2. On a collision the nearest claimant keeps the cell; the others search again with it removed. Cells already acquired (the 10x10 pre-selected centre) are never claimed. In training, displaced points are written back into the parameter. |
| Gradient | Forward is the hard mask. Backward is a bilinear splat of each continuous position onto its 4 neighbouring cells (straight-through). Acquired cells are excluded from the splat so points are not pulled toward the pre-selected centre. |
| Scope | `num_step=1` only (one fixed set of positions, input-independent like LOUPE). `data_parallel` and non-`straight_through` mask modes are rejected. AutoSamp's measurement-noise injection (`noise_std`) is not implemented. |

Trainer changes: the positions get their own parameter group (`autosamp_lr`, default 0.01 = AutoSamp's `lr_enc`),
with `weight_decay=0` and Adam `eps=1e-16`. The small `eps` was needed because position gradients scale with
k-space amplitude and are then shrunk by the global clip, which pushed them under Adam's default `eps=1e-8` and froze
them in a toy test at data scales <= 1e-3. The real pipeline normalizes k-space to ~1, so this is a safeguard.

Config: `sampler.sampler_type=autosamp_2d`, `sampler.autosamp_init=uniform|normal`, `sampler.autosamp_lr`.
Launcher: `scripts/train_autosamp2d_varnet_r8_brain_knee.sh`. Tests: `tests/test_autosamp_sampler.py`
(21 tests, passing on the project venv, torch 2.6.0 / Python 3.13).

## Setup
Same as the baselines: 256x256, R=8 (sparsity 0.125), `brain_knee_balanced` (validation on 344 brain and 344 knee
slices, 23 and 10 volumes), VarNet trained from scratch jointly with the sampler, batch 16, reconstructor lr 3e-4,
50 epochs, 10x10 pre-selected centre, one sampling step, seed 42. ~5 min per epoch. Metrics are the trainer's own masked
per-anatomy validation metrics, taken at the epoch saved as `best_model.pt`.

Runs:
- `autosamp2d_varnet_r8_brain_knee`: uniform init, lr 0.01
- `autosamp2d_varnet_r8_brain_knee_lr0.05`: uniform init, lr 0.05
- `autosamp2d_varnet_r8_brain_knee_normalinit`: normal init, lr 0.01

Run `lr0.05` was added because the REINFORCE v2 baseline got a tuned policy lr, so a single untuned AutoSamp run
would be an unfair comparison.

## Results
Final validation, best checkpoint per run:

| Run | Epochs (best) | Brain SSIM | Brain PSNR | Brain NMSE | Knee SSIM | Knee PSNR | Knee NMSE |
|---|---|---|---|---|---|---|---|
| LOUPE | 50 (50) | 0.9299 | 36.08 | 0.0142 | 0.9493 | 33.90 | 0.0134 |
| REINFORCE v1 | 50 (50) | 0.9067 | 33.21 | 0.0228 | 0.9351 | 32.44 | 0.0189 |
| REINFORCE v2 | 50 (48) | 0.9263 | 35.69 | 0.0149 | 0.9484 | 33.84 | 0.0136 |
| AutoSamp, uniform init, lr 0.01 | 50 (42) | 0.9227 | 35.10 | 0.0164 | 0.9476 | 33.47 | 0.0146 |
| AutoSamp, uniform init, lr 0.05 | 50 (42) | 0.9228 | 35.11 | 0.0163 | 0.9477 | 33.47 | 0.0146 |
| AutoSamp, normal init | 50 (46) | 0.9303 | 36.18 | 0.0139 | 0.9521 | 34.07 | 0.0128 |
| REINFORCE vd2.5, 100-epoch schedule (**still running**, epoch 75 at the time of writing) | 75 (74) | 0.9308 | 36.26 | 0.0138 | 0.9519 | 34.18 | 0.0125 |

Earlier 100-epoch results from the existing runs (not rerun here): LOUPE 0.9327 / 0.9526, REINFORCE v2 0.9311 / 0.9525
(brain / knee SSIM).

![Sampler comparison](../reports/sampler_comparison_r8/overview.png)

The figure shows each sampler's mean effective pattern for brain and knee, brain minus knee, brain minus LOUPE, the
radial sampling density, and the validation curves. Table and figure are regenerated with
`python scripts/compare_samplers_r8.py` (outputs in `reports/sampler_comparison_r8/`).

### Reading the table
- **Uniform-init AutoSamp is a random-mask result.** It is 0.98 dB (brain) and 0.43 dB (knee) below LOUPE, and
  about 0.004 SSIM below REINFORCE v2 on brain. Both learning rates give the same numbers, and the learned mask is
  still uniform at 0.12 density in every radial ring.
- **Normal-init AutoSamp matches LOUPE.** It is +0.10 dB and +0.0004 SSIM on brain, and +0.17 dB and +0.0028 SSIM on
  knee, and ~1.08 dB above the uniform-init run on brain. The brain difference to LOUPE is within what a seed change
  would likely produce (one seed per run, see Caveats). The knee difference is larger but still a single-seed result.
- **The ~1 dB gap between the two inits is the init, not learning.** See below.

## Why the positions do not learn
Displacement from the recovered seed-42 initial positions (replaying `train_sequential`'s RNG):

| | Uniform lr 0.01 | Uniform lr 0.05 | Normal init |
|---|---|---|---|
| Median displacement (cells) | 2.1 | 2.1 | 2.6 |
| 95th percentile (cells) | 5.4 | 5.0* | 6.4 |
| Largest radial-band change | 38 of 8,092 points | not measured | 44 of 8,092 points |
| Coordinates within 0.02 of a snap boundary (x.5; 0.04 if uniform) | 0.53 | 0.83 | 0.47 |

*From the epoch-25 checkpoint; the other columns are from the final best checkpoints.

- Per-ring density is unchanged from init in every ring for all three runs. The one apparent exception is the
  innermost ring, where a few init points landed inside the 10x10 pre-selected centre and were pushed out on the first
  forward pass. That is mechanical, not learning.
- Adam moves each coordinate by about `lr` per step, which would allow 80-400 cells of travel over ~8.4k steps. The
  points stay within ~2-3 cells and a 5x higher lr only packs more of them onto the boundary.

**Mechanism** (found by a parallel analysis in `reports/AUTOSAMP_REINFORCE_STALL.md`; the boundary clustering numbers
above reproduce independently on the final checkpoints):
- For a point in `[n, n+1)`, the splat gradient along an axis is `g[n+1] - g[n]`, with `g = dL/dmask`.
- If the point snaps to `n`, cell `n` is sampled and `n+1` is not. To first order, adding a sample at the empty cell
  always looks better than keeping the existing one, so the step pushes the point toward `n+1`.
- Once the point crosses `n + 0.5`, the roles swap and the gradient flips back. The point parks on the boundary,
  its cell flickers (about 20-25% of points change cell per step late in training), and it never reaches the integer
  where the splat would start comparing it with farther cells.

This corrects my earlier interim explanation (noisy, weak local gradients): noise is not the main limit, the
estimator itself traps the points.

### What it means for the init comparison
The normal-init run performs like a fixed, variable-density random mask plus a well-trained VarNet. Without a
frozen-init control (`autosamp_lr=0` from the same init) the data cannot say whether the ~2-cell jitter helps, hurts,
or is neutral.

## Brain and knee share one pattern
Loupe, REINFORCE and AutoSamp all learn an input-independent mask, and these runs have `anatomy_conditioning: false`,
so one pattern serves every slice. Brain minus knee is exactly 0 (max |diff| = 0.00) for every sampler here. The
validation k-space has no zero-padding at 256x256 for either anatomy, so valid-region masking does not separate them
either.

**Known gap:** `autosamp_2d` ignores the anatomy labels. Enabling `sampler.anatomy_conditioning=true` with it would
silently train one shared set of positions while the config says conditioning is on. LOUPE and REINFORCE do keep one
mask per anatomy under that flag. AutoSamp should either raise an error for the combination or learn positions per
anatomy; neither is done yet.

## Caveats
- **One seed per run.** Differences below roughly 0.001 SSIM should not be interpreted.
- **Best-epoch selection uses the validation set** that the table reports, for every run alike, so all rows share the
  same mild optimistic bias.
- **No frozen-init control** for the normal-init run.
- **REINFORCE vd2.5 is another session's run and incomplete** (epoch 75/100 with a 100-epoch schedule), so its row is
  not directly comparable with the 50-epoch rows.
- **Reconstructor and sampler train jointly**, so a mask's quality is entangled with how well its VarNet trained.
- Environment: the `dl` conda env is empty on this machine; training and tests used the project `.venv` (torch 2.6.0).
  GPUs are shared and one early smoke test hit an out-of-memory error from another job's allocation, then ran
  fine on a free GPU.

## Suggested next steps
1. **Frozen-init control:** `sampler.autosamp_init=normal sampler.autosamp_lr=0`, same seed. This is the one run that
   separates "the init" from "the learning".
2. **Fix the estimator.** The closest analogue of AutoSamp's NUFFT is a continuous forward: use the bilinear splat (or
   a sinc/Kaiser kernel) as the training-time mask weights, anneal toward a hard nearest-cell assignment, and snap only
   at eval. Using the k-space gradient directly (AutoSamp's actual signal) would need multicoil k-space passed into
   the sampler.
3. **Or drop positions for Cartesian sampling.** Per-cell logits (LOUPE, REINFORCE) already cover the space and
   give the exact budget through top-k.
4. Make `autosamp_2d` reject `anatomy_conditioning`, or add per-anatomy positions.

## Files
- Code: `activemri/samplers/autosamp.py`; edits to `samplers/sampler.py`, `models/sequential_sampling.py`,
  `training/trainer.py`, `config.py`, `configs/config.yaml`, `scripts/train_sequential.py`.
- Tests: `tests/test_autosamp_sampler.py`.
- Scripts: `scripts/train_autosamp2d_varnet_r8_brain_knee.sh`, `scripts/compare_samplers_r8.py`.
- Results: `reports/sampler_comparison_r8/{metrics.md,overview.png}`; checkpoints and logs under `checkpoints/autosamp2d_*`.
- Related: `reports/AUTOSAMP_REINFORCE_STALL.md`, `reports/LOUPE_VS_REINFORCE_R8.md`.
