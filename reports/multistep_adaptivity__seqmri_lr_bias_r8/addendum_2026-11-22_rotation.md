# Addendum 2026-11-22: does rotating a scan rotate the sampling density? (post hoc, not pre-registered)

The user asked for this probe after reading the report. It runs inference only, on the 12 checkpoints this report scores,
and changes no decision row.

## Method

`reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/rotation_probe.py` (`score`, `summarise`, `figure`).

- **Input.** Each of the 307 dev slices is rotated by T, an exact 90° rotation about the k-space centre:
  `rot90` and then a 1-pixel roll on the centred 320 × 320 grid. T commutes with the centred FFT (max error 8e-7 on a
  random test), so the image rotates the same way. The k-space, the target and the draw thresholds (seed 1234) are all
  rotated, so a perfectly equivariant sampler would draw exactly rotated masks.
- **Density.** Each step's sampling scores before the budget rescaling (`unscaled`). That is LOUPE's sigmoid map, or
  seqmri_2d's softplus over the slice's maximum. They are taken per step and summed over the steps. The budget-rescaled
  probabilities are not used: rescaling depends on the acquired region, which rotates with the scan. In a first version
  they gave LOUPE's fixed map w = +0.22.
- **Rotation index.** The comparison region Ω is k-space acquired in both orientations, outside both calibration centres.
  On Ω, p (density of the slice), q (density of the rotated slice) and T(p) are each turned into z-scores. With
  d = z(T(p)) − z(p):

  w = ⟨z(q) − z(p), d⟩ / ⟨d, d⟩

  w = 0 means the density stays where it was; w = 1 means it rotates with the anatomy. Standardising means a change of
  scale or offset alone counts as no movement.
- **Pooling.** Each run's pooled w is the sum over slices of the numerators divided by the sum of the denominators. The
  95% CI comes from resampling whole volumes (2,000 draws). The adaptive − twin gaps are paired on the same slices.
- **Run.** Lab server, GPU 1, local copies of the checkpoints and the data, train tier S loaded only to build the data
  loaders. Each run's unrotated PSNR here matches its score file in `scores/`.

## Results (`rotation/summary.json`)

| Run | w, steps summed | step 0 | step 1 | step 2 | step 3 | PSNR → rotated |
|---|---|---|---|---|---|---|
| F | +0.421 [+0.398, +0.456] | +0.242 | +0.482 | +0.574 | +0.588 | 33.48 → 29.21 |
| Fb | +0.193 [+0.189, +0.197] | 0.000 | +0.286 | +0.398 | +0.426 | 33.23 → 28.13 |
| L | 0.000 | 0.000 | | | | 33.37 → 24.81 |
| F_r150 | +0.363 [+0.323, +0.417] | +0.213 | +0.414 | +0.538 | +0.584 | 33.51 → 29.12 |
| Fb_r150 | +0.185 [+0.181, +0.189] | 0.000 | +0.275 | +0.393 | +0.416 | 33.15 → 27.97 |
| L_r150 | 0.000 | 0.000 | | | | 33.35 → 24.34 |
| F_lrs | +0.457 [+0.423, +0.495] | +0.308 | +0.490 | +0.565 | +0.576 | 33.52 → 28.25 |
| Fb_lrs | +0.192 [+0.176, +0.212] | 0.000 | +0.279 | +0.394 | +0.440 | 33.38 → 27.25 |
| F_route | +0.335 [+0.287, +0.403] | +0.139 | +0.385 | +0.475 | +0.471 | 33.33 → 30.60 |
| Fb_route | +0.202 [+0.197, +0.210] | 0.000 | +0.216 | +0.368 | +0.414 | 33.27 → 27.74 |
| F_routebias | +0.351 [+0.311, +0.405] | +0.158 | +0.346 | +0.470 | +0.483 | 33.39 → 28.70 |
| Fb_routebias | +0.221 [+0.211, +0.235] | 0.000 | +0.257 | +0.396 | +0.375 | 33.38 → 23.14 |

How much more each adaptive run's density rotates than its twin's (paired, 95% volume CI):

| Pair | w gap |
|---|---|
| F − Fb | +0.228 [+0.201, +0.263] |
| F_r150 − Fb_r150 | +0.178 [+0.136, +0.235] |
| F_lrs − Fb_lrs | +0.265 [+0.245, +0.287] |
| F_route − Fb_route | +0.133 [+0.081, +0.193] |
| F_routebias − Fb_routebias | +0.130 [+0.097, +0.174] |

- **Sanity checks hold.** LOUPE gives exactly 0, and every twin gives exactly 0 at step 0, where it reads only a fixed
  tensor and the calibration mask.
- **The twins' later steps** reach 0.22–0.44. They read their own earlier draws in the mask channel, and those draws
  rotate with the thresholds and the acquired region. This is the floor that reading the data has to beat.
- **Every adaptive run rotates more than its twin**, with every CI above 0, and at step 0 by 0.14–0.31 against 0.
  F_lrs moves most; the two route runs move least.
- **No run rotates fully.** Every w is below 0.5, so the density stays closer to where it was than to the rotated
  position. On the example slice, the one-sided wedge (the partial-Fourier side) stays where it was in F_lrs's rotated
  density as well.

## Reading

- The adaptive sampler does read the scan: part of its density follows a 90° rotation that its twin's does not. The
  larger part is a fixed, position-bound prior that every arm shares.
- This agrees with the main result. Reading the data moves the density, but the reconstructions gain little from it
  (B2 +0.009; F_lrs − Fb_lrs +0.14).
- Hypothesis, untested: the route and the per-step maps put more of the sampler's density into fixed per-position terms,
  which would explain their lower rotation gaps (+0.13 against +0.18 to +0.27).

## Caveats

- **The rotated input is out of distribution.** No run ever trained on rotated brains, and PSNR drops 2.7–10 dB. F's
  sampler reads in-loop reconstructions of the rotated scan, so its inputs are degraded too. So w measures how the
  trained sampler reacts to a rotated scan, not an equivariance it was trained to have.
- **The acquired extent rotates with the scan**, and Ω keeps only what both orientations acquired. On narrow and
  one-sided volumes that is a small part of k-space.
- **The twins' floor comes from reading their own draws.** With the thresholds left unrotated, the floor would be
  different. That variant was not run.
- One draw seed, seed-42 checkpoints, post hoc.

## Figure

`figures/rotation_probe.png` shows three things:
- every run's w with its CI;
- w by step for each adaptive run against its twin;
- for the first dev slice, p, q and T(p) for F_lrs, Fb_lrs and L, with Ω outlined.
