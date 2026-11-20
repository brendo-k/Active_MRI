# Addendum 2026-11-20: a relative placebo bar for the conjugate-symmetry probe

This changes one sub-row of the mechanism readout in `proposal.md`, before any arm trained. The layout readout, the
value of the half, the guards and the noise-floor rule are unchanged.

## Why

- The frozen row: L_rss "at least 90% conjugate, at 10% or more of the self-response, with placebo points below 2%".
- The 2% came from October's probe of a 1-cascade LOUPE network (`reports/PARTIAL_FOURIER_REPORT.md`):
  - placebo points responded below 1% of the self-response, and at 1.8% and 2.4% without the sense mask;
  - even there, their antilinear fractions were 0.45 and 0.21, so their response was not conjugate.
- Today's 6-cascade VarNets spread one perturbation across much of k-space. fscr1's seed-42 tier-F runs, probed at run
  state 1 with `scripts/diagnostics/conjugate_symmetry_probe.py` (24 dev slices × 6 locations, n = 144 per point;
  `probe_fscr1.json`), give:

| Run | −k: antilinear fraction / response | Row placebo: fraction / response | Column placebo: fraction / response | −k minus row placebo | −k minus column placebo |
|---|---|---|---|---|---|
| L_1_s42_fscr1_tF | 0.927 / 0.82 | 0.475 / 0.36 | 0.491 / 0.34 | 0.45 | 0.44 |
| F_4_s42_fscr1_tF | 0.972 / 0.76 | 0.471 / 0.18 | 0.520 / 0.18 | 0.50 | 0.45 |
| Fb_4_s42_fscr1_tF | 0.985 / 0.92 | 0.563 / 0.16 | 0.547 / 0.17 | 0.42 | 0.44 |

- Fractions and responses are medians over the perturbations. A response is relative to the response at k itself.
- Placebo points respond at 16–36% of the self-response, so every run fails the 2% bar.
- Their antilinear fraction is 0.47–0.56. That is what a response with no conjugate relation to the perturbation
  gives: its linear and antilinear parts are equally likely. At −k the fraction is 0.93–0.98.
- So the 2% bar measures how far a network spreads one perturbation. It does not test whether the network fills −k by
  conjugate symmetry.

## The new sub-row

- It replaces "with placebo points below 2%". The median antilinear fraction at −k must be at least 0.30 above each
  placebo's, taking the row-mirrored and the column-mirrored placebos separately.
- The mechanism row now reads: L_rss, in both seeds, at least 90% conjugate at −k, at 10% or more of the
  self-response, and at least 0.30 above each placebo's antilinear fraction.
- It is still reported for every arm and for fscr1's F, Fb and L.
- `scripts/diagnostics/conjugate_symmetry_probe.py` now writes both differences for each run
  (`partner_minus_placebo_anti`). Nothing else in the probe changed.

## When it was set

- The bar was set after seeing fscr1's probe above, and before any arm of this experiment trained.
  - Run state 1's validation job failed on a test-harness bug, and its training jobs were cancelled before they started
    (`jobs.txt`).
- 0.30 is 0.12 below the smallest difference fscr1's runs show (0.42).
- A network that does not fill −k by conjugate symmetry leaves −k with a fraction near the placebos', a difference near
  0.

## Renamed 2026-11-20

- This experiment was renamed from `loupe_loss_partial_fourier` to `loss_domain__loupe_loss_partial_fourier`, under the
  loss_domain line. The rename covers the branch, the worktree, the report and launch folders and the CoreHPC folder.
- `proposal.md` is frozen and keeps the old paths.
