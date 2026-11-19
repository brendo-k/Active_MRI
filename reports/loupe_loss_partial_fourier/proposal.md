# Does an RSS loss make LOUPE learn partial Fourier, and what is the one-sided half worth? (proposal, 2026-11-19)

| | |
|---|---|
| Status | **Frozen 2026-11-19** as the first commit on `exp/loupe_loss_partial_fourier` (experiment loop, step 2). Approved by the user on 2026-11-19. The decision rule does not change after this commit; any change is stated in the report with its reason. |
| `main` | `d498afd`, clean. `git fetch` could not authenticate (publickey), so main's sync with origin is unverified: it is 3 commits ahead of the last fetched `origin/main`. Nothing is pushed. |

- **Topic / branch / report:** `loupe_loss_partial_fourier` · `exp/loupe_loss_partial_fourier` ·
  `reports/loupe_loss_partial_fourier/`
- **Question.** Learned LOUPE maps have a partial-Fourier layout: symmetric within about 15 columns of DC, one-sided
  beyond. At 32 or more columns from DC, 4% of fscr1's tier-F map lies right of the DC column, and 9–14% of the
  loss-domain RSS maps. An RSS-magnitude loss does not charge for the phase errors that a one-sided acquisition causes; a
  complex-image loss does. Hypothesis: the loss decides the layout.
  - The only earlier test was loss_domain. Its round-1 complex map was more two-sided (33% right beyond 32 columns),
    but its round-2 complex map was one-sided (12%). Both complex arms were broken by the ESPIRiT phase convention.
- **Roadmap:** mask design under the scored loss. It tells later experiments whether a mask learned under one loss carries over.

## Arms (FLAIR tier F, R = 8, LOUPE + 6 × 18 VarNet, 10 × 10 centre, 100 epochs at 1e-3 with ×0.1 at 90; seeds 42 and 43)

| Arm | Loss | LOUPE map |
|---|---|---|
| L_rss (control, rerun at the new run state) | RSS L1, as now | free |
| L_cplx | complex-image L1: output Σ_c conj(S_c)·IFFT(k̂_c) against target Σ_c conj(S_c)·IFFT(y_c), both through the model's own maps, detached; same loss region as RSS | free |
| L_rss_sym | RSS L1 | point-symmetric |
| L_cplx_sym | complex L1 | point-symmetric |

- **The symmetric map:** logit(i, j) = ½ [w(i, j) + w((320 − i) mod 320, (320 − j) mod 320)], about DC at (160, 160).
  - Row and column 0 pair with themselves.
  - Draws stay inside the acquired band and are rescaled to the budget, so an in-band point whose partner falls outside
    the band keeps its own probability.
- **Unit tests:**
  - the map is exactly symmetric, and the realised fraction is 0.125;
  - the complex loss is 0 for a perfect reconstruction;
  - the complex loss is unchanged when the maps are rotated by a common phase;
  - the maps stay RSS-normalised.

## Readouts and decision rule (both seeds)

| Readout | Rule |
|---|---|
| **Layout (primary):** the share of the learned map right of the DC column, at 32 to 98 columns from DC, rows 33–287 | L_rss ≤ 0.15 and L_cplx ≥ 0.30 in both seeds: the loss decides the layout. L_cplx ≤ 0.15 in both: it does not. Otherwise inconclusive. (Symmetric = 0.50) |
| **Value of the half under RSS:** L_rss − L_rss_sym, paired pooled PSNR | ≥ +0.10 dB with the CI above 0 in both seeds: partial Fourier is worth at least 0.1 dB. CI including 0: worth nothing measurable |
| Value under the complex loss: L_cplx − L_cplx_sym | reported; expected to include 0 |
| **Mechanism:** the conjugate-symmetry probe (`scripts/diagnostics/probe_conjugate_symmetry.py`, ported to the manifest loader and to multi-step models with the final mask held fixed): a perturbation at a sampled k, and the response at the unsampled −k | L_rss: at least 90% conjugate, at 10% or more of the self-response, with placebo points below 2%. Reported for every arm and for fscr1's F, Fb and L |

- **Compare PSNR and SSIM only within a loss** (free against symmetric map), never complex against RSS. The complex target
  has no RSS noise floor, so scores against the RSS target are biased.
- **Guards:**
  - pooled SSIM ≥ 0.80;
  - each symmetric arm within 0.03 SSIM of its free arm, and L_cplx within 0.03 of L_rss, or debug before reporting;
  - realised fraction 0.125 ± 0.002.
- **Noise floor:** two seeds; every rule needs both.

## Expected outcome

- L_rss stays one-sided (about 0.05; fscr1 L_tF 0.044).
- If phase is the reason, L_cplx lands at 0.25–0.45.
- L_rss − L_rss_sym: +0.05 to +0.2 dB. On the older R=4 brain data, a kx-symmetrised mask lost 0.08 dB (`reports/PARTIAL_FOURIER_REPORT.md`).
- L_cplx − L_cplx_sym about 0.
- If L_cplx is also one-sided, the layout comes from somewhere else (the coils or the VarNet), and loss_domain's round-1
  map was an artefact of the broken arm.

## Code (run state 1) and compute

- The complex loss.
- `sampler.loupe_point_symmetric`.
- The probe port.
- The committed mask-asymmetry scripts, so the report shows curves, the learned maps and recons.
- 8 LOUPE runs at tier F, about 4–5 GPU-h each: about 35 GPU-h on CoreHPC. The probe runs locally on GPU 0, in about 2 GPU-h.
