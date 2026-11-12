# P2 contrasts (11-11, from `../contrasts.json`)

Paired per-slice PSNR_Masked differences at epoch 30, pooled over 1938 val slices, with 95% volume-cluster bootstrap CIs.
Computed on the lab server from the CoreHPC score files (`p1/scores`, `p2/scores`); the console output is `../contrasts.txt`.

## Primary endpoint G_K = A_K − T_K (dB)

| K | seed 42 | seed 43 |
|---|---------|---------|
| 1 | +0.014 [+0.010, +0.019] | +0.001 [−0.003, +0.005] |
| 4 | −0.001 [−0.005, +0.004] | −0.002 [−0.009, +0.005] |
| 16 | +0.016 [+0.009, +0.025] | +0.018 [+0.011, +0.026] |

- G_K − G_1: K=4 −0.015 / −0.003; K=16 +0.002 / +0.017 (seed 42 / 43).
- Twin-vs-twin floor (base 42): T_K − T_K′ is −0.016 [−0.022, −0.010] at K=1 and −0.017 [−0.023, −0.011] at K=16, so
  F_1 = 0.016 and F_16 = 0.017, and 2·F_16 = 0.034. G_16 is about the size of the floor.

## Against the frozen rules (proposal section 7)

- **PASS: no.** G_K* ≥ +0.10 dB fails at both K in both seeds (largest +0.018), as do G_K* − G_1 ≥ +0.05 and G_K* ≥ 2·F*.
- **FAIL, condition 1: holds.** G_K < +0.05 dB at K=4 and K=16 in both seeds.
- **FAIL, recon-input ablation: holds.** G(A_4^rec) = A_4^rec − T_4 = +0.006 [+0.001, +0.011] < +0.05.
- **FAIL, from-scratch check: not run.** P2b (A_16^scr's within-model swap cost < +0.05) is required before FAIL can be
  declared. Its LOUPE start (VD p=4 per the proposal, or uniform per `feedback_no_vd_start`) is waiting on the user.
- The guards hold in every run (pooled SSIM 0.863–0.866, realised fraction 0.125 ± 0.002, no slice below 20 dB).

## Within-model readouts (adaptive arms)

| Arm | trajectory swap cost (dB) | largest per-step data-swap movement |
|-----|---------------------------|-------------------------------------|
| A_1_s42, A_1_s43 | +0.052, +0.045 | 6% |
| A_4_s42, A_4_s43 | +0.029, +0.031 | 10% |
| A_16_s42, A_16_s43 | +0.036, +0.032 | 23–25% |
| A_4_s42_rec | +0.036 | 11% |
| A_16_s42_img | +0.032 | 23% |

The adaptive heads are used: they move up to a quarter of a step's samples when another slice's data is swapped in, and
their own trajectories beat donors' by 0.03–0.05 dB inside the same model. That does not turn into a gain over the
data-blind twin.

## Reported only

- **Spacing effect** T_K − T_1: K=4 +0.059 / +0.049; K=16 +0.078 [+0.066, +0.090] / +0.062 [+0.051, +0.073]. Below the
  +0.10 dB bar in the "either way" clause, so the mask-aware static sampler is not proposed on these numbers.
- **Usefulness** U_K = A_K − T_1: K=4 +0.058 / +0.047; K=16 +0.094 / +0.081, CIs above 0 in both seeds. Almost all of
  it is the spacing effect, since G_16 ≈ 0.02.
- **Channel ablation:** A_16 − A_16^img = +0.003 [−0.001, +0.006]; G(A_16^img) = +0.014 [+0.006, +0.022].
- Per condition, G_16 is ≈ 0 in the brain and +0.01 to +0.03 in the knee, in both seeds.
- Every adaptive arm and twin here fine-tunes from a base whose LOUPE mask was learned from the VD p=4 start.
