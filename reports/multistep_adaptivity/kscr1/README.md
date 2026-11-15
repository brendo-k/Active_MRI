# kscr1: knee-only from-scratch pair and floor twin (CoreHPC, run state 3 = 707a5c9)

The recipe is iteration 1 of `../addendum_2026-11-11.md`: `seq_adapt_2d`, K=16, 100 epochs (lr step at 90), LOUPE from
uniform noise, knee PD + PDFS, R=8, a 12-cascade VarNet. Training validates on the dev half of the knee val volumes, and the
scores here are on that half (669 slices, 20 volumes). The holdout half is untouched.

| Arm | Job | Role |
|-----|-----|------|
| A_16_s42_kscr1 | 3027891 | adaptive arm |
| T_16_s42_kscr1 | 3027892 | data-blind twin, same seed, so it starts bitwise identical to A |
| T_16_s142_kscr1 | 3027893 | floor twin, seed 142 |

- **All three completed** on 11-12 on the RTX PRO 6000, in 5.8 to 6.2 h each including the final scoring (about 220 s per
  epoch). Logs and scores are in this folder. `contrasts.json` and `contrasts.txt` come from
  `eval_multistep_adaptivity.py contrast --pair G_16_kscr1=A_16_s42_kscr1-T_16_s42_kscr1 --pair F_16_kscr1=T_16_s42_kscr1-T_16_s142_kscr1`.
- **Results** (PSNR in dB, with 95% CIs from resampling volumes):

  | Readout | knee-PD | knee-PDFS | pooled |
  |---|---|---|---|
  | G_16 = A − T | +0.102 [+0.072, +0.130] | +0.051 [+0.032, +0.072] | +0.077 [+0.058, +0.094] |
  | Floor T − T′ | −0.006 [−0.031, +0.017] | +0.010 [−0.007, +0.026] | +0.002 [−0.013, +0.016] |
  | Swap cost of A (own minus donor trajectory) | +0.190 [+0.128, +0.266] | +0.088 [+0.056, +0.116] | +0.139 [+0.104, +0.180] |

  The swap cost is 0 for both twins by construction.
- **Per-step data-swap movement of A** (pooled share of a step's draws that move when the data is swapped):
  - step 1: 27%;
  - steps 2–6: 9–10%;
  - then rising to a peak of 36% at step 15 (the zero-indexed step 14);
  - the last step: 27%.
- **Guards pass:**
  - pooled SSIM 0.865 (A), 0.864 (T) and 0.864 (T′);
  - realised fraction 0.125;
  - no slice below 20 dB.
- **Against the addendum's steering signal:** the swap cost (+0.139 pooled) is clearly above the fine-tunes' +0.03 to +0.05.
  - G_16 is well above twice the floor: F_16 = |mean(T − T′)| = 0.002, and G_16's CI does not overlap the floor's.
  - G_16 is still below the +0.10 dB pooled bar that the confirmation will apply.
  - Caveats: one seed, the dev half only, and a floor from a single twin pair.
