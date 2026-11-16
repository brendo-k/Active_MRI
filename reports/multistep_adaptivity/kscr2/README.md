# kscr2: a reconstruction in the loop and a clairvoyant ceiling (CoreHPC, run state 4 = b21c6b2)

The recipe is iteration 2 of `../addendum_2026-11-15.md`. It is kscr1's recipe (`seq_adapt_2d`, 100 epochs with the lr
step at 90, LOUPE from uniform noise, knee PD + PDFS, R=8) with a 6-cascade VarNet. Every arm uses seed 42. The scores
are on the dev half of the knee val volumes: 669 slices from 20 volumes, slice hash 720f599c, the same slices as kscr1.
The holdout half is untouched. Steps are zero-indexed (0–15) throughout.

| Arm | Job | Wall time | Its sampler reads |
|---|---|---|---|
| T_16_s42_kscr2 | 3073268 | 5.2 h | a fixed tensor: the data-blind twin, the shared control |
| A_16_s42_kscr2 | 3073269 | 5.5 h | y ⊙ M_i, magnitude, as in kscr1 |
| C_16_s42_kscr2 | 3073270 | 5.5 h | the fully sampled k-space: the clairvoyant ceiling |
| S_16_s42_kscr2 | 3073272 | 9.5 h | also 4 ACS virtual coils of y ⊙ M_i and of the in-loop VarNet's prediction: SeqMRI-style |
| T_1_s42_kscr2 | 3073271 | 2.2 h | nothing: the one-step twin, i.e. static LOUPE |

- **Validation:** job 3073267 passed all 14 checks on 11-15 (`validate_3073267/`, `multistep_validate-3073267.log`).
- **All five completed** on 11-15/16 on the RTX PRO 6000 with exit 0, using 27.9 GPU-h in all (the addendum estimated 27).
  - Training time per epoch at epoch 100: T 178 s, A 183 s, C 186 s, S 312 s, T_1 64 s.
  - S's four in-loop VarNet passes add 70% per epoch and triple the peak memory (78.8 against 26.3 GiB).
- **wandb** (project SeqMRI):
  - training runs: kvwzces4 (T_16), vnodonfo (A), 792hycn3 (C), mutu2yqe (S), v3pdjmd4 (T_1);
  - scoring runs: pl68zbvz (T_16), ctq8ijz8 (A), 6crbxrro (C), iyjs1b1e (S), iq5hnarq (T_1).
- **Contrasts:**
  - `contrasts.json` and `contrasts.txt` come from `eval_multistep_adaptivity.py contrast --scores scores` with the pairs in
    run.md 5c, plus `T16_minus_T1`, `S_minus_T1` and `C_minus_T1`.
  - `contrasts_c_minus_s.*` adds C − S.
  - `contrasts_cascades.*` compares against kscr1. It was run on a folder that links `../kscr1/scores` and `scores`.

## Results

PSNR in dB, with 95% CIs from resampling volumes:

| Arm | knee-PD | knee-PDFS | pooled | SSIM pooled |
|---|---|---|---|---|
| T_1 | 36.270 | 33.046 | 34.670 | 0.864 |
| T_16 | 36.328 | 33.076 | 34.714 | 0.865 |
| A_16 | 36.392 | 33.130 | 34.774 | 0.866 |
| S_16 | 36.341 | 33.184 | 34.774 | 0.866 |
| C_16 | 36.710 | 33.305 | 35.020 | 0.872 |

| Contrast | knee-PD | knee-PDFS | pooled |
|---|---|---|---|
| G_A = A − T | +0.064 [+0.042, +0.086] | +0.054 [+0.031, +0.077] | +0.059 [+0.043, +0.074] |
| G_S = S − T | +0.013 [−0.014, +0.039] | +0.108 [+0.062, +0.150] | +0.060 [+0.033, +0.085] |
| G_C = C − T | +0.382 [+0.306, +0.459] | +0.229 [+0.164, +0.289] | +0.306 [+0.255, +0.355] |
| S − A | −0.051 [−0.079, −0.021] | +0.053 [+0.024, +0.087] | +0.001 [−0.021, +0.023] |
| C − A | +0.318 [+0.255, +0.382] | +0.174 [+0.128, +0.217] | +0.247 [+0.207, +0.286] |
| C − S | +0.369 [+0.288, +0.453] | +0.121 [+0.072, +0.165] | +0.246 [+0.197, +0.296] |
| A − T_1: the gap to one-step LOUPE | +0.122 [+0.092, +0.151] | +0.084 [+0.053, +0.113] | +0.103 [+0.080, +0.123] |
| T_16 − T_1: its data-blind part | +0.057 [+0.042, +0.072] | +0.030 [+0.014, +0.047] | +0.044 [+0.033, +0.056] |
| S − T_1 | +0.071 [+0.041, +0.099] | +0.138 [+0.082, +0.192] | +0.104 [+0.071, +0.135] |
| C − T_1 | +0.440 [+0.356, +0.522] | +0.259 [+0.185, +0.327] | +0.350 [+0.293, +0.404] |

The swap cost is the own trajectory minus a donor's trajectory, both on the slice's own data:

| Swap cost | knee-PD | knee-PDFS | pooled |
|---|---|---|---|
| A | +0.149 [+0.107, +0.188] | +0.077 [+0.052, +0.103] | +0.113 [+0.088, +0.137] |
| S | +0.242 [+0.193, +0.284] | +0.135 [+0.096, +0.169] | +0.189 [+0.157, +0.217] |
| C, inflated by the leak (below) | +1.188 [+0.958, +1.399] | +0.787 [+0.737, +0.842] | +0.989 [+0.868, +1.103] |

The swap cost is 0 for both twins by construction.

- **Per-step data-swap movement** (pooled share of a step's draws that move when the data is swapped):
  - **A:** 9% at step 0, then 14–18% in steps 1–10, rising to 37% at step 14 and ending at 20% at step 15. kscr1's A had
    27% at step 0 and 36% at step 14.
  - **S:** 35% at step 0 and 39% at step 5, then 22% at step 6, rising again to 41% at step 14. Step 6 is the first step
    that reads an in-loop reconstruction.
  - **C:** 49% at step 0, rising to 75% at step 14.
  - **T_16 and T_1:** 0 at every step.
- **Guards pass:**
  - pooled SSIM 0.864–0.872 (knee-PDFS 0.80–0.81 per condition, as in kscr1);
  - realised fraction 0.125;
  - no slice below 20 dB;
  - every arm reproduces its own steps, and the twins' swap null is exactly 1.

## Against the outcome map (fixed in the addendum)

- **Row 1 (S clearly above A) fails.**
  - S − A pooled is +0.001 [−0.021, +0.023].
  - S's swap cost did rise over A's (+0.189 against +0.113), but "clearly above" needs both conditions.
  - By condition (post hoc, one seed), S is worse than A on knee-PD (−0.051) and better on knee-PDFS (+0.053).
- **At face value, row 2 holds:** C is well above both A (+0.247) and S (+0.246).
  - The reading: the information exists, but partial data does not reveal it to this sampler.
  - The next step would be what the early steps acquire.
- **The leak check (`leak/README.md`, 11-16, run state 5 = b8fff95) keeps row 2 by its pre-registered rule, and retracts
  the +0.25 dB ceiling.**
  - C's mask is computed from the full k-space, so it can tell the VarNet about k-space it never measured. A's and S's
    masks depend only on measured data, so they carry no such code.
  - **Through T_16's VarNet,** which cannot read C's masks as a code, C's swap cost minus A's is +0.184 [+0.078, +0.272].
    - So C's choices are slice-matched in a way that VarNet still uses.
    - The pooled effect comes from knee-PD.
    - T_1's VarNet does not confirm it: +0.064 [−0.081, +0.183].
  - **None of C's +0.247 over A survives through T_16's VarNet:** the gap is −0.083 [−0.150, −0.033]. That readout
    answers one way only.
    - C's swap cost falls from +0.99 to +0.26.
    - C's VarNet loses 0.37–0.41 dB on every other arm's masks.
  - **So how much a better partial-data sampler could gain over A is unknown,** and well under +0.25 dB.
- **The gap to one-step LOUPE** is A − T_1 = +0.103 pooled.
  - +0.044 of it is the data-blind 16-step spacing (T_16 − T_1), so adaptivity proper is A − T = +0.059.
  - S did not widen the gap: S − T_1 = +0.104.
- **Swap cost as a steering signal:** S's swap cost rose with no gain in G. A VarNet that co-adapts to an arm's own masks
  also raises it, so on its own it is not evidence of benefit.

## Reported only

- **6 vs 12 cascades** (`contrasts_cascades.txt`):
  - A6 − A12 = +0.113 [+0.090, +0.139] and T6 − T12 = +0.131 [+0.106, +0.157], pooled.
  - The runs differ in run state (707a5c9 against b21c6b2) and in initialisation. The CIs resample slices only, not
    training seeds, so this is unreplicated.
  - G_A was +0.077 at 12 cascades and is +0.059 at 6; A's swap cost was +0.139 and is +0.113. The CIs overlap.
- **Compute:** each extra 6-cascade VarNet pass costs about 32 s per epoch and 13 GiB. Reconstructing before every step
  would need activation checkpointing to fit on the card.

## Caveats

- One seed, the dev half only, and no new floor twin. kscr1's floor (+0.002 [−0.013, +0.016]) stands in.
- C's own-VarNet numbers are mostly that VarNet reading C's masks (`leak/README.md`). Through a VarNet that never saw
  them, C's masks are no better than A's.
- Order-free only. Nothing here is a SPACE gain.
