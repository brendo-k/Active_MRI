# fscr1: SeqMRI's design on all the FLAIR data (CoreHPC, run state 7 = d58087f)

The design and the reading rule are in `../addendum_2026-11-16_fscr1.md`, written before the runs.

| Arm | Sampler | Train data | Job | Wall time | wandb (train, score) |
|---|---|---|---|---|---|
| F_4_s42_fscr1_tF | `seqmri_2d` with the signed-log input, 4 steps | tier F | 3090320 | 19.1 h | pz9ml9l1, igs4u7rc |
| Fb_4_s42_fscr1_tF | its data-blind twin | tier F | 3090321 | 6.4 h | ptm1frx8, 9xlz3ghq |
| L_1_s42_fscr1_tF | static one-step LOUPE | tier F | 3090322 | 4.1 h | wrmr5kvu, gbahk74y |
| F_4_s42_fscr1_tS5 | as F | tier S, every 5th slice | 3090323 | 1.4 h | p4kcarwq, vl27bknd |
| Fb_4_s42_fscr1_tS5 | as Fb | tier S, every 5th slice | 3090324 | 0.6 h | w3076j1d, zoxgmylc |
| L_1_s42_fscr1_tS5 | as L | tier S, every 5th slice | 3090325 | 0.4 h | mqjmiygw, ptqmcvoj |

F_tF's `epoch_010.pt` was scored by job 3099177 (wandb x0f6kh6q). Wall times include each job's final scoring.

- **Validation:** job 3090319 passed all 16 checks on 11-17 (`validate_3090319/`, `multistep_validate-3090319.log`).
- **All six completed** on 11-17/18 on the RTX PRO 6000 with exit 0.
  - With the validation job and the epoch-10 score that is 32.5 GPU-h; the addendum estimated about 45.
  - Training time per epoch at epoch 100: F 674 s, Fb 223 s, L 139 s at tier F; 34, 14 and 8 s at tier S5.
  - Peak memory at tier F: 35.8, 26.2 and 20.9 GiB.
- **Data:**
  - Tier F is 339 volumes, 5,139 training slices: about 32,000 optimizer steps over 100 epochs.
  - Tier S5 is 238 slices of 80 volumes: about 1,500 steps.
- **Scoring:** the dev half, 307 slices of 20 FLAIR val volumes (slice hash 1874f6fc), draw seeds 1234 and 4321.
  - Training and scoring ran at d58087f with a clean tree. The holdout half is untouched.
- **wandb:** project SeqMRI. The offline runs were synced on 11-18 from `/export/scratch/bkadota/multistep_adaptivity_sync` (`sync_fscr1.log` there).
- **Files:**
  - `scores/`;
  - `contrasts.{json,txt}`: the pairs in run.md 5d;
  - `contrasts_descriptive.{json,txt}`: Fb − L, each arm's data gain, and F's epoch 100 against its epoch 10;
  - `by_scan_type.{json,txt}` from `analysis/by_scan_type.py`: post hoc, see below;
  - the slurm logs and `fscr1_jobs.txt`.

## Results

| Arm | PSNR | SSIM | NMSE |
|---|---|---|---|
| F_tF | 33.476 | 0.896 | 0.00435 |
| Fb_tF | 33.226 | 0.890 | 0.00478 |
| L_tF | 33.367 | 0.891 | 0.00458 |
| F_tS5 | 31.943 | 0.873 | 0.00605 |
| Fb_tS5 | 31.977 | 0.871 | 0.00614 |
| L_tS5 | 31.873 | 0.870 | 0.00619 |
| F_tF at epoch 10 | 32.438 | 0.880 | 0.00545 |

Pooled PSNR contrasts in dB, with 95% CIs from resampling volumes:

| Contrast | tier F | tier S5 | tier F minus tier S5 |
|---|---|---|---|
| U = F − L (the primary at tier F) | +0.109 [−0.034, +0.287] | +0.070 [−0.010, +0.159] | +0.039 [−0.136, +0.254] |
| G = F − Fb | +0.250 [+0.076, +0.467] | −0.034 [−0.131, +0.082] | +0.284 [+0.184, +0.404] |
| B = Fb − L (descriptive) | −0.141 [−0.216, −0.071] | +0.104 [−0.061, +0.245] | −0.245 [−0.335, −0.136] |

- **Each arm's data gain** (tier F minus tier S5): F +1.533 [+1.366, +1.704], Fb +1.249 [+1.081, +1.418], L +1.494
  [+1.250, +1.715].
- **F_tF from epoch 10 to epoch 100:** +1.038 [+0.921, +1.151].
- **SSIM (descriptive):** U_tF +0.0053 [+0.0015, +0.0099], G_tF +0.0061 [+0.0007, +0.0122].
- **F reads its data; the signed log did its job:**

  | | Data-swap movement, steps 0–3 | Zero-data movement, steps 0–3 | Swap cost through its own VarNet |
  |---|---|---|---|
  | F_tF, epoch 100 | 25%, 28%, 34%, 40% | 37%, 45%, 59%, 70% | +0.188 [+0.132, +0.247] |
  | F_tF, epoch 10 | 16%, 10%, 12%, 19% | 46%, 24%, 35%, 46% | +0.062 [+0.037, +0.089] |
  | F_tS5, epoch 100 | 10%, 9%, 13%, 18% | 53%, 25%, 28%, 42% | +0.050 [+0.037, +0.064] |
  | kscr3's F, raw input | 0.7%, 0.2%, 0.2%, 0.2% | density unchanged | −0.000 |

  - The swap's donors share the slice's acquired region and coil count: 245 slices of 16 volumes. Four volumes (62
    slices) have no donor.
  - Own-VarNet swap costs include the VarNet's co-adaptation to its arm's masks (kscr2's leak check).
- **Guards pass in all seven scores:**
  - pooled SSIM 0.870–0.896;
  - realised fraction 0.125;
  - no draw outside the acquired region;
  - no slice below 20 dB;
  - both twins' swap null is exactly 1;
  - every `seqmri_2d` score reproduces its own steps.
- **Early warning:** it did not fire.
  - At epoch 10 a data swap moved 10–19% of F's draws per step, against kscr3's 0.2–0.7%.
  - It ran late, though. The score job waited in the queue and ran at 09:02 on 11-18, five hours after F had finished.

## Reading, by the rule fixed before the runs

| Contrast, pooled PSNR | Result | Row it falls in |
|---|---|---|
| F_tF − L_tF, +0.10 dB (primary) | +0.109, CI includes 0 | **no gap at full data** |
| F_tF − Fb_tF, +0.05 dB | +0.250, CI above 0 | reading the scan is worth something: a patient-specific gain, with the credit-route caveat |
| (F_tF − L_tF) − (F_tS5 − L_tS5), +0.10 dB | +0.039, CI includes 0 | **no evidence that data changes the gap** |
| (F_tF − Fb_tF) − (F_tS5 − Fb_tS5), +0.05 dB | +0.284, CI above 0 | more data grows the adaptive gain |

- **The design was tested.** The clause for an F that is data-blind at epoch 100 does not apply: a data swap moves
  25–40% of its draws.
- **The two twin rows pass only because the twin trails LOUPE.** On the same slices and draws, F − Fb = (F − L) − (Fb − L)
  exactly:
  - G_tF = U_tF − B_tF = +0.109 + 0.141 = +0.250;
  - G_data = U_data − B_data = +0.039 + 0.245 = +0.284.
  - U's CI includes 0 in both. Neither pass shows F beating the best static design; both show the twin falling behind
    one-step LOUPE: by 0.14 dB at tier F, and by 0.25 dB more than at tier S5.
  - That fits the caveat stated before the runs: under SeqMRI's routes the twin's steps 0–2 learn only through the
    weights they share with step 3. Its data gain (+1.25) is the smallest of the three arms.
  - The cause is not separated. At tier F the twin also has the lowest train loss and the highest val loss of the three
    arms (2.288e-2 and 2.488e-2 at epoch 100), which fits its VarNet overfitting to a near-fixed mask.
- **Against the expected outcomes:**
  - F − L (+0.109) is inside the expected −0.1 to +0.2.
  - F − Fb (+0.25) is above the expected 0 to +0.1, and its difference of differences (+0.28) is above the expected 0 to
    +0.05. Both misses are the twin's shortfall.
  - The first difference of differences (+0.04) is just above the expected −0.3 to 0: F − L was not larger at tier S5.
  - L's convergence was not checked directly. Its trainer val loss moves by about 1% after epoch 40.
- **The user's question, whether more data and longer training help:**
  - Every arm gains +1.25 to +1.53 dB from tier S5 to tier F (about 21 times the slices and the optimizer steps).
  - F's lead over LOUPE does not detectably change: +0.07 at tier S5, +0.11 at tier F.
- **Scope:** one seed. The seed spread of joint mask and VarNet training (0.3–0.5 dB) is not in these CIs. The primary
  did not pass, so by the rule seed 43 is not earned.

## Post hoc: where F's +0.11 over LOUPE comes from (a hypothesis, not a finding)

The split below was chosen after sorting the 20 dev volumes by F − L, so it was not pre-registered.

- **Three volumes carry it.** F − L is positive on 8 of the 20 volumes.
  - Three 1.5 T Aera volumes (AXFLAIR_202, 48 slices) give +0.86 on average: +0.55, +0.84 and +1.19. Their k-space is
    264 columns wide with 198 acquired lines; most dev volumes are 320 wide with 224.
  - The other 17 volumes give −0.03.
- **Split by k-space width** (`by_scan_type.txt`, with the scorer's volume bootstrap):

  | Dev volumes | Volumes | L_tF | F − L, tier F | F − L, tier S5 | F − Fb, tier F | F's swap cost |
  |---|---|---|---|---|---|---|
  | narrower than the 320-column grid | 9 | 33.09 | +0.360 [+0.117, +0.640] | +0.133 [−0.019, +0.297] | +0.448 [+0.078, +0.866] | +0.165 |
  | full width | 11 | 33.60 | −0.099 [−0.119, −0.079] | +0.018 [−0.046, +0.069] | +0.086 [+0.054, +0.117] | +0.202 |

- **The groups are confounded.**
  - 5 of the 9 narrower volumes are 1.5 T Aera; the full-width group is mostly 3 T Skyra.
  - Split by field strength instead, F − L is +0.48 at 1.5 T (6 volumes) and −0.05 at 3 T (14 volumes).
  - 206 of the 339 tier-F training volumes are full width.
- **The hypothesis: F's lead is scan-type routing that it learned from the data.**
  - LOUPE learns one map, mostly for full-width scans, and restricts it to a narrower scan's acquired region.
  - F can see the narrower region, and perhaps the lower SNR at 1.5 T, and place its samples for them.
- **Two observations fit it:**
  - The two 16-coil AXFLAIR_202 volumes are each other's only swap donors. Their swap costs are +0.03 and +0.06, against
    F − L of +0.84 and +1.19. F's masks for one of these scans serve the other nearly as well, so the gain belongs to
    the scan type, not to the patient.
  - At tier S5, with every 5th slice of 10 such training volumes instead of 51 full volumes, F's lead on them is
    absent (−0.24, −0.22 and +0.24).
- **The full-width CI holds only slices and draws, not seeds.** "F trails a converged LOUPE on full-width scans" is a
  hypothesis too.
- **The test is a header-routed LOUPE:** one map per k-space width or per scanner.

## Proposed next steps (not run; for the user's verdict)

1. **A header-routed LOUPE at tier F:** one map per acquired-region class. If it closes the gap on the narrower scans,
   F's lead is routing. This is roadmap item 1.
2. **Cross-decoding,** as in kscr2's leak check (`eval_multistep_adaptivity.py cross`): F's masks through L's and Fb's
   VarNets. That separates F's masks from its VarNet's co-adaptation. It needs the checkpoints, which are on CoreHPC;
   `cross` takes `--data-root`, so it can run on either platform.
3. **A better twin:** one whose steps 0–2 get credit for their own draws. Whether that closes the twin's shortfall is
   not known, since the shortfall's cause is not separated.
4. **Seed 43,** only if one of these makes a case for it.

## Caveats

- One seed, the dev half only; the holdout half is untouched.
- The CIs hold slices and draws, not training seeds.
- FLAIR only, with a 10×10 centre and a 6 × 18 VarNet, so these numbers do not compare with kscr1–3's knee numbers.
- Order-free only. Nothing here is a SPACE gain.
