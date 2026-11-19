# Does a multi-step adaptive sampler beat its data-blind twin at R = 8? (2026-11-09 to 2026-11-19)

| | |
|---|---|
| Code | results produced at `d58087f` (multistep_adaptivity: run state 7, fscr1 SeqMRI's design on all the FLAIR data); branch `exp/multistep_adaptivity` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 35 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 375 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |
| Code drift | changed after the results were produced: `reports/multistep_adaptivity/fscr1/analysis/by_scan_type.py`, `reports/multistep_adaptivity/fscr1/analysis/render_figures.py` |

- **Branch**: `exp/multistep_adaptivity`. Each phase's results were produced at its own run state (table below); the last
  is run state 7, `d58087f`. The proposal was frozen at `55618e8`.
- **Proposal vs outcome**: the frozen decision rule (`proposal.md` section 7), then each addendum's own rule.

  | Criterion (frozen before the runs) | Number | Met |
  |---|---|---|
  | PASS: G_K* ≥ +0.10 dB pooled in both base seeds, CI above 0 (K* = 4 or 16) | largest G_K is +0.018 (K=16, seed 43) | no |
  | PASS: G_K* − G_1 ≥ +0.05 dB in both seeds | K=16: +0.002 / +0.017 | no |
  | PASS: G_K* ≥ 2 × the twin-vs-twin floor | G_16 +0.016 / +0.018 against 2 × 0.017 | no |
  | FAIL 1: G_K < +0.05 dB at K = 4 and 16 in both seeds | at most +0.018 | yes |
  | FAIL 2: the recon-input ablation G(A_4^rec) < +0.05 dB | +0.006 | yes |
  | FAIL 3: from-scratch pair's swap cost < +0.05 dB | replaced on 11-11 by the knee subproblem (`addendum_2026-11-11.md`) | not run as frozen |
  | kscr1 (11-11 addendum): swap cost clearly above the fine-tunes' +0.03 to +0.05 | +0.139 | yes |
  | kscr1: G_16 ≥ +0.10 dB pooled, to earn the confirmation pair | +0.077 | no |
  | kscr2 (11-15 addendum), row 1: S clearly above A | S − A +0.001 | no |
  | kscr2, row 2: C well above A and S (information exists, partial data hides it) | +0.247 / +0.246; kept by the leak check's rule, ceiling retracted | yes, at face value |
  | kscr3 (11-16 addendum), pilot: F − L ≥ +0.10 dB, CI above 0, same sign on PD and PDFS | +0.367 (PD +0.51, PDFS +0.22) | yes; F was data-blind (post-hoc probes) |
  | fscr1 (11-16 fscr1 addendum), primary: F − L ≥ +0.10 dB at tier F, CI above 0 | +0.109 [−0.034, +0.287] | no: "no gap at full data" |
  | fscr1: F − Fb ≥ +0.05 dB at tier F, CI above 0 | +0.250 [+0.076, +0.467] | as written; the twin trails LOUPE by 0.141 |
  | fscr1: (F − L) tier F − tier S5 ≥ +0.10 dB, CI above 0 | +0.039 [−0.136, +0.254] | no: no evidence data changes the gap |
  | fscr1: (F − Fb) tier F − tier S5 ≥ +0.05 dB, CI above 0 | +0.284 [+0.184, +0.404] | as written; same twin caveat |

  By section 7 the experiment did not pass. FAIL was not formally declared, because its third condition was replaced
  by the knee subproblem on the user's decision of 2026-11-11.
- **Recommendation and verdict**: the user's verdict on 2026-11-19 is to merge. The adaptive gain over a data-blind twin
  was not shown; the merge keeps the multi-step samplers (`seq_adapt_2d`, `seqmri_2d`), the scorer, the fixed
  dev/holdout validation halves and every phase's report on `main` for the follow-ups.

## Question

Does acquiring the R = 8 budget in K steps, each chosen from the data measured so far, give a patient-specific mask that
reconstructs better than an identical sampler that never sees the data (its data-blind twin)? The roadmap
(`AGENTS.md`, step 3) puts multi-step sampling behind modelling the SPACE echo train, so this is an order-free
go/no-go: masks are 2D sets of k-space samples, with no acquisition order or scan time. PSNR and SSIM are scored inside
the trainer's apodized per-anatomy metric mask. A contrast "X − Y" is paired per slice: positive means X is better.

## Runs compared

Every run uses an end-to-end VarNet with learned sensitivity maps and R = 8 over the acquired k-space. The detailed phase
reports are the `README.md` in each sub-folder.

| Phase (folder) | Run state | Data, scored on | Arms | Settings | Health |
|---|---|---|---|---|---|
| P1 (`p1/`) | 44c3fba | local v1 FLAIR + knee PD/PDFS, tier S; 1938 val slices | B_42, B_43: LOUPE + VarNet bases | 12 cascades, 100 epochs, LOUPE from the VD p=4 start | 33.80 / 33.81 dB, SSIM 0.863 |
| P2 (`p2/`) | 44c3fba | as P1 | A_K, T_K for K ∈ {1, 4, 16} × bases 42, 43; floor twins T_1, T_16 (seed 142); A_4^rec, A_16^img | fine-tunes of B_42/B_43, 30 epochs at lr 1e-4 | pooled SSIM 0.863–0.866 |
| kscr1 (`kscr1/`) | 707a5c9 | knee PD + PDFS from scratch; dev half, 669 slices / 20 volumes | A_16, T_16, floor T_16 (seed 142) | `seq_adapt_2d`, 12 cascades, 100 epochs, LOUPE from uniform noise | SSIM 0.864–0.865 |
| kscr2 (`kscr2/`) | b21c6b2 (leak check b8fff95) | as kscr1 | T_16, A_16, C_16 (reads full k-space), S_16 (SeqMRI-style in-loop recon), T_1 | as kscr1 at 6 cascades | 34.67–35.02 dB, SSIM 0.864–0.872 |
| kscr3 (`kscr3/`) | 0329845, lab server | knee, every 5th slice of 80 volumes (544 slices); dev half | F_4 (`seqmri_2d`, raw input), L_1 (LOUPE), Lc_1 (LOUPE per contrast) | 4 cascades, 40 × 40 centre, 100 epochs | 33.75–34.12 dB, SSIM 0.845–0.856 |
| fscr1 (`fscr1/`) | d58087f | FLAIR tier F (339 volumes, 5,139 slices) and tier S5 (238 slices); dev half, 307 slices / 20 volumes | F_4 (`seqmri_2d`, signed-log input), Fb_4 (its data-blind twin), L_1 (LOUPE), at both tiers | 6 × 18 VarNet, 10 × 10 centre, 100 epochs, lr step at 90 | 31.87–33.48 dB, SSIM 0.870–0.896 |

All runs use seed 42 unless named; P2 adds base seed 43. Checkpoints are untracked: they stay on CoreHPC scratch
(`~/scratch/multistep_adaptivity/runs`), with copies of kscr1 and kscr2 in `~/data/seqmri_checkpoints/multistep_kscr{1,2}`,
kscr3 in `~/data/seqmri_checkpoints/multistep_kscr3` and fscr1 in `/export/scratch/bkadota/multistep_adaptivity_sync/runs`.

## Methods

1. **Scoring.** `scripts/eval/eval_multistep_adaptivity.py score` draws every arm's masks with the same seeded
   thresholds (draw seeds 1234 and 4321), reconstructs with the arm's own VarNet and scores PSNR, SSIM and NMSE inside
   the metric mask. Guards: pooled SSIM ≥ 0.80, realised fraction 0.125 ± 0.002, no draw outside the acquired region.
2. **Contrasts.** `contrast` pairs two runs per slice and pools; the 95% CI resamples volumes (cluster bootstrap), so it
   covers slices and draws but not training seeds. A difference of differences, e.g. (F − L) at tier F minus (F − L) at
   tier S5, is computed on the same slices and draws.
3. **Twins.** A data-blind twin is the same sampler with a fixed tensor in place of its data channels; with the same seed
   it starts bitwise identical to its adaptive arm. G = adaptive − twin; the floor is twin − twin across seeds.
4. **Data dependence.** The swap cost is a slice's metric with its own trajectory minus with a donor slice's trajectory,
   through the same VarNet; the data-swap movement is the share of a step's draws that move when a donor's data is read.
5. **Leak check.** `cross` decodes every arm's masks through every other arm's VarNet (kscr2), to separate information
   in the masks from a VarNet reading its own masks as a code.
6. **Post hoc (not pre-registered).** `reports/multistep_adaptivity/fscr1/analysis/by_scan_type.py` splits fscr1's
   contrasts by k-space width and field strength. kscr3's probes test whether its F reads its input
   (`reports/multistep_adaptivity/kscr3/analysis/blind_probe.py`, `reports/multistep_adaptivity/kscr3/analysis/rotate_probe.py`)
   and how speckled the masks are (`reports/multistep_adaptivity/kscr3/analysis/loupe_speckle.py`,
   `reports/multistep_adaptivity/kscr3/analysis/per_scan_speckle.py`,
   `reports/multistep_adaptivity/kscr3/analysis/render_mask_compare.py`).
   `reports/multistep_adaptivity/fscr1/analysis/render_figures.py` renders fscr1's figures from the final checkpoints and
   draws the training curves from the slurm logs.
7. **Training.** CoreHPC runs went through `scripts/launch/hpc/multistep_adaptivity/train.sbatch`, which builds each arm
   from `common.sh` and scores its final checkpoint; kscr3 trained on the lab server with
   `scripts/launch/train_multistep_kscr3_local.sh`.

## Results

**P2 (fine-tunes from a static base).** G_K = A_K − T_K, pooled PSNR in dB:

| K | seed 42 | seed 43 |
|---|---|---|
| 1 | +0.014 [+0.010, +0.019] | +0.001 [−0.003, +0.005] |
| 4 | −0.001 [−0.005, +0.004] | −0.002 [−0.009, +0.005] |
| 16 | +0.016 [+0.009, +0.025] | +0.018 [+0.011, +0.026] |

The twin-vs-twin floor is 0.016 (K=1) and 0.017 (K=16), so G_16 is the size of the floor. The adaptive heads are used
(a data swap moves up to 23–25% of a step's draws; swap cost +0.03 to +0.05) but give nothing over the twin. The 16-step
twin beats the one-step twin by +0.078 / +0.062: spacing, which is data-blind.

**kscr1 and kscr2 (knee, from scratch, seed 42).** Pooled PSNR in dB:

| Contrast | kscr1 (12 cascades) | kscr2 (6 cascades) |
|---|---|---|
| G = A − T | +0.077 [+0.058, +0.094] | +0.059 [+0.043, +0.074] |
| Floor T − T′ | +0.002 [−0.013, +0.016] | not rerun |
| Swap cost of A | +0.139 [+0.104, +0.180] | +0.113 [+0.088, +0.137] |
| S − A (in-loop reconstruction) | | +0.001 [−0.021, +0.023] |
| C − A (clairvoyant, own VarNet) | | +0.247 [+0.207, +0.286] |
| C − A through T_16's VarNet (leak check) | | −0.083 [−0.150, −0.033] |
| A − T_1 (gap to one-step LOUPE) | | +0.103 [+0.080, +0.123] |
| T_16 − T_1 (its data-blind part) | | +0.044 [+0.033, +0.056] |

From scratch, the adaptive arm beats its twin by 0.06–0.08 dB, below the +0.10 bar. Reading an in-loop reconstruction
adds nothing, and the clairvoyant ceiling does not survive decoding through a VarNet that never saw its masks.

**kscr3 (SeqMRI's design, raw input, knee).** F − L +0.367 [+0.306, +0.433] and F − Lc +0.358 [+0.294, +0.422] pooled.
F's swap cost is 0.000 and a data swap moves 0.2–0.7% of its draws: its raw k-space input is about 1e-5 of the ACS
peak, so its U-Net reads only the mask. The gain is a data-blind 4-step mask against a LOUPE with about 3,400 optimizer
steps.

**fscr1 (SeqMRI's design, signed-log input, FLAIR).** Pooled PSNR in dB:

| Contrast | tier F | tier S5 | tier F − tier S5 |
|---|---|---|---|
| F − L | +0.109 [−0.034, +0.287] | +0.070 [−0.010, +0.159] | +0.039 [−0.136, +0.254] |
| F − Fb | +0.250 [+0.076, +0.467] | −0.034 [−0.131, +0.082] | +0.284 [+0.184, +0.404] |
| Fb − L (descriptive) | −0.141 [−0.216, −0.071] | +0.104 [−0.061, +0.245] | −0.245 [−0.335, −0.136] |

F − Fb = (F − L) − (Fb − L) exactly, so its pass comes from the twin trailing LOUPE, not from a gap between F and LOUPE.
Every arm gains 1.25–1.53 dB from tier S5 to tier F. The signed log makes F read its data: a data swap moves 25%, 28%,
34% and 40% of its draws in steps 0–3 at epoch 100 (kscr3: 0.2–0.7%), and its own-VarNet swap cost is +0.188
[+0.132, +0.247]. In the training curves (`fscr1/curves.json`), only F still gains after the learning-rate step at
epoch 90: mean trainer validation PSNR over epochs 91–100 minus 80–89 is +0.13 dB for F, +0.01 for L and −0.02 for Fb;
no run's validation loss turns up (minimum at epochs 93–98).

Post hoc (`fscr1/by_scan_type.txt`): the three 1.5 T AXFLAIR_202 dev volumes, whose k-space is narrower than the
320-column grid, carry F's lead (F − L +0.86 there, −0.03 on the other 17 volumes). On the figure slice from one of them,
F's final mask is 64% right of the centre column, while F on the 3 T slices (9–10%), the twin (4–6%) and LOUPE (16–17%)
stay on the left half; F at tier S5 does not switch (22%).

## Conclusions

- At R = 8, no multi-step sampler showed an adaptive gain over its matched data-blind twin by the frozen bars: G ≤ +0.018
  dB fine-tuned (P2, the size of the floor) and +0.06 to +0.08 dB from scratch (kscr1, kscr2; one seed, knee).
- Fine-tuned (P2), nearly all of the multi-step gain over one-step LOUPE is data-blind spacing: T_16 − T_1 is +0.078 /
  +0.062 of A_16 − T_1 = +0.094 / +0.081. From scratch (kscr2), spacing is +0.044 of the +0.103 gap and the adaptive
  part, A − T, is +0.059.
- SeqMRI's design reads its data only when the input is scaled to be readable (fscr1's signed log against kscr3's raw
  input). Reading the data did not give a gap over LOUPE at full data (fscr1 primary, CI includes 0).
- More data and longer training raised every arm by 1.25–1.53 dB and did not change F's lead over LOUPE (+0.039, CI
  includes 0).
- Hypothesis, untested: F's fscr1 lead is scan-type routing for the 1.5 T narrow-k-space scans; a header-routed LOUPE
  is the test.
- Hypothesis, untested: F is less converged than L and Fb at epoch 100, or its sampler learning rate was too high until
  the step at epoch 90. fscr1 trained F's sampler with Adam at 1e-3; the SeqMRI paper states 1e-3 for 2D, but the
  sequential commands in its code's README use 5e-5 (with 1e-3 for LOUPE).

## Caveats

- One seed in kscr1–fscr1; the CIs cover slices and draws, not training seeds, whose spread for joint mask and VarNet
  training is 0.3–0.5 dB.
- P2's arms fine-tune from bases whose LOUPE masks started from the VD p=4 prior; the from-scratch phases start from
  uniform noise.
- kscr3 differs from the other phases in cascades (4), centre (40 × 40) and data (20% of slices), and its LOUPE was not
  converged.
- fscr1's twin trails LOUPE (the credit-route caveat stated before the runs), so F − Fb there is not adaptivity.
- Knee-PDFS SSIM is 0.78–0.81 per condition in P1, kscr2 and kscr3, against the 0.80 line; the pooled guard passes and the
  value is equal across arms. In fscr1, 27–41 of 307 slices per run have SSIM below 0.80, mostly the noisy reference of
  AXFLAIR_202_6000450; pooled SSIM is 0.870–0.896.
- The trainer's validation PSNR uses the last batch's data range and reads 1.2–1.4 dB above the scored PSNR; the curves
  compare runs within a panel only.
- fscr1's figures were rendered after the runs (2026-11-18) from the final checkpoints on the lab server: PSNR is within
  0.03 dB of the scored values and F's per-step sample counts within 2% (`fscr1/render_checks.json`).
- The scan-type split, the side switch and the per-epoch windows were chosen after looking at the results.
- Order-free only: no claim about SPACE until echo-train evolution is modelled.

## Figures

All in `fscr1/figures/`, rendered after the runs from fscr1's final checkpoints on the scorer's three figure slices
(`brain_q25`, `brain_q50`, `brain_q75`: the middle slice of the dev volumes at those quantiles of the sorted names).

- `curves.png`: trainer validation PSNR, validation loss and training loss per epoch for the six fscr1 runs, tier F and
  tier S5.
- `recon_<slice>_tF.png`: the reference, each tier-F arm's reconstruction, a zoom and the error inside the metric mask.
- `sampling_<slice>_tF.png`: the fully sampled k-space, LOUPE's learned map, each tier-F arm's final mask and its local
  sampling density.
- `steps_F_tF_<slice>.png`: tier-F F's four steps: draw probability, new samples, cumulative mask, in-loop
  reconstruction and error.
- `density_tF.png`, `density_tS5.png`: each arm's sampling density on the three slices side by side.
- `render_figures.py plot --all-figures` also writes tier S5's recon and sampling figures and the twin's and LOUPE's step
  grids (not committed, to keep the folder small). kscr3's probe figures go to `~/seqmri_figures/` (untracked).

## Next steps

- Resume all three fscr1 tier-F arms for 50 more epochs at lr 1e-4, and retrain F (with its twin) at a lower sampler
  learning rate, to separate step count from learning rate.
- Give F and its twin a learned per-pixel (per-step) logit bias, so that both contain LOUPE and the twin can no longer
  trail it.
- Port `scripts/diagnostics/probe_conjugate_symmetry.py` to these checkpoints, to test whether the one-sided masks use
  conjugate symmetry (partial Fourier).
- A header-routed LOUPE (one map per k-space width or scanner) against F on the 1.5 T scans.

## Reproduce

`bash reports/multistep_adaptivity/run.sh` from the repository root regenerates the contrasts, the leak check, the
post-hoc analyses and the fscr1 figures from the score files here. Section 1 (`RUN_COREHPC=1`, on CoreHPC) retrains and
rescores every run; the leak check and the fscr1 render stage need the checkpoints named above and GPU 0. Check the
folder with `python scripts/report_manifest.py check reports/multistep_adaptivity`.
