# Is SeqMRI's sampler held back by its training or by its parametrisation? (2026-11-19 to 2026-11-22)

| | |
|---|---|
| Code | results produced at `e8d6d39` (multistep_adaptivity__seqmri_lr_bias_r8: the report's analysis script and run.sh); branch `exp/multistep_adaptivity__seqmri_lr_bias_r8` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 12 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 74 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Branch**: `exp/multistep_adaptivity__seqmri_lr_bias_r8` (renamed 2026-11-20 from `exp/seqmri_lr_bias_r8`). Proposal frozen at
  `c9d2206`; arm B replaced before it ran by `addendum_2026-11-19_route.md`. Every run trained at run state 3 (`49be699`,
  which carries the Muon resume fix `86b9a22`) and was scored by `train.sbatch` on the same card right after training.
- **Proposal vs outcome** (seed-42 screen; pass = at least +0.10 dB with the 95% volume-bootstrap CI above 0):

  | Row (frozen before the runs) | Contrast | Number | Met |
  |---|---|---|---|
  | R1: longer training opens F's lead | (F − L) at epoch 150 minus at epoch 100 | +0.056 [+0.035, +0.081] | no: positive, below +0.10 |
  | S1: the sampler rate held F back | F_lrs − F | +0.042 [+0.020, +0.064] | no: positive, below +0.10 |
  | B1: per-step per-pixel maps help F | F_routebias − F_route | +0.065 [+0.029, +0.097] | no: positive, below +0.10 |
  | B2: reading the data pays over a twin with the same maps | F_routebias − Fb_routebias | +0.009 [−0.123, +0.173] | no: data-blind |
  | B3: the twin no longer trails LOUPE | Fb_routebias − L | +0.013 [−0.048, +0.072] | yes (CI includes 0) |
  | Guards: pooled SSIM ≥ 0.80 and within 0.02 of the control; realised 0.125 ± 0.002; no draw outside the acquired region | | SSIM 0.889–0.897; realised 0.1249–0.1251; 0 draws outside | yes |

- **Recommendation**: park. None of the three explanations is worth +0.10 dB on its own, so no arm earns seed 43. With
  a twin that is credited at every step and has per-pixel maps, F's adaptive gain is zero (B2). The one large secondary
  gap (F_lrs − Fb_lrs +0.14) is a twin that still trails LOUPE on full-width volumes, as in fscr1. The verdict is the
  user's.

## Question

In fscr1 (`reports/multistep_adaptivity/fscr1/README.md`), the 4-step SeqMRI sampler F was +0.11 dB over LOUPE with a CI
including 0, and its +0.25 dB lead over its data-blind twin was the twin trailing LOUPE. F was the only arm still
improving after the learning-rate step at epoch 90. This experiment asks which of three things held F back: too few
epochs (arm R), a sampler learning rate 20× the original SeqMRI code's (arm S), or a sampler with no per-pixel parameters
(arm B). PSNR is the scorer's pooled PSNR on the FLAIR dev half, in dB; differences are paired per slice.

## Runs compared

All runs: FLAIR tier F (339 volumes, 5,139 training slices), R = 8, 6 × 18 VarNet, 10 × 10 calibration centre, seed 42,
step schedule 1e-3 then ×0.1 at epoch 90, CoreHPC RTX PRO 6000. F = `seqmri_2d` 4-step sampler with signed-log input;
Fb = its data-blind twin; L = one-step LOUPE.

| Run | Change from its fscr1 control | Epochs | Wall time | Pooled PSNR | SSIM | NMSE |
|---|---|---|---|---|---|---|
| F_4_s42_fscr1_tF (control) | — | 100 | 19.1 h | 33.476 | 0.8962 | 0.0043 |
| Fb_4_s42_fscr1_tF (control) | — | 100 | 6.4 h | 33.226 | 0.8901 | 0.0048 |
| L_1_s42_fscr1_tF (control) | — | 100 | 4.1 h | 33.367 | 0.8908 | 0.0046 |
| F_4_s42_fscr1_tF_r150 | resumed at 1e-4, optimizer state included | 150 | +9.5 h | 33.512 | 0.8972 | 0.0043 |
| Fb_4_s42_fscr1_tF_r150 | resumed | 150 | +3.2 h | 33.146 | 0.8892 | 0.0049 |
| L_1_s42_fscr1_tF_r150 | resumed | 150 | +2.1 h | 33.347 | 0.8906 | 0.0046 |
| F_4_s42_lrs_tF | sampler in its own group at 1e-4 (1e-5 after epoch 90) | 100 | 19.0 h | 33.518 | 0.8969 | 0.0043 |
| Fb_4_s42_lrs_tF | same | 100 | 6.4 h | 33.380 | 0.8952 | 0.0045 |
| F_4_s42_route_tF | `sampler.seqmri_union_route` (every step's draws credited) | 100 | 19.0 h | 33.324 | 0.8937 | 0.0045 |
| Fb_4_s42_route_tF | same | 100 | 9.0 h | 33.276 | 0.8900 | 0.0047 |
| F_4_s42_routebias_tF | route + one learned 320 × 320 logit map per step | 100 | 18.9 h | 33.389 | 0.8941 | 0.0044 |
| Fb_4_s42_routebias_tF | same | 100 | 9.1 h | 33.380 | 0.8921 | 0.0046 |

All models are healthy (SSIM 0.889–0.897). The validation jobs (`jobs.txt`) checked before training that each resume
restarts at lr 1e-4 in every optimizer group with the loss continuing, that every arm reads its data or (twins) does not,
and that every run records the shipped commit. F_lrs's log shows three lr groups, 0.001, 0.001 and 0.0001.

## Methods

1. **Scores.** `scripts/eval/eval_multistep_adaptivity.py score --gradients` on the FLAIR dev half (20 volumes, 307
   slices, hash 1874f6fc), two draw seeds (1234, 4321), run by `train.sbatch` after training. The fscr1 controls' scores
   (`d58087f`) are reused; the pre-launch check scored F_tF with run-state code bit-identically on 24 slices × 2 draws.
2. **Contrasts.** `contrast` subcommand: per-slice paired differences of pooled PSNR, 95% CI from resampling whole
   volumes (2,000 draws). R1 is a difference of differences. Decision rows in `contrasts.json`, secondary contrasts in
   `contrasts_secondary.json`.
3. **Scan-type split** (post hoc, not in the proposal): the same bootstrap within volumes whose k-space is narrower than
   the 320-column grid (9) against full-width ones (11), and 1.5 T (6) against 3 T (14), from the manifest
   (`reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/summarise.py tables`, `by_scan_type.txt`).
4. **Data-swap movement and swap cost** (descriptive): the scorer's share of each step's draws that moves when another
   slice's data is fed in, and the PSNR a run loses decoding a donor's mask (`summary.txt`).
5. **Curves**: the trainer's per-epoch validation PSNR (its own metric, about 1.4 dB above the scorer's) and training
   loss, parsed from the slurm logs in `logs/` and fscr1's (`curves.json`, `figures/curves.png`).

## Results

**Decision rows and secondary contrasts** (`figures/contrasts.png`):

| Contrast | PSNR difference (dB) | Reading |
|---|---|---|
| R1 (F − L) at 150 minus at 100 | +0.056 [+0.035, +0.081] | F +0.036 from 100 to 150; L −0.020 |
| S1 F_lrs − F | +0.042 [+0.020, +0.064] | small |
| B1 F_routebias − F_route | +0.065 [+0.029, +0.097] | small |
| B2 F_routebias − Fb_routebias | +0.009 [−0.123, +0.173] | no adaptive gain |
| B3 Fb_routebias − L | +0.013 [−0.048, +0.072] | the twin matches LOUPE |
| F_route − F (the route alone) | −0.152 [−0.191, −0.109] | the route costs F |
| Fb_route − Fb (the route alone) | +0.050 [+0.023, +0.076] | and helps the twin a little |
| Fb_routebias − Fb_route (maps on the twin) | +0.104 [+0.065, +0.150] | the maps help the twin |
| Fb_r150 − Fb | −0.080 [−0.098, −0.064] | the twin loses with more epochs |
| Fb_lrs − Fb | +0.154 [−0.023, +0.369] | the lower sampler rate helps the twin |
| F_r150 − L_r150 | +0.165 [+0.001, +0.364] | |
| F_lrs − L | +0.151 [−0.003, +0.336] | |
| F_lrs − Fb_lrs | +0.138 [+0.100, +0.174] | see the split below |
| Fb_lrs − L | +0.013 [−0.128, +0.186] | |

The three rows that test an explanation of F's late gain are all positive with CIs above 0, and all below the +0.10 bar.
The route costs F 0.15 dB, so F_routebias (33.389) ends below fscr1's F (33.476), and the per-step maps then recover
less than half of that.

**By scan type** (post hoc, `by_scan_type.txt`):

| Group | Volumes | F_lrs − L | F_lrs − Fb_lrs | Fb_lrs − L | F_r150 − L_r150 | F_routebias − Fb_routebias |
|---|---|---|---|---|---|---|
| narrower than the grid | 9 | +0.439 [+0.208, +0.713] | +0.190 [+0.138, +0.240] | +0.249 [+0.004, +0.527] | +0.451 [+0.178, +0.769] | +0.154 [−0.124, +0.460] |
| full width | 11 | −0.088 [−0.116, −0.059] | +0.094 [+0.061, +0.130] | −0.182 [−0.213, −0.151] | −0.072 [−0.097, −0.047] | −0.112 [−0.143, −0.078] |
| 1.5 T | 6 | +0.528 [+0.179, +0.886] | +0.195 [+0.121, +0.266] | +0.333 [−0.026, +0.689] | +0.591 [+0.192, +0.975] | +0.324 [−0.013, +0.672] |
| 3 T | 14 | −0.013 [−0.081, +0.065] | +0.113 [+0.079, +0.147] | −0.126 [−0.177, −0.070] | −0.020 [−0.071, +0.038] | −0.128 [−0.149, −0.103] |

Every F lead over LOUPE comes from the narrower-than-grid, mostly 1.5 T volumes; on full-width volumes F trails LOUPE by
0.07–0.09 dB, as fscr1's F did (−0.10). On full width, F_lrs's +0.09 over its twin is the twin's 0.18 dB deficit to
LOUPE less F's own 0.09.

**Data swap and swap cost** (`summary.txt`, `figures/data_swap.png`):

| Run | Draws moved by a data swap, steps 0–3 | Swap cost (dB) |
|---|---|---|
| F (fscr1) | 0.25, 0.28, 0.34, 0.40 | +0.188 [+0.132, +0.247] |
| F_r150 | 0.23, 0.25, 0.33, 0.40 | +0.190 [+0.134, +0.248] |
| F_lrs | 0.33, 0.31, 0.31, 0.35 | +0.206 [+0.151, +0.264] |
| F_route | 0.32, 0.28, 0.32, 0.27 | +0.059 [+0.042, +0.077] |
| F_routebias | 0.21, 0.13, 0.25, 0.21 | +0.079 [+0.061, +0.097] |

Every F reads its data; every twin moves nothing (swap null 1.0). Under the route F moves as many draws (F_route) or
fewer (F_routebias), and a donor's mask costs it a third to two-fifths as much.

**Curves** (`figures/curves.png`): resumed F keeps climbing slowly to epoch 150 while L is flat and the twin falls
steadily (its training loss keeps dropping, its validation PSNR drops 0.1 dB). Both F route arms are still rising at
epoch 100 and their training loss is higher and spikier than F's (F_route spikes near epochs 42–48 and 87).

## Conclusions

- More epochs, a lower sampler rate, and per-step per-pixel maps each add 0.04–0.07 dB to F. None reaches the
  +0.10 dB bar, so none alone explains the gap the proposal asked about (R1, S1, B1).
- Given per-pixel maps and credit at every step, the data-blind twin catches up with LOUPE (B3: +0.013) and F no longer
  beats it (B2: +0.009). Pooled, there is no adaptive gain over a twin of equal strength at R = 8.
- The route alone costs F 0.15 dB at 100 epochs while F's curves are still rising. Hypothesis, untested: the route's
  extra gradient paths slow F's convergence rather than lower its ceiling.
- The twin is sensitive to how it is trained: the lower sampler rate (+0.15, CI includes 0) and the per-step maps (+0.10)
  both help it, and more epochs hurt it (−0.08). fscr1's twin was a weak control.
- F's lead over LOUPE sits in the 9 narrower-than-grid volumes (and the 6 at 1.5 T); on full-width volumes F_lrs and
  F_r150 trail LOUPE. Hypothesis, untested: this is routing by scan type (k-space width / scanner), which a per-header
  LOUPE could match, not patient-specific adaptivity.

## Caveats

- One training seed. The CIs cover slices and volumes, not training; F − L moves 0.3–0.5 dB between seeds and twin
  against twin about 0.05 dB. S1 and B1 compare separately trained runs, so their +0.04 and +0.07 are inside that floor.
  R1 continues the same runs, so it is not.
- The route arms are under-converged at 100 epochs (curves still rising), so B1–B3 compare F and the twin at a budget
  where the route has cost F 0.15 dB. A longer schedule could change B1 and B2.
- The scan-type split was chosen after looking (it repeats fscr1's post hoc split); with 6–9 volumes per group its CIs
  are wide.
- The fscr1 controls were scored at `d58087f` and the new runs at `49be699`. The scorer is unchanged, and run-state code
  scored F_tF bit-identically on a 24-slice subset before launch; the full 307 slices were not rescored.
- The trainer's validation PSNR in the curves is about 1.4 dB above the scorer's (different data range); compare
  shapes, not levels, with the tables.
- The proposal's other descriptive readouts (each map's high-frequency share and right-of-centre share) were not
  computed; they need the checkpoints rendered on a GPU.
- Training logged to wandb offline on CoreHPC; those runs are not synced yet. Checkpoints are untracked (CoreHPC
  `~/scratch/multistep_adaptivity__seqmri_lr_bias_r8{,_rs2}/runs`).

## Figures

- `figures/curves.png`: trainer validation PSNR and training loss per epoch for each arm against its fscr1 control.
- `figures/contrasts.png`: the decision rows and secondary contrasts with CIs against the +0.10 dB bar.
- `figures/data_swap.png`: the share of each step's draws a data swap moves, for each F run.
- `figures/scorer/<run>/brain_q{25,50,75}.png`: the scorer's step grids on three dev slices (volumes at the 25th, 50th
  and 75th PSNR percentile, draw seed 1234): per step the data seen, the rescaled sampling probability, the loss
  gradient, the new draws, the mask, the zero-filled image, the reconstruction and its error map. Nine runs.

## Next steps

- Header-routed LOUPE: one map per k-space width or scanner, to test whether F's narrow-volume lead is scan-type
  routing (the cheapest explanation of the only place F wins).
- If adaptivity is pursued further, use a twin trained like Fb_lrs or Fb_routebias, which matches LOUPE pooled, and
  stratify every contrast by k-space width.
- To judge the route fairly, train F_route and F_routebias past 100 epochs (or until the validation curve flattens).
- Sync the offline wandb runs.

## Reproduce

`bash reports/multistep_adaptivity__seqmri_lr_bias_r8/run.sh` from the repository root regenerates the contrasts, tables
and figures from `scores/` without a GPU. Rescoring needs the checkpoints on CoreHPC and a GPU (command in `run.sh`).
Check with `python scripts/report_manifest.py check reports/multistep_adaptivity__seqmri_lr_bias_r8`.
