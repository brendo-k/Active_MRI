# Do wider U-Nets help a brain VarNet at 80 volumes, and is depth or width the better use of parameters? With a tier-L extension at 300 volumes (2026-11-15, 2026-11-16)

| | |
|---|---|
| Code | results produced at `c7d6ed2` (brain_varnet_width tier L: summary script, Slurm mail on END/FAIL); branch `exp/brain_varnet_width` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 27 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 117 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

**Status: merged (user verdict 2026-11-16). Reconstructor: 6×18, also for tier-L/F work for now.**

- **Branch:** `exp/brain_varnet_width`.
  - Proposal frozen at `5dcbeb8` (`reports/brain_varnet_width/proposal.md`).
  - Seed-42 runs trained at `c19ef3b`; seed-43 runs and all tier-S scoring at `5c2fb4e`.
  - Tier-L runs trained at `c334d89` (CoreHPC); tier-L scoring and summary at `c7d6ed2`.
  - The 6×18 and 12×18 cells are the 80-volume runs of `reports/brain_varnet_scaling/report.md`, trained at `35f8f2f`
    (c12_v80_s42, no early stopping) or `8485567`.

**Proposal vs outcome** (full brain val PSNR at `best_model.pt`; differences are paired per slice, 95% bootstrap CI
over volumes):

| Frozen criterion | Outcome | Met? |
|---|---|---|
| Validity: every run finishes, SSIM ≥ 0.80, no NaNs or skipped steps | 13/13 new runs finished; SSIM 0.954–0.957; 0 skipped steps | met |
| Q1 wider helps: best 26/36-channel cell beats 18 at the same cascades by ≥ 0.15 dB in both seeds | best wider − 18: −0.07 (3 casc.), −0.01 / −0.04 (6 casc., s42 / s43), −0.09 (12 casc.); never positive | not met |
| Q1 width saturated: \|Δ\| < 0.05 dB | 6 cascades, both seeds: −0.015 [−0.05, +0.02], −0.038 [−0.06, −0.01]. At 3 and 12 cascades (s42) wider is 0.07–0.09 dB *worse* | met at 6 cascades; elsewhere wider is slightly worse |
| Q1 18 needed: 12 channels loses ≥ 0.15 dB to 18 (s42, 6 and 12 cascades) | 6 casc. −0.19 [−0.25, −0.14]; 12 casc. −0.07 [−0.10, −0.05] (s43: +0.09 [+0.07, +0.12]) | not met |
| Q2 depth wins: 6×26 and 12×18 both beat 3×36 by ≥ 0.15 dB in both seeds | 6×26 − 3×36: +0.29 / **+0.10**; 12×18 − 3×36: +0.25 / +0.16 | not met (6×26 in s43 is 0.05 short) → **inconclusive** |
| Q2 width wins / size alone | 3×36 is last in all four comparisons; all CIs exclude 0 | not met |
| Exploratory: equal-size cells within 0.1 dB | spread 0.31 (15 M, s42), 0.11 (15 M, s43, two cells), 0.29 / 0.16 (30 M), 0.01 (60 M) | no: 3 cascades trail at 15 M and 30 M |
| Expected: 26/36 channels add ≤ +0.1 dB; 12 channels lose 0.2–0.4 dB | wider adds −0.41 to −0.01; 12 channels lose 0.07–0.19 (s42), gain 0.09 at 12 casc. (s43) | width met; 12-channel loss smaller than predicted |
| Expected: 3×36 trails 6×26 ≈ 12×18 by 0.1–0.3 dB | 3×36 trails by 0.10–0.29; 6×26 vs 12×18: −0.04 (s42), +0.06 (s43) | met |

**Frozen next step:** the cheapest cell within 0.05 dB of the best becomes the reconstructor for the static vs learned
mask rerun. On the two-seed means, the best is **6×18** (37.18 dB), which is also the cheapest of the cells within
0.05 dB (12×12 37.17, 12×18 37.16) at 0.82 s per step. So the rerun keeps 6×18.

**Recommendation: merge.**
- Q1 has a clear answer even though the frozen rule calls only the 6-cascade part "saturated": at 80 volumes no width
  above 18 channels helps at any depth, and the widest models are worse.
- Q2 points the same way in all four seed × pair comparisons (more cascades beat wider, CIs exclude 0), but misses the
  pre-set 0.15 dB bar once (6×26 vs 3×36 in s43, +0.10). Record it as inconclusive by the rule, direction consistent.
- Nothing changes on main's defaults. The branch adds the width launcher, a worker that claims a job only once its GPU
  has room, and the eval and summary scripts.

**Tier-L extension (2026-11-16).** The addendum `reports/brain_varnet_width/proposal_tier_l.md` was frozen at `70ddf1c`
before any tier-L run. Runs trained on CoreHPC at `c334d89`; the summary script is committed at `c7d6ed2`.
- Cells {6, 12} cascades × {18, 26} channels on tier L (300 FLAIR volumes), seeds 42 and 43.
- 6×18 on tier S on CoreHPC, as the platform control.

| Frozen criterion (tier L) | Outcome | Met? |
|---|---|---|
| Validity: every run finishes, SSIM ≥ 0.80, no NaNs or skipped steps | 10/10 finished; SSIM 0.955–0.959; 0 non-finite losses, 0 skipped steps | met |
| Platform control: CoreHPC 6×18 tier S within 0.10 dB of local (37.23 s42, 37.13 s43) | −0.23 [−0.27, −0.19] (s42), −0.09 [−0.11, −0.06] (s43) | **not met** (s42) |
| Q3 capacity pays: best of 6×26 / 12×18 / 12×26 beats 6×18 at L by ≥ 0.15 dB in both seeds | best is 12×18: +0.04 [+0.02, +0.06] / +0.06 [+0.05, +0.08] | not met |
| Q3 still saturated: all three within 0.05 dB of 6×18 in both seeds | 6×26 s43 −0.08; 12×18 s43 +0.06 | not met → **inconclusive** |
| Q4 data still pays: 6×18 L − 6×18 S (CoreHPC) ≥ +0.3 dB in both seeds | +0.46 [+0.40, +0.53] / +0.45 [+0.40, +0.50] | **met** |
| Expected: data +0.4 to +0.8 dB; capacity 0 to +0.15 dB | +0.45 / +0.46; capacity −0.08 to +0.06 | met |

Because the platform control failed, the rule says:
- data-gain statements use CoreHPC against CoreHPC only, which is what Q4 above does;
- the gap-growth table below, which compares cells across tiers, is confounded by platform.

**Frozen next step (tier L):** the cheapest cell within 0.05 dB of the best two-seed mean at L. The best is 12×18
(37.53 dB); 12×26 (37.50) is also within 0.05; the cheaper of the two by wall-clock is **12×18**.
- 12×18 costs 140 s per epoch against 94 s for 6×18 (1.48×).
- This is a knife-edge:
  - 6×18 (37.477) misses the 37.480 cut-off by 0.003 dB.
  - At `last_model.pt`, 6×18 and 12×18 tie at 37.53 dB, and the rule would pick 6×18.
- The pick is reported as the rule gives it; the choice of reconstructor for tier-L work is the user's.
- **User decision (2026-11-16): use 6×18 for now**, overriding the knife-edge rule pick.

**Recommendation (tier L):** fold into the merge.
- More data clearly helps: +0.45 dB from 80 to 300 volumes, both seeds, same platform.
- Extra capacity buys at most +0.06 dB at 300 volumes. That is a small, consistent gain for 12×18 at `best_model`,
  and it vanishes at `last_model`.
- Suggestion: use 6×18 when cost matters and 12×18 when the last 0.05 dB matters. The user decides.

## Question
`reports/brain_varnet_scaling/report.md` found that, at 80 training volumes, adding cascades stops helping at about 6.
This experiment varies the second size knob, U-Net width (`recon.chans`, the first level's channels, doubling over 4
pooling levels, so parameters per cascade grow with chans²), and asks (1) whether wider cascades help, and (2) at equal
parameter count, whether parameters are better spent on more cascades or on wider ones. The answer picks the
reconstructor for the static vs learned brain mask rerun (roadmap step 1, anatomy routing). PSNR is in dB on the full
brain val set; higher is better.

## Runs compared
All runs: E2E VarNet, brain FLAIR only, local v1 tier S, all 80 training volumes (1,211 slices), frozen shared mask S
(`loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt`), R=8 of acquired, Muon + LayerNorm, L1 on RSS, batch 16,
lr 1e-3 dropping 10× at epoch 90 of 100 (7,600 steps), validation every 5 epochs on the pinned 10-volume brain subset,
early-stop patience 5. 12×26 and 12×36 use `recon.checkpoint_cascades=true` (recomputes activations in backward;
changes speed and memory only). Checkpoints: `checkpoints/bvw_c{C}_ch{K}_s{seed}` (new) and
`checkpoints/bvs_c{6,12}_v80_s{42,43}` (reused 18-channel cells).

| Cascades × chans | Params (M) | s / step | Stopped at step | Seed 42 PSNR | Seed 43 PSNR | SSIM | Train − val gap (dB) |
|---|---|---|---|---|---|---|---|
| 3×12 | 4.1 | 0.59 | 7,600 | 36.83 | — | 0.954 | 0.72 |
| 3×18 | 8.2 | 0.63 | 7,600 | 36.99 | — | 0.955 | 0.77 |
| 3×26 | 16.2 | 0.67 | 7,600 | 36.92 | — | 0.955 | 1.12 |
| 3×36 | 30.3 | 0.78 | 6,080 / 6,080 | 36.92 | 36.99 | 0.955 | 0.86 / 0.84 |
| 6×12 | 7.4 | 0.72 | 7,600 | 37.04 | — | 0.956 | 1.04 |
| 6×18 (reused) | 15.5 | 0.82 | 6,840 / 6,460 | **37.23** | 37.13 | 0.957 | 0.80 / 0.80 |
| 6×26 | 31.6 | 0.92 | 5,320 / 7,600 | 37.21 | 37.09 | 0.957 | 0.77 / 1.13 |
| 6×36 | 59.7 | 1.12 | 6,080 | 37.06 | — | 0.956 | 1.14 |
| 12×12 | 13.9 | 1.01 | 6,080 / 7,600 | 37.10 | **37.25** | 0.956 / 0.957 | 0.75 / 1.07 |
| 12×18 (reused) | 30.3 | 1.21 | 7,600 / 7,600 | 37.17 | 37.15 | 0.956 | 1.11 / 1.55 |
| 12×26 | 62.3 | 1.95 | 5,700 | 37.08 | — | 0.956 | 1.09 |
| 12×36 | 118.7 | 2.47 | 6,460 | 36.76 | — | 0.954 | 1.50 |

Two values are seed 42 / seed 43. s / step is the training wall-clock per optimiser step on one A6000 (seed 42). All
SSIM values are in the healthy 0.95 range.

**Tier-L runs** (CoreHPC, one RTX Pro 6000 each, `scripts/launch/hpc/brain_varnet_width_l/run.sbatch`, array job
3079379).
- Same recipe as above except the training set:
  - tier L is 300 FLAIR volumes (4,545 slices), the 80 tier-S volumes plus 220 more;
  - 100 epochs is about 28,400 steps.
- Python 3.14 and torch 2.12 on CoreHPC, against Python 3.13 locally.
- Checkpoints are under `checkpoints/bvwl_{L,S}_c{C}_ch{K}_s{seed}`.
- Seeds are given as 42 / 43.
- Train − val gap: train PSNR on the 80 tier-S volumes minus full-val PSNR, at `best_model.pt`.
- Scoring: `scripts/eval/brain_varnet_width_eval.py --local-data`. Criteria, tables and figure:
  `scripts/eval/brain_varnet_width_tier_l_summary.py`.

| Cell | Tier | Params (M) | s / epoch | Stopped at epoch | PSNR best | PSNR last | SSIM | Train − val gap |
|---|---|---|---|---|---|---|---|---|
| 6×18 | L | 15.5 | 94 | 85 / 100 | 37.46 / 37.49 | 37.55 / 37.51 | 0.959 / 0.958 | 0.69 / 0.81 |
| 6×26 | L | 31.6 | 105 | 80 / 85 | **37.51** / 37.42 | 37.47 / 37.35 | 0.959 / 0.958 | 0.78 / 0.83 |
| 12×18 | L | 30.3 | 140 | 70 / 65 | 37.50 / **37.56** | 37.53 / 37.54 | 0.959 / 0.959 | 0.78 / 0.73 |
| 12×26 | L | 62.3 | 159 | 95 / 55 | 37.47 / 37.53 | 37.32 / 37.56 | 0.958 / 0.959 | 1.32 / 0.67 |
| 6×18 | S (CoreHPC) | 15.5 | 28 | 60 / 75 | 37.00 / 37.04 | 37.11 / 37.11 | 0.955 / 0.955 | 0.64 / 0.79 |

## Methods
1. **Training:** `scripts/launch/queue_brain_varnet_width.sh` queued the 10 new seed-42 cells on GPUs 0–3, then the
   frozen seed-43 picks (3×36 and 6×26 always; the top two new seed-42 cells, 6×26 and 12×12, so 12×12 was added).
2. **Scoring:** `scripts/eval/brain_varnet_width_eval.py` scores `best_model.pt` and `last_model.pt` on the full brain
   val set (40 volumes, 605 slices) and on the training slices. Mask S is thresholded with common per-slice Bernoulli
   draws (seeds 1234 and 4321, averaged), so every run sees the same mask on each slice; scores use the trainer's
   apodized metric mask. The reused cells' score files come from `reports/brain_varnet_scaling/eval/`, made with the
   same recipe.
3. **Summary:** `scripts/eval/brain_varnet_width_summary.py` builds the table, the per-slice paired differences with
   95% bootstrap CIs over volumes (resampling volumes, not slices), the frozen Q1/Q2 verdicts, the equal-size
   diagonals and the figures. Train − val gap = train PSNR minus full-val PSNR at `best_model.pt`.

## Results

**Tier L, capacity against 6×18** (paired differences at L, dB; both arms on CoreHPC):

| Cell − 6×18 | s42, best | s43, best | s42, last | s43, last |
|---|---|---|---|---|
| 6×26 | +0.05 [+0.02, +0.07] | −0.08 [−0.11, −0.05] | −0.08 [−0.11, −0.06] | −0.17 [−0.19, −0.14] |
| 12×18 | +0.04 [+0.02, +0.06] | +0.06 [+0.05, +0.08] | −0.03 [−0.05, −0.01] | +0.03 [+0.01, +0.05] |
| 12×26 | +0.01 [−0.03, +0.05] | +0.03 [+0.01, +0.06] | −0.23 [−0.28, −0.19] | +0.05 [+0.04, +0.07] |

- 12×18 is the only cell ahead of 6×18 in both seeds at `best_model.pt`, by 0.04–0.06 dB.
- Wider at 6 cascades flips sign with the seed.
- No gain survives at `last_model.pt` in both seeds.

**Tier L, data and platform** (paired, dB):

| Comparison | s42 | s43 |
|---|---|---|
| 6×18 L − 6×18 S, both on CoreHPC (Q4) | +0.46 [+0.40, +0.53] | +0.45 [+0.40, +0.50] |
| 6×18 L (CoreHPC) − 6×18 S (local), confounded | +0.23 | +0.36 |
| 6×18 S, CoreHPC − local, `best_model.pt` | −0.23 [−0.27, −0.19] | −0.09 [−0.11, −0.06] |
| 6×18 S, CoreHPC − local, `last_model.pt` | −0.09 [−0.11, −0.07] | −0.03 [−0.04, −0.01] |

- 300 volumes add +0.45 dB over 80 volumes on the same platform.
- The platform gap is three times smaller at `last_model.pt` than at `best_model.pt`.

**Exploratory: does capacity gain more with data?**
- Measure: (X − 6×18) at L (CoreHPC) minus (X − 6×18) at S (local).
- Confounded by platform, though the platform offset cancels to first order inside each tier.
- Results:
  - 6×26: +0.06 (s42), −0.04 (s43);
  - 12×18: +0.10 (s42), +0.04 (s43);
  - 12×26: +0.16 (s42; no S run in s43).
- Depth gains a little relative to 6×18 as data grows; width at 6 cascades does not.

**Exploratory: equal steps** (trainer val L1 on the 10-volume subset, lower is better).
- Every tier-L cell at epoch 25 (about 7,100 steps) is at 0.0296–0.0302.
- Every local tier-S run at its best (≤ 7,600 steps) is at 0.0310–0.0319.
- So at the same number of steps, more data already wins; the tier-L gain is not just longer training.

Figure: `figures/tier_l_cells.png`.

**Width at fixed depth** (wider minus 18 channels, dB, seed 42 unless marked):

| Cascades | 12 − 18 | 26 − 18 | 36 − 18 |
|---|---|---|---|
| 3 | −0.16 [−0.21, −0.11] | −0.08 [−0.09, −0.06] | −0.07 [−0.11, −0.04] |
| 6 | −0.19 [−0.25, −0.14] | −0.01 [−0.05, +0.02]; s43 −0.04 [−0.06, −0.01] | −0.16 [−0.21, −0.12] |
| 12 | −0.07 [−0.10, −0.05]; s43 +0.09 [+0.07, +0.12] | −0.09 [−0.12, −0.07] | −0.41 [−0.47, −0.36] |

No width above 18 channels beats 18 at any depth; 36 channels is clearly worse at 6 and 12 cascades.

**Depth vs width at about 30 M parameters** (paired differences, dB):

| Pair | Seed 42 | Seed 43 |
|---|---|---|
| 6×26 − 3×36 | +0.29 [+0.23, +0.36] | +0.10 [+0.06, +0.15] |
| 12×18 − 3×36 | +0.25 [+0.19, +0.32] | +0.16 [+0.08, +0.25] |
| 12×18 − 6×26 | −0.04 [−0.07, −0.01] | +0.06 [+0.01, +0.11] |

3 cascades lose to 6 or 12 cascades of the same size in all four comparisons; 6×26 and 12×18 swap order between seeds.

**Two-seed means** of the cells run in both seeds: 6×18 37.18, 12×12 37.17, 12×18 37.16, 6×26 37.15, 3×36 36.96 dB.
The top four are within 0.03 dB.

## Conclusions
- **Width does not help at 80 volumes.** 18 channels is at or above every wider cell at 3, 6 and 12 cascades
  (Δ −0.01 to −0.41 dB). The two largest models (12×26, 12×36) are worse and have the largest train − val gaps
  (1.1–1.5 dB), consistent with over-capacity for 80 volumes. That this is over-fitting rather than optimisation
  trouble at large width is a hypothesis, untested (lr 1e-3 was not tuned per width; no NaNs or skipped steps).
- **Depth beats width at equal size, below about 6 cascades.** 3-cascade models trail 6- and 12-cascade models of the
  same size by 0.10–0.31 dB in both seeds; between 6 and 12 cascades the order flips with the seed.
- **Above 3 cascades, size barely matters.** Cells from 14 M (12×12) to 32 M (6×26) parameters land within 0.03 dB on
  the two-seed mean, which is smaller than the 0.1–0.15 dB seed-to-seed change of a single cell here.
- **The reconstructor for the next experiment stays 6×18** (best two-seed mean and the cheapest within 0.05 dB).
- **Tier L: data still pays, capacity barely does.**
  - Going from 80 to 300 volumes adds +0.45 dB to 6×18 in both seeds, both on CoreHPC.
  - Doubling the parameters adds at most +0.06 dB at 300 volumes (12×18, `best_model.pt` only).
  - The 80-volume saturation of model size mostly holds at 300 volumes. Depth gains slightly with data; width does not.
- **Tier-L next step by the frozen rule: 12×18**, by 0.003 dB over 6×18, and the two tie at `last_model.pt`.
- **Platform control failed in s42 (−0.23 dB).**
  - Hypothesis, untested: the cause is checkpoint selection, not hardware.
  - The CoreHPC runs early-stopped at epochs 60 and 75, against 90 and 85 locally, on the noisy 10-volume subset.
  - Their best trainer-val losses match the local runs (0.0313 / 0.0312 vs 0.0313 / 0.0313).
  - At `last_model.pt` the gap is −0.09 / −0.03 dB, within the 0.10 tolerance.

## Caveats
- Only 5 cells have two seeds; the other 7 are seed 42 only. Seed-to-seed changes of one cell reach 0.15 dB (12×12),
  the size of several Q1 differences, so single-seed rows in the width table are indicative only.
- The 12×12 seed-43 run beats 12×18 by 0.09 dB, the opposite of seed 42 (−0.07); "18 needed" is not established.
- `best_model.pt` is chosen on the trainer's 10-volume val subset, which is noisier than the full set and ranked cells
  differently (6×12 looked worst there; it is mid-table on the full set). See the data audit in
  `reports/brain_varnet_scaling/report.md`.
- Early stopping cut runs at different steps (5,320–7,600), so the largest models saw fewer steps; their
  `last_model.pt` is lower than `best_model.pt`, so more steps would not have helped them.
- lr 1e-3 for all widths was a proposal choice, not tested per width.
- Seed-42 scoring ran with the eval and summary scripts before they were committed in `5c2fb4e`; the committed files
  are the same ones, and the summary was rerun at `5c2fb4e`.
- Reused 18-channel cells were trained at older commits (`35f8f2f`, `8485567`); c12_v80_s42 ran without early stopping.
- Proposal readouts not produced: PSNR against wall-clock as a figure (the table gives s / step and stop step) and
  validation curves.

- Tier L:
  - **Platform control.** The tier-L and CoreHPC tier-S runs used Python 3.14, torch 2.12 and RTX Pro 6000 bf16; the local runs used Python 3.13 and A6000s. The control failed at `best_model.pt` in s42, so tier-L numbers compare to tier-S numbers only within CoreHPC.
  - **Unequal schedules.** Early stopping ended several tier-L runs before the lr drop at epoch 90: 12×18 at 70 / 65, 12×26 s43 at 55, 6×26 at 80 / 85.
  - **12×26 s42 is an outlier.** Its train − val gap is 1.32 dB, about double the others (0.67–0.83), and it loses 0.15 dB from best to last.
  - **Trainer-val subset not confirmed.** That CoreHPC drew the same 10-volume subset was not checked directly against a file list, though it uses the same manifest and draw seed. Supporting evidence: 150 slices, and trainer-val losses matching the local runs.
  - **No per-cell tuning.** lr and the 100-epoch budget were not tuned per cell at L.

## Figures
- `figures/full_val_grid.png`, `figures/full_val_grid_s43.png`: full-val PSNR heat map over cascades × channels.
- `figures/full_val_vs_params.png`, `figures/full_val_vs_params_s43.png`: PSNR against parameter count, equal-size
  groups marked.
- `figures/tier_l_cells.png`: full-val PSNR at peak for the four tier-L cells, tier S (local) against tier L (CoreHPC), both seeds.

## Next steps
- Rerun static vs learned brain masks (S, M, U) with the 6×18 reconstructor, as frozen.
- Done at tier L (above): data +0.45 dB, capacity ≤ +0.06 dB. Next, tier F (all 339 usable FLAIR volumes, on main),
  with 6×18 or 12×18.
- Early stopping on the 10-volume subset picks noisy checkpoints (the CoreHPC S runs, the 12×18 L runs). Consider
  validating on the full 40 volumes, or always reporting `last_model.pt`.

## Reproduce
`bash reports/brain_varnet_width/run.sh` from the repo root. It needs the checkpoints linked under `checkpoints/`
(`~/data/seqmri_checkpoints/brain_varnet_width` and the four reused `brain_varnet_scaling` runs), the brain_varnet_scaling
score files, and one GPU. The tier-L part needs the CoreHPC runs copied to
`~/data/seqmri_checkpoints/brain_varnet_width_l` (rsync line in `run.sh`; training is the sbatch above). Check with `python scripts/report_manifest.py check reports/brain_varnet_width`.
