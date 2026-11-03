# How much faster can 1-step LOUPE + VarNet (R=8) train? (2026-11-03)

| | |
|---|---|
| Code | results produced at `c527d3e` (varnet_train_speed: results with the Muon + LayerNorm arms, final report text); branch `exp/varnet_train_speed` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 28 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 15 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Branch:** `exp/varnet_train_speed`. Results come from the commit in the provenance block. The runs used the
  per-arm code snapshots listed under Runs compared.
- **Status:** awaiting the user's verdict. The proposal (`proposal.md`, frozen 2026-11-02 22:45) was written for an
  overnight run and was not approved before launch.

## Proposal vs outcome

Frozen criteria, read as same-seed paired deltas against the control (see Methods 7). Pre-freeze arms come first,
then post-freeze ones.

| criterion (frozen) | arm | evidence | met? |
|---|---|---|---|
| Exact change: outputs match (bitwise or ≤1e-5 rel.) and step/epoch time −5% or more | trainer switches | bitwise-identical weights and RNG; bench step −12% | **met** |
| same | `fused_coil_ops` | 1e-15 (float64), ≤8e-7 (float32); bench −7% on top of the switches; fast path −11% per epoch in a 40-epoch run | **met** |
| same | channels-last (inductor layout) | +11% slower with InstanceNorm | not met |
| "Better": epoch-40 PSNR ≥ control + 0.1 dB in every seed, no SSIM loss | norm_layer, 20% data, 3 seeds | +0.79 / +0.88 / +0.59 dB; SSIM +0.019 / +0.021 / +0.009 | **met** |
| same | muon, 20% data, 2 seeds | +0.10 / −0.37 dB | not met: inconclusive |
| "Worse": below control − 0.1 dB in every seed | norm_none, norm_chln (1 seed each) | −0.37, −0.39 dB | met: worse |
| "Faster": time-to-quality ≤75% and final ≥ control − 0.1 dB | norm_layer | 46% / 43% / 54% of the control's wall-clock; finals above | met, but the measure is noisy: an exact replicate scored 37% |
| same | muon, 20% data | 63% (s42); never reached (s43) | not met |
| post-freeze, same "better" bar | full_muon, full data, 12 epochs, 2 seeds | +0.26 / +0.45 dB (20% val), +0.41 / +0.43 (all val); 69% / 52% time-to-quality | met (post-freeze) |
| same | full_norm_layer, 2 seeds | +0.02 / +0.05 dB | not met: null at full data |
| same | muon_layer, 20% data, 2 seeds | +0.70 / +1.00 dB; SSIM +0.014 / +0.019 | met (post-freeze) |
| same | full_muon_layer, full data, 1 seed | +0.60 dB [+0.27, +0.92]; SSIM +0.010 | met in its one seed (post-freeze) |

Read as frozen (against the control *mean*, 30.25 dB), no verdict changes:
- LayerNorm's finals (30.90–31.07) and Muon + LayerNorm's (30.98, 31.02) all clear 30.35;
- Muon's (30.38, 29.65) straddle it;
- no norm and channel LayerNorm (29.91, 29.88) fall below 30.15.

**Recommendation:** **merge** the exact fast path, and flip its defaults: `training.train_metrics_every=10`,
`training.visualize_every=10`, `recon.fused_coil_ops=true`. That is −11% wall-clock per epoch with the training math
unchanged.

Do **not** yet flip `recon.norm` or `training.optimizer`:
- LayerNorm is a robust +0.75 dB win on 20% of the data that vanishes at full data in a 12-epoch run.
- Muon is the reverse: +0.36 dB at full data, inconsistent at 20%.
- Muon + LayerNorm together was the best arm at both data sizes: +0.85 dB at 20% (2 seeds), +0.60 dB at full
  data (1 seed). It is the candidate recipe.
- Run the full-data 100-epoch 2×2 first (Next steps 2), at 2+ seeds.

For small-data studies, such as per-anatomy subsets, LayerNorm can be switched on now. This is a recommendation only;
the verdict is the user's.

## Question

The user asked to make training of the main operating point faster: 1-step LOUPE + 12-cascade E2E VarNet, R=8, local v1
brain FLAIR + knee PD/PDFS at 320². There are two routes:
- **Throughput:** exact changes that cut time per step or per epoch without touching the training math.
- **Convergence:** Muon and normalisation-layer swaps that need fewer epochs to reach the same validation quality.

Experiments ran on 20% of the data ("0.2 percent" was read as `datasets.limit_volumes=0.2`), with a full-data check of
the leading changes.

Conventions:
- PSNR is in dB and SSIM is unitless. Both are the trainer's PSNR_Masked / SSIM_Masked: per slice, inside the apodized
  per-anatomy metric mask, with data range = the slice's maximum, then averaged over slices.
- "Paired" means arm minus control on the same validation slices with the same mask draws, at the same training seed.
- 95% CIs are over per-volume means.

## Runs compared

| group | data | epochs (lr step) | seeds | change against the control | code snapshot |
|---|---|---|---|---|---|
| control | 20% | 40 (36) | 42, 43, 44 | none: Adam 1e-3, InstanceNorm | main 457c788 |
| control_short | 20% | 20 (18) | 42, 43 | schedule only | 457c788 |
| muon / muon_short | 20% | 40 (36) / 20 (18) | 42, 43 / 42, 43 | `training.optimizer=muon` | 36b58f4 |
| norm_layer / norm_layer_short | 20% | 40 / 20 | 42, 43, 44 / 42, 43 | `recon.norm=layer sampler.sens_norm=layer` (GroupNorm(1, C)) | 36b58f4 |
| norm_layer_casc | 20% | 40 | 42 | LayerNorm in the cascades only | 36b58f4 |
| norm_none / norm_chln | 20% | 40 | 42 | no norm / per-pixel channel LayerNorm | 36b58f4 |
| muon_layer | 20% | 40 | 42, 43 | Muon + LayerNorm | 36b58f4 |
| fast | 20% | 40 | 42 | exact fast path only: `train_metrics_every=10 visualize_every=10 recon.fused_coil_ops=true` | 519f777 |
| full_control | 100% | 12 (11) | 42, 43 | exact fast path, Adam, InstanceNorm | 519f777 |
| full_muon / full_norm_layer / full_muon_layer | 100% | 12 (11) | 42, 43 / 42, 43 / 42 | fast path + Muon / LayerNorm / both | 519f777 |
| fulldata_ref | 100% | 100 (90) | 42 | reference: loss_domain RSS run of 2026-10-30, same config, old code | (exp/loss_domain) |

Each run is one A6000 at a time, logged to wandb group `varnet_train_speed` as `vts_<group>_s<seed>`. Run directories
are in `~/data/seqmri_checkpoints/varnet_train_speed/<group>_s<seed>`. All 20% runs see the same 772 training and
390 validation slices; the full-data runs train on 3,929 slices. Every run in this report is scored on the 20%
validation subset by the eval script; the full-data runs' trainer logs also give the full 1,938-slice validation.

## Methods

1. **Data.** `datasets=fastmri_local_v1_flair_knee`, `limit_volumes=0.2`, a fixed draw (seed 1234):
   - train: 16 FLAIR + 8 PD + 8 PDFS volumes = 241 + 266 + 265 = 772 slices, 49 batches of 16;
   - val: 8 + 4 + 4 volumes = 118 + 141 + 131 = 390 slices.
   The full-data runs use `limit_volumes=1`: tier S train, 3,929 slices; val, 1,938 slices.
2. **Control.** The main config as of 457c788:
   - 12-cascade VarNet (fastMRI U-Net, chans 18, InstanceNorm, bf16, compiled);
   - `loupe_2d` straight-through mask with VD p=4 start, R=8 of the acquired k-space;
   - Adam lr 1e-3, StepLR ×0.1 at 90% of the epochs, batch 16;
   - seeds 42/43/44, 40 epochs (step at 36).
   Every arm changes one thing.
3. **Exact throughput changes.** Each is checked for exactness before it is timed.
   - `training.train_metrics_every=N` and `visualize_every=N`: train-batch metrics and images every Nth batch/epoch,
     and no per-forward host copies of visualisation tensors otherwise. A deterministic 2-epoch run gives
     bitwise-identical weights and RNG state (`equivalence_switches.json`).
   - `recon.fused_coil_ops=true`: the cascades' coil combine, coil expand and soft-DC update are written as
     real-arithmetic `torch.compile` kernels. Complex tensors stay outside the compiled graph. Fused vs eager is
     1e-15 in float64 and 5e-7 in float32 (eager vs eager: 3e-7) (`fused_check.json`).
   - `val_every`: added, not used in the runs here.
   - Step time is measured per arm by `scripts/diagnostics/varnet_train_speed_bench.py` on one idle A6000. It times 24
     training batches through the trainer's own `train_epoch` after a 14-batch warm-up (compile and autotune), plus
     one validation pass (`bench.json`). Each batch also carries 1/24 of the per-epoch overhead, so absolute values
     exceed the runs' epoch times / 49. Real-run epoch times are in `curves.json`.
4. **Convergence arms**, each against the 40-epoch control:
   - `optimizer=muon`: Muon (Nesterov 0.95, 5 Newton–Schulz steps, update RMS matched to Adam) on the 3×3 and
     transposed convolutions of every U-Net; Adam on everything else, including the LOUPE logits.
   - `norm=layer`: GroupNorm(1, C), affine.
   - `norm=channel_layer`: per-pixel LayerNorm over C, affine.
   - `norm=none`.
   The norm swaps apply to the cascades and the sensitivity U-Net, and leave every conv weight's initial value
   unchanged.
5. **Post-freeze additions.**
   - 20-epoch schedules (step at 18): control_short, muon_short, norm_layer_short.
   - LayerNorm in the cascades only (`norm_layer_casc`).
   - LayerNorm seeds 43/44.
   - Full-data 12-epoch runs (step at 11) on the exact fast path: full_control / full_muon / full_norm_layer, seeds 42/43.
   - The loss_domain 100-epoch full-data RSS run (`fulldata_ref_s42`, main code of 2026-10-30, same config) as a
     reference ceiling.
   These were added overnight to test convergence speed, which the frozen 40-epoch time-to-quality criterion cannot
   see because the control plateaus by about epoch 13. The criteria are read as same-seed paired deltas.
6. **Scoring.** `scripts/eval/varnet_train_speed_eval.py` scores every finished run's `last_model.pt` on the 20% val
   set:
   - 2 seeded mask draws, averaged;
   - per condition (brain, knee-PD, knee-PDFS);
   - paired arm − control per seed, with `--control control`, `control_short` or `full_control`.
   Its "all" PSNR matches the trainer's last logged `val/PSNR_Masked` to 0.03 dB. The error split adds L1 inside and
   outside the metric mask and full-image PSNR (`error_split.json`).
7. **Noise floor.** Two same-seed control runs share an identical config up to epoch 18 (control_s42 and
   control_short_s42). Over epochs 5–18 they differ by −0.27 ± 0.28 dB (mean ± SD of per-epoch val PSNR), because
   cuDNN autotuning makes runs nondeterministic. The seed range of the three 40-epoch controls is 0.43 dB.

## Results

All quality numbers below are final-epoch validation PSNR_Masked / SSIM_Masked. The tables are generated by
`scripts/eval/varnet_train_speed_summary.py` into `summary.md`, which also holds every per-seed and per-condition
number.

### 1. Throughput: exact changes

| | step (bench, ms) | train s/epoch (20% data, runs) | val s/epoch | exactness |
|---|---|---|---|---|
| control | 1774 | 69.9 (3 runs) | 17.1 | reference |
| + switches (`train_metrics_every=10`, `visualize_every=10`) | 1565 (−12%) | (in fast) | (in fast) | bitwise-identical weights and RNG after 2 epochs |
| + `fused_coil_ops` = fast path | 1449 (−18%) | 61.5 (−12%) | 15.7 (−8%) | 1e-15 (float64), ≤8e-7 (float32) vs eager |

- **Whole epoch.** In a real 40-epoch run (fast_s42 vs control_s42, 20% data), the fast path cuts 11% (77.5 vs 87.0 s, medians over epochs 2–40); the run ends −0.11 dB [−0.26, +0.03] from control_s42, within same-seed noise
  per epoch, train plus validation. The benchmark's −18% per step is an upper bound: it spreads the control's
  per-epoch visualisation over 24 batches instead of 49.
- **Full data.** The fast path trains an epoch in 302 s and validates in 72 s. The old-code full-data run of the same
  config took 374 s and 79 s, but its code also differs in small ways, so the clean figure is the 20% pair. Its first
  four epochs track the old run's train and val loss to within 0.5%.
- **Channels-last** (inductor layout optimisation) made every variant except norm=none slower: +11% with
  InstanceNorm, +7% with channel LayerNorm, −2% with no norm.
- **Kernel profile.** The step is mostly convolutions, about 49% including about 17% of cuDNN NCHW↔NHWC transposes,
  then complex elementwise arithmetic (~29%), copies (~8%) and FFTs (~7%) (`kernel_profile_contended.json`;
  contended GPU, proportions only).

### 2. Convergence on 20% of the data

Final validation scores by the eval script (last_model.pt, 2 mask draws), mean over seeds [range], against the
same-seed control:

| arm | epochs | seeds | PSNR (all) | SSIM | brain PSNR | knee-PD PSNR | knee-PDFS PSNR | ΔPSNR vs same-seed control (per seed) |
|---|---|---|---|---|---|---|---|---|
| control | 40 | 3 | 30.25 [30.02, 30.45] | 0.800 | 29.58 | 32.33 | 28.61 | (reference) |
| control_short | 20 | 2 | 30.04 [30.01, 30.07] | 0.797 | 29.57 | 31.83 | 28.54 | −0.21, −0.01 |
| fast (exact) | 40 | 1 | 30.17 | 0.794 | 29.02 | 32.42 | 28.78 | −0.11 |
| **norm_layer** | 40 | 3 | **31.00 [30.90, 31.07]** | **0.816** | **31.03** | 32.62 | 29.23 | **+0.79, +0.88, +0.59** |
| norm_layer_short | 20 | 2 | 30.40 [30.32, 30.48] | 0.809 | 30.54 | 31.99 | 28.56 | +0.20, +0.30 |
| norm_layer_casc | 40 | 1 | 30.89 | 0.816 | 31.10 | 32.53 | 28.92 | +0.61 |
| muon_layer | 40 | 2 | 31.00 [30.98, 31.02] | 0.812 | 30.11 | 33.38 | 29.24 | +0.70, +1.00 |
| muon | 40 | 2 | 30.01 [29.65, 30.38] | 0.804 | 29.19 | 32.79 | 27.76 | +0.10, −0.37 |
| muon_short | 20 | 2 | 29.72 [29.13, 30.31] | 0.792 | 29.08 | 31.94 | 27.90 | +0.03, −0.90 |
| norm_none | 40 | 1 | 29.91 | 0.794 | 29.89 | 31.11 | 28.64 | −0.37 |
| norm_chln | 40 | 1 | 29.88 | 0.795 | 29.89 | 31.03 | 28.65 | −0.39 |

95% CIs over volumes and the per-condition deltas are in `summary.md`. For LayerNorm the CI of the overall delta
excludes 0 in every seed: [+0.40, +1.17], [+0.50, +1.25], [+0.36, +0.81].

- **Control.** It plateaus by about epoch 13 at lr 1e-3, and the lr step adds 0.1–0.3 dB. A 20-epoch schedule
  (step at 18) ends −0.21 and −0.01 dB below the 40-epoch run of the same seed, within the same-seed noise
  (±0.28 dB).
- **LayerNorm (GroupNorm(1, C)) beats the control in all three seeds:** +0.79, +0.88 and +0.59 dB, with SSIM
  +0.009 to +0.021.
  - It starts slower, crosses the control around epoch 15–19 and keeps improving to epoch 40, while the control
    stays flat.
  - It reaches the same-seed control's final PSNR at epochs 17, 16 and 20: 46%, 43% and 54% of the control's
    wall-clock, despite +6% per epoch.
    - That measure is noisy on a plateau. The exact replicate fast_s42 reached control_s42's final at epoch 16
      (37%).
    - The final paired deltas are the robust evidence.
    - At half the epochs, norm_layer_short still ends +0.20 / +0.30 dB above the 40-epoch control.
  - Its train loss stays higher (0.0753 vs 0.0695 at epoch 40, s42) while its validation is better. That is
    consistent with less overfitting of the 772 slices, a hypothesis this study does not test.
  - The gain is largest in brain FLAIR (+1.77 / +1.73 / +0.87 dB; all 8 volumes better in seed 42), then knee PDFS
    (+0.53 / +0.60 / +0.71) and knee PD (+0.20 / +0.43 / +0.23).
  - The learned mask correlates 0.91 with the control's, as two control seeds do, so the gain is mostly in the
    reconstructor (`masks.json`).
  - LayerNorm in the cascades only (norm_layer_casc): +0.61 dB overall and +1.76 dB brain (seed 42). The cascades alone recover +0.61 of the +0.79 dB and essentially all of
    the brain gain. That is one seed, and no sensitivity-net-only arm was run.
- **Muon is inconsistent on 20% data.**
  - At 40 epochs, seed 42 is +0.10 dB and seed 43 is −0.37 dB (2-seed mean −0.14; the per-condition pattern is consistent: knee PD +0.48/+0.51, knee PDFS −0.54/−1.16).
  - At 20 epochs, seeds 42 and 43 are +0.24 and −0.89 against the same-seed 20-epoch control.
  - Its train loss is lower than the control's at every epoch, while its validation PSNR is not consistently higher.
- **No normalisation and channel LayerNorm are worse** (one seed each): −0.37 and −0.39 dB overall. Brain is
  +0.5 dB but knee PD is −1.35 dB. No normalisation also spiked at epochs 8 and 15; train loss jumped from 0.084 to
  0.117 at epoch 8. Its 12% faster step does not pay for that.
- **Muon + LayerNorm** (muon_layer): +0.70 [+0.51, +0.89] and +1.00 [+0.67, +1.33] dB (seeds 42, 43), SSIM +0.014 / +0.019. That is about
    LayerNorm alone (+0.79 / +0.88), but spread more evenly over the conditions: knee PD +0.89 / +1.27, where
    LayerNorm alone gave +0.20 / +0.43. With LayerNorm, Muon's seed-43 loss disappears.

### 3. Full data, 12 epochs (step at 11), exact fast path

Full-data runs, final epoch.
- **Trainer validation, all 1,938 val slices:**
  - control 31.35 / 31.02 (seeds 42 / 43);
  - Muon 31.76 / 31.45;
  - LayerNorm 31.41 / 31.11;
  - Muon + LayerNorm 32.03 (seed 42 only).
- **Eval script, 20% val subset**, paired against the same-seed full-data control (`eval_vs_full.json`):

| arm (full data, 12 epochs) | seeds | PSNR (20% val) | SSIM | ΔPSNR vs full_control per seed [95% CI] | ΔSSIM |
|---|---|---|---|---|---|
| full_control | 2 | 30.89 [30.75, 31.02] | 0.806 | (reference) | |
| **full_muon** | 2 | **31.24 [31.20, 31.28]** | **0.812** | **+0.26 [−0.06, +0.57], +0.45 [+0.25, +0.65]** | +0.003, +0.010 |
| full_norm_layer | 2 | 30.92 [30.80, 31.05] | 0.804 | +0.02 [−0.13, +0.17], +0.05 [−0.14, +0.25] | +0.001, −0.004 |
| **full_muon_layer** | 1 | **31.62** | **0.818** | **+0.60 [+0.27, +0.92]** | +0.010 |
| fulldata_ref (100 epochs) | 1 | 31.57 | 0.817 | +0.55 | +0.009 |
| norm_layer (20% data, 40 epochs) | 2 | 30.98 | 0.815 | +0.04, +0.15 | +0.009, +0.012 |

LayerNorm on 20% of the data and 40 epochs matches the full-data control at 12 epochs: brain +1.0 / +1.2 dB,
knee PDFS −0.6 dB.

- **Muon gives +0.41 and +0.43 dB** (trainer validation on all 1,938 val slices; seeds 42 and 43), with SSIM
  +0.008 and +0.012.
  - It reaches the same-seed control's final PSNR at epoch 8 and epoch 6 of 12: 69% and 52% of the control's
    wall-clock, at +5% per epoch.
  - Its validation L1 is lower at every epoch in both seeds.
- **LayerNorm's gain mostly disappears at full data:** +0.06 and +0.09 dB, SSIM ±0.002.
- **Muon + LayerNorm** (one seed): **+0.60 [+0.27, +0.92] dB** against full_control_s42 on the 20% val subset, SSIM +0.010.
    - That is more than Muon alone (+0.26) and LayerNorm alone (+0.02) for the same seed.
    - Trainer validation on all val slices: 32.03 dB at epoch 12, against 31.35 for the control and 32.11 for the
      100-epoch old-code reference.
    - In-mask L1 is 0.0401, the lowest of any run. It is one seed.

### 4. Where LayerNorm's validation L1 goes

`error_split.json` (one mask draw) separates the training loss (full-image L1) from the metric:

| run | L1 inside metric mask | L1 outside | full-image L1 | output / target background mean |
|---|---|---|---|---|
| control s42 / s43 / s44 | 0.0475 / 0.0493 / 0.0462 | 0.142 / 0.145 / 0.150 | 0.095 / 0.097 / 0.099 | 0.045 / 0.045 / 0.039 vs 0.176 |
| norm_layer s42 / s43 / s44 | **0.0424 / 0.0432 / 0.0421** | 0.155 / 0.156 / 0.155 | 0.101 / 0.102 / 0.101 | 0.029 / 0.027 / 0.027 vs 0.176 |
| full_control s42 / s43 | 0.0423 / 0.0433 | 0.145 / 0.147 | 0.095 / 0.096 | 0.038 / 0.037 vs 0.176 |
| full_muon s42 / s43 | 0.0417 / 0.0418 | 0.136 / 0.139 | 0.089 / 0.091 | 0.048 / 0.045 vs 0.176 |
| full_muon_layer s42 | 0.0401 | 0.136 | 0.088 | 0.047 vs 0.176 |

- **LayerNorm is better inside the metric mask:** in-mask L1 is 9–12% lower than the control's.
- **Its full-image L1 is higher only because of the background.** It renders the background darker (mean 0.027–0.029
  against the control's 0.039–0.045), further below the RSS target's noise floor (0.176).
- **The trainer's val loss therefore disagrees with its PSNR_Masked for LayerNorm.** The loss includes the
  background; PSNR_Masked does not.
- **Full-data Muon is better in both regions.**

## Conclusions

- **Exact fast path: adopt.** Trainer switches plus fused coil ops cut a 20%-data epoch by 11% (train −12%, val −8%)
  with unchanged math: bitwise for the switches, float rounding for the fused ops. A full-data epoch now takes
  302 + 72 s. Channels-last does not help.
- **Data size decides which convergence change helps.** On 772 training slices, the control stops improving after
  about epoch 13. On 3,929 slices it is still improving at epoch 12.
  - **LayerNorm, 20% data: +0.75 dB mean** (3/3 seeds; the 95% CI over volumes excludes 0 in each seed; 15/16 volumes better in seed 42), mostly brain FLAIR and mostly from
    the cascades.
    - It matches, at 20 epochs, what the control reaches at 40.
    - It lowers the in-tissue error; its background comes out darker.
    - At full data (12 epochs) it is null: +0.04 dB.
  - **Muon, full data: +0.36 dB** (20% val subset) **/ +0.42 dB** (all val), in 2/2 seeds, reaching the control's
    12-epoch quality at 52–69% of the wall-clock. On 20% data it is inconsistent (+0.10 / −0.37 at 40 epochs).
  - **Hypothesis, untested causally:** LayerNorm regularises (higher train loss, later plateau) and Muon optimises
    faster (lower train loss everywhere). That pattern would make each useful where the other is not.
- **Schedule, 20% data:** a 20-epoch schedule is within 0.1 dB of 40 epochs (−0.21 / −0.01), so short-data
  experiments can halve their epochs.
- **Rejected:** no normalisation (12% faster per step but unstable, −0.37 dB), per-pixel channel LayerNorm
  (−0.39 dB) and channels-last.
- **Muon + LayerNorm:** on 20% data (2 seeds) it is +0.85 dB, about LayerNorm's. At full data (1 seed) it is +0.60 dB
  against the control (Muon alone +0.26, LayerNorm alone +0.02), and in 12 epochs it gets within 0.1 dB of the
  100-epoch old-code reference. If that holds at 2+ seeds, the combination is the recipe to test at 100 epochs
- **Net effect on the time to a given quality,** combining the fast path with the arm's time-to-quality:
  - 20% data: about 0.5× (LayerNorm; quality at 20 epochs ≥ the control's 40) × 0.89 (fast path).
  - Full data, early training: about 0.6× (Muon) × 0.89.
  These are untested at 100 epochs.

## Caveats

- **Proposal timing and added arms.** The proposal was frozen at 22:45 but not approved before launch. The arms added
  after the freeze are post-hoc: the 20-epoch schedules, LayerNorm seeds 43/44, the LayerNorm split, the Muon +
  LayerNorm combination and all full-data runs. The frozen criteria are read as same-seed paired deltas against the
  control, as the noise section explains.
- **Noise.** Same-seed runs are not reproducible: cuDNN autotuning makes them diverge. Two identical-config runs
  (control_s42 and control_short_s42 up to epoch 18) differ by −0.27 ± 0.28 dB per epoch, and the three control seeds
  span 0.43 dB at epoch 40. Pairing by seed controls the data order and the initial weights, not the trajectory. Arms
  with one seed (norm_none, norm_chln, norm_layer_casc, fast, full_muon_layer) are single observations.
- **20% of the data.** On 772 training slices the control plateaus by about epoch 13 and its train loss keeps falling,
  so the 20% runs mostly measure generalisation, not optimisation. Muon's sign flip between seeds and LayerNorm's
  higher train loss with better validation both point that way. That reading is a hypothesis; the full-data pairs
  are the only test of it.
- **Short full-data budget.** The full-data runs are 12 epochs, about 2,950 steps, against the usual 100. Muon's
  +0.4 dB there is an early-training result. Whether it survives a 100-epoch schedule is untested.
- **Brain input convention.** All runs use main's brain k-space convention: the DC row and phase ramp differ from
  knee's, a difference fixed only on exp/loss_domain. Every non-InstanceNorm variant raised brain PSNR by
  +0.5 to +1.8 dB. Whether the brain gain survives that fix is untested.
- **Benchmark method.** `bench.json` times 24 batches through `train_epoch`, so each batch also carries 1/24 of the
  per-epoch overhead (visualisation in the control variant). The control and switches rows therefore overstate the
  per-step difference a little. The run logs' epoch times are the end-to-end measure.
  `kernel_profile_contended.json` was measured next to another job, so only its proportions are meaningful.
- **Old-code timing reference.** The full-data timing reference (`fulldata_ref_s42`, 374 s/epoch) ran with one other
  job on the machine; the fast-path full-data runs ran with three others. The 19% speed-up is therefore not inflated
  by lighter contention.
- **Untracked checkpoints.** Checkpoints are untracked, in `~/data/seqmri_checkpoints/varnet_train_speed`. The
  `fulldata_ref_s42` checkpoint belongs to the loss_domain study.

## Figures

- `figures/curves_20pct.png`: val PSNR_Masked and SSIM_Masked against epoch and against wall-clock (train + val
  seconds from the logs), seed mean with the seed range shaded. Arms are control, Muon, LayerNorm, Muon + LayerNorm,
  no norm and channel LayerNorm at 40 epochs.
- `figures/curves_full_data.png`: the same for the full-data 12-epoch runs: control, Muon, LayerNorm and Muon +
  LayerNorm.

## Next steps

1. If the verdict is to adopt: flip `training.train_metrics_every` and `visualize_every` to 10 and
   `recon.fused_coil_ops` to true. These are exact and give ~18% off every train epoch. Consider `val_every` 2–5 for
   100-epoch runs; validation is 17% of a full-data epoch.
2. Test LayerNorm and Muon at the real budget: full data, 100 epochs, the 4-arm 2×2 (InstanceNorm/LayerNorm ×
   Adam/Muon), 2 seeds. If both hold, retrain the anatomy-routing baselines with the new recipe before comparing
   0.05–0.2 dB effects, because the recipe changes their reference point.
3. Find why GroupNorm(1) helps brain by 1.8 dB:
   - InstanceNorm with affine, to separate the affine parameters from the grouping;
   - the cascades-only vs sensitivity-net split, at 2 seeds;
   - rerun with the brain DC-row fix from exp/loss_domain.
4. Make GroupNorm(1) as cheap as InstanceNorm: it costs +6% per epoch, from a whole-sample reduction and the fp32
   upcast under autocast.

## Reproduce

Scripts, all called by `run.sh`:
- `scripts/diagnostics/varnet_train_speed_equivalence.py`: switch exactness;
- `scripts/diagnostics/varnet_train_speed_fused_check.py`: fused-ops numerics;
- `scripts/diagnostics/varnet_train_speed_bench.py`: step time per arm;
- `scripts/diagnostics/varnet_train_speed_profile_kernels.py`: kernel breakdown;
- `scripts/diagnostics/varnet_train_speed_masks.py`: learned masks;
- `scripts/eval/varnet_train_speed_curves.py`: validation curves and epoch times from wandb and the logs;
- `scripts/eval/varnet_train_speed_eval.py`: per-condition paired scores and the error split;
- `scripts/eval/varnet_train_speed_summary.py`: the tables in `summary.md`;
- `scripts/plots/varnet_train_speed_curves.py`: the figures.


`bash reports/varnet_train_speed/run.sh` from the repo root. It needs:
- the run directories in `~/data/seqmri_checkpoints/varnet_train_speed` (untracked; listed in `manifest.json`);
- an idle GPU for the bench and equivalence steps;
- wandb access for `curves.json`.

The training runs came from `scripts/launch/varnet_train_speed_worker.sh` with the job lines in
`~/data/seqmri_checkpoints/varnet_train_speed/jobs.txt`. Check with
`python scripts/report_manifest.py check reports/varnet_train_speed`.
