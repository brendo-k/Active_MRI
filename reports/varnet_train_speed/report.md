# How much faster can 1-step LOUPE + VarNet (R=8) train? (2026-11-03)

PROVENANCE_BLOCK

- **Branch:** `exp/varnet_train_speed`. Results come from the commit in the provenance block. The runs used the
  per-arm code snapshots listed under Runs compared.
- **Status:** awaiting the user's verdict. The proposal (`proposal.md`, frozen 2026-11-02 22:45) was written for an
  overnight run and was not approved before launch.

## Proposal vs outcome

CRITERIA_TABLE

**Recommendation:** RECOMMENDATION

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
| + `fused_coil_ops` = fast path | 1449 (−18%) | FAST_TRAIN | FAST_VAL | 1e-15 (float64), ≤8e-7 (float32) vs eager |

- **Whole epoch.** In a real 40-epoch run (fast_s42 vs control_s42, 20% data), the fast path cuts FAST_EPOCH_PCT
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

TABLE_20PCT

- **Control.** It plateaus by about epoch 13 at lr 1e-3, and the lr step adds 0.1–0.3 dB. A 20-epoch schedule
  (step at 18) ends −0.21 and −0.01 dB below the 40-epoch run of the same seed, within the same-seed noise
  (±0.28 dB).
- **LayerNorm (GroupNorm(1, C)) beats the control in all three seeds:** +0.79, +0.84 and +0.59 dB, with SSIM
  +0.016–0.020.
  - It starts slower, crosses the control around epoch 15–19 and keeps improving to epoch 40, while the control
    stays flat.
  - It reaches the same-seed control's final PSNR at epochs 17, 19 and 15, at about 46% of the control's
    wall-clock, despite +6% per epoch.
  - Its train loss stays higher (0.0753 vs 0.0695 at epoch 40, s42): it overfits the 772 slices less.
  - The gain is largest in brain FLAIR (+1.8 dB in seed 42, all 8 volumes better), then knee PDFS (+0.5) and knee PD (+0.2).
  - The learned mask correlates 0.91 with the control's, as two control seeds do, so the gain is mostly in the
    reconstructor (`masks.json`).
  - LayerNorm in the cascades only (norm_layer_casc): CASC_RESULT.
- **Muon is inconsistent on 20% data.**
  - At 40 epochs, seed 42 is +0.13 dB and seed 43 is MUON_S43_DELTA.
  - At 20 epochs, seeds 42 and 43 are +0.24 and −0.89 against the same-seed 20-epoch control.
  - Its train loss is always lower than the control's, so it fits faster and generalises unevenly on little data.
- **No normalisation and channel LayerNorm are worse** (one seed each): −0.37 and −0.39 dB overall. Brain is
  +0.5 dB but knee PD is −1.35 dB. No normalisation also spiked at epochs 8 and 15; train loss jumped from 0.084 to
  0.117 at epoch 8. Its 12% faster step does not pay for that.
- **Muon + LayerNorm** (muon_layer): COMBO_20PCT.

### 3. Full data, 12 epochs (step at 11), exact fast path

TABLE_FULL

- **Muon gives +0.41 and +0.43 dB** (trainer validation on all 1,938 val slices; seeds 42 and 43), with SSIM
  +0.008 and +0.012.
  - It reaches the same-seed control's final PSNR at epoch 8 and epoch 6 of 12: 69% and 52% of the control's
    wall-clock, at +5% per epoch.
  - Its validation L1 is lower at every epoch in both seeds.
- **LayerNorm's gain mostly disappears at full data:** +0.06 and +0.09 dB, SSIM ±0.002.
- **Muon + LayerNorm** (one seed): COMBO_FULL.

### 4. Where LayerNorm's validation L1 goes

ERROR_SPLIT

## Conclusions

CONCLUSIONS

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
  seconds from the logs), seed mean with the seed range shaded. Arms are control, Muon and the three norm swaps at
  40 epochs.
- `figures/curves_full_data.png`: the same for the full-data 12-epoch runs.

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

`bash reports/varnet_train_speed/run.sh` from the repo root. It needs:
- the run directories in `~/data/seqmri_checkpoints/varnet_train_speed` (untracked; listed in `manifest.json`);
- an idle GPU for the bench and equivalence steps;
- wandb access for `curves.json`.

The training runs came from `scripts/launch/varnet_train_speed_worker.sh` with the job lines in
`~/data/seqmri_checkpoints/varnet_train_speed/jobs.txt`. Check with
`python scripts/report_manifest.py check reports/varnet_train_speed`.
