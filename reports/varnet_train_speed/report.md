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

RUNS_TABLE

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

RESULTS

## Conclusions

CONCLUSIONS

## Caveats

CAVEATS

## Figures

- `figures/curves_20pct.png`: val PSNR_Masked and SSIM_Masked against epoch and against wall-clock (train + val
  seconds from the logs), seed mean with the seed range shaded. Arms are control, Muon and the three norm swaps at
  40 epochs.
- `figures/curves_full_data.png`: the same for the full-data 12-epoch runs.

## Next steps

NEXT_STEPS

## Reproduce

`bash reports/varnet_train_speed/run.sh` from the repo root. It needs:
- the run directories in `~/data/seqmri_checkpoints/varnet_train_speed` (untracked; listed in `manifest.json`);
- an idle GPU for the bench and equivalence steps;
- wandb access for `curves.json`.

The training runs came from `scripts/launch/varnet_train_speed_worker.sh` with the job lines in
`~/data/seqmri_checkpoints/varnet_train_speed/jobs.txt`. Check with
`python scripts/report_manifest.py check reports/varnet_train_speed`.
