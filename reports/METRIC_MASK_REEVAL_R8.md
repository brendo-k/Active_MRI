# Metric mask re-evaluation (R=8 / R=4 checkpoints), 2026-10-20

Every checkpoint's `last_model.pt` was run through the trainer's validation path on its own val set (688 slices: 344 brain, 344 knee), and the
same outputs were scored twice: under the legacy mask (`target > 0.1 * max`) and under the new apodized per-anatomy metric mask
(`activemri/utils/metric_mask.py`). One mask draw per run (the trainer's seeded eval generator). The legacy columns reproduce the final logged
val metrics of the run (checked on `loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep`: brain 34.93 dB / 0.913 / 0.0174, knee 32.71 / 0.929 / 0.0172).

Script: `scripts/eval/reeval_metric_mask_r8.py`; per-run JSON in `reports/metric_mask_reeval_r8/`.

Not evaluated: the six `mnet_varnet_*` runs (no checkpoint saved). The `varnet_caipi2d_r4_promptmr_*` runs and
`loupe2d_cgsense_r8_brain_knee_prototable_bilinear_const_100ep` were first scored mid-training and re-scored after they finished (all at their last epoch).

## Findings

- The new mask scores higher for every run. Brain PSNR rises 1.5 to 5.0 dB (the 0.7 dB gain of `shared_bilinear_nomask_50ep` is the exception, below) and knee
  PSNR 1.1 to 3.9 dB. NMSE falls 40-80% in brain and 25-55% in knee. The legacy mask drops the dark interior and, on ghosted slices, covers the noisy
  background, so absolute numbers are not comparable across the change.
- Knee SSIM moves less and not uniformly: down (-0.002 to -0.010) for the learned-sense-net CG-SENSE LOUPE runs, up for ecalib CG-SENSE (+0.006) and for
  VarNet runs (+0.004 to +0.007).
- Same-sign comparisons, larger gaps (deltas are brain / knee, new minus shared):
  CG-SENSE R=8 anatcond-shared: legacy +0.12 / -0.15 dB, new +0.22 / -0.24 dB; sharedinit: legacy +0.12 / -0.03, new +0.21 / -0.06.
  VarNet LOUPE anatcond_100ep minus conv floor0.05: legacy +0.44 / +0.28 dB, new +1.00 / +0.66 dB.
  Random-mask VarNet minus conv floor0.05: legacy -1.42 / -0.59 dB, new -2.72 / -1.22 dB.
- One flip: CG-SENSE with ecalib sense maps (`ecalib_c10`) minus the shared learned-sense-net run on brain is +0.39 dB under the legacy mask and
  -0.27 dB under the new one (knee: +0.61 -> +1.13 dB).
- CAIPI PromptMR R=4 (new mask): cond seeds 42/43 reach 41.61 / 41.73 dB brain and 37.82 / 37.86 dB knee against 41.69 / 37.85 for the label-blind run, so the
  cond-blind gap is inside the seed spread.
- Seed spread is small (`shared` seeds 42/43/bilinear: brain 37.11-37.13, knee 34.08-34.12 under the new mask).
- `shared_bilinear_nomask_50ep` (sense net with no body mask, `sense_mask_method=null`) scored 1.6 dB above `shared_bilinear_50ep` on brain under the legacy mask
  (36.55 vs 34.93) but only +0.09 dB under the new one (37.22 vs 37.13); knee +1.05 dB (35.14 vs 34.09). This fits the earlier finding that sense-mask
  support dominates val error, but it was not tested further here.

## Table

| run | epoch | anatomy | PSNR legacy | PSNR new | SSIM legacy | SSIM new | NMSE legacy | NMSE new |
|---|---|---|---|---|---|---|---|---|
| conv2d_varnet_r8_brain_knee_floor0.05 | 50/50 | brain | 36.28 | 39.80 | 0.9309 | 0.9800 | 0.0138 | 0.0042 |
| conv2d_varnet_r8_brain_knee_floor0.05 | 50/50 | knee | 34.07 | 37.02 | 0.9511 | 0.9554 | 0.0128 | 0.0066 |
| conv2d_varnet_r8_brain_knee_floor0.05_ahain | 50/50 | brain | 36.63 | 40.64 | 0.9335 | 0.9824 | 0.0131 | 0.0035 |
| conv2d_varnet_r8_brain_knee_floor0.05_ahain | 50/50 | knee | 34.26 | 37.47 | 0.9529 | 0.9578 | 0.0123 | 0.0061 |
| conv2d_varnet_r8_brain_knee_floor0.05_clamp10_leak0.05 | 50/50 | brain | 36.44 | 40.11 | 0.9323 | 0.9807 | 0.0134 | 0.0039 |
| conv2d_varnet_r8_brain_knee_floor0.05_clamp10_leak0.05 | 50/50 | knee | 34.19 | 37.34 | 0.9522 | 0.9568 | 0.0125 | 0.0062 |
| conv2d_varnet_r8_brain_knee_floor0.05_draws2 | 50/50 | brain | 36.49 | 40.24 | 0.9264 | 0.9795 | 0.0134 | 0.0038 |
| conv2d_varnet_r8_brain_knee_floor0.05_draws2 | 50/50 | knee | 34.21 | 37.36 | 0.9491 | 0.9545 | 0.0125 | 0.0062 |
| conv2d_varnet_r8_brain_knee_floor0.05_imginput | 50/50 | brain | 36.47 | 40.15 | 0.9324 | 0.9808 | 0.0134 | 0.0039 |
| conv2d_varnet_r8_brain_knee_floor0.05_imginput | 50/50 | knee | 34.20 | 37.34 | 0.9520 | 0.9566 | 0.0125 | 0.0062 |
| conv2d_varnet_r8_brain_knee_floor0.05_imginput_headbias | 50/50 | brain | 36.51 | 40.22 | 0.9328 | 0.9810 | 0.0133 | 0.0038 |
| conv2d_varnet_r8_brain_knee_floor0.05_imginput_headbias | 50/50 | knee | 34.20 | 37.36 | 0.9521 | 0.9568 | 0.0125 | 0.0062 |
| conv2d_varnet_r8_brain_knee_floor0.05_imginput_headmlp_ln_clipsplit | 50/50 | brain | 36.39 | 39.97 | 0.9320 | 0.9803 | 0.0136 | 0.0041 |
| conv2d_varnet_r8_brain_knee_floor0.05_imginput_headmlp_ln_clipsplit | 50/50 | knee | 34.15 | 37.23 | 0.9516 | 0.9560 | 0.0127 | 0.0064 |
| loupe2d_cgsense_r4_brain_knee_anatcond | 50/50 | brain | 35.81 | 38.76 | 0.9223 | 0.9714 | 0.0149 | 0.0053 |
| loupe2d_cgsense_r4_brain_knee_anatcond | 50/50 | knee | 33.60 | 35.67 | 0.9453 | 0.9436 | 0.0140 | 0.0088 |
| loupe2d_cgsense_r4_brain_knee_shared | 50/50 | brain | 35.82 | 38.80 | 0.9224 | 0.9715 | 0.0148 | 0.0053 |
| loupe2d_cgsense_r4_brain_knee_shared | 50/50 | knee | 33.57 | 35.62 | 0.9447 | 0.9429 | 0.0141 | 0.0089 |
| loupe2d_cgsense_r8_brain_knee_anatcond | 50/50 | brain | 35.05 | 37.35 | 0.9152 | 0.9639 | 0.0170 | 0.0075 |
| loupe2d_cgsense_r8_brain_knee_anatcond | 50/50 | knee | 32.54 | 33.84 | 0.9256 | 0.9172 | 0.0178 | 0.0133 |
| loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep | 100/100 | brain | 35.08 | 37.38 | 0.9152 | 0.9638 | 0.0170 | 0.0075 |
| loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep | 100/100 | knee | 32.51 | 33.79 | 0.9252 | 0.9165 | 0.0179 | 0.0134 |
| loupe2d_cgsense_r8_brain_knee_anatcond_sharedinit | 50/50 | brain | 35.05 | 37.34 | 0.9147 | 0.9634 | 0.0171 | 0.0076 |
| loupe2d_cgsense_r8_brain_knee_anatcond_sharedinit | 50/50 | knee | 32.66 | 34.02 | 0.9282 | 0.9205 | 0.0173 | 0.0127 |
| loupe2d_cgsense_r8_brain_knee_dcm_anatcond | 50/50 | brain | 34.67 | 36.70 | 0.9110 | 0.9595 | 0.0182 | 0.0087 |
| loupe2d_cgsense_r8_brain_knee_dcm_anatcond | 50/50 | knee | 32.23 | 33.36 | 0.9200 | 0.9099 | 0.0191 | 0.0149 |
| loupe2d_cgsense_r8_brain_knee_dcm_prototable | 50/50 | brain | 34.64 | 36.65 | 0.9111 | 0.9594 | 0.0182 | 0.0088 |
| loupe2d_cgsense_r8_brain_knee_dcm_prototable | 50/50 | knee | 32.20 | 33.31 | 0.9202 | 0.9100 | 0.0193 | 0.0150 |
| loupe2d_cgsense_r8_brain_knee_dcm_shared | 50/50 | brain | 34.43 | 36.32 | 0.9080 | 0.9563 | 0.0189 | 0.0095 |
| loupe2d_cgsense_r8_brain_knee_dcm_shared | 50/50 | knee | 32.42 | 33.66 | 0.9239 | 0.9149 | 0.0184 | 0.0140 |
| loupe2d_cgsense_r8_brain_knee_ecalib_c10 | 50/50 | brain | 35.32 | 36.86 | 0.9276 | 0.9635 | 0.0158 | 0.0082 |
| loupe2d_cgsense_r8_brain_knee_ecalib_c10 | 50/50 | knee | 33.30 | 35.21 | 0.9386 | 0.9445 | 0.0148 | 0.0093 |
| loupe2d_cgsense_r8_brain_knee_ecalib_c24 | 50/50 | brain | 35.31 | 36.85 | 0.9271 | 0.9629 | 0.0158 | 0.0083 |
| loupe2d_cgsense_r8_brain_knee_ecalib_c24 | 50/50 | knee | 33.25 | 35.13 | 0.9376 | 0.9432 | 0.0150 | 0.0095 |
| loupe2d_cgsense_r8_brain_knee_prototable | 50/50 | brain | 35.05 | 37.35 | 0.9152 | 0.9639 | 0.0170 | 0.0075 |
| loupe2d_cgsense_r8_brain_knee_prototable | 50/50 | knee | 32.42 | 33.64 | 0.9237 | 0.9146 | 0.0182 | 0.0138 |
| loupe2d_cgsense_r8_brain_knee_prototable_bilinear_const_100ep | 100/100 | brain | 35.07 | 37.37 | 0.9151 | 0.9637 | 0.0170 | 0.0075 |
| loupe2d_cgsense_r8_brain_knee_prototable_bilinear_const_100ep | 100/100 | knee | 32.50 | 33.77 | 0.9248 | 0.9160 | 0.0179 | 0.0134 |
| loupe2d_cgsense_r8_brain_knee_shared | 50/50 | brain | 34.93 | 37.13 | 0.9131 | 0.9617 | 0.0174 | 0.0079 |
| loupe2d_cgsense_r8_brain_knee_shared | 50/50 | knee | 32.69 | 34.08 | 0.9291 | 0.9219 | 0.0172 | 0.0126 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep | 50/50 | brain | 34.93 | 37.13 | 0.9130 | 0.9614 | 0.0174 | 0.0079 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep | 50/50 | knee | 32.71 | 34.09 | 0.9290 | 0.9216 | 0.0172 | 0.0126 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43 | 50/50 | brain | 34.92 | 37.11 | 0.9129 | 0.9613 | 0.0174 | 0.0079 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_50ep_seed43 | 50/50 | knee | 32.72 | 34.12 | 0.9292 | 0.9219 | 0.0171 | 0.0125 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep | 100/100 | brain | 34.92 | 37.11 | 0.9128 | 0.9612 | 0.0174 | 0.0080 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep | 100/100 | knee | 32.72 | 34.11 | 0.9290 | 0.9216 | 0.0171 | 0.0125 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_nomask_50ep | 50/50 | brain | 36.55 | 37.22 | 0.9494 | 0.9634 | 0.0101 | 0.0078 |
| loupe2d_cgsense_r8_brain_knee_shared_bilinear_nomask_50ep | 50/50 | knee | 34.98 | 35.14 | 0.9505 | 0.9441 | 0.0101 | 0.0096 |
| loupe2d_cgsense_r8_brain_knee_shared_cont | 50/50 | brain | 34.89 | 37.06 | 0.9126 | 0.9610 | 0.0175 | 0.0080 |
| loupe2d_cgsense_r8_brain_knee_shared_cont | 50/50 | knee | 32.68 | 34.04 | 0.9285 | 0.9209 | 0.0173 | 0.0127 |
| loupe2d_cgsense_r8_brain_knee_shared_seed43 | 50/50 | brain | 34.91 | 37.11 | 0.9130 | 0.9616 | 0.0174 | 0.0079 |
| loupe2d_cgsense_r8_brain_knee_shared_seed43 | 50/50 | knee | 32.71 | 34.10 | 0.9292 | 0.9219 | 0.0172 | 0.0126 |
| loupe2d_varnet_r8_brain_knee_anatcond_100ep | 100/100 | brain | 36.72 | 40.80 | 0.9341 | 0.9827 | 0.0129 | 0.0034 |
| loupe2d_varnet_r8_brain_knee_anatcond_100ep | 100/100 | knee | 34.35 | 37.68 | 0.9540 | 0.9590 | 0.0121 | 0.0058 |
| random_mask_varnet_r8_brain_knee | 50/50 | brain | 34.86 | 37.08 | 0.9220 | 0.9707 | 0.0171 | 0.0075 |
| random_mask_varnet_r8_brain_knee | 50/50 | knee | 33.48 | 35.80 | 0.9475 | 0.9505 | 0.0147 | 0.0087 |
| varnet_caipi2d_r4_promptmr_blind_seed42 | 50/50 | brain | 36.70 | 41.69 | 0.9315 | 0.9859 | 0.0120 | 0.0024 |
| varnet_caipi2d_r4_promptmr_blind_seed42 | 50/50 | knee | 33.93 | 37.85 | 0.9547 | 0.9619 | 0.0110 | 0.0047 |
| varnet_caipi2d_r4_promptmr_cond_seed42 | 50/50 | brain | 36.65 | 41.61 | 0.9312 | 0.9859 | 0.0121 | 0.0024 |
| varnet_caipi2d_r4_promptmr_cond_seed42 | 50/50 | knee | 33.93 | 37.82 | 0.9547 | 0.9617 | 0.0110 | 0.0047 |
| varnet_caipi2d_r4_promptmr_cond_seed43 | 50/50 | brain | 36.70 | 41.73 | 0.9315 | 0.9861 | 0.0120 | 0.0024 |
| varnet_caipi2d_r4_promptmr_cond_seed43 | 50/50 | knee | 33.94 | 37.86 | 0.9549 | 0.9619 | 0.0110 | 0.0047 |
