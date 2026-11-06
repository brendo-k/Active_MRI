# How much less wall-clock does Muon + LayerNorm VarNet + LOUPE need to reach its 100-epoch quality? (2026-11-04)

| | |
|---|---|
| Code | results produced at `60eb6dc` (muon_layer_speed_curves: the panel title names the reference); branch `exp/muon_layer_speed` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 13 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 29 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |
| Training commit | not recorded for these runs (they predate `run_info.json`) |

- **Branch:** `exp/muon_layer_speed`, squash-merged into `main`. Results come from the commit in the provenance block. The runs used
  the two code snapshots listed under Runs compared.
- **Status:** adopted 2026-11-06. On 2026-11-04 the user set Muon + LayerNorm as the project default, ahead of the
  100-epoch 2×2. The branch carries that change in `553f760`. Proposal frozen 2026-11-03 23:00 (`proposal.md`). Everything here is
  seed 42: seed 43 and `e50_step` did not fit before the 08:00 limit, so any criterion that needs two seeds stays open.
- **Verdict (user, 2026-11-06):** merge, with Muon + LayerNorm as the default. The finished 2×2 supports that default: +0.35 and
  +0.33 dB over Adam + InstanceNorm at 100 epochs (`reports/muon_layernorm_100ep/report.md`). The exact step-time code comes with the
  merge, so the deviation from the frozen 1e-5 bar (Results 2) came in with it rather than as a separate decision. The schedule,
  batch size and `recon.chans` keep their defaults, and the EMA option stays off.

## Proposal vs outcome

Frozen criteria, read as same-seed paired deltas against the 100-epoch reference. A criterion over two seeds is decided on one
seed only where that seed already fails it. For example, "neither seed below −0.20 dB" fails on one seed. A mean over two seeds stays open.

| criterion (frozen) | arm | evidence | met? |
|---|---|---|---|
| Exact change: match to float rounding (bitwise, or ≤1e-5 relative) **and** bench step or epoch time −5% or more | exact bundle | speed: real epochs −12% (301 vs 343 s), bench step −18.8%. Match: ≤9.4e-9 in float64, but 4.6e-5 to 5.8e-4 in fp32 and 0.46% to 8.4% in bf16. A same-math control moves a 2-epoch training as far as the new code does (2.71e-2 vs 2.73e-2) | speed **met**; 1e-5 bar **not met as written**, met in float64 only |
| Bitwise-identical removal of unused work: adopted at any gain | unused forward work | not shown bitwise in the real pipeline, because it was only run inside the bundle | not decided |
| Faster: mean ΔPSNR ≥ −0.10 dB, neither seed below −0.20 dB, ΔSSIM ≥ −0.002, wall-clock ≤50% | `e30_bs8` | −0.142 dB [−0.180, −0.104], ΔSSIM −0.0023, 28% | not met on seed 42; still open, seed 43 would need about −0.06 dB or better and ΔSSIM −0.0017 or better |
| same | `e30_step` | −0.307 dB [−0.361, −0.252], ΔSSIM −0.0045, 23% | **not met, decided** |
| same | `e30_cos` | −0.424 dB [−0.496, −0.351], ΔSSIM −0.0062, 27% | **not met, decided** |
| same | `e30_cos_muon2e3` | −0.457 dB [−0.533, −0.381], ΔSSIM −0.0066, 24% | **not met, decided** |
| same | `e50_step` | not run | not decided |
| Worse: mean ΔPSNR below −0.25 dB | `e30_step`, `e30_cos`, `e30_cos_muon2e3` | −0.307, −0.424, −0.457 dB on one seed | provisional: met on seed 42, the mean needs seed 43 |
| same | `e30_bs8` | −0.142 dB | not met on seed 42 |
| EMA adopted: ≥0.05 dB over the last weights in both seeds of an arm | `e30_bs8`, `e30_step`, `e30_cos`, `e30_cos_muon2e3`, `e30_step_c16` | −0.024, −0.052, −0.022, −0.016, −0.024 dB | **not met, decided**: no arm can clear it in both seeds |
| no frozen criterion (post-freeze arm) | `e30_step_c16` | −0.394 dB [−0.459, −0.329], ΔSSIM −0.0055, 19%; −0.087 dB [−0.103, −0.072] against `e30_step` | reported, not judged |

**Recommendation** (revised 2026-11-04): **merge**, with Muon + LayerNorm as the project default.

The default is the user's decision, made ahead of the 100-epoch 2×2. This study took the recipe as given and does not test it
against Adam + InstanceNorm. The evidence so far is `reports/varnet_train_speed/report.md` and seed 42 of the 2×2, which ended at
33.86 dB / 0.863 against 33.51 dB / 0.859 in trainer validation at 100 epochs (`curves_reference.json`). If the finished 2×2
disagrees, reverting `553f760` undoes the change.

That commit sets five YAML defaults:

| setting | new default | why |
|---|---|---|
| `training.optimizer` | `muon` | the user's decision |
| `recon.norm`, `sampler.sens_norm` | `layer` | the user's decision |
| `training.fused_adam` | `true` | changes no math; its speed-up was measured on this recipe only |
| `training.channels_last_unets` | `sens` | the profile gives −13.4% with it and −8.1% without; measured with LayerNorm only |

With those five removed from `e30_step_s42`'s overrides, `train.py --print-config` resolves to the config that run trained with;
only `output.out_dir` and `output.run_name` differ. A run that goes back to InstanceNorm should also set
`training.channels_last_unets=false`, because `reports/varnet_train_speed/report.md` found channels-last +11% slower there.
Caveats, under **The default change**, lists where the new default is untested or refuses a checkpoint.

Merge the exact step-time code with it. That is 12% off every real epoch on the recipe, and it nearly cancels what the recipe
costs: 301 s per epoch, against 340 s on the old step code and 298 s for Adam + InstanceNorm (`curves_reference.json`; different
GPUs, so not a paired measure). The price is that bf16 runs after the merge do not reproduce runs before it bitwise, and the
numeric 1e-5 bar holds only in float64. That is a deviation from the frozen wording, so it is the user's call. Results 2 gives
the evidence.

Do **not** change the default schedule or batch size yet. `e30_bs8` is the only arm still able to pass, and it needs seed 43.
Do not adopt the EMA or `training.muon_lr=0.002`, and leave `recon.chans` at 18 until 16 channels has been run to convergence.
The EMA option can stay on the branch, off by default, since it costs nothing when off. Dropping it is one commit if it is not wanted.

## Question

The request was to speed up training of the main operating point, both the forward and backward pass (step time) and the number of
steps to convergence, on top of Muon + LayerNorm, which the user set as the baseline without waiting for the 100-epoch 2×2.
The frozen question is how much less wall-clock 1-step LOUPE + 12-cascade E2E VarNet (R=8, full local v1 FLAIR + knee) needs to
reach its 100-epoch quality. It matters because effects on the roadmap are 0.05–0.2 dB against 0.3–0.5 dB seed spread, so
every question needs several seeds, and a 100-epoch run takes 11.3 h.

Conventions: PSNR_Masked and SSIM_Masked inside the apodized per-anatomy metric mask. Δ is arm minus reference, per slice, same
seed, averaged per volume. The 95% CI is over volumes for fixed checkpoints, so it holds no run-to-run variance. A "step" is one optimizer update.

## Runs compared

All runs: 1-step LOUPE (`loupe_2d`, VD init power 4), 12-cascade E2E VarNet with 18 channels unless stated, R=8, full local v1 FLAIR +
knee at 320², `training.optimizer=muon recon.norm=layer sampler.sens_norm=layer`, seed 42, batch 16 unless stated. Arms also keep
an EMA (decay 0.999) and validate every 5 epochs. The reference validates every epoch. Checkpoints live under
`~/data/seqmri_checkpoints/`. wandb ids are in `manifest.json`.

| run | checkpoint | change from the reference | epochs | code | PSNR / SSIM (paired eval) | trainer final |
|---|---|---|---|---|---|---|
| reference `ml100_muon_layer_s42` | `muon_layernorm_100ep/muon_layer_s42` | none; lr step at 90 | 100 | main's fast path | 33.86 / 0.863 | 33.86 / 0.863 |
| `e30_cos_s42` | `muon_layer_speed/e30_cos_s42` | cosine lr | 30 | `0920f8b`, old step code | 33.44 / 0.857 | 33.43 |
| `e30_step_s42` | `muon_layer_speed/e30_step_s42` | lr step at 27 | 30 | `e61a166`, exact bundle | 33.55 / 0.859 | 33.55 |
| `e30_bs8_s42` | `muon_layer_speed/e30_bs8_s42` | batch 8, same lr, lr step at 27 | 30 | `e61a166`, exact bundle | 33.72 / 0.861 | 33.71 |
| `e30_cos_muon2e3_s42` | `muon_layer_speed/e30_cos_muon2e3_s42` | cosine lr, `training.muon_lr=0.002` | 30 | `e61a166`, exact bundle | 33.40 / 0.857 | 33.40 |
| `e30_step_c16_s42` (post-freeze) | `muon_layer_speed/e30_step_c16_s42` | as `e30_step`, plus `recon.chans=16`, all U-Nets channels-last | 30 | `e61a166`, exact bundle | 33.46 / 0.858 | 33.46 |
| `ref_maskof_e30_step_s42` (exploratory) | `muon_layer_speed/ref_maskof_e30_step_s42` | reference networks, `e30_step`'s mask | none | scored only | 33.70 / 0.862 | n/a |
| `e30_step_maskof_ref_s42` (exploratory) | `muon_layer_speed/e30_step_maskof_ref_s42` | `e30_step`'s networks, reference mask | none | scored only | 33.54 / 0.859 | n/a |

The exact bundle, in the order it was built:
1. Unused forward work removed for LOUPE: the sampler's input image, the zero-filled image in training, the k-space prediction after the last step.
2. `_foreach_` ops in Muon and the gradient statistics, and `training.fused_adam=true`.
3. A hand-written backward pass for the cascades' coil operations (`activemri/reconstructors/fused_coil_ops.py`): compiled kernels, running gradient sums, unscaled FFTs with the scale folded into the kernels.
4. `gc.freeze()` after the first epoch.
5. Channels-last for the sensitivity net only (`training.channels_last_unets=sens`), whose channel counts are multiples of 8.

## Methods

1. **Profiles.** `scripts/diagnostics/muon_layer_speed_profile.py`: per-phase median milliseconds of a training step on an idle GPU,
   at each stage of the bundle, cumulative.
2. **Exactness, one step.** `scripts/diagnostics/muon_layer_speed_gradcheck.py` and `scripts/diagnostics/muon_layer_speed_fused_check.py`:
   relative distance between the hand-written backward and autograd, for outputs and gradients, at full size in float64, fp32 with TF32 off, and bf16.
3. **Exactness, a real training.** `scripts/diagnostics/muon_layer_speed_equivalence.py`: two epochs on 5% of the data through `train.py`
   with the autotuner off, four times. The old code twice, then the new code with and without the arms' options. Those options change
   no math, so that last pair shows how far a pure rounding change moves a training.
4. **Bench.** `scripts/diagnostics/muon_layer_speed_bench.py`: each arm in its own process on an idle GPU, 14 warm-up batches, then 24 timed
   batches and one validation pass on 20% of the data.
5. **Convergence arms.** Trained from `queue_jobs.txt` by `scripts/launch/varnet_train_speed_worker.sh` on GPUs 2–3, one change from the reference each. Crashed runs would rerun in full; none crashed.
6. **Paired scores.** `scripts/eval/varnet_train_speed_eval.py`, driven by `queue_rolling_eval.sh`: uncompiled fp32, 2 seeded mask draws, all
   1938 validation slices (80 volumes), per condition, one val-loader batch size for every run so the mask draws match. `<arm>_ema_s42` is the EMA checkpoint scored as a run of its own.
7. **Mask swap (exploratory, added after the freeze).** `scripts/diagnostics/muon_layer_speed_swap_masks.py` crosses `e30_step`'s and the
   reference's masks and networks into a 2×2 without retraining. A swapped network sees a mask it never trained on, so each swapped cell also carries a mismatch cost.
8. **Masks.** `scripts/diagnostics/varnet_train_speed_masks.py`: p(k) = sigmoid(slope × logits), correlation with the reference's mask, mean p by radial band.
9. **Tables and figure.** `scripts/eval/varnet_train_speed_curves.py` reads wandb and the run logs. `scripts/eval/muon_layer_speed_summary.py` decides the criteria
   and writes `summary.md`, which holds every table below. `scripts/plots/muon_layer_speed_curves.py` draws the figure.

## Results

### 1. Step time

| stage (cumulative) | step ms | vs control | forward | backward | optimizer | sens net | steady GiB |
|---|---|---|---|---|---|---|---|
| control, old code | 1379.2 | | 459.0 | 789.1 | 88.3 | 323.0 | - |
| + unused work, `_foreach_`, fused Adam (`cheap`) | 1317.6 | −4.5% | 434.4 | 788.1 | 65.9 | 322.2 | - |
| + hand-written backward (`fused_backward`) | 1308.5 | −5.1% | 412.4 | 752.3 | 101.3 | 323.2 | 35.3 |
| + `gc.freeze` (`gc_freeze`) | 1267.3 | −8.1% | 411.1 | 750.4 | 72.1 | 322.9 | 35.3 |
| + sensitivity net channels-last (`group_senscl`) | 1194.0 | −13.4% | 396.9 | 704.5 | 64.6 | 264.5 | 35.3 |
| 16 channels, every U-Net channels-last (`chans16_nhwc`), a model change | 947.4 | −31.3% | 334.8 | 531.1 | 49.4 | 265.9 | 33.6 |

Most of the saving is spread across small changes. Only the last stage clears 5% alone (1267.3 to 1194.0 ms, −5.8%).

| measure | old code | exact bundle | change | 16 channels | change |
|---|---|---|---|---|---|
| Real training epoch, median train s | 343 (`e30_cos`) | 301 (`e30_step`) | −12% | 234 (`e30_step_c16`) | −32% |
| Bench step s | 1.781 | 1.446 | −18.8% | 1.207 | −32.3% |
| Bench validation pass s (25 batches) | 19.4 | 13.7 | −30% | 15.8 | −19% |
| Bench steady peak GiB | 36.3 | 36.2 | | 34.5 | |

Real epochs are the number to plan on. The bench agrees on direction, but its absolute step times are higher than real runs (1.45 s against about 1.22 s),
and it overstates the gain. It times a truncated 24-batch epoch, including that epoch's start-up. Whether that explains the gap was not measured.

Tried and dropped:
- A LayerNorm that stays in bf16: +7.1% to +7.8%, slower.
- Channels-last on every U-Net at 18 channels: −3.4%, which is worse than the sensitivity net alone.
- NUMA pinning: 1288 ms against 1309 ms unpinned, −1.5%, from before the reference-cycle fix (`profile_fused_pinned.json`). Not pursued.

### 2. Exactness

Relative distance, hand-written backward against autograd, one step at full size:

| precision | output | reconstructor gradients | sensitivity net gradients | mask gradient |
|---|---|---|---|---|
| float64 | 1.6e-09 | 8.2e-10 | 9.4e-09 | 7.3e-09 |
| fp32, TF32 off | 5.2e-07 | 4.6e-05 | 5.4e-05 | 5.8e-04 |
| fp32, TF32 off, smooth loss | 5.2e-07 | 5.1e-05 | 6.7e-05 | 5.3e-04 |
| bf16, as trained | 6.5e-03 | 4.6e-03 | 5.3e-03 | 8.4e-02 |

The distance follows the precision: about 1e-9 in float64, 5e-5 in fp32, 0.5% in bf16. In float64 the two passes agree, which is the evidence that the math is the same.
The 1e-5 bar cannot be read as met in the precision training uses.

Two epochs on 5% of the data, relative distance between final weights:

| pair | whole model | median parameter | worst parameter | mask |
|---|---|---|---|---|
| old vs old, rerun | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| old vs new, with the arms' options | 2.73e-02 | 9.34e-03 | 1.37e-01 | 1.80e-03 |
| old vs new, without the options | 2.67e-02 | 8.45e-03 | 1.48e-01 | 1.74e-03 |
| new without vs new with the options (same math) | 2.71e-02 | 8.75e-03 | 1.36e-01 | 1.79e-03 |

Last train loss 0.08205 (old, both runs), 0.08215 (new with the options), 0.08212 (new without). Last val loss 0.101, 0.101, 0.1015.

Same code, same settings gives the same weights bitwise, so the old-to-new distance is not run noise. The control that changes no math moves a training just as far, 2.71e-2 against 2.73e-2, so the new code's distance is no larger than a pure rounding change's. That supports reading it as rounding. It does not meet the 1e-5 bar, and it says nothing about a full-length run.

### 3. Convergence arms

| arm | PSNR | SSIM | ΔPSNR [95% CI] | ΔSSIM | Δ brain / knee-PD / knee-PDFS | median train s per epoch | wall-clock h | share of reference | steady h | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `e30_bs8` | 33.717 | 0.8609 | −0.142 [−0.180, −0.104] | −0.0023 | −0.11 / −0.31 / +0.01 | 332 | 3.19 | 28% | 2.89 | provisional inconclusive |
| `e30_step` | 33.552 | 0.8587 | −0.307 [−0.361, −0.252] | −0.0045 | −0.30 / −0.53 / −0.08 | 301 | 2.65 | 23% | 2.62 | not faster, decided; provisional worse |
| `e30_cos` | 33.435 | 0.8570 | −0.424 [−0.496, −0.351] | −0.0062 | −0.41 / −0.73 / −0.12 | 343 | 3.02 | 27% | 2.99 | not faster, decided; provisional worse |
| `e30_cos_muon2e3` | 33.402 | 0.8566 | −0.457 [−0.533, −0.381] | −0.0066 | −0.50 / −0.75 / −0.12 | 303 | 2.69 | 24% | 2.66 | not faster, decided; provisional worse |
| `e30_step_c16` (post-freeze) | 33.465 | 0.8577 | −0.394 [−0.459, −0.329] | −0.0055 | −0.42 / −0.65 / −0.11 | 234 | 2.10 | 19% | 2.07 | post-freeze, no frozen criterion |

The reference took 11.30 h (9.50 train + 1.80 val), or 9.86 h at the arms' val cadence. Against that, the shares are 21–32%, still under the 50% bar.
"Steady h" uses the median epoch, which removes the stretch below where scoring shared `e30_bs8`'s GPU.

No 30-epoch arm reached the bar. `e30_bs8` came closest, 0.04 dB and 0.0003 SSIM short of it. The knee-PD deficit is the largest in every arm, and knee-PDFS barely moves.

Between arms, seed 42, per slice (exploratory, no frozen criterion):

| pair | same code snapshot? | ΔPSNR [95% CI] | ΔSSIM |
|---|---|---|---|
| `e30_bs8` − `e30_step` | yes | +0.165 [+0.143, +0.187] | +0.0022 |
| `e30_step_c16` − `e30_step` | yes | −0.087 [−0.103, −0.072] | −0.0010 |
| `e30_step` − `e30_cos` | no | +0.117 [+0.097, +0.137] | +0.0018 |
| `e30_cos_muon2e3` − `e30_cos` | no | −0.033 [−0.046, −0.020] | −0.0004 |

Trainer val PSNR_Masked by epoch:

| run | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| 100-epoch reference | 32.31 | 32.76 | 33.03 | 33.25 | 33.34 | 33.48 |
| `e30_cos` | 32.33 | 32.86 | 33.15 | 33.31 | 33.40 | 33.43 |
| `e30_step` | 32.27 | 32.85 | 33.03 | 33.23 | 33.40 | 33.55 |
| `e30_bs8` | 32.48 | 33.06 | 33.26 | 33.37 | 33.50 | 33.71 |
| `e30_cos_muon2e3` | 32.52 | 32.95 | 33.13 | 33.30 | 33.38 | 33.40 |
| `e30_step_c16` | 32.46 | 32.86 | 33.06 | 33.18 | 33.31 | 33.46 |

`e30_bs8` stays above the reference at every epoch, and its final score matches the reference's smoothed curve at epoch 53. `e30_step` matches epoch 38, `e30_step_c16` epoch 30, and the two cosine arms epochs 27–28.
At constant lr the reference's smoothed curve reaches the bar (33.76 dB) at epoch 65, about 5.7 h at `e30_step`'s epoch time. That is an estimate from the curve, not a run.

### 4. Where the 0.31 dB gap sits: mask swap

Paired score PSNR / SSIM, seed 42, from `eval.json`:

| | 100-epoch networks | 30-epoch networks (`e30_step`) |
|---|---|---|
| **100-epoch mask** | 33.86 / 0.863 | 33.54 / 0.859 |
| **30-epoch mask** | 33.70 / 0.862 | 33.55 / 0.859 |

| change | with the other part fixed at | ΔPSNR [95% CI] |
|---|---|---|
| 30-epoch to 100-epoch networks | the 100-epoch mask | +0.323 [+0.263, +0.384] |
| same | the 30-epoch mask | +0.149 [+0.099, +0.199] |
| 30-epoch to 100-epoch mask | the 100-epoch networks | +0.158 [+0.137, +0.179] |
| same | the 30-epoch networks | −0.017 [−0.038, +0.005] |

Both routes around the 2×2 sum to the 0.31 dB gap. The networks account for 0.15–0.32 dB of it and the mask for −0.02 to +0.16 dB, depending on the route.
This is a bracket, not a split: the difference between routes is the mismatch cost plus co-adaptation of network and mask.

Masks at 30 epochs are still moving. Correlation of p(k) with the 100-epoch mask is 0.73–0.78:

| run | mean p | correlation with reference | 0–20 px | 20–40 | 40–80 | 80–120 | 120–160 | 160–230 |
|---|---|---|---|---|---|---|---|---|
| reference | 0.1328 | 1.000 | 0.803 | 0.513 | 0.285 | 0.153 | 0.070 | 0.003 |
| `e30_cos` | 0.1302 | 0.762 | 0.879 | 0.573 | 0.309 | 0.145 | 0.051 | 0.000 |
| `e30_step` | 0.1292 | 0.775 | 0.866 | 0.555 | 0.292 | 0.148 | 0.055 | 0.000 |
| `e30_bs8` | 0.1291 | 0.741 | 0.839 | 0.541 | 0.286 | 0.149 | 0.058 | 0.001 |
| `e30_cos_muon2e3` | 0.1294 | 0.738 | 0.887 | 0.583 | 0.305 | 0.141 | 0.051 | 0.000 |
| `e30_step_c16` | 0.1299 | 0.726 | 0.888 | 0.577 | 0.298 | 0.145 | 0.053 | 0.000 |

Every short run keeps more of its budget at the centre than the reference, and less at high frequency.

### 5. EMA

| arm | EMA minus last weights, dB |
|---|---|
| `e30_bs8` | −0.024 |
| `e30_step` | −0.052 |
| `e30_cos` | −0.022 |
| `e30_cos_muon2e3` | −0.016 |
| `e30_step_c16` | −0.024 |

The EMA was below the last weights in every arm, against an expected +0.03 to +0.1 dB.

## Conclusions

- The exact bundle takes 12% off a real epoch (343 to 301 s) and 18.8% off a bench step, with no change in peak memory. Evidence: Results 1.
- In float64 the hand-written backward agrees with autograd to under 1e-8, so it is the same math. In bf16 it is not the same numbers, and a change with no math in it moves a 2-epoch training just as far (2.71e-2 against 2.73e-2). Evidence: Results 2.
- No schedule tested reaches 100-epoch quality in 30 epochs on seed 42. Evidence: Results 3.
- At 30 epochs, batch 8 was worth +0.165 dB over batch 16 for 10% more time per epoch, the largest single effect measured here, on one seed. It is the only arm still able to pass.
- The largest part of the gap at 30 epochs sits in the networks (0.15–0.32 dB), with the mask at −0.02 to +0.16 dB. The mask is not shown to be irrelevant. Evidence: Results 4.
- The EMA at decay 0.999 does not help. Evidence: Results 5.
- 16 channels takes a real epoch from 301 to 234 s on top of the exact bundle (−22%) and costs 0.087 dB [0.072, 0.103] at 30 epochs, same snapshot, one seed. It is a model change with no frozen criterion, and whether that cost holds at convergence is untested.
- Hypothesis, untested: convergence here is limited by the number of updates, not samples. `e30_bs8` took about 14,800 steps and matches the reference at epoch 53, about 13,100 steps.
- Hypothesis, untested: `e30_step`'s EMA is the worst of the four because a 0.999 horizon, about 4 epochs at batch 16, straddles its lr drop at epoch 27.
- Hypothesis, untested: Muon is why 0.5% gradient rounding becomes a 2.7% weight distance in two epochs. Newton–Schulz gives every singular direction the same size, including the ones rounding dominates.

## Caveats

- **One seed.** Every convergence number is seed 42. CIs are over volumes for fixed checkpoints and hold no run-to-run variance, and every effect here sits inside the 0.3–0.5 dB seed spread in `AGENTS.md`. Treat between-arm differences as candidates. The verdicts marked provisional need seed 43, and `ml100_muon_layer_s43` does not exist yet.
- **Cross-snapshot pairs.** `e30_cos` ran on the old step code and the other arms on the new. `e30_step` − `e30_cos` and `e30_cos_muon2e3` − `e30_cos` therefore mix their change with a rounding change, which by itself decorrelates a training (Results 2). So the data do not show that a higher Muon lr fails to help. Only `e30_bs8` − `e30_step` and `e30_step_c16` − `e30_step` are same-snapshot.
- **The exact claim.** No full-length run compares old and new code on one schedule. The evidence is one step in three precisions and two epochs on 5% of the data. An ad-hoc rerun of the old code with no committed script once drifted by 2.3e-3 for a reason not found, so bitwise reproducibility is not guaranteed even for the same code.
- **The bundle.** The changes were adopted together. Most were individually under the 5% bar. The stages were profiled from different commits, which `run.sh` lists, and the removal of unused work was not separately shown bitwise.
- **Post-freeze additions.** `e30_step_c16`, the mask swap and the equivalence job were added after the freeze. `e30_step_c16` first read `e30_cos_c16` and was changed to its current schedule before it started. The mask swap is exploratory. Its cells carry a mismatch cost, and it covers one pair of checkpoints.
- **Deviations from the proposal's compute plan.** The job order changed: `e30_step_c16`, the equivalence job and the bench went ahead of `e50_step`, so `e50_step` and seed 43 never ran. `e30_step_c16` finished after the 07:15 target, inside the 08:00 limit. Scoring ran on a rolling loop rather than at the end.
- **`e30_bs8` wall-clock.** Its epochs 2–5 shared GPU 3 with the first scoring pass and took 558–658 s against about 335 s, adding about 15 min. The "steady h" column corrects for that.
- **`e30_cos` wall-clock.** It carries the old code's 343 s per epoch, so its 27% share is not comparable with the others' 23–24%.
- **Trainer against paired eval.** The eval is uncompiled fp32 with 2 seeded draws, and the trainer is bf16. `e30_bs8`'s trainer validation also uses a different batch size, so different mask draws. "Matches reference epoch" uses trainer numbers and is approximate.
- **Validation cadence.** The reference validated every epoch and the arms every 5, which helps the arms' shares (Results 3 gives both).
- **Baseline.** Muon + LayerNorm is taken as true on the user's instruction. The 100-epoch 2×2 was still running on GPUs 0–1 throughout, and they were not used.
- **The default change.** It rests on that instruction, not on this study.
  - The recipe was measured on 12-cascade VarNet + LOUPE at R=8 only. `recon=unet` (the config's default group), `recon=cg_sense`, `recon=sdum` and the PromptMR backbone also change, untested: they now train under Muon, and their sensitivity nets take LayerNorm. The PromptMR U-Net has no norm layers of its own, so `recon.norm` does nothing there.
  - Saved checkpoints keep their own settings. Evals rebuild the model from the options stored in the checkpoint, and the dataclass and `builders.py` fallbacks stay Adam + InstanceNorm for options saved before those fields existed.
  - Two paths now stop with an error on a mismatch. `output.pretrained_recon` with an InstanceNorm checkpoint needs `recon.norm=instance sampler.sens_norm=instance`. PPO and DQN build their environment's networks with InstanceNorm and ignore `recon.norm`, so they cannot load a LayerNorm checkpoint.
  - Launch scripts that relied on the old defaults now train the new recipe. The earlier default is `training.optimizer=adam recon.norm=instance sampler.sens_norm=instance training.channels_last_unets=false`.
  - It was checked on CPU only, with the test suite and the config comparison in the recommendation, because GPUs 0–1 hold the 2×2 and GPUs 2–3 were not cleared for use. No training ran on commit `553f760` itself, though `e30_step_s42` ran the same resolved config at `e61a166`.
- **Knee-PDFS SSIM.** It is about 0.78 in the reference and every arm, below the 0.80 line in `AGENTS.md`. Overall SSIM is 0.86. It predates this study: the controls in `reports/varnet_train_speed/eval.json` sit at 0.71–0.72. It was not debugged here. Hypothesis, untested: fat-suppressed knee is the lowest-SNR contrast, and SSIM against a noisy target is capped.
- **Uncommitted code.** The bf16 LayerNorm behind `profile_layer_bf16*.json` was never committed; the class is kept in `layer_bf16_reverted.txt`.
- **Training commit row.** The provenance block's row refers only to the EMA and mask-swap directories, which are scored-only and have no training of their own. Every trained run has its commit in `manifest.json`: `0920f8b` for `e30_cos`, `e61a166` for the other arms, `91932da` for the reference.

## Figures

- `figures/curves.png`: trainer val PSNR_Masked of the reference and the arms, against epoch and against wall-clock, with the bar at the reference's final minus 0.10 dB. The reference validated every epoch, so its wall-clock line sits to the right of the arms.

## Next steps

1. Run seed 43 of `e30_bs8` and pair it with `ml100_muon_layer_s43`, which now exists. That settles the one open verdict.
2. Test the updates hypothesis with one longer and one smaller-batch arm: `e50_bs8` (about 4.8 h, about 42% of the reference) and `e30_bs4`, both with the lr step at 90% of the run.
3. If 16 channels is still of interest, run it through a full schedule at two seeds, since 30 epochs cannot say what it costs at convergence.
4. Find out why knee-PD carries the largest deficit in every arm, since that is where the remaining dB are.

## Reproduce

`bash reports/muon_layer_speed/run.sh` from the repo root, with the two code snapshots checked out as its header describes. It needs the checkpoints in
`~/data/seqmri_checkpoints/muon_layer_speed` and `~/data/seqmri_checkpoints/muon_layernorm_100ep`, one idle GPU and wandb access.
Check it with `python scripts/report_manifest.py check reports/muon_layer_speed`.
