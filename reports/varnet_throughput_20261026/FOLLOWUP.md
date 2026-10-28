# Full-resolution follow-up: two- and twelve-cascade VarNet

Date: 2026-10-26. This follow-up completes the full-resolution cuDNN search, normalization substitutions, and fused FFT-shift experiments requested after the initial report. All arms use 256x256 images. The first suite used the two-cascade BF16 VarNet; a short follow-up then compared baseline, fused shifts, and moments plus fused shifts on the target twelve-cascade configuration. The 128x128 sensitivity experiment had already lost about 0.1 dB and is not reconsidered here.

## Clean same-GPU timing repeats

After the shared-GPU suite, all GPUs were idle. Five two-cascade arms then ran serially on GPU 2 with identical settings (skip 10, profile 6, time 20 steps, replay one batch). The bracketing baseline was stable at 302.0/303.4 ms per step.

| Two-cascade arm | Step ms | GPU work ms/step | Sensitivity forward+backward GPU ms |
|---|---:|---:|---:|
| Baseline before | 302.0 | 293.6 | 146.9 |
| Exhaustive cuDNN search (`benchmark_limit=0`) | 299.2 | 292.8 | 144.9 |
| `SpatialMomentsNorm` | 274.1 | 265.2 | 117.2 |
| Fused FFT shifts | 294.0 | 282.7 | 147.1 |
| Baseline after | 303.4 | 295.2 | 148.1 |

Relative to the bracketing baseline mean, moments reduced step latency by 9.4% (about 10.4% more steps per second) and GPU work by 9.9%; fused shifts reduced latency by 2.9% (about 3.0% more steps per second) and GPU work by 4.0%. Exhaustive cuDNN search showed no useful gain. Earlier shared-GPU values were badly inflated by contention and should not be mixed with these clean repeats.

The original `peak_gib` records the process high-water allocation, including compilation, autotuning, and profiler work; it does not isolate steady-state training memory. A post-reset measurement over the final timed steps was added for the target-model comparison below. The older two-cascade process high-water was 17.89 GiB for moments versus about 8.74 GiB for baseline, but this alone does not show that steady-state activation memory doubled.

## Target twelve-cascade timing and memory

With GPU 2 still idle, baseline, fused-only, and moments plus fused shifts ran serially with the same short schedule (skip 6, profile 4, time 12 steps, replay one batch) and explicit `recon.num_cascades=12`.

| Twelve-cascade arm | Step ms | GPU work ms/step | Sensitivity forward+backward GPU ms | Steady post-reset peak GiB |
|---|---:|---:|---:|---:|
| Baseline | 825.7 | 811.4 | 147.5 | 19.547 |
| Fused FFT shifts | 780.4 | 767.2 | 146.8 | 19.547 |
| `SpatialMomentsNorm` + fused shifts | 752.3 | 738.9 | 117.3 | 19.542 |

Fused shifts reduced this short-run step latency by 5.5% (about 5.8% more steps per second); the combined candidate reduced latency by 8.9% (about 9.8% more steps per second) relative to baseline. The combined arm's sensitivity GPU time was about 30 ms/step lower, while fused shifts alone left that phase unchanged as expected. Post-reset timed-window memory was effectively the same in all three arms; overall process high-water was also about 19.55 GiB. This resolves the available-memory concern for this twelve-cascade configuration on the idle A6000, but these are short profiling runs, not epoch-level throughput or convergence results.

The isolated sensitivity-body benchmark on a real `[208,2,256,256]` input measured 291.1 ms for InstanceNorm and 224.1 ms for `SpatialMomentsNorm` (23% lower), with 2.95% relative L2 parameter-gradient difference. GroupNorm with `num_groups == num_channels` (one channel per group) measured 291.3 ms and had 4.02% relative L2 gradient difference. This microbenchmark was collected on the shared GPU and remains directional.

## Same-weight reconstruction validation

Validation used the same two-cascade, epoch-49 checkpoint across all 688 held-out slices (344 brain, 344 knee), with masked-pixel PSNR/SSIM. It is a local numerical comparison, not a retraining or convergence study.

| Candidate | Brain mean PSNR delta | Knee mean PSNR delta | Largest per-slice PSNR loss | Output check |
|---|---:|---:|---:|---|
| SpatialMomentsNorm | -0.00019 dB | -0.00016 dB | -0.02324 dB | max absolute output difference 0.1035 |
| GroupNorm, one channel per group | +0.00003 dB | +0.00014 dB | -0.02135 dB | max absolute output difference 0.0974 |
| fused FFT shifts | 0 | 0 | 0 | bitwise identical on all 688 slices |

Mean SSIM deltas for moments and GroupNorm were within 3.4e-6 of baseline; fused shifts were identical. These comparisons support near-identical inference metrics for the two normalization substitutions and exact inference equivalence for the fused-shift implementation, but they do not establish equal training trajectories. The earlier correctness check found bitwise-equal permutation outputs and input gradients, plus bitwise-equal reconstruction, loss and parameter gradients in the tested two-cascade pass.

The exhaustive cuDNN validation was paired directly using saved per-slice arrays after asserting that anatomy and volume ordering matched across all 688 slices. Deltas below are `benchmark_limit=0` minus default limit 10:

| Anatomy | Metric | Mean delta | Min slice delta | Max slice delta |
|---|---|---:|---:|---:|
| Brain (344 slices) | PSNR | +0.000138 dB | -0.016197 dB | +0.013504 dB |
| Brain | SSIM | +0.00000259 | -0.00013586 | +0.00018578 |
| Knee (344 slices) | PSNR | -0.000172 dB | -0.013817 dB | +0.010818 dB |
| Knee | SSIM | +0.00000055 | -0.00023462 | +0.00016064 |

This confirms negligible mean metric differences for exhaustive search, but its clean repeated timing showed no performance benefit.

The combined twelve-cascade arm was not independently run through the held-out validation set. Its quality evidence comes from the separately tested moments and fused-shift arms: moments had near-zero mean metric deltas on the fixed checkpoint, and fused shifts were bitwise identical on every validation output and on the tested full-pass gradients. This supports the combination as a candidate for a controlled training run, not a claim of unchanged convergence or learning trajectory.

## Recommendation

- Do not enable exhaustive cuDNN search; clean two-cascade timing and paired validation show no practical gain.
- Fused FFT shifts are the first candidate to consider because tested outputs and gradients are bitwise identical; they improved twelve-cascade throughput by about 5.8% in the short idle-GPU profile.
- Moments plus fused shifts produced the lowest measured twelve-cascade time and about 9.8% higher measured throughput, with no steady post-reset memory increase. It has not been evaluated or retrained as a combined arm, so use it only as a candidate for a controlled quality and convergence run.
- Production code and defaults remain unchanged. No other training process was stopped or signaled.

## Artifacts

Shared-GPU artifacts use `fullres_*`; idle two-cascade repeats are `idle_{baseline_before,autotune,moments,fused,baseline_after}.json` with `fullres_idle_suite.json`; idle target-model repeats are `idle12_{baseline_before,fused,combined}.json` with `fullres_idle12_suite.json`. Paired exhaustive validation is in `fullres_autotune_paired_validation.json`. Same-weight quality results are `fullres_validation.json` and `fullres_validation_slices.json`; the isolated normalization benchmark is `sens_norm_shared_gpu.json`.
