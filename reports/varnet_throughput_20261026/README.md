Measured on 2026-10-26: sensitivity estimation is the largest component of the two-cascade training step. The obvious additional layout/chunking switches did **not** improve throughput. A fused FFT-shift prototype preserves outputs and gradients bitwise in the checks performed, but its end-to-end speedup remains unverified because another training job started on its GPU.

**Experiment and scope**

Question: which implementation changes can improve training throughput without changing the reconstruction model, objective, or useful image quality? Evidence: CPU/CUDA traces of the actual `Sequential_Trainer.train_epoch`, unprofiled step latency, and identical-input/weight/upstream-gradient comparisons. This is technical profiling and local numerical validation, not a completed convergence study.

The baseline uses an RTX A6000, PyTorch 2.6.0+cu124, two VarNet cascades (18 channels, four pools), batch size 16, 256x256 images, balanced brain T2/knee data, 16 padded coils, fixed R=8 static-VD mask, seed 42. Both U-Net types use bfloat16, `torch.compile` and cuDNN benchmarking. Sensitivity estimation remains full resolution, with eight initial channels; padded coils are already skipped in buckets of 16. The profiled four batches run 192/208/224/240 coil images through the sensitivity U-Net, versus 16 images through each reconstruction U-Net.

The running 12-cascade `spacing_r8/...white_loupe_s0_50ep_seed42/train_sequential.log` was also inspected: it already enables compiled bfloat16 U-Nets and cuDNN benchmarking, batch size 16. Its sampling mask differs from the static-VD mask used here. Enabling these existing defaults is not an additional speedup for that job.

Each full-loop run warms up for 16 steps, warms the profiler for two steps, records eight steps, then runs 20 unprofiled steps. Four real batches are replayed, keeping the per-slice metric cache warm. Data loading uses four workers. The existing profiler reports the median of 19 post-export step intervals (one interval is excluded at the boundary). No validation, full epoch, or 12-cascade profiling was performed. Cold-cache preprocessing, new coil-bucket autotuning, first-time compilation, and unseen batches can make full epochs slower. Metrics/logging remain active, with W&B offline.

**Where the time goes**

Baseline: **308.7 ms/step**, approximately **51.8 slices/s**, peak allocated CUDA memory **9.43 GiB**. The profiled steps take 315.8 ms; GPU activity occupies 94.8% of the recorded window. GPU work totals 298.5 ms/step. The baseline ran on GPU 2 before the competing job there began.

| Phase | GPU ms/step | Share of GPU work |
|---|---:|---:|
| Sensitivity forward | 57.9 | 19.4% |
| Sensitivity backward | 93.9 | 31.5% |
| Cascade U-Net forwards, combined | 17.7 | 5.9% |
| Other VarNet forward operations | 23.8 | 8.0% |
| Loss, final combination and cascades backward | 73.2 | 24.5% |
| Remaining work | 32.0 | 10.7% |

Within sensitivity forward+backward, approximately **61.4 ms** is convolution/GEMM kernels, **44.0 ms** is cuDNN NCHW/NHWC conversion, **27.2 ms** is compiled fused kernels (largely normalization/activation and their gradients), and **14.0 ms** is other elementwise work. Its forward IFFT is only **0.8 ms**, and its centering permutations are **1.6 ms**. The sensitivity bottleneck is primarily the learned U-Net, not the ACS IFFT.

The shape-recording run attributes **61.9 ms** to sensitivity convolution calls at 256x256, and **27.0 ms** at 128x128, including the conversion kernels launched inside those calls. These overlap the preceding kernel categories; do not add them. The largest individual convolution families are the full-resolution 8-to-8-channel 3x3 backward calls (23.1 ms/step across the two occurrences), followed by 16-to-8-channel 3x3 backward (11.7 ms) and 8-to-8-channel forward (10.2 ms). Attribution excludes the metric convolutions.

Across the whole step, layout conversions cost **57.1 ms**, FFT centering rolls **25.5 ms**, and actual FFT kernels **12.6 ms**. The profile's broad `conv / gemm` category includes layout copies; `kernel_attribution.json` separates them.

`gradient_stats` appears to consume 146 ms on the CPU, but launches only 0.7 ms of GPU work and contains about 1.1 ms of GPU idle time. Its `.tolist()` waits for queued backward work. Deleting diagnostics would not recover 146 ms. Likewise, CPU waiting in visualization copies is not all incremental copy cost. Loader waiting is effectively zero in this replay experiment; that does not establish zero I/O overhead for fresh data.

**Measured alternatives**

| Experiment | Full step, ms | Conclusion |
|---|---:|---|
| Current settings, GPU 2 | 308.7 | Baseline |
| Disable layout optimization only for sensitivity compilation, GPU 3 | 306.8 | Within run/device variation; sensitivity GPU time actually unchanged |
| Sensitivity coil chunks of 32, GPU 2 | 314.1 | Slower; no useful whole-step memory reduction |
| Fused FFT shifts, GPU 2 | 452.6 | Invalid comparison: another training job started on GPU 2 |

The isolated sensitivity-body benchmark uses the captured first real coil batch `[208,2,256,256]`, identical initial weights, identical upstream gradients, five warmup iterations, and 20 CUDA-event-timed forward/backward iterations without optimizer updates. It gives:

| Body implementation | Median forward+backward, ms | Output relative L2 difference | Weight-gradient relative L2 difference |
|---|---:|---:|---:|
| Current | 131.6 | reference | reference |
| Disable layout optimization | 131.9 | 0 | 0 |
| Explicit channels-last input and weights | 156.2 | 1.11% | 3.03% |
| Coil chunks 32 | 142.8 | 1.10% | 3.19% |
| Coil chunks 64 | 138.9 | 1.10% | 3.19% |
| Coil chunks 128 | 135.3 | 0.37% | 1.54% |

Chunking preserves the mathematical model because these are independent coils with per-instance normalization and zero dropout, but bfloat16 kernel selection/gradient accumulation changes numerical results. The microbenchmark's later timings should be considered exploratory: other jobs were starting on the machine near its completion. The clean full-loop chunk-32 result independently rejects that candidate. No result supports changing to explicit channels-last or chunking here. PyTorch's [general performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) recommends testing channels-last with AMP; it does not guarantee a benefit for this U-Net.

**Changes worth pursuing**

1. **Reduce sensitivity convolution/layout traffic without changing the architecture.** The measured 44 ms of sensitivity layout copies is a substantial target. Try a wider cuDNN algorithm search (`torch.backends.cudnn.benchmark_limit = 0`) in a fresh process and compare on the same idle GPU. This changes kernel selection rather than the model, but numerical agreement and eventual validation quality still need checks. PyTorch documents that zero tries every available v8-API algorithm in its [backend reference](https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark_limit). This experiment was **not run**, and there is no measured speedup to promise. If autotuning cannot reduce traffic, a layout-preserving normalization/convolution implementation is a more involved candidate; simply setting channels-last has already failed the timing test.
2. **Fuse the two spatial FFT-shift permutations.** The diagnostic prototype compiles `torch.roll` over the real view of complex tensors, producing one permutation kernel rather than sequential spatial copies. It keeps the FFT, normalization, precision and mathematical model unchanged. Eighteen even/odd-shape/dimension checks passed with bitwise-equal outputs and input gradients. A complete two-cascade sensitivity+VarNet pass also had bitwise-equal reconstruction, loss and all parameter gradients. The initial correctness run hit the compiler's eight-graph cache limit; it was rerun with capacity 64 so the final checks remained compiled. The candidate trace replaces the roll kernels with `triton_poi_fused_roll_0`, but the GPU became shared during that run, invalidating an end-to-end speedup estimate. Eliminating *all* baseline roll time would cap the two-cascade gain at about **9% throughput**; a real implementation will recover less. Repeat a same-GPU baseline/candidate comparison before integrating it. This targets the reconstruction physics path more than sensitivity estimation itself.
3. **Keep the existing useful optimizations.** Compiled bfloat16 U-Nets, full-precision FFTs/normalization, and skipping padded coils are already enabled. Do not attribute these to this investigation as new changes. Do not freeze or cache learned sensitivity maps during joint training: their weights change at every step. Caching them changes training.

Downsampling sensitivity estimation is not a no-quality-loss recommendation. The existing, separately collected three-seed results in `../unet_speed_varnet2_r8/eval_sens128.json` report mean PSNR changes versus the full-resolution baseline of approximately -0.108 dB brain/-0.122 dB knee for 128x128, and -0.095/-0.206 dB for 64x64. Those are prior repository results, not new training runs or an independent validation here. Altering width/depth/upsampling or freezing maps likewise needs reconstruction-quality experiments.

For 12 cascades, sensitivity estimation is still executed once per training forward (`estimate_sense=false`), while cascade work increases. Its 51% share at two cascades must not be carried over to twelve. At two cascades, even a hypothetical 2x sensitivity speedup only implies about **1.33x whole-step throughput** using the measured baseline. A trustworthy 12-cascade estimate requires a short profile of that configuration; no uncontended GPU remained for that comparison.

**Artifacts and reproduction**

Production training code and defaults were not edited. Diagnostic scripts are in `scripts/diagnostics/`: `varnet_throughput_probe.py`, `sens_body_benchmark.py`, `check_fused_shifts.py`, and `summarize_varnet_throughput.py`. The pre-existing profiler was reused without editing it. JSON summaries, shape-grouped operator tables, numerical checks and source SHA256 hashes are saved here. Chrome traces and offline run directories remain local and are ignored by this directory's `.gitignore`. The captured real normalized coil batch is `/tmp/seqmri_sens_probe.pt`; it is not added to the repository.

From the repository root, with an exclusively available GPU:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 CC=gcc CXX=g++ .venv/bin/python \
  scripts/diagnostics/varnet_step_profile.py \
  --name baseline_repeat --skip 16 --active 8 --timed 20 --replay 4 \
  --out-dir reports/varnet_throughput_20261026 datasets.num_workers=4

rtk proxy env CUDA_VISIBLE_DEVICES=2 CC=gcc CXX=g++ .venv/bin/python \
  scripts/diagnostics/varnet_throughput_probe.py --fused-shifts \
  --name fused_repeat --skip 16 --active 8 --timed 20 --replay 4 \
  --out-dir reports/varnet_throughput_20261026 datasets.num_workers=4
```

Use new names for further runs. Then repeat without `--replay` to measure fresh-batch overhead and with `recon.num_cascades=12` for the target model. Include a second baseline after the candidate to detect clock/load drift. No other user's training process was interrupted; the initial profiler on GPU 0 was stopped when a conflicting job appeared, and its incomplete run was not used.
