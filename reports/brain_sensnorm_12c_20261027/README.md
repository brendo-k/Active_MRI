# Brain-only sensitivity-normalization convergence study

Started 2026-10-27. The preceding setup was interrupted by a worker usage limit before any training was launched. The study below is now launched through a durable supervisor.

Stopped at the user's request on October 27 after both seed-42 arms saved epoch 1. The automatic checkpoint resume scheduled for **October 27, 2026, 8:00 PM America/Los_Angeles (PDT)**, `at` job **12**, executed but failed during preflight. The original dataset directory `/home/bkadota/data/fastmri/multicoil_brain` was missing; the repository's legacy split notes document deletion of that dataset tree on October 27. Checked October 28: no study workers are running, both epoch-1 checkpoints remain, and final results are unavailable. The current workspace config uses a different dataset and split; switching to it would change this study and requires a fresh matched experiment or restoration of the original data. Scheduling details are in `schedule.json`; the failure traceback is in `scheduled_resume.log` in the experiment directory.

Question: does replacing sensitivity-U-Net InstanceNorm with the full-resolution spatial-moments implementation change final reconstruction quality after training a 12-cascade VarNet from scratch?

Both arms use 12 cascades, 18 initial reconstruction channels, four pools, 256x256 images, the same fixed R=8 static-VD mask, compiled BF16 sensitivity and reconstruction U-Nets, batch size 16, Adam learning rate 0.001, and 50 epochs with the existing step schedule at epoch 40. The candidate replaces only sensitivity normalization with FP32 spatial statistics and channels-last layout. Both retain the existing FFT implementation. Sensitivity maps remain trainable and full resolution.

Two paired seeds (42 and 43) use identical initialization and sample order within each pair. Preflight and each worker assert matching state hashes, dataset membership hashes and sampler order. Training is brain T2 only: 1,350 slices from 89 volumes. Validation is brain only: 344 slices from 23 different volumes. The split is constructed using the preceding balanced dataset recipe, then the knee data is excluded from both loaders. There is no training/validation volume overlap.

GPUs 2/3 run baseline/moments for seed 42, then moments/baseline for seed 43, balancing device assignment. Other GPU jobs are left running. The supervisor waits for its required GPUs to become available before starting a new pair. W&B is offline. Code, configs and mask are frozen in the experiment directory; running jobs are insulated from subsequent workspace edits.

Primary endpoint: final epoch-50 checkpoint, with no best-checkpoint selection. Automatic evaluation reports masked-pixel PSNR and SSIM, plus the prior apodized-image PSNR/SSIM as secondary metrics. Candidate-minus-baseline differences are paired by slice and volume. Confidence intervals bootstrap validation volumes; the combined interval averages these two seeds before bootstrapping and does not represent uncertainty across arbitrary training seeds. No equivalence or acceptable-loss margin is assumed. In particular, a non-significant difference is not proof of no performance loss.

Live artifacts are in `checkpoints/brain_sensnorm_12c_20261027/`:

- `status.json`: supervisor state and active PIDs.
- `preflight.json`: pairing checks and exact sample counts.
- `<arm>_seed<seed>/status.json` and `epochs.jsonl`: epoch-level loss, metrics and timings.
- `<arm>_seed<seed>/last_model.pt` and `best_model.pt`: atomic, resumable checkpoints with arm metadata and RNG states.
- `paired_seed42.json`, `paired_seed43.json`: automatically generated final per-slice scores.
- `comparison.json` and `RESULTS.md`: automatic final comparison after all four runs finish.

Checkpoints identify the normalization arm. Use `load_study_model` in the frozen `code/scripts/diagnostics/brain_sensnorm_study.py` to rebuild the matching model; the candidate has an additional `inner` component in its sensitivity-body state keys. Loading it as an unchanged baseline model is not supported.

The reproducible launcher is `scripts/diagnostics/brain_sensnorm_study.py`. Running `launch --root checkpoints/brain_sensnorm_12c_20261027` refuses to duplicate a live supervisor. If interrupted, restarting the launcher resumes existing epoch checkpoints and restores optimizer, scheduler and RNG states. Final performance results are pending until the experiment completes.
