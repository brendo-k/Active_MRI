# Phase 0 checks (2026-11-09 and 11-10, `exp/multistep_adaptivity`, base `main` 9123f44)

Fixes 1–5 of the proposal's section 3, and what shows they hold. Fix 5, the K-step eval and the per-step logging, has
its own section. The two ablation inputs and the CoreHPC validation job (P0 item 7) follow at the end.

| # | Fix | Evidence |
|---|---|---|
| 1 | The union M_{i+1} = M_i + (draw_i − M_i.detach()) keeps every step's straight-through increment, so the final loss reaches each step's choices | `step_gradients_main_vs_branch.txt`: at K=4 on main only step 3's probabilities get a gradient, on the branch all four; `test_final_loss_reaches_every_steps_probabilities` (loupe_2d and seq_adapt_2d) |
| 2 | Every step reads the measured y ⊙ M_i, never FFT(recon · S) | `test_steps_read_only_the_kspace_measured_so_far`: values outside the final mask can change without moving any step's probabilities, while changing the ACS does move them |
| 3 | One reconstruction per forward, of M_K | `step_gradients_main_vs_branch.txt` (main 4 per forward at K=4, branch 1); `test_one_reconstruction_per_forward` at K = 1, 4, 16 |
| 4 | `seq_adapt_2d` with `sampler.data_blind`: the twin reads M_i, i/K and the header (extent, coil fraction, condition), never the k-space values | `test_twin_ignores_kspace_values_but_reads_its_mask`, `test_adaptive_arm_and_twin_start_bitwise_identical` (same weights, same random stream, identical masks and output at init) |

Also:
- **K=1 is bitwise unchanged** for loupe_2d (shared and per-condition) and conv_2d (sense and rss input): loss, output,
  mask and every gradient (`k1_bitwise_vs_main.txt`; `scripts/diagnostics/multistep_k1_reference.py` run on both
  checkouts on CPU, compared with `multistep_k1_compare.py`).
- **Budget**: every step's rescaled probabilities sum to budget/K of the slice's acquired k-space at K = 1, 4, 16, and
  the realised count stays within 5 sd of the budget (`test_each_step_spends_an_equal_share_of_the_acquired_budget`).
- **Optimizer**: the residual U-Net trains in its own last Adam group at `sampler.residual_lr`, outside Muon
  (`test_residual_trains_in_its_own_adam_group_outside_muon`).
- **Full suite**: 221 passed, 2 skipped (GPU 0).

## Selection mechanics at init (P0 item 5)

`union_marginals_muon_layer_s42.txt` (`scripts/diagnostics/multistep_union_marginals.py`, the pre-flip `muon_layer_s42`
LOUPE map, 128 draws). Spending one static map's budget in K equal steps flattens it a little:

- Locations with a K=1 marginal of 0.80–0.95 are covered 0.75 of the time at K=4 and 0.73 at K=16, against 0.85.
- The 0.05–0.50 bins gain about +0.03 to +0.05.
- The mean per-location change is about 0.011–0.012, and the total count is preserved.

A and T share this, so it cancels in G_K. It enters T_K − T_1 and the pilot's plumbing gate (T_16 − B_42), so it is
reported beside them.

## Smoke runs through `train.py` (wandb offline, 1 epoch, `datasets.limit_volumes`, batch 16, 320², R=8)

- **K=4, adaptive arm (GPU 0) and twin (GPU 1)**, `datasets.limit_volumes=0.05`. Both exit 0, including the visualised
  batch (four per-step panels) and a validation pass. Both log train 0.1631 and val 0.1589.
  - Their last checkpoints carry `data_blind` False and True, so the flag reaches the model through yaml, builder,
    TrainConfig and build_model.
  - Both residual output layers have moved off zero (max |w| about 1.2e-3).
  - 44 of 46 residual tensors differ between the arms. The equal losses come from a residual still small after one
    epoch, not from broken wiring.
- **K=16 vs K=1, adaptive arm**, `datasets.limit_volumes=0.1`.
  - K=1 trains: an epoch of 113 s train and 35 s validation, compile included.
  - K=16 at batch 16 first ran out of memory on a 48 GB A6000, in the residual U-Net's forward pass, both with and
    without `recon.checkpoint_cascades=true`. `nvidia-smi` showed 48.7 GB reserved at K=16 and 46.7 GB at K=1.
  - The fix is the per-step activation checkpointing of r_θ planned in proposal section 4.2. It is now
    `sampler.residual_checkpoint: true`, the default, and leaves the gradients unchanged
    (`test_residual_checkpointing_keeps_the_gradients`).
  - With it, K=16 trains without cascade checkpointing: 160 s train and 45 s validation per epoch, against 113 s and
    35 s at K=1. That is about 1.4×, rough because compile time is included. Train and validation losses match K=1 to
    1e-4.
  - `nvidia-smi` peaks are reserved memory, not true peaks, so the P0 timing runs should log
    `torch.cuda.max_memory_allocated` to measure the real headroom.

## Fix 5: the K-step eval, and per-step logging

`scripts/eval/eval_multistep_adaptivity.py`, built on `activemri/utils/multistep.py`, which the trainer's logging also
uses.

```bash
# each run, as it finishes: per-slice scores, swap cost, data-swap sensitivity, guards, step-grid figures, wandb eval run
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_multistep_adaptivity.py score --run <run dir> --name A_16_s42 \
    --out reports/multistep_adaptivity/scores --gradients            # --checkpoint epoch_010.pt for the pilot gate
# any time, from the saved scores: G_K, the twin floor, T_K - T_1, U_K, with volume-cluster bootstrap CIs
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/scores \
    --pair G_16_s42=A_16_s42-T_16_s42 --out reports/multistep_adaptivity/contrasts.json
```

What it guarantees, and the checks:
- **Shared draws.** Every Bernoulli threshold is keyed by (draw seed, val slice, step), so A_K and T_K, and any two
  runs on the same slices, share their random numbers, whatever the batching. `contrast` refuses runs scored on
  different slices or draw seeds.
- **The trainer's metric path.** The two K=4 smoke runs score 16.666 / 16.654 dB PSNR_Masked and 0.1752 / 0.1749
  SSIM_Masked. Their logged `val/` values are 16.668 / 16.660 and 0.1759 / 0.1756, which is draw noise.
- **One reconstruction path.** Own and swapped masks go through `reconstruct_from_mask`, the forward's own final
  reconstruction, bitwise equal to the forward's output at K=1 and K=4. cuDNN runs deterministic in the eval.
- **Donors** match on condition, acquired-location mask and coil count, from another volume. On the full val set every
  knee slice has one, and 515 of 605 brain slices (85%; 13 brain header signatures).
- **Nulls on real data** (the two K=4 smoke checkpoints, 120 evenly spaced full-val slices, 113 with a donor):

  | | twin (data-blind) | adaptive |
  |---|---|---|
  | donor's final mask equals its own | 1.000 (must be 1) | 0.009 |
  | swap cost, PSNR | 0.0 [0.0, 0.0] | +0.0001 [−0.0002, +0.0004] |
  | step i's new samples moved by a data swap | 0 at every step (must be 0) | 2.4e-4 to 3.0e-4 |
  | residual change under the swap (RMS logits) | 0 | 2.2e-3 to 2.8e-3 |
  | own data in the swap function reproduces every step | yes | yes |

  The adaptive column is the positive control: its residual is one epoch out of a zero start, so its data use is tiny
  but not zero.
- **Unit tests** (`tests/test_multistep_eval.py`, 13): thresholds, reconstruct-from-mask = forward output, the swap
  null with its positive control, the swap function with its positive control, the stratified bootstrap, the trainer's
  metrics, donor matching, the recorder and the figure.
- `contrast` of a run with itself gives 0 [0, 0]; a wandb eval run (offline test) receives means, guards, swap cost,
  per-step data-swap values and the figures.

Per-step logging in training (wandb and the run's visualization folder):
- **Every metric batch:** `train/steps/grad_norm_i`, the norm of the loss gradient reaching step i's probabilities;
  `train/steps/last_grad_share`, which is 1 if only the last step learns and 1/K for an even pull; also
  `new_samples_i` and `residual_rms_i`. `grad/sampler_residual_norm` and `grad/sampler_static_norm` split the
  sampler's gradient. In the K=4 smoke all four steps get a gradient of about 1e-4, and the last step's share is 0.30.
- **Every val epoch:** `val/steps/new_samples_i` and `residual_rms_i`, averaged over the val set.
- **On visualised epochs**, `train/{anatomy}/steps` and `val/{anatomy}/steps` images, one row per step. The columns are
  the k-space measured before the step, the step's rescaled probabilities, the residual, the loss gradient, the new
  samples, the mask after, and the zero-filled image, reconstruction and error of that mask. The same figure comes from
  the eval, for fixed mid-volume slices of each condition.
- `training.save_epochs=[10]` keeps `epoch_010.pt` for the pilot gate.
- K=16 at batch 16 still trains with all of it on.

Two more checks:
- **K=16** (60 full-val slices, 56 with a donor). The twin is the K=16 smoke weights with `data_blind` set, since the
  null must hold whatever the weights.
  - Twin: donor mask equals its own 1.000, moved fraction 0 at all 16 steps, swap cost 0.0 [0.0, 0.0].
  - Adaptive: donor mask equals its own 0.018, moved fraction 1.6e-4 to 3.5e-4 per step, residual change about
    2.4e-3, swap cost −0.0002 [−0.0009, +0.0006] dB.
  - About 54 s per 60 slices per arm with both draw seeds, swap and sensitivity, so roughly 30 min per K=16 run on the
    full val set. That is consistent with section 9's eval estimate.
- **The P1 base path** (`loupe_2d`, K=1, base recipe, 1 epoch, 5% data):
  - Train exits 0 and writes its step figures.
  - The eval runs with sensitivity skipped, since LOUPE reads no data. Its donor masks equal its own (1.000) and its swap
    cost is exactly 0, as for any data-independent mask.

The summaries behind these numbers are in `eval_smoke/`. The checkpoints were 1-epoch smoke runs and stayed in /tmp.

## Ablation inputs (2026-11-10)

Both are built before anything ships, so every CoreHPC run records one commit.

- **A_16^img, `sampler.data_channels=[image]`.** The k-space channel (b) gets the twin's fixed values, and the image
  channel (a) keeps the slice's data. The architecture is unchanged, so A_16^img, A_16 and T_16 share every weight and
  draw the same masks at init (`test_channel_ablation_reads_the_image_channel_alone_and_starts_as_its_arm`).
- **A_4^rec, `sampler.input=recon` with `sampler.input_recon_checkpoint`.** At every step the residual also reads a
  frozen copy of the base's reconstructor and sensitivity net, reconstructing y ⊙ M_i, divided by the ACS image peak
  (the scale of channel (a)).
  - The copies load the base's weights when the fine-tune starts, stay in eval mode, get no gradient and are outside
    the optimizer. They are saved with the run, so the eval rebuilds them
    (`test_recon_input_arm_loads_the_base_into_both_reconstructors_and_trains_only_one`).
  - The residual's first convolution gains one input channel. Its weights for the existing inputs are unchanged and the
    new ones come from their own generator, so every other weight and the random stream match A_4, and the two draw
    the same masks at init (`test_recon_input_arm_shares_every_weight_but_one_input_and_starts_as_its_arm`).
  - The eval's data swap also swaps the donor's reconstruction
    (`test_counterfactual_step_reads_the_donors_data_through_the_recon_input_too`).
- K=1 is still bitwise equal to `main` after these changes (loupe_2d shared and per-condition, conv_2d sense and rss).
  The full suite gives 238 passed and 4 skipped on CPU; the 2 extra skips need a GPU.
- **Smokes** through `train.py` and the eval (1 epoch, 5% data, fine-tuned from the 1-epoch base smoke, GPU 1):
  - A_16^img trains and scores, with a peak of 39.1 GiB.
  - A_4^rec first stopped at startup: the base loader counted the frozen copy's keys as missing. That is fixed, and the
    loader test above covers it.
  - It then stopped in validation on `RecompileLimitExceeded`. The frozen copy's no-grad calls add compiled variants of
    the fused data-consistency kernel, which compiles with `fullgraph=True`, past dynamo's default limit of 8.
    `allow_recompiles()` (`activemri/utils/compilers.py`) now raises the limit to 32 wherever the code compiles. No
    main-arm run reached 8 variants (none of their logs shows the limit), so they compile exactly as before.
  - After both fixes A_4^rec trains, with a peak of 38.9 GiB, and its eval runs, with the own-data swap reproducing
    every step.

## CoreHPC (P0 item 7)

`scripts/launch/hpc/multistep_adaptivity/` holds `run.md` (ship, submit, bring back), `validate.sbatch`, `train.sbatch`
(one arm, then its eval), `eval.sbatch` (e.g. the pilot's `epoch_010.pt`) and `common.sh` (arm recipes, data roots,
the GPU pin).

- **Data reference.** `reference_slices.json` lists the (volume, slice) pairs the lab server serves: train 160 volumes
  and 3,929 slices (brain 80 / 1,211, knee-PD 40 / 1,369, knee-PDFS 40 / 1,349), val 80 and 1,938 (40 / 605, 20 / 673,
  20 / 660). The loader skips a file it cannot open with only a warning, so the job compares the exact lists, not
  counts. The manifest loader crops the readout, so it never reads the `sense` datasets only the local copies carry.
- **GPU pin: RTX Pro 6000** (the user's choice, 2026-11-10). Every job is submitted with
  `--gres=gpu:nvidia_rtx_pro_6000:1` and `GPU_NAME='RTX PRO 6000'`. It checks `torch.cuda.get_device_name()` against
  `GPU_NAME` and exits on any other card. `sinfo` showed 7 such nodes with 2 cards each (`ggpu1-13` to `ggpu1-19`), and
  NVIDIA's spec gives 96 GB per card. The validation job's first log line gives the full device name.
- **Memory.** The trainer now logs each epoch's `torch.cuda.max_memory_allocated` (`system/peak_memory_gib`, and
  `PeakMemory` in the epoch line), not the allocator reserve `nvidia-smi` shows.
- **Lab-server dry run** of the whole validation job (`LAB_SERVER_TEST=1`, GPU 0, from a shallow clone as `run.md` ships
  it), on this code before the ablation inputs and the recompile fix. All 12 checks passed:
  - the commit and a clean tree; tests 235 passed, 2 skipped; the same slices as the reference, in the same loader order;
  - the K=4 twin: donor's final mask equals its own in every slice, 0 new samples moved at every step, swap cost
    0.0 [0.0, 0.0] dB (113 slices, 7 without a donor). The adaptive arm moved 0.15–0.26% of each step's new samples,
    the positive control;
  - every smoke run recorded the shipped commit, with no dirty files.
  - Peak memory and epoch times (10% of a train epoch, compile included, then the full val set):

    | run | peak memory | train | val |
    |---|---|---|---|
    | base (K=1 LOUPE) | 36.5 GiB | 85 s | 117 s |
    | A_4 | 36.6 GiB | 122 s | 163 s |
    | T_4 | 36.6 GiB | 121 s | 160 s |
    | A_16 | 38.1 GiB | 160 s | 220 s |

    The 48 GB that `nvidia-smi` showed at K=16 was the allocator's reserve; the true peak is 38 GiB, so an 80 GB card
    has room to spare.
- **Second lab-server dry run**, on the launch commit `de7af04` (2026-11-10): inconclusive.
  - Another training queue had started on GPU 0 four minutes earlier. The A_4, T_4 and A_16 smokes ran out of memory
    beside it, and they pushed that queue's run out of memory twice.
  - Every check that got memory passed: the commit, the tests, the data, and the base smoke and its score.
  - The dry run was stopped, and the CoreHPC job is the gate instead.
- **CoreHPC validation job 2996412** (run state 1, `de7af04`, 2026-11-10) failed. It was submitted at 14:37 and
  started at 14:42 on `ggpu1-14`, not at the scheduler's 23:18 estimate, and it ended at 14:43.
  - The card was an `NVIDIA RTX PRO 6000 Blackwell Server Edition`, on python 3.14.0 and torch 2.12.0+cu130.
  - The commit check and the tests passed. Every check after them failed on a Hydra parse error:
    `common.sh` passed `datasets.roots.local_batch2=~/...` with the `~` unquoted. The lab-server dry runs never reached
    it, because `LAB_SERVER_TEST=1` leaves the data roots out.
  - Its logs are in `corehpc/`.
- **Run state 2** (`44c3fba`) quotes that root.
  - It adds `tests/test_multistep_launch.py`, which builds all 20 arms' CoreHPC arguments as `train.sbatch` does and
    composes them through `train.py` and Hydra, checking the data roots and each arm's recipe. Against the old
    `common.sh` it fails with the same `OverrideParseException`.
  - All 240 files the job loads exist on CoreHPC under the two roots (train 80 `hpc` + 80 `local_batch2`, val 80 `hpc`).
  - The CPU suite gives 259 passed, 4 skipped (GPU only).
- **CoreHPC validation job 2996689** (run state 2, 2026-11-10, `ggpu1-19`): all 15 checks passed, so P0 is done. Its
  logs are in `corehpc/`.
  - The commit was `44c3fba`, clean, and the tests gave 263 passed (the 4 GPU tests included).
  - The data matched the lab-server reference, with the same slices in the same loader order for train and val.
  - The K=4 twin's nulls were exact: the donor's final mask equals its own in every slice, 0 new samples moved at every
    step, and the swap cost is 0.0 [0.0, 0.0] dB (113 slices from 71 volumes, 7 without a donor). The adaptive arm
    moved 0.15–0.26% of each step's new samples.
  - All 6 smoke runs recorded the shipped commit.
  - Epoch lines on the RTX PRO 6000 (10% of a train epoch, compile included, then the full val set):

    | run | peak memory | train | val |
    |---|---|---|---|
    | base (K=1 LOUPE) | 35.7 GiB | 37 s | 49 s |
    | A_4 | 36.2 GiB | 99 s | 86 s |
    | T_4 | 36.2 GiB | 49 s | 53 s |
    | A_16 | 38.1 GiB | 65 s | 85 s |
    | A_4^rec | 36.4 GiB | 66 s | 137 s |
    | A_4^img | 36.2 GiB | 47 s | 57 s |

- **P1 submitted** 2026-11-10 from run state 2: B_42 is job 2997083 and B_43 is job 2997084, each with a 1-day limit.
  The AGENTS.md roadmap line now reads "order-free go/no-go running" on this branch, as agreed for the start of P1.

