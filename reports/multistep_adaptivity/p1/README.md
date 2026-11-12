# Phase 1: bases and the K=16 pilot (CoreHPC, run state 2 = 44c3fba)

Times are CoreHPC local (PST), as `sacct` prints them. Logs and score files are copied here as each job ends.

## Base runs (LOUPE + VarNet, K=1, 100 epochs, the whole local v1 tier, RTX PRO 6000)

|                                          | B_42                          | B_43                          |
|------------------------------------------|-------------------------------|-------------------------------|
| Job, node                                | 2997083, ggpu1-18             | 2997084, ggpu1-18             |
| Ran                                      | 11-10 15:46 to 20:38 (4:52:26) | 11-10 15:46 to 20:38 (4:52:00) |
| State                                    | COMPLETED 0:0                 | COMPLETED 0:0                 |
| Epoch                                    | ~121 s train, ~49 s val       | ~121 s train, ~48 s val       |
| ValLoss, epoch 1 to 100                  | 0.0369 to 0.0275              | 0.0439 to 0.0267              |
| PSNR_Masked: all / brain / PD / PDFS     | 33.80 / 32.90 / 35.99 / 32.40 | 33.81 / 32.99 / 35.93 / 32.41 |
| SSIM_Masked: all / brain / PD / PDFS     | 0.863 / 0.883 / 0.921 / 0.784 | 0.863 / 0.885 / 0.921 / 0.785 |
| Realised fraction of acquired k-space    | 0.1250                        | 0.1250                        |
| wandb: train, eval                       | r7dx548o, jiq041qv            | fbrthybg, 3c7i4dnr            |

- Both runs recorded commit 44c3fba with no uncommitted changes, and both were scored on the same 1938 val slices (hash
  78f07c15).
- **Guards pass.**
  - Pooled SSIM is 0.863, against the 0.80 bar.
  - The realised fraction is within 0.002 of 0.125.
  - No slice failed below 20 dB.
- Knee-PDFS SSIM is 0.78, as in every arm of `muon_layernorm_100ep`; the proposal's SSIM guard is pooled for that reason.
- The swap cost is exactly 0.0 dB [0.0, 0.0], over 1848 slices from 74 volumes. That is expected: the base's LOUPE mask is
  the same for every slice.
- The logs show no NaN, error or traceback. Their warnings are harmless:
  - PyTorch deprecation notices: `size_average`, the scheduler's `epoch` argument, and `lr_scheduler.step()` called before
    `optimizer.step()` on the first step.
  - A `log(0)` in the k-space panel of the training figures (`trainer.py:868`), which draws every 10 epochs.
- The two seeds land 0.01 dB apart pooled.
- **Open: the LOUPE start.** The bases start LOUPE from the VD p=4 prior (`sampler.loupe_vd_init_power=4`), as the
  proposal frozen on 11-10 at 13:08 specifies. The user's note from 11-10 at 13:30 (`feedback_no_vd_start`) says new LOUPE
  arms should start from uniform noise instead. A_K and T_K share each base, so the start cannot by itself open a gap
  between them, but it sets the base they adapt from. Raised with the user on 11-10; on 11-11 they kept these bases
  for the fine-tunes and chose a uniform-noise start for the from-scratch check (`../addendum_2026-11-11.md`).

## Pilot: the seed-42 K=16 pair

- A_16_s42 is job 3000520 and T_16_s42 is job 3000521. Both were submitted 11-10 20:48 with a 1-day limit, both fine-tune
  from `runs/B_42/last_model.pt` for 30 epochs, and both keep `epoch_010.pt`.
- Slurm's start estimate was 11-11 00:25. Both started at 21:04 on ggpu1-18, at about 6 minutes per epoch: A_16 takes
  ~290 s to train and ~52 s to validate, T_16 ~280 s and ~45 s. Peak memory is 38.4 GiB.
- Both saved `epoch_010.pt` (T_16 at 22:01, A_16 at 22:05). The gate evals were jobs 3001401 (A_16, 8.5 min,
  ggpu1-17) and 3001402 (T_16, 7 min, ggpu1-14). Both completed 0:0, on commit 44c3fba with no uncommitted changes.

### Gates at epoch 10 (proposal section 8): all pass

| Gate | Rule | A_16_s42 | T_16_s42 | Pass |
|------|------|----------|----------|------|
| Stability | finite loss, at most 10 consecutive skipped steps | finite, 0 skipped | finite, 0 skipped | yes |
| Stability | PSNR ≥ B_42 − 0.2 dB = 33.60 | 33.90 | 33.87 | yes |
| Plumbing | realised fraction 0.125 ± 0.002 | 0.1250 | 0.1250 | yes |
| Plumbing | T_16 − B_42 within [−0.2, +0.3] dB | | +0.074 | yes |
| Head used | data swap moves ≥ 2% of a step's new samples at some step ≥ 1 (A); exactly 0 (T) | 3.8% at step 1, up to 20.3% at step 14 | 0 at every step | yes |

- **Selection mechanics behind T_16 − B_42** (P0 item 5, `p0/README.md`): spending one static map's budget in 16 equal
  steps flattens it a little. Locations with a K=1 marginal of 0.80–0.95 are covered 0.73 of the time instead of 0.85,
  and the 0.05–0.50 bins gain about +0.03 to +0.05. That was measured on the pre-flip `muon_layer_s42` map, not B_42's.
  It cancels in G_K.
- **A_16's data-swap movement**, pooled, as the share of each step's new samples that move when another slice's data is
  swapped in, steps 0 to 15: 4.1, 3.8, 1.6, 1.0, 0.8, 0.7, 0.7, 0.8, 0.9, 1.1, 1.8, 3.7, 8.6, 16.8, 20.3, 19.5%.
  - It is highest at the last steps.
  - In the knee it reaches 24% (PD and PDFS); in the brain it peaks at 12%.
  - In P0's 1-epoch smoke it was 0.15–0.26%.
- **Twin nulls hold at epoch 10.**
  - T_16's swap cost is exactly 0.0 dB [0.0, 0.0], and its donor masks equal its own on every slice.
  - Replaying each slice with its own data reproduces every step, for both arms.
- **A_16's within-model trajectory swap cost** is +0.033 dB [0.025, 0.042] pooled (1848 slices, 74 volumes). Its knee-PD
  slices cost +0.053.
- **Not a result:** at epoch 10, A_16 is 0.025 dB above T_16 pooled (unpaired means). The decision uses the paired G_K
  at epoch 30, in both seeds.
- The pilot runs finished at epoch 30: A_16_s42 at 11-11 00:12 (3:07:42), T_16_s42 at 11-10 23:59 (2:54:54), both
  COMPLETED 0:0. Their final scores are in `scores/`; the contrasts are in `../p2/contrasts.md`.

## For the final sync (run.md section 6)

- Training runs write wandb under `runs/<arm>/wandb/`, because the trainer passes `dir=exp_dir`. Evals write it under
  `scores/wandb/`. The top-level `wandb/` that section 6 copies stays empty, so copy `runs/*/wandb/offline-run-*` as well.
- Training runs land in wandb group `runs`, the run folder's parent; evals land in `multistep_adaptivity`. Regroup the
  training runs when everything is synced.
- B_42, B_43 and their evals were synced on 11-10 from a copy on the lab server.
