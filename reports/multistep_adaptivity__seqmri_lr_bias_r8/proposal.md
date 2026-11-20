# Is SeqMRI's sampler held back by its training or by its parametrisation? (proposal, 2026-11-19)

| | |
|---|---|
| Status | **Frozen 2026-11-19** as the first commit on `exp/seqmri_lr_bias_r8` (experiment loop, step 2). Approved by the user on 2026-11-19. The decision rule does not change after this commit; any change is stated in the report with its reason. |
| `main` | `d498afd`, clean. `git fetch` could not authenticate (publickey), so main's sync with origin is unverified: it is 3 commits ahead of the last fetched `origin/main`. Nothing is pushed. |

- **Topic / branch / report:** `seqmri_lr_bias_r8` · `exp/seqmri_lr_bias_r8` · `reports/seqmri_lr_bias_r8/`
- **Question.** In fscr1 (`reports/multistep_adaptivity/`), the 4-step SeqMRI sampler F was the only arm still improving
  after the learning-rate step at epoch 90: +0.13 dB of trainer validation PSNR (epochs 91–100 against 80–89), against
  LOUPE's +0.01 and the twin's −0.02. Three explanations:
  - F needs more steps;
  - its sampler's learning rate is too high. fscr1 used Adam at 1e-3; the sequential commands in the SeqMRI code's
    README use 5e-5, with 1e-3 for LOUPE;
  - it has no per-pixel parameters. A convolutional sampler cannot place structure at fixed k-space positions: on
    LOUPE's map, a U-Net fits the pixel band with R² 0.006, and a per-voxel bias with 0.86
    (`conv_sampler_smoothness_r8`).

  Roadmap step 3, order-free.
- **Control.** fscr1 at run state d58087f, seed 42: F_tF (`seqmri_2d`, signed-log input), Fb_tF (its data-blind twin) and
  L_tF (LOUPE). FLAIR tier F, R = 8, 6 × 18 VarNet, 10 × 10 centre, 100 epochs at 1e-3, ×0.1 at epoch 90, scored on the
  FLAIR dev half (307 slices). Checkpoints and scores are reused.

## Arms (each differs from its control in one thing)

| Arm | Change | Control |
|---|---|---|
| R: F, Fb, L resumed | 50 more epochs at 1e-4 from each epoch-100 checkpoint, optimizer state included, into new run folders (`checkpoint=` the old `last_model.pt`, `num_epochs=150`) | the same run at epoch 100 |
| S: F_lrs, Fb_lrs | the sampler in its own Adam group at 1e-4 (1e-5 after the step at 90), from scratch | F_tF, Fb_tF |
| B: F_bias, Fb_bias | a learned per-step, per-pixel logit map b_t (4 × 320 × 320, zero at the start, its own Adam group at the sampler's rate) added to the U-Net output before the softplus. It doubles as step conditioning: the shared U-Net has no step input | F_tF, Fb_tF |

S uses 1e-4, the rate F had when it last improved, rather than the README's 5e-5. The two are within a factor of 2, and
1e-4 matches the observed gain.

## Readouts and decision rule (paired per slice, pooled PSNR, 95% volume bootstrap; seed-42 screen)

| Row | Contrast | Pass | CI includes 0 | CI below 0 |
|---|---|---|---|---|
| R1 | (F − L) at epoch 150 minus (F − L) at epoch 100 | ≥ +0.10 dB, CI above 0: longer training opens F's lead | no evidence | LOUPE gains more |
| S1 | F_lrs − F_tF | ≥ +0.10, CI above 0: the sampler rate held F back | no effect | the lower rate hurts |
| B1 | F_bias − F_tF | ≥ +0.10, CI above 0: per-pixel parameters help F | no effect | they hurt |
| B2 | F_bias − Fb_bias | ≥ +0.10, CI above 0: reading the data pays over a twin with the same per-pixel maps | data-blind | reading the data costs |
| B3 | Fb_bias − L_tF | tested, not assumed: CI above or including 0 means the twin no longer trails LOUPE | | still trails |

- Secondary: each new F against LOUPE (≥ +0.10, CI above 0), F at epoch 150 minus F at epoch 100, SSIM and NMSE.
- Descriptive: data-swap movement, the high-frequency share of each map, and the right-of-centre share.
- Guards: pooled SSIM ≥ 0.80, and within 0.02 of its control; realised fraction 0.125 ± 0.002; no draw outside the
  acquired region.
- **Noise floor.** One seed: the CIs cover slices and draws, not training seeds (spread 0.3–0.5 dB). A pass earns seed
  43 for that arm and its controls. That costs about 30 GPU-h for the seed-43 fscr1 controls, plus the arm itself.

## Expected outcome

- R1: +0.05 to +0.15. F gained +0.13 and LOUPE +0.01 in the ten epochs after the step.
- S1: 0 to +0.2.
- B1: +0.1 to +0.3. Pixel structure gives spacing: a Poisson-disc draw beat a white one by +0.17 to +0.39 dB on the older
  data.
- B2: 0 to +0.1. B3: about 0.
- If R1 holds, the 100-epoch S and B arms are compared at an under-converged budget; the report says so.

## Code (run state 1) and compute

- `seqmri_2d` routes its parameters to a group at `sampler_lr` (F and Fb alike).
- `sampler.seqmri_step_bias`, with the bias's norm logged. A unit test checks that a zero bias reproduces F's first forward
  pass bit for bit and that the twin carries the same bias.
- Resume: a validation job that fails unless both parameter groups log 1e-4 at epoch 101.
- The mask-asymmetry scripts and fscr1's figure renderer are committed, so the report shows curves, recons and per-step
  masks.
- Order: R first (about 15 GPU-h), then S (about 26) and B (about 27): about 68 GPU-h on CoreHPC (RTX PRO 6000). Crashed
  runs are rerun in full.
