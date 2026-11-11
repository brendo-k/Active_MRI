# Does a brain-only VarNet gain from learning its own mask instead of the frozen shared one? (R=8, 2026-11-10)

| | |
|---|---|
| Code | results produced at `373530d` (brain_static_vs_loupe: paired t over seeds and S support check in the summary); branch `exp/brain_static_vs_loupe` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 9 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 6 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Status**: awaiting the user's verdict.
- **Branch**: `exp/brain_static_vs_loupe`. Runs trained at `f218677` (detached snapshot `../SeqMRI-worktrees/run-bsl-f218677`). Eval
  script unchanged since `fa42786`; summary at `373530d`. Frozen proposal: `proposal.md` (commit `8d6b778`).
- **Proposal vs outcome** (primary contrast loupe_shared − static, PSNR on the full brain val set):

| Frozen criterion | Number | Met? |
|---|---|---|
| 1. Validity: brain SSIM ≥ 0.80 in all 9 runs, all finished | min SSIM 0.949; 9/9 finished | met |
| 2. loupe_shared relocated ≥ 10% of the budget in every seed | 13.3%, 15.9%, 13.0% | met |
| 3. Loupe wins: mean ≥ +0.10 dB, 3/3 positive, > static seed spread | mean −0.120 dB | not met |
| 4. No effect: \|mean\| < 0.05 dB | 0.120 dB | not met |
| 5. Static wins: mean ≤ −0.10 dB, 3/3 negative | mean −0.120 dB, but 2/3 negative (s42 +0.009) | not met |
| 6. Otherwise inconclusive | | **outcome: inconclusive, leaning static** |
| Secondary: loupe_uniform − static (same thresholds) | −0.047, −0.035, −0.409; mean −0.164, 3/3 negative | static wins (driven by s44) |
| Start effect: loupe_shared − loupe_uniform | +0.057, −0.192, +0.266; mean +0.044 | rule says "no effect", but per-seed effects are large and opposite |

- **Recommendation**: park. Learning the mask on brain did not beat the frozen shared mask. The question (a negative or
  null for "a brain-specific mask helps once joint training is removed") is answered as far as this budget allows, and
  no code needs to reach `main`.

## Question
A mask per condition barely beats one shared mask on local v1 (+0.07 / +0.12 dB brain, `reports/local_v1_flair_knee/`).
Two explanations:
- the mask shape barely matters for brain;
- a single jointly trained VarNet holds back the per-condition masks.

This experiment removes joint training. A VarNet is trained on brain only, with the shared mask either frozen
(static) or still learning (LOUPE). If a brain-specific mask were worth something once the VarNet is brain-only, LOUPE
should beat static.

R=8 counts each slice's acquired k-space (`budget_over=acquired`). PSNR is in dB. Mask columns are the phase-encode
axis (kx).

## Runs compared
Common to all runs:
- Data: brain FLAIR only (`datasets.train_anatomies=[brain]`), a quarter of local v1 (`limit_volumes=0.25`: 20 volumes, 301 slices).
- Model and optimisation: 12-cascade VarNet, main defaults (Muon + LayerNorm, `sense_mask_method=null`); batch 16, lr 1e-3; 65 epochs, lr step at 58.
- Sampler: `loupe_2d` straight-through, slope 10, `anatomy_conditioning=true` (3 condition classes; only the brain slot sees data).
- Validation every 5 epochs; seeds 42, 43 and 44.

Each run is scored at its last epoch (`last_model.pt`) on all 40 brain val volumes (605 slices). The best-val-loss
checkpoint is not used, because that val loss includes knee slices, which these models never train on.

S, the shared mask, comes from `loss_domain_rss_r8_flair_knee_100ep_seed42/best_model.pt`:
- one loupe_2d mask learned jointly on full local v1 (FLAIR + knee PD/PDFS), 100 epochs, from a VD p=4 start;
- trained at commit 676675b, which includes the brain-flip fix.

| Arm | Mask | Mean PSNR (min–max over seeds) | Mean SSIM | Mean NMSE |
|---|---|---|---|---|
| static | S frozen (`fixed_mask_checkpoints`), Bernoulli-sampled each step | 36.21 (35.99–36.43) | 0.9507 | 0.0064 |
| loupe_shared | starts at S, trainable (`init_mask_checkpoint`) | 36.09 (35.76–36.44) | 0.9503 | 0.0065 |
| loupe_uniform | starts from uniform noise (`loupe_vd_init_power=0`), trainable | 36.04 (35.79–36.39) | 0.9504 | 0.0065 |

The realised sampling fraction is 0.081 of the 320² grid in every run (R=8 of FLAIR's acquired extent). Every SSIM is
0.949–0.952, so all runs are healthy.

wandb group `brain_static_vs_loupe`:

| Seed | static | loupe_shared | loupe_uniform |
|---|---|---|---|
| s42 | `0k3d7vgy` | `7unl5kqb` | `5qdljg10` |
| s43 | `t1a4cwld` | `q8z1q76v` | `a0i0oydt` |
| s44 | `fq1fd31m` | `ssnhiwe8` | `v5l42w0j` |

## Methods
1. **Training.** Queue `scripts/launch/queue_brain_static_vs_loupe.sh`, one worker per GPU (GPUs 0–1), seed-major order.
   - The worker is `scripts/launch/brain_static_vs_loupe_worker.sh`.
   - Before launch, the frozen arm's brain-slot logits were checked bit-identical to S after a 2-epoch smoke test.
   - Seeds give the same VarNet initialisation across arms, because parameter shapes are identical.
2. **Scoring** (`scripts/eval/brain_static_vs_loupe_eval.py`).
   - Each run's own mean brain probability map is thresholded with common per-slice Bernoulli draws: two draws per slice, seeds 1234 and 4321, averaged.
   - Metrics use the trainer's apodized metric mask.
   - Paired difference = per-slice metric of arm A minus arm B, same seed. CI = 95% bootstrap over the 40 volumes.
   - Budget relocated = ½ Σ|p_arm − p_static| / Σ p_static over the mean brain map. The static map equals S.
3. **Criteria** (`scripts/eval/brain_static_vs_loupe_summary.py`) applies the thresholds frozen in `proposal.md`. Two noise measures:
   - Seed spread = max − min of an arm's mean PSNR over seeds.
   - Seed swap = arm A at seed i minus arm B at seed j (i ≠ j): what an unpaired comparison would see.

## Results
Paired PSNR difference (dB) per seed, with 95% CI over volumes:

| Contrast | s42 | s43 | s44 | Mean | Seed-swap range |
|---|---|---|---|---|---|
| loupe_shared − static | +0.009 [−0.02, +0.04] | −0.227 [−0.31, −0.14] | −0.142 [−0.18, −0.10] | −0.120 | −0.67 to +0.45 |
| loupe_uniform − static | −0.047 [−0.10, +0.00] | −0.035 [−0.13, +0.07] | −0.409 [−0.49, −0.34] | −0.164 | −0.64 to +0.39 |
| loupe_shared − loupe_uniform | +0.057 [+0.00, +0.11] | −0.192 [−0.29, −0.10] | +0.266 [+0.20, +0.34] | +0.044 | −0.62 to +0.65 |

- **Reading:** letting the mask learn on brain never beat the frozen shared mask by more than its noise. It was worse
  in 5 of 6 paired comparisons.
- **Noise:**
  - Seed spread is 0.44 dB (static), 0.68 (loupe_shared) and 0.59 (loupe_uniform), several times every mean effect.
  - Each paired CI covers only its own seed's ~600 slices, so a CI that excludes zero does not make the effect
    reproducible: s43 and s44 disagree on the sign of the start effect.
  - Over seeds, paired t = mean / (SD/√3) is −1.74 (loupe_shared − static), −1.34 (loupe_uniform − static) and +0.33
    (start effect), on 2 df. None is significant, so "leaning static" and the secondary "static wins" verdict are
    weak.

SSIM and NMSE follow PSNR. Paired SSIM differences are within ±0.002 everywhere (loupe_shared − static: +0.0005,
−0.0006, −0.0011).

Budget relocated from S:

| Run | s42 | s43 | s44 |
|---|---|---|---|
| loupe_shared | 13.3% | 15.9% | 13.0% |
| loupe_uniform | 44.3% | 47.6% | 44.1% |

The LOUPE masks did move. loupe_shared moved 13–16% of the budget. Full-data runs moved 42–57%, but from a VD start,
not from an already-learned mask, so the two are not directly comparable. The figure shows that loupe_shared's move is
a re-weighting inside S's support:
- more weight near the centre and along the horizontal band;
- less on the outer lobes;
- no sampling outside S's sharp-edged support.

That support is measured, not just seen:
- Within the acquired extent (where any loupe_uniform map is above 0), S has p = sigmoid(10·w) < 1e-3 on 66.7% of
  the area.
- After 65 epochs, loupe_shared's largest p there is 0.003, 0.009 and 0.001 (s42, s43, s44).

loupe_uniform ends as a diffuse, roughly radial blob. After 1,235 steps it still carries visible noise.

## Conclusions
- With a brain-only VarNet, a learned brain mask did not beat the frozen shared mask: mean −0.12 dB (loupe_shared) and
  −0.16 dB (loupe_uniform), against 0.44 dB seed spread. Removing joint training did not reveal a brain-specific mask
  gain. So the explanation "the shared VarNet holds back per-condition masks" is not supported for brain at this
  budget.
- The frozen shared mask was at least as good as the learned mask in 5 of 6 paired comparisons (the exception is
  loupe_shared s42, +0.009 dB). Within the measured range, mask shape matters little for brain, consistent with the
  earlier small per-condition gains.
- **Hypothesis, untested:** LOUPE's mask moves on 301 slices partly fit the training set or co-adapt with a VarNet
  that is still training. That would explain why learning slightly hurts rather than ties. A full-data brain-only
  rerun would test it.
- From a uniform start, LOUPE reached the static score in 2 of 3 seeds (−0.05, −0.03 dB) despite a still-noisy mask.
  It fell 0.41 dB short in s44. The start effect is seed-dependent rather than consistent in either direction.

## Caveats
- **Small budget.**
  - The mask moved, but 1,235 steps is about 1/13 of a full-data run.
  - A brain-only run on full data might let LOUPE find a better brain mask. This experiment cannot rule that out.
- **S's support.**
  - S has a sharp-edged support: p < 1e-3 on two thirds of the acquired extent, and no loupe_shared run exceeded
    p = 0.009 there (measured).
  - That loupe_shared *cannot* leave it, because the sigmoid gradient vanishes there, is a hypothesis, untested. The
    measurement only shows that it did not leave in 1,235 steps.
  - A "brain-specific mask" here therefore means a re-weighting of S, not a free search.
  - loupe_uniform has no such limit, but its mask is under-converged.
- **The shared mask came from a VD p=4 start and full data,** while the brain-only VarNets trained on a quarter of the
  data. The static arm thus carries extra information (full data, both anatomies) in its mask. This favours static,
  and is inherent to the question.
- **Three seeds.** The verdict for loupe_uniform − static rests on s44 (−0.41 dB); s42 and s43 are within ±0.05.
- **Fixed scoring checkpoint.** `last_model.pt` was fixed in advance (val loss includes knee). Other checkpoints were not scored.
- **Interrupted first attempts.** The first attempts of static_s42 ran out of memory while another session's jobs
  were on GPU 0. It was rerun in full from scratch, and no partial run is used. Logs are in the queue's `failed_oom/`.
- **Untracked inputs.** Checkpoints and S are untracked (see `manifest.json`).

## Figures
- `figures/mean_brain_maps.png`: mean brain sampling probability (after budget rescaling) on the full brain val set.
  - Per seed: static (= S), loupe_shared, loupe_uniform, and loupe_shared − static.
  - The black square is the always-acquired centre.

## Next steps
1. Brain-only, full-data rerun of static vs loupe_shared (80 volumes, 1,211 slices, about 4,900 steps at 65 epochs).
   This tests whether the small negative is a small-data artefact.
2. Joint-VarNet test from the proposal: frozen per-condition masks × {blind, FiLM} recon, 2×2. It asks whether recon
   conditioning, not the mask, is where joint training loses.
3. Loss-landscape traversal (user's open question): starts beyond S and uniform, e.g. uniform start with a higher mask
   lr or temperature annealing. Score them against the same static control.

## Reproduce
`bash reports/brain_static_vs_loupe/run.sh` from the repo root.
- Needs the 9 checkpoints in `~/data/seqmri_checkpoints/brain_static_vs_loupe/` (linked from `checkpoints/bsl_*`) and one idle GPU, about 45 min.
- Check: `python scripts/report_manifest.py check reports/brain_static_vs_loupe`.
