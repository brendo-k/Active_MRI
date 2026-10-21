# Does an anatomy label help a PromptMR+-style VarNet on fixed CAIPI R=4 sampling? (2026-10-21)

## Question
A VarNet that can already adapt to its input (PromptMR+'s adaptive input, momentum layer and input-driven prompts)
is trained on a fixed 2D-CAIPI pattern over brain, knee-PD and knee-PDFS. Does also giving it each slice's class
improve the reconstruction, hurt it, or change nothing? The comparison isolates the label: the control has the
same parameters, initialisation and data order, and only the label it is fed differs.

Conventions: PSNR in dB, inside the apodized per-anatomy metric mask (`activemri/utils/metric_mask.py`). "Cond -
blind" > 0 means the label helps. "PromptMR++" was asked for but does not exist in the literature; the
conditioning below is PromptMR's prompt block with the class added to the prompt selection.

## Runs compared
All: `train.py --model sequential`, VarNet 12 cascades, chans 18, `caipi_2d` 2x2 shift 1 (R=4 plus the 10x10
calibration centre), `brain_knee_balanced` with classes [brain, knee-PD, knee-PDFS], lr 1e-3 (2 warmup epochs) stepped
x0.1 at epoch 40, 50 epochs, batch 4. Launcher `scripts/launch/train_varnet_caipi_promptmr_anatcond.sh`; checkpoints under
`/tmp/bkadota_checkpoints/varnet_caipi_promptmr_anatcond/` (symlinked from `checkpoints/`). Scored at
`last_model.pt` (epoch 50).

| Arm | Seed | Run dir | wandb | Label fed |
|---|---|---|---|---|
| cond | 42 | `varnet_caipi2d_r4_promptmr_cond_seed42` | ej68kqb7 | own class |
| blind | 42 | `varnet_caipi2d_r4_promptmr_blind_seed42` | 75gex8ds | class 0 for every slice |
| cond | 43 | `varnet_caipi2d_r4_promptmr_cond_seed43` | m8hsaz22 | own class |
| blind | 43 | `varnet_caipi2d_r4_promptmr_blind_seed43` | mgpv2b1j | class 0 for every slice |

Model (83.6M parameters; a plain 12-cascade VarNet is about 30M):
- adaptive input (`recon.adaptive_input`): each cascade's U-Net sees A^H A x, A^H y and a carried latent;
- momentum layer (`recon.momentum_history=11`, 40.3M parameters): at every decoder level, the level's input is
  fused with the same level's inputs from the 11 earlier cascades (1x1 conv, then a channel-attention block), as
  PromptMR+ eq. 10;
- prompt (`recon.prompt_conditioning`, 13.3M parameters): at every decoder level, 5 learned prompt maps mixed by
  a softmax over logits from the pooled features plus a linear map of a 16-d class embedding, resized, 3x3 conv,
  fused by a 1x1 conv. The label enters only through these logits: 3,888 label-specific parameters in all.
- blind arm (`recon.label_blind=true`): same model, every slice fed class 0, so the prompt selection depends on the
  features alone, as in PromptMR. The `.hydra/config.yaml` files differ only in `label_blind`, seed and out_dir.

Data: train 89 brain volumes / 1350 slices, 40 knee volumes / 1350 slices; val 23 brain volumes / 344 slices and
10 knee volumes / 344 slices (4 knee-PD, 6 knee-PDFS volumes).

Validation at epoch 50 (both metrics are defined in Methods; all models healthy, SSIM 0.96-0.99):

| Arm | Seed | Brain PSNR per slice / trainer | Brain SSIM per slice | Knee PSNR per slice / trainer | Knee SSIM per slice | NMSE brain / knee (trainer) |
|---|---|---|---|---|---|---|
| cond | 42 | 41.92 / 41.61 | 0.9864 | 39.88 / 37.82 | 0.9597 | 0.0024 / 0.0047 |
| blind | 42 | 41.91 / 41.69 | 0.9865 | 39.89 / 37.85 | 0.9599 | 0.0024 / 0.0047 |
| cond | 43 | 42.06 / 41.73 | 0.9866 | 39.90 / 37.86 | 0.9600 | 0.0024 / 0.0047 |
| blind | 43 | 42.04 / 41.75 | 0.9866 | 39.92 / 37.88 | 0.9601 | 0.0024 / 0.0047 |

Final val loss (the training objective) per run: 0.05104 / 0.05095 (seed 42 cond / blind), 0.05089 / 0.05085
(seed 43).

## Methods
All numbers from `scripts/eval/eval_varnet_caipi_anatcond.py --seeds 42 43` (output
`analysis/varnet_caipi_anatcond/eval.json`, log `eval.log`). Every val slice goes through the model and is scored
inside the metric mask (`_apply_signal_mask` with `metric_mask_from_real_kspace`), two ways:
- **per slice**: PSNR / SSIM of each slice with data range = that slice's max, averaged over slices;
- **trainer**: the trainer's validation metric (`_AnatomyStats`): PSNR = 10 log10(R^2 / MSE pooled over the
  group's slices), with R the target max of the group's last batch (torchmetrics keeps the last `data_range` set).
  It weights high-error slices more than the per-slice mean does. Both arms see the same targets, so a trainer
  gap is exactly -10 log10(MSE_cond / MSE_blind); its absolute per-class values depend on which batch came last and
  are not comparable across classes. The brain / knee values reproduce `reports/METRIC_MASK_REEVAL_R8.md` (seed
  42: cond 41.61 / 37.82, blind 41.69 / 37.85 dB).

1. **Arm gap** (cond - blind), per class and knee pooled, per seed. Paired over slices; 95% CI over per-volume
   mean differences, normal (1.96) and Student-t (t at n_volumes - 1 df: 3.18 for knee-PD's 4 volumes, 2.57 for
   knee-PDFS's 6). Also the share of volumes whose mean difference is > 0.
2. **Label swap**: the cond model given one class for every slice, minus the same model given each slice's own
   class. A wrong label that costs nothing would mean the model ignores the label.
3. **Context against plain VarNet**: final logged val metrics (wandb summaries, legacy `target > 0.1 * max` mask)
   of the earlier plain 12-cascade VarNet on the same CAIPI pattern (rbdddrmf) and the same with bottleneck FiLM
   conditioning on [brain, knee-PD, knee-PDFS] (qpaflt54). Their checkpoints were not kept, so they cannot be
   re-scored under the new mask.

## Results
**The label changes nothing measurable.** Cond - blind PSNR (dB), per seed; per-slice difference with its
Student-t 95% CI over volumes, the share of volumes where cond is better, and the trainer-metric difference:

| Class | Volumes | Seed 42 per slice [t-CI] | Seed 43 per slice [t-CI] | Volumes cond better (42 / 43) | Trainer seed 42 / 43 |
|---|---|---|---|---|---|
| brain | 23 | +0.007 [-0.018, +0.033] | +0.019 [+0.005, +0.032] | 78% / 74% | -0.077 / -0.014 |
| knee-PD | 4 | +0.027 [-0.125, +0.179] | -0.002 [-0.143, +0.139] | 50% / 25% | -0.013 / +0.001 |
| knee-PDFS | 6 | -0.031 [-0.098, +0.036] | -0.023 [-0.055, +0.008] | 50% / 17% | -0.030 / -0.021 |
| knee (pooled) | 10 | -0.008 [-0.064, +0.047] | -0.015 [-0.056, +0.026] | 50% / 20% | -0.025 / -0.015 |

Every gap is within 0.08 dB, and all but the seed-42 brain trainer gap are within 0.031 dB. Averaged over seeds
(per slice): brain +0.013 dB, knee-PD +0.012, knee-PDFS -0.027, knee -0.012, all with t-CIs spanning 0. Brain flips
sign between metrics (per slice +0.007 / +0.019, trainer -0.077 / -0.014). Only knee-PDFS has the same sign in
both seeds and both metrics, at -0.02 to -0.03 dB. SSIM gaps are all within 0.0003. The seed-to-seed gap
difference (up to 0.06 dB in the brain trainer metric) is as large as any gap.

**The conditioned model does use the label.** PSNR (dB, per slice) when every slice is given one class, minus its
own class; seed 42 / seed 43, Student-t 95% CI in brackets for seed 42:

| Given label | True brain | True knee-PD | True knee-PDFS |
|---|---|---|---|
| brain | 0 | -0.09 [-0.18, +0.00] / -0.10 | -0.45 [-0.87, -0.03] / -0.41 |
| knee-PD | -0.34 [-0.40, -0.28] / -0.15 | 0 | -0.44 [-0.79, -0.10] / -0.14 |
| knee-PDFS | -0.82 [-0.98, -0.66] / -0.38 | -0.32 [-0.67, +0.02] / -0.02 | 0 |

A wrong label costs 0.1 to 0.8 dB, against arm gaps of at most 0.08 dB. Every wrong label hurts on average; none
helps. The trainer metric gives the same picture (e.g. brain given knee-PDFS: -0.60 / -0.23 dB). How much the
model relies on the label differs by seed. On brain slices given a knee label, seed 43 loses about half as much as
seed 42 (-0.15 / -0.38 vs -0.34 / -0.82 dB). On the knee-PD / knee-PDFS swap it loses much less (-0.02 and -0.14 vs
-0.32 and -0.44). The brain label on knee slices costs the same in both seeds.

**Context: PromptMR+ vs plain VarNet** (legacy metric mask, final logged val, one run each; not comparable to the
tables above):

| Run | Brain PSNR | Knee PSNR | Brain SSIM | Knee SSIM |
|---|---|---|---|---|
| plain VarNet, rbdddrmf | 36.59 | 33.92 | 0.9311 | 0.9548 |
| plain VarNet + bottleneck FiLM label, qpaflt54 | 36.43 | 33.78 | 0.9299 | 0.9540 |
| PromptMR+ blind, seed 42 (75gex8ds) | 36.70 | 33.93 | 0.9315 | 0.9547 |
| PromptMR+ cond, seed 42 / 43 | 36.65 / 36.70 | 33.93 / 33.94 | 0.9312 / 0.9315 | 0.9547 / 0.9549 |

Under the legacy mask, PromptMR+ (83.6M parameters) is +0.1 dB on brain and +0.01 dB on knee over the 30M plain
VarNet. The bottleneck-FiLM label cost 0.15 dB brain and 0.14 dB knee on plain VarNet. The prompt label here costs
0 to 0.05 dB on the same metric.

## Conclusions
- Feeding the class to this PromptMR+ VarNet on fixed CAIPI R=4 neither helps nor hurts by a measurable amount.
  Every per-class gap over 2 seeds and 2 metrics is within ±0.08 dB, and the sign is not consistent across seeds or
  metrics, except a -0.02 to -0.03 dB knee-PDFS deficit whose CIs include 0.
- The null result is not the model ignoring the label: given a wrong label, the conditioned model loses 0.1 to
  0.8 dB. The label-blind model with the same parameters reaches the same quality without it. So the class adds
  nothing the network cannot get from the input. Why (e.g. that the anatomy and contrast are readable from the
  zero-filled image) is a hypothesis, untested.
- How much the conditioned model relies on the label is not stable across seeds, but its accuracy is. Some
  wrong-label costs differ by 2x or more between seeds. The label is one of several equivalent routes to the same
  solution.
- For deployment the conditioned model is strictly worse. It has no accuracy gain and a 0.1 to 0.8 dB loss when the
  class is wrong (e.g. a mislabelled body part or fat-suppression flag). It also refuses to run without a label.
- Consistent with the earlier plain-VarNet bottleneck-FiLM result (-0.15 dB, legacy mask, one seed): across two
  injection sites and two architectures, labelling the reconstructor has not helped on CAIPI R=4.

## Caveats
- Two seeds. Knee has 10 val volumes (4 knee-PD, 6 knee-PDFS), so the per-class knee CIs are wide (about ±0.15 dB
  for knee-PD). Effects below about 0.05 dB on knee cannot be resolved. The seed-averaged CI covers volume variance
  only, not seed variance.
- The two metrics disagree in sign on brain (per slice +0.01 / +0.02, trainer -0.08 / -0.01). The trainer metric
  pools MSE, so a few high-error slices dominate it. Which slices drive the difference was not checked. The
  outlier brain val volume AXT2_201_2010571 is a candidate (hypothesis, untested).
- The seed-43 blind arm started (2026-10-20 19:55) after the metric-mask commit f6e1e9a, whereas the seed-43 cond
  arm started at 13:08. Checked:
  - that commit changes only the trainer's metric computation, not the loss, data or RNG;
  - the model, conditioning, data and sampler sources were last modified before 13:08;
  - the two runs' configs differ only in `label_blind` and the seed;
  - they are scored at `last_model.pt`, so metric-based checkpoint selection plays no part.
  Its logged val metrics use the new mask while the other three logs use the legacy one. This report re-scores all
  four identically.
- Only one form of conditioning was tested: the label can only bias the 5-way prompt-selection softmax, through
  3,888 parameters. Richer injection (label-specific prompt maps, FiLM at every level, PromptMR+'s attention-based
  prompt fusion) was not tested.
- One easy regime: fixed CAIPI R=4, brain about 42 dB. A label could matter at higher acceleration or with learned,
  per-anatomy masks; untested.
- The plain-VarNet context runs were unseeded, single runs, scored with the legacy mask and not re-scorable.
- This supersedes the CAIPI line in `reports/METRIC_MASK_REEVAL_R8.md`, which compared both cond seeds with the
  seed-42 blind run only.

## Figures
None. Per-run reconstructions and masks are in each run's wandb page (IDs above) and `visualizations/`.

## Next steps
1. Test the redundancy hypothesis: linear probe for brain / knee-PD / knee-PDFS on the blind arm's pooled
   decoder features, and on the zero-filled CAIPI image.
2. Check the label where it should matter more: the same cond/blind pair at CAIPI R=8, or with the per-anatomy
   learned masks (anatomy routing, roadmap step 1).
3. If conditioning is kept anywhere, train with label dropout, so that a missing or wrong label degrades to the
   blind model instead of losing up to 0.8 dB.
4. Re-train the plain-VarNet CAIPI baseline with seed 42 and score it under the new mask, to measure what the 2.8x
   larger PromptMR+ model buys.
