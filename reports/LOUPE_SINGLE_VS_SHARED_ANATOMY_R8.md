# Why does a shared LOUPE mask beat single-anatomy ones? Data, steps, mask and reconstructor (R=8, 2026-10-12)

## Question
A learned 2D LOUPE mask and its VarNet reconstructor can be trained on brain and knee together (one shared
mask) or on one anatomy alone. Single-anatomy training gives a specialised mask and reconstructor but half the
data per epoch. This report asks (1) whether specialising helps, (2) whether the shared model's lead at equal
epochs comes from having more training slices or more gradient steps, and (3) how much of a model's quality comes from its mask and how
much from its reconstructor. All numbers are on the validation slices, scored with the trainer's
background-masked metric; "brain" and "knee" are scored separately; PSNR is in dB. Mask columns are the
phase-encode axis.

## Runs compared
Same setup for all: `recon=varnet` (12 cascades, 18 channels), `datasets=brain_knee_balanced`,
`sampler=loupe_2d`, R=8 (12.5% sampling), batch size 16, lr 3e-4, weight decay 0.0, training seed 42 (one seed per
configuration). The balanced train set has 1350 slices per anatomy (89 brain and 40 knee volumes), 2700 in total;
the val set has 344 slices per anatomy (23 brain and 10 knee volumes). Single-anatomy runs validate on both
anatomies.

| Name | Checkpoint | wandb | Epochs (lr step) | Train data | Mask | Best epoch |
|---|---|---|---|---|---|---|
| shared | `checkpoints/loupe2d_varnet_r8_brain_knee` | `azd6r64k` | 50 (40) | brain + knee, 2700 slices | one learned | 49 |
| shared_100ep | `checkpoints/loupe2d_varnet_r8_brain_knee_100ep` | `1k9d0xdj` | 100 (90) | brain + knee, 2700 slices | one learned | 90 |
| anatcond | `checkpoints/loupe2d_varnet_r8_brain_knee_anatcond` | `pjysura8` | 50 (40) | brain + knee, anatomy label to sampler | one learned per anatomy | 48 |
| brainonly | `checkpoints/loupe2d_varnet_r8_brainonly_stepmatched` | `obhu7ax7` | 100 (80) | brain only, 1350 slices | learned | 61 |
| kneeonly | `checkpoints/loupe2d_varnet_r8_kneeonly_stepmatched` | `slhb0p0y` | 100 (80) | knee only, 1350 slices | learned | 71 |
| brainonly2700 | `checkpoints/loupe2d_varnet_r8_brainonly_2700slices_100ep` | `u5e1xbsq` | 100 (90) | brain only, 2700 slices (177 volumes) | learned | 80 |
| fixedmasks | `checkpoints/loupe2d_varnet_r8_brain_knee_fixedsinglemasks_100ep` | `1mj0aie5` | 100 (90) | brain + knee, 2700 slices | brainonly's and kneeonly's masks, frozen | 91 |
| brainonly1350_200ep | `checkpoints/loupe2d_varnet_r8_brainonly_1350slices_200ep` | `1ahao515` | 200 (180) | brain only, 1350 slices | learned | 129 |
| fixed2700mask | `checkpoints/loupe2d_varnet_r8_brain_knee_fixed2700brainmask_100ep` | `g449cnkq` | 100 (90) | brain + knee, 2700 slices | brainonly2700's (brain) and kneeonly's (knee) masks, frozen | 86 |

"Step-matched": a single-anatomy epoch has half the slices, so 100 epochs is about the gradient work of 50 shared
epochs. Measured epoch time is 138 s (single-anatomy) against 263 s (shared, 50 epochs) and 278 s (shared, 100
epochs), so `brainonly` and `kneeonly` are matched to `shared`, not to `shared_100ep`. Step counts were not
compared directly. `brainonly2700` has the slice count and number of steps of `shared_100ep`;
`brainonly1350_200ep` has the same number of steps with half the unique slices. The best epoch is
chosen by validation loss over both anatomies and uses the trainer's epoch field.

`fixedmasks` is a fresh VarNet and sensitivity net trained from scratch on brain + knee. Its sampler has one
mask slot per anatomy, filled from the `best_model.pt` of `brainonly` (brain slot) and `kneeonly` (knee slot)
and frozen (`sampler.fixed_mask_checkpoints`, `sampler.anatomy_conditioning=true`). Only the reconstructor
trains. `fixed2700mask` is the same with the brain slot taken from `brainonly2700`'s `best_model.pt`. For both,
the saved masks at the best and last epoch are bit-identical to their source masks (epochs 91 and 99 for
`fixedmasks`, 86 and 99 for `fixed2700mask`).

Validation metrics, mean over two mask draws (PSNR dB / SSIM):

| Run | Brain | Knee |
|---|---|---|
| shared | 36.07 / 0.9299 | 33.90 / 0.9493 |
| shared_100ep | 36.55 / 0.9327 | 34.15 / 0.9527 |
| anatcond | 36.25 / 0.9310 | 34.05 / 0.9510 |
| anatcond, labels swapped | 36.25 / 0.9307 | 33.97 / 0.9502 |
| brainonly | 35.98 / 0.9292 | 28.26 / 0.9224 |
| kneeonly | 35.05 / 0.9257 | 33.97 / 0.9507 |
| brainonly2700, best-val epoch | 36.43 / 0.9319 | 29.57 / 0.9340 |
| brainonly2700, epoch 100 | 36.48 / 0.9323 | 29.45 / 0.9327 |
| fixedmasks | 36.39 / 0.9318 | 34.14 / 0.9520 |
| fixedmasks, labels swapped | 36.39 / 0.9317 | 34.06 / 0.9511 |
| brainonly1350_200ep, best-val epoch | 36.12 / 0.9303 | 28.29 / 0.9268 |
| brainonly1350_200ep, epoch 200 | 36.06 / 0.9297 | 28.02 / 0.9226 |
| fixed2700mask | 36.52 / 0.9327 | 34.19 / 0.9523 |
| fixed2700mask, labels swapped | 36.47 / 0.9323 | 34.17 / 0.9521 |

All SSIM values are above 0.92, so no run looks broken.

## Methods
1. **Training.** The single-anatomy runs used `datasets.train_anatomies=[brain]` or `[knee]` through
   `scripts/train_loupe2d_varnet_r8_brain_knee.sh` with `training.num_epochs=100 training.lr_step_size=80`.
   `brainonly1350_200ep` is the `brainonly` setup with `training.num_epochs=200 training.lr_step_size=180`.
   `brainonly2700` adds `datasets.train_slices=2700`, which gives the trained anatomy 2700 train slices instead
   of the balanced 1350 (the loader starts from all 200 brain T2 train volumes and trims to 2700; val is
   unchanged). `fixedmasks` uses `sampler.anatomy_conditioning=true` and
   `sampler.fixed_mask_checkpoints={brain: ..., knee: ...}` with 100 epochs and lr step 90 (`fixedmasks` takes both
   masks from the 1350-slice single-anatomy runs, `fixed2700mask` takes the brain mask from `brainonly2700`). The shared and
   conditioned runs were trained earlier with the same script and other epoch settings.
2. **Paired evaluation.** `scripts/compare_anatcond_r8.py` loads each checkpoint, runs every model on the same
   val slices with the same random-number stream for the Bernoulli mask draws, and scores with the trainer's
   masked metric. It repeats this for two mask-draw seeds (1234, 4321) and averages them. Every run is scored at
   its best-val-loss epoch, plus `brainonly2700` and `brainonly1350_200ep` also at their last epoch (`_last`).
3. **Paired statistics.** Per-slice difference between two models, averaged over slices. The 95% confidence
   interval (CI) is `1.96 * std / sqrt(n)` over per-volume means of that difference, with n = 23 brain volumes or
   10 knee volumes. "Better on x% of volumes" is the share of volumes with a positive mean difference.
   The CI covers slice and volume variation only; it does not include variation between training seeds.
4. **Label swap.** A conditioned model (`anatcond`, `fixedmasks`) is given the other anatomy's label, so each
   slice is sampled with the other anatomy's mask and the same reconstructor.
5. **Mask freeze check.** The saved `sampler.mask_net.mask` of `fixedmasks` was compared with the source
   checkpoints with `torch.equal`. `tests/test_fixed_masks.py` covers the freezing and the optimizer exclusion.

## Results
### Does specialising help? (single-anatomy against shared, own anatomy, PSNR dB)
The CI is 95% over volumes.

| Comparison | Brain-only on brain | Knee-only on knee |
|---|---|---|
| vs shared (50 epochs, step-matched) | +0.24 [+0.11, +0.37], better on 83% of volumes | +0.52 [+0.21, +0.83], 90% |
| vs anatcond (50 epochs) | -0.12 [-0.19, -0.04], 13% | -0.15 [-0.43, +0.13], 60% |
| vs shared_100ep | -0.92 [-1.10, -0.74], 0% | -0.53 [-0.70, -0.36], 0% |

Against the step-matched shared run, specialising gains 0.24 dB (brain) and 0.52 dB (knee). The 100-epoch shared
run beats both specialised models.

### Is the shared model's lead data, or steps? (brain, PSNR dB)
`brainonly2700` has the same number of train slices and steps as `shared_100ep`, but all brain. `brainonly1350_200ep`
has the same steps with half the unique slices.

| Comparison | Brain | Knee |
|---|---|---|
| brainonly2700 (2700 slices) - brainonly (1350 slices, 100 epochs) | +0.80 [+0.63, +0.96], better on 100% of volumes | +1.71 [+0.95, +2.46] |
| brainonly1350_200ep - brainonly (same slices, twice the epochs), best-val epochs | +0.44 [+0.35, +0.54], 96% | +0.52 [+0.21, +0.83] |
| brainonly1350_200ep - brainonly2700 (same steps, half the unique slices), best-val epochs | -0.36 [-0.44, -0.27], 0% | -1.19 [-1.92, -0.45] |
| same, epoch 200 against epoch 100 | -0.82 [-0.98, -0.67], 0% | -1.66 [-2.39, -0.93] |
| brainonly2700, best-val epoch - shared_100ep | -0.12 [-0.16, -0.09], 0% | -5.88 [-7.63, -4.13] |
| brainonly2700, epoch 100 - shared_100ep | +0.03 [-0.001, +0.065], 83% | -5.86 [-7.61, -4.12] |
| brainonly1350_200ep - shared_100ep, best-val epochs | -0.48 [-0.59, -0.37], 0% | -7.07 [-8.61, -5.52] |
| brainonly1350_200ep - shared_100ep, epoch 200 against 100 | -0.79 [-0.94, -0.64], 0% | -7.53 [-9.15, -5.90] |
| brainonly1350_200ep: epoch 200 - best-val epoch (129) | -0.31 [-0.38, -0.24], 4% | -0.46 [-0.69, -0.23] |

Twice the steps on the same 1350 brain slices helps by 0.44 dB, but it overfits: the final epoch is 0.31 dB below
the best-val epoch (train loss 0.0536, val loss 0.0621 at epoch 200 against 0.0613 at epoch 129). Twice the unique
slices at the same steps is worth a further 0.36 dB (best-val epochs) to 0.82 dB (final epochs). A brain-only model
with 2700 brain slices ties the joint model on brain (+0.03 dB at epoch 100, -0.12 dB at the best-val epoch).

### Mask against reconstructor (frozen masks, joint reconstructor, PSNR dB, own anatomy unless noted)

| Comparison | Brain | Knee | What it isolates |
|---|---|---|---|
| fixedmasks - brainonly / kneeonly | +0.53 [+0.45, +0.61] | +0.45 [+0.14, +0.76] | same mask, joint reconstructor with twice the data and steps against a specialised one |
| fixedmasks - shared_100ep | -0.39 [-0.51, -0.27] | -0.08 [-0.36, +0.20] | same epochs, frozen 1350-slice masks against a jointly learned shared mask |
| fixed2700mask - fixedmasks | +0.21 [+0.14, +0.29] | +0.06 [-0.001, +0.13] | same joint training, brain mask learned on 2700 instead of 1350 slices (knee mask identical) |
| fixed2700mask - shared_100ep | -0.18 [-0.23, -0.13] | -0.02 [-0.30, +0.27] | the remaining gap to the jointly learned shared mask |
| fixedmasks - fixedmasks (labels swapped) | -0.05 [-0.08, -0.01] | +0.31 [+0.22, +0.40] | mask specificity with a joint reconstructor |
| fixed2700mask - fixed2700mask (labels swapped) | +0.14 [+0.10, +0.18] | +0.07 [+0.04, +0.10] | the same, with the 2700-slice brain mask |
| fixedmasks - anatcond | +0.41 [+0.32, +0.51] | +0.30 [+0.19, +0.41] | not like-for-like: 100 epochs against 50 |

With the mask held fixed, a reconstructor trained on both anatomies for twice as many steps gains about 0.5 dB over a
specialised one. A brain mask learned on 2700 slices instead of 1350 is worth 0.21 dB in the joint reconstructor,
about half of the 0.39 dB brain gap that the 1350-slice masks leave to `shared_100ep`. The knee mask is the same in
both runs, and knee results do not move. The label swap shows that the 2700-slice brain mask is anatomy-specific on
brain (-0.14 dB when replaced by the knee mask), unlike the 1350-slice one (+0.05 dB).

### Other effects (PSNR dB, paired per slice)

| Comparison | Brain | Knee |
|---|---|---|
| shared_100ep - shared (twice the epochs) | +1.16 [+0.88, +1.44] | +1.05 [+0.64, +1.45] |
| anatcond - shared | +0.36 [+0.28, +0.44] | +0.67 [+0.46, +0.88] |
| anatcond - anatcond (labels swapped) | -0.02 [-0.05, +0.01] | +0.29 [+0.21, +0.37] |

Single-anatomy models on the other anatomy (PSNR minus shared): brain-only on knee -6.54 [-8.08, -5.00];
knee-only on brain -1.63 [-1.97, -1.29]. SSIM differences have the same sign as PSNR except where the PSNR change
is small or the CI includes 0 (for example brain-only on brain against shared: SSIM -0.0008, CI includes 0, and
`brainonly2700` against `shared_100ep` on brain: SSIM -0.0007, CI includes 0). All numbers are in
`reports/single_anatomy_r8/paired.json`.

## Conclusions
- Against a shared run with the same gradient work, a single-anatomy mask plus reconstructor gains about
  0.24 dB (brain) and 0.52 dB (knee) on its own anatomy. It does not beat the anatomy-conditioned model, which
  has one reconstructor for both anatomies.
- The shared model's lead at 100 epochs comes from the number of unique training slices, not from steps. At equal
  steps, half the unique slices costs 0.36 dB (best-val epochs) to 0.82 dB (final epochs) on brain, and a
  brain-only model on 1350 slices overfits within 200 epochs (final epoch 0.31 dB below the best).
- The extra slices need not be brain: with the same total slices and steps, a brain-only model ties the
  brain+knee model on brain (+0.03 dB at epoch 100, -0.12 dB at the best-val epoch). So 1350 knee slices helped
  brain recon about as much as 1350 more brain slices did, within the roughly 0.1 dB this design can resolve.
  Pooling anatomies is a way to get more slices; the data does not show a benefit specific to cross-anatomy training.
- A reconstructor trained on both anatomies is worth about 0.5 dB over a specialised one when the mask is held
  fixed, which is consistent with the data explanation above.
- Mask quality also scales with data: a brain mask learned on 2700 brain slices is 0.21 dB better in a joint
  reconstructor than one learned on 1350, and closes about half of the brain gap to the jointly learned shared mask.
  The remaining 0.18 dB (brain) is unexplained. Hypothesis, untested: it comes from the mask being learned together
  with its reconstructor (co-adaptation), or from the knee data in the mask's training.
- Mask specificity differs by anatomy: knee needs its own mask (-0.31 dB when swapped with the 1350-slice brain mask,
  -0.07 dB with the 2700-slice one, -0.29 dB in `anatcond`). The brain mask is anatomy-specific only for the
  2700-slice version (-0.14 dB when replaced by the knee mask).
- Hypothesis, untested: a more stable or transferable representation is not needed to explain the shared model's
  lead, but nothing here tests representation stability directly.

## Caveats
- One training seed per configuration. Differences of about 0.1 dB (for example `brainonly2700` against
  `shared_100ep`, or brain-only against anatcond) are not resolvable, and the CIs ignore seed variance.
- The checkpoint rule changes some results. Best-val-loss selection is over both anatomies, so for single-anatomy
  runs it can pick an earlier epoch than ideal for their own anatomy: `brainonly2700` scores 0.16 dB higher on
  brain at epoch 100 than at epoch 80. The other runs in the table are scored at the best-val epoch only, so
  the `brainonly`, `kneeonly` and frozen masks inherit this selection; `fixedmasks` uses their `best_model.pt`
  masks from epochs 61 and 71, before the lr drop at 80.
- `fixedmasks` against `anatcond` and `shared` mixes epoch count with the mask difference (100 against 50 epochs);
  the like-for-like comparison is `shared_100ep`.
- `brainonly1350_200ep` overfits after about epoch 130, so its best-val epoch (129) and last epoch differ by 0.31 dB.
  The two checkpoint rules give different gaps to `brainonly2700` (-0.36 against -0.82 dB); the conclusion that unique
  slices matter holds under both. The best-val rule selects on the combined brain+knee loss, which a brain-only model
  does not optimise, so the best-val rows are noisier than the final-epoch rows.
- The knee-slice-equals-brain-slice reading rests on one seed per run and the 0.1 dB resolution above; it compares
  `brainonly2700` with `shared_100ep` only at their final epochs (+0.03 dB), while their best-val epochs differ by -0.12 dB.
- The step-matching rests on epoch times, not counted steps. The lr step is 80 of 100 for `brainonly` and
  `kneeonly`, 40 of 50 for `shared`, and 90 of 100 for the 100-epoch runs.
- The first launches of `brainonly1350_200ep` and `fixed2700mask` on 2026-10-10 failed with CUDA out-of-memory errors
  because other jobs took the same GPUs while their dataset scans (minutes long) were still running; the runs reported
  here are the relaunch with the same settings.
- The first launch of the single-anatomy runs on 2026-10-06 exited about 20 s after start with no error. The
  runs reported here are the relaunch; `train_sequential.log` in each `brainonly`/`kneeonly` directory contains
  both attempts. The cause of the first failure was not found. The first launch of `brainonly2700` ran out of GPU
  memory because another job was sharing the device; it was relaunched unchanged.
- The 50-epoch single-anatomy control (`brainonly_epochmatched`) was not run.
- `scripts/compare_anatcond_r8.py` was extended for this comparison; it now loads all the checkpoints listed above.
- `datasets.train_slices` and `sampler.fixed_mask_checkpoints` are new options added for these runs. Default
  behaviour of the balanced loader was confirmed unchanged (89 brain and 40 knee volumes, 1350 slices each).

## Figures
None produced. Numbers: `reports/single_anatomy_r8/paired.json` (aggregate and paired statistics) and
`reports/single_anatomy_r8/per_slice.pt` (per-slice PSNR/SSIM and volume ids).

## Next steps
1. **Freeze the shared mask.** Train a joint reconstructor from scratch on `shared_100ep`'s own learned mask (both
   anatomy slots set to it). If it matches `shared_100ep`, the mask accounts for the remaining 0.18 dB brain gap;
   if it falls short, co-adaptation of mask and reconstructor does.
2. **Test the cross-anatomy transfer reading.** Train brain-only on 2700 brain slices against brain+knee with
   1350 + 1350, and brain-only against a joint model with a smaller knee share, to check whether knee slices
   substitute for brain slices across the range.
3. **Measure seed variance.** Repeat `shared_100ep`, `brainonly2700`, `brainonly1350_200ep` and `fixed2700mask`
   with a second seed so that differences of about 0.1 dB can be judged.
4. **Fix checkpoint selection.** For single-anatomy runs, choose the epoch on their own anatomy's val loss, or
   report the final epoch, so that comparisons do not depend on a joint-loss rule.
