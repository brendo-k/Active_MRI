# Why does the per-anatomy CG-SENSE mask lose on knee at R=8? (2026-10-20)

## Question
With CG-SENSE at R=8, the learned parts are the LOUPE mask(s) and the sensitivity U-Net. The reconstructor uses
10 CG iterations, lamda 0 and no learned prior. In that setup, one mask per anatomy beats the shared mask on brain but
loses to it on knee. Under VarNet the per-anatomy mask wins on both anatomies (`LOUPE_SINGLE_VS_SHARED_ANATOMY_R8.md`).
The questions:
1. Is the knee loss a property of knee, or of how the knee mask was trained?
2. In particular, is it the initialisation?
3. Did every per-class run start the same way?

Conventions:
- PSNR is the trainer's signal-masked per-slice PSNR in dB.
- Deltas are paired over the same slices, with common seeded Bernoulli mask draws.
- 95% CIs are volume-level bootstraps.
- Two kinds of delta appear and are not interchangeable:
  - **run − run:** each run scored with its own mask and its own sense net.
  - **mask effect:** two masks scored on one fixed sense net (anat_own − shared).

## Runs compared
All runs: CG-SENSE R=8 (R=4 where stated), balanced brain T2 + knee, LOUPE slope 10, lr 1e-3, batch 16.

| Run | Path (`checkpoints/loupe2d_cgsense_…`) | wandb | Epochs / lr / sens upsampling / seed | Mask rows (starting logits) |
|---|---|---|---|---|
| shared, 50 ep | `r8_brain_knee_shared` | xb6tna6g | 50 / step at 40 / conv / 42 | shared (draw 0) |
| shared, 50 ep, seed 43 | `r8_brain_knee_shared_seed43` | 5qa8q3ls | 50 / step / conv / 43 | shared (seed-43 draw 0) |
| shared, 50 ep bilinear | `r8_brain_knee_shared_bilinear_50ep` | wcxaz648 | 50 / step / bilinear / 42 | shared (draw 0) |
| shared, 50 ep bilinear, seed 43 | `r8_brain_knee_shared_bilinear_50ep_seed43` | wkr6xaqt | 50 / step / bilinear / 43 | shared (seed-43 draw 0) |
| anatomy, 50 ep | `r8_brain_knee_anatcond` | tk3c5ry8 | 50 / step / conv / 42 | brain (draw 0), knee (draw 1) |
| protocol, 50 ep | `r8_brain_knee_prototable` | 3dbidc2q | 50 / step / conv / 42 | brain (draw 0), knee-PD (draw 1), knee-PDFS (draw 2) |
| shared, 100 ep | `r8_brain_knee_shared_bilinear_const_100ep` | u2xei0pq | 100 / constant / bilinear / 42 | shared (draw 0) |
| anatomy, 100 ep | `r8_brain_knee_anatcond_bilinear_const_100ep` | kikc9627 | 100 / constant / bilinear / 42 | brain (draw 0), knee (draw 1) |
| control: shared, continued | `r8_brain_knee_shared_cont` | l57hbia4 | +50 / constant / bilinear / 42 | shared (from the shared 100 ep mask) |
| test: anatomy, shared-init | `r8_brain_knee_anatcond_sharedinit` | zc266a5v | +50 / constant / bilinear / 42 | brain and knee (both from the shared 100 ep mask) |
| R=4 shared | `r4_brain_knee_shared` | bu3ma9t2 | 50 / step / conv / 42 | shared (draw 0) |
| R=4 anatomy | `r4_brain_knee_anatcond` | nd02wof1 | 50 / step / conv / 42 | brain (draw 0), knee (draw 1) |

"Draw k" is the k-th logit field drawn by `ProbMask` from the run's seed (Results 3).

## Methods
1. **Cross-evaluation** (`scripts/diagnostics/anatcond_cgsense_cross_eval_r8.py --pair {50ep,100ep,sharedinit}`).
   - Every run's reconstructor (its sense net plus CG-SENSE) is scored with every run's masks, so a run − run gap
     splits into a mask part and a sense-net part.
   - Label swap: a per-class model is given the other anatomy's mask.
   - Val split: 688 slices.
     - Brain: 344 slices, 23 volumes.
     - Knee: 344 slices, 10 volumes (knee-PD 135, knee-PDFS 209).
   - `--split train` scores the slices the masks were fitted to.
     - Shared-init: all 1350 per anatomy.
     - 100 ep: every 4th slice, 338 brain and 337 knee.
   - Also reported:
     - the training loss (unmasked L1, the same quantity as the trainer's `F.l1_loss`);
     - each map's share of sampling probability by k-space radius;
     - the pixel correlation between probability maps.
2. **Shared-init test** (`scripts/launch/train_loupe2d_cgsense_r8_sharedinit.sh`; the decision rule is in its header
   and was fixed before the runs).
   - Both arms load the shared 100 ep run's last checkpoint: its mask goes into every class row, plus its sense net
     and a fresh Adam.
   - Both then train 50 more epochs at constant lr 1e-3, seed 42.
   - The test arm has one mask row per anatomy; the control keeps one shared mask.
3. **Starting draws** (`scripts/diagnostics/cgsense_init_draws_r8.py`, output
   `analysis/anatcond_cgsense/init_draws.json`).
   - `ProbMask` draws one uniform logit field per class, in class order, from the global RNG right after the
     trainer seeds it.
   - So `torch.manual_seed(seed); ProbMask(shape, slope, 3)` reproduces each run's starting logits.
   - The script correlates every row's final logits with draws 0–2, and every row's final probability map with
     every other row's.
4. Earlier numbers are reused from three sources:
   - `analysis/cgsense_mask_variants/eval.md` (per-protocol run);
   - `reports/cgsense_anatcond/paired_r4.json` (R=4);
   - `reports/sens_upsample_r8/seeds_comparison.json` (seed 42 vs 43 shared masks).

## Results

### 1. From scratch, brain wins and knee loses (val, run − run)

| Comparison | Brain | Knee | Knee-PD | Knee-PDFS |
|---|---|---|---|---|
| anatomy − shared, 50 ep | +0.101 [+0.039, +0.185] | −0.145 [−0.230, −0.078] | −0.118 | −0.163 |
| protocol − shared, 50 ep | +0.089 [+0.027, +0.180] | −0.433 [−0.485, −0.366] | −0.407 | −0.449 |
| anatomy − shared, 100 ep | +0.158 [+0.093, +0.238] | −0.250 [−0.343, −0.171] | −0.155 | −0.312 |
| R=4 anatomy − shared, 50 ep | −0.049 [−0.078, −0.020] | +0.066 [+0.018, +0.114] | | |

The per-protocol run gave each knee protocol its own mask, and it lost more than the per-anatomy run, not less.

### 2. The knee gap is the mask, not the sense net (val, mask effect)

| 100 ep pair | Brain | Knee | Knee-PD | Knee-PDFS |
|---|---|---|---|---|
| mask effect, shared sense net | +0.163 [+0.107, +0.243] | −0.246 [−0.307, −0.190] | −0.171 | −0.294 |
| sense-net effect, shared mask | −0.011 [−0.028, +0.004] | −0.020 [−0.073, +0.028] | −0.008 | −0.028 |
| label swap: knee given the brain mask (shared sense net) | | +0.215 [+0.101, +0.348] | +0.046 | +0.324 |

The 50 ep pair shows the same thing: knee mask effect −0.232 [−0.311, −0.168], sense-net effect +0.06 (CI
[−0.01, +0.14]).

In-sample, the 100 ep knee mask still loses (every 4th training slice):
- Knee: −0.147 [−0.229, −0.079]. Knee-PD gains (+0.062 / +0.115 on the shared / per-anatomy sense net), but
  knee-PDFS loses −0.341.
- Its training loss on knee is higher than the shared mask's: L1 0.05553 vs 0.05495.

So the knee row did not even fit its own training data as well as the shared mask did.

### 3. Every per-class run started its non-brain rows from fresh draws, and from the same ones

Pixel correlation between each row's final logits and the logits drawn from its seed:

| Row | Draw 0 | Draw 1 | Draw 2 |
|---|---|---|---|
| shared masks, seed 42 (50 ep conv, 50 ep bilinear, 100 ep) | +0.42 to +0.52 | 0.01 | 0.00 |
| shared masks, seed 43 (correlated with seed 43's draws) | +0.51 / +0.53 | −0.01 | 0.00 |
| brain row: anatomy 50 ep, anatomy 100 ep, protocol 50 ep | +0.49 / +0.40 / +0.49 | 0.00 | 0.00 |
| knee row: anatomy 50 ep, anatomy 100 ep | 0.00 | +0.59 / +0.48 | 0.00 |
| knee-PD row: protocol 50 ep | 0.00 | +0.57 | 0.00 |
| knee-PDFS row: protocol 50 ep | 0.00 | 0.01 | +0.62 |
| R=4: shared / brain row / knee row | +0.63 / +0.63 / 0.00 | 0.00 / 0.00 / +0.68 | 0.00 |
| shared-init: control / brain row / knee row | +0.39 / +0.40 / +0.40 | 0.01 | 0.00 |

1. **Training keeps the starting pixel layout.**
   - Rows from the same draw end up correlated across runs, even with different epochs, lr schedule and upsampling.
     The shared 50 ep conv vs shared 100 ep bilinear maps correlate 0.81. The knee rows of the 50 ep and 100 ep
     anatomy runs correlate 0.76, and the protocol run's knee-PD row correlates 0.69 / 0.62 with them.
   - Rows from different draws correlate 0.05–0.16. The seed-42 and seed-43 shared masks correlate 0.12–0.15.
2. **All three R=8 knee results share one starting draw.** All runs used seed 42, so the knee rows of both anatomy
   runs and the protocol run's knee-PD row all started from draw 1.
   - The brain row always started from draw 0, the same draw as its shared baseline.
   - So the brain-wins / knee-loses pattern of the from-scratch runs is confounded with whether a row started from
     its baseline's draw.
3. **Draw 1 itself is probably not the culprit.**
   - A shared mask barely depends on its draw: seed 43 minus seed 42 is +0.046 dB on knee (conv) and +0.035
     (bilinear), and −0.014 / −0.013 on brain.
   - The R=4 knee row started from the same draw 1, and the R=4 anatomy run beat its shared run on knee (+0.07,
     run − run).
   - What the losing knee rows have in common is a random start trained on only the knee half of each batch.
     Whether any random start would lose at R=8 is untested.

This covers every LOUPE per-class run. The DCM variants were not checked.

### 4. Shared-init test: starting from the shared mask removes most of the knee loss

Mask effect, anat_own − shared mask, val (344 slices per anatomy):

| | Shared sense net | Per-anatomy sense net | Run − run |
|---|---|---|---|
| Brain | +0.154 [+0.110, +0.221] | +0.158 [+0.113, +0.223] | +0.184 [+0.134, +0.253] |
| Knee | **−0.035 [−0.076, +0.002]** | **−0.019 [−0.061, +0.020]** | +0.018 [−0.029, +0.069] |
| Knee-PD | +0.017 [−0.015, +0.043] | +0.041 [−0.001, +0.071] | +0.101 [+0.036, +0.135] |
| Knee-PDFS | −0.068 [−0.118, −0.032] | −0.057 [−0.103, −0.023] | −0.035 [−0.064, −0.009] |

The same effect in-sample (1350 slices per anatomy):

| | Shared sense net | Per-anatomy sense net | Run − run |
|---|---|---|---|
| Brain | +0.159 [+0.132, +0.190] | +0.164 [+0.136, +0.198] | +0.171 [+0.146, +0.205] |
| Knee | +0.004 [−0.025, +0.030] | +0.027 [−0.004, +0.056] | +0.048 [+0.009, +0.082] |
| Knee-PD | +0.075 [+0.041, +0.107] | +0.111 [+0.078, +0.141] | +0.151 [+0.115, +0.184] |
| Knee-PDFS | −0.061 [−0.083, −0.041] | −0.050 [−0.071, −0.032] | −0.047 [−0.068, −0.025] |

- **Sense net.** The per-anatomy run's sense net is now slightly better on both anatomies: +0.027 to +0.053 on val,
  all CIs above 0.
- **Label swap on the shared sense net.** Knee-PDFS is better with the brain mask than with the knee mask: +0.098
  [+0.039, +0.168] on val, +0.143 in-sample. Knee-PD is worse with it: −0.094 on val, −0.313 in-sample. Brain given
  the knee mask: −0.655.
- **Training loss** (unmasked L1, per-anatomy sense net, in-sample):

  | Anatomy | Own row | Shared mask |
  |---|---|---|
  | Knee | 0.05424 | 0.05415 |
  | Knee-PD | 0.02564 | 0.02566 |
  | Knee-PDFS | 0.08056 | 0.08036 |
  | Brain | 0.06070 | 0.06112 |

How far each map moved from the start:

| Map | Share of probability by radius (px) 0–8 / 8–16 / 16–32 / 32–64 / ≥64 | Pixel corr. with start |
|---|---|---|
| start (shared 100 ep) | 0.4 / 2.8 / 8.7 / 24.7 / **63.4%** | |
| control, shared continued | 0.4 / 2.6 / 8.5 / 24.5 / **63.9%** | 0.993 |
| test, brain row | 0.4 / 2.7 / 9.0 / 26.9 / **61.0%** | 0.966 |
| test, knee row | 0.5 / 2.4 / 6.5 / 19.7 / **71.0%** | 0.912 |

The brain and knee rows of the test arm correlate 0.85 with each other.

Against the decision rule:

| Outcome | Criterion | Met? |
|---|---|---|
| Init / basin | knee mask effect CI covers 0 | yes, on both sense nets |
| | knee row's drift and r ≥ 64 share track the brain row and the control | no: it moves outward (71.0% vs 61.0% and 63.9%) and decorrelates more (0.91 vs 0.97 and 0.99) |
| Knee gradient pulls it away | knee mask effect negative | no overall; yes for knee-PDFS (−0.06, CI below 0 on val and in-sample) |
| | knee row decorrelates or flattens more than both | yes |

Neither outcome holds cleanly:
- The random start accounts for about 0.2 of the 0.25 dB knee loss.
- A residual of about −0.06 dB on knee-PDFS remains, and it comes with the knee row drifting outward.

### 5. What drives the residual is not settled
Two explanations fit, and the existing runs do not separate them.
- **Gradient noise.** The knee row gets half of each batch.
  - It drifted further from the start than either other map, yet gained nothing on its own training slices (+0.004
    dB; L1 0.05424 vs 0.05415).
  - The brain row, from the same start with the same batch share, gained +0.16 in-sample.
  - Large drift without an in-sample gain is what a noise-driven walk looks like. The 100 ep from-scratch knee row
    had the same symptom: a higher training loss than the shared mask.
- **PD/PDFS conflict.** One knee row serves two protocols that want different masks.
  - In-sample, knee-PD gains (+0.08 to +0.11) and knee-PDFS loses (−0.05 to −0.06).
  - Knee-PDFS prefers the brain mask (+0.10 to +0.14); knee-PD does not (−0.09 to −0.31).
  - A row that trades PD against PDFS would look like this. It does not by itself explain why the summed knee
    training loss fails to improve.

## Conclusions
- **Answer to "could it be the init?": mostly yes.**
  - In every earlier per-class LOUPE run, the non-brain rows started from fresh random draws while the brain row
    shared its baseline's draw.
  - Started from the trained shared mask instead, the knee row ties the shared mask on val: −0.02 to −0.04 dB, CI
    covering 0, against −0.25 from scratch.
  - Strictly, the shared-init test changed two things at once: which draw, and random vs trained starting state. It
    supports "a random start trained on half the batch underperforms" more than "draw 1 is a bad draw". A shared mask
    is insensitive to its draw (≤0.05 dB), and the R=4 knee row from draw 1 won.
- **Answer to "was that every run?": yes**, for every LOUPE per-class run.
  - Seed 42 throughout. Knee and knee-PD always started from draw 1, knee-PDFS from draw 2, brain from draw 0.
  - The three R=8 knee results are therefore one starting draw, not three independent ones.
- **The per-protocol result (−0.43 dB on knee) is not evidence against protocol masks.** It combined fresh draws for
  both knee rows with about a quarter of each batch per knee row. It has not been tested from a good start.
- **Per-anatomy masks under CG-SENSE R=8 are worth about +0.15 dB on brain and about 0 on knee.** The brain gain is
  stable across 50 ep, 100 ep and the shared-init test (+0.10 to +0.18).
- **A residual knee-PDFS loss of about 0.06 dB remains**, with the knee row moving its samples outward. Whether this
  is gradient noise or a PD/PDFS conflict is open.

## Caveats
- Knee has 10 val volumes (PD 135 slices, PDFS 209) and 40 training volumes. The CIs are volume bootstraps, so knee
  CIs rest on 10 units.
- The shared-init test is one seed per arm. Both arms restart Adam, so the warm start is not a pure continuation.
- The 100 ep in-sample split scored every 4th training slice (338 / 337); the shared-init split scored all 1350.
- The seed 42 vs 43 shared comparison is a mean difference without a CI (per-slice SD 0.13–0.18 dB).
- The R=4 numbers come from the earlier paired evaluation (`paired_r4.json`), not from the cross-evaluation, and are
  not decomposed into mask and sense net.
- The shared-init sense-net effect (+0.03 to +0.05 dB, favouring the per-anatomy run) is reported but not explained.
  It is small next to the mask effects.

## Figures
No new figures.
- Masks of the from-scratch variants: `reports/figures/cgsense_mask_variants_r8.png`.
- Numbers: `analysis/anatcond_cgsense/cross_eval_{50ep,100ep,100ep_train,sharedinit,sharedinit_train}.{json,log}`
  and `analysis/anatcond_cgsense/init_draws.json`.
- `cross_eval_sharedinit_smoke*` are an earlier every-64th-slice check, taken with the control at about epoch 33.
  Not used here.

## Next steps
1. **Per-protocol masks from the shared start.** Three rows (brain, knee-PD, knee-PDFS), all loaded from the shared
   100 ep mask; 50 epochs, constant lr 1e-3, seed 42. The existing `shared_cont` run is the control, so only one new
   run is needed. This separates the two explanations in Results 5:
   - **Conflict:** the PD row gains and the PDFS row holds or gains, in-sample and on val, without moving outward.
   - **Noise:** the PDFS row, now with about a quarter of each batch, drifts more than the two-class knee row did and
     still loses in-sample.

   The cross-evaluation script needs one change first: it takes the class label from `batch['anatomy']`, so it
   cannot address protocol rows.
2. **Draw 1 vs any random start.** Repeat the from-scratch anatomy run with seed 43 (50 ep, bilinear, step lr)
   against `shared_bilinear_50ep_seed43`. If knee again loses about 0.2 dB, random starts in general are the problem.
3. If noise is confirmed: lower the mask lr or decay it, or average the per-class gradient over more slices, before
   trying more mask classes.
