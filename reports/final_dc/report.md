# Does a final data-consistency step on the scored output help R=8 LOUPE + VarNet? Stage 1, inference only (2026-11-09)

| | |
|---|---|
| Code | results produced at `d867595` (final_dc: summary script that decides the frozen criteria, and run.sh); branch `exp/final_dc` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 10 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 15 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |

- **Branch:** `exp/final_dc`. Scores produced at `eab75aa` (`scripts/eval/final_dc_eval.py`, unchanged since); criteria and
  tables at `d867595`. Nothing was trained.
- **Status:** awaiting the human verdict.
- **Proposal:** `proposal.md`, frozen at `3ff952b` on 2026-11-09.

### Proposal vs outcome

All numbers are PSNR_Masked dB at λ = 1 minus λ = 0, on the same checkpoint and slices.

| # | Criterion (frozen) | Evidence | Met? |
|---|---|---|---|
| 0 | λ = 0 reproduces each published eval within 0.003 dB | gap +0.0000 dB in all 10 checkpoints (2x2: max per-slice gap 0.0000) | **met** |
| C1 | Mean over the 8 checkpoints ≥ +0.10 dB, positive in all 8 | mean −2.002 dB; range −1.038 (`muon_s43`) to −2.205 (`muon_layer_s42`); 0 of 8 positive | not met |
| C2 | ΔSSIM_Masked ≥ −0.002 in both `muon_layer` seeds | −0.0075, −0.0310 | not met |
| C3 | No condition worse by ≥ 0.10 dB in both seeds of any recipe | every recipe is worse in every condition, in both seeds | not met |
| F | Failure: mean < +0.05 dB, or either `muon_layer` seed negative | both hold | **failure** |
| Side (a) | kspace_gt's deficit to rss (same λ) shrinks to ≤ 50% of 2.39 dB | the deficit becomes a +0.43 dB lead at λ = 1, but only because rss falls 2.24 dB | met as worded, for the wrong reason |
| Side (b) | kspace_gt's knee-PDFS output/target mean ≥ 0.95 at λ = 1 | 0.883 (0.862 at λ = 0) | not met: the step does not recover the gap |

| Prediction (frozen) | Observed |
|---|---|
| Brain and knee-PD: PSNR up, +0.1 to +0.5 dB | down: −1.36 and −0.89 dB (8-checkpoint means) |
| Knee-PDFS may go down, output darkens below ~0.997 | down −3.73 dB; output/target mean 0.992 → 0.891 |
| SSIM either sign | down at λ = 1 (−0.0094); +0.0004 at λ = 0.25 |
| kspace_gt brightens only a little: knee-PDFS mean 0.87 → ~0.89, gap not recovered; same-λ gap shrinks mostly because rss darkens | 0.862 → 0.883; gap to the rss control −2.39 → −1.81 dB; rss at λ = 1 −2.24 dB |

### Recommendation

**Park.** Under the frozen rule the stage-1 result is a failure, so nothing is trained. Do not add a final DC step, hard or
with a scalar gate, at inference to RSS-trained checkpoints. Every λ from 0.25 up lowers PSNR. What this study does add
is a finding about the reconstructor, below. It bears on the loss/target choice and on multi-step sampling (Next steps).

## Question

The scored output of 1-step LOUPE + 12-cascade E2E VarNet (R=8) does not reproduce the measured samples. The last cascade's
U-Net term moves them after its DC update. The output then re-expands the coil-combined image x through the maps (S x)
before the RSS. Putting the measured coil k-space back at the sampled points, `K + λ M E (k0 − K)` with K = F(S x) E, is
free at inference. Does it raise PSNR / SSIM? Side readout, at the user's request: does it close the intensity gap of the
k-space-loss arm of `reports/loss_domain/report.md` (branch `exp/loss_domain`)? PSNR is in dB. "Error at the samples" is
‖M E (k0 − K)‖ / ‖M E k0‖ per slice.

## Runs compared

| Group | Checkpoints | Scored with | PSNR_Masked / SSIM_Masked at λ = 0 |
|---|---|---|---|
| 2x2 (RSS L1 loss) | `muon_layernorm_100ep` `control`, `muon`, `layer`, `muon_layer` × seeds 42, 43 (`reports/muon_layernorm_100ep/report.md`) | training snapshot `91932da` (`SeqMRI-worktrees/run-ml100-91932da`): main's loader, old brain flip | 33.28–33.86 / 0.855–0.863 |
| loss_domain `rss` (RSS L1) | `loss_domain_rss_r8_flair_knee_100ep_seed42` | `exp/loss_domain` (fixed brain flip) | 33.62 / 0.860 |
| loss_domain `kspace_gt` (L1 on coil k-space vs the measured data) | `loss_domain_kspace_gt_r8_flair_knee_100ep_seed42` | `exp/loss_domain` | 31.23 / 0.850 |

All of them are 12-cascade E2E VarNets with learned maps (`SensetivityModel_mc`, no body multiply), shared `loupe_2d` (VD start
p=4), R=8 of the acquired extent, 100 epochs, local v1 FLAIR + knee at 320². Knee-PDFS SSIM_Masked is about 0.78 in every
checkpoint, as in the source reports.

## Methods

1. **Step, post-forward** (`scripts/eval/final_dc_eval.py`). For each val batch, the model's own `recon_complex` x,
   `sense_maps` S and binary `mask` M feed `K = F(S x) E`, `K_λ = K + λ M E (k0 − K)` and `output = sqrt(Σ_c |F⁻¹ K_λ|² + 1e-12)`,
   for λ ∈ {0, 0.25, 0.5, 0.75, 1}. E is `_kspace_extent`. λ = 0 equals the model's output bitwise (max gap 0 in every run).
2. **Scoring.** All 1938 val slices, mask draws seeded 1234 and 4321 (global RNG), averaged per slice. PSNR_Masked, SSIM_Masked,
   NMSE and output/target mean are taken inside the apodized metric mask, with data range = the slice's max. Each checkpoint
   loads and draws exactly as its study's own eval does (`scripts/eval/varnet_train_speed_eval.py`, `eval_loss_domain.py`).
   `reports/final_dc/run.sh` runs it on GPUs 0–1.
3. **Statistics** (`scripts/eval/final_dc_summary.py`). Paired per slice against λ = 0. The 95% CI is ±1.96 SE over
   per-volume means.
4. **Post-freeze diagnostics.** Same forward pass; these explain the result and decide nothing.
   - The per-pixel common phase of S x against the measured coil images, φ = ∠Σ_c conj(c_c) S_c x. Reported as the
     tissue-weighted mean |φ| (weight = target × metric mask).
   - **Oracle alignment:** x e^{−iφ} before the step. φ comes from the fully sampled data, noise phase included, so it is an
     upper bound with leakage.
   - **Estimated alignment:** φ from the sampled data only, comparing M k0 and M K through one Gaussian low-pass
     (σ = 8 px). σ was fixed before the full run and not tuned.
   - **Maps floor:** the error at the samples of the fully sampled image passed through the maps.
   - **Error share:** the share of the coil k-space error inside E that sits at the sampled points.
   - **Gain+offset fit:** the oracle per-slice affine fit of `reports/loss_domain/diag/intensity_fit.py`.

## Results

### 1. Hard DC lowers PSNR on every RSS-trained checkpoint (λ = 1 − λ = 0)

| checkpoint | PSNR λ=0 | ΔPSNR all [95% CI] | ΔSSIM | Δ brain | Δ knee-PD | Δ knee-PDFS |
|---|---|---|---|---|---|---|
| control_s42 | 33.51 | −2.196 [−2.575, −1.816] | −0.0069 | −1.469 | −1.005 | −4.076 |
| control_s43 | 33.28 | −2.093 [−2.469, −1.717] | −0.0056 | −1.273 | −1.027 | −3.933 |
| muon_s42 | 33.64 | −2.140 [−2.515, −1.765] | −0.0073 | −1.530 | −0.883 | −3.979 |
| muon_s43 | 33.45 | −1.038 [−1.285, −0.792] | −0.0033 | −0.692 | −0.217 | −2.193 |
| layer_s42 | 33.53 | −2.198 [−2.576, −1.820] | −0.0067 | −1.517 | −0.973 | −4.071 |
| layer_s43 | 33.37 | −2.131 [−2.511, −1.752] | −0.0070 | −1.439 | −0.898 | −4.022 |
| muon_layer_s42 | 33.86 | −2.205 [−2.579, −1.831] | −0.0075 | −1.569 | −0.978 | −4.039 |
| muon_layer_s43 | 33.61 | −2.012 [−2.320, −1.704] | −0.0310 | −1.375 | −1.128 | −3.496 |

The loss is 1.0–2.2 dB, every per-checkpoint CI excludes zero, and it is largest in knee-PDFS.

### 2. No scalar gate helps (mean over the 8 checkpoints; ΔPSNR dB / ΔSSIM against λ = 0)

| condition | λ = 0.25 | λ = 0.5 | λ = 0.75 | λ = 1 |
|---|---|---|---|---|
| all | −0.512 / +0.0004 | −1.286 / −0.0027 | −1.832 / −0.0071 | −2.002 / −0.0094 |
| brain | −0.273 / +0.0008 | −0.780 / −0.0010 | −1.195 / −0.0036 | −1.358 / −0.0051 |
| knee-PD | −0.206 / −0.0000 | −0.568 / −0.0016 | −0.826 / −0.0034 | −0.889 / −0.0042 |
| knee-PDFS | −1.043 / +0.0006 | −2.482 / −0.0055 | −3.440 / −0.0140 | −3.726 / −0.0186 |

PSNR falls monotonically with λ in every condition, so neither a scalar gate nor a per-condition one has a positive setting.
A gate by k-space radius, which would leave the densely sampled centre to the network, was not tested.

### 3. Why: the output's phase and brightness differ from the data (post-freeze; means by condition: all / brain / knee-PD / knee-PDFS)

| readout | 2x2 (8 checkpoints, RSS L1) | loss_domain rss (RSS L1) | loss_domain kspace_gt |
|---|---|---|---|
| phase offset |φ| in tissue (rad) | 0.233 / 0.195 / 0.145 / 0.357 | 0.280 / 0.227 / 0.164 / 0.446 | 0.063 / 0.056 / 0.034 / 0.099 |
| error at the samples | 0.400 / 0.386 / 0.229 / 0.587 | 0.438 / 0.416 / 0.245 / 0.654 | 0.142 / 0.131 / 0.081 / 0.215 |
| maps floor | 0.173 / 0.156 / 0.120 / 0.243 | 0.173 / 0.159 / 0.118 / 0.242 | 0.147 / 0.136 / 0.083 / 0.221 |
| error at the samples, oracle-aligned | 0.208 / 0.185 / 0.133 / 0.305 | 0.209 / 0.188 / 0.132 / 0.306 | 0.148 / 0.138 / 0.084 / 0.223 |
| share of coil k-space error at the samples | 0.486 | 0.554 | 0.120 |
| Δ hard DC | −2.002 / −1.358 / −0.889 / −3.726 | −2.241 / −1.518 / −1.044 / −4.126 | **+0.582** / +0.397 / +0.397 / +0.940 |
| Δ oracle-aligned, no DC | +0.022 / +0.069 / +0.000 / +0.000 | +0.017 / +0.053 / +0.000 / +0.000 | +0.020 / +0.064 / −0.000 / −0.000 |
| Δ oracle-aligned + hard DC | −1.085 / −0.218 / −0.269 / −2.713 | −1.118 / −0.201 / −0.297 / −2.796 | +0.862 / +0.700 / +0.643 / +1.234 |
| Δ estimated-aligned + hard DC | −2.247 / −2.075 / −1.018 / −3.658 | −2.608 / −2.431 / −1.268 / −4.137 | +0.582 / +0.397 / +0.396 / +0.941 |
| output/target mean, λ = 0 → 1 | 0.995 → 0.944; PDFS 0.992 → 0.891 | 0.997 → 0.939; PDFS 0.994 → 0.882 | 0.930 → 0.940; PDFS 0.862 → 0.883 |
| Δ with oracle gain+offset fit, λ = 1 | −0.215 / −0.209 / −0.143 / −0.294 | −0.198 / −0.192 / −0.154 / −0.249 | +2.652 / +1.693 / +1.332 / +4.876 |

Δ are against the same run's raw λ = 0 PSNR. Reading:
- **Phase.** The RSS-trained models' coil images sit 0.15–0.45 rad from the measured data's phase. Their error at the samples
  (0.40) is more than twice the maps floor (0.17); after the oracle rotation it is near the floor (0.21). The offset is as large
  for rss on the fixed loader as on the old-flip 2x2, and it is present in knee, which was never flipped. So it goes with the
  RSS loss, not the brain flip. kspace_gt, same loader, k-space loss: 0.06 rad, error at the floor.
- **Brightness.** Raw hard DC darkens the output: knee-PDFS 0.99 → 0.88–0.92, 6 of 8 at 0.882–0.883, the same as kspace_gt
  at λ = 1. An oracle affine fit takes the raw λ = 1 loss from −2.0 to −0.2 dB. That raw darkening mixes two causes: phase
  cancellation and any overwrite of the network's brightness. With the phase aligned, hard DC still costs a near-constant
  −1.06 to −1.11 dB in all 8 checkpoints (`summary.json`, `mechanism_2x2_per_run`), while SSIM rises (+0.0014 to +0.0046).
  Brightness and the affine fit were not recorded for the aligned output. Hypothesis, untested: the aligned residual is
  mostly an intensity offset, not lost structure.
- **The estimated alignment** (sampled data only, σ = 8 px) is worse than none (−2.25 vs −2.00 dB). It rotates low-signal
  regions by a noisy phase.

### 4. Side readout: the step does not recover kspace_gt's intensity gap (seed 42)

| comparison | all | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|
| kspace_gt λ=1 − rss λ=0 | −1.811 [−2.243, −1.379] | −1.068 [−1.397, −0.739] | −0.387 [−0.846, +0.072] | −3.944 [−4.841, −3.046] |
| kspace_gt − rss, both λ=0 | −2.393 [−2.898, −1.887] | −1.465 [−1.871, −1.059] | −0.784 [−1.322, −0.245] | −4.884 [−5.901, −3.866] |
| kspace_gt − rss, both λ=1 | +0.431 [+0.360, +0.501] | +0.450 [+0.378, +0.522] | +0.657 [+0.475, +0.839] | +0.182 [+0.111, +0.253] |
| kspace_gt λ=1 − λ=0 | +0.582 [+0.503, +0.661] | +0.397 [+0.314, +0.479] | +0.397 [+0.301, +0.493] | +0.940 [+0.806, +1.075] |
| rss λ=1 − λ=0 | −2.241 [−2.620, −1.863] | −1.518 [−1.792, −1.244] | −1.044 [−1.338, −0.750] | −4.126 [−4.960, −3.291] |

Against the production comparison, the RSS control as scored today, the step recovers 0.58 of kspace_gt's 2.39 dB (24%).
Its knee-PDFS brightness moves from 0.862 to 0.883, against a target of 0.95. The same-λ comparison turns positive only because
rss falls 2.24 dB.

## Conclusions

- **Measured:** a hard final DC step at inference lowers PSNR_Masked by 1.0–2.2 dB on all 8 RSS-trained checkpoints (frozen
  verdict: failure). Every λ ≥ 0.25 lowers it too. Knee-PDFS loses the most (−3.7 dB).
- **Measured:** the RSS-trained models' coil images carry a per-pixel common phase 0.15–0.45 rad from the measured data. The
  offset does not depend on the brain flip, and it accounts for about half of the hard-DC loss: −2.0 dB raw, −1.1 dB after
  oracle alignment.
- **Measured:** raw hard DC darkens the RSS-trained outputs to the knee-PDFS brightness of the k-space-loss model at λ = 1
  (0.88), and an oracle affine fit recovers all but about 0.2 dB of the raw loss. With the phase aligned, a constant −1.1 dB
  remains while SSIM rises; its brightness was not recorded. Hypothesis, consistent with the frozen prediction but not tested
  directly (no aligned brightness readout, no noise-only control): an RSS-trained network reproduces the target's noise floor
  as low-frequency signal, DC overwrites it with measured data from the densely sampled centre, and that noise bias is the
  remaining loss.
- **Measured:** the k-space-loss model gains +0.58 dB from hard DC (+0.94 in knee-PDFS) but stays 1.81 dB below the RSS
  control. The step does not recover the contrast gap, because what it lacks is the target's noise floor, which data
  consistency cannot add.
- **Measured:** estimating the phase from the sampled data alone (one fixed estimator) makes the step worse, not better.

## Caveats

- **Inference only.** The networks were trained without the step and co-adapted to that. A model trained with the step in
  its forward pass could behave differently; this study does not test that.
- **Post-freeze additions.** The phase diagnostics, both alignments and the maps floor were added after a 2-batch smoke test
  showed the hard-DC loss. They decide no criterion. The oracle alignment uses the fully sampled data, including its noise
  phase, so its numbers are an upper bound with leakage. No output/target mean was recorded for the aligned variants.
- **The side-readout criterion (a)** was first coded as "gap ratio between 0 and 0.5", which rejects a sign flip. It was
  recoded after the results to follow the frozen wording ("the deficit shrinks to ≤ 50%"). That turns (a) from not met to met,
  but (b) still fails, so the side verdict is unchanged.
- **One seed** for loss_domain rss and kspace_gt. The 2x2 has two seeds per recipe, on the old brain flip.
- **Retrospective target.** The target is the RSS of the same noisy samples, so copying samples is exact against it. On
  prospective data the step would copy noise into the image instead.
- **Not investigated:** `muon_layer_s43`'s larger SSIM drop (−0.031 vs −0.003 to −0.008 elsewhere), and `muon_s43`'s smaller
  loss (−1.04 dB, error share 0.23 vs 0.43–0.59).
- Checkpoints are untracked (`manifest.json` lists them). The 2x2 runs predate `run_info.json`, so their training commit is
  null in the manifest; it is `91932da` (`reports/muon_layernorm_100ep/report.md`).

## Figures

None.

## Next steps

1. **No inference-time DC** for RSS-trained checkpoints. Keep today's output.
2. **Multi-step sampling (roadmap step 3).** The next step's sampler input is `F(S x) E` (`sequential_sampling.py`, the
   `pred_kspace` update), and this study measures it 40% off the measured data at the sampled points. Check what the sampler
   sees before trusting multi-step results. Hypothesis, untested: the phase drift matters there even though the RSS metric
   is blind to it.
3. **Training with the step in the forward pass** would be a new proposal, not stage 2. It removes the phase drift by
   construction. Prediction: it still costs PSNR in knee-PDFS against the RSS target, because the noise floor would then
   have to come from unsampled k-space only.
4. **The target.** kspace_gt and DC fail for the same reason: the RSS target carries a noise floor that consistency with the
   data cannot produce. Scoring and training against a noise-debiased target, for example an RSS with the noise floor removed
   using a background or noise-scan estimate, is the change that would let consistency-preserving reconstruction compete.
   It is a benchmark change, so it needs its own proposal.

## Reproduce

`bash reports/final_dc/run.sh` from the repo root. It needs:
- GPUs 0–1, about 1 h;
- the 10 checkpoints linked from `checkpoints/`;
- the local v1 raw data;
- the two code trees the checkpoints trained with: `SeqMRI-worktrees/run-ml100-91932da` and the `exp/loss_domain` worktree.

Scoring skips existing `scores/*.npz`; delete them to rescore. Check the folder with
`python scripts/report_manifest.py check reports/final_dc`.
