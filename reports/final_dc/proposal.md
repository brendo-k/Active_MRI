# Proposal: final_dc, stage 1 (inference only)

**Status:** frozen 2026-11-09. The user approved it in chat ("approved, run stage 1 with your recommendations. Also try it with
the k-space version. This could recover the poor contrast differences."). The k-space-loss side readout below was added at that
request. Stage 2 (training with the step) is not part of this proposal.

- **Topic / branch:** `final_dc`, branch `exp/final_dc`, report `reports/final_dc/`.
- **Roadmap:** every roadmap comparison (anatomy routing, ACS budget) sits on 1-step LOUPE + 12-cascade E2E VarNet at R=8. If a
  measurable part of its scored error sits at the sampled k-space locations, a final data-consistency (DC) step is a gain for all
  arms, and, trained in, a direct mask gradient for LOUPE (stage 2).

## Question

The scored output loses data consistency twice:
1. The last cascade computes `k - eta * M * (k - k0) - U-Net term` from the same `k`. In the 2x2 checkpoints the last cascade's
   eta is 0.83-1.00, but the U-Net term moves the sampled points in the same update.
2. The VarNet returns x = sum_c conj(S_c) F^-1 k_c, and the output re-expands S x before the RSS (`_coil_rss`). That projects every
   coil onto the maps' rank-1 model, so even consistent coil k-space loses consistency. The loss_domain rss control misses the
   measured samples by about 40% (relative, `reports/loss_domain` on `exp/loss_domain`).

How much of the scored error sits at the sampled locations, and does putting the measured samples back raise PSNR / SSIM?

## The step

Between the re-expansion and the RSS (`sequential_sampling.py`, the `_coil_rss` call in `forward`), applied after the forward
pass to the model's own `recon_complex`, `sense_maps` and `mask`:

```
K   = F(S x) * E                     # E = acquired extent (_kspace_extent), as in _coil_rss
K_l = K + lambda * M * E * (k0 - K)  # k0 = measured coil k-space; lambda = 1 is hard replace
out = sqrt(sum_c |F^-1 K_l|^2 + RSS_EPSILON)
```

lambda = 0 is today's output exactly. No weights change and nothing is trained.

## Checkpoints and scoring

- **Main set:** the 8 `muon_layernorm_100ep` checkpoints (`control`, `muon`, `layer`, `muon_layer` x seeds 42, 43;
  `~/data/seqmri_checkpoints/muon_layernorm_100ep/<arm>_s<seed>/last_model.pt`). Scored with their training code, the detached
  snapshot `run-ml100-91932da` (main's loader, old brain flip), and its loader built from `control_s42/.hydra/config.yaml`,
  as `scripts/eval/varnet_train_speed_eval.py` did for that report.
- **Side readout (user request):** loss_domain round 2 `rss` and `kspace_gt`, seed 42
  (`~/data/seqmri_checkpoints/loss_domain/runs_v2/`), scored with the `exp/loss_domain` code (fixed brain flip) and the loader of
  `scripts/eval/eval_loss_domain.py` there. Question: does the step close kspace_gt's intensity gap to rss?
- **Metric:** all 1938 val slices, mask draws seeded 1234 and 4321 (global RNG seeded before each pass), averaged per slice;
  PSNR_Masked, SSIM_Masked and NMSE inside the apodized per-anatomy metric mask, data range = the slice's max(target), by condition
  (brain, knee-PD, knee-PDFS) and pooled.
- **Control:** lambda = 0. It must reproduce each checkpoint's published eval to within 0.003 dB (2x2: per slice against
  `reports/muon_layernorm_100ep/eval.per_slice.npz`; loss_domain: the means in `eval_v2/summary.json`), or the run is debugged first.
- **Treatment:** lambda = 1 (hard replace). lambda in {0.25, 0.5, 0.75} is reported as a curve only. Choosing lambda on val would be
  selection, so no lambda other than 1 can pass.
- **Readouts:** per slice and lambda: PSNR, SSIM, NMSE, output/target mean inside the metric mask. At lambda 0 and 1: the oracle
  gain+offset PSNR (as `reports/loss_domain/diag/intensity_fit.py`). At lambda 0: the relative error of K at the sampled points,
  the same error for the fully sampled image passed through the maps (the maps' floor), and the share of the coil k-space error
  energy inside E that sits at the sampled points.

## Noise floor

Each comparison is the same checkpoint at lambda 1 vs lambda 0, paired per slice, so there is no training noise. CIs are 95%
over per-volume means. Sign consistency across the 8 checkpoints (4 recipes x 2 seeds) is the test.

## Expected outcomes

- **Brain and knee-PD (higher SNR): PSNR up.** The network's error at the sampled points is replaced by the exact values, and the
  target is built from the same samples. Guess: +0.1 to +0.5 dB.
- **Knee-PDFS (lowest SNR): may go down.** An RSS-trained network matches the target's noise floor by inflating |x|, a smooth,
  low-frequency component of its coil k-space. Hard DC overwrites the densely sampled centre with measured data, which carries no
  such floor, and adds back only about 1/R = 1/8 of the noise power. The output would darken (output/target mean below today's
  ~0.997), in the direction of kspace_gt's deficit.
- **SSIM:** either sign. The copied high-k samples carry noise the network had smoothed away.
- **Error share:** the share of coil k-space error at the sampled points bounds what lambda = 1 can remove in the coil images.
- **Side readout:** kspace_gt brightens only a little (about 1/8 of the noise-floor power returns: knee-PDFS mean ratio ~0.87 ->
  ~0.89) and does **not** recover the gap by itself. The gap to rss under the same lambda shrinks mostly because rss darkens.
- **If PDFS falls at lambda = 1 while an intermediate lambda helps**, that is the case for a per-condition or per-k-radius gate in
  stage 2, not a pass here.

## Success criteria (stage 1 -> propose stage 2)

All of:
1. lambda = 1 improves PSNR_Masked: mean over the 8 checkpoints >= +0.10 dB, and positive in all 8.
2. SSIM guard: delta SSIM_Masked >= -0.002 in both `muon_layer` seeds.
3. Condition guard: no condition worse by >= 0.10 dB in both seeds of any recipe.

**Failure:** mean gain < +0.05 dB, or either `muon_layer` seed negative. Anything else is **inconclusive**: report it, train nothing.

**Side readout (descriptive, one seed, not a gate):** the step "recovers the contrast gap" if both hold at lambda = 1:
(a) kspace_gt's PSNR deficit to rss (same lambda) shrinks to <= 50% of its lambda = 0 size (-2.39 dB), and (b) kspace_gt's
knee-PDFS output/target mean is >= 0.95. Also reported: kspace_gt at lambda = 1 against rss at lambda = 0.

## Compute

One forward per checkpoint serves every lambda. 10 checkpoints on GPUs 0-1 (two processes, each loader built once), about
10-15 min per checkpoint: about 1 h. Smoke test first on 2 batches. Crashed scoring jobs are rerun in full.
