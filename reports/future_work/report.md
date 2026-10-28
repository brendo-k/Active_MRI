# Future Work: Improving Adaptive MRI Sampling (2026-10-26)

| | |
|---|---|
| Code | results produced at `a84b2ba` (Skip the zero-padded coils in the sensitivity net's U-Net); branch `main` |
| Metric mask | **apodized**: k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20 |
| Runs | 4 checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |
| Results | 3 files next to this report, hashed in `manifest.json` |
| Reproduce | `bash run.sh` from the repo root |
| Working tree | 17 tracked files were modified when the manifest was written |
| Training commit | not recorded for these runs (they predate `run_info.json`) |

**Document type: research synthesis and experiment proposal.** The provenance block identifies the repository state when this report was assembled, not the training commit of historical models. The JSON files here are copies of existing evaluations, not new reconstruction results. No training or GPU experiment was launched for this report. Historical metrics retain their original definitions; all proposed performance gains are hypotheses or decision thresholds.

## Question

Can adaptive sampling improve SSIM, PSNR, and NMSE beyond a single population-level LOUPE mask, and beyond the current anatomy-conditioned alternative? Which experiments would distinguish better mask optimization from useful patient-specific adaptation, and could sequential sampling or reinforcement learning produce a 1–2 dB PSNR improvement?

The immediate implementation is deliberately simple: **a CNN takes the ACS region and outputs a mask-bank index**. Richer routing, encoding features, and sequential acquisition are subsequent experiments. R=8 is the primary comparison; other acceleration factors require their own matched baselines. PSNR is reported in dB; SSIM and NMSE are unitless.

## Runs compared

This report summarizes prior comparisons and defines future arms; it does not rerank checkpoints through a new evaluation.

| Existing evidence | Models or artifacts | Role |
|---|---|---|
| Population versus anatomy LOUPE | `checkpoints/loupe2d_varnet_r8_brain_knee_shared_100ep_lr1e-3`, `checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep` | Historical seed-42 VarNet comparison |
| Second training seed | `checkpoints/loupe2d_varnet_r8_brain_knee_shared_100ep_lr1e-3_seed43`, `checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep_lr1e-3_seed43` | Historical seed-43 comparison |
| Initialization controls | [Existing initialization summary](../ismrm2027/vdinit_summary.json), copied locally as [vdinit_summary.json](vdinit_summary.json) | Variable-density and common-start results; inspect before duplicating runs |
| Adaptivity diagnostics | [Mechanisms](../ADAPTIVITY_MECHANISMS_R8.md), [subject co-adaptation](../SUBJECT_COADAPTATION_R8.md), [CG-SENSE ACS gate](../ACS_MASK_GATE_CGSENSE_R8.md) | Transferability, optimization, and encoding clues |
| Simple router | [cnn_router.py](../../activemri/samplers/cnn_router.py), [usage notes](../../docs/ACS_ROUTER.md) | Implemented; no performance result established here |

The historical anatomy pairs use R=8, a 12-cascade VarNet, 100 training epochs, and balanced brain/knee data. The earlier [metric-mask re-evaluation](../metric_mask_reeval_r8/report.md) supplies absolute metrics and provenance for its evaluated checkpoints. Its aggregate scores must not be directly substituted for the paired per-slice statistics below.

## Methods

### 1. Immediate experiment: the simple ACS CNN router

The implemented path is:

```text
ACS complex data
  → mean-centered log magnitude
  → 3×3 Conv + ReLU
  → 3×3 Conv + ReLU
  → global average pooling
  → linear layer with K outputs
  → mask index
```

Start with **K=2 masks, 32 CNN channels, and the existing 10×10 ACS**. The router uses no anatomy labels, coil metadata, uncertainty estimator, or reconstruction encoder. Index 0/1 has no predefined anatomical meaning.

During training, hard Gumbel-softmax selects one learned LOUPE probability map in the forward pass and supplies a soft surrogate gradient to the CNN. The mask bank and router train through the existing reconstruction loss. At evaluation the router uses argmax. The selected map still undergoes the existing budget rescaling and Bernoulli sampling: the routing index is deterministic, but the binary mask is a seeded draw and its count is controlled in expectation.

The implementation is registered as `cnn_router_2d` in [sampler.py](../../activemri/samplers/sampler.py). `ACSRouter.forward` returns indices from normalized patches; `CNNRoutedMaskBank.indices` extracts the ACS from an NHWC observation and returns deterministic indices. `logits` exposes scores for training. Current bank entries start from independent LOUPE random initializations; the implementation does not automatically import trained anatomy masks.

Proposed controls:

| Arm | Purpose |
|---|---|
| Population LOUPE | Primary baseline |
| Anatomy-conditioned LOUPE | Current routing comparator |
| ACS CNN, K=2 | Learn mask choice directly from measurements |
| Same CNN/bank with a constant input | Separate routing benefit from architecture and bank training |
| Best fixed bank member | Determine whether a learned expert is simply a better population mask |
| Shuffled routing indices within anatomy | Test whether within-anatomy choices matter |
| Ground-truth best-of-bank selection | Diagnose available headroom for this bank and reconstructor |

Log expert usage, routing entropy, mask similarity, per-volume gains, and performance of each fixed expert. Do not force equal expert usage: a dominant expert may be appropriate. A collapsed router with a large best-of-bank gap suggests an optimization or predictability problem; collapse with no gap suggests little reason to route this bank.

Keep training exposure and reconstruction initialization matched. Additional mask parameters consume random numbers, so a common seed alone does not guarantee identical reconstructor initialization or data order. Use separate RNG streams or explicitly copy initial states. The present router has one decision per supplied example; volume aggregation is a separate implementation requirement for SPACE.

### 2. Better discrete optimization: LOUPE, BASS, and gradient-proposed swaps

**Hypothesis, untested:** straight-through gradients may misrank useful binary-mask changes.

Starting from the same good mask and a fixed reconstructor, compare continued LOUPE training, BASS refinement, and a hybrid that proposes swaps using gradients and accepts them only after measuring actual reconstruction loss. Match reconstruction-evaluation budgets and report wall time. The hybrid is BASS-inspired, not a reproduction of BASS.

For removal i and addition j, compare the gradient prediction `g[j] - g[i]` with the measured loss change. Include random and spatially stratified proposals. Keep ACS and sample count fixed, and use separate training batches for proposal and acceptance checks.

Run population and subject-specific searches with equal effort. If both improve similarly, pursue better population optimization. Extra subject-specific improvement must survive held-out evaluation before it motivates a router. Full-reference optimization on a test subject is a privileged diagnostic, not a deployable policy or a certified global upper bound.

[BASS](https://www.nature.com/articles/s41598-021-97995-w) is a discrete sampling optimizer, not inherently a patient-adaptive method. [Alternating BASS/VarNet learning](https://arxiv.org/abs/2110.14703) provides a precedent for combining discrete acquisition optimization with gradient-trained reconstruction. An offline BASS teacher can also support a gradient-trained router, but that pipeline is hybrid rather than fully gradient-based.

### 3. Optional router upgrades after the simple experiment

**Encoding-conditioned routing.** Add header coil count first, then ACS-derived sensitivity geometry. The CG-SENSE gate identified a few-coil subgroup with greater transferable mask gains, although coil count and scanner model were confounded. Compare anatomy, coil-class, and encoding-conditioned arms with shuffled-feature controls. Spatial sensitivity overlap matters beyond total coil energy or global effective rank. The [spectral-moment sampling framework](https://arxiv.org/abs/1710.07837) provides a physical basis for studying encoding-dependent geometry.

Before claiming an encoding benefit, check coil truncation versus consistent calibration-derived coil compression. Improving the input pipeline is not adaptive-mask gain; apply changes to every arm.

**Residual LOUPE.** A later compact alternative is a population logit map plus a small conditioned residual, with shrinkage toward the baseline. This should add physical information or structured geometry; simply repeating an ACS-to-smooth-density head would duplicate earlier approximately null experiments.

**Benefit-supervised routing.** With a fixed bank and common reconstructor, evaluate each training volume under every mask and train the router to predict `loss(mask_j) - loss(population)`. This distinguishes consequential choices from near ties. Generate labels with subject-disjoint cross-fitting and repeated mask draws where necessary. Offline fully sampled training data permits direct supervision without RL.

For subsequent joint training, minimize `sum_j q_j(observation) * loss_j`, where each loss comes from the actual candidate acquisition. This is distinct from averaging probability maps and drawing a mask from that mixture. Alternate expert, router, and reconstructor updates; score hard selection at inference.

[SUNO](https://arxiv.org/html/2501.09799v4) combines alternating per-example mask optimization with low-frequency nearest-neighbor retrieval. A learned benefit router is an alternative to its retrieval step. Its knee E2E-VarNet R=8 table reports BASS at 29.99 dB and SUNO at 29.57 dB, so SUNO does not establish a guaranteed adaptive advantage. The earlier in-house oracle-medoid bank also failed to provide useful headroom. Expand a bank only after demonstrating that its best-of-bank selection can outperform the baseline.

### 4. Sequential sampling: acquire information beyond the ACS

**Hypothesis, untested:** newly acquired mid/high-frequency measurements reveal which remaining samples are useful, even when the ACS alone does not.

Start with ACS followed by four acquisition stages sharing the remaining budget. Give the policy all acquired measurements, the current mask, reconstruction, and remaining budget. Compare an input-independent staged policy, an ACS-only staged policy, and a policy whose observations update after each stage. Also retain population and anatomy LOUPE. Match total acquisition and reconstruction/training resources.

Train a differentiable policy against final reconstruction quality, starting near a good LOUPE solution. Train the reconstructor at intermediate sampling fractions. The updated-observation arm must beat the ACS-only and input-independent twins to establish feedback benefit.

A simple extension of the router lets each expert propose the next fixed-size set of unacquired locations, with one index selected per stage. This makes the decision space small. It is not implemented in `cnn_router_2d`; the current module continues to read the ACS only.

Before expensive policy training, optimize privileged continuations from a fixed initial measurement block on development subjects, with multiple starts and the baseline continuation included. A consistently large improvement motivates learning; small achieved improvement argues for caution but does not prove no better policy exists.

[End-to-end SeqMRI](https://arxiv.org/html/2105.06460v2) reports R=8 SSIM of 0.9060 for LOUPE versus 0.9107 for four-step sampling. That supports the hypothesis, but its single-coil, cropped 128×128 U-Net setting does not establish a 1–2 dB gain over the present multicoil VarNet. SSIM changes cannot be converted directly into PSNR.

### 5. RL and the role of larger GPUs

Use PPO as a comparison after the differentiable sequential test, ideally warm-started from a useful policy. Freeze the reconstructor initially, then alternate policy and reconstruction updates. Start with 4–8 decisions per episode, acquiring a batch of locations at each decision, rather than a long series of individual choices.

Per-stage error reduction with an undiscounted return telescopes to final error improvement. Intermediate rewards provide feedback but do not eliminate credit-assignment difficulty. A router choosing among a few acquisition strategies is a smaller RL problem than a free policy over every k-space location.

The existing [PPO report](../PPO_SAMPLER.md) documents a smoke-tested implementation and a long-horizon setup, not a demonstrated performance improvement. Earlier [REINFORCE diagnostics](../LOUPE_VS_REINFORCE_R8.md) identified weak per-location gradient signal and reduced unique-slice exposure when multiple draws occupied the same batch. Spend additional GPU memory on multiple rollouts while preserving distinct-subject exposure, intermediate reconstructions, and matched controls/seeds. Do not assume a larger policy alone solves either problem.

Published RL results are mixed: [Pineda et al.](https://arxiv.org/html/2007.10469v2) found the dataset-specific policy slightly better than subject-specific policies on most metrics; [Bakker et al.](https://arxiv.org/html/2010.16262v2) identified poorer gradient signal and less adaptation with longer-horizon optimization. More computation makes a stronger experiment possible, not a large gain certain.

### 6. Sampling geometry and objective alignment

Compare fixed/adaptive density crossed with fixed/adaptive geometry. Use a strong variable-density Poisson-disc reference and test anisotropic spacing or encoding-weighted penalties. LOUPE probability maps already learn some spatial spacing; the question is whether explicit scan-dependent geometry adds value. Validate any relaxed selection gradient using actual binary acquisitions. The unstable DCM implementation and the radial REINFORCE top-K failure are warnings against changing the selection rule only at evaluation.

Compare whole-image L1 with per-example normalized squared error and normalized error plus SSIM. The current evaluation support differs from the default training loss, and batch-normalized NMSE is not the same as averaging per-example NMSE. Fix support from the reference and add full-image or standard-crop secondary metrics. Apply objective changes to both population and adaptive models so general reconstruction improvement is not attributed to routing.

### 7. Evaluation and acquisition constraints

Use subject-disjoint training, tuning, and final test sets. Treat repeatedly inspected validation data as development data. Report paired volume-level changes and confidence intervals, seed-specific results, and at least three seeds for finalists. Retain trainer aggregates separately from subject-weighted metrics; keep crop, support, range, reference, precision, and aggregation consistent.

Count all calibration samples. In particular, the historical `ecalib_c10` arm used maps derived from a 24×24 centre; it is a privileged-map diagnostic unless the extra calibration acquisition is accounted for. The simple router currently uses Bernoulli masks: report realized counts and use matched draws, or introduce the same validated exact-count rule for all arms.

Hold CG iterations, regularization, and sensitivity estimation fixed after tuning them. The [CG-SENSE sweep](../CGSENSE_HYPERPARAMS_R8.md) found strong early-stopping effects. For VarNet, separate mask substitution under a common reconstructor from complete-system comparisons with matched dedicated training.

For SPACE, verify the ACS-first assumption. The deployed ky–kz pattern is shared across the volume; per-slice fastMRI point masks are a proxy. Sequential decisions must preserve or model echo-dependent signal weighting, feasible acquisition blocks, and feedback latency. Retrospective static k-space simulation is an algorithmic screen, not a scanner validation. Adaptive acceleration, additional pilot measurements, and dynamic imaging are separate studies requiring matched acquisition-time controls.

## Results

These are previously recorded findings and implementation status, not newly measured performance.

| Historical comparison, anatomy minus population | Seed 42 | Seed 43 |
|---|---:|---:|
| Knee mask substitution under the shared VarNet | +0.040 dB | +0.090 dB |
| Knee complete-model comparison | +0.131 dB | +0.183 dB |
| Brain complete-model comparison | −0.046 dB | +0.128 dB |

Sources: [anatomy_pair_seed42.json](anatomy_pair_seed42.json) and [anatomy_pair_seed43.json](anatomy_pair_seed43.json), copied from `reports/ismrm2027/`; the original evaluator is `scripts/eval/eval_anatomy_pair_ismrm_r8.py`. These are paired per-slice PSNR summaries under apodized support. Their volume-bootstrap intervals are in the JSONs and do not incorporate between-training-seed uncertainty; for example, the seed-43 knee whole-model interval crosses zero. The table does not establish corresponding NMSE gains.

The newer copied initialization summary reports mixed outcomes, including a mixed verdict for same-start initialization. It is already available, so the earlier suggestion to wait for the VD-start runs is superseded. Its different code/precision setup must be kept separate from the historical table. No checkpoints were reloaded to audit that summary for this report.

The mechanisms and subject-coadaptation reports find large in-sample oracle opportunities with poor transfer to other slices. A failed predictor that also fails to fit training labels, even with full k-space, is not an information-theoretic bound. It may reflect representation, optimization, or nonunique mask targets. These findings motivate a small controlled router test and do not rule out sequential feedback.

| Component | Status |
|---|---|
| ACS CNN → index, hard selection, sampler integration | Implemented in the working tree |
| Router and related regression checks | Previous implementation session recorded 14 passing tests; not rerun for this documentation task |
| Router reconstruction-quality gain | Not established |
| Coil-aware or benefit-supervised router | Proposed |
| Updated-observation sequential router and new PPO comparison | Proposed |

Tests in [test_cnn_router.py](../../tests/test_cnn_router.py) cover ACS responsiveness, exclusion of outside-ACS data and labels, selection of a single bank member, finite gradients to the CNN/bank, and preservation of ACS and expected budget. These verify mechanics, not generalization or performance.

## Conclusions

- The immediate next experiment is the small ACS CNN router with two masks. More complex encoders are optional follow-ups.
- A better trained model or mask bank is not sufficient evidence of adaptation. Require improvement over matched static and anatomy controls, and test whether patient-specific choices matter.
- BASS can diagnose discrete-optimization limits or provide offline supervision; SUNO suggests a dictionary approach but does not guarantee superiority over population masks.
- Sequential acquisition is a distinct hypothesis because it observes new measurements. Its potential is not settled by ACS-only failures.
- **Planning judgment, not a forecast:** a sub-dB average sequential gain is more defensible than assuming 1–2 dB over the strong R=8 VarNet baseline. One dB is an ambitious target; two dB requires compelling headroom evidence. With fixed range and reference, these correspond to approximately 21% and 37% less squared error, respectively (`1 - 10**(-delta_dB/10)`). Mean PSNR gains need not imply those reductions in arithmetic mean NMSE.
- A proposed near-term success criterion is +0.3 dB over population LOUPE, additional benefit over anatomy LOUPE, lower NMSE, and no material SSIM regression. Fix the SSIM tolerance before evaluation; this criterion is not an expected result.

## Caveats

- Earlier flat reports mix legacy and newer metric conventions. The [metric-mask re-evaluation](../metric_mask_reeval_r8/report.md) explains why absolute values cannot be pooled.
- Small subgroup findings, scanner/coil confounding, repeated validation use, and training-seed variance limit current conclusions.
- The current simple router uses ACS magnitude only, learns a bank from random starts, and selects per example. It does not implement the richer proposed router, trained-mask import, volume pooling, or sequential feedback.
- Gumbel-softmax supplies a biased gradient surrogate; stochastic training and deterministic argmax evaluation must both be monitored.
- The report's HEAD is an assembly reference. Router files and some source evaluations were untracked, and other project files were modified when it was written. It is not a reproducible training commit.
- Historical source JSONs are copied for traceability. Reproducing their reconstructions requires original checkpoints, data, and evaluator settings. The report's `run.sh` does not perform that recomputation.

## Figures

None. The architecture is shown as a text diagram; comparison and status tables are above.

## Next steps

1. Train the two-mask ACS CNN with population/anatomy baselines and an input-independent control; log all three reconstruction metrics and routing behavior.
2. Evaluate best-of-bank headroom and shuffled routing. If a useful bank exists but routing fails, try benefit supervision; if the bank offers no gain, inspect discrete optimization and encoding-specific experts.
3. Run the four-stage updated-observation experiment with matched ACS-only and static staged twins. Add PPO only after assessing headroom and training stability.
4. Validate the winning hypothesis on true 3D data with a volume-level acquisition decision and an appropriate SPACE echo-train model.

## Reproduce

`bash reports/future_work/run.sh` refreshes the three historical JSON snapshots and checks the report manifest. It requires the original JSON files and the project Python environment, but no MRI dataset, GPU, or reconstruction run. It overwrites the snapshots; if their sources have changed, the hash check will flag the drift. Review the changed evidence before regenerating the manifest.

Report conformance:

```bash
rtk proxy .venv/bin/python scripts/report_manifest.py check reports/future_work
```

Optional implementation checks, separate from report reproduction:

```bash
rtk proxy .venv/bin/python -m pytest -q \
  tests/test_cnn_router.py tests/test_mask_bank_cg.py tests/test_smooth_adapt.py
```

Proposed training command, **not run for this report**:

```bash
rtk proxy .venv/bin/python train.py --model sequential --recon varnet \
  --dataset brain_knee_balanced --acceleration 8 \
  sampler.sampler_type=cnn_router_2d sampler.bank_size=2 \
  sampler.adapt_hidden=32 sampler.anatomy_conditioning=false \
  --out-dir checkpoints/cnn_router_r8
```

Further design detail: [adaptive-mask experiment plan](../../docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md), [simple-router usage](../../docs/ACS_ROUTER.md), and [prior literature synthesis](../LITERATURE_ADAPTIVE_SAMPLING_R8.md). This report takes precedence over the earlier plan's priority order: the simple CNN router is the first experiment.
