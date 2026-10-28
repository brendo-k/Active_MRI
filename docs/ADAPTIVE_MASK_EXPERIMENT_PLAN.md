# Experiments to improve adaptive sampling over population LOUPE

2026-10-24. Research proposal based on existing reports, current source, and primary literature. No new training or reconstruction experiments were run for this proposal. Historical results below are evidence for prioritization, not a newly standardized benchmark.

Assumption: R=8 remains the main endpoint, with one mask per volume selected from the initial ACS and available scan metadata. For SPACE, all readouts share the ky–kz mask. A different R or additional pilot data must have a separate, acquisition-matched control.

The most promising direction is **conditioning sampling geometry on the scan's encoding capability**, coupled with an audit of the discrete optimization. A larger ACS-to-pixel CNN is a low-priority investment. SUNO and BASS are useful methodological controls, but neither establishes that patient adaptation must outperform a strong static mask.

1. **What the existing work establishes—and leaves open.**

   - [The adaptivity mechanisms study](../reports/ADAPTIVITY_MECHANISMS_R8.md) finds substantial per-image oracle improvement, but most of it is fine-scale and does not transfer to adjacent slices. [Subject co-adaptation](../reports/SUBJECT_COADAPTATION_R8.md) similarly distinguishes in-sample mask fitting from improvement on held-out slices. Neither is an achievable clinical ceiling.
   - A failed predictor that also fails to fit its training labels—even when given full k-space—does **not** establish that the information is absent. Nonunique good masks also make pixelwise mask regression a questionable target. These results argue against the tested representation and training procedure; they do not prove an information-theoretic impossibility.
   - [The ACS/CG-SENSE study](../reports/ACS_MASK_GATE_CGSENSE_R8.md) identifies a small few-coil brain subgroup with stronger transferable gains. Its coil-count split was post hoc, scanner model is confounded, and the larger gains cannot be projected to the full dataset.
   - [The later initialization investigation](../reports/CGSENSE_ANATOMY_MASKS_R8.md) qualifies the earlier protocol-routing failure: the class masks started from different draws and received fewer examples per update. The current [VD initialization launcher](../scripts/launch/train_loupe2d_varnet_r8_vdinit_anatomy.sh) already proposes matched population/anatomy starts. Complete and inspect that experiment before duplicating it; the launcher is not evidence of completed results.
   - The newer [seed 42](../reports/ismrm2027/anatomy_pair_seed42.json) and [seed 43](../reports/ismrm2027/anatomy_pair_seed43.json) evaluations separate mask and reconstructor changes. Knee mask replacement under the shared reconstructor gives about +0.040/+0.090 dB, while whole-run differences are +0.131/+0.183 dB. Brain whole-run differences change sign. These are per-slice PSNR summaries, not the trainer's aggregate PSNR; the files do not establish NMSE improvement.
   - [Metric-mask re-evaluation](../reports/metric_mask_reeval_r8/report.md) shows that old and new absolute metrics cannot be pooled. Historical confidence intervals generally do not include training-seed variance. Re-score retained candidates together.

2. **First experiment: determine whether the optimizer leaves usable improvement behind.**

   Hypothesis: the straight-through gradient ranks discrete mask changes poorly enough to limit both population and adaptive masks. BASS is useful here because it evaluates changes to the actual acquisition set.

   Start from the same trained population and anatomy masks. Freeze CG-SENSE settings/sensitivity estimation, or a common VarNet. On training volumes, compare: continued LOUPE optimization; a BASS refinement; and a hybrid that uses gradients to propose swaps and accepts them only after measuring the actual reconstruction loss. The hybrid is BASS-inspired, not a faithful BASS reproduction. Match reconstruction-evaluation budgets and report runtime separately.

   For a proposed removal i and addition j, compare the surrogate prediction g[j]−g[i] with the measured loss change of the binary mask. Include random and spatially stratified proposals, not just the proposals the gradient favors. Report rank correlation, sign accuracy, and actual accepted improvement across different starting masks. Use multiple slices per training volume, fresh training batches for acceptance checks, and fixed ACS/sample count.

   Run population and per-volume searches with identical effort. Include the unchanged starting mask among the candidates. Score searches on separate data/draws; test-volume searches that access complete data are diagnostic oracles only. If population refinement captures essentially the same improvement as per-volume refinement, pursue better population optimization rather than a patient router.

   BASS itself is a discrete optimizer and is not inherently patient-adaptive. Its original formulation and its alternating extension provide the relevant controls: [BASS, Zibetti et al. 2021](https://www.nature.com/articles/s41598-021-97995-w), [alternating BASS/VarNet learning, Zibetti et al. 2022](https://arxiv.org/abs/2110.14703).

3. **Highest-priority adaptive experiment: coil-aware LOUPE with shared residuals.**

   Hypothesis: scans require different sampling because their spatial encoding differs, even when their low-resolution anatomical appearance is similar.

   Compare these progressively more informative conditions:

   | Arm | Condition supplied to sampler | Purpose |
   |---|---|---|
   | Population | Constant | User's baseline |
   | Anatomy | Brain/knee label | Current adaptive benchmark |
   | Coil class | Anatomy and raw coil-count class | Cheap test of the observed subgroup effect |
   | Encoding features | Anatomy plus ACS-derived sensitivity geometry and noise features | Test whether physical encoding adds more than the header |
   | Shuffled encoding | Encoding features permuted within anatomy/coil class | Isolate patient-specific benefit |

   Use shared logits plus small conditioned residuals, initialized at zero:

   `z_v(k) = z_population(k) + delta_anatomy(k) + sum_j a_j(features_v) B_j(k)`.

   Begin with 4–8 coefficients controlling anisotropy, density, and a few structured spatial components. A header lookup and a small MLP can use the same head. Use shrinkage toward the population mask and balanced subject-level sampling to protect small classes. All class rows start from the same mask; separate RNG streams keep reconstruction initialization and batch order identical between arms.

   Features should include spatial coil overlap/encoding measures, not only total coil energy. Compare header count with covariance/effective-rank summaries and sensitivity-correlation features. Use representations robust to coil permutation and basis changes; a global effective rank alone cannot describe spatial alias separation. Normalize sensitivity maps consistently and measure sensitivity to calibration error.

   The existing smooth-head experiment was approximately null. This proposal therefore needs the **new physical input** and a geometry-aware objective or mask family; simply repeating a smooth head with an anatomical image is not the experiment.

   A supporting physical approach is to regularize the mask with an encoding-conditioning criterion derived from the sensitivity operator, then retain image-quality loss as the final objective. [Levine and Hargreaves](https://arxiv.org/abs/1710.07837) relate sampling geometry to spectral moments of the information matrix. A gradient-trained conditional extension is a proposal here, not their published result. Any continuous relaxation is only a search surrogate; measure the resulting binary acquisition and its reconstruction.

   Before attributing a benefit to coil conditioning, check the loader's truncation of scans with more than 16 coils. Compare with a consistent, energy-preserving coil-compression pipeline, using a basis estimated from available calibration data. Update every comparison arm identically. Improvement from retaining coil information is a reconstruction/input improvement, not adaptive-mask gain.

   Initial decision: advance if the conditioned arm beats anatomy LOUPE on previously unseen few-coil volumes with consistent seed direction and no material average regression. Require additional volumes before making a scanner-independent claim.

4. **Gradient-trained SUNO alternative: predict reconstruction benefit, not oracle pixels.**

   SUNO alternates reconstruction training and per-example discrete mask optimization, then retrieves a mask using low-frequency image similarity. Its knee E2E-VarNet R=8 table favors population BASS over SUNO (29.99 versus 29.57 dB). This supports using SUNO as a comparator, not assuming a large adaptive gain. [SUNO, methods and Table II](https://arxiv.org/html/2501.09799v4).

   Build a small bank of 2–4 exact-budget masks per anatomy, including the anatomy baseline. Train the experts on groups of training volumes, with shared initialization and shrinkage. Use coil geometry and ACS features to predict each mask's reconstruction loss or improvement relative to the baseline. At deployment choose one mask for the whole volume.

   A gradient-trained gate can minimize `mean_v sum_j q_j(features_v) L(recon(mask_j, data_v), target_v)`. This averages losses from actual binary acquisitions, not measurements from a physically unrealizable average mask. Gates receive ordinary gradients; mask experts use a validated discrete-gradient surrogate. Score hard expert selection at validation and deployment.

   Alternate mask, gate, and reconstructor updates. Keep mask-training diversity the same for all routing controls: best fixed expert; anatomy choice; nearest neighbor; learned benefit router; shuffled router; and diagnostic best-of-bank selection using ground truth. Generate benefit labels with subject-disjoint cross-fitting to reduce training-subject memorization.

   The crucial cheap test is best-of-bank headroom. The prior [oracle-medoid test](../reports/LITERATURE_ADAPTIVE_SAMPLING_R8.md) already failed. Revisit a bank only if training experts across multiple volumes, encoding-based conditions, and matched reconstruction training produce positive headroom. Reject a bank whose perfect selector cannot beat anatomy LOUPE by the desired margin. A best-of-bank result is an upper bound for that bank and reconstructor only.

   If BASS/ICD produces useful training masks, it can be an offline teacher for a gradient-trained gate. This is a hybrid, not end-to-end gradient-only learning. For a strictly gradient-only project, replace the teacher with constrained gradient-based mask optimization and retain the same controls.

5. **Test mask geometry separately from sampling density.**

   Independent Bernoulli draws have no explicit mechanism to enforce within-draw spacing. However, learned spatial probability maps can already favor dispersed positions: the existing spacing analysis shows this, particularly for CG-SENSE. Do not assume that LOUPE ignores geometry.

   Compare existing LOUPE, a fixed variable-density Poisson-disc baseline, and an exact-count structured sampler with tunable anisotropic spacing. A candidate gradient version uses perturbed top-K with a relaxed backward pass and a sensitivity-weighted pairwise penalty. Start from the strong LOUPE layout and verify gradients with experiment 2. The existing deterministic DCM run had near-tied scores and unstable sets; simply switching to deterministic top-K is not sufficient.

   Use a 2×2 comparison: fixed/adaptive density crossed with fixed/adaptive geometry. Match approximate radial density when testing geometry, and keep count and ACS exactly equal. Test a constant-input twin of every conditional architecture. This identifies whether the benefit comes from better masks generally or from scan-specific geometry.

   Preserve the intended sampling distribution at inference. The radial REINFORCE report already demonstrates that naive top-K conversion can turn a useful density into a low-pass disc. Freeze representative valid draws and fine-tune, or train with the intended exact-count mechanism; never silently change the acquisition rule only at evaluation.

6. **Align the objective with SSIM, PSNR, and NMSE.**

   Current `SequentialMRI.loss` defaults to whole-image L1; its NMSE option is a ratio of norms over the batch. The evaluation path applies an apodized support mask. Changing `loss_type` alone therefore does not guarantee alignment with the reported subject-level endpoints.

   Compare L1 with per-example normalized squared error and normalized squared error plus SSIM. Fix the metric support from the reference, never from each candidate reconstruction. Include full-image/cropped metrics as secondary endpoints so a gain does not merely move error outside the scoring region. Specify whether SSIM is computed on zeroed images or by averaging local SSIM over support; those are different definitions.

   Apply every objective to population and adaptive arms. Factor the result into objective improvement of the population baseline and additional adaptive improvement. A loss change that improves both equally is useful but does not increase evidence for adaptation. Optimize a prespecified objective; select one model on the tuning split and report all three metrics on test data without cherry-picking a checkpoint per metric.

7. **Evaluation and execution order.**

   Establish subject-disjoint train/tuning/test partitions. The small validation set used repeatedly in the reports is a development set now. Retain a fresh final test set; use cross-validation within development data for cheap screens.

   Use the same sampling budget including calibration. In particular, the reported `ecalib_c10` arm used maps obtained from a 24×24 centre. It is a privileged-map diagnostic unless those extra calibration measurements are counted or supplied by an acquisition common to every arm.

   Keep CG iterations, regularization, sensitivity estimation, reconstruction capacity, optimization steps, precision, and checkpoint selection matched. Tune the CG solver first, then hold it fixed: [the solver sweep](../reports/CGSENSE_HYPERPARAMS_R8.md) shows strong early-stopping effects. For VarNet, evaluate each candidate mask under both a common reconstructor trained on the candidate mask family and a comparably trained dedicated reconstructor. Include the population mask in that family.

   Use paired subject summaries, volume-bootstrap intervals, at least three training seeds for finalists, and explicit between-seed results. Keep trainer aggregates for historical continuity, but report subject-weighted PSNR, SSIM, and NMSE separately. Use common random numbers for probabilistic mask comparisons and evaluate multiple draws or fixed deployment masks. Report anatomy/coil subgroup results and the number of subjects in each.

   Suggested order:

   | Stage | Work | Decision |
   |---|---|---|
   | A | Complete matched VD-start comparison; audit gradients/discrete swaps; standardize scoring and calibration | Is optimization a limiting factor? |
   | B | Population/anatomy/coil-class/encoding residuals, CG-SENSE first | Is the physical condition predictive on unseen volumes? |
   | C | Promote successful geometry/conditioning to matched VarNet training | Does the gain survive a strong reconstructor? |
   | D | Small benefit-routed expert bank, only after best-of-bank screen | Is benefit predictable from available data? |
   | E | Validate the winning hypothesis on actual 3D ky–kz data | Does it survive the acquisition model? |

   Proposed practical target, not a forecast: +0.3 dB PSNR over population LOUPE, positive incremental benefit over anatomy LOUPE, lower NMSE, and no material SSIM regression. Specify the SSIM tolerance before testing. For a fixed target, range, and support, +0.3 dB corresponds to about 6.7% less squared error; +0.5 dB to about 10.9%. Mean PSNR differences do not imply identical reductions in arithmetic mean NMSE.

   For actual SPACE, verify the ACS-first assumption and echo-train scheduling. Use a forward model consistent with echo-dependent contrast before comparing order-sensitive masks. A common ky–kz mask on true 3D data is materially different from 2D point masks on independent fastMRI slices. This change is a more meaningful open question than another larger CNN on the existing surrogate dataset.

   Additional pilot measurements or adaptive acceleration are separate studies. They may create a larger useful effect, but need equal total acquisition-time controls and do not answer the original fixed-budget question. Multi-step acquisition stays outside this plan until SPACE echo-train evolution is modeled.

The near-term choice is **coil-aware residual LOUPE plus discrete-optimization diagnostics**, followed by structured geometry if the diagnostics justify it. SUNO-style retrieval is worth revisiting only with a useful mask bank and a routing target tied to reconstruction benefit. There is presently insufficient evidence to promise a >0.5 dB VarNet gain at fixed R=8 from ACS-only mask adaptation.
