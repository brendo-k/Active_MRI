# Multi-step adaptivity at R=8: does acquiring the budget in K steps make the mask patient-specific? (design + proposal, 2026-11-09)

| | |
|---|---|
| Status | **Frozen 2026-11-10**, as the first commit on `exp/multistep_adaptivity` (experiment loop, step 2). Approved 2026-11-09; section 12 lists what was settled with the user. The decision rule does not change after this commit. |
| Topic / branch / report | `multistep_adaptivity` · `exp/multistep_adaptivity` · `reports/multistep_adaptivity/` |
| Roadmap | Step 3 (multi-step), run as an order-free go/no-go before anyone models the echo train. It runs ahead of step 2 (budget R) because the user asked for it. |
| `main` | `9123f44`, clean, `0 0` against `origin/main` (checked 2026-11-09). On 2026-11-10 `git fetch` could not authenticate, so that count is against the last fetched ref |
| Decided with the user | 2026-11-09: (1) order-free proxy first; (2) a shared base per seed + paired fine-tunes, plus one from-scratch check at the largest K; (3) K ∈ {1, 4, 16}; (4) keep the +0.10 dB pass and +0.05 dB fail bars; (5) 30-epoch fine-tunes, with no extension; (6) the AGENTS.md roadmap line changes on this branch when P1 starts. 2026-11-10: (7) keep the k-space input channel (b), and add an ablation that removes it (A_16^img, reported only); (8) keep both false-negative checks, with the from-scratch pair run only if PASS does not hold after P2; (9) run on CoreHPC |

## TL;DR

- **Question.** Split the R=8 budget into K steps, each chosen from the k-space acquired so far in the same scan. Does that make
  masks patient-specific in a way that improves the reconstruction, compared with the same K steps run blind to the data?
- **Design.** For each seed (42, 43), train one LOUPE + VarNet base. Fine-tune it six ways: K ∈ {1, 4, 16}, each as an adaptive
  arm A_K and a data-blind twin T_K. Within a seed and a K, the arms start bitwise identical and differ only in whether the
  sampler's residual network sees the measured k-space.
- **Primary endpoint.** G_K = PSNR_Masked(A_K) − PSNR_Masked(T_K), paired per slice. K=1 is the null anchor: it is the
  ACS → mask-shape case, which earlier work closed.
- **Pass** if G_K ≥ +0.10 dB at K=4 or K=16 in both seeds with the CI above 0, it beats G_1 by ≥ 0.05 dB, and it is at least
  twice the twin-vs-twin noise floor. **Fail** if G_K < +0.05 dB at both K in both seeds, and both false-negative checks agree:
  the from-scratch pair's within-model swap cost is below +0.05 dB, and the recon-input ablation gains less than +0.05 dB.
- **Compute.** 20 runs on CoreHPC, about 127 GPU-hours. Two of them, the from-scratch pair (about 33 GPU-hours), run only if
  PASS does not hold after P2. These are estimates; the P0 CoreHPC job replaces them with measured step times.
- **Prior.** No published study has tested multi-step 2D sampling with a multi-coil VarNet against a mask-aware twin.
  - Bakker 2022 (multi-coil VarNet) is single-pass with 1D lines, so it speaks to K=1 only, where it agrees with our 1-step
    closure.
  - The only 2D multi-step result is SeqMRI (single-coil U-Net, 128²): 4 steps beat its twin by +0.35 to +0.43 SSIM points.
    That twin was blind to its own mask too, so the gain includes spacing.
  - G_1 ≈ 0 is expected. The size and sign of G_4 and G_16 are what this experiment measures. A pass is necessary for SPACE
    but not sufficient.

## 1. Why this question, and why order-free first

- **1-step adaptivity is closed on this data.** The ACS can predict at most about 0.06 dB per slice and about 0.01 dB per
  volume. Adjacent slices have 0.98-similar ACS, but their per-slice best masks correlate only 0.03. Adaptive conv samplers end
  near-static: slice-vs-mean correlation 0.97–0.99, and they trail LOUPE by 0.16–0.20 dB. Sources:
  `reports/LITERATURE_ADAPTIVE_SAMPLING_R8.md`, `reports/ADAPTIVITY_MECHANISMS_R8.md`, `reports/patient_specific_r8_local_v1/`.
- **The literature's adaptive gains all come from multi-step acquisition, and they are small.**
  - SeqMRI (single-coil knee, U-Net, 128², 2D points) at 8x: 1-step beats its non-seq twin by +0.07 SSIM points, and 4-step
    by +0.41. The non-seq twin sees neither the data nor its own mask, so the 4-step gain includes spacing.
  - Bakker 2020 (multi-step, 1D lines, single-coil) realises about 10% of its per-image oracle gap. Pineda 2020 (multi-step,
    1D) finds a dataset-specific policy as good as a subject-specific one.
  - Bakker 2022 (multi-coil VarNet, 4 virtual coils, 128²) found the best policies are static (mutual information 0). It is
    single-pass with 1D lines, so it bears on K=1 only.
  - Every multi-step null is 1D; the one 2D multi-step study is positive. 2D multi-step sampling with a multi-coil VarNet
    against a mask-aware twin has not been tested.
- **Where the headroom might be.** Per-slice ground-truth oracle masks gain +0.91 dB [0.80, 1.03] over a fitted per-anatomy map.
  Almost all of it is pixel-scale: blurred to a 4-px density, the oracle keeps only +0.07 dB. None of it can be predicted
  from the ACS. These are legacy numbers (pre-local-v1, older code), so they are context here, not a control. At step i > 0 the sampler sees the slice's own
  high-frequency data. That is the first input that could locate this structure.
- **Why order-free is roughly an upper bound.**
  - In SPACE, each sample's weighting depends on its echo position. An adaptive policy must view-order every step's new samples
    as they come, so ordering constraints take freedom away from it.
  - A static mask can be view-ordered once, for the whole mask.
  - So a gain that is absent order-free should also be absent in SPACE. A gain that shows up order-free still has to survive an
    echo-train model.
  - The bound is only approximate, because the static twin also pays some ordering cost in SPACE.

## 2. What "more adaptive" means, operationally

- **Adaptivity gain**: G_K, as defined in the TL;DR. The adaptive arm A_K and its twin T_K share:
  - the base checkpoint, the sampler architecture, K and the selection mechanics;
  - the seeded mask draws, the data order and the header information.

  Only the k-space values reach A_K and not T_K.
- **The twin T_K must still see its own previous mask M_i and the step index.** A sampler that knows where its earlier random
  draws landed can fill the gaps, which gives blue-noise spacing. That gain is not adaptive. On legacy data and the legacy
  metric, a Poisson-disc draw of the LOUPE density beat an independent draw by about +0.15 dB in brain and +0.05 dB in knee
  (`reports/spacing_vdinit_r8/`). A twin blind to both data and mask would credit that spacing gain to adaptivity.
  - The existing `fixed_input` path cannot serve as this twin: it replaces the whole observation.
- **Both arms get the header.** The header here means the condition (brain / knee-PD / knee-PDFS), the active coil count and the
  acquired extent.
  - Brain and knee are trivially easy to tell apart from the data.
  - The knee-specific mask alone was worth 0.24–0.29 dB to knee slices in an older label swap, and the coil count shifts
    the best brain mask (`docs/idea.md`).
  - A twin without the header would credit header routing to adaptivity. G_K therefore measures adaptivity beyond what the
    header already tells you.
- **Behavioural adaptivity** (secondary) is the data-swap sensitivity. Hold M_i, the header and the thresholds fixed, and replace
  the measured data (both data channels) with another slice's. Then count how many of the step's new samples move.
  - It is exactly 0 for T_K and for A_K at init, so any nonzero value is data use.
  - It is measured per step, whether or not the use helps.
  - Plain slice-to-slice mask variation would not work as this measure. Each slice's available region differs after its own
    draws, so p_i varies across slices in both arms, even with a dead head.
- **Usefulness** is kept separate from whether adaptivity exists. It is U_K = PSNR(A_K) − PSNR(T_1), the margin over the best
  static fine-tune. It is reported but does not decide pass or fail.

## 3. The current code at K > 1: must-fix list (Phase 0)

| # | Issue | Where | Consequence | Fix | Test |
|---|---|---|---|---|---|
| 1 | Only the last step's choices get gradient | `old_mask = pred_dict['mask'].detach()`, `activemri/models/sequential_sampling.py:569`; the loss uses only the final output (`:624`) | At K=16, 15/16 of the budget would be drawn by a network trained only on its last-step behaviour | Carry the straight-through union across steps; detach only where M_i is used for availability and as input | Every step's probabilities get a nonzero gradient from the final loss at K=4; K=1 stays bitwise unchanged |
| 2 | At steps > 0 the sampler reads predicted k-space, not measured k-space | `pred_kspace = FFT(recon · S) · extent`, `:587` | `final_dc` measured it about 40% off the data at sampled points, and the default `coil_combine=sense` input has random phase | Build the input from the measured y ⊙ M_i, RSS-combined | Leakage: the step-i probabilities do not change when unacquired k-space is perturbed |
| 3 | Every step runs the full VarNet | `:385` | It reconstructs partial masks (R > 100 after step 0 of K=16) it never trained on, at K× the cost | With measured input, reconstruct only the final mask M_K during training; intermediate reconstructions only at eval, under no_grad, for figures | One reconstructor call per training forward |
| 4 | No data-blind twin with mask access | `fixed_input` replaces the whole observation and is reachable only through `training.num_step=0` → 1 step (`activemri/training/builders.py:155`) | No valid twin for K > 1 | A `sampler.data_blind` flag that keeps M_i, the step index and the header | The probabilities do not depend on k-space values but do vary with M_i |
| 5 | Single-step guards | `propose_mask` (`:482`), `fixed_mask` and `mask_policy` raise when `num_step != 1` | `eval_patient_specific_r8.py`'s machinery does not carry over | A new K-step eval script (section 5) | Null calibrations in section 5 |

## 4. Design

### 4.1 Fixed for every arm

- **Data**: `fastmri_local_v1_flair_knee` on current `main`, which includes the brain-flip fix `6ee2565`, at
  `misc.resolution=[320,320]`.
  - Train: 160 volumes, 3,929 slices.
  - Val: 80 volumes, 1,938 slices (brain FLAIR 605, knee-PD 673, knee-PDFS 660).
  - On CoreHPC the same manifest rows are read from `~/scratch/data/fastmri/{anatomy}/data`, through the `datasets.roots`
    overrides in `configs/datasets/fastmri_local_v1_base.yaml` (`hpc` for brain and val, `local_batch2` for the train knee
    rows). The manifest loader crops the readout, so it never reads the `sense` datasets that only the local copies carry.
- **Reconstructor**: E2E VarNet with 12 cascades and 18 channels, Muon + LayerNorm (the `main` defaults), `dc_mode=where`.
- **Budget**: R=8 over acquired k-space (`sampler.budget_over=acquired`), with the 10×10 centre preselected. The remaining budget
  is split equally over K steps: about 2,200–3,200 new samples per step at K=4, and about 550–800 at K=16.
- **Metric**: the trainer's PSNR_Masked, SSIM_Masked and NMSE, inside the apodized per-anatomy metric mask. Every number is
  measured on the current loader. Nothing is compared with numbers from before 2026-11-09.

### 4.2 Sampler (`seq_adapt_2d`, new)

At step i = 0 … K−1, with mask M_i (M_0 is the 10×10 centre):

- **Logits**: logits_i = slope · L_c + r_θ(z_i), where:
  - L_c is a static logit map per condition. All three start as copies of the base's shared LOUPE logits and are trainable.
  - r_θ is a small U-Net: chans 16 or 32 (picked from the P0 timings), depth 4. Its last convolution starts at zero, and its
    output is bounded to 6·tanh(·/6).
- **Inputs z_i**: two data channels, then the mask and header channels.
  - **Data, adaptive arm**:
    - (a) the RSS of IFFT(y ⊙ M_i), divided by the peak of the ACS's RSS image;
    - (b) the coil-RSS magnitude of y ⊙ M_i itself, divided by its ACS peak and log-scaled to [0, 1], with 0 where nothing is
      measured. It lines up pixel for pixel with the logits, so a local rule such as "sample next to measured energy" needs
      no global transform. Channel (b) was added at P0 and approved by the user on 2026-11-10. The A_16^img ablation
      measures what it contributes.
  - **Data, twin**: a fixed tensor from a local, seeded generator. The twin's sampler is never handed the measured k-space,
    so its blindness is structural, and the global random stream is not touched.
  - **Both arms**: M_i, the acquired extent, a constant i/K channel, the active coil count divided by the coil dimension, and
    a one-hot condition (brain / knee-PD / knee-PDFS).
- **Selection**: rescale sigmoid(logits_i) over the available region (valid and not in M_i) to (B − |M_0|)/K expected new
  samples. Draw a Bernoulli mask with seeded thresholds, using the straight-through estimator. Then M_{i+1} = M_i ∪ b_i, with the
  straight-through increments kept in the graph.
- **Reconstruction**: one VarNet pass on M_K.
- **Identical start**: r_θ is built under `torch.random.fork_rng`, as `EncoderMLPHead` was, so every other weight and the RNG
  stream match the twin bitwise. Because r_θ starts at 0, A_K ≡ T_K at initialisation. This is asserted on the first batch:
  identical masks, and the loss equal to within GPU non-determinism.
- **Memory**: if P0's memory check needs it, r_θ gets activation checkpointing per step.

### 4.3 Training

- **Stage 1, the base B_s** (s ∈ {42, 43}): the `muon_layer` recipe of `reports/muon_layernorm_100ep/`, run on current `main`.
  Every existing brain checkpoint predates the flip fix, so none of them can be a control.

  ```
  recon=varnet datasets=fastmri_local_v1_flair_knee misc.resolution=[320,320] sampler.accelerations=[8.0]
  sampler.budget_over=acquired training.num_step=1 training.batch_size=16 training.num_epochs=100
  training.lr_step_size=90 sampler.sampler_type=loupe_2d sampler.loupe_vd_init_power=4
  sampler.anatomy_conditioning=false datasets.limit_volumes=1 training.seed=s
  ```

- **Stage 2, the fine-tunes**, starting from B_s:
  - Everything trains: the VarNet, the sensitivity net, L_c and r_θ. Keeping the VarNet trainable guards against H2, under
    which a reconstructor co-adapted to a static mask taxes deviations from it.
  - Each fine-tune gets a fresh optimizer, with lr 1e-4 for every group (B_s's final lr), held constant for 30 epochs.
  - r_θ trains in its own Adam group at `sampler.residual_lr` (1e-4). It stays out of the Muon group that takes the U-Nets'
    hidden convolutions, so its lr means what this section says. A P0 test checks the grouping.
  - A_K and T_K use the same fine-tune seed, so they see the same data order and the same draws.

### 4.4 Arms (20 runs)

| Group | Runs | Purpose |
|---|---|---|
| Bases | B_42, B_43 | Shared start; static reference |
| Sweep (for each base seed) | T_1, A_1, T_4, A_4, T_16, A_16 | Primary endpoint G_K; spacing effect T_K − T_1; usefulness U_K |
| Noise floor (base 42) | T_1′, T_16′: fine-tune seed 142, so a different data order and different draws | Twin-vs-twin floor F_K |
| From-scratch check (seed 42), run only if PASS does not hold after P2 | A_16^scr, T_16^scr: 100 epochs on the base recipe, starting from the same init (L from the VD-4 prior, r_θ = 0) | False-negative check for H2 |
| Input ablation (base 42) | A_4^rec: z_i also gets the frozen B_42 VarNet's magnitude reconstruction of y ⊙ M_i, under no_grad (SeqMRI-style). Its comparator is T_4 | False-negative check: is the measured-only input the bottleneck? |
| Channel ablation (base 42), always run | A_16^img: A_16 with channel (b) replaced by the twin's fixed values, so its sampler sees the measured data only as the image channel (a). Same architecture, base, init and fine-tune seed as A_16 and T_16, so the three start identical | Reported only: does the k-space channel carry the adaptivity? Readout A_16 − A_16^img, paired, with its CI |

## 5. Measurements

The new eval script is `scripts/eval/eval_multistep_adaptivity.py`. It scores each run's `last_model.pt` (no selection on val)
on all 1,938 val slices.

- **Draws**: 2 seeds (1234, 4321) define every step's thresholds. A and T get the same thresholds, so they share random numbers,
  and each slice is scored with its own data range.
- **CIs**: 95% volume-cluster bootstrap of the slice-weighted mean (volumes resampled within each condition). Reported per
  condition and pooled.

**Primary**
- G_K, for each seed and K, per condition and pooled.

**Noise floor**
- F_K = |mean(T_K − T_K′)|, with its CI, at K=1 and K=16.

**Secondary**
1. **Trajectory swap cost** (within one model, so no training noise):
   - Run A_K's full trajectory on a donor slice, using the target's thresholds. The donor is another volume's slice with
     the same condition, acquired-location mask (which fixes the extent, the available region and the budget) and active
     coil count. Slices with no such donor are left out of the swap analysis and counted. On the val set every knee slice
     has one, and 515 of the 605 brain slices (85%).
   - Apply the donor's final mask to the target.
   - The swap cost is PSNR(own) − PSNR(donor).
   - Null calibration: with the header matched, T_K's trajectory is the same for target and donor, so its swap cost is exactly
     0. This tests the eval code.
   - Known caveat: the swap cost includes the penalty of someone else's deviation (patient_specific: +0.05 cost, but own minus
     condition mean ≈ 0), so it is supporting evidence, not the primary endpoint.
2. **Per-step data-swap sensitivity**, the behavioural-adaptivity curve.
   - What it measures: the fraction of step i's new samples that move when the measured data (both data channels) is
     swapped for the header-matched donor's from item 1, with M_i, the header and the thresholds held fixed. Also the RMS
     change in r_θ.
   - It is exactly 0 for T_K.
   - The slice-vs-condition-mean correlation of p_i is reported only as an A-vs-T comparison. It falls with step index in both
     arms, because each slice's available region differs.
3. **Spacing effect**: T_K − T_1. **Usefulness**: U_K = A_K − T_1. Also both against B_s.
4. SSIM_Masked and NMSE, with the same pairing.
5. Optional, eval-only: switch to the donor's data from step j onward, to find which steps carry G_K.

**Guards and sanity**
- **SSIM**: pooled SSIM_Masked ≥ 0.80 for every run. Knee-PDFS is about 0.78 in every arm already (`muon_layernorm_100ep`), so
  its guard is no more than 0.005 below its own base.
- **Budget**: the realised fraction is 0.125 ± 0.002 of acquired k-space at every K, and each step's count is about budget/K.
- **Twin flat in K**: |T_16 − T_1| ≤ 0.25 dB. A larger gap means a plumbing problem, not spacing.
- **K=1 null**: |G_1| ≤ 0.05 dB. If G_1 > +0.05 in both seeds, that contradicts the 1-step closure, so audit for leakage before
  reading the K > 1 results.
- **Failed reconstructions**: slices below 20 dB are listed. Known cases are PDFS file1001159, file1001818, file1000344 and
  file1000528. The primary endpoint includes them, and a sensitivity analysis leaves them out.
- **Base**: B_s pooled PSNR_Masked within ±0.5 dB of `muon_layer_s42/s43` (33.86 / 33.61, before the flip fix). Otherwise stop
  and check.

## 6. Expected outcomes

| Quantity | Prediction | What other results would mean |
|---|---|---|
| G_1 | ≈ 0 (\|G\| ≤ 0.03 dB) | > +0.05: leakage or header routing slipping through; audit |
| G_4 | 0 to +0.05 dB | ≥ +0.10: adaptivity appears within a few steps |
| G_16 | Uncertain, sign included; 0 to +0.15 dB is the likely range. For > 0: SeqMRI's 2D multi-step gain, though part of it is spacing. For ≈ 0: our 1-step closure and the 1D multi-step nulls. Optimistic: up to +0.3 (10–35% of a ~0.9 dB per-slice ceiling) | Grows with K: within-scan feedback carries information, so go on to the echo-train model. Flat: within-scan feedback does not help this reconstructor |
| Data-swap sensitivity of A_K by step | Grows with step index if feedback is used (T_K: 0 by construction) | Sensitive but G ≈ 0: adaptivity the reconstructor cannot exploit (a H2-type tax) |
| Swap cost of A_16 | Tracks G_16, and is smaller | > 0 while G ≈ 0: masks are slice-specific but no better than the twin's |
| T_16 − T_1 (spacing) | 0 to +0.15 dB | ≥ +0.10 in both seeds: a deployable non-adaptive by-product (section 8, P3) |
| U_16 | G_16 plus spacing; most likely about +0 to +0.15 | G > 0 but U < 0: adaptivity exists but the parameterisation loses statically |
| SSIM, NMSE | Follow PSNR (SSIM ±0.002) | SSIM against PSNR: check the metric path |
| G(A_4^rec) − G_4 | ≈ 0 | ≥ +0.05: the measured-only input is the bottleneck, so a fail cannot be declared from the main arms |
| Scratch pair: A_16^scr's swap cost and data-swap sensitivity | Same reading as the fine-tunes | Scratch adapts (swap cost ≥ +0.05) but the fine-tunes do not: the shared base suppresses adaptivity (H2). Scratch A − T is supporting evidence only: one from-scratch seed, no measured floor |
| A_16 − A_16^img (the k-space channel) | ≈ 0 if G_16 ≈ 0. If G_16 > 0, a positive share of it, since the image channel needs a global transform to inform k-space choices | ≥ +0.05 with G_16 > 0: the adaptivity runs through the k-space channel. ≈ 0 with G_16 > 0: the image channel carries it. ≤ −0.05: channel (b) hurts, which the report flags |

## 7. Decision rule (frozen at the proposal commit)

Here K* ∈ {4, 16}. F* = F_16 for K* = 16, and max(F_1, F_16) for K* = 4.

- **PASS**: all of the following hold for some K*.
  - G_K* ≥ +0.10 dB pooled in both base seeds, with each CI above 0.
  - G_K* − G_1 ≥ +0.05 dB in both seeds.
  - G_K* ≥ 2·F*.
  - No condition has G_K* ≤ −0.10 dB in both seeds.
  - The guards hold.
- **FAIL**: all of the following hold.
  - G_K < +0.05 dB at K=4 and at K=16, in both seeds.
  - The from-scratch pair, which runs whenever PASS does not hold after P2, agrees on within-model readouts: A_16^scr's
    trajectory swap cost < +0.05 dB pooled.
    A_16^scr − T_16^scr is reported as supporting evidence, not used as a condition: it is one from-scratch seed, and its floor
    is unmeasured.
  - The ablation agrees: G(A_4^rec) < +0.05 dB.
- **INCONCLUSIVE**: anything else.
  - Add base seed 44 at the best K: B_44, A, T, about 1 day.
  - The 3-seed rule is then: mean G_K* ≥ +0.10 dB, all three positive, at least two with the CI above 0, and G_K* − G_1 ≥ +0.05
    in all three.
  - If only the from-scratch check or the ablation disagrees, the case goes back to the user with a costed from-scratch or
    recon-input sweep.
- **Usefulness** (reported only): U_K* has its CI above 0 in both seeds, meaning the adaptive arm beats the best static
  fine-tune.
- **Channel ablation** (reported only): A_16 − A_16^img with its CI, and G(A_16^img) = A_16^img − T_16. It enters neither
  PASS, FAIL nor INCONCLUSIVE.

The +0.10 dB bar is the low end of the effects that matter here. It is about 2× the expected paired twin floor; F_K is measured,
and the rule above enforces that factor. The bar is well above what the ACS can predict (≤ 0.06 dB per slice).

## 8. Roadmap

```
P0 infra + tests + CoreHPC job ──► P1 bases + pilot ──gate──► P2 sweep + floor + ablations ──► P2b from-scratch pair, unless PASS ──► report ──► user verdict
                                                                                                                                                 │
                     PASS ──► P3a echo-train model (own design doc) ──► P3b in-house SPACE  ◄─────────────────────────────────────────────────────┤
                     FAIL ──► park, write the closure, back to roadmap step 2 (ACS → budget R)  ◄─────────────────────────────────────────────────┤
                     INCONCLUSIVE ──► seed 44 at the best K ──► same rule (3-seed version)  ◄──────────────────────────────────────────────────────┘
```

Every phase lives on the one branch `exp/multistep_adaptivity`, because it is a single question. P3a and P3b would be new
branches.

**P0, infrastructure** (no GPU results; about 1–2 days of code)
1. Approve the proposal. Then create the branch and worktree, and freeze the proposal as the first commit.
2. Make fixes 1–5 from section 3. Add the `seq_adapt_2d` sampler, and the config keys `sampler.data_blind`,
   `sampler.input={measured,recon}`, `sampler.residual_chans`, `sampler.residual_lr` and the header inputs. Before P2, add
   the recon input for A_4^rec and `sampler.data_channels` for A_16^img (which data channels read the slice's data; the rest
   get the twin's fixed values), each with a test that it starts identical to A_K.
3. Add tests:
   - gradient to every step;
   - K=1 bitwise equal to `main` for `loupe_2d`;
   - leakage;
   - one recon call per training forward;
   - data-blind invariance;
   - A ≡ T at init;
   - per-step and total budget at K ∈ {1, 4, 16};
   - r_θ in its own Adam group at `sampler.residual_lr`, not in the Muon group.
4. Write the K-step eval script and check its null calibrations:
   - an at-init A/T pair gives G = 0 exactly;
   - T_K's swap cost is 0 with header-matched donors;
   - data-swap sensitivity is 0 for T_K and for A at init.
5. Measure the selection mechanics at init.
   - Re-normalising one static map to budget/K at every step does not reproduce its K=1 marginals at intermediate
     probabilities.
   - Monte-Carlo each location's union coverage at K=4 and K=16 against K=1, from B_s's map.
   - The distortion is shared by A and T, so it cancels in G_K. It does enter T_K − T_1 and the pilot's plumbing gate, so
     report it beside them.
6. Smoke-test every arm through `train.py` for a few batches on GPUs 0–1.
   - Time a step and record peak memory at K ∈ {1, 4, 16}, with chans 16 and 32, and with and without checkpointing.
   - Pick r_θ's width so that a K=16 step costs ≤ 2.5× a K=1 step.
   - Replace section 9's estimates with these numbers.
   - If K=16 still does not fit in 46 GB with checkpointing, substitute K=8. This is the only change to K allowed in advance.
7. Validate on CoreHPC, in one SLURM job on the pinned GPU type (section 9), never on the login node. The local checks ran
   under uv (Python 3.13); CoreHPC runs the conda `dl` env (Python 3.14, torch 2.12). The job:
   - runs the full test suite, including A ≡ T at init and the eval nulls;
   - checks that every train (tier S) and val manifest row resolves under the `roots` overrides, for both source types, and
     that the loader yields 3,929 train and 1,938 val slices;
   - trains a 1-epoch K=4 A/T smoke pair and scores it with the eval: the twin's swap cost and data-swap sensitivity must be
     exactly 0;
   - runs K=16 at batch 16 with `torch.cuda.max_memory_allocated` logged, and times K ∈ {1, 4, 16};
   - checks that `run_info.json` records the shipped commit.

**Gate P0 → P1**: all tests pass, every arm smoke-tests, and the CoreHPC job passes.

**P1, bases + pilot**
- Train B_42 and B_43 in parallel.
- The pilot is the first 10 epochs of the real seed-42 K=16 pair, A_16 and T_16, so it costs no extra compute. Both runs
  keep `epoch_010.pt` (`training.save_epochs=[10]`), and the gates are scored on it by `scripts/eval/eval_multistep_adaptivity.py`,
  the same code as the final eval. The per-step curves logged to wandb during training are a monitor, not the gate. Gates at
  epoch 10:
  - **Stability**: finite loss, at most 10 consecutive skipped steps, both arms ≥ B_42 − 0.2 dB.
  - **Plumbing**: realised fraction 0.125 ± 0.002, and T_16 − B_42 within [−0.2, +0.3] dB.
  - **Head used**: at some step ≥ 1, A_16's data-swap sensitivity moves ≥ 2% of the step's new samples. For T_16, and for A
    at init, it is exactly 0.
    - If not, retry once with `sampler.residual_lr` at 3e-4; this retry is decided in advance.
    - If it is still not used, continue: a static adaptive arm is a legitimate result.
- If the stability or plumbing gate fails, the runs count as debug runs, not results. Fix and relaunch.

**P2, sweep**: the remaining 10 fine-tunes, the floor pair and the two ablations, A_4^rec and A_16^img. Each is evaluated as
soon as it finishes.

**P2b**: the from-scratch K=16 pair, run whenever PASS does not hold on the P2 results (so FAIL and INCONCLUSIVE both trigger
it). A PASS does not use it.

**Report**: written with the `experiment-report` skill, with the criteria table. Then wait for the verdict.

**P3, conditional**
- **PASS**: the echo-train model gets its own design doc. It covers:
  - the signal weighting of the variable-flip-angle train per echo position, estimated by EPG;
  - per-step view ordering into echo trains;
  - a contrast-matched reference;
  - both arms under the same constraints, with the static twin view-ordered globally.

  Then retest at the best K, then move to the in-house SPACE data. That data is 40+40 volumes, and its IRB is still unconfirmed.
- **FAIL**: close multi-step for fastMRI proxies with VarNet at R=8. Record G_K against the ceiling context. Return to step 2,
  ACS → budget R; a sequential "when to stop" rule is the natural link to it. Edits to the AGENTS.md roadmap and to memory notes
  are made only with the user's OK.
- **Either way**: if T_16 − T_1 ≥ +0.10 dB in both seeds, propose the sequential, mask-aware static sampler separately. Once a draw
  is frozen it is a fixed mask with learned spacing, so it deploys in SPACE without the ordering problem.

## 9. Compute plan (estimates; P0 replaces them)

The assumptions, in A6000 hours (the CoreHPC job measures the pinned type):
- A 1-step LOUPE + VarNet epoch takes about 301 s on one A6000 (`muon_layer_speed`).
- Relative to that epoch, a K=1 epoch costs about 1.05×, K=4 about 1.25× and K=16 about 2×.
- A_4^rec costs about 2× K=4, from 3 extra frozen-VarNet forwards per step. A_16^img costs the same as A_16.

| Phase | Runs | GPU-h | Wall-clock, 4 GPUs at a time |
|---|---|---|---|
| P0 smoke + timing, local (done) and the CoreHPC job | short | ~5 | ~3 h |
| Bases | 2 × 100 epochs | ~17 | ~9 h |
| Sweep | 12 × 30 epochs | ~43 | ~11 h |
| Noise floor + ablations | 4 × 30 epochs | ~19 | ~6 h |
| From-scratch pair, only if PASS does not hold | 2 × 100 epochs at K=16 | ~33 | ~17 h |
| Eval | 20 runs | ~10 | ~3 h |
| **Total** | **20 runs** | **~127, or ~94 without the from-scratch pair** | **~2 days, or ~1.3 without it** |
| If inconclusive: seed 44 at the best K | 1 base + 2 fine-tunes | ~25 | ~1 day |

- **Platform**: CoreHPC, partition `gpu`, account `hpc_core`, 4-day limit. The longest run, a 100-epoch K=16 from-scratch arm,
  is about 17 A6000-hours.
- **One GPU type for the whole experiment**: every run and every eval uses one type with at least 80 GB, either H100 NVL
  (94 GB) or RTX Pro 6000 (96 GB). The CoreHPC job picks it from `sinfo` and records its gres string in
  `reports/multistep_adaptivity/p0/README.md`, and it does not change after P1 starts. Mixing types would add a numerical
  difference between paired arms that is not the treatment. L40S cards (48 GB) are excluded: K=16 at batch 16 reserved 48.0
  of 48.6 GB on an A6000.
- **Code**: the branch ships as a `git bundle`, cloned into a new folder on CoreHPC scratch, so each run's `run_info.json`
  records its real commit. The diverged `~/scratch/code/active_MRI` copy is not touched.
- **wandb**: offline on the compute nodes. The offline run folders are copied back with rsync and synced from the lab server.
- Each run is evaluated as soon as it finishes, on the same GPU type. A crashed run is rerun in full.
- On OOM, fall back to `recon.checkpoint_cascades=true`, then to r_θ checkpointing.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| H2 false negative: a base co-adapted to a static mask taxes deviations | The VarNet trains during the fine-tune; the from-scratch K=16 pair; within-model swap and diversity readouts |
| The residual collapses to a static pattern, as the MLP heads did | Zero-init, bounded residual; per-condition L_c carries the static part; the pilot's data-swap gate with one retry decided in advance |
| Spacing or header routing credited as adaptivity | The twin sees M_i, the step index and the header |
| The measured-only input hides information a reconstruction would expose | The A_4^rec ablation is part of the fail rule |
| Memory or time at K=16 | A small r_θ, per-step checkpointing, P0 timing; the only allowed fallback is K=8 |
| Leakage of unacquired k-space into the sampler | Unit test; the K=1 null |
| Seed noise: 0.25 dB between `muon_layer` seeds; paired from-scratch arm differences ranged +0.12 to +0.33 dB across 4 contexts | Shared bases, the same fine-tune seed, shared random numbers in eval, a measured twin floor, two base seeds |
| The order-free proxy credits masks the scanner cannot acquire in that order | A pass is necessary but not sufficient; P3a models the order before any SPACE claim |
| CoreHPC's software stack differs from the local one the P0 checks ran on | The CoreHPC job (P0 item 7) reruns the tests, the A/T nulls and the memory check there before P1 |
| Paired arms on different GPU types | One type for every run and eval (section 9) |

## 11. Out of scope

- **Echo-train and view-ordering physics**: P3a.
- **RL/PPO policies**: training is straight-through only. Bakker 2022's learned policies (single-pass, 1D) came out static,
  and our PPO samplers stayed below the static ones.
- **Variable budget or adaptive stopping**: roadmap step 2.
- **Scan latency** between steps.
- **1D line sampling**.
- **A SeqMRI-style reconstruction input at every K**: only the A_4^rec ablation tests it.

## 12. Settled with the user (2026-11-09 and 2026-11-10)

1. **Bars**: keep the +0.10 dB pass bar and the +0.05 dB fail bar.
2. **Fine-tune recipe**: 30 epochs at a constant 1e-4, with no extension.
3. **False-negative checks** (the from-scratch K=16 pair, about 33 GPU-h, and the A_4^rec ablation, about 6 GPU-h) stay as
   fail-rule conditions. Without them a null reads only as "no gain fine-tuned from a static base with zero-filled input".
   On 2026-11-10 the recommendation was to keep both and to run the from-scratch pair only if PASS does not hold after P2.
   The user asked to freeze after that, which is taken as accepting it.
4. **AGENTS.md**: the roadmap line "Multi-step … blocked until echo-train evolution is modelled" changes to "order-free
   go/no-go running" on this branch when P1 starts. The `no_multistep_space` memory note needs its own OK.
5. **Channel (b)**, the log-scaled measured k-space magnitude, stays (2026-11-10). The user asked for an ablation of it:
   A_16^img, reported only.
6. **Platform**: the runs go to CoreHPC (2026-11-10), behind the CoreHPC job of P0 item 7.
