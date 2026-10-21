# Is single-pass, ACS-conditioned adaptive sampling a dead end at R=8? The literature against our results (2026-10-19)

## Question
Every input-adaptive sampler trained here at R=8 ends near-static, and per-slice oracle masks cannot be predicted from
the calibration centre (`reports/ADAPTIVITY_MECHANISMS_R8.md`). Setup: brain + knee, VarNet, a 10 x 10 calibration
centre (the ACS), 256 x 256 point masks. Several papers report gains from adaptive sampling. Do they contradict our
results, and is ACS-conditioned adaptivity worth more work?

Conventions:
- *Single-pass*: the mask is chosen once, from the calibration data.
- *Multi-step*: later choices see data acquired earlier in the same scan.
- *Non-adaptive twin*: a static mask trained with the same architecture, loss, optimiser and reconstructor as the
  adaptive model.
- *SSIM points*: SSIM x 100.
- In-house PSNR is the trainer's background-masked metric. Literature numbers are as reported in each paper.

## Runs compared
**Primary papers** (full text read):

| Paper | Data, test size | Coils, matrix | Mask, R | Reconstructor | Calibration given to the sampler | Decision |
|---|---|---|---|---|---|---|
| MNet: Huang & Ravishankar, IEEE TCI 2022 (arXiv 2111.09212) | fastMRI knee, 1,300 test images | single-coil, 320 x 320 | 1D lines, 4x / 8x | U-Net, MoDL | 16 / 8 centre lines (5% / 2.5% of k-space) | single-pass |
| SeqMRI: Yin et al., ML4H 2021 (2105.06460) | fastMRI knee, 1,851 test slices | single-coil, k-space cropped to 128 x 128 | 2D points 4x / 8x / 16x; 1D lines 4x | U-Net | 1/8 of the budget at the centre (0.8-3.1% of k-space) | 1, 2 or 4 steps |
| Bakker et al., MIDL 2022 (2203.16392) | fastMRI knee, 7,135 val slices | 4 virtual coils, 128 x 128 | 1D, 4x / 8x | E2E-VarNet, 5-7 cascades | 10 / 4 centre lines (7.8% / 3.1%) | single-pass |
| SUNO: Gautam et al., IEEE TCI 2026 (2501.09799) | fastMRI knee 104, brain 60 test slices | multi-coil, knee 640 x 368 | 1D, 4x / 8x | ZS-SSL, E2E-VarNet, MoDL | 30 / 15 centre lines, a third of the budget | single-pass (nearest-neighbour lookup) |
| Xu & Öksüz, Neurocomputing 2025, DOI 10.1016/j.neucom.2025.130116 | ACDC 798, fastMRI knee 1,245 test slices | single-coil; k-space simulated from [0, 1]-normalised images | 1D, 4x / 6x | pretrained PromptMR (knee) | centre lines (count not stated) | multi-step (PPO, one line per step) |

Xu & Öksüz is paywalled. It was read as section 3.3 of Xu's 2026 ITU PhD thesis ("based on" the paper) and the KSRO
repository README.

**References chased** (those that test adaptivity):

| Paper | Setting | Decision | Why it is here |
|---|---|---|---|
| Pineda et al., MICCAI 2020 (2007.10469) | single-coil knee 640 x 368, pretrained cResNet | multi-step (DDQN) | subject-specific vs dataset-specific policy |
| Bakker, van Hoof & Welling, NeurIPS 2020 (2010.16262) | single-coil knee 128 x 128; brain 256 x 256 simulated from ground-truth images; pretrained U-Net | multi-step (16 / 28 acquisitions) | adaptive policy vs a static oracle fitted on the test set |
| Sanchez, Krawczuk & Cevher 2021, ch. 6-7 of Sanchez's thesis (2312.04327) | replications of the two above | multi-step | fixed greedy mask (sLBCS) vs RL; Fourier vs image-domain sampling |
| Levine & Hargreaves, IEEE TMI 2017 (1710.07837) | 3D breast, 16 channels, SENSE (linear) | single-pass, from coil maps | the physical case for calibration-driven sampling |
| Yen et al., ICML 2024 (2406.04318); dSUNO, Gautam et al. 2026 (2602.13984) | pathology classification at 5-10% sampling; dynamic cardiac (26 / 52 test slices) | multi-step; single-pass lookup | positive results in other regimes |

**In-house runs referenced.** Final logged wandb validation metrics, one training seed each.
- VarNet runs share 12 cascades, 18 channels, R=8 (12.5% of 256 x 256), the 10 x 10 centre preselected and
  batch 16.
- The lr 1e-3 runs use `brain_knee_balanced`. The lr 3e-4 runs predate several config keys (e.g. recon
  `dc_mode`), and `azd6r64k` uses LOUPE slope 10 rather than 5. They are not exact copies of the lr 1e-3 runs.

| Name | wandb | Checkpoint | lr, epochs | Val brain PSNR / SSIM | Val knee PSNR / SSIM |
|---|---|---|---|---|---|
| LOUPE (static) | `uo8d68n0` | `checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05/last_model.pt` | 1e-3, 50 | 36.55 / 0.933 | 34.26 / 0.953 |
| conv_2d, 10 x 10 k-space input | `ziuji11f` | `checkpoints/conv2d_varnet_r8_brain_knee_floor0.05/last_model.pt` | 1e-3, 50 | 36.28 / 0.931 | 34.07 / 0.951 |
| conv_2d, image input + bias head | `lj113kv4` | `checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput_headbias/` | 1e-3, 50 | 36.51 / 0.933 | 34.20 / 0.952 |
| conv_2d, image input + MLP head | `069x1apw` | `checkpoints/conv2d_varnet_r8_brain_knee_floor0.05_imginput_headmlp_ln_clipsplit/` | 1e-3, 50 | 36.40 / 0.932 | 34.15 / 0.952 |
| uniform random masks | `fkg5xsd6` | `checkpoints/random_mask_varnet_r8_brain_knee/last_model.pt` | 1e-3, 50 | 34.85 / 0.922 | 33.48 / 0.947 |
| LOUPE shared | `azd6r64k` | deleted | 3e-4, 50 | 36.08 / 0.930 | 33.90 / 0.949 |
| LOUPE per-anatomy | `pjysura8` | deleted | 3e-4, 50 | 36.25 / 0.931 | 34.05 / 0.951 |
| LOUPE shared | `1k9d0xdj` | deleted | 3e-4, 100 | 36.54 / 0.933 | 34.13 / 0.953 |
| LOUPE per-anatomy | `xum09g7g` | `checkpoints/loupe2d_varnet_r8_brain_knee_anatcond_100ep/last_model.pt` | 1e-3, 100 | 36.72 / 0.934 | 34.35 / 0.954 |
| CG-SENSE LOUPE shared | `u2xei0pq` | `checkpoints/loupe2d_cgsense_r8_brain_knee_shared_bilinear_const_100ep` | 1e-3 constant, 100 | 34.92 / 0.913 | 32.71 / 0.929 |
| CG-SENSE LOUPE per-anatomy | `kikc9627` | `checkpoints/loupe2d_cgsense_r8_brain_knee_anatcond_bilinear_const_100ep` | 1e-3 constant, 100 | 35.08 / 0.915 | 32.51 / 0.925 |

## Methods
1. **Retrieval.** Full text came from arXiv via the `paper-search` CLI, except Xu & Öksüz (above). References were
   chased from the five papers' bibliographies wherever they compare adaptive and static sampling. Not read:
   - Van Gorp et al. 2021 (A-DPS): the OpenReview PDF did not download;
   - Seeger et al. 2010 (Bayesian design for CS-MRI): paywalled.
2. **Two questions per paper:**
   - (i) single-pass or multi-step;
   - (ii) is there a non-adaptive twin?

   Also recorded:
   - a shuffle control (a scan scored with another scan's adaptive mask);
   - test-set size and whether paired statistics are reported;
   - data realism: coils, cropping, magnitude-derived k-space. Real-valued images give conjugate-symmetric k-space,
     which raw data does not have.
3. **Adaptivity effect** = adaptive minus non-adaptive twin, from the papers' tables. Where no twin exists, the
   comparison is adaptive minus the best static baseline, with the confound named.
4. **Realised vs oracle.** Where a paper reports a per-image oracle (a mask optimised with the ground truth of that
   image), realised adaptive gain / oracle gain is computed against the same static reference.
5. **In-house evidence** is re-read from existing outputs, not new experiments:
   - `reports/conv_vs_loupe_r8/label_gonogo.json`, `reports/conv_vs_loupe_r8/adaptivity.json`;
   - `analysis/label_gonogo/{acs_vs_label_similarity,oracle_blur_density,smooth_predictor,best_of_k_oracle,oracle_init_identifiability}.json`;
   - `analysis/label_gonogo/reweighted/privileged_full_gain_lr1e-3_dev0.json`;
   - `reports/adaptivity_next_steps_r8/screen_{observed10,centre24,full}.json`;
   - `reports/anatcond_r8/{paired,label_swap_100ep}.json` (from `scripts/eval/compare_anatcond_r8.py` and
     `scripts/eval/label_swap_anatcond.py`);
   - `analysis/anatcond_cgsense/cross_eval_100ep{,_train}.json`;
   - wandb summaries.

   One new number is the per-volume base PSNR spread (`scripts/diagnostics/per_volume_base_psnr_spread.py` ->
   `analysis/subject_coadapt/per_volume_base_psnr_spread.json`). It covers the 22 volumes of the block-split
   subject co-adaptation runs, with the LOUPE VarNet `uo8d68n0` as base, averaged over each volume's scored slices.
   - Label "predictability" is the correlation of a predicted mask deviation with the oracle label's deviation from
     the anatomy mean.

## Results
### 1. Paper by paper: was adaptivity itself tested?

| Paper | Single-pass? | Non-adaptive twin? | What adaptivity itself added |
|---|---|---|---|
| SeqMRI | 1-step arm | Yes: "non-seq", the same network fed a random tensor | 2D: 1-step minus non-seq +0.10 / +0.07 / +0.15 SSIM points (4x / 8x / 16x). 1D 4x: 1-step ≈ non-seq. Larger gains need steps 2-4 (section 2) |
| Bakker 2022 | Yes | Yes: mutual-information (MI) analysis of the policies, LOUPE, equispaced | The best policies have MI = 0 with their input, i.e. they are static. The 8x win (93.26 vs LOUPE 91.37 SSIM points) is a better static mask. A policy placed after the first cascade made no significant difference |
| MNet | Yes | No | 8x NMSE 0.0415 vs LOUPE 0.0451, but the two were trained differently: LOUPE end-to-end with an unnormalised L2 loss, MNet by alternating label generation with normalised losses. At 4x, equispaced SSIM 0.7658 vs MNet-SEP 0.7660. No constant-input or shuffle test |
| SUNO | Yes | Partly: LOUPE and BASS population masks, no shuffle or random-neighbour control | Knee VarNet PSNR: 4x SUNO 32.55 vs BASS 32.48; 8x BASS 29.99 vs SUNO 29.57. Brain VarNet: 33.51 vs LOUPE 33.40 (4x), 29.51 vs 29.24 (8x). No paired statistics, so significance cannot be judged. Per-scan training masks also diversify the reconstructor's training masks, a confound |
| Xu & Öksüz | No | No: random-mask baselines only | Knee 4x SSIM: random 0.60, reward variants 0.67-0.68. Says nothing about adaptivity. The reconstructor is a fixed pretrained PromptMR, and k-space comes from magnitude images |
| Pineda 2020 | No | Yes: dataset-specific DDQN (input = step count only) | The dataset-specific DDQN matches or beats the subject-specific one on most metrics, e.g. MSE AUC 7.94 vs 8.00, with equal SSIM AUC (30-line start) |
| Bakker 2020 | No | Yes: static greedy oracle fitted on the test set | Knee: the adaptive greedy policy +0.0017 / +0.0021 SSIM (base / long horizon). Brain: +0.0007 |
| Sanchez et al. 2021 | No | Yes: sLBCS fixed greedy mask | Fixed mask ≈ both RL methods: RL minus sLBCS is -0.0007 to +0.0015 SSIM across settings, and rankings flip with preprocessing |

The SUNO authors write that whether scan-adaptive sampling beats population-based sampling "remains open in the deep
learning setting".

**Reading:** every study that has a non-adaptive twin finds single-pass adaptivity at or near zero. Positive
single-pass claims (MNet, SUNO) lack the twin or the shuffle control.

### 2. SeqMRI's own ablation is the single-pass test
SSIM points on the fastMRI knee test set (SeqMRI Fig. 6 and Tables 1-3). "Beats LOUPE" is the share of test images
better than LOUPE (Table 3; 1D from Fig. 10).

| Setting | LOUPE | Non-seq | 1-step | 2-step | 4-step | Beats LOUPE: non-seq / 1-step / 4-step |
|---|---|---|---|---|---|---|
| 2D 4x | 92.44 | 92.56 | 92.66 | 92.74 | 92.91 | 74% / 77% / 97% |
| 2D 8x | 90.60 | 90.66 | 90.73 | 90.95 | 91.07 | 60% / 57% / 93% |
| 2D 16x | 88.73 | 88.67 | 88.82 | 88.88 | 89.10 | 47% / 51% / 77% |
| 1D 4x | 89.52 | 90.89 | 90.84 | 91.20 | 91.08 | 95% / 93% / 97% |

**Reading:**
- One adaptive step adds 0.07-0.15 SSIM points over the static twin in 2D and nothing in 1D. At 8x the 1-step
  model beats LOUPE on fewer images than the static twin does.
- The 1D gain over LOUPE (+1.4 points) is the framework, not adaptivity.
- Gains of +0.35-0.43 points need 2-4 steps that see newly acquired high-frequency data. The 1D values are read from
  the figure text.

### 3. Realised vs oracle adaptivity

| Study | Static reference | Per-image oracle (uses ground truth) | Realised adaptive | Realised / oracle |
|---|---|---|---|---|
| Bakker 2020, knee, base horizon (SSIM) | static oracle 0.7213 | 0.7379 (+0.0166) | 0.7230 (+0.0017) | 10% |
| Bakker 2020, knee, long horizon | 0.7421 | 0.7623 (+0.0202) | 0.7442 (+0.0021) | 10% |
| Bakker 2020, brain, base horizon | 0.9099 | 0.9141 (+0.0042) | 0.9106 (+0.0007) | 17% |
| Pineda setting (Sanchez Table 6.4), SSIM AUC | sLBCS 0.6079 | 0.6341 (+0.026) | subject-specific DDQN 0.6139 (+0.006) | 23% |
| same, PSNR at 100 lines | sLBCS 30.21 dB | 30.68 (+0.47) | 29.93 (-0.28) | < 0 |
| SUNO knee MoDL 4x (PSNR) | BASS 31.67 | 31.87 (+0.20) | 31.74 (+0.07) | 35%, no CI |
| SUNO knee MoDL 8x | BASS 28.99 | 29.60 (+0.61) | 28.66 (-0.33) | < 0 |
| Ours, per slice (frozen conv ep50 VarNet, the 128 labelled val slices) | fitted per-anatomy map | +0.91 dB [+0.80, +1.03] | predictor -0.001 [-0.020, +0.017]; kNN -0.015 [-0.032, +0.004] | ≈ 0 |
| Ours, smooth band (4 px density blur) | anatomy-mean label | +0.066 [+0.044, +0.091] | ridge on 10 x 10 +0.009 [+0.003, +0.015]; kNN +0.017 [+0.009, +0.024] | 13-26% |

All literature rows above are multi-step or a dictionary lookup. **Reading:** wherever an oracle gap is reported, at
most about a third of it is realised. In-house the share is 0 per slice, and 13-26% of a gap that is itself 0.07 dB.

### 4. Population and reconstructor effects are far larger than adaptivity

| Change | Effect | Source |
|---|---|---|
| RL policy vs fixed greedy mask (Bakker 2020 setting) | -0.0007 to +0.0015 SSIM | Sanchez Tables 6.2-6.3 |
| U-Net -> cResNet reconstructor, same setting | +0.006 to +0.017 SSIM | same |
| Highest-mean-power mask ("Spectrum") vs LOUPE, SeqMRI 2D 4x / 8x / 16x | -0.05 / -0.22 / -0.36 SSIM points | SeqMRI Table 1 |
| Ours: LOUPE vs uniform random masks, each with its own VarNet | +1.70 dB brain, +0.78 dB knee | `uo8d68n0` vs `fkg5xsd6` |
| Ours: 50 -> 100 epochs, shared LOUPE, lr 3e-4 (lr step at 40 vs 90) | brain +1.16 [+0.88, +1.44], knee +1.05 [+0.64, +1.45] dB paired (logged: +0.47 / +0.23) | `azd6r64k` vs `1k9d0xdj`, `reports/anatcond_r8/paired.json` |
| Ours: per-anatomy vs shared run, 50 epochs, lr 3e-4 | brain +0.36 [+0.28, +0.44], knee +0.67 [+0.46, +0.88] dB paired (logged: +0.17 / +0.15). The mask-specific part (label swap) is ≈ 0 / +0.29 | `pjysura8` vs `azd6r64k`, same file |
| Ours: best ACS-input sampler vs LOUPE | -0.04 / -0.06 dB (logged) | `lj113kv4` vs `uo8d68n0` |

About the paired rows:
- They come from `scripts/eval/compare_anatcond_r8.py`: per-slice PSNR with a per-slice data range, the mean over
  slices, and a 95% CI over per-volume means.
- The script's aggregate reproduces the logged numbers. The per-slice statistic weights slices differently and comes
  out larger.
- The checkpoints behind them have since been deleted, so the file cannot be regenerated.

None of the ACS-input samplers is more adaptive than a static mask: each slice's map correlates 0.98-0.99 with its
anatomy mean, and the MLP head gives 1.000 (`adaptivity.json`). **Reading:** a population-level mask choice is
nearly as good as the learned one. Training length and the anatomy split each move PSNR more than any ACS-conditioned
sampler has.

### 5. What the 10 x 10 ACS can and cannot see (ours)

| Quantity | Value | Source |
|---|---|---|
| ACS similarity of adjacent slices (complex, mean) | 0.971 | `acs_vs_label_similarity.json` |
| Oracle-deviation correlation of adjacent slices: pixel band (below 4 px) / smooth band | 0.030 / 0.53 | same |
| Same, nearest-ACS slice of another volume | -0.001 / 0.21 | same |
| Oracle gain vs density blur sigma 0 / 1 / 2 / 4 / 16 px | +0.92 / +0.22 / +0.10 / +0.066 / +0.049 dB | `oracle_blur_density.json` |
| Label predictability, input 10 x 10 / 24 x 24 / full k-space (val; train fit) | 0.060 (0.054) / 0.059 (0.053) / 0.123 (0.108) | `screen_*.json` |
| Same, full k-space, gain-weighted loss | 0.154 (0.143) | `privileged_full_gain_lr1e-3_dev0.json` |
| Oracle refit from the LOUPE start vs the original, per-slice deviation correlation (raw / 4 px) | 0.56 / 0.74 | `oracle_init_identifiability.json` |
| Perfect pick from K per-anatomy oracle medoids, K = 4 / 16 / 64 | -0.13 / -0.08 / -0.10 dB (random pick -0.24 / -0.30 / -0.34) | `best_of_k_oracle.json` |
| One mask per volume: shared fraction pixel / smooth; held-out correlation pixel / smooth | 0.0027 / 0.128; 0.009 / 0.27 | `acs_vs_label_similarity.json` |

**Reading:**
- About 90% of the per-slice oracle gain is pixel-scale structure.
- That structure changes between slices whose ACS are 97% alike.
- Starts disagree about where it goes.
- Even full k-space predicts it at only 0.12-0.15 correlation, so a larger calibration region is not the missing
  ingredient.
- `ADAPTIVITY_MECHANISMS_R8.md` converts these caps to about 0.06 dB per slice and about 0.01 dB per volume.

### 6. Anatomy routing: the one routable lever

| Contrast | Brain | Knee | Note |
|---|---|---|---|
| VarNet per-anatomy minus shared run, 50 epochs, lr 3e-4, paired per slice | +0.36 [+0.28, +0.44], 96% of 23 volumes better | +0.67 [+0.46, +0.88], 90% of 10 volumes better | `paired.json`; logged aggregates +0.17 / +0.15; one seed each |
| Label swap in the 50-epoch per-anatomy run (own mask minus the other anatomy's) | -0.02 [-0.05, +0.01] | +0.29 [+0.21, +0.37], 10 of 10 volumes | `paired.json` |
| VarNet per-anatomy (lr 1e-3) minus shared (lr 3e-4), 100 epochs, logged | +0.18 dB | +0.22 dB | lr-confounded |
| Label swap in `xum09g7g`, 100 epochs (other anatomy's mask minus own) | -0.009 [-0.030, +0.013] | -0.241 [-0.312, -0.169], 10 of 10 volumes worse | `label_swap_100ep.json` |
| CG-SENSE own per-anatomy mask minus shared, 100 epochs (344 slices) | +0.16 [+0.09, +0.24] | -0.25 [-0.34, -0.17] | knee mask loses even on training slices: -0.14 [-0.23, -0.06] (`cross_eval_100ep_train.json`) |
| CG-SENSE label swap | -0.98 [-1.26, -0.82] | +0.18 [+0.05, +0.32] | the brain mask is better on knee than the knee mask |

**Reading:**
- Under VarNet only the knee mask is anatomy-specific: giving knee slices the brain mask costs 0.24-0.29 dB, while
  the brain mask can be swapped for free.
- The paired run difference is larger (+0.36 / +0.67 dB). Its part not explained by the mask is about +0.37 dB on
  both anatomies.
- **Hypothesis, untested** (one seed per run): that part is run-to-run variance of 50-epoch lr-3e-4 runs, which are
  far from converged (at lr 3e-4, 50 -> 100 epochs is worth +1.1 dB per slice, +0.47 / +0.23 logged).
- Under CG-SENSE the pattern reverses: the brain mask is specific, and the knee mask is worse than the brain mask on
  knee. So the value of a per-anatomy mask depends on the reconstructor, and per-class mask optimisation can fail.

### 7. Per-volume quality at fixed R=8 (the budget lever)

| Volumes | Min | 10th pct | Median | 90th pct | Max |
|---|---|---|---|---|---|
| all, n = 22 | 22.6 | 34.9 | 40.0 | 42.6 | 46.1 dB |
| brain, n = 16 | 22.6 | 30.2 | 39.9 | 42.6 | 43.5 dB |
| knee, n = 6 | 39.3 | 39.3 | 41.2 | 43.9 | 46.1 dB |

**Reading:** volume-to-volume quality at one R spans about 8 dB between the 10th and 90th percentiles. That is about
25-30 times the mask-specific anatomy effect (0.24-0.29 dB, knee).

## Conclusions
- **Where a single-pass adaptive sampler has a matched non-adaptive twin, adaptivity adds about zero:**
  - SeqMRI 1-step: +0.07-0.15 SSIM points in 2D, about 0 in 1D;
  - Bakker 2022: the best multi-coil policies are static;
  - Pineda and Sanchez: static ≈ adaptive.

  Our conv samplers agree: ≤ LOUPE, with slice maps correlating 0.98-0.99 with the anatomy mean. The literature
  does not contradict our result.
- **The positive claims come from four places.**
  - Multi-step acquisition: SeqMRI steps 2-4, Bakker 2020, Pineda.
  - Comparisons without a non-adaptive twin or shuffle control: MNet, SUNO, Xu & Öksüz.
  - Small test sets without paired statistics: SUNO.
  - Different regimes: dynamic cardiac (dSUNO), pathology classification (Yen 2024).
- **"Adaptive architecture beats LOUPE" can mean "better static mask".** Examples: Bakker 2022 at 8x, our image-input
  + bias-head sampler, SeqMRI's 1D non-seq.
- **The size of the calibration centre is not the bottleneck.** 24 x 24 and full-k-space predictors fail to fit the
  per-slice labels too.
- **Measured mechanism.** The per-slice oracle gain is pixel-scale structure that decorrelates between adjacent slices
  while their ACS do not. The ACS-predictable share is at most about 0.06 dB per slice and about 0.01 dB per volume.
  SPACE needs one ky-kz mask per volume, so per-slice gains would not deploy there even if they were predictable.
- **Hypothesis, untested here.** With a linear-Gaussian model, the posterior covariance does not depend on the
  measured values, so the optimal remaining samples are fixed in advance.
  - Adaptivity can only pay when the observed data moves *where* the uncertainty is.
  - Fourier samples are spread over the whole image. What a 10 x 10 centre reveals (scale or SNR, support, coil
    geometry, contrast class) mostly rescales the per-frequency uncertainty or reshapes it smoothly.
  - That predicts the ACS tells you how many samples to take more than which ones.
  - Consistent with this: SeqMRI's highest-mean-power mask is within 0.05-0.36 SSIM points of LOUPE, and in
    Sanchez ch. 7 adaptivity pays in image-domain but not Fourier-domain sampling.
- **Verdict.** ACS -> mask shape at fixed R is closed for fastMRI-like fixed-protocol data with a jointly trained
  VarNet. This is not a claim that adaptive sampling cannot work anywhere.
  - The only positive routable lever is anatomy. Under VarNet the knee mask is worth 0.24-0.29 dB to knee slices
    (label swap), and the brain mask is not specific.
  - Larger paired run differences (+0.36 / +0.67 dB, one seed each) are not attributable to the mask.

## Caveats
- **Literature:**
  - Numbers are as reported, not reproduced.
  - SeqMRI's 1D values come from figure text, and its Table 3 and Fig. 12 disagree for 4-step at 8x (92.6% vs 89.3%).
  - SUNO reports no paired intervals.
  - Xu & Öksüz was read from the thesis chapter, which may differ from the journal version.
  - A-DPS and Seeger 2010 were not read.
  - The five primary papers were chosen by the user, and references were chased only where they test adaptivity.
    Work from 2023-2026 was not searched systematically beyond dSUNO and Yen 2024.
- **Cross-study comparisons mix metrics** (SSIM points, SSIM AUC, PSNR), datasets and crops. The ratios in section 3
  compare effect sizes, not methods.
- **In-house:**
  - All runs are single training seeds.
  - The oracle and predictor numbers go through one frozen VarNet (conv ep50), and the oracle keeps about 70% of its
    gain on other conv-trained VarNets.
  - fastMRI 2D slices stand in for SPACE ky-kz planes.
  - The paired anatomy statistics use a per-slice data range and are larger than differences of the logged
    aggregates. The checkpoints behind them have been deleted.
  - The 100-epoch pair differs in lr, lr schedule and config keys.
  - The per-volume spread covers 22 volumes (16 brain, 6 knee), scored on the held-out half of each volume with that
    script's metric, and includes two low brain outliers (22.6 and 25.6 dB).
- **The linear-Gaussian argument is a heuristic.** A trained VarNet is not a Gaussian posterior mean.

## Figures
None new. Related figures:
- `reports/conv_vs_loupe_r8/masks.png`, `radial.png`: conv vs LOUPE masks and radial profiles.
- `reports/figures/mask_spacing_r8.png`: CG-SENSE vs VarNet mask spacing.

## Next steps
The chosen order is anatomy routing now, then the ACS choosing the budget R, then multi-step.

1. **Anatomy routing (current).**
   - Train an lr-matched 100-epoch pair (shared vs per-anatomy, same code, config and lr schedule), two seeds each.
     Score paired per slice with a label swap. Two seeds are needed to tell whether the non-mask part of the
     50-epoch difference (about +0.37 dB) is run-to-run variance.
   - Fix per-class mask optimisation before trusting per-class gains: initialise the per-anatomy masks from the
     shared mask and decay the lr. Test protocol classes (PD / PDFS) as a variant, and recheck the CG-SENSE knee sign
     flip.
   - Route on the DICOM header (`bodyPartExamined`), not an ACS classifier.
2. **ACS -> budget R (next).**
   - The objective becomes scan time at matched quality.
   - Cheap first test: does the ACS rank per-volume PSNR at fixed R=8 (section 7)? Then fit an R-selection rule and
     score it as time saved at a quality floor.
3. **Multi-step (later).**
   - This is where the literature's adaptive gains are: SeqMRI +0.35-0.43 SSIM points over its static twin at
     4 steps, single-coil 128 x 128. Bakker 2020 realises about 10% of its oracle gap.
   - Prerequisite for SPACE: model the echo-train signal evolution, so that acquisition order sets each sample's
     weighting. Without it the simulation credits masks the scanner cannot realise with the assumed contrast.
4. **Dropped:**
   - per-slice or per-volume ACS -> mask-shape predictors;
   - larger calibration centres;
   - SUNO-style mask dictionaries through a frozen reconstructor.
