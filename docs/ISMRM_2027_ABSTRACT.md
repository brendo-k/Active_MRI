# ISMRM 2027 abstract: working draft (anatomy-specific vs shared vs scan-adaptive learned masks, R=8)

Deadline **28 Oct 2026** (ECHO; confirm the time zone there). Co-author draft target: **Fri 23 Oct**.
Limits: Impact 40 words · Synopsis 100 words (Motivation / Goals / Approach / Results, text only, no citations) ·
body 750 words · ≤5 figures, captions ≤500 characters · one Preview Figure · code/data links in Acknowledgements only.
`[[...]]` = number pending or decision open. Numbers come only from the scripts in "Evidence map"; all use the trainer's
current metric mask. Never mix in legacy-metric numbers (`reports/metric_mask_reeval_r8/report.md`).

**Pre-set decision rule (2026-10-20, before results):** the anatomy result leads if the matched pair shows per-anatomy >
shared with the 95% CI excluding 0 on at least one anatomy and no significant loss on the other, in both seeds. Otherwise
the adaptivity result leads and the anatomy result is reported as knee mask specificity (label swap).

---

## Title options (1 holds only if the pair passes the decision rule)
1. Adapt the k-space mask to the anatomy, not to the scan: learned sampling at R=8
2. Anatomy-specific learned undersampling masks outperform a shared mask, while scan-adaptive sampling collapses to a static one

Brenden Kadota, Yang Yang, [[affiliations]]

## Impact (≤40 words; 36 now; bracket depends on the matched pair)
Learned sampling can be deployed as one learned mask per anatomy, selected from the scan's body-part header: this
[[helped knee at no cost to brain]], while scan-adaptive samplers gained ≤0.1 dB and trailed a static mask.

## Synopsis (≤100 words)
**Motivation:** Learned undersampling masks can be specialised per anatomy or per scan, but which level pays off at a fixed
acceleration is unclear.
**Goals:** Compare one shared learned mask with anatomy-specific masks and with scan-adaptive samplers, each trained
jointly with a multi-coil variational network at R=8.
**Approach:** Probabilistic masks and calibration-conditioned convolutional samplers were trained on fastMRI brain and knee,
with matched shared/per-anatomy pairs, and validated on in-house undersampled 3D SPACE.
**Results:** Per-anatomy masks improved PSNR by [[x]] dB (brain) and [[y]] dB (knee). Scan-adaptive samplers converged to
near-static masks and trailed the static mask by 0.07-0.64 dB.

---

## Abstract body (≤750 words)

### Introduction
Learned k-space sampling optimises the undersampling pattern jointly with the reconstruction [1-3]. The pattern can be
specialised at two levels. A population-level mask, for example one per anatomy, can be chosen by the scanner from the
protocol. A scan-level mask is chosen from data acquired first, as in active or adaptive sampling [4-7]. Where adaptive
samplers were compared with a matched static twin, adaptivity added little [5,7-9]. We test both levels in one
multi-coil setting: does a per-anatomy learned mask beat a shared one, and does a sampler that sees each scan's calibration
data beat a static mask?

### Methods
**Data.** fastMRI multi-coil brain (T2) and knee (PD, PDFS) [10], k-space centre-cropped to 256×256 (brain readout
oversampling removed first), coils zero-padded or truncated to 16. Balanced split: 1350 training slices per anatomy
(89 brain, 40 knee volumes) and 344 validation slices per anatomy (23 brain, 10 knee volumes). Masks are 2D point patterns
on this grid, emulating the ky–kz plane of 3D Cartesian imaging, with R=8 (12.5% of samples) including a fully sampled
10×10 centre.

**Reconstruction.** An end-to-end variational network [11] (12 cascades, 18 channels, sensitivity network on the centre),
trained jointly with the sampler using an L1 loss and Adam (batch 16).

**Samplers.** (i) LOUPE [1]: learnable logits, sigmoid probabilities rescaled to the budget, Bernoulli draws with a
straight-through gradient. The *shared* model learns one map; the *per-anatomy* model learns one map per anatomy, selected
by the anatomy label (at the scanner, a lookup on DICOM BodyPartExamined). The pair is identical except for the label:
100 epochs, lr 10⁻³ (×0.1 at epoch 90), seeds 42 and 43. (ii) Scan-adaptive: a U-Net maps the coil-combined 10×10
calibration k-space (or its image) to the probability map; [[4]] variants. These, LOUPE, a fixed Gaussian variable-density
mask and uniform random masks were each trained with a matched 50-epoch recipe (lr 10⁻³, ×0.1 at epoch 40).

**Evaluation.** PSNR and SSIM of magnitude images multiplied by a foreground mask computed from the fully sampled data
(identical for all methods; data range = per-slice reference maximum), averaged over two common mask draws. Differences
are paired per slice with 95% bootstrap CIs over volumes. A mask × reconstructor cross-evaluation separates the mask's
contribution; a label swap tests anatomy specificity; a shuffle control scores each slice with the mask its sampler
proposed for a slice from another volume of the same anatomy.

**External validation.** In-house prospectively undersampled 3D SPACE, 40 brain and 40 knee volumes ([[IRB statement:
UNCONFIRMED, must be checked with the PI before submission]]); the same shared/per-anatomy pair trained self-supervised
[12]; endpoint: error on held-out acquired k-space in held-out subjects, paired per volume.

### Results
[[Fill from `reports/ismrm2027/anatomy_pair_seed42.json`, `..._seed43.json`, `paired_50ep_vs_loupe.json`,
`reports/adaptivity_next_steps_r8/eval_ismrm_conv.json`, SPACE readout.]]
- Per-anatomy minus shared (run − run): brain [[+a [CI]]], knee [[+b [CI]]] dB PSNR, seeds 42 / 43 (Fig. 2).
- Mask effect on a fixed reconstructor: [[...]]. Label swap in the per-anatomy model (other anatomy's mask, same
  reconstructor; new metric, 2 draws; computed inline 2026-10-20, to be reproduced by `anatomy_pair_seed42.json`):
  knee −0.17 [−0.19, −0.14] dB, 10/10 volumes worse (PD −0.19, 4/4; PDFS −0.15, 6/6); brain −0.003 [−0.03, +0.02], 11/23.
  The knee mask is knee-specific and serves both knee protocols; brain slices do as well with the knee mask.
  Both maps put ~70% of samples at kx < 0; rms radius 54 / 58 px (ky / kx) brain, 57 / 57 px knee.
- Learned maps (Fig. 1): [[describe: knee map shifts samples toward ... ; brain map ...]].
- Scan-adaptive samplers: none of six beat static LOUPE: brain −0.14 to −0.64 dB, knee −0.07 to −0.44 dB PSNR (all
  brain CIs < 0; knee CI covers 0 only for the 2-draw variant). Per-slice maps correlate 0.99 with their anatomy mean and the
  brain and knee mean maps 0.988-0.999. Within each adaptive model, the shuffle control (mask made for another volume's
  input) costs 0 to +0.11 dB and the anatomy swap 0 to +0.18 dB, largest for the variant closest to LOUPE (2 draws:
  +0.113 [+0.103, +0.165] / +0.182 [+0.166, +0.253]); the MLP-head variant is exactly static (0 / 0). These costs include
  the co-adapted reconstructor's mismatch, so they bound the value of adaptivity from above. Source:
  `reports/adaptivity_next_steps_r8/eval_ismrm_conv.json`. LOUPE beats a fixed Gaussian VD mask by +0.67 [0.50, 0.84] / +0.50 [0.29, 0.70] dB and uniform
  random masks by +2.71 / +1.69 dB (brain / knee) (Fig. 4).
  Source: `reports/ismrm2027/paired_50ep_vs_loupe.json` (2026-10-20); LOUPE reference 39.34 / 38.47 dB, SSIM 0.9807 / 0.9502.
  Two static LOUPE runs differing only in the gradient floor differ by −0.02 / −0.03 dB (same seed, not a seed-variance estimate).
- SPACE: per-anatomy minus shared held-out k-space error [[...]] in [[n]] held-out subjects (Fig. 5).

### Discussion and conclusion
(User's paragraph, 2026-10-23, with the "can't learn the optimum" sentences added. The user wrote "~0.2 dB brain /
~0.1 dB knee"; replaced with the matched VarNet pair numbers until they say which run theirs came from.)

Here we show that learning an anatomy-specific sampling mask yields modest gains: +0.13 and +0.18 dB PSNR in the knee
in two training seeds, and a seed-dependent effect in the brain (−0.05 and +0.13 dB). The knee result is consistent with
our initial hypothesis, since different anatomies have different underlying k-space energy distributions, and prior
k-space sampling studies have reported distinct optimal sampling patterns for brain and knee imaging [1]. The main
limitation is that the gain from an anatomy-specific mask is modest. Masks optimised on each slice's fully sampled data
reconstruct that slice better still, but this scan-specific optimum cannot be learned in the current single-pass setup,
where the mask is fixed before acquisition. Adjacent slices have nearly identical calibration data (correlation 0.97),
yet the slice-specific parts of their optimised masks are almost uncorrelated (r = 0.06), and samplers given each scan's
calibration data converged to near-static masks. Previous studies have increased adaptability by sampling k-space
sequentially, building the mask up over multiple acquisition steps so that later samples are chosen from data already
acquired [4-7]. However, prospective implementation of such approaches must account for k-space reordering and for the
signal evolution of the variable-flip-angle echo train. Addressing these acquisition constraints is a necessary step
toward translating learned sampling strategies to prospective scans.

[[Old draft also listed: two training seeds, retrospective undersampling of 2D fastMRI slices, one acceleration, SPACE
scored against held-out k-space. Keep?]]

---

## Figures (≤5, captions ≤500 characters)
1. **Learned sampling maps.** Shared, brain and knee LOUPE maps (mean effective probability), fixed VD, radial profiles.
   Source: `anatomy_pair_seed42.pt` (`mean_maps`). Preview Figure = the three maps without text.
2. **Anatomy pair.** Forest plot: run − run, mask effect on each reconstructor, label swap; brain / knee-PD / knee-PDFS;
   both seeds.
3. **Examples.** Brain and knee slices near the median PSNR (not volume AXT2_201_2010571, a ~25 dB outlier): reference,
   shared, per-anatomy, error maps ×[[5]].
4. **Scan-adaptive samplers.** (a) slice-vs-mean map correlation; (b) shuffle-control cost; (c) Δ PSNR vs LOUPE for conv
   variants, VD, random.
5. **SPACE external validation.** Per-subject paired Δ held-out k-space error; one example per anatomy.

## Evidence map (claim → source)
| Claim | Script / output | Status |
|---|---|---|
| Matched pair config differs only in `anatomy_conditioning` | `.hydra/config.yaml` diff (checked 2026-10-20) | done |
| Pair results, seed 42 | `scripts/eval/eval_anatomy_pair_ismrm_r8.py --tag seed42` | shared run due ~00:00 Oct 21 |
| Pair results, seed 43 | same, `--tag seed43` | queued on GPU 3, due ~Oct 21 evening |
| 50-epoch table vs LOUPE | `scripts/eval/paired_vs_reference.py` → `reports/ismrm2027/paired_50ep_vs_loupe.json` | done 2026-10-20 |
| Shuffle control, adaptivity | `scripts/eval/eval_adaptive_samplers_r8.py --tag ismrm_conv` | done 2026-10-20 |
| Oracle per-slice gain is not predictable (optional, Discussion) | `reports/ADAPTIVITY_MECHANISMS_R8.md` | legacy metric: re-score or state as loss/correlation |
| Discussion: adjacent slices' ACS correlation 0.97, oracle-mask deviations r = 0.06 (no dB numbers; oracle labels fitted on training slices through the frozen conv ep50 VarNet, not the LOUPE pair) | `scripts/diagnostics/acs_vs_label_similarity_r8.py` → `reports/label_gonogo_r8/acs_vs_label_similarity.json` (`same volume, 1 slices apart`: `acs_complex` mean 0.971, `label_unsplit` mean 0.059, 513 pairs) | used 2026-10-23 |
| Discussion: knee +0.13 / +0.18, brain −0.05 / +0.13 (run − run) | `reports/ismrm2027/anatomy_pair_seed42.json`, `..._seed43.json` (`contrasts.run_minus_run.psnr`) | used 2026-10-23 |
| "Distinct optimal patterns for brain and knee" | LOUPE [1] abstract: "significantly different for brain vs knee MRI scans" (arXiv:1907.11374, checked 2026-10-23) | done |
| SPACE pair | user's runs (rsync to `/tmp/bkadota_space/`) | due Oct 23; IRB unconfirmed |

## References (checked against arXiv/primary records 2026-10-20 unless marked [[verify]])
1. Bahadir CD, Wang AQ, Dalca AV, Sabuncu MR. Deep-learning-based optimization of the under-sampling pattern in MRI. IEEE TCI 2020;6:1139-1152.
2. Aggarwal HK, Jacob M. J-MoDL: joint model-based deep learning for optimized sampling and reconstruction. IEEE JSTSP 2020;14:1151-1162.
3. Zhang J, et al. Extending LOUPE for k-space under-sampling pattern optimization in multi-coil MRI. MLMIR 2020:91-101.
4. Zhang Z, Romero A, Muckley MJ, Vincent P, Yang L, Drozdzal M. Reducing uncertainty in undersampled MRI reconstruction with active acquisition. CVPR 2019:2049-2058.
5. Pineda L, Basu S, Romero A, Calandra R, Drozdzal M. Active MR k-space sampling with reinforcement learning. MICCAI 2020:23-33 (arXiv:2007.10469).
6. Bakker T, van Hoof H, Welling M. Experimental design for MRI by greedy policy search. NeurIPS 2020 (arXiv:2010.16262).
7. Yin T, Wu Z, Sun H, Dalca AV, Yue Y, Bouman KL. End-to-end sequential sampling and reconstruction for MRI. Proc ML4H, PMLR 2021;158:261-281 (arXiv:2105.06460).
8. Bakker T, Muckley M, Romero-Soriano A, Drozdzal M, Pineda L. On learning adaptive acquisition policies for undersampled multi-coil MRI reconstruction. MIDL 2022 (arXiv:2203.16392).
9. Sanchez T. Learning to sample in Cartesian MRI. PhD thesis, EPFL, doi:10.5075/epfl-thesis-9981 (arXiv:2312.04327) [[verify thesis year]].
10. Zbontar J, et al. fastMRI: an open dataset and benchmarks for accelerated MRI. arXiv:1811.08839 (2018).
11. Sriram A, et al. End-to-end variational networks for accelerated MRI reconstruction. MICCAI 2020:64-73 (arXiv:2004.06688).
12. Yaman B, Hosseini SAH, Moeller S, Ellermann J, Uğurbil K, Akçakaya M. Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data. MRM 2020;84:3172-3191.

Citation map for the body: learned sampling [1-3]; adaptive / active sampling [4-7]; matched static twins [5,7-9]
(Pineda: dataset- vs subject-specific policy; SeqMRI: non-sequential twin; Bakker 2022: policies with zero mutual
information with their input; Sanchez: fixed greedy mask vs RL); data [10]; VarNet [11]; SSDU [12].

## Acknowledgements (not shown to reviewers)
[[grants]] · code: [[repo link]]
