# ISMRM 2027 abstract: DRAFT TEMPLATE (single-step multi-coil LOUPE, R=8)

Format follows "ISMRM Abstract 2027" in Drive (Synopsis / Introduction / Theory / Methods / Results / Discussion / Figures / References).
`[[...]]` = fill in or decide. Numbers marked (†) come from project notes, are single-seed and have open confounds. Re-run before submitting.

---

## Synopsis

**Motivation:** Learned k-space sampling methods are mostly demonstrated on single-coil or single-anatomy data. Whether one learned pattern can serve several anatomies in a multi-coil setting, and what the mask contributes compared with the reconstructor, is unclear. Sequential (adaptive) samplers add complexity, and their benefit over a single learned static pattern is rarely tested.

**Goal:** To learn a single-step k-space undersampling pattern jointly with a multi-coil unrolled reconstruction at R=8, and to measure how much is gained by anatomy-specific patterns and by input-adaptive sampling.

**Approach:** A LOUPE-style probabilistic mask, with a straight-through relaxation, is trained end-to-end with a sensitivity-estimating variational network on fastMRI brain and knee. Comparisons: variable-density and random masks, REINFORCE and PPO policy samplers, and a convolutional sampler that conditions on the calibration region or image.

**Results:** [[headline: learned mask vs variable density, dB PSNR / SSIM, brain and knee]]. Anatomy-conditioned masks gave (†) +0.18 dB (brain) and +0.22 dB (knee). Input-adaptive samplers produced near-static masks and did not beat the learned static mask.

**Impact:** [[one sentence: a single learned static pattern captures most of the achievable gain at R=8, so scanner implementation can use a fixed per-anatomy pattern.]]

---

## Title options
1. [[Joint Learning of Single-Step k-Space Sampling and Multi-Coil Reconstruction: How Much Does Adaptivity Add at R=8?]]
2. [[Is a Learned Static Mask Enough? Single-Step LOUPE vs Sequential and Input-Adaptive k-Space Sampling]]

Brenden Kadota, [[Charles Milliard]], [[Mark Chiew]]

---

## Introduction

Accelerated MRI depends on both the sampling pattern and the reconstruction. Variable-density random sampling [Lustig 2007] is the usual hand-designed choice. Learning the pattern jointly with the reconstructor was introduced by LOUPE [Bahadir 2020], which parameterises per-location sampling probabilities and relaxes the Bernoulli draw so the pattern trains by backpropagation. Extensions include multi-coil LOUPE [Zhang 2020], joint sampling-and-reconstruction with J-MoDL [Aggarwal 2020], learned sampling with model-based recovery [Huijben 2020], continuous and trajectory-based optimisation [Sherry 2020, Weiss 2021, Wang 2022], stabilised sampling preferences [Sun 2024] and variational-information-based sampling for 3D MRI [Alkan 2024].

A separate line of work chooses k-space lines sequentially, conditioned on what has been acquired: active acquisition [Zhang 2019], reinforcement-learning policies [Pineda 2020], greedy policy search [Bakker 2020] and end-to-end sequential sampling and reconstruction [Yin 2021]. These methods are expected to adapt to each subject. Whether they beat a well-trained static learned mask at matched acceleration, particularly with a strong multi-coil unrolled reconstructor, has received little direct comparison.

Here we train a single-step LOUPE sampler with a multi-coil variational network on fastMRI brain and knee at R=8. We then ask three questions: (1) how much does the learned mask add over variable density, (2) is anatomy-specific sampling worthwhile, and (3) do sequential or input-adaptive samplers add anything over a static learned pattern?

## Theory

Sampling probabilities are `P = N(σ(W))`, where `W` holds learnable logits, `σ` is a sigmoid with slope [[5]] and `N` rescales `P` to a mean of 1/R. [[Confirm the slope and normalisation against `activemri/models/`.]] The mask is drawn as `M = 1[P > U(0,1)]`, with a straight-through gradient [Bengio 2013] in the backward pass [[slope / relaxation: check the sampler code]]. The reconstructor is an unrolled variational network with a learned sensitivity-map network [Sriram 2020, Hammernik 2018] and a U-Net backbone [Ronneberger 2015]. Data consistency uses [[`where` / `multiply`: check `recon.dc_mode`]].

**Compared samplers** (all at R=8):
- (a) Random and variable-density masks.
- (b) Learned static mask (LOUPE).
- (c) Anatomy-conditioned LOUPE, with one mask per anatomy.
- (d) REINFORCE policy sampler, which draws masks and uses a score-function gradient.
- (e) PPO sequential sampler.
- (f) Convolutional sampler that takes the 10x10 centre k-space or the image as input.

## Methods

**Data:** fastMRI brain and knee multi-coil [Zbontar 2018]. Balanced split: 1350 training slices per anatomy (89 brain / 40 knee volumes), 344 validation slices per anatomy (23 brain / 10 knee volumes). [[test set? no held-out test split is recorded in the notes.]] R=8, 16 coils, 256x256, 10x10 fully sampled centre. [[Confirm the exact acceleration-vs-centre convention.]]

**Reconstruction:** VarNet, 12 cascades, 18 channels, sensitivity-estimation network [Sriram 2020]. Adam [Kingma 2015], batch size 16, lr [[1e-3 / 3e-4: runs differ]], 100 epochs with a step decay at epoch 90, seed 42.

**Evaluation:** PSNR and SSIM [Wang 2004] on background-masked magnitude images, reported per anatomy. [[Paired per-slice comparisons with bootstrap CIs; add seeds.]]

**Controls:** cross-mask evaluation (one run's mask with another run's reconstructor) to separate mask effects from reconstructor effects, and a label-swap test for the anatomy-conditioned model.

## Results (placeholders)

| Method (R=8) | Brain PSNR / SSIM | Knee PSNR / SSIM |
|---|---|---|
| Random | [[ ]] | [[ ]] |
| Variable density | [[ ]] | [[ ]] |
| LOUPE, shared mask, 100 ep | 36.54 / 0.9327 † | 34.13 / 0.9526 † |
| LOUPE, anatomy-conditioned, 100 ep | 36.72 / 0.9341 † | 34.35 / 0.9540 † |
| Conv sampler (centre k-space) | 36.28 / 0.9309 † | 34.07 / 0.9511 † |
| REINFORCE | [[ ]] | [[ ]] |
| PPO | [[ ]] | [[ ]] |

† Single seed. The shared and conditioned 100-epoch runs differ in learning rate (3e-4 vs 1e-3), so the +0.18 / +0.22 dB is not yet attributable to anatomy-specific masks. A matched rerun is needed. The conv-sampler row comes from a different run set (LOUPE reference uo8d68n0: 36.55 / 34.26), so it is not directly comparable with the other two LOUPE rows.

Findings to state once confirmed:
- The learned mask beats variable density by [[ ]] dB.
- Label swap: the knee mask is anatomy-specific (-0.24 dB with the brain mask), while the brain mask is not (-0.01 dB, CI spans 0).
- Adaptive samplers: per-slice mask correlation with the anatomy mean is 0.99 or higher, so the learned patterns are close to static. The conv sampler trails LOUPE by 0.2 to 0.27 dB. Part of this (about 0.1 to 0.2 dB) is mask quality and the rest is reconstructor co-adaptation.
- Policy samplers: PPO is indistinguishable from random in the early runs, and REINFORCE trails LOUPE by 0.38 dB on brain. [[Check against the final runs.]]

## Discussion and Conclusions

[[Core claim to settle: a single learned static pattern is a strong baseline at R=8, and apparent adaptivity mostly reflects a better static mask plus a co-adapted reconstructor.]] Caveats: single seed, co-adaptation of mask and reconstructor, a mismatch tax when the mask is swapped, [[no prospective acquisition]]. Future work: [[multi-seed, matched learning rate, 3D / SPACE sampling, prospective scanning]].

---

## Figures (plan)
1. Pipeline diagram: probabilistic mask -> straight-through draw -> sensitivity net + VarNet; the compared samplers.
2. Bar plot of PSNR / SSIM per anatomy per sampler, with paired CIs.
3. Learned probability maps: brain, knee, shared vs conditioned, next to variable density.
4. Example reconstructions with error maps [[pick slices by PSNR rank; the brain val volume AXT2_201_2010571 is a ~25 dB outlier, exclude or flag it]].
5. Adaptivity diagnostic: per-slice mask correlation, and the cross-mask matrix (mask vs reconstructor swap).

---

## References

Verified against your 2025 reference list unless marked **[verify]**.

1. Bahadir CD, Wang AQ, Dalca AV, Sabuncu MR. Deep-Learning-Based Optimization of the Under-Sampling Pattern in MRI. *IEEE Trans Comput Imaging* 6:1139-1152 (2020).
2. Yin T, Wu Z, Sun H, Dalca AV, Yue Y, Bouman KL. End-to-End Sequential Sampling and Reconstruction for MRI. *Proc Machine Learning for Health (MLHC)* (2021). arXiv:2105.06460. **[verify venue]**
3. Zhang J, et al. Extending LOUPE for K-Space Under-Sampling Pattern Optimization in Multi-coil MRI. In: *Machine Learning for Medical Image Reconstruction*, Springer, 91-101 (2020).
4. Aggarwal HK, Jacob M. J-MoDL: Joint Model-Based Deep Learning for Optimized Sampling and Reconstruction. *IEEE J Sel Top Signal Process* 14:1151-1162 (2020).
5. Huijben IAM, Veeling BS, van Sloun RJG. Learning Sampling and Model-Based Signal Recovery for Compressed Sensing MRI. *ICASSP* 8906-8910 (2020).
6. Sherry F, et al. Learning the Sampling Pattern for MRI. *IEEE Trans Med Imaging* 39:4310-4321 (2020).
7. Sun H, Tian C, Xiao J, Yang Y. Learn Stable MRI Under-Sampling Pattern With Decoupled Sampling Preference. *IEEE Trans Comput Imaging* 10:246-260 (2024).
8. Alkan C, et al. AutoSamp: Autoencoding k-space Sampling via Variational Information Maximization for 3D MRI. *IEEE Trans Med Imaging* (2024).
9. Weiss T, Senouf O, Vedula S, Michailovich O, Zibulevsky M, Bronstein A. PILOT: Physics-Informed Learned Optimized Trajectories for Accelerated MRI. *MELBA* (2021). **[verify]**
10. Wang G, Luo T, Nielsen JF, Noll DC, Fessler JA. B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI. *IEEE Trans Med Imaging* 41:2318-2330 (2022). **[verify]**
11. Zhang Z, Romero A, Muckley MJ, Vincent P, Yang L, Drozdzal M. Reducing Uncertainty in Undersampled MRI Reconstruction with Active Acquisition. *CVPR* 2049-2058 (2019).
12. Pineda L, Basu S, Romero A, Calandra R, Drozdzal M. Active MR k-space Sampling with Reinforcement Learning. *MICCAI* 23-33 (2020).
13. Bakker T, van Hoof H, Welling M. Experimental design for MRI by greedy policy search. *NeurIPS* (2020).
14. Lustig M, Donoho D, Pauly JM. Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magn Reson Med* 58:1182-1195 (2007).
15. Sriram A, et al. End-to-End Variational Networks for Accelerated MRI Reconstruction. *MICCAI* (2020). arXiv:2004.06688.
16. Hammernik K, et al. Learning a variational network for reconstruction of accelerated MRI data. *Magn Reson Med* 79:3055-3071 (2018).
17. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv:1505.04597 (2015).
18. Bengio Y, Léonard N, Courville A. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv:1308.3432 (2013).
19. Zbontar J, et al. fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv:1811.08839 (2018).
20. Uecker M, et al. ESPIRiT: an eigenvalue approach to autocalibrating parallel MRI. *Magn Reson Med* 71:990-1001 (2014). [[only if the final text cites ESPIRiT / PISCO maps]]
21. Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 (2014).
22. Wang Z, Simoncelli EP, Bovik AC. Multiscale structural similarity for image quality assessment. *Asilomar Conf. Signals, Systems & Computers* 1398-1402 (2003).

Open decisions: (1) headline claim, either "learned static ≈ adaptive" or "anatomy-specific masks help"; (2) a matched-lr, second-seed rerun to resolve the anatomy-mask confound; (3) whether to include the REINFORCE and PPO comparisons given the word limit.
