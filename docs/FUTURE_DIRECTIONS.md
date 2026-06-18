# Future Directions for SeqMRI

## 1. Reconstructor Upgrades

- **Unrolled networks (VarNet / E2E-VarNet):** Replace the UNet reconstructor with a model-based unrolled architecture. CG-SENSE is already there as a non-learned baseline — an unrolled CG with learned regularization is the natural next step. This would likely give the biggest single improvement in reconstruction quality and provide better gradients to the sampler.
- **Diffusion-based or score-based reconstruction:** Use a pretrained diffusion prior as the reconstructor. The sequential sampling policy could query a score model's uncertainty to decide where to sample next — a strong fit for the existing uncertainty-guided framework.

## 2. Sampler Architecture

- **Transformer-based sampler:** The current sampler is an MLP/ResNet that flattens k-space features. A transformer over the set of candidate lines (or 2D patches) could capture long-range dependencies between acquired and unacquired measurements much better.
- **Multi-scale / coarse-to-fine sampling:** First pick broad regions, then refine within selected regions. The current single-scale approach can't express this naturally.
- **Variable-budget sampling:** Train a single model that generalizes across acceleration factors (e.g., condition on target sparsity) rather than training separate models per acceleration.

## 3. Training Methodology

- **Reinforcement learning:** REINFORCE/PPO with the reconstruction metric as reward avoids the straight-through estimator approximation. A clean RL setup with the current architecture could work well, especially for non-differentiable reconstructors.
- **Self-supervised / unsupervised training:** Remove the requirement for fully-sampled ground truth during training. Use consistency losses (SSDU-style) or equivariant constraints so the model can be trained on prospectively undersampled data.
- **Curriculum training over steps:** Start training with 1 step, then progressively add more sequential steps. This avoids the credit-assignment difficulty of long sequential rollouts early in training.

## 4. Data & Generalization

- **3D volumetric sampling:** Extend from 2D slice-by-slice to full 3D k-space sampling (k_x, k_y, k_z). The `line_constrained` option already handles 1D lines — the natural extension is 2D planes or arbitrary 3D trajectories.
- **Cross-anatomy generalization:** Training a single universal sampler (perhaps anatomy-conditioned) and measuring zero-shot transfer across brain/knee/prostate/breast.
- **Prospective validation:** Move from retrospective (simulated undersampling of fully-sampled data) to prospective experiments on a real scanner, where the model actually drives acquisition in real-time.

## 5. Clinical Utility

- **Task-driven sampling:** Instead of optimizing for image quality (SSIM/PSNR), optimize for downstream task performance — e.g., segmentation accuracy, lesion detection AUC, or radiologist reading time. (Additionally can optimize for segmentation mask based on localizer).
- **Uncertainty quantification for stopping criteria:** Use the uncertainty maps (already computed) to decide when to *stop* acquiring — adaptive scan time rather than fixed budget.
- **Patient-adaptive acceleration:** Let the model choose different total acceleration per patient based on scan complexity.

## 6. Engineering / Reproducibility

- **Hydra/config-based experiments:** Replace the argparse approach with Hydra or a YAML config system for better experiment tracking and sweeps.
- **Standardized benchmarking:** Evaluate against fastMRI leaderboard protocols and publish reproducible baselines.
- **ONNX/TorchScript export** for real-time deployment on scanner hardware.

## Priority Recommendation

The reconstructor is the weakest link — upgrading from UNet to an unrolled architecture (VarNet-style) would improve reconstruction quality substantially, and the better gradients flowing back through the reconstructor would also improve the learned sampling policy. After that, a transformer-based sampler would be the natural second step.
