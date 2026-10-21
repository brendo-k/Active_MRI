# Per-subject co-adaptation (R=8, LOUPE VarNet uo8d68n0)

Paired per-slice PSNR differences (dB), slice-weighted mean [95% CI over volumes]. Scored on the odd
half H of each validation volume; F = the even half. Typical = signal outside the sense support <= 30%.

Left out of the ceiling tables: file_brain_AXT2_201_2010571.h5

### Ground-truth ceiling, brain_block_param (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_param all | brain_block_param typical |
|---|---|---|
| mask (fit on F) - base | -0.154 [-0.205, -0.107] | -0.179 [-0.219, -0.127] |
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.244 [+0.204, +0.335] | +0.278 [+0.226, +0.360] |
| param (fit on F) - base | +0.002 [-0.004, +0.007] | +0.002 [-0.006, +0.008] |
| param_in (fit on F+H) - base | +0.005 [-0.003, +0.011] | +0.006 [-0.004, +0.012] |
| param_ctrl (other subjects) - base | -0.004 [-0.010, +0.002] | -0.004 [-0.010, +0.002] |
| param - param_ctrl (subject-specific) | +0.005 [-0.004, +0.013] | +0.006 [-0.004, +0.014] |
| param_in - mask_in (family vs free per-volume oracle) | -0.239 [-0.328, -0.201] | -0.272 [-0.358, -0.225] |
| grid (fit on F) - base | -0.056 [-0.139, +0.005] | -0.065 [-0.150, +0.001] |
| grid_in (fit on F+H) - base | -0.008 [-0.053, +0.025] | -0.010 [-0.058, +0.024] |
| grid_ctrl (other subjects) - base | -0.006 [-0.014, +0.001] | -0.007 [-0.015, +0.001] |
| grid - grid_ctrl (subject-specific) | -0.049 [-0.139, +0.014] | -0.058 [-0.157, +0.011] |
| grid_in - mask_in (family vs free per-volume oracle) | -0.252 [-0.336, -0.215] | -0.288 [-0.369, -0.245] |

brain_block_param: 22 volumes, 123 slices

### Per-volume family parameters, brain_block_param: reliability across 22 volumes

| parameter | mean F fit | sd across volumes (F fit) | corr(F fit, H fit) | corr(ctrl fit, H fit) |
|---|---|---|---|---|
| scale | -0.018 | 0.059 | +0.48 | -0.01 |
| anisotropy_a | +0.030 | 0.046 | +0.45 | +0.11 |
| anisotropy_b | -0.009 | 0.035 | +0.72 | -0.48 |
| tilt | -0.009 | 0.073 | +0.11 | +0.16 |

grid offset (centred on the mean over volumes): corr(F fit, H fit) within a volume +0.42, between volumes -0.01 (22 volumes)

### Ground-truth ceiling, knee_block_param (fit on F, score H; slice distance to F 3-22)

| contrast | knee_block_param all | knee_block_param typical |
|---|---|---|
| mask (fit on F) - base | -0.116 [-0.171, -0.059] | -0.116 [-0.173, -0.062] |
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.170 [+0.099, +0.226] | +0.170 [+0.099, +0.228] |
| param (fit on F) - base | -0.040 [-0.066, -0.014] | -0.040 [-0.065, -0.014] |
| param_in (fit on F+H) - base | +0.005 [-0.013, +0.029] | +0.005 [-0.014, +0.030] |
| param_ctrl (other subjects) - base | -0.016 [-0.032, +0.001] | -0.016 [-0.032, -0.000] |
| param - param_ctrl (subject-specific) | -0.024 [-0.060, +0.007] | -0.024 [-0.059, +0.008] |
| param_in - mask_in (family vs free per-volume oracle) | -0.165 [-0.230, -0.092] | -0.165 [-0.226, -0.085] |
| grid (fit on F) - base | -0.084 [-0.155, -0.024] | -0.084 [-0.154, -0.020] |
| grid_in (fit on F+H) - base | +0.016 [-0.009, +0.045] | +0.016 [-0.009, +0.043] |
| grid_ctrl (other subjects) - base | -0.021 [-0.034, -0.008] | -0.021 [-0.034, -0.009] |
| grid - grid_ctrl (subject-specific) | -0.063 [-0.142, +0.011] | -0.063 [-0.141, +0.013] |
| grid_in - mask_in (family vs free per-volume oracle) | -0.154 [-0.221, -0.073] | -0.154 [-0.221, -0.072] |

knee_block_param: 10 volumes, 80 slices

### Per-volume family parameters, knee_block_param: reliability across 10 volumes

| parameter | mean F fit | sd across volumes (F fit) | corr(F fit, H fit) | corr(ctrl fit, H fit) |
|---|---|---|---|---|
| scale | -0.038 | 0.038 | +0.43 | -0.41 |
| anisotropy_a | -0.069 | 0.060 | +0.58 | -0.43 |
| anisotropy_b | -0.031 | 0.092 | +0.66 | +0.72 |
| tilt | +0.020 | 0.091 | +0.52 | -0.26 |

grid offset (centred on the mean over volumes): corr(F fit, H fit) within a volume +0.56, between volumes -0.06 (10 volumes)

Mean correlation of the fitted per-volume probability map with the LOUPE map: brain_block_param: mask 0.7050, mask_in 0.7805; knee_block_param: mask 0.8197, mask_in 0.8607

### Per-volume mask oracle vs fit length, brain (same volumes as brain_block_maskin1500)

| fit steps | folder | arm | volumes | PSNR vs base (dB) |
|---|---|---|---|---|
| 150 | brain_block_full | mask_in | 4 of 4 | +0.242 |
| 600 | brain_block_maskin600 | mask_in | 4 of 4 | +0.384 |
| 1500 | brain_block_maskin1500 | mask_in | 4 of 4 | +0.527 |
| 1500 | brain_block_mask1500 | mask | 4 of 4 | -0.200 |

### Per-volume mask oracle vs fit length, knee (same volumes as knee_block_maskin1500)

| fit steps | folder | arm | volumes | PSNR vs base (dB) |
|---|---|---|---|---|
| 150 | knee_block_full | mask_in | 2 of 3 | +0.052 |
| 600 | knee_block_maskin600 | mask_in | 3 of 3 | +0.213 |
| 1500 | knee_block_maskin1500 | mask_in | 3 of 3 | +0.323 |
| 1500 | knee_block_mask1500 | mask | 3 of 3 | -0.198 |

### Self-supervised per-slice adaptation (PSNR change vs base, dB, mean over slices)

| method | loss | lr | rho | sense net | slices | @10 | @25 | @50 | @100 | @200 | @500 | @1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| echo | acquired | 1e-05 |  | frozen | 2 |  | -0.317 | -0.297 | -0.374 | -0.437 |  |  |
| echo | acquired | 3e-05 |  | frozen | 2 |  | -0.378 | -0.421 | -0.423 | -0.456 |  |  |
| echo | acquired | 0.0001 |  | frozen | 2 |  | -0.557 | -0.466 | -0.518 | -0.585 | -1.178 | -2.097 |
| echo | acquired | 0.0003 |  | frozen | 2 |  | -0.609 | -0.592 | -0.685 | -0.988 | -2.065 | -3.444 |
| echo | heldout | 1e-05 | 0.2 | frozen | 2 |  | -0.067 | -0.195 | -0.404 | -0.861 | -1.660 |  |
| echo | heldout | 0.0001 | 0.2 | frozen | 2 |  | -0.863 | -2.276 | -2.687 | -2.915 | -3.113 |  |
| echo | unseen | 1e-05 |  | frozen | 2 |  | -0.199 | -0.140 | -0.207 | -0.290 |  |  |
| echo | unseen | 1e-05 |  | frozen | 4 |  |  |  |  |  |  |  |
| echo | unseen | 1e-05 |  | frozen | 4 |  |  |  |  |  |  |  |
| echo | unseen | 3e-05 |  | frozen | 2 |  | -0.121 | -0.258 | -0.309 | -0.386 |  |  |
| ssdu | coil | 3e-06 | 0.4 | frozen | 2 | -0.020 | -0.207 | -1.095 | -3.660 |  |  |  |
| ssdu | coil | 1e-05 | 0.4 | frozen | 2 | -0.092 | -1.347 | -5.247 | -6.358 |  |  |  |
| ssdu | image | 3e-06 | 0.4 | frozen | 2 | -0.021 | -0.210 | -1.072 | -3.335 |  |  |  |
| ssdu | image | 1e-05 | 0.4 | trained | 2 |  | -1.380 | -5.044 | -6.873 | -6.723 |  |  |

