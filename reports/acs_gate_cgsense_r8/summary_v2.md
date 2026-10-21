
[brain] 23 volumes, base 35.889 dB
| arm | vs base | vs its control |
|---|---|---|
| mask | +0.166 [+0.077, +0.270] | +0.182 [+0.081, +0.305] |
| mask_s4 | +0.151 [+0.054, +0.266] | +0.173 [+0.066, +0.320] |
| mask_s8 | +0.146 [+0.055, +0.250] | +0.167 [+0.066, +0.301] |
| grid | +0.160 [+0.078, +0.256] | +0.157 [+0.068, +0.258] |
| mask_ctrl | -0.015 [-0.063, +0.029] |  |
| mask_s4_ctrl | -0.022 [-0.077, +0.025] |  |
| mask_s8_ctrl | -0.021 [-0.077, +0.023] |  |
| grid_ctrl | +0.003 [-0.041, +0.046] |  |
| mask_in | +0.224 [+0.139, +0.327] |  |
| mask_s4_in | +0.150 [+0.061, +0.252] |  |
| mask_s8_in | +0.140 [+0.058, +0.235] |  |
| grid_in | +0.155 [+0.073, +0.245] |  |

[knee] 10 volumes, base 38.517 dB
| arm | vs base | vs its control |
|---|---|---|
| mask | -0.133 [-0.247, -0.033] | -0.055 [-0.128, +0.011] |
| mask_s4 | -0.218 [-0.380, -0.093] | -0.073 [-0.160, +0.006] |
| mask_s8 | -0.205 [-0.350, -0.089] | -0.057 [-0.138, +0.009] |
| grid | -0.187 [-0.324, -0.079] | -0.053 [-0.120, +0.010] |
| mask_ctrl | -0.078 [-0.142, -0.029] |  |
| mask_s4_ctrl | -0.145 [-0.244, -0.076] |  |
| mask_s8_ctrl | -0.148 [-0.246, -0.077] |  |
| grid_ctrl | -0.134 [-0.240, -0.065] |  |
| mask_in | +0.010 [-0.016, +0.036] |  |
| mask_s4_in | -0.071 [-0.095, -0.047] |  |
| mask_s8_in | -0.080 [-0.104, -0.054] |  |
| grid_in | -0.060 [-0.086, -0.034] |  |

[brain] smooth arms minus control: concentration, and without 2 volumes (--exclude, or scored on < 2 slices)
| arm | median | positive | top-5 share of sum | corr(base PSNR, gain) | mean without excluded |
|---|---|---|---|---|---|
| grid | +0.063 | 19/23 | 0.75 | -0.31 | +0.091 [+0.044, +0.148] |
| mask_s4 | +0.030 | 19/23 | 0.84 | -0.33 | +0.092 [+0.032, +0.169] |
| mask_s8 | +0.041 | 19/23 | 0.83 | -0.33 | +0.089 [+0.032, +0.153] |

[knee] smooth arms minus control: concentration, and without 0 volumes (--exclude, or scored on < 2 slices)
| arm | median | positive | top-5 share of sum | corr(base PSNR, gain) | mean without excluded |
|---|---|---|---|---|---|
| grid | -0.025 | 3/10 |  | +0.10 | -0.053 [-0.120, +0.010] |
| mask_s4 | -0.047 | 4/10 |  | -0.24 | -0.073 [-0.159, +0.004] |
| mask_s8 | -0.044 | 4/10 |  | -0.18 | -0.057 [-0.131, +0.007] |

Stage 1 (smooth held-out arm - control >= +0.05 dB, interval above 0): PASS {"brain": ["grid", "mask_s4", "mask_s8"], "knee": []}
