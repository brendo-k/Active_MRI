
[brain] 23 volumes, base 35.889 dB
| arm | vs base | vs its control |
|---|---|---|
| mask | -0.007 [-0.189, +0.195] | +0.292 [+0.090, +0.529] |
| mask_s4 | -0.021 [-0.202, +0.184] | +0.219 [+0.018, +0.475] |
| mask_s8 | +0.006 [-0.161, +0.183] | +0.214 [+0.017, +0.452] |
| grid | +0.211 [+0.095, +0.351] | +0.213 [+0.088, +0.354] |
| mask_ctrl | -0.299 [-0.405, -0.190] |  |
| mask_s4_ctrl | -0.240 [-0.348, -0.141] |  |
| mask_s8_ctrl | -0.208 [-0.320, -0.117] |  |
| grid_ctrl | -0.002 [-0.053, +0.047] |  |
| mask_in | +0.258 [+0.091, +0.454] |  |
| mask_s4_in | +0.030 [-0.152, +0.227] |  |
| mask_s8_in | +0.047 [-0.112, +0.223] |  |
| grid_in | +0.202 [+0.086, +0.332] |  |

[knee] 10 volumes, base 38.517 dB
| arm | vs base | vs its control |
|---|---|---|
| mask | -0.542 [-0.744, -0.346] | -0.305 [-0.461, -0.159] |
| mask_s4 | -0.747 [-1.054, -0.501] | -0.322 [-0.511, -0.148] |
| mask_s8 | -0.684 [-0.970, -0.457] | -0.282 [-0.462, -0.128] |
| grid | -0.389 [-0.577, -0.228] | -0.242 [-0.407, -0.107] |
| mask_ctrl | -0.237 [-0.331, -0.149] |  |
| mask_s4_ctrl | -0.425 [-0.596, -0.303] |  |
| mask_s8_ctrl | -0.402 [-0.575, -0.280] |  |
| grid_ctrl | -0.147 [-0.237, -0.082] |  |
| mask_in | -0.087 [-0.142, -0.030] |  |
| mask_s4_in | -0.333 [-0.387, -0.278] |  |
| mask_s8_in | -0.319 [-0.371, -0.268] |  |
| grid_in | -0.107 [-0.142, -0.075] |  |

[brain] smooth arms minus control: concentration, and without 2 volumes (--exclude, or scored on < 2 slices)
| arm | median | positive | top-5 share of sum | corr(base PSNR, gain) | mean without excluded |
|---|---|---|---|---|---|
| grid | +0.052 | 19/23 | 0.79 | -0.37 | +0.128 [+0.053, +0.221] |
| mask_s4 | -0.033 | 11/23 | 1.15 | -0.39 | +0.085 [-0.067, +0.259] |
| mask_s8 | -0.029 | 11/23 | 1.14 | -0.38 | +0.082 [-0.055, +0.235] |

[knee] smooth arms minus control: concentration, and without 0 volumes (--exclude, or scored on < 2 slices)
| arm | median | positive | top-5 share of sum | corr(base PSNR, gain) | mean without excluded |
|---|---|---|---|---|---|
| grid | -0.164 | 1/10 |  | -0.32 | -0.242 [-0.401, -0.113] |
| mask_s4 | -0.273 | 1/10 |  | -0.26 | -0.322 [-0.506, -0.153] |
| mask_s8 | -0.248 | 1/10 |  | -0.22 | -0.282 [-0.452, -0.132] |

Stage 1 (smooth held-out arm - control >= +0.05 dB, interval above 0): PASS {"brain": ["grid", "mask_s4", "mask_s8"], "knee": []}
