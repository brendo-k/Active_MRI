# Per-subject co-adaptation (R=8, LOUPE VarNet uo8d68n0)

Paired per-slice PSNR differences (dB), slice-weighted mean [95% CI over volumes]. Scored on the odd
half H of each validation volume; F = the even half. Typical = signal outside the sense support <= 30%.

### Ground-truth ceiling, brain_block_full (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_full all | brain_block_full typical |
|---|---|---|
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.218 [+0.160, +0.272] | +0.256 [+0.184, +0.297] |
| recon@25 - base | +0.021 [+0.010, +0.033] | +0.024 [+0.010, +0.035] |
| recon_generic@25 - base (continued training) | -0.001 [-0.012, +0.009] | -0.001 [-0.012, +0.012] |
| recon@25 - recon_generic@25 (subject vs continued training) | +0.022 [+0.009, +0.036] | +0.025 [+0.010, +0.038] |
| joint@25 - base | +0.045 [+0.023, +0.068] | +0.052 [+0.025, +0.073] |
| joint@25 - recon@25 (mask gain on a co-adapted VarNet) | +0.024 [+0.008, +0.043] | +0.028 [+0.008, +0.045] |
| recon@50 - base | +0.016 [-0.001, +0.035] | +0.018 [+0.001, +0.039] |
| recon_generic@50 - base (continued training) | -0.018 [-0.032, -0.005] | -0.022 [-0.035, -0.007] |
| recon@50 - recon_generic@50 (subject vs continued training) | +0.035 [+0.024, +0.049] | +0.041 [+0.028, +0.052] |
| joint@50 - base | +0.029 [-0.002, +0.063] | +0.033 [-0.004, +0.069] |
| joint@50 - recon@50 (mask gain on a co-adapted VarNet) | +0.013 [-0.012, +0.039] | +0.014 [-0.014, +0.040] |
| recon@100 - base | -0.013 [-0.034, +0.010] | -0.016 [-0.037, +0.012] |
| recon_generic@100 - base (continued training) | -0.012 [-0.029, +0.004] | -0.015 [-0.030, +0.005] |
| recon@100 - recon_generic@100 (subject vs continued training) | -0.001 [-0.022, +0.023] | -0.001 [-0.025, +0.023] |
| joint@100 - base | -0.021 [-0.061, +0.021] | -0.027 [-0.070, +0.024] |
| joint@100 - recon@100 (mask gain on a co-adapted VarNet) | -0.008 [-0.039, +0.025] | -0.011 [-0.046, +0.025] |
| recon@200 - base | -0.098 [-0.135, -0.053] | -0.115 [-0.152, -0.062] |
| recon_generic@200 - base (continued training) | -0.003 [-0.013, +0.006] | -0.005 [-0.015, +0.005] |
| recon@200 - recon_generic@200 (subject vs continued training) | -0.094 [-0.136, -0.049] | -0.111 [-0.148, -0.055] |
| joint@200 - base | -0.140 [-0.206, -0.074] | -0.168 [-0.226, -0.092] |
| joint@200 - recon@200 (mask gain on a co-adapted VarNet) | -0.042 [-0.082, +0.003] | -0.052 [-0.093, -0.005] |
| recon@400 - base | -0.387 [-0.492, -0.283] | -0.457 [-0.533, -0.332] |
| recon_generic@400 - base (continued training) | -0.012 [-0.035, +0.008] | -0.014 [-0.035, +0.010] |
| recon@400 - recon_generic@400 (subject vs continued training) | -0.376 [-0.467, -0.281] | -0.443 [-0.509, -0.328] |
| joint@400 - base | -0.484 [-0.610, -0.348] | -0.577 [-0.669, -0.414] |
| joint@400 - recon@400 (mask gain on a co-adapted VarNet) | -0.097 [-0.151, -0.040] | -0.119 [-0.168, -0.054] |

brain_block_full: 16 volumes, 95 slices

### Ground-truth ceiling, knee_block_full (fit on F, score H; slice distance to F 3-22)

| contrast | knee_block_full all | knee_block_full typical |
|---|---|---|
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.165 [+0.052, +0.244] | +0.165 [+0.049, +0.244] |
| recon@25 - base | -0.018 [-0.060, +0.017] | -0.018 [-0.055, +0.019] |
| recon_generic@25 - base (continued training) | -0.014 [-0.035, +0.006] | -0.014 [-0.035, +0.006] |
| recon@25 - recon_generic@25 (subject vs continued training) | -0.004 [-0.040, +0.033] | -0.004 [-0.041, +0.033] |
| joint@25 - base | -0.012 [-0.040, +0.016] | -0.012 [-0.039, +0.017] |
| joint@25 - recon@25 (mask gain on a co-adapted VarNet) | +0.006 [-0.026, +0.041] | +0.006 [-0.025, +0.041] |
| recon@50 - base | -0.006 [-0.054, +0.043] | -0.006 [-0.054, +0.043] |
| recon_generic@50 - base (continued training) | -0.029 [-0.064, +0.008] | -0.029 [-0.065, +0.008] |
| recon@50 - recon_generic@50 (subject vs continued training) | +0.024 [-0.041, +0.078] | +0.024 [-0.041, +0.078] |
| joint@50 - base | -0.020 [-0.068, +0.027] | -0.020 [-0.069, +0.029] |
| joint@50 - recon@50 (mask gain on a co-adapted VarNet) | -0.014 [-0.066, +0.040] | -0.014 [-0.064, +0.045] |
| recon@100 - base | -0.025 [-0.068, +0.018] | -0.025 [-0.065, +0.018] |
| recon_generic@100 - base (continued training) | +0.012 [-0.014, +0.039] | +0.012 [-0.014, +0.037] |
| recon@100 - recon_generic@100 (subject vs continued training) | -0.037 [-0.087, +0.001] | -0.037 [-0.087, +0.001] |
| joint@100 - base | -0.086 [-0.150, -0.022] | -0.086 [-0.150, -0.022] |
| joint@100 - recon@100 (mask gain on a co-adapted VarNet) | -0.062 [-0.124, +0.017] | -0.062 [-0.121, +0.017] |
| recon@200 - base | -0.094 [-0.141, -0.051] | -0.094 [-0.141, -0.051] |
| recon_generic@200 - base (continued training) | -0.045 [-0.076, -0.010] | -0.045 [-0.076, -0.011] |
| recon@200 - recon_generic@200 (subject vs continued training) | -0.048 [-0.113, +0.018] | -0.048 [-0.111, +0.018] |
| joint@200 - base | -0.207 [-0.279, -0.139] | -0.207 [-0.283, -0.139] |
| joint@200 - recon@200 (mask gain on a co-adapted VarNet) | -0.113 [-0.197, -0.006] | -0.113 [-0.197, -0.016] |
| recon@400 - base | -0.263 [-0.373, -0.192] | -0.263 [-0.372, -0.191] |
| recon_generic@400 - base (continued training) | +0.029 [+0.003, +0.056] | +0.029 [+0.003, +0.058] |
| recon@400 - recon_generic@400 (subject vs continued training) | -0.292 [-0.407, -0.204] | -0.292 [-0.406, -0.204] |
| joint@400 - base | -0.436 [-0.505, -0.357] | -0.436 [-0.510, -0.355] |
| joint@400 - recon@400 (mask gain on a co-adapted VarNet) | -0.173 [-0.271, -0.062] | -0.173 [-0.271, -0.064] |

knee_block_full: 6 volumes, 48 slices

### Ground-truth ceiling, brain_block_maskin600 (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_maskin600 all | brain_block_maskin600 typical |
|---|---|---|
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.384 [+0.167, +0.558] | +0.384 [+0.167, +0.558] |
| recon@25 - base | +0.021 [-0.008, +0.055] | +0.021 [-0.008, +0.055] |
| recon_on_mask_in@25 - base | +0.408 [+0.177, +0.596] | +0.408 [+0.177, +0.596] |
| recon_on_mask_in@25 - mask_in (VarNet gain on the oracle mask) | +0.024 [-0.010, +0.068] | +0.024 [-0.010, +0.068] |
| recon@50 - base | +0.012 [-0.031, +0.069] | +0.012 [-0.031, +0.069] |
| recon_on_mask_in@50 - base | +0.419 [+0.210, +0.608] | +0.419 [+0.210, +0.608] |
| recon_on_mask_in@50 - mask_in (VarNet gain on the oracle mask) | +0.035 [+0.007, +0.077] | +0.035 [+0.007, +0.077] |
| recon@100 - base | -0.035 [-0.075, +0.025] | -0.035 [-0.075, +0.025] |
| recon_on_mask_in@100 - base | +0.385 [+0.201, +0.564] | +0.385 [+0.201, +0.564] |
| recon_on_mask_in@100 - mask_in (VarNet gain on the oracle mask) | +0.001 [-0.040, +0.042] | +0.001 [-0.040, +0.042] |

brain_block_maskin600: 4 volumes, 24 slices

### Ground-truth ceiling, knee_block_maskin600 (fit on F, score H; slice distance to F 3-20)

| contrast | knee_block_maskin600 all | knee_block_maskin600 typical |
|---|---|---|
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.213 [-0.032, +0.436] | +0.213 [-0.032, +0.436] |
| recon@25 - base | +0.017 [-0.008, +0.037] | +0.017 [-0.008, +0.037] |
| recon_on_mask_in@25 - base | +0.270 [+0.017, +0.473] | +0.270 [+0.017, +0.473] |
| recon_on_mask_in@25 - mask_in (VarNet gain on the oracle mask) | +0.057 [+0.037, +0.085] | +0.057 [+0.037, +0.085] |
| recon@50 - base | +0.113 [-0.003, +0.252] | +0.113 [-0.003, +0.252] |
| recon_on_mask_in@50 - base | +0.363 [+0.047, +0.579] | +0.363 [+0.047, +0.579] |
| recon_on_mask_in@50 - mask_in (VarNet gain on the oracle mask) | +0.150 [+0.027, +0.344] | +0.150 [+0.027, +0.344] |
| recon@100 - base | +0.077 [-0.041, +0.214] | +0.077 [-0.041, +0.214] |
| recon_on_mask_in@100 - base | +0.362 [+0.000, +0.657] | +0.362 [+0.000, +0.657] |
| recon_on_mask_in@100 - mask_in (VarNet gain on the oracle mask) | +0.149 [-0.009, +0.422] | +0.149 [-0.009, +0.422] |

knee_block_maskin600: 3 volumes, 24 slices

### Ground-truth ceiling, brain_block_maskin1500 (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_maskin1500 all | brain_block_maskin1500 typical |
|---|---|---|
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.527 [+0.335, +0.704] | +0.527 [+0.335, +0.738] |

brain_block_maskin1500: 4 volumes, 24 slices

### Ground-truth ceiling, knee_block_maskin1500 (fit on F, score H; slice distance to F 3-20)

| contrast | knee_block_maskin1500 all | knee_block_maskin1500 typical |
|---|---|---|
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.323 [+0.080, +0.577] | +0.323 [+0.080, +0.577] |

knee_block_maskin1500: 3 volumes, 24 slices

### Ground-truth ceiling, brain_block_mask1500 (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_mask1500 all | brain_block_mask1500 typical |
|---|---|---|
| mask (fit on F) - base | -0.200 [-0.349, -0.022] | -0.200 [-0.349, -0.022] |

brain_block_mask1500: 4 volumes, 24 slices

### Ground-truth ceiling, knee_block_mask1500 (fit on F, score H; slice distance to F 3-20)

| contrast | knee_block_mask1500 all | knee_block_mask1500 typical |
|---|---|---|
| mask (fit on F) - base | -0.198 [-0.266, -0.124] | -0.198 [-0.266, -0.124] |

knee_block_mask1500: 3 volumes, 24 slices

### Ground-truth ceiling, brain (fit on F, score H)

| contrast | brain all | brain typical |
|---|---|---|
| mask (fit on F) - base | -0.099 [-0.172, -0.020] | -0.103 [-0.178, -0.017] |
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.235 [+0.171, +0.288] | +0.242 [+0.175, +0.305] |
| mask_ctrl (other subjects) - base | -0.147 [-0.173, -0.125] | -0.152 [-0.176, -0.128] |
| mask - mask_ctrl (subject-specific) | +0.048 [-0.038, +0.129] | +0.049 [-0.030, +0.138] |
| recon@50 - base | +0.129 [+0.088, +0.178] | +0.133 [+0.092, +0.196] |
| recon_ctrl@50 - base | -0.053 [-0.068, -0.037] | -0.055 [-0.071, -0.037] |
| recon@50 - recon_ctrl@50 (subject-specific) | +0.182 [+0.142, +0.229] | +0.188 [+0.144, +0.252] |
| joint@50 - base | +0.188 [+0.119, +0.294] | +0.194 [+0.121, +0.322] |
| joint@50 - joint_ctrl@50 (subject-specific) | +0.301 [+0.229, +0.395] | +0.311 [+0.234, +0.427] |
| joint@50 - recon@50 (mask gain on a co-adapted VarNet) | +0.060 [+0.022, +0.114] | +0.062 [+0.021, +0.124] |
| recon@100 - base | +0.154 [+0.108, +0.213] | +0.159 [+0.111, +0.230] |
| recon_ctrl@100 - base | -0.119 [-0.153, -0.088] | -0.123 [-0.156, -0.092] |
| recon@100 - recon_ctrl@100 (subject-specific) | +0.274 [+0.218, +0.341] | +0.283 [+0.225, +0.362] |
| joint@100 - base | +0.200 [+0.114, +0.313] | +0.206 [+0.117, +0.343] |
| joint@100 - joint_ctrl@100 (subject-specific) | +0.436 [+0.338, +0.550] | +0.450 [+0.344, +0.593] |
| joint@100 - recon@100 (mask gain on a co-adapted VarNet) | +0.045 [-0.003, +0.109] | +0.047 [-0.003, +0.122] |
| recon@200 - base | +0.141 [+0.086, +0.213] | +0.146 [+0.089, +0.230] |
| recon_ctrl@200 - base | -0.258 [-0.322, -0.201] | -0.266 [-0.328, -0.205] |
| recon@200 - recon_ctrl@200 (subject-specific) | +0.400 [+0.326, +0.483] | +0.412 [+0.335, +0.505] |
| joint@200 - base | +0.179 [+0.076, +0.319] | +0.185 [+0.080, +0.346] |
| joint@200 - joint_ctrl@200 (subject-specific) | +0.600 [+0.484, +0.732] | +0.619 [+0.495, +0.771] |
| joint@200 - recon@200 (mask gain on a co-adapted VarNet) | +0.038 [-0.026, +0.112] | +0.039 [-0.022, +0.123] |
| recon@400 - base | -0.034 [-0.103, +0.051] | -0.034 [-0.108, +0.065] |
| recon_ctrl@400 - base | -0.662 [-0.812, -0.505] | -0.683 [-0.822, -0.528] |
| recon@400 - recon_ctrl@400 (subject-specific) | +0.628 [+0.507, +0.773] | +0.649 [+0.523, +0.791] |
| joint@400 - base | +0.021 [-0.113, +0.184] | +0.022 [-0.118, +0.212] |
| joint@400 - joint_ctrl@400 (subject-specific) | +0.881 [+0.704, +1.060] | +0.910 [+0.722, +1.102] |
| joint@400 - recon@400 (mask gain on a co-adapted VarNet) | +0.055 [-0.020, +0.138] | +0.056 [-0.021, +0.156] |

brain: 12 volumes, 92 slices

### Ground-truth ceiling, knee (fit on F, score H)

| contrast | knee all | knee typical |
|---|---|---|
| mask (fit on F) - base | -0.027 [-0.027, -0.027] | -0.027 [-0.027, -0.027] |
| mask_in (per-volume mask oracle, fit on F+H) - base | +0.124 [+0.124, +0.124] | +0.124 [+0.124, +0.124] |
| mask_ctrl (other subjects) - base | -0.047 [-0.047, -0.047] | -0.047 [-0.047, -0.047] |
| mask - mask_ctrl (subject-specific) | +0.020 [+0.020, +0.020] | +0.020 [+0.020, +0.020] |
| recon@50 - base | +0.767 [+0.767, +0.767] | +0.767 [+0.767, +0.767] |
| recon_ctrl@50 - base | -0.383 [-0.383, -0.383] | -0.383 [-0.383, -0.383] |
| recon@50 - recon_ctrl@50 (subject-specific) | +1.150 [+1.150, +1.150] | +1.150 [+1.150, +1.150] |
| joint@50 - base | +0.817 [+0.817, +0.817] | +0.817 [+0.817, +0.817] |
| joint@50 - joint_ctrl@50 (subject-specific) | +1.250 [+1.250, +1.250] | +1.250 [+1.250, +1.250] |
| joint@50 - recon@50 (mask gain on a co-adapted VarNet) | +0.050 [+0.050, +0.050] | +0.050 [+0.050, +0.050] |
| recon@100 - base | +0.882 [+0.882, +0.882] | +0.882 [+0.882, +0.882] |
| recon_ctrl@100 - base | -0.763 [-0.763, -0.763] | -0.763 [-0.763, -0.763] |
| recon@100 - recon_ctrl@100 (subject-specific) | +1.645 [+1.645, +1.645] | +1.645 [+1.645, +1.645] |
| joint@100 - base | +0.971 [+0.971, +0.971] | +0.971 [+0.971, +0.971] |
| joint@100 - joint_ctrl@100 (subject-specific) | +1.828 [+1.828, +1.828] | +1.828 [+1.828, +1.828] |
| joint@100 - recon@100 (mask gain on a co-adapted VarNet) | +0.089 [+0.089, +0.089] | +0.089 [+0.089, +0.089] |
| recon@200 - base | +0.968 [+0.968, +0.968] | +0.968 [+0.968, +0.968] |
| recon_ctrl@200 - base | -1.119 [-1.119, -1.119] | -1.119 [-1.119, -1.119] |
| recon@200 - recon_ctrl@200 (subject-specific) | +2.087 [+2.087, +2.087] | +2.087 [+2.087, +2.087] |
| joint@200 - base | +1.053 [+1.053, +1.053] | +1.053 [+1.053, +1.053] |
| joint@200 - joint_ctrl@200 (subject-specific) | +2.291 [+2.291, +2.291] | +2.291 [+2.291, +2.291] |
| joint@200 - recon@200 (mask gain on a co-adapted VarNet) | +0.085 [+0.085, +0.085] | +0.085 [+0.085, +0.085] |
| recon@400 - base | +1.050 [+1.050, +1.050] | +1.050 [+1.050, +1.050] |
| recon_ctrl@400 - base | -1.452 [-1.452, -1.452] | -1.452 [-1.452, -1.452] |
| recon@400 - recon_ctrl@400 (subject-specific) | +2.503 [+2.503, +2.503] | +2.503 [+2.503, +2.503] |
| joint@400 - base | +1.148 [+1.148, +1.148] | +1.148 [+1.148, +1.148] |
| joint@400 - joint_ctrl@400 (subject-specific) | +2.771 [+2.771, +2.771] | +2.771 [+2.771, +2.771] |
| joint@400 - recon@400 (mask gain on a co-adapted VarNet) | +0.098 [+0.098, +0.098] | +0.098 [+0.098, +0.098] |

knee: 1 volumes, 8 slices

### Ground-truth ceiling, knee_block (fit on F, score H; slice distance to F 3-20)

| contrast | knee_block all | knee_block typical |
|---|---|---|
| recon@50 - base | +0.124 [-0.019, +0.267] | +0.124 [-0.019, +0.267] |
| recon_ctrl@50 - base | -0.405 [-0.715, -0.095] | -0.405 [-0.715, -0.095] |
| recon@50 - recon_ctrl@50 (subject-specific) | +0.529 [+0.076, +0.982] | +0.529 [+0.076, +0.982] |
| recon@100 - base | +0.093 [-0.049, +0.236] | +0.093 [-0.049, +0.236] |
| recon_ctrl@100 - base | -0.733 [-1.327, -0.139] | -0.733 [-1.327, -0.139] |
| recon@100 - recon_ctrl@100 (subject-specific) | +0.826 [+0.089, +1.563] | +0.826 [+0.089, +1.563] |
| recon@200 - base | -0.007 [-0.118, +0.105] | -0.007 [-0.118, +0.105] |
| recon_ctrl@200 - base | -1.074 [-1.947, -0.200] | -1.074 [-1.947, -0.200] |
| recon@200 - recon_ctrl@200 (subject-specific) | +1.067 [+0.082, +2.052] | +1.067 [+0.082, +2.052] |
| recon@400 - base | -0.294 [-0.347, -0.240] | -0.294 [-0.347, -0.240] |
| recon_ctrl@400 - base | -1.242 [-2.200, -0.284] | -1.242 [-2.200, -0.284] |
| recon@400 - recon_ctrl@400 (subject-specific) | +0.949 [+0.045, +1.853] | +0.949 [+0.045, +1.853] |

knee_block: 2 volumes, 16 slices

### Ground-truth ceiling, brain_block_fresh (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_fresh all | brain_block_fresh typical |
|---|---|---|
| recon@25 - base | +0.070 [+0.070, +0.070] | +0.070 [+0.070, +0.070] |
| recon_generic@25 - base (continued training) | -0.021 [-0.021, -0.021] | -0.021 [-0.021, -0.021] |
| recon@25 - recon_generic@25 (subject vs continued training) | +0.091 [+0.091, +0.091] | +0.091 [+0.091, +0.091] |
| recon@50 - base | +0.095 [+0.095, +0.095] | +0.095 [+0.095, +0.095] |
| recon_generic@50 - base (continued training) | -0.013 [-0.013, -0.013] | -0.013 [-0.013, -0.013] |
| recon@50 - recon_generic@50 (subject vs continued training) | +0.108 [+0.108, +0.108] | +0.108 [+0.108, +0.108] |
| recon@100 - base | +0.053 [+0.053, +0.053] | +0.053 [+0.053, +0.053] |
| recon_generic@100 - base (continued training) | -0.058 [-0.058, -0.058] | -0.058 [-0.058, -0.058] |
| recon@100 - recon_generic@100 (subject vs continued training) | +0.112 [+0.112, +0.112] | +0.112 [+0.112, +0.112] |
| recon@200 - base | -0.060 [-0.060, -0.060] | -0.060 [-0.060, -0.060] |
| recon_generic@200 - base (continued training) | +0.030 [+0.030, +0.030] | +0.030 [+0.030, +0.030] |
| recon@200 - recon_generic@200 (subject vs continued training) | -0.090 [-0.090, -0.090] | -0.090 [-0.090, -0.090] |
| recon@400 - base | -0.387 [-0.387, -0.387] | -0.387 [-0.387, -0.387] |
| recon_generic@400 - base (continued training) | -0.035 [-0.035, -0.035] | -0.035 [-0.035, -0.035] |
| recon@400 - recon_generic@400 (subject vs continued training) | -0.352 [-0.352, -0.352] | -0.352 [-0.352, -0.352] |

brain_block_fresh: 1 volumes, 6 slices

### Ground-truth ceiling, brain_block_adam (fit on F, score H; slice distance to F 3-8)

| contrast | brain_block_adam all | brain_block_adam typical |
|---|---|---|
| recon@25 - base | +0.080 [+0.080, +0.080] | +0.080 [+0.080, +0.080] |
| recon_generic@25 - base (continued training) | -0.040 [-0.040, -0.040] | -0.040 [-0.040, -0.040] |
| recon@25 - recon_generic@25 (subject vs continued training) | +0.120 [+0.120, +0.120] | +0.120 [+0.120, +0.120] |
| recon@50 - base | +0.070 [+0.070, +0.070] | +0.070 [+0.070, +0.070] |
| recon_generic@50 - base (continued training) | -0.015 [-0.015, -0.015] | -0.015 [-0.015, -0.015] |
| recon@50 - recon_generic@50 (subject vs continued training) | +0.086 [+0.086, +0.086] | +0.086 [+0.086, +0.086] |
| recon@100 - base | +0.022 [+0.022, +0.022] | +0.022 [+0.022, +0.022] |
| recon_generic@100 - base (continued training) | -0.093 [-0.093, -0.093] | -0.093 [-0.093, -0.093] |
| recon@100 - recon_generic@100 (subject vs continued training) | +0.115 [+0.115, +0.115] | +0.115 [+0.115, +0.115] |
| recon@200 - base | -0.115 [-0.115, -0.115] | -0.115 [-0.115, -0.115] |
| recon_generic@200 - base (continued training) | +0.018 [+0.018, +0.018] | +0.018 [+0.018, +0.018] |
| recon@200 - recon_generic@200 (subject vs continued training) | -0.133 [-0.133, -0.133] | -0.133 [-0.133, -0.133] |
| recon@400 - base | -0.544 [-0.544, -0.544] | -0.544 [-0.544, -0.544] |
| recon_generic@400 - base (continued training) | -0.056 [-0.056, -0.056] | -0.056 [-0.056, -0.056] |
| recon@400 - recon_generic@400 (subject vs continued training) | -0.488 [-0.488, -0.488] | -0.488 [-0.488, -0.488] |

brain_block_adam: 1 volumes, 6 slices

### Ground-truth ceiling, knee_block_fresh (fit on F, score H; slice distance to F 3-20)

| contrast | knee_block_fresh all | knee_block_fresh typical |
|---|---|---|
| recon@25 - base | +0.022 [+0.022, +0.022] | +0.022 [+0.022, +0.022] |
| recon_generic@25 - base (continued training) | -0.061 [-0.061, -0.061] | -0.061 [-0.061, -0.061] |
| recon@25 - recon_generic@25 (subject vs continued training) | +0.083 [+0.083, +0.083] | +0.083 [+0.083, +0.083] |
| recon@50 - base | +0.253 [+0.253, +0.253] | +0.253 [+0.253, +0.253] |
| recon_generic@50 - base (continued training) | -0.169 [-0.169, -0.169] | -0.169 [-0.169, -0.169] |
| recon@50 - recon_generic@50 (subject vs continued training) | +0.422 [+0.422, +0.422] | +0.422 [+0.422, +0.422] |
| recon@100 - base | +0.216 [+0.216, +0.216] | +0.216 [+0.216, +0.216] |
| recon_generic@100 - base (continued training) | -0.170 [-0.170, -0.170] | -0.170 [-0.170, -0.170] |
| recon@100 - recon_generic@100 (subject vs continued training) | +0.386 [+0.386, +0.386] | +0.386 [+0.386, +0.386] |
| recon@200 - base | +0.143 [+0.143, +0.143] | +0.143 [+0.143, +0.143] |
| recon_generic@200 - base (continued training) | -0.060 [-0.060, -0.060] | -0.060 [-0.060, -0.060] |
| recon@200 - recon_generic@200 (subject vs continued training) | +0.203 [+0.203, +0.203] | +0.203 [+0.203, +0.203] |
| recon@400 - base | -0.281 [-0.281, -0.281] | -0.281 [-0.281, -0.281] |
| recon_generic@400 - base (continued training) | -0.173 [-0.173, -0.173] | -0.173 [-0.173, -0.173] |
| recon@400 - recon_generic@400 (subject vs continued training) | -0.108 [-0.108, -0.108] | -0.108 [-0.108, -0.108] |

knee_block_fresh: 1 volumes, 8 slices

### Ground-truth ceiling, knee_block_adam (fit on F, score H; slice distance to F 3-20)

| contrast | knee_block_adam all | knee_block_adam typical |
|---|---|---|
| recon@25 - base | +0.287 [+0.287, +0.287] | +0.287 [+0.287, +0.287] |
| recon_generic@25 - base (continued training) | -0.179 [-0.179, -0.179] | -0.179 [-0.179, -0.179] |
| recon@25 - recon_generic@25 (subject vs continued training) | +0.466 [+0.466, +0.466] | +0.466 [+0.466, +0.466] |
| recon@50 - base | +0.303 [+0.303, +0.303] | +0.303 [+0.303, +0.303] |
| recon_generic@50 - base (continued training) | -0.160 [-0.160, -0.160] | -0.160 [-0.160, -0.160] |
| recon@50 - recon_generic@50 (subject vs continued training) | +0.464 [+0.464, +0.464] | +0.464 [+0.464, +0.464] |
| recon@100 - base | +0.196 [+0.196, +0.196] | +0.196 [+0.196, +0.196] |
| recon_generic@100 - base (continued training) | -0.127 [-0.127, -0.127] | -0.127 [-0.127, -0.127] |
| recon@100 - recon_generic@100 (subject vs continued training) | +0.322 [+0.322, +0.322] | +0.322 [+0.322, +0.322] |
| recon@200 - base | -0.030 [-0.030, -0.030] | -0.030 [-0.030, -0.030] |
| recon_generic@200 - base (continued training) | -0.083 [-0.083, -0.083] | -0.083 [-0.083, -0.083] |
| recon@200 - recon_generic@200 (subject vs continued training) | +0.052 [+0.052, +0.052] | +0.052 [+0.052, +0.052] |
| recon@400 - base | -0.458 [-0.458, -0.458] | -0.458 [-0.458, -0.458] |
| recon_generic@400 - base (continued training) | -0.171 [-0.171, -0.171] | -0.171 [-0.171, -0.171] |
| recon@400 - recon_generic@400 (subject vs continued training) | -0.287 [-0.287, -0.287] | -0.287 [-0.287, -0.287] |

knee_block_adam: 1 volumes, 8 slices

Mean correlation of the fitted per-volume probability map with the LOUPE map: brain_block_full: mask_in 0.7884, joint 0.5582; knee_block_full: mask_in 0.8708, joint 0.6707; brain_block_maskin600: mask_in 0.5268; knee_block_maskin600: mask_in 0.6400; brain_block_maskin1500: mask_in 0.4246; knee_block_maskin1500: mask_in 0.5124; brain_block_mask1500: mask 0.3793; knee_block_mask1500: mask 0.4819; brain: mask 0.7342, mask_in 0.8026, mask_ctrl 0.7376, joint 0.5762, joint_ctrl 0.5714; knee: mask 0.8290, mask_in 0.8589, mask_ctrl 0.8652, joint 0.6947, joint_ctrl 0.7001; knee_block: ; brain_block_fresh: ; brain_block_adam: ; knee_block_fresh: ; knee_block_adam: 

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

### ZS-SSL, brain_echo_vol_unseen_lr1e-5

| contrast | brain_echo_vol_unseen_lr1e-5 all | brain_echo_vol_unseen_lr1e-5 typical |
|---|---|---|
| ssl_vol@100 - base | -0.200 [-0.200, -0.200] | -0.200 [-0.200, -0.200] |
| ssl_vol@200 - base | -0.285 [-0.285, -0.285] | -0.285 [-0.285, -0.285] |
| ssl_vol@25 - base | -0.221 [-0.221, -0.221] | -0.221 [-0.221, -0.221] |
| ssl_vol@50 - base | -0.082 [-0.082, -0.082] | -0.082 [-0.082, -0.082] |
| ssl_vol@stop - base | -0.285 [-0.285, -0.285] | -0.285 [-0.285, -0.285] |

brain_echo_vol_unseen_lr1e-5: 1 volumes, 8 slices

ssl_vol Gamma-selected stop steps: [0]

### ZS-SSL, knee_echo_vol_unseen_lr1e-5

| contrast | knee_echo_vol_unseen_lr1e-5 all | knee_echo_vol_unseen_lr1e-5 typical |
|---|---|---|
| ssl_vol@100 - base | -0.194 [-0.194, -0.194] | -0.194 [-0.194, -0.194] |
| ssl_vol@200 - base | -0.248 [-0.248, -0.248] | -0.248 [-0.248, -0.248] |
| ssl_vol@25 - base | -0.045 [-0.045, -0.045] | -0.045 [-0.045, -0.045] |
| ssl_vol@50 - base | -0.116 [-0.116, -0.116] | -0.116 [-0.116, -0.116] |
| ssl_vol@stop - base | -0.248 [-0.248, -0.248] | -0.248 [-0.248, -0.248] |

knee_echo_vol_unseen_lr1e-5: 1 volumes, 8 slices

ssl_vol Gamma-selected stop steps: [0]

