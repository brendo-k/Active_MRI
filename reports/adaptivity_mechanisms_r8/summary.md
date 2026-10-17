# Adaptivity mechanisms, generated tables

## Gradient anatomy (z = score map; m = binary mask, outside the centre)

| operating point | reconstructor | L1 loss | pop | anat | slice | slice SNR/draw | anat SNR/draw | pop SNR @16 | reliability (16 draws) | neighbour corr | other-volume corr | m: slice | m: pop share in dead region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conv ep50 | conv ep50 (co-adapted) | 0.0471 | 0.026 | 0.025 | 0.949 | 0.17 | 0.0045 | 0.06 | 0.76 | 0.12 | -0.05 | 0.782 | 0.31 |
| conv ep4 | conv ep4 (co-adapted) | 0.0505 | 0.043 | 0.044 | 0.913 | 0.33 | 0.0159 | 0.19 | 0.87 | 0.21 | -0.06 | 0.798 | 0.16 |
| LOUPE | LOUPE (co-adapted) | 0.0465 | 0.030 | 0.028 | 0.942 | 0.18 | 0.0054 | 0.08 | 0.77 | 0.10 | -0.05 | 0.775 | 0.31 |
| conv ep50 | LOUPE | 0.0470 | 0.069 | 0.070 | 0.860 | 0.19 | 0.0156 | 0.20 | 0.77 | 0.18 | -0.06 | 0.705 | 0.72 |
| conv ep50 | conv ep4 | 0.0520 | 0.318 | 0.099 | 0.583 | 0.26 | 0.0436 | 1.73 | 0.84 | 0.43 | -0.06 | 0.538 | 0.01 |
| conv ep50 | uniform-random-mask | 0.0602 | 0.155 | 0.069 | 0.776 | 0.46 | 0.0405 | 0.97 | 0.88 | 0.65 | -0.01 | 0.770 | 0.01 |
| LOUPE | conv ep50 | 0.0474 | 0.169 | 0.043 | 0.788 | 0.22 | 0.0120 | 0.61 | 0.82 | 0.32 | -0.07 | 0.694 | 0.06 |

Slice-specific z energy kept at scales >= sigma / neighbour corr of the blurred signal:

| operating point | reconstructor | sigma 1 px | sigma 2 px | sigma 4 px | sigma 8 px |
|---|---|---|---|---|---|
| conv ep50 | conv ep50 (co-adapted) | 0.323 / 0.26 | 0.155 / 0.40 | 0.072 / 0.54 | 0.033 / 0.64 |
| conv ep4 | conv ep4 (co-adapted) | 0.393 / 0.41 | 0.235 / 0.60 | 0.143 / 0.77 | 0.085 / 0.83 |
| LOUPE | LOUPE (co-adapted) | 0.221 / 0.21 | 0.100 / 0.36 | 0.047 / 0.52 | 0.023 / 0.66 |
| conv ep50 | LOUPE | 0.385 / 0.37 | 0.226 / 0.54 | 0.135 / 0.67 | 0.081 / 0.78 |
| conv ep50 | conv ep4 | 0.533 / 0.67 | 0.394 / 0.80 | 0.290 / 0.90 | 0.201 / 0.94 |
| conv ep50 | uniform-random-mask | 0.715 / 0.82 | 0.548 / 0.89 | 0.374 / 0.94 | 0.249 / 0.97 |
| LOUPE | conv ep50 | 0.290 / 0.56 | 0.170 / 0.70 | 0.107 / 0.81 | 0.068 / 0.88 |

Pull toward the other sampler, corr(-mean_i dL/dm_i, p_other - p_own) outside the centre:

| operating point (own masks) | reconstructor | corr |
|---|---|---|
| conv ep50 | conv ep50 (co-adapted) | 0.025 |
| conv ep50 | LOUPE | 0.203 |
| conv ep50 | conv ep4 | -0.039 |
| conv ep50 | uniform-random-mask | 0.018 |
| LOUPE | LOUPE (co-adapted) | 0.161 |
| LOUPE | conv ep50 | 0.097 |

## Transport through the conv sampler network

| sampler | update | full update: common vs population | full update: anatomy offset | self cosine | self cosine vs blurred (4 px) target | control: population push cosine | control: anatomy push cosine | free-map alignment | net alignment (full update) | neighbour/self response | other/self response | |other-volume change| / |own change| | brain push: knee/brain | knee push: brain/knee |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conv ep4 | adam | 0.168 | 0.030 | 0.044 | 0.111 | 0.082 | 0.073 | 0.87 | -0.004 | 0.98 | 0.72 | 0.51 | 1.00 | 0.90 |
| conv ep4 | sgd | 0.067 | 0.087 | 0.115 | 0.246 | 0.126 | 0.260 | 0.87 | -0.014 | 0.99 | 0.95 | 0.94 | 1.02 | 0.97 |
| conv ep50 | adam | 0.027 | -0.010 | 0.008 | 0.030 | 0.031 | 0.022 | 0.76 | -0.002 | 0.89 | 0.28 | 0.39 | 1.10 | 0.45 |
| conv ep50 | sgd | 0.048 | 0.043 | 0.015 | 0.051 | 0.036 | 0.034 | 0.76 | -0.002 | 0.82 | 0.35 | 0.27 | 0.66 | 0.70 |

## Per-slice free masks fitted through each frozen reconstructor (gain over its shared refit, dB)

| reconstructor | shared PSNR | shared - prior | own slice | neighbour slice | other volumes | slice vs shared map corr | neighbour deviation corr | other-volume deviation corr |
|---|---|---|---|---|---|---|---|---|
| conv ep50 (co-adapted) | 39.07 | +0.25 [+0.20, +0.31] | +0.31 [+0.25, +0.37] | -0.30 [-0.36, -0.25] | -0.37 [-0.40, -0.33] | 0.713 | 0.714 | 0.687 |
| LOUPE | 39.32 | +0.32 [+0.26, +0.38] | +0.24 [+0.19, +0.29] | -0.34 [-0.41, -0.28] | -0.40 [-0.45, -0.35] | 0.707 | 0.713 | 0.685 |
| conv ep4 | 36.58 | +0.94 [+0.72, +1.15] | +0.31 [+0.26, +0.36] | -0.33 [-0.41, -0.26] | -0.57 [-0.71, -0.46] | 0.807 | 0.658 | 0.610 |
| uniform-random-mask | 35.41 | +1.04 [+0.70, +1.43] | -0.04 [-0.51, +0.45] | -0.40 [-0.88, +0.12] | -0.97 [-1.50, -0.46] | 0.772 | 0.619 | 0.489 |

## Tolerance to mask deviations (PSNR change vs the reconstructor's base masks, dB; r = map corr)

| reconstructor | base PSNR | reshape sigma 0.5 | reshape sigma 1 | reshape sigma 1.5 | reshape sigma 2.5 | ring 0.1 | ring 0.25 | ring 0.5 | ring 1 |
|---|---|---|---|---|---|---|---|---|---|
| conv_last | 38.45 | -0.24 [-0.28, -0.19] (r 0.95) | -0.93 [-1.12, -0.79] (r 0.84) | -1.83 [-2.23, -1.54] (r 0.75) | -3.39 [-4.19, -2.99] (r 0.65) | -0.03 [-0.06, -0.03] | -0.12 [-0.17, -0.10] | -0.31 [-0.44, -0.27] | -1.01 [-1.30, -0.87] |
| conv_ep4 | 36.29 | -0.15 [-0.19, -0.13] (r 0.97) | -0.58 [-0.72, -0.52] (r 0.91) | -1.17 [-1.46, -1.06] (r 0.84) | -2.54 [-3.17, -2.30] (r 0.73) | -0.01 [-0.03, -0.01] | -0.06 [-0.10, -0.04] | -0.20 [-0.29, -0.16] | -0.81 [-1.04, -0.72] |
| loupe_last | 39.06 | -0.13 [-0.16, -0.10] (r 0.97) | -0.59 [-0.74, -0.49] (r 0.91) | -1.41 [-1.80, -1.21] (r 0.82) | -3.37 [-4.18, -2.92] (r 0.65) | -0.03 [-0.05, -0.03] | -0.09 [-0.13, -0.08] | -0.19 [-0.26, -0.18] | -0.41 [-0.54, -0.37] |
| random | 34.74 | -0.06 [-0.13, +0.03] (r 0.95) | -0.16 [-0.27, -0.05] (r 0.84) | -0.30 [-0.45, -0.18] (r 0.75) | -0.64 [-0.85, -0.45] (r 0.65) | +0.05 [+0.03, +0.10] | +0.16 [+0.12, +0.21] | +0.24 [+0.19, +0.29] | +0.17 [+0.12, +0.26] |

## Fine-tune control: mismatch vs information

| reconstructor | reshape sigma | fine-tune steps | penalty, frozen | penalty after adapting | mismatch (frozen - adapted) | perturbed-trained cost on base masks |
|---|---|---|---|---|---|---|
| conv_last | 1.0 | 1000 | +0.93 [+0.78, +1.11] | +0.72 [+0.60, +0.84] | +0.21 [+0.17, +0.30] | +0.09 [+0.07, +0.12] |
| loupe_last | 1.0 | 1000 | +0.59 [+0.50, +0.74] | +0.44 [+0.37, +0.53] | +0.14 [+0.11, +0.23] | +0.04 [+0.02, +0.06] |

## Toy (final evaluation of each arm)

| arm | sampler | purity | steps | logit lr | recon | val PSNR | class-mean mask corr | own-band share A / B | swap cost (dB) | share of stochastic locations | class from hidden features (linear probe), init / trained |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_frozen_gen_net-label_s0 | net (label, head zero) | 1.0 | 3000 | 0.005 | frozen | 33.45 | 0.645 | 0.62 / 0.61 | 6.73 | 0.171 | 1.00 / 1.00 |
| A_frozen_gen_net-label_s1 | net (label, head zero) | 1.0 | 3000 | 0.005 | frozen | 33.52 | 0.655 | 0.62 / 0.61 | 6.97 | 0.167 | 1.00 / 1.00 |
| A_frozen_gen_net-magnitude_s0 | net (magnitude, head zero) | 1.0 | 3000 | 0.005 | frozen | 33.29 | 0.727 | 0.59 / 0.58 | 5.25 | 0.169 | 0.97 / 0.98 |
| A_frozen_gen_net-magnitude_s1 | net (magnitude, head zero) | 1.0 | 3000 | 0.005 | frozen | 33.36 | 0.751 | 0.58 / 0.56 | 5.00 | 0.163 | 0.97 / 0.98 |
| A_frozen_gen_net-raw_s0 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 32.68 | 0.995 | 0.46 / 0.47 | 0.39 | 0.163 | 0.70 / 0.87 |
| A_frozen_gen_net-raw_s1 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 32.57 | 1.000 | 0.46 / 0.45 | 0.07 | 0.159 | 0.69 / 0.75 |
| A_frozen_gen_static_s0 | static | 1.0 | 3000 | 0.005 | frozen | 31.81 | 1.000 | 0.42 / 0.42 | 0.00 | 0.176 |  |
| A_frozen_gen_static_s1 | static | 1.0 | 3000 | 0.005 | frozen | 31.79 | 1.000 | 0.42 / 0.42 | 0.00 | 0.178 |  |
| A_frozen_gen_table_s0 | table | 1.0 | 3000 | 0.005 | frozen | 32.69 | 0.653 | 0.56 / 0.56 | 6.14 | 0.197 |  |
| A_frozen_gen_table_s1 | table | 1.0 | 3000 | 0.005 | frozen | 32.69 | 0.651 | 0.56 / 0.56 | 6.12 | 0.198 |  |
| B_joint_scratch_net-label_s0 | net (label, head zero) | 1.0 | 6000 | 0.005 | joint | 46.47 | 0.779 | 0.50 / 0.50 | 12.82 | 0.078 | 1.00 / 1.00 |
| B_joint_scratch_net-label_s1 | net (label, head zero) | 1.0 | 6000 | 0.005 | joint | 46.61 | 0.816 | 0.48 / 0.48 | 12.88 | 0.076 | 1.00 / 1.00 |
| B_joint_scratch_net-magnitude_s0 | net (magnitude, head zero) | 1.0 | 6000 | 0.005 | joint | 45.72 | 0.975 | 0.42 / 0.45 | 2.35 | 0.078 | 0.97 / 0.98 |
| B_joint_scratch_net-magnitude_s1 | net (magnitude, head zero) | 1.0 | 6000 | 0.005 | joint | 44.94 | 1.000 | 0.43 / 0.42 | 0.10 | 0.073 | 0.97 / 0.98 |
| B_joint_scratch_net-raw_s0 | net (raw, head zero) | 1.0 | 6000 | 0.005 | joint | 44.14 | 1.000 | 0.44 / 0.40 | -0.02 | 0.084 | 0.70 / 0.67 |
| B_joint_scratch_net-raw_s1 | net (raw, head zero) | 1.0 | 6000 | 0.005 | joint | 44.79 | 1.000 | 0.40 / 0.42 | -0.05 | 0.086 | 0.69 / 0.66 |
| B_joint_scratch_static_s0 | static | 1.0 | 6000 | 0.005 | joint | 39.96 | 1.000 | 0.43 / 0.43 | 0.00 | 0.164 |  |
| B_joint_scratch_static_s1 | static | 1.0 | 6000 | 0.005 | joint | 40.65 | 1.000 | 0.42 / 0.44 | 0.00 | 0.155 |  |
| B_joint_scratch_table_s0 | table | 1.0 | 6000 | 0.005 | joint | 41.36 | 0.640 | 0.57 / 0.56 | 13.12 | 0.180 |  |
| B_joint_scratch_table_s1 | table | 1.0 | 6000 | 0.005 | joint | 41.98 | 0.639 | 0.57 / 0.56 | 13.85 | 0.171 |  |
| C_lockin_coadapted_frozen_net_s0 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 42.33 | 1.000 | 0.41 / 0.42 | -0.02 | 0.078 | 0.70 / 0.68 |
| C_lockin_coadapted_frozen_net_s1 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 43.16 | 1.000 | 0.40 / 0.42 | 0.00 | 0.074 | 0.69 / 0.77 |
| C_lockin_coadapted_frozen_table_s0 | table | 1.0 | 3000 | 0.005 | frozen | 42.10 | 0.851 | 0.50 / 0.51 | 7.03 | 0.158 |  |
| C_lockin_coadapted_frozen_table_s1 | table | 1.0 | 3000 | 0.005 | frozen | 43.27 | 0.846 | 0.51 / 0.53 | 7.46 | 0.139 |  |
| C_lockin_coadapted_net_s0 | net (raw, head zero) | 1.0 | 6000 | 0.005 | joint | 46.56 | 1.000 | 0.40 / 0.42 | 0.06 | 0.069 | 0.70 / 0.67 |
| C_lockin_coadapted_net_s1 | net (raw, head zero) | 1.0 | 6000 | 0.005 | joint | 46.83 | 1.000 | 0.41 / 0.43 | 0.02 | 0.084 | 0.69 / 0.72 |
| C_lockin_coadapted_table_s0 | table | 1.0 | 6000 | 0.005 | joint | 46.48 | 0.770 | 0.53 / 0.53 | 13.99 | 0.121 |  |
| C_lockin_coadapted_table_s1 | table | 1.0 | 6000 | 0.005 | joint | 47.85 | 0.784 | 0.53 / 0.54 | 14.31 | 0.122 |  |
| D_from_static_generalist_frozen_net_s0 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 33.21 | 0.785 | 0.57 / 0.55 | 4.82 | 0.164 | 0.70 / 0.98 |
| D_from_static_generalist_frozen_net_s1 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 33.30 | 0.773 | 0.57 / 0.56 | 4.67 | 0.163 | 0.69 / 0.97 |
| D_from_static_generalist_frozen_table_s0 | table | 1.0 | 3000 | 0.005 | frozen | 33.43 | 0.655 | 0.62 / 0.60 | 6.57 | 0.179 |  |
| D_from_static_generalist_frozen_table_s1 | table | 1.0 | 3000 | 0.005 | frozen | 33.43 | 0.661 | 0.62 / 0.61 | 6.54 | 0.179 |  |
| D_from_static_generalist_joint_net_s0 | net (raw, head zero) | 1.0 | 6000 | 0.005 | joint | 45.85 | 1.000 | 0.42 / 0.42 | -0.04 | 0.077 | 0.70 / 0.68 |
| D_from_static_generalist_joint_net_s1 | net (raw, head zero) | 1.0 | 6000 | 0.005 | joint | 46.45 | 1.000 | 0.41 / 0.44 | 0.03 | 0.083 | 0.69 / 0.82 |
| D_from_static_generalist_joint_table_s0 | table | 1.0 | 6000 | 0.005 | joint | 46.35 | 0.730 | 0.54 / 0.56 | 14.38 | 0.123 |  |
| D_from_static_generalist_joint_table_s1 | table | 1.0 | 6000 | 0.005 | joint | 47.18 | 0.742 | 0.54 / 0.55 | 14.64 | 0.122 |  |
| E_frozen_gen_net-raw_p0.8 | net (raw, head zero) | 0.8 | 3000 | 0.005 | frozen | 32.66 | 1.000 | 0.44 / 0.45 | 0.01 | 0.161 | 0.61 / 0.67 |
| E_frozen_gen_static_p0.8 | static | 0.8 | 3000 | 0.005 | frozen | 31.93 | 1.000 | 0.42 / 0.41 | 0.00 | 0.181 |  |
| E_frozen_gen_table_p0.8 | table | 0.8 | 3000 | 0.005 | frozen | 32.13 | 0.894 | 0.49 / 0.48 | 0.96 | 0.201 |  |
| E_joint_scratch_net-raw_p0.8 | net (raw, head zero) | 0.8 | 6000 | 0.005 | joint | 45.31 | 1.000 | 0.43 / 0.39 | -0.02 | 0.095 | 0.61 / 0.58 |
| E_joint_scratch_static_p0.8 | static | 0.8 | 6000 | 0.005 | joint | 40.59 | 1.000 | 0.42 / 0.42 | 0.00 | 0.157 |  |
| E_joint_scratch_table_p0.8 | table | 0.8 | 6000 | 0.005 | joint | 40.64 | 0.900 | 0.48 / 0.48 | 1.61 | 0.182 |  |
| Q_frozen_tablerecon_from_static_net_s0 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 42.18 | 1.000 | 0.43 / 0.41 | -0.01 | 0.080 | 0.70 / 0.71 |
| Q_frozen_tablerecon_from_static_net_s1 | net (raw, head zero) | 1.0 | 3000 | 0.005 | frozen | 43.38 | 1.000 | 0.41 / 0.42 | 0.01 | 0.080 | 0.69 / 0.79 |

Toy score-map gradient at the B-static mask, through the co-adapted static reconstructor or the generalist:

| run | pop energy | class energy | image energy | noise per draw | class fraction | class SNR per draw | class offset vs ideal move |
|---|---|---|---|---|---|---|---|
| G_gradients_coadapted_s0 | 2.83e-11 | 4.36e-11 | 2.99e-10 | 5.03e-08 | 0.117 | 0.0009 | 0.54 |
| G_gradients_coadapted_s1 | 1.35e-11 | 2.12e-11 | 2.14e-10 | 2.23e-08 | 0.085 | 0.0009 | 0.59 |
| G_gradients_generalist_s0 | 2.77e-11 | 5.07e-11 | 4.17e-10 | 5.06e-09 | 0.102 | 0.0100 | 0.60 |
| G_gradients_generalist_s1 | 3.78e-11 | 5.02e-11 | 3.99e-10 | 4.42e-09 | 0.103 | 0.0114 | 0.63 |
| G_gradients_tablerecon_s0 | 3.60e-11 | 1.06e-10 | 6.13e-10 | 5.90e-08 | 0.140 | 0.0018 | 0.61 |
| G_gradients_tablerecon_s1 | 2.25e-11 | 5.70e-11 | 2.88e-10 | 3.32e-08 | 0.155 | 0.0017 | 0.60 |

Net fitted to the table's per-class logit maps by direct regression:

| net input | relative error | class difference, fitted / target |
|---|---|---|
| raw | 0.077 | 0.845 |
| magnitude | 0.059 | 0.909 |
| label | 0.000 | 1.000 |

Class decodability from the acquired 4x4 centre (held-out accuracy):

| purity | raw: linear | raw: MLP | magnitude: linear | magnitude: MLP |
|---|---|---|---|---|
| 0.65 | 0.518 | 0.604 | 0.666 | 0.629 |
| 0.8 | 0.498 | 0.775 | 0.806 | 0.799 |
| 1.0 | 0.493 | 0.985 | 0.980 | 0.980 |
