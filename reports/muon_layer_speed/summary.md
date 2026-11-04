# muon_layer_speed summary

Paired against the same-seed 100-epoch reference on the full validation set, 2 mask draws.

| arm | seed | PSNR | SSIM | delta PSNR [95% CI] | delta SSIM | brain / PD / PDFS delta | epochs | median train s | wall-clock h (train + val) | share | share, reference at the arms' val cadence | steady h | EMA - last dB | EMA verdict | matches reference epoch | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e30_bs8 | 42 | 33.717 | 0.8609 | -0.142 [-0.180, -0.104] | -0.0023 | -0.11 / -0.31 / +0.01 | 30 | 332 | 3.19 (3.07 + 0.12) | 28% | 32% | 2.89 | -0.024 | not adopted, decided | 53 | provisional inconclusive (1 of 2 seeds) |
| e30_cos_muon2e3 | 42 | 33.402 | 0.8566 | -0.457 [-0.533, -0.381] | -0.0066 | -0.50 / -0.75 / -0.12 | 30 | 303 | 2.69 (2.56 + 0.13) | 24% | 27% | 2.66 | -0.016 | not adopted, decided | 27 | not faster, decided; provisional worse (1 of 2 seeds) |
| e30_cos | 42 | 33.435 | 0.8570 | -0.424 [-0.496, -0.351] | -0.0062 | -0.41 / -0.73 / -0.12 | 30 | 343 | 3.02 (2.89 + 0.14) | 27% | 31% | 2.99 | -0.022 | not adopted, decided | 28 | not faster, decided; provisional worse (1 of 2 seeds) |
| e30_step | 42 | 33.552 | 0.8587 | -0.307 [-0.361, -0.252] | -0.0045 | -0.30 / -0.53 / -0.08 | 30 | 301 | 2.65 (2.54 + 0.11) | 23% | 27% | 2.62 | -0.052 | not adopted, decided | 38 | not faster, decided; provisional worse (1 of 2 seeds) |

Reference, 100 epochs: 11.30 h (9.50 train + 1.80 val), median 340 s per epoch; 9.86 h at the arms' val cadence.

Between arms, same seed, per slice (exploratory):

| arm minus arm | seed | delta PSNR [95% CI] | delta SSIM |
|---|---|---|---|
| e30_step - e30_cos | 42 | +0.117 [+0.097, +0.137] | +0.0018 |
| e30_cos_muon2e3 - e30_cos | 42 | -0.033 [-0.046, -0.020] | -0.0004 |
| e30_bs8 - e30_step | 42 | +0.165 [+0.143, +0.187] | +0.0022 |
| e30_step_maskof_ref - e30_step | 42 | -0.017 [-0.038, +0.005] | +0.0004 |
| ref - ref_maskof_e30_step | 42 | +0.158 [+0.137, +0.179] | +0.0015 |
| ref_maskof_e30_step - e30_step | 42 | +0.149 [+0.099, +0.199] | +0.0029 |
| ref - e30_step_maskof_ref | 42 | +0.323 [+0.263, +0.384] | +0.0041 |

Constant-lr estimate: the reference's smoothed curve reaches 33.76 dB at epoch 65, 5.7 h at e30_step's epoch time.

| bench arm | step s | change | val pass s | steady peak GiB |
|---|---|---|---|---|
| control | 1.781 | +0.0% | 19.4 | 36.3 |
| exact | 1.446 | -18.8% | 13.7 | 36.2 |
| exact_16_channels | 1.207 | -32.3% | 15.8 | 34.5 |

Trainer val PSNR_Masked by epoch:

| run | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| 100-epoch reference | 32.31 | 32.76 | 33.03 | 33.25 | 33.34 | 33.48 |
| e30_cos_s42 | 32.33 | 32.86 | 33.15 | 33.31 | 33.40 | 33.43 |
| e30_step_s42 | 32.27 | 32.85 | 33.03 | 33.23 | 33.40 | 33.55 |
| e30_bs8_s42 | 32.48 | 33.06 | 33.26 | 33.37 | 33.50 | 33.71 |
| e30_cos_muon2e3_s42 | 32.52 | 32.95 | 33.13 | 33.30 | 33.38 | 33.40 |
| e30_step_c16_s42 | 32.46 | 32.86 | 33.06 | 33.18 | - | - |

Absolute scores per condition (PSNR / SSIM):

| run | epochs | all | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|---|
| ref_s42 | 100 | 33.86 / 0.863 | 33.02 / 0.885 | 36.04 / 0.921 | 32.40 / 0.784 |
| e30_bs8_s42 | 30 | 33.72 / 0.861 | 32.91 / 0.883 | 35.72 / 0.918 | 32.41 / 0.783 |
| e30_bs8_ema_s42 | 30 | 33.69 / 0.861 | 32.89 / 0.882 | 35.69 / 0.918 | 32.39 / 0.783 |
| e30_cos_muon2e3_s42 | 30 | 33.40 / 0.857 | 32.53 / 0.877 | 35.29 / 0.912 | 32.28 / 0.781 |
| e30_cos_muon2e3_ema_s42 | 30 | 33.39 / 0.856 | 32.50 / 0.877 | 35.27 / 0.912 | 32.28 / 0.781 |
| e30_cos_s42 | 30 | 33.44 / 0.857 | 32.61 / 0.878 | 35.31 / 0.912 | 32.28 / 0.781 |
| e30_cos_ema_s42 | 30 | 33.41 / 0.857 | 32.58 / 0.878 | 35.29 / 0.912 | 32.27 / 0.781 |
| e30_step_maskof_ref_s42 | 30 | 33.54 / 0.859 | 32.75 / 0.881 | 35.48 / 0.916 | 32.27 / 0.781 |
| e30_step_s42 | 30 | 33.55 / 0.859 | 32.72 / 0.880 | 35.50 / 0.915 | 32.32 / 0.782 |
| e30_step_ema_s42 | 30 | 33.50 / 0.858 | 32.66 / 0.879 | 35.43 / 0.914 | 32.30 / 0.781 |
| ref_maskof_e30_step_s42 | 100 | 33.70 / 0.862 | 32.91 / 0.883 | 35.82 / 0.919 | 32.27 / 0.784 |

Learned mask p(k), against the same-seed reference, and mean p by radial band in pixels from the centre:

| run | mean p | correlation | 0-20 | 20-40 | 40-80 | 80-120 | 120-160 | 160-230 |
|---|---|---|---|---|---|---|---|---|
| ref_s42 | 0.1328 | 1.000 | 0.803 | 0.513 | 0.285 | 0.153 | 0.070 | 0.003 |
| e30_cos_s42 | 0.1302 | 0.762 | 0.879 | 0.573 | 0.309 | 0.145 | 0.051 | 0.000 |
| e30_step_s42 | 0.1292 | 0.775 | 0.866 | 0.555 | 0.292 | 0.148 | 0.055 | 0.000 |
| e30_bs8_s42 | 0.1291 | 0.741 | 0.839 | 0.541 | 0.286 | 0.149 | 0.058 | 0.001 |
| e30_cos_muon2e3_s42 | 0.1294 | 0.738 | 0.887 | 0.583 | 0.305 | 0.141 | 0.051 | 0.000 |

| profile | step ms | change | forward | backward | grad_stats | clip | optimizer | sens net ms | steady GiB |
|---|---|---|---|---|---|---|---|---|---|
| control | 1379.2 | +0.0% | 459.0 | 789.1 | 25.5 | 7.1 | 88.3 | 323.0 | - |
| channels_last | 1332.7 | -3.4% | 463.8 | 749.8 | 23.8 | 6.0 | 78.1 | 265.5 | - |
| chans16_nhwc | 947.4 | -31.3% | 334.8 | 531.1 | 14.5 | 7.0 | 49.4 | 265.9 | 33.6 |
| cheap | 1317.6 | -4.5% | 434.4 | 788.1 | 12.0 | 6.0 | 65.9 | 322.2 | - |
| fused_backward | 1308.5 | -5.1% | 412.4 | 752.3 | 20.5 | 11.1 | 101.3 | 323.2 | 35.3 |
| fused_pinned | 1288.4 | -6.6% | 411.8 | 749.9 | 19.5 | 9.0 | 87.5 | 323.9 | 39.7 |
| gc_freeze | 1267.3 | -8.1% | 411.1 | 750.4 | 14.1 | 7.0 | 72.1 | 322.9 | 35.3 |
| group_senscl | 1194.0 | -13.4% | 396.9 | 704.5 | 11.3 | 6.0 | 64.6 | 264.5 | 35.3 |
| layer_bf16 | 1486.8 | +7.8% | 457.3 | 921.0 | 18.1 | 7.0 | 72.4 | 275.2 | 33.3 |
| layer_bf16_nchw | 1476.5 | +7.1% | 455.1 | 919.2 | 12.8 | 6.1 | 70.9 | 273.9 | 33.3 |
| layer_bf16_senscl | 1483.1 | +7.5% | 455.6 | 919.2 | 14.6 | 7.0 | 75.9 | 273.9 | 33.3 |

Relative distance, hand-written backward against autograd, one step:

| precision | output | reconstructor | sensitivity net | mask |
|---|---|---|---|---|
| float64, full size | 1.6e-09 | 8.2e-10 | 9.4e-09 | 7.3e-09 |
| fp32, TF32 off | 5.2e-07 | 4.6e-05 | 5.4e-05 | 5.8e-04 |
| fp32, TF32 off, smooth loss | 5.2e-07 | 5.1e-05 | 6.7e-05 | 5.3e-04 |
| bf16, as trained | 6.5e-03 | 4.6e-03 | 5.3e-03 | 8.4e-02 |

Two epochs on 5% of the data, final weights, relative distance:

| pair | whole model | median parameter | worst parameter | mask |
|---|---|---|---|---|
| before vs before_again | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| before vs after | 2.73e-02 | 9.34e-03 | 1.37e-01 | 1.80e-03 |

| run | last train loss | last val loss |
|---|---|---|
| before | 0.08205 | 0.101 |
| before_again | 0.08205 | 0.101 |
| after | 0.08215 | 0.101 |
