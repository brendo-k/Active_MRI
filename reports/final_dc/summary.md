# final_dc summary

## Reproduction (lambda = 0 against the published eval, PSNR_Masked dB)

| run | published | lambda = 0 | mean gap | max slice gap | within 0.003 dB |
|---|---|---|---|---|---|
| ml100_control_s42 | 33.513 | 33.513 | +0.0000 | 0.0000 | yes |
| ml100_control_s43 | 33.279 | 33.279 | +0.0000 | 0.0000 | yes |
| ml100_muon_s42 | 33.637 | 33.637 | +0.0000 | 0.0000 | yes |
| ml100_muon_s43 | 33.450 | 33.450 | +0.0000 | 0.0000 | yes |
| ml100_layer_s42 | 33.527 | 33.527 | +0.0000 | 0.0000 | yes |
| ml100_layer_s43 | 33.370 | 33.370 | +0.0000 | 0.0000 | yes |
| ml100_muon_layer_s42 | 33.859 | 33.859 | +0.0000 | 0.0000 | yes |
| ml100_muon_layer_s43 | 33.606 | 33.606 | +0.0000 | 0.0000 | yes |
| loss_domain_rss | 33.623 | 33.623 | +0.0000 | — | yes |
| loss_domain_kspace_gt | 31.230 | 31.230 | +0.0000 | — | yes |

## Frozen criteria (2x2, lambda = 1 minus lambda = 0)

Mean over the 8 checkpoints: -2.002 dB. Verdict: **failure** (C1 False, C2 False, C3 False, failure rule True).

| checkpoint | PSNR λ=0 | ΔPSNR all [95% CI] | ΔSSIM all | Δ brain | Δ knee-PD | Δ knee-PDFS |
|---|---|---|---|---|---|---|
| control_s42 | 33.51 | -2.196 [-2.575, -1.816] | -0.0069 | -1.469 | -1.005 | -4.076 |
| control_s43 | 33.28 | -2.093 [-2.469, -1.717] | -0.0056 | -1.273 | -1.027 | -3.933 |
| layer_s42 | 33.53 | -2.198 [-2.576, -1.820] | -0.0067 | -1.517 | -0.973 | -4.071 |
| layer_s43 | 33.37 | -2.131 [-2.511, -1.752] | -0.0070 | -1.439 | -0.898 | -4.022 |
| muon_layer_s42 | 33.86 | -2.205 [-2.579, -1.831] | -0.0075 | -1.569 | -0.978 | -4.039 |
| muon_layer_s43 | 33.61 | -2.012 [-2.320, -1.704] | -0.0310 | -1.375 | -1.128 | -3.496 |
| muon_s42 | 33.64 | -2.140 [-2.515, -1.765] | -0.0073 | -1.530 | -0.883 | -3.979 |
| muon_s43 | 33.45 | -1.038 [-1.285, -0.792] | -0.0033 | -0.692 | -0.217 | -2.193 |

Worse by >= 0.10 dB in both seeds: control brain, control knee-PD, control knee-PDFS, muon brain, muon knee-PD, muon knee-PDFS, layer brain, layer knee-PD, layer knee-PDFS, muon_layer brain, muon_layer knee-PD, muon_layer knee-PDFS.

## Lambda curve (mean over the 8 checkpoints; ΔPSNR dB / ΔSSIM against lambda = 0)

| condition | l0.25 | l0.5 | l0.75 | l1 |
|---|---|---|---|---|
| all | -0.512 / +0.0004 | -1.286 / -0.0027 | -1.832 / -0.0071 | -2.002 / -0.0094 |
| brain | -0.273 / +0.0008 | -0.780 / -0.0010 | -1.195 / -0.0036 | -1.358 / -0.0051 |
| knee-PD | -0.206 / -0.0000 | -0.568 / -0.0016 | -0.826 / -0.0034 | -0.889 / -0.0042 |
| knee-PDFS | -1.043 / +0.0006 | -2.482 / -0.0055 | -3.440 / -0.0140 | -3.726 / -0.0186 |

## Mechanism (post-freeze diagnostics; means by condition)

Phase offsets are tissue-weighted mean |angle| in rad; errors are relative norms at the sampled points; Δ are PSNR dB against the same run at lambda = 0. Oracle alignment takes the phase from the fully sampled data (noise phase included), so it is an upper bound with leakage; the estimated alignment uses the sampled data only (σ = 8 px).

### 2x2 mean (old flip)

| readout | all | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|
| phase offset (rad) | 0.233 | 0.195 | 0.145 | 0.357 |
| error at samples | 0.400 | 0.386 | 0.229 | 0.587 |
| maps floor | 0.173 | 0.156 | 0.120 | 0.243 |
| error at samples, oracle-aligned | 0.208 | 0.185 | 0.133 | 0.305 |
| share of error at samples | 0.486 | 0.482 | 0.509 | 0.468 |
| Δ hard DC | -2.002 | -1.358 | -0.889 | -3.726 |
| Δ oracle-aligned, no DC | +0.022 | +0.069 | +0.000 | +0.000 |
| Δ oracle-aligned + hard DC | -1.085 | -0.218 | -0.269 | -2.713 |
| Δ estimated-aligned + hard DC | -2.247 | -2.075 | -1.018 | -3.658 |
| estimate phase error (rad) | 0.109 | 0.105 | 0.070 | 0.153 |
| output/target mean, λ=0 | 0.995 | 0.998 | 0.996 | 0.992 |
| output/target mean, λ=1 | 0.944 | 0.963 | 0.980 | 0.891 |
| Δ oracle gain+offset fit, λ=0 | +0.136 | +0.047 | +0.148 | +0.207 |
| Δ oracle gain+offset fit, λ=1 | -0.215 | -0.209 | -0.143 | -0.294 |

### loss_domain rss (fixed flip)

| readout | all | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|
| phase offset (rad) | 0.280 | 0.227 | 0.164 | 0.446 |
| error at samples | 0.438 | 0.416 | 0.245 | 0.654 |
| maps floor | 0.173 | 0.159 | 0.118 | 0.242 |
| error at samples, oracle-aligned | 0.209 | 0.188 | 0.132 | 0.306 |
| share of error at samples | 0.554 | 0.540 | 0.574 | 0.545 |
| Δ hard DC | -2.241 | -1.518 | -1.044 | -4.126 |
| Δ oracle-aligned, no DC | +0.017 | +0.053 | +0.000 | +0.000 |
| Δ oracle-aligned + hard DC | -1.118 | -0.201 | -0.297 | -2.796 |
| Δ estimated-aligned + hard DC | -2.608 | -2.431 | -1.268 | -4.137 |
| estimate phase error (rad) | 0.097 | 0.097 | 0.056 | 0.138 |
| output/target mean, λ=0 | 0.997 | 1.000 | 0.998 | 0.994 |
| output/target mean, λ=1 | 0.939 | 0.960 | 0.976 | 0.882 |
| Δ oracle gain+offset fit, λ=0 | +0.121 | +0.036 | +0.147 | +0.172 |
| Δ oracle gain+offset fit, λ=1 | -0.198 | -0.192 | -0.154 | -0.249 |

### loss_domain kspace_gt (fixed flip)

| readout | all | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|
| phase offset (rad) | 0.063 | 0.056 | 0.034 | 0.099 |
| error at samples | 0.142 | 0.131 | 0.081 | 0.215 |
| maps floor | 0.147 | 0.136 | 0.083 | 0.221 |
| error at samples, oracle-aligned | 0.148 | 0.138 | 0.084 | 0.223 |
| share of error at samples | 0.120 | 0.102 | 0.142 | 0.113 |
| Δ hard DC | +0.582 | +0.397 | +0.397 | +0.940 |
| Δ oracle-aligned, no DC | +0.020 | +0.064 | -0.000 | -0.000 |
| Δ oracle-aligned + hard DC | +0.862 | +0.700 | +0.643 | +1.234 |
| Δ estimated-aligned + hard DC | +0.582 | +0.397 | +0.396 | +0.941 |
| estimate phase error (rad) | 0.063 | 0.056 | 0.034 | 0.099 |
| output/target mean, λ=0 | 0.930 | 0.956 | 0.973 | 0.862 |
| output/target mean, λ=1 | 0.940 | 0.962 | 0.977 | 0.883 |
| Δ oracle gain+offset fit, λ=0 | +2.487 | +1.603 | +1.170 | +4.641 |
| Δ oracle gain+offset fit, λ=1 | +2.652 | +1.693 | +1.332 | +4.876 |

## Side readout: does the step recover kspace_gt's intensity gap? (seed 42, descriptive)

| comparison | all | brain | knee-PD | knee-PDFS |
|---|---|---|---|---|
| kspace_gt λ=1 − rss λ=0 | -1.811 [-2.243, -1.379] | -1.068 [-1.397, -0.739] | -0.387 [-0.846, +0.072] | -3.944 [-4.841, -3.046] |
| kspace_gt − rss, both λ=0 | -2.393 [-2.898, -1.887] | -1.465 [-1.871, -1.059] | -0.784 [-1.322, -0.245] | -4.884 [-5.901, -3.866] |
| kspace_gt − rss, both λ=1 | +0.431 [+0.360, +0.501] | +0.450 [+0.378, +0.522] | +0.657 [+0.475, +0.839] | +0.182 [+0.111, +0.253] |
| rss λ=1 − λ=0 | -2.241 [-2.620, -1.863] | -1.518 [-1.792, -1.244] | -1.044 [-1.338, -0.750] | -4.126 [-4.960, -3.291] |
| kspace_gt λ=1 − λ=0 | +0.582 [+0.503, +0.661] | +0.397 [+0.314, +0.479] | +0.397 [+0.301, +0.493] | +0.940 [+0.806, +1.075] |

knee-PDFS output/target mean: rss l0 0.994, rss l1 0.882, kspace_gt l0 0.862, kspace_gt l1 0.883.
Same-λ deficit: 2.393 dB at λ=0, 0.000 dB at λ=1 (signed gap ratio -0.18). (a) True, (b) False: recovers = **False**.
