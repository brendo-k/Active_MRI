Generalist reconstructor (step 6000) vs the LOUPE run's own, 86 val slices, PSNR dB:

| masks | LOUPE run recon | generalist | difference |
|---|---|---|---|
| loupe | 39.06 | 38.87 | -0.18 |
| conv | 38.62 | 38.54 | -0.08 |
| loupe_reshaped | 38.34 | 38.46 | +0.12 |

| run | sampler | reconstructor | val PSNR brain / knee | anatomy swap cost (dB) | slice swap cost (dB) | slice vs mean corr brain / knee | brain vs knee corr | offset mean abs / across-slice std (logits) |
|---|---|---|---|---|---|---|---|---|
| loupe_original | LOUPE (start point) | LOUPE run, co-adapted | 38.094 / 39.966 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 1.0000 |  |
| static_joint_R8 | static (offset off) | joint fine-tune | 38.177 / 40.017 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 1.0000 |  |
| adapt16_joint_R8 | adaptive head, 16 x 16 grid | joint fine-tune | 38.191 / 40.004 | +0.003 [+0.000, +0.005] | +0.001 [-0.001, +0.002] | 0.9999 / 0.9999 | 0.9999 | 2.338 / 0.097 |
| anattable_joint_R8 | one map per anatomy | joint fine-tune | 38.161 / 40.050 | +0.058 [+0.031, +0.060] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 0.9754 |  |
| adapt16_frozen-loupe-recon_R8 | adaptive head, 16 x 16 grid | LOUPE run, frozen | 38.055 / 39.975 | -0.003 [-0.008, -0.001] | +0.001 [-0.002, +0.005] | 0.9999 / 0.9998 | 0.9998 | 2.254 / 0.083 |
| adapt16_frozen-generalist_R8 | adaptive head, 16 x 16 grid | generalist, frozen | 37.142 / 39.113 | -0.012 [-0.084, +0.062] | +0.007 [-0.030, +0.091] | 0.9979 / 0.9972 | 0.9999 | 1.405 / 0.181 |
| static_frozen-generalist_R8 | static (offset off) | generalist, frozen | 37.084 / 39.107 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 1.0000 |  |

| comparison (per-slice PSNR, dB) | all | brain | knee |
|---|---|---|---|
| adapt16_joint_R8 - static_joint_R8 | +0.000 [-0.002, +0.013] | +0.014 [+0.007, +0.020] | -0.013 [-0.027, +0.001] |
| anattable_joint_R8 - static_joint_R8 | +0.009 [-0.011, +0.010] | -0.016 [-0.023, -0.007] | +0.033 [+0.019, +0.046] |
| adapt16_frozen-loupe-recon_R8 - loupe_original | -0.015 [-0.036, -0.014] | -0.038 [-0.049, -0.028] | +0.009 [-0.002, +0.018] |
| adapt16_frozen-generalist_R8 - static_frozen-generalist_R8 | +0.032 [-0.010, +0.113] | +0.058 [-0.006, +0.146] | +0.006 [-0.089, +0.123] |
| static_joint_R8 - loupe_original | +0.067 [+0.047, +0.095] | +0.083 [+0.060, +0.104] | +0.051 [-0.014, +0.100] |
