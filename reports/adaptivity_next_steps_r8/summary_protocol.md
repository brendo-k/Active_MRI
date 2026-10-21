Generalist reconstructor (step 6000) vs the LOUPE run's own, 86 val slices, PSNR dB:

| masks | LOUPE run recon | generalist | difference |
|---|---|---|---|
| loupe | 39.06 | 38.87 | -0.18 |
| conv | 38.62 | 38.54 | -0.08 |
| loupe_reshaped | 38.34 | 38.46 | +0.12 |

| run | sampler | reconstructor | val PSNR brain / knee | anatomy swap cost (dB) | slice swap cost (dB) | slice vs mean corr brain / knee | brain vs knee corr | offset mean abs / across-slice std (logits) |
|---|---|---|---|---|---|---|---|---|
| static_joint_R8 | static (offset off) | joint fine-tune | 38.177 / 40.017 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 1.0000 |  |
| anattable_joint_R8 | one map per anatomy | joint fine-tune | 38.161 / 40.050 | +0.058 [+0.030, +0.060] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 0.9754 |  |
| prototable_joint_R8 | one map per protocol (brain / knee-PD / knee-PDFS) | joint fine-tune | 38.159 / 40.036 | +0.060 [+0.040, +0.071] | +0.008 [-0.005, +0.019] | 1.0000 / 0.9926 | 0.9818 |  |

| run | val PSNR knee-PD / knee-PDFS | protocol swap cost, knee (dB) | PD vs PDFS corr |
|---|---|---|---|
| static_joint_R8 | 42.793 / 38.224 | +0.000 [+0.000, +0.000] | 1.0000 |
| anattable_joint_R8 | 42.849 / 38.242 | +0.000 [+0.000, +0.000] | 1.0000 |
| prototable_joint_R8 | 42.875 / 38.202 | +0.019 [-0.037, +0.085] | 0.9691 |

| comparison (per-slice PSNR, dB) | all | brain | knee | knee-PD | knee-PDFS |
|---|---|---|---|---|---|
| anattable_joint_R8 - static_joint_R8 | +0.009 [-0.011, +0.010] | -0.016 [-0.023, -0.007] | +0.033 [+0.019, +0.047] | +0.056 [+0.040, +0.070] | +0.018 [+0.008, +0.025] |
| prototable_joint_R8 - static_joint_R8 | +0.001 [-0.017, +0.008] | -0.018 [-0.026, -0.008] | +0.019 [-0.011, +0.055] | +0.082 [+0.069, +0.097] | -0.022 [-0.038, -0.002] |
| prototable_joint_R8 - anattable_joint_R8 | -0.008 [-0.012, +0.003] | -0.002 [-0.003, -0.000] | -0.014 [-0.035, +0.012] | +0.026 [+0.019, +0.030] | -0.040 [-0.062, -0.013] |
