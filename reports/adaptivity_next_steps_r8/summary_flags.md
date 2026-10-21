Generalist reconstructor (step 6000) vs the LOUPE run's own, 86 val slices, PSNR dB:

| masks | LOUPE run recon | generalist | difference |
|---|---|---|---|
| loupe | 39.06 | 38.87 | -0.18 |
| conv | 38.62 | 38.54 | -0.08 |
| loupe_reshaped | 38.34 | 38.46 | +0.12 |

| run | sampler | reconstructor | val PSNR brain / knee | anatomy swap cost (dB) | slice swap cost (dB) | slice vs mean corr brain / knee | brain vs knee corr | offset mean abs / across-slice std (logits) |
|---|---|---|---|---|---|---|---|---|
| loupe_flags_original | loupe_flags_original |  | 37.623 / 39.283 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 1.0000 |  |
| static_joint_flags_R8 | static (offset off) | flagged VarNet, joint fine-tune | 37.724 / 39.410 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 1.0000 |  |
| adapt16_joint_flags_R8 | adaptive head, 16 x 16 grid | flagged VarNet, joint fine-tune | 37.732 / 39.390 | +0.001 [-0.000, +0.002] | +0.000 [-0.001, +0.002] | 0.9999 / 0.9999 | 1.0000 | 3.283 / 0.073 |
| anattable_joint_flags_R8 | one map per anatomy | flagged VarNet, joint fine-tune | 37.691 / 39.443 | +0.035 [-0.014, +0.032] | +0.000 [+0.000, +0.000] | 1.0000 / 1.0000 | 0.9816 |  |

| comparison (per-slice PSNR, dB) | all | brain | knee |
|---|---|---|---|
| adapt16_joint_flags_R8 - static_joint_flags_R8 | -0.007 [-0.007, +0.006] | +0.007 [+0.002, +0.013] | -0.020 [-0.030, -0.007] |
| anattable_joint_flags_R8 - static_joint_flags_R8 | -0.000 [-0.024, -0.002] | -0.033 [-0.040, -0.026] | +0.033 [+0.021, +0.041] |
| static_joint_flags_R8 - loupe_flags_original | +0.114 [+0.088, +0.126] | +0.102 [+0.077, +0.124] | +0.127 [+0.092, +0.152] |
