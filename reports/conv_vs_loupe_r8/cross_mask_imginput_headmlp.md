Checkpoints: last_model.pt (conv epoch 50, LOUPE epoch 50). Mean mask density: conv 0.1250, loupe 0.1250, loupe_blur 0.1250.

| Reconstructor | Mask | Brain PSNR / SSIM (trainer) | Knee PSNR / SSIM (trainer) |
|---|---|---|---|
| loupe | loupe | 36.55 / 0.9328 | 34.26 / 0.9529 |
| loupe | conv | 36.06 / 0.9303 | 33.95 / 0.9503 |
| loupe | loupe_blur | 36.46 / 0.9321 | 34.23 / 0.9524 |
| loupe | loupe_reseed | 36.55 / 0.9328 | 34.26 / 0.9529 |
| conv | loupe | 29.56 / 0.8836 | 29.23 / 0.8945 |
| conv | conv | 36.36 / 0.9319 | 34.15 / 0.9516 |
| conv | loupe_blur | 29.55 / 0.8836 | 29.18 / 0.8940 |
| conv | conv_reseed | 36.39 / 0.9320 | 34.15 / 0.9516 |

| Paired difference | Brain PSNR (dB), 95% CI | Brain SSIM | Knee PSNR (dB), 95% CI | Knee SSIM |
|---|---|---|---|---|
| conv recon: conv_reseed - conv | +0.001 [-0.008, +0.013] | +0.0001 | +0.004 [-0.002, +0.012] | -0.0000 |
| conv recon: loupe - conv | -9.158 [-10.447, -7.671] | -0.0470 | -8.583 [-10.200, -6.664] | -0.0591 |
| conv recon: loupe_blur - conv | -9.204 [-10.463, -7.751] | -0.0472 | -8.686 [-10.307, -6.638] | -0.0596 |
| loupe recon: conv - loupe | -1.064 [-1.241, -0.858] | -0.0025 | -0.880 [-0.973, -0.736] | -0.0029 |
| loupe recon: loupe_blur - loupe | -0.175 [-0.199, -0.148] | -0.0008 | -0.120 [-0.137, -0.094] | -0.0006 |
| loupe recon: loupe_reseed - loupe | +0.000 [-0.008, +0.010] | +0.0000 | -0.004 [-0.013, +0.005] | -0.0000 |
| own vs own: conv - loupe | -0.337 [-0.427, -0.249] | -0.0009 | -0.418 [-0.508, -0.298] | -0.0015 |
