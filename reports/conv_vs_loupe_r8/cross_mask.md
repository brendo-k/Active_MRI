Checkpoints: last_model.pt (conv epoch 50, LOUPE epoch 50). Mean mask density: conv 0.1249, loupe 0.1250, loupe_blur 0.1250.

| Reconstructor | Mask | Brain PSNR / SSIM (trainer) | Knee PSNR / SSIM (trainer) |
|---|---|---|---|
| loupe | loupe | 36.55 / 0.9328 | 34.26 / 0.9529 |
| loupe | conv | 36.35 / 0.9317 | 34.15 / 0.9518 |
| loupe | loupe_blur | 36.46 / 0.9321 | 34.23 / 0.9524 |
| loupe | loupe_reseed | 36.55 / 0.9328 | 34.26 / 0.9529 |
| conv | loupe | 36.23 / 0.9305 | 34.06 / 0.9509 |
| conv | conv | 36.28 / 0.9309 | 34.07 / 0.9511 |
| conv | loupe_blur | 36.06 / 0.9290 | 33.97 / 0.9496 |
| conv | conv_reseed | 36.28 / 0.9309 | 34.07 / 0.9511 |

| Paired difference | Brain PSNR (dB), 95% CI | Brain SSIM | Knee PSNR (dB), 95% CI | Knee SSIM |
|---|---|---|---|---|
| conv recon: conv_reseed - conv | +0.001 [-0.008, +0.010] | -0.0000 | +0.002 [-0.006, +0.010] | +0.0000 |
| conv recon: loupe - conv | -0.186 [-0.248, -0.125] | -0.0006 | -0.058 [-0.109, -0.004] | -0.0002 |
| conv recon: loupe_blur - conv | -0.534 [-0.633, -0.426] | -0.0021 | -0.356 [-0.456, -0.235] | -0.0017 |
| loupe recon: conv - loupe | -0.460 [-0.546, -0.372] | -0.0011 | -0.392 [-0.468, -0.288] | -0.0011 |
| loupe recon: loupe_blur - loupe | -0.175 [-0.199, -0.148] | -0.0008 | -0.120 [-0.137, -0.094] | -0.0006 |
| loupe recon: loupe_reseed - loupe | +0.000 [-0.008, +0.010] | +0.0000 | -0.004 [-0.013, +0.005] | -0.0000 |
| own vs own: conv - loupe | -0.585 [-0.685, -0.469] | -0.0018 | -0.574 [-0.718, -0.397] | -0.0020 |
