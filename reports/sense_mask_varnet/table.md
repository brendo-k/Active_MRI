| arm | epochs | PSNR_Masked | SSIM_Masked | NMSE | brain PSNR | knee PSNR | brain NMSE | knee NMSE |
|---|---|---|---|---|---|---|---|---|
| pre_network | 50 | 31.26 | 0.8154 | 0.0127 | 35.79 | 28.99 | 0.0105 | 0.0136 |
| null | 50 | 32.52 | 0.8395 | 0.0068 | 38.01 | 30.97 | 0.0057 | 0.0073 |
| post_network | 50 | 31.92 | 0.8281 | 0.0088 | 37.35 | 29.92 | 0.0070 | 0.0097 |

Difference from `pre_network` (arm minus control; negative NMSE is better):

| arm | PSNR_Masked | SSIM_Masked | NMSE | brain PSNR | knee PSNR | brain NMSE | knee NMSE |
|---|---|---|---|---|---|---|---|
| null | +1.26 | +0.0242 | -0.0059 | +2.22 | +1.98 | -0.0048 | -0.0064 |
| post_network | +0.67 | +0.0127 | -0.0038 | +1.56 | +0.93 | -0.0035 | -0.0040 |
