"""Custom torchmetrics metrics for MRI reconstruction evaluation."""

import torch
import torch.nn.functional as F
from torchmetrics import Metric
import skimage.metrics


class NMSE(Metric):
    """Normalized Mean Squared Error (NMSE) metric."""

    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_nmse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        batch_size = target.shape[0]
        for i in range(batch_size):
            gt = target[i].flatten()
            pred = preds[i].flatten()
            self.sum_nmse += torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2
            self.count += 1

    def compute(self):
        return self.sum_nmse / self.count


class MaskedPSNR(Metric):
    """Peak Signal-to-Noise Ratio computed only over masked (foreground) voxels."""

    full_state_update = False

    def __init__(self, data_range=1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = data_range
        self.add_state("sum_psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            raise ValueError("mask must be provided for MaskedPSNR")

        batch_size = target.shape[0]

        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        for i in range(batch_size):
            pred = preds[i]
            tgt = target[i]
            msk = mask[i]

            diff = (tgt - pred) ** 2
            masked_diff = (diff * msk).sum()
            masked_count = msk.sum()
            masked_count = torch.clamp(masked_count, min=1.0)

            mse = masked_diff / masked_count
            data_range = self.data_range if self.data_range > 0 else tgt.max()

            psnr = 10 * torch.log10((data_range ** 2) / mse)
            self.sum_psnr += psnr
            self.count += 1

    def compute(self):
        return self.sum_psnr / self.count


class MaskedSSIM(Metric):
    """Structural Similarity Index Measure computed only over masked (foreground) voxels."""

    full_state_update = False

    def __init__(self, data_range=1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = data_range
        self.add_state("sum_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            raise ValueError("mask must be provided for MaskedSSIM")

        batch_size = target.shape[0]

        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        for i in range(batch_size):
            pred = preds[i].cpu().numpy()
            tgt = target[i].cpu().numpy()
            msk = mask[i].cpu().numpy()

            if pred.ndim == 3:
                pred = pred.squeeze(0)
            if tgt.ndim == 3:
                tgt = tgt.squeeze(0)
            if msk.ndim == 3:
                msk = msk.squeeze(0)

            data_range = self.data_range if self.data_range > 0 else tgt.max()

            if msk.sum() == 0:
                ssim = torch.tensor(0.0, device=self.device)
            else:
                ssim = skimage.metrics.structural_similarity(
                    pred, tgt,
                    data_range=float(data_range),
                    mask=msk > 0.5,
                )
                ssim = torch.tensor(ssim, dtype=torch.float32, device=self.device)

            self.sum_ssim += ssim
            self.count += 1

    def compute(self):
        return self.sum_ssim / self.count


SKIMAGE_SSIM_WINDOW = 7
SKIMAGE_SSIM_K1 = 0.01
SKIMAGE_SSIM_K2 = 0.03


def ssim_map(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """skimage.metrics.structural_similarity's SSIM map of each [B, 1, H, W] slice pair, with its defaults and the
    target slice's maximum as the data range (a 7x7 uniform window, sample covariance, K1 0.01, K2 0.03), on the region
    skimage averages: the window's half-width cropped from every edge, [B, 1, H - 6, W - 6].

    Batched, on the inputs' device, in float64. Inside the cropped region every window lies within the image, so a
    stride-1 average pool is skimage's uniform filter there (a convolution would run in TF32 on the GPU).
    """
    x, y = preds.double(), target.double()
    data_range = y.amax(dim=(1, 2, 3), keepdim=True)

    def local_mean(image):
        return F.avg_pool2d(image, SKIMAGE_SSIM_WINDOW, stride=1)

    cov_norm = SKIMAGE_SSIM_WINDOW ** 2 / (SKIMAGE_SSIM_WINDOW ** 2 - 1)
    ux, uy = local_mean(x), local_mean(y)
    vx = cov_norm * (local_mean(x * x) - ux * ux)
    vy = cov_norm * (local_mean(y * y) - uy * uy)
    vxy = cov_norm * (local_mean(x * y) - ux * uy)
    c1 = (SKIMAGE_SSIM_K1 * data_range) ** 2
    c2 = (SKIMAGE_SSIM_K2 * data_range) ** 2
    return (2 * ux * uy + c1) * (2 * vxy + c2) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))


def masked_ssim(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SSIM [B] (float64) of each [B, 1, H, W] slice pair over its [B, 1, H, W] mask: `ssim_map` averaged over the
    mask's pixels in the map's region. A slice with no mask pixels there scores 0."""
    half = SKIMAGE_SSIM_WINDOW // 2
    weights = mask[..., half:-half, half:-half].double()
    count = weights.sum(dim=(1, 2, 3))
    total = (ssim_map(preds, target) * weights).sum(dim=(1, 2, 3))
    return torch.where(count > 0, total / count.clamp(min=1), 0.0)


class MaskedMetricAccumulator:
    """Per-slice PSNR and SSIM over the metric mask, averaged over every slice seen.

    Before 2026-10-23 SSIM_Masked was the SSIM of the whole slice, background included: it called skimage's
    structural_similarity, which has no mask argument and silently ignored the `mask=` passed to it.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.psnr_values = []
        self.ssim_values = []

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Add the batch's slices: preds, target and mask are [B, 1, H, W] or [B, H, W]."""
        preds, target, mask = (x.unsqueeze(1) if x.dim() == 3 else x for x in (preds, target, mask))
        mse = ((target - preds) ** 2 * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        psnr = 10 * torch.log10(target.amax(dim=(1, 2, 3)) ** 2 / mse)
        psnr, ssim = torch.stack([psnr.double(), masked_ssim(preds, target, mask)]).tolist()
        self.psnr_values += psnr
        self.ssim_values += ssim

    def compute(self):
        """Return mean PSNR and SSIM over accumulated batches."""
        psnr = torch.tensor(sum(self.psnr_values) / len(self.psnr_values)) if self.psnr_values else torch.tensor(0.0)
        ssim = torch.tensor(sum(self.ssim_values) / len(self.ssim_values)) if self.ssim_values else torch.tensor(0.0)
        return {"PSNR_Masked": psnr, "SSIM_Masked": ssim}

    def reset(self):
        """Reset accumulated values."""
        self.psnr_values = []
        self.ssim_values = []
