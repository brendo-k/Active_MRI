"""Custom torchmetrics metrics for MRI reconstruction evaluation."""

import torch
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


class MaskedMetricAccumulator:
    """Simple accumulator for masked PSNR/SSIM computed only over foreground voxels."""

    def __init__(self, device="cpu"):
        self.device = device
        self.psnr_values = []
        self.ssim_values = []

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Update metrics with batch, computing only over masked regions."""
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

            self._update_psnr(pred, tgt, msk)
            self._update_ssim(pred, tgt, msk)

    def _update_psnr(self, pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor):
        """Compute PSNR over masked region."""
        diff = (tgt - pred) ** 2
        masked_diff = (diff * mask).sum()
        masked_count = mask.sum()
        masked_count = torch.clamp(masked_count, min=1.0)

        mse = masked_diff / masked_count
        data_range = tgt.max()
        psnr = 10 * torch.log10((data_range ** 2) / mse)
        self.psnr_values.append(psnr.item())

    def _update_ssim(self, pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor):
        """Compute SSIM over masked region using skimage."""
        pred_np = pred.cpu().numpy()
        tgt_np = tgt.cpu().numpy()
        mask_np = mask.cpu().numpy()

        if pred_np.ndim == 3:
            pred_np = pred_np.squeeze(0)
        if tgt_np.ndim == 3:
            tgt_np = tgt_np.squeeze(0)
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)

        data_range = tgt_np.max()

        if mask_np.sum() == 0:
            ssim = 0.0
        else:
            ssim = skimage.metrics.structural_similarity(
                pred_np,
                tgt_np,
                data_range=float(data_range),
                mask=mask_np > 0.5,
            )

        self.ssim_values.append(ssim)

    def compute(self):
        """Return mean PSNR and SSIM over accumulated batches."""
        psnr = torch.tensor(sum(self.psnr_values) / len(self.psnr_values)) if self.psnr_values else torch.tensor(0.0)
        ssim = torch.tensor(sum(self.ssim_values) / len(self.ssim_values)) if self.ssim_values else torch.tensor(0.0)
        return {"PSNR_Masked": psnr, "SSIM_Masked": ssim}

    def reset(self):
        """Reset accumulated values."""
        self.psnr_values = []
        self.ssim_values = []
