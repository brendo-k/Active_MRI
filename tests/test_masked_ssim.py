"""masked_ssim against skimage.metrics.structural_similarity, whose SSIM map it averages over the mask."""
import torch
from skimage.metrics import structural_similarity

from activemri.utils.metrics import SKIMAGE_SSIM_WINDOW, MaskedMetricAccumulator, masked_ssim

HALF = SKIMAGE_SSIM_WINDOW // 2


def noisy_pair(seed, size=64):
    generator = torch.Generator().manual_seed(seed)
    target = torch.rand(2, 1, size, size, generator=generator)
    return target + 0.1 * torch.randn(2, 1, size, size, generator=generator), target


def skimage_map(pred, target):
    _, full = structural_similarity(pred[0].numpy(), target[0].numpy(), data_range=float(target.max()), full=True)
    return torch.from_numpy(full).double()[HALF:-HALF, HALF:-HALF]


def test_a_full_mask_gives_skimage_ssim():
    preds, target = noisy_pair(0)
    expected = [structural_similarity(p[0].numpy(), t[0].numpy(), data_range=float(t.max())) for p, t in zip(preds, target)]
    assert torch.allclose(masked_ssim(preds, target, torch.ones_like(target)), torch.tensor(expected).double(), atol=1e-6)


def test_a_partial_mask_averages_skimage_ssim_map_over_its_pixels():
    preds, target = noisy_pair(1)
    mask = torch.zeros_like(target)
    mask[..., 10:40, 20:50] = 1
    inside = mask[0, 0, HALF:-HALF, HALF:-HALF] > 0
    expected = torch.stack([skimage_map(p, t)[inside].mean() for p, t in zip(preds, target)])
    assert torch.allclose(masked_ssim(preds, target, mask), expected, atol=1e-6)


def test_only_the_masked_region_counts_and_an_empty_mask_scores_zero():
    preds, target = noisy_pair(2)
    preds[..., :32, :] = target[..., :32, :]
    exact_rows = torch.zeros_like(target)
    exact_rows[..., :32 - SKIMAGE_SSIM_WINDOW, :] = 1
    assert torch.allclose(masked_ssim(preds, target, exact_rows), torch.ones(2).double())
    assert torch.equal(masked_ssim(preds, target, torch.zeros_like(target)), torch.zeros(2).double())

    accumulator = MaskedMetricAccumulator()
    accumulator.update(preds, target, exact_rows)
    assert abs(accumulator.compute()['SSIM_Masked'].item() - 1.0) < 1e-6
