"""Metric support mask: where PSNR/SSIM/NMSE are computed.

Built only from the fully sampled k-space and the anatomy, so it is identical for every run: Gaussian-apodize
the k-space, take the RSS, threshold at a fraction of its max, close, fill holes and keep the largest component.
"""
import torch
import torch.nn.functional as F

APODIZATION_SIGMA = 16
MASK_PARAMS = {'brain': {'fraction': 0.15, 'radius': 6}, 'knee': {'fraction': 0.06, 'radius': 12}}
PROPAGATION_STEPS_PER_CHECK = 16


def apodized_rss(kspace, sigma=APODIZATION_SIGMA):
    """RSS of Gaussian-windowed k-space; kspace is complex [B, C, H, W] with DC at the centre."""
    height, width = kspace.shape[-2:]
    ky = torch.arange(height, device=kspace.device) - height // 2
    kx = torch.arange(width, device=kspace.device) - width // 2
    window = torch.exp(-(ky[:, None] ** 2 + kx[None, :] ** 2).float() / (2 * sigma ** 2))
    image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace * window, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
    return image.abs().pow(2).sum(1).sqrt()


def _disk(radius, device):
    offsets = torch.arange(-radius, radius + 1, device=device)
    return ((offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius ** 2).float()[None, None]


def _dilate(mask, kernel):
    return F.conv2d(mask, kernel, padding=kernel.shape[-1] // 2) > 0


def _close(mask, radius):
    """Binary closing with a disk, on a zero-padded copy so the image border does not bleed into the mask."""
    kernel = _disk(radius, mask.device)
    padded = F.pad(mask.float(), (radius,) * 4)
    closed = ~_dilate((~_dilate(padded, kernel)).float(), kernel)
    return closed[..., radius:-radius, radius:-radius]


def _cross_max(values):
    """Max over each pixel and its four neighbours (4-connectivity)."""
    return torch.maximum(F.max_pool2d(values, (1, 3), stride=1, padding=(0, 1)), F.max_pool2d(values, (3, 1), stride=1, padding=(1, 0)))


def _propagate(state, allowed):
    """Spread `state` through `allowed` pixels with 4-connectivity until nothing changes."""
    while True:
        previous = state
        for _ in range(PROPAGATION_STEPS_PER_CHECK):
            state = _cross_max(state) * allowed
        if torch.equal(state, previous):
            return state


def _fill_holes(mask):
    background = ~mask
    border = torch.zeros_like(background)
    border[..., 0, :] = border[..., -1, :] = border[..., :, 0] = border[..., :, -1] = True
    outside = _propagate((background & border).float(), background.float()) > 0
    return mask | (background & ~outside)


def _largest_component(mask):
    batch, _, height, width = mask.shape
    allowed = mask.float()
    ids = torch.arange(1, height * width + 1, device=mask.device, dtype=torch.float32).view(1, 1, height, width) * allowed
    labels = _propagate(ids, allowed).long().flatten(1)
    sizes = torch.zeros(batch, height * width + 1, device=mask.device, dtype=torch.long)
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    sizes[:, 0] = 0
    largest = sizes.argmax(dim=1, keepdim=True)
    return ((labels == largest) & (labels > 0)).view(batch, 1, height, width)


def _group_mask(image, fraction, radius):
    peak = image.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    mask = image.unsqueeze(1) > fraction * peak
    return _largest_component(_fill_holes(_close(mask, radius)))


@torch.no_grad()
def metric_support_mask(kspace, anatomies):
    """[B, 1, H, W] float mask from complex fully sampled k-space [B, C, H, W] and the per-sample anatomy names."""
    unsupported = sorted(set(anatomies) - MASK_PARAMS.keys())
    if unsupported:
        raise ValueError(f'No metric mask parameters for anatomy {unsupported}; supported: {sorted(MASK_PARAMS)}')
    image = apodized_rss(kspace)
    mask = torch.zeros(image.shape[0], 1, *image.shape[1:], device=image.device)
    for anatomy, params in MASK_PARAMS.items():
        selected = torch.tensor([name == anatomy for name in anatomies], device=image.device)
        if selected.any():
            mask[selected] = _group_mask(image[selected], **params).float()
    return mask


def metric_mask_from_real_kspace(kspace, anatomies):
    """`metric_support_mask` for k-space stored as a real view [B, C, H, W, 2], as fastMRI batches carry it."""
    return metric_support_mask(torch.view_as_complex(kspace.contiguous()), list(anatomies))
