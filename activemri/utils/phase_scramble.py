"""Smooth random image-domain phase, applied identically to every coil.

A phase field shared by all coils leaves the RSS magnitude image (the training
target) unchanged, but breaks the conjugate symmetry of k-space that a real-valued
image would have. Bands above the sensitivity net's centre-k-space bandwidth
(+-5 cycles/FOV) cannot be absorbed into the estimated sensitivity maps.
"""
import torch

from activemri.utils.image_processing import fft_2d_img, ifft_2d_img


def random_phase_field(batch, shape, low, high, std, device, generator=None):
    """Band-limited Gaussian field [batch, H, W] with `std` radians, spatial frequencies in cycles/FOV."""
    height, width = shape
    noise = torch.randn(batch, height, width, device=device, generator=generator)
    freq_y = torch.fft.fftfreq(height, device=device) * height
    freq_x = torch.fft.fftfreq(width, device=device) * width
    radius = torch.hypot(freq_y[:, None], freq_x[None, :])
    band = ((radius >= low) & (radius <= high)).float()
    field = torch.fft.ifft2(torch.fft.fft2(noise) * band).real
    return std * field / field.std(dim=(-2, -1), keepdim=True)


def scramble_phase(kspace, field):
    """Multiply every coil image by exp(i * field). kspace: complex [B, C, H, W]."""
    return fft_2d_img(ifft_2d_img(kspace) * torch.exp(1j * field)[:, None])


def scramble_phase_real(kspace, std, band, generator=None):
    """`scramble_phase` on the real-view [B, C, H, W, 2] layout the dataloaders produce."""
    complex_kspace = torch.view_as_complex(kspace.contiguous())
    field = random_phase_field(
        kspace.shape[0], kspace.shape[2:4], band[0], band[1], std, kspace.device, generator)
    return torch.view_as_real(scramble_phase(complex_kspace, field))
