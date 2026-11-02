"""The brain loader's up-down flip keeps k-space DC on the centre row, so it adds no phase ramp to the image."""
import torch

from activemri.datasets.fastmri_data import flip_rows_about_centre
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img


def test_dc_stays_on_the_centre_row():
    kspace = torch.zeros(3, 320, 320, dtype=torch.complex64)
    kspace[:, 160, 160] = 1
    assert torch.equal(flip_rows_about_centre(kspace), kspace)


def test_flipping_kspace_flips_the_image_with_no_phase_change():
    image = torch.randn(2, 320, 320, dtype=torch.complex64, generator=torch.Generator().manual_seed(0))
    from_kspace = ifft_2d_img(flip_rows_about_centre(fft_2d_img(image)))
    assert torch.allclose(from_kspace, flip_rows_about_centre(image), atol=1e-5)


def test_a_plain_flip_would_ramp_the_phase():
    image = torch.randn(320, 320, dtype=torch.complex64, generator=torch.Generator().manual_seed(0))
    from_kspace = ifft_2d_img(fft_2d_img(image).flip(-2))
    assert torch.allclose(from_kspace.abs(), flip_rows_about_centre(image).abs(), atol=1e-5)
    assert not torch.allclose(from_kspace, flip_rows_about_centre(image), atol=1e-3)
