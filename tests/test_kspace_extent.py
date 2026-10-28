"""The model's output is formed like the target: the RSS of coil k-space zero-filled outside the acquired extent.

Run with ``.venv/bin/python -m unittest tests.test_kspace_extent`` from the repo root.

fastMRI FLAIR acquires fewer phase encodes (and on 512-readout scans fewer readout rows) than the 320 grid, so the
loader zero-fills the rest, and its target is the RSS of that zero-filled k-space. A reconstructor that returns the
true image must then score a perfect match. The VarNet's regularizer is still left free outside the extent: VarNets
trained without the restriction move up to half their k-space energy there between cascades, and confining them at
evaluation loses ~1.5 dB on FLAIR. Training with it is untested.
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.models.sequential_sampling import SequentialSampling  # noqa: E402
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img  # noqa: E402

BATCH, COILS, SIZE = 2, 5, 32
LIVE_COILS = 4
ACQUIRED_ROWS = slice(4, 28)
ACQUIRED_COLS = slice(7, 25)


class TrueImageReconstructor(nn.Module):
    """Returns the true image whatever it is given, and records the k-space support it was passed (None = free)."""

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.kspace_support = None

    def forward(self, masked_kspace, sense_maps, mask, kspace_support=None):  # noqa: ARG002
        self.kspace_support = kspace_support
        return self.image


def acquired_extent():
    extent = torch.zeros(BATCH, 1, SIZE, SIZE)
    extent[..., ACQUIRED_ROWS, ACQUIRED_COLS] = 1
    return extent


def zero_filled_scan(seed=0):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(BATCH, 1, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    maps = torch.randn(BATCH, COILS, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    maps[:, LIVE_COILS:] = 0
    kspace = fft_2d_img(image * maps) * acquired_extent()
    return image, maps, kspace


def model_for(image):
    return SequentialSampling(TrueImageReconstructor(image), shape=(SIZE, SIZE), sparsity=0.25, sampler_type='loupe_2d',
                              num_step=1, preselect_num=4, recon_type='varnet')


class OutputMatchesZeroFilledTarget(unittest.TestCase):
    def test_extent_is_the_acquired_rectangle(self):
        _, _, kspace = zero_filled_scan()
        model = model_for(None)
        torch.testing.assert_close(model._kspace_extent(torch.view_as_real(kspace)), acquired_extent())

    def test_extent_ignores_points_missing_inside_it(self):
        _, _, kspace = zero_filled_scan()
        holdout = torch.rand(BATCH, 1, SIZE, SIZE, generator=torch.Generator().manual_seed(1)) < 0.4
        model = model_for(None)
        torch.testing.assert_close(model._kspace_extent(torch.view_as_real(kspace * ~holdout)), acquired_extent())

    def test_true_image_scores_the_target_exactly(self):
        image, maps, kspace = zero_filled_scan()
        target = ifft_2d_img(kspace).abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        model = model_for(image).eval()
        with torch.no_grad():
            output = model(target, torch.view_as_real(kspace), sense_maps=maps)['output']
        torch.testing.assert_close(output, target, rtol=1e-4, atol=1e-4)
        self.assertIsNone(model.reconstructor.kspace_support)

    def test_unprojected_output_does_not(self):
        image, maps, kspace = zero_filled_scan()
        target = ifft_2d_img(kspace).abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        unprojected = image.abs() * maps.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        self.assertGreater((unprojected - target).abs().max().item(), 0.1)


if __name__ == '__main__':
    unittest.main()
