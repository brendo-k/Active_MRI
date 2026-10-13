"""End-to-end equivalence of masked k-space -> sensitivity model -> VarNet -> image against fastMRI.

Run with ``.venv/bin/python -m unittest tests.test_end_to_end_vs_fastmri`` from the repo root.

The in-house pipeline is ``SensetivityModel_mc`` followed by the ``VarNet`` cascades. fastMRI's
``VarNet.forward`` returns the RSS magnitude of the final k-space, so the in-house final k-space is
passed through the same finishing step before comparison. Two strengths of comparison:
  * bitwise: the reference's sensitivity net is replaced by a stub returning the in-house maps, which
    isolates cascades, layouts, masks and the final combine from the 1 ulp RSS rounding difference;
  * tolerance: the unmodified reference, with k-space zeroed in the centre columns outside the 10x10
    box so that fastMRI's column strip and the in-house box select the same calibration data.
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from fastmri import complex_abs, ifft2c, rss, rss_complex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc  # noqa: E402
from activemri.utils.image_processing import fft_2d_img  # noqa: E402
from tests.test_sensitivity_model_vs_fastmri import CENTRE, copy_unet_weights  # noqa: E402
from tests.test_varnet_vs_fastmri import make_inputs, make_models, our_cascade_outputs  # noqa: E402

THRESHOLD = 0.1
POST_NETWORK_THRESHOLD = 0.9  # an untrained network's output is too flat for 0.1 to mask anything


class StubSensitivityModel(nn.Module):
    def __init__(self, maps):
        super().__init__()
        self.maps = maps

    def forward(self, *args, **kwargs):
        return self.maps


def make_pipeline(cfg):
    ours_varnet, reference = make_models(cfg["num_cascades"], cfg["chans"], cfg["pools"])
    ours_sense = SensetivityModel_mc(2, 2, 8).eval()
    copy_unet_weights(ours_sense.model, reference.sens_net.norm_unet.unet)
    return ours_sense, ours_varnet, reference


def box_calibration_only(kspace):
    """Zero centre-column samples outside the 10x10 box so strip and box calibration coincide."""
    height, width = kspace.shape[-2:]
    outside_box_rows = torch.ones(height, dtype=torch.bool)
    outside_box_rows[height // 2 - CENTRE // 2: height // 2 + CENTRE // 2] = False
    centre_columns = torch.zeros(width, dtype=torch.bool)
    centre_columns[width // 2 - CENTRE // 2: width // 2 + CENTRE // 2] = True
    return kspace * ~(outside_box_rows[:, None] & centre_columns[None, :])


def disc_phantom_kspace(batch, coils, height, width, radius=8, seed=2):
    """Multi-coil k-space of a disc, so the signal support (and hence the sense mask) is nontrivial."""
    generator = torch.Generator().manual_seed(seed)
    rows = torch.arange(height)[:, None] - height // 2
    columns = torch.arange(width)[None, :] - width // 2
    disc = ((rows ** 2 + columns ** 2) <= radius ** 2).float()
    coil_gains = torch.randn(batch, coils, 1, 1, dtype=torch.complex64, generator=generator)
    return fft_2d_img(coil_gains * disc)


def our_image(ours_sense, ours_varnet, kspace, mask, mask_method=None):
    sense_maps, _ = ours_sense(kspace, mask_method=mask_method)
    final_kspace = torch.view_as_complex(our_cascade_outputs(ours_varnet, kspace, sense_maps, mask)[-1])
    return rss(complex_abs(ifft2c(torch.view_as_real(final_kspace))), dim=1), sense_maps


def reference_mask(mask):
    return mask[:, :1, :1, :, None].bool()


class EndToEndMatchesFastMRI(unittest.TestCase):
    CONFIGS = [
        dict(num_cascades=3, chans=4, pools=2, batch=2, coils=4, height=36, width=40),
        dict(num_cascades=2, chans=8, pools=4, batch=1, coils=6, height=64, width=48),
    ]

    def inputs(self, cfg):
        kspace, _, mask = make_inputs(cfg["batch"], cfg["coils"], cfg["height"], cfg["width"])
        return box_calibration_only(kspace), mask

    def test_image_is_bitwise_equal_given_identical_sense_maps(self):
        for cfg in self.CONFIGS:
            for mask_method in (None, "pre_network", "post_network"):
                with self.subTest(mask_method=mask_method, **cfg), torch.no_grad():
                    ours_sense, ours_varnet, reference = make_pipeline(cfg)
                    kspace, mask = self.inputs(cfg)
                    actual, sense_maps = our_image(ours_sense, ours_varnet, kspace, mask, mask_method)
                    reference.sens_net = StubSensitivityModel(torch.view_as_real(sense_maps).contiguous())
                    expected = reference(torch.view_as_real(kspace).contiguous(), reference_mask(mask))
                    self.assertEqual(actual.shape, (cfg["batch"], cfg["height"], cfg["width"]))
                    self.assertTrue(torch.equal(actual, expected), f"max abs diff {(actual - expected).abs().max():.3e}")

    def test_image_matches_unmodified_fastmri_to_float32_rounding(self):
        for cfg in self.CONFIGS:
            with self.subTest(**cfg), torch.no_grad():
                ours_sense, ours_varnet, reference = make_pipeline(cfg)
                kspace, mask = self.inputs(cfg)
                actual, _ = our_image(ours_sense, ours_varnet, kspace, mask)
                expected = reference(
                    torch.view_as_real(kspace).contiguous(), reference_mask(mask), num_low_frequencies=CENTRE
                )
                scale = expected.abs().max().item()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5 * scale)

    def test_native_output_is_complex_sense_combination_of_final_kspace(self):
        cfg = self.CONFIGS[0]
        with torch.no_grad():
            ours_sense, ours_varnet, _ = make_pipeline(cfg)
            kspace, mask = self.inputs(cfg)
            sense_maps, _ = ours_sense(kspace)
            output = ours_varnet(kspace, sense_maps, mask.float())
            self.assertEqual(output.shape, (cfg["batch"], 1, cfg["height"], cfg["width"]))
            self.assertEqual(output.dtype, torch.complex64)

            final_kspace = torch.view_as_complex(our_cascade_outputs(ours_varnet, kspace, sense_maps, mask)[-1])
            expected = (torch.view_as_complex(ifft2c(torch.view_as_real(final_kspace))) * sense_maps.conj()).sum(1, keepdim=True)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


class MaskedSenseMapsAreConsistent(unittest.TestCase):
    cfg = dict(num_cascades=2, chans=4, pools=2, batch=2, coils=4, height=36, width=40)

    def setUp(self):
        torch.manual_seed(0)
        self.sense = SensetivityModel_mc(2, 2, 8).eval()
        self.kspace = disc_phantom_kspace(self.cfg["batch"], self.cfg["coils"], self.cfg["height"], self.cfg["width"])

    def maps(self, mask_method):
        with torch.no_grad():
            return self.sense(self.kspace, mask_method=mask_method)

    def test_output_shapes_and_dtypes(self):
        maps, mask = self.maps("pre_network")
        self.assertEqual(maps.shape, self.kspace.shape)
        self.assertEqual(maps.dtype, torch.complex64)
        self.assertEqual(mask.shape, (self.cfg["batch"], 1, self.cfg["height"], self.cfg["width"]))
        self.assertEqual(sorted(mask.unique().tolist()), [0.0, 1.0], "phantom should give a nontrivial mask")
        self.assertIsNone(self.maps(None)[1])

    def test_pre_network_mask_is_thresholded_rss_of_centre_image(self):
        _, mask = self.maps("pre_network")
        rss_image = rss_complex(ifft2c(torch.view_as_real(self.sense.mask_center(self.kspace))), dim=1)
        expected = (rss_image > THRESHOLD * rss_image.amax(dim=(-2, -1), keepdim=True)).float().unsqueeze(1)
        self.assertTrue(torch.equal(mask, expected))

    def test_post_network_mask_is_thresholded_rss_of_network_output(self):
        from tests.test_sensitivity_model_vs_fastmri import make_models as make_sense_models

        torch.manual_seed(0)
        ours, reference = make_sense_models(8)
        ours.eval()
        with torch.no_grad():
            _, mask = ours(self.kspace, mask_method="post_network", mask_threshold=POST_NETWORK_THRESHOLD)
            images, batch = reference.chans_to_batch_dim(ifft2c(torch.view_as_real(ours.mask_center(self.kspace))))
            output = reference.batch_chans_to_chan_dim(reference.norm_unet(images), batch)
            rss_output = rss_complex(output, dim=1)
        expected = (rss_output > POST_NETWORK_THRESHOLD * rss_output.amax(dim=(-2, -1), keepdim=True)).float().unsqueeze(1)
        self.assertEqual(sorted(mask.unique().tolist()), [0.0, 1.0], "threshold should give a nontrivial mask")
        self.assertTrue(torch.equal(mask, expected))

    def test_masked_maps_are_unmasked_maps_times_mask(self):
        plain, _ = self.maps(None)
        for mask_method in ("pre_network", "post_network"):
            with self.subTest(mask_method=mask_method):
                masked, mask = self.maps(mask_method)
                self.assertTrue(torch.equal(masked, plain * mask))
                self.assertTrue(torch.all(masked[(mask == 0).expand_as(masked)] == 0))


if __name__ == "__main__":
    unittest.main()
