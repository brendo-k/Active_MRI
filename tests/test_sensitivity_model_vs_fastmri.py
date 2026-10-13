"""Bit-for-bit equivalence of SensetivityModel_mc against fastMRI's SensitivityModel.

Run with ``.venv/bin/python -m unittest tests.test_sensitivity_model_vs_fastmri`` from the repo root.

The in-house U-Net is a separate implementation with its own parameter layout, so weights are copied
layer by layer (fastMRI's final 1x1 conv has a bias, the in-house one does not; the reference bias is
zeroed). Known, intentional differences:
  * The in-house model always keeps a fixed 10x10 box of k-space centre; fastMRI keeps the central
    ``num_low_frequencies`` columns derived from the mask. The reference is therefore fed the already
    box-masked k-space with ``num_low_frequencies=10``, which selects the same columns.
  * The final RSS normalisation rounds differently (``abs().pow(2)`` here, squared real/imag parts in
    fastMRI), so the U-Net output is compared bitwise and the normalised maps to float32 rounding.
  * ``+1e-20`` in the normalisation and ``clamp(min=1e-8)`` on the RSS are no-ops for realistic data.
"""
import sys
import unittest
from pathlib import Path

import torch
from fastmri import ifft2c
from fastmri.models.varnet import SensitivityModel as ReferenceSensitivityModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc  # noqa: E402

CENTRE = 10


def copy_layers(sources, targets):
    for source, target in zip(sources, targets):
        for index in (0, 4):
            target[index].weight.data = source[index].weight.data.clone()


def copy_unet_weights(ours, reference):
    """Copy the in-house backbone `ours` into fastMRI's `reference` Unet."""
    our_down = [ours.down_sample_layers[0].layers] + [layer.conv.layers for layer in ours.down_sample_layers[1:]]
    ref_down = [block.layers for block in reference.down_sample_layers] + [reference.conv.layers]
    copy_layers(our_down, ref_down)

    our_up = [layer.conv.layers for layer in ours.up_sample_layers]
    ref_up = [block.layers for block in reference.up_conv[:-1]] + [reference.up_conv[-1][0].layers]
    copy_layers(our_up, ref_up)

    for our_layer, ref_block in zip(ours.up_sample_layers, reference.up_transpose_conv):
        ref_block.layers[0].weight.data = our_layer.up.layers[0].weight.data.clone()

    final = reference.up_conv[-1][1]
    final.weight.data = ours.conv1d.weight.data.clone()
    final.bias.data.zero_()


def make_models(chans, seed=0):
    torch.manual_seed(seed)
    ours = SensetivityModel_mc(2, 2, chans).eval()
    reference = ReferenceSensitivityModel(chans=chans, num_pools=4, mask_center=True).eval()
    copy_unet_weights(ours.model, reference.norm_unet.unet)
    return ours, reference


def make_kspace(batch, coils, height, width, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, coils, height, width, dtype=torch.complex64, generator=generator)


def reference_sense_maps(ours, reference, kspace):
    boxed = ours.mask_center(kspace)
    height, width = kspace.shape[-2:]
    mask = torch.zeros(kspace.shape[0], 1, 1, width, 1, dtype=torch.bool)
    mask[:, :, :, width // 2 - CENTRE // 2: width // 2 + CENTRE // 2] = True
    return reference(torch.view_as_real(boxed).contiguous(), mask, num_low_frequencies=CENTRE)


class SensitivityModelMatchesFastMRI(unittest.TestCase):
    CONFIGS = [
        dict(chans=4, batch=2, coils=4, height=36, width=40),
        dict(chans=8, batch=1, coils=6, height=64, width=48),
    ]

    def test_unet_output_is_bitwise_equal(self):
        for cfg in self.CONFIGS:
            with self.subTest(**cfg), torch.no_grad():
                ours, reference = make_models(cfg["chans"])
                kspace = make_kspace(cfg["batch"], cfg["coils"], cfg["height"], cfg["width"])

                captured = []
                original_unnorm = ours.unnorm
                ours.unnorm = lambda *args: captured.append(original_unnorm(*args)) or captured[-1]
                ours(kspace)
                ours.unnorm = original_unnorm

                ref_images = reference.chans_to_batch_dim(
                    ifft2c(torch.view_as_real(ours.mask_center(kspace)).contiguous())
                )[0]
                expected = reference.norm_unet(ref_images)
                actual = captured[0].reshape(expected.shape[0], 2, *expected.shape[2:4]).permute(0, 2, 3, 1).unsqueeze(1)
                self.assertTrue(torch.equal(actual, expected), f"max abs diff {(actual - expected).abs().max():.3e}")

    def test_sense_maps_match_to_float32_rounding(self):
        for cfg in self.CONFIGS:
            with self.subTest(**cfg), torch.no_grad():
                ours, reference = make_models(cfg["chans"])
                kspace = make_kspace(cfg["batch"], cfg["coils"], cfg["height"], cfg["width"])
                actual, _ = ours(kspace)
                expected = reference_sense_maps(ours, reference, kspace)
                torch.testing.assert_close(torch.view_as_real(actual), expected, rtol=1e-5, atol=1e-6)

    def test_center_mask_is_a_box_unlike_fastmri_column_strip(self):
        ours, _ = make_models(4)
        kept = ours.mask_center(torch.ones(1, 1, 36, 40, dtype=torch.complex64)).abs()[0, 0]
        self.assertEqual(int(kept.sum()), CENTRE * CENTRE)


if __name__ == "__main__":
    unittest.main()
