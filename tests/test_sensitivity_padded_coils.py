"""SensetivityModel_mc skips the all-zero coils the loader pads with, and that changes nothing.

Run with ``.venv/bin/python -m unittest tests.test_sensitivity_padded_coils`` from the repo root.

The reference runs the U-Net on every coil, as the model did before the skip. Float64, so batch-size-dependent
convolution roundoff stays far below the tolerances.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.SensetivityModel_mc import COIL_BUCKET, SensetivityModel_mc  # noqa: E402

BATCH, COILS, SIZE = 2, 16, 32
PHYSICAL_COILS = (4, 9)


def padded_kspace(seed=1):
    generator = torch.Generator().manual_seed(seed)
    kspace = torch.randn(BATCH, COILS, SIZE, SIZE, dtype=torch.complex128, generator=generator)
    for sample, physical in enumerate(PHYSICAL_COILS):
        kspace[sample, physical:] = 0
    return kspace


def run(model, kspace, weights, every_coil):
    if every_coil:
        model.coils_to_run = lambda images: torch.arange(images.shape[0])
    unet_batch = []
    hook = model.model.register_forward_hook(lambda module, inputs, output: unet_batch.append(inputs[0].shape[0]))
    model.zero_grad(set_to_none=True)
    maps, mask = model(kspace, mask_method='pre_network')
    (maps * weights).real.sum().backward()
    hook.remove()
    if every_coil:
        del model.coils_to_run
    return maps.detach(), mask, {name: p.grad.clone() for name, p in model.named_parameters()}, unet_batch[0]


class PaddedCoilsAreSkippedExactly(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SensetivityModel_mc(2, 2, 4).double()
        self.kspace = padded_kspace()
        self.weights = torch.randn_like(self.kspace)

    def test_unet_sees_live_coils_rounded_up_to_the_bucket(self):
        *_, unet_batch = run(self.model, self.kspace, self.weights, every_coil=False)
        live = sum(PHYSICAL_COILS)
        self.assertEqual(unet_batch, -(-live // COIL_BUCKET) * COIL_BUCKET)
        self.assertLess(unet_batch, BATCH * COILS)

    def test_maps_mask_and_gradients_match_running_every_coil(self):
        maps, mask, grads, _ = run(self.model, self.kspace, self.weights, every_coil=False)
        ref_maps, ref_mask, ref_grads, ref_batch = run(self.model, self.kspace, self.weights, every_coil=True)
        self.assertEqual(ref_batch, BATCH * COILS)
        self.assertTrue(torch.equal(mask, ref_mask))
        torch.testing.assert_close(maps, ref_maps)
        for name, grad in ref_grads.items():
            torch.testing.assert_close(grads[name], grad, msg=name)

    def test_padded_coils_get_zero_maps(self):
        maps, *_ = run(self.model, self.kspace, self.weights, every_coil=False)
        for sample, physical in enumerate(PHYSICAL_COILS):
            self.assertEqual(maps[sample, physical:].abs().max().item(), 0.0)
            self.assertGreater(maps[sample, :physical].abs().amax(dim=(-2, -1)).min().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
