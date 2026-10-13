"""Bit-for-bit equivalence of the in-house VarNet cascades against fastMRI's reference VarNet.

Run with ``.venv/bin/python -m unittest tests.test_varnet_vs_fastmri`` from the repo root.

Known, intentional differences (so the comparison maps parameters rather than assuming they match):
  * Sensitivity maps are supplied externally here; both models are fed the same maps.
  * The in-house output is the sensitivity-weighted complex coil combination, fastMRI returns the
    RSS magnitude, so the final k-space of the cascades is compared, plus the in-house combine.
"""
import copy
import sys
import unittest
from pathlib import Path

import torch
from fastmri import fft2c, ifft2c
from fastmri.models.varnet import VarNet as ReferenceVarNet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.VarNet import VarNet  # noqa: E402
from activemri.utils.image_processing import ifft_2d_img  # noqa: E402


def make_models(num_cascades, chans, pools, seed=0):
    torch.manual_seed(seed)
    ours = VarNet(num_cascades=num_cascades, chans=chans, pools=pools).eval()
    for cascade in ours.cascades:
        cascade.dc_weight.data = torch.rand(1) + 0.5

    reference = ReferenceVarNet(num_cascades=num_cascades, chans=chans, pools=pools).eval()
    for ref_cascade, our_cascade in zip(reference.cascades, ours.cascades):
        ref_cascade.model.unet.load_state_dict(our_cascade.model.unet.state_dict())
        ref_cascade.dc_weight.data = our_cascade.dc_weight.data.clone()
    return ours, reference


def make_inputs(batch, coils, height, width, seed=1):
    generator = torch.Generator().manual_seed(seed)
    kspace = torch.randn(batch, coils, height, width, dtype=torch.complex64, generator=generator)
    sense = torch.randn(batch, coils, height, width, dtype=torch.complex64, generator=generator)
    sense = sense / sense.abs().pow(2).sum(1, keepdim=True).sqrt()
    columns = torch.rand(batch, width, generator=generator) < 0.4
    columns[:, width // 2 - 3: width // 2 + 3] = True
    mask = columns[:, None, None, :].expand(batch, 1, height, width)
    return kspace * mask, sense, mask


def to_reference_layout(complex_tensor):
    return torch.view_as_real(complex_tensor).contiguous()


def reference_cascade_outputs(reference, kspace, sense, mask):
    ref_kspace = to_reference_layout(kspace)
    ref_sense = to_reference_layout(sense)
    ref_mask = mask[:, :1, :1, :, None].bool()
    current = ref_kspace.clone()
    outputs = []
    for cascade in reference.cascades:
        current = cascade(current, ref_kspace, ref_mask, ref_sense)
        outputs.append(current)
    return outputs


def our_cascade_outputs(ours, kspace, sense, mask):
    current = kspace.clone()
    outputs = []
    for cascade in ours.cascades:
        current = cascade(current, kspace, mask, sense)
        outputs.append(torch.view_as_real(current))
    return outputs


class VarNetMatchesFastMRI(unittest.TestCase):
    CONFIGS = [
        dict(num_cascades=3, chans=4, pools=2, batch=2, coils=4, height=36, width=40),
        dict(num_cascades=2, chans=8, pools=4, batch=1, coils=6, height=64, width=48),
    ]

    def test_each_cascade_is_bitwise_equal(self):
        for cfg in self.CONFIGS:
            with self.subTest(**cfg), torch.no_grad():
                ours, reference = make_models(cfg["num_cascades"], cfg["chans"], cfg["pools"])
                kspace, sense, mask = make_inputs(cfg["batch"], cfg["coils"], cfg["height"], cfg["width"])
                expected = reference_cascade_outputs(reference, kspace, sense, mask)
                actual = our_cascade_outputs(ours, kspace, sense, mask)
                for index, (a, e) in enumerate(zip(actual, expected)):
                    self.assertTrue(torch.equal(a, e), f"cascade {index}: max abs diff {(a - e).abs().max():.3e}")

    def test_forward_equals_manual_cascade_loop(self):
        cfg = self.CONFIGS[0]
        with torch.no_grad():
            ours, _ = make_models(cfg["num_cascades"], cfg["chans"], cfg["pools"])
            kspace, sense, mask = make_inputs(cfg["batch"], cfg["coils"], cfg["height"], cfg["width"])
            final_k = torch.view_as_complex(our_cascade_outputs(ours, kspace, sense, mask)[-1])
            expected = (ifft_2d_img(final_k) * sense.conj()).sum(dim=1, keepdim=True)
            self.assertTrue(torch.equal(ours(kspace, sense, mask.float()), expected))

    def test_dc_weight_defaults_to_one_like_fastmri(self):
        ours, reference = VarNet(num_cascades=2, chans=4, pools=2), ReferenceVarNet(num_cascades=2, chans=4, pools=2)
        for our_c, ref_c in zip(ours.cascades, reference.cascades):
            self.assertTrue(torch.equal(our_c.dc_weight, ref_c.dc_weight))

    def test_legacy_logit_checkpoint_is_converted(self):
        ours, _ = make_models(2, 4, 2)
        legacy = ours.state_dict()
        logits = [torch.tensor(0.3), torch.tensor(-1.2)]
        for index, logit in enumerate(logits):
            legacy[f"cascades.{index}.dc_weight"] = logit
        ours.load_state_dict(legacy)
        for cascade, logit in zip(ours.cascades, logits):
            self.assertTrue(torch.equal(cascade.dc_weight, torch.sigmoid(logit).reshape(1)))

    def test_fft_matches_fastmri(self):
        kspace, _, _ = make_inputs(2, 4, 36, 40)
        ref = to_reference_layout(kspace)
        self.assertTrue(torch.equal(torch.view_as_real(ifft_2d_img(kspace)), ifft2c(ref)))
        from activemri.utils.image_processing import fft_2d_img
        self.assertTrue(torch.equal(torch.view_as_real(fft_2d_img(kspace)), fft2c(ref)))

    def test_gradients_match_to_float32_rounding(self):
        # Backward runs through complex autograd here vs real-pair ops in fastMRI: equal up to rounding, not bitwise.
        cfg = self.CONFIGS[0]
        ours, reference = make_models(cfg["num_cascades"], cfg["chans"], cfg["pools"])
        ours.train(), reference.train()
        kspace, sense, mask = make_inputs(cfg["batch"], cfg["coils"], cfg["height"], cfg["width"])

        (our_cascade_outputs(ours, kspace, sense, mask)[-1] ** 2).sum().backward()
        (reference_cascade_outputs(reference, kspace, sense, mask)[-1] ** 2).sum().backward()

        for index, (our_c, ref_c) in enumerate(zip(ours.cascades, reference.cascades)):
            for (name, p_ours), (_, p_ref) in zip(our_c.model.unet.named_parameters(), ref_c.model.unet.named_parameters()):
                torch.testing.assert_close(p_ours.grad, p_ref.grad, rtol=1e-4, atol=1e-6 * p_ref.grad.abs().max().item(), msg=lambda m: f"cascade {index} {name}: {m}")


if __name__ == "__main__":
    unittest.main()
