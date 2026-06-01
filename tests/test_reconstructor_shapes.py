"""
Shape regression tests for CG-SENSE and SequentialUnet step_forward.

Each test targets a specific shape bug encountered during development:
  - CGSENSEReconstructor output must be [B, 1, H, W] real
  - conjugate_gradient_sense output must be [B, H, W] complex
  - step_forward 'zero_recon' must be [B, 1, H, W] real (compatible with loss target)
  - visualization dict entries must be 2-D numpy arrays (for imshow)
  - loss function must not receive a 5-D zero_filled tensor
"""

import importlib.util
import sys
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers to load modules without triggering the broken matplotlib __init__
# ---------------------------------------------------------------------------

def _load_reconstructors():
    spec = importlib.util.spec_from_file_location(
        "reconstructors",
        ROOT / "activemri/baselines/loupe_codes/reconstructors.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, H, W = 2, 4, 64, 64


@pytest.fixture()
def multicoil_inputs():
    kspace = torch.randn(B, C, H, W, dtype=torch.complex64)
    sense = torch.randn(B, C, H, W, dtype=torch.complex64)
    # normalise sense maps so they're vaguely realistic
    sense = sense / sense.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    mask = (torch.rand(B, 1, H, W) > 0.5).float()
    target = torch.rand(B, 1, H, W)
    return kspace, sense, mask, target


# ---------------------------------------------------------------------------
# CG-SENSE reconstructor
# ---------------------------------------------------------------------------

class TestCGSENSEShapes:
    def setup_method(self):
        self.mod = _load_reconstructors()

    def test_cg_function_output_shape(self, multicoil_inputs):
        kspace, sense, mask, _ = multicoil_inputs
        x = self.mod.conjugate_gradient_sense(kspace, sense, mask, num_iters=3)
        assert x.shape == (B, H, W), f"Expected ({B},{H},{W}), got {x.shape}"

    def test_cg_function_output_is_complex(self, multicoil_inputs):
        kspace, sense, mask, _ = multicoil_inputs
        x = self.mod.conjugate_gradient_sense(kspace, sense, mask, num_iters=3)
        assert x.is_complex()

    def test_cg_module_output_shape(self, multicoil_inputs):
        kspace, sense, mask, _ = multicoil_inputs
        recon = self.mod.CGSENSEReconstructor(num_iters=3)
        out = recon(kspace, sense, mask)
        assert out.shape == (B, 1, H, W), f"Expected ({B},1,{H},{W}), got {out.shape}"

    def test_cg_module_output_is_real(self, multicoil_inputs):
        kspace, sense, mask, _ = multicoil_inputs
        recon = self.mod.CGSENSEReconstructor(num_iters=3)
        out = recon(kspace, sense, mask)
        assert not out.is_complex(), "CGSENSEReconstructor should return a real magnitude tensor"

    def test_cg_module_output_nonnegative(self, multicoil_inputs):
        kspace, sense, mask, _ = multicoil_inputs
        recon = self.mod.CGSENSEReconstructor(num_iters=3)
        out = recon(kspace, sense, mask)
        assert (out >= 0).all(), "Magnitude output should be non-negative"

    def test_cg_with_lamda(self, multicoil_inputs):
        kspace, sense, mask, _ = multicoil_inputs
        recon = self.mod.CGSENSEReconstructor(num_iters=5, lamda=1e-3)
        out = recon(kspace, sense, mask)
        assert out.shape == (B, 1, H, W)

    def test_batch_independence(self, multicoil_inputs):
        """Each item in the batch should reconstruct independently."""
        kspace, sense, mask, _ = multicoil_inputs
        recon = self.mod.CGSENSEReconstructor(num_iters=5)
        out_batch = recon(kspace, sense, mask)

        out_single = recon(kspace[:1], sense[:1], mask[:1])
        assert torch.allclose(out_batch[:1], out_single, atol=1e-5)


# ---------------------------------------------------------------------------
# step_forward shapes (tested via a minimal SequentialUnet with recon_type='lls')
# ---------------------------------------------------------------------------

def _make_sequential_unet(recon_type="lls", shape=(H, W), num_step=2):
    from activemri.baselines.differential_sampler import SequentialUnet
    from activemri.reconstructors.CG_Sense import CGSENSEReconstructor
    from activemri.reconstructors.UNet import LOUPEUNet

    if recon_type == 'unet':
        reconstructor = LOUPEUNet(2, 2, 8, 2, 0.0)
    elif recon_type == 'lls':
        reconstructor = CGSENSEReconstructor(num_iters=3, lamda=0.0)
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

    return SequentialUnet(
        reconstructor=reconstructor,
        shape=list(shape),
        sparsity=0.25,
        num_step=num_step,
        preselect=False,
        recon_type=recon_type,
    ).eval()


@pytest.fixture()
def step_forward_inputs():
    """Minimal tensors to call step_forward directly."""
    from activemri.baselines.sequential_sampling_codes.image_processing import fft_2d_img

    target = torch.rand(B, 1, H, W)
    # build a plausible full_kspace [B, C, H, W, 2]
    img = torch.randn(B, C, H, W, dtype=torch.complex64)
    kspace_complex = fft_2d_img(img)
    full_kspace = torch.view_as_real(kspace_complex)

    sense = torch.randn(B, C, H, W, dtype=torch.complex64)
    sense = sense / sense.abs().pow(2).sum(dim=1, keepdim=True).sqrt()

    mask = (torch.rand(B, 1, H, W) > 0.5).float()
    # full_kspace is [B, C, H, W, 2]; mask needs a trailing dim to broadcast over the last (real/imag) axis
    pred_kspace = full_kspace * mask.unsqueeze(-1)  # [B, C, H, W, 2]
    old_recon = torch.zeros_like(target)
    uncertainty_map = torch.zeros_like(mask)

    return target, full_kspace, pred_kspace, mask, old_recon, uncertainty_map, sense


class TestStepForwardShapes:
    def _call_step(self, recon_type, step_forward_inputs):
        target, full_kspace, pred_kspace, mask, old_recon, uncertainty_map, sense = step_forward_inputs
        model = _make_sequential_unet(recon_type=recon_type)
        with torch.no_grad():
            pred_dict = model.step_forward(
                full_kspace, pred_kspace, mask, old_recon, target, step=0,
                sense_maps=sense, uncertainty_map=uncertainty_map,
            )
        return pred_dict

    @pytest.mark.parametrize("recon_type", ["lls", "unet"])
    def test_zero_recon_shape(self, recon_type, step_forward_inputs):
        """zero_recon must be [B, 1, H, W] — same as target — for the loss."""
        pred_dict = self._call_step(recon_type, step_forward_inputs)
        zr = pred_dict["zero_recon"]
        assert zr.shape == (B, 1, H, W), (
            f"[{recon_type}] zero_recon shape {zr.shape} incompatible with target {(B,1,H,W)}"
        )

    @pytest.mark.parametrize("recon_type", ["lls", "unet"])
    def test_zero_recon_is_real(self, recon_type, step_forward_inputs):
        pred_dict = self._call_step(recon_type, step_forward_inputs)
        assert not pred_dict["zero_recon"].is_complex(), (
            f"[{recon_type}] zero_recon should be a real tensor for loss comparison"
        )

    @pytest.mark.parametrize("recon_type", ["lls", "unet"])
    def test_mask_shape(self, recon_type, step_forward_inputs):
        pred_dict = self._call_step(recon_type, step_forward_inputs)
        assert pred_dict["mask"].shape == (B, 1, H, W)

    @pytest.mark.parametrize("recon_type", ["lls", "unet"])
    def test_vis_masked_kspace_is_2d(self, recon_type, step_forward_inputs):
        """imshow requires a 2-D array; masked_kspace[0] must be [H, W]."""
        model = _make_sequential_unet(recon_type=recon_type)
        target, full_kspace, pred_kspace, mask, old_recon, uncertainty_map, sense = step_forward_inputs
        with torch.no_grad():
            model.step_forward(
                full_kspace, pred_kspace, mask, old_recon, target, step=0,
                sense_maps=sense, uncertainty_map=uncertainty_map,
            )
        masked_k = model.data_for_vis[0]["masked_kspace"]
        assert masked_k.ndim >= 2
        frame = masked_k[0] if masked_k.ndim == 3 else masked_k
        assert frame.ndim == 2, (
            f"[{recon_type}] masked_kspace[0] has shape {masked_k[0].shape if masked_k.ndim==3 else masked_k.shape}, need 2-D for imshow"
        )

    @pytest.mark.parametrize("recon_type", ["lls", "unet"])
    def test_vis_zero_filled_recon_is_2d(self, recon_type, step_forward_inputs):
        model = _make_sequential_unet(recon_type=recon_type)
        target, full_kspace, pred_kspace, mask, old_recon, uncertainty_map, sense = step_forward_inputs
        with torch.no_grad():
            model.step_forward(
                full_kspace, pred_kspace, mask, old_recon, target, step=0,
                sense_maps=sense, uncertainty_map=uncertainty_map,
            )
        zfr = model.data_for_vis[0]["zero_filled_recon"]
        assert zfr.ndim == 2, f"[{recon_type}] zero_filled_recon vis entry has shape {zfr.shape}"


# ---------------------------------------------------------------------------
# Loss compatibility
# ---------------------------------------------------------------------------

class TestLossShapeCompatibility:
    @pytest.mark.parametrize("recon_type", ["lls", "unet"])
    def test_l1_loss_does_not_raise(self, recon_type, step_forward_inputs):
        """The loss function must accept zero_recon without a shape broadcast error."""
        import torch.nn.functional as F
        target, full_kspace, pred_kspace, mask, old_recon, uncertainty_map, sense = step_forward_inputs
        model = _make_sequential_unet(recon_type=recon_type)
        with torch.no_grad():
            pred_dict = model.step_forward(
                full_kspace, pred_kspace, mask, old_recon, target, step=0,
                sense_maps=sense, uncertainty_map=uncertainty_map,
            )
        zero_filled = pred_dict["zero_recon"]
        # must not raise RuntimeError about tensor size mismatch
        loss = F.l1_loss(zero_filled, target)
        assert loss.ndim == 0, "loss should be a scalar"
