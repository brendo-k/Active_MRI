"""The VarNet's output and gradients against the centred formulation, in which every FFT shifts in and out.

`centred_reference` writes out VarNet.forward with fft_2d_img / ifft_2d_img around every FFT, using the model's own
modules, so the comparison holds for whatever k-space layout the cascades use internally. Shifts only move elements,
so at an even size, in float32, the output and nearly every gradient must be bit-for-bit equal. The exception is a
scalar DC weight's gradient, a sum over every k-space location, which a different element order rounds differently.

An odd size runs in float64 and matches to rounding: the CPU's vectorised loops finish a row whose length is not a
multiple of the vector width with a scalar tail, whose complex multiply rounds differently (no fused multiply-add),
and a shift moves other elements into that tail.

Run with ``python -m pytest tests/test_varnet_uncentered_kspace.py`` from the repo root.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.VarNet import VarNet  # noqa: E402
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img  # noqa: E402

BATCH, COILS = 2, 3
# (height, width, dtype): even and odd, since fftshift and ifftshift differ for odd sizes.
CASES = [(32, 24, torch.float32), (33, 27, torch.float64)]
# Odd sizes: the tolerance relative to the largest value, so that gradients which are zero but for rounding (a conv
# bias before an instance norm) do not count as mismatches.
ODD_SIZE_TOLERANCE = 1e-12
CONFIGS = {
    "where": dict(),
    "multiply_dc_map_support": dict(dc_mode="multiply", dc_weight_map=True, support=True),
    "adaptive_momentum_checkpointed": dict(adaptive_input=True, momentum_history=2, checkpoint_cascades=True),
    "csme": dict(csme=True, support=True),
    "csme_adaptive_multiply": dict(csme=True, adaptive_input=True, dc_mode="multiply"),
}


def centred_reference(model, masked_kspace, sense_maps, mask, full_kspace=None, kspace_support=None):
    """VarNet.forward on centred k-space, every FFT an fft_2d_img / ifft_2d_img."""
    def sens_reduce(kspace, maps):
        return (ifft_2d_img(kspace) * maps.conj()).sum(dim=1, keepdim=True)

    # Same order of operations as VarNet.forward: autograd sums the gradients of a tensor's consumers in the order
    # they were created, so another order rounds the input gradients differently.
    adaptive = model.adaptive_input
    ref_kspace = full_kspace if model.dc_mode == "multiply" else masked_kspace
    kspace = masked_kspace.clone()
    if model.csme:
        sense_maps = model.cascades[0].sense_model(ifft_2d_img(masked_kspace)) if adaptive else None
    zero_filled = latent = history = None
    if adaptive:
        zero_filled = latent = sens_reduce(masked_kspace, sense_maps)
    for cascade in model.cascades:
        if cascade.sense_model is not None and sense_maps is None:
            sense_maps = cascade.sense_model(ifft_2d_img(kspace))
        if model.dc_mode == "multiply":
            soft_dc = mask * (kspace - ref_kspace) * cascade.dc_weight
        else:
            zero = torch.zeros(1, device=kspace.device, dtype=kspace.dtype)
            soft_dc = torch.where(mask.bool(), kspace - ref_kspace, zero) * cascade.dc_weight
        image = sens_reduce(kspace, sense_maps)
        if adaptive:
            image = torch.cat([image, sens_reduce(mask * kspace, sense_maps), latent, zero_filled], dim=1)
        if cascade.has_momentum:
            output, history = cascade.model(image, None, history)
        else:
            output = cascade.model(image, None)
        correction, latent = output.chunk(2, dim=1) if adaptive else (output, latent)
        model_term = fft_2d_img(correction * sense_maps)
        if kspace_support is not None:
            model_term = model_term * kspace_support
        kspace = kspace - soft_dc - model_term
        used_maps = sense_maps
        if model.csme:
            sense_maps = None
    return sens_reduce(kspace, used_maps)


def make_model(config, height, width, dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    model = VarNet(
        num_cascades=3, chans=4, pools=2, dc_mode=config.get("dc_mode", "where"),
        dc_weight_shape=(height, width) if config.get("dc_weight_map") else None,
        adaptive_input=config.get("adaptive_input", False), momentum_history=config.get("momentum_history", 0),
        csme=config.get("csme", False),
        checkpoint_cascades=config.get("checkpoint_cascades", False),
    ).train()
    # Random rather than all-ones DC weights, so a DC weight map applied at the wrong k-space location shows.
    for cascade in model.cascades:
        cascade.dc_weight.data = torch.rand_like(cascade.dc_weight) + 0.5
    return model.to(device=device, dtype=dtype)


def make_inputs(height, width, dtype=torch.float32, device="cpu", seed=1):
    generator = torch.Generator().manual_seed(seed)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    full_kspace = torch.randn(BATCH, COILS, height, width, dtype=complex_dtype, generator=generator)
    sense = torch.randn(BATCH, COILS, height, width, dtype=complex_dtype, generator=generator)
    sense = sense / sense.abs().pow(2).sum(1, keepdim=True).sqrt()
    columns = torch.rand(BATCH, width, generator=generator) < 0.4
    columns[:, width // 2 - 3:width // 2 + 3] = True  # an ACS block
    mask = columns[:, None, None, :].expand(BATCH, 1, height, width).to(dtype)
    support = (torch.rand(BATCH, 1, height, width, generator=generator) < 0.9).to(dtype)
    return tuple(tensor.to(device) for tensor in (full_kspace, sense, mask, support))


def output_and_gradients(forward, model, inputs, use_full, use_support):
    model.zero_grad(set_to_none=True)
    full_kspace, sense, mask, support = (tensor.clone().requires_grad_() for tensor in inputs)
    # The mask also gets gradient through the masked k-space, as the sampler's does in training.
    output = forward(full_kspace * mask, sense, mask, full_kspace if use_full else None,
                     support if use_support else None)
    torch.view_as_real(output).pow(2).sum().backward()
    gradients = {name: p.grad for name, p in model.named_parameters()}
    gradients.update(full_kspace=full_kspace.grad, sense_maps=sense.grad, mask=mask.grad, support=support.grad)
    return output.detach(), gradients





@pytest.mark.parametrize("height,width,dtype", CASES)
@pytest.mark.parametrize("name", CONFIGS)
def test_output_and_gradients_match_the_centred_formulation(name, height, width, dtype):
    check_against_centred_reference(name, height, width, dtype)





def check_against_centred_reference(name, height, width, dtype):
    """Asserts that VarNet's output and gradients are those of `centred_reference`, and returns the output."""
    config = CONFIGS[name]
    model = make_model(config, height, width, dtype)
    inputs = make_inputs(height, width, dtype)
    use_full, use_support = config.get("dc_mode") == "multiply", config.get("support", False)
    exact = height % 2 == 0 and width % 2 == 0

    def model_forward(masked, sense, mask, full, support):
        return model(masked, sense, mask, full_kspace=full, kspace_support=support)

    def reference_forward(masked, sense, mask, full, support):
        return centred_reference(model, masked, sense, mask, full_kspace=full, kspace_support=support)

    output, gradients = output_and_gradients(model_forward, model, inputs, use_full, use_support)
    expected, expected_gradients = output_and_gradients(reference_forward, model, inputs, use_full, use_support)

    def matches(actual, reference, scale):
        if exact:
            return torch.equal(actual, reference)
        return torch.allclose(actual, reference, rtol=ODD_SIZE_TOLERANCE, atol=ODD_SIZE_TOLERANCE * scale)

    assert matches(output, expected, expected.abs().max()), f"output: max abs diff {(output - expected).abs().max():.3e}"
    assert gradients.keys() == expected_gradients.keys()
    largest = max(gradient.abs().max() for gradient in expected_gradients.values() if gradient is not None)
    mismatches = []
    for key, gradient in gradients.items():
        reference = expected_gradients[key]
        if gradient is None or reference is None:
            if not (gradient is None and reference is None):
                mismatches.append(f"{key}: gradient on one side only")
        elif exact and key.endswith("dc_weight") and gradient.numel() == 1:
            if not torch.allclose(gradient, reference, rtol=1e-5, atol=0):
                mismatches.append(f"{key}: {gradient.item():.8e} vs {reference.item():.8e}")
        elif not matches(gradient, reference, largest):
            mismatches.append(f"{key}: max abs diff {(gradient - reference).abs().max():.2e}, "
                              f"largest gradient {largest:.2e}")
    assert not mismatches, "\n".join(mismatches)
    return output
