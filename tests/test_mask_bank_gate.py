import torch

from activemri.samplers.mask_bank import MaskBank

SHAPE = (32, 32)


def kspace(batch=6):
    torch.manual_seed(0)
    return torch.randn(batch, *SHAPE, 4)  # NHWC; the gate reads the centre block


def pushed(bank, scale=20.0):
    """Drive the gate head's bias so mask 0 dominates, as training does when the gate collapses."""
    with torch.no_grad():
        bank.gate_head.bias.copy_(torch.tensor([scale] + [0.0] * (bank.bank_size - 1)))
    return bank


def test_defaults_are_a_plain_softmax():
    bank = MaskBank(SHAPE, 5.0, 8, bank_size=4)
    x = kspace()
    w = bank.weights(x)
    assert torch.allclose(w, torch.full_like(w, 0.25))  # zero-initialised head: uniform
    pushed(bank)
    plain = torch.softmax(bank.gate_head(bank.gate(bank.calibration_features(x))), dim=-1)
    assert torch.allclose(bank.weights(x), plain)


def test_unbounded_gate_starves_the_other_masks_but_the_limit_does_not():
    x = kspace()
    starved = pushed(MaskBank(SHAPE, 5.0, 8, bank_size=4))
    bounded = pushed(MaskBank(SHAPE, 5.0, 8, bank_size=4, gate_logit_limit=3.0, gate_temperature=1.0))
    assert starved.weights(x)[:, 1:].max() < 1e-8
    assert bounded.weights(x)[:, 1:].min() > 3e-4  # >= exp(-2 * 3) / (1 + ...) relative floor
    for bank, expect_grad in ((starved, False), (bounded, True)):
        bank.zero_grad()
        bank(x).sum().backward()
        grad = bank.mask.grad[1:].abs().max().item()
        assert (grad > 1e-6) == expect_grad


def test_temperature_flattens():
    x = kspace()
    sharp = pushed(MaskBank(SHAPE, 5.0, 8, bank_size=4, gate_temperature=1.0), scale=4.0)
    flat = pushed(MaskBank(SHAPE, 5.0, 8, bank_size=4, gate_temperature=4.0), scale=4.0)
    assert flat.weights(x).max() < sharp.weights(x).max()
