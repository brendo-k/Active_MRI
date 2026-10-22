import torch

from activemri.reconstructors.CG_Sense import CGSENSEReconstructor
from activemri.samplers.mask_bank import MaskBank
from activemri.samplers.sampler import Sampler
from activemri.training.trainer import _cg_iters_by_label


def test_cg_iters_by_label_matches_separate_runs():
    torch.manual_seed(0)
    b, c, h, w = 4, 3, 16, 16
    sense = torch.randn(b, c, h, w, dtype=torch.complex64)
    kspace = torch.randn(b, c, h, w, dtype=torch.complex64)
    mask = (torch.rand(b, 1, h, w) < 0.4).float()
    labels = torch.tensor([0, 1, 1, 0])
    mixed = CGSENSEReconstructor(lamda=1e-3, iters_by_label=[8, 5])(kspace * mask, sense, mask, anatomy_labels=labels)
    for label, iters in ((0, 8), (1, 5)):
        alone = CGSENSEReconstructor(iters, 1e-3)(kspace * mask, sense, mask)
        picked = labels == label
        assert torch.allclose(mixed[picked], alone[picked], atol=1e-5)


def test_cg_iters_by_label_names():
    classes = ['brain', 'knee-PD', 'knee-PDFS']
    assert _cg_iters_by_label({'brain': 8, 'knee': 5}, classes) == [8, 5, 5]
    assert _cg_iters_by_label(None, classes) is None


def test_mask_bank_starts_as_bank_mean_and_routes():
    torch.manual_seed(0)
    bank = MaskBank((32, 32), slope=10, calibration_size=8, bank_size=8)
    kspace = torch.randn(3, 32, 32, 2)
    prob = bank(kspace)
    assert prob.shape == (3, 1, 32, 32)
    assert torch.allclose(bank.weights(kspace), torch.full((3, 8), 1 / 8))
    prob.sum().backward()
    assert bank.mask.grad is not None
    bank.gate_head.bias.data[0] = 5.0
    assert (bank.weights(kspace)[:, 0] > 0.9).all()


def test_sampler_accepts_bank_type():
    sampler = Sampler(sampler_type='bank_adapt_2d', shape=(32, 32), calibration_size=8)
    assert isinstance(sampler.mask_net, MaskBank)
