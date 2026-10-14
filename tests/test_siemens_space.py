"""Self-supervised Siemens SPACE pipeline: the hold-out split, the SSDU loss, and the k-space support."""
import h5py
import numpy as np
import pytest
import torch

from activemri.datasets.siemens_space_data import (
    BLOCK_GUARD, SiemensSpaceDataset, draw_holdout, split_slices,
)
from activemri.models.sequential_sampling import SequentialSampling
from activemri.reconstructors.VarNet import VarNetBlock, NormUnet
from activemri.utils.image_processing import fft_2d_img

COILS, HEIGHT, WIDTH, READOUT = 4, 32, 16, 40


@pytest.fixture
def volume(tmp_path):
    rng = np.random.default_rng(0)
    omega = np.zeros((HEIGHT, WIDTH), np.uint8)
    omega[4::2, ::2] = 1
    kspace = (rng.standard_normal((READOUT, COILS, HEIGHT, WIDTH)) + 1j * rng.standard_normal((READOUT, COILS, HEIGHT, WIDTH))).astype(np.complex64)
    kspace *= omega
    path = tmp_path / 'volume.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('kspace', data=kspace)
        f.create_dataset('sense', data=np.ones((READOUT, COILS, HEIGHT, WIDTH), np.complex64) / 2)
        f.create_dataset('reference', data=np.ones((READOUT, HEIGHT, WIDTH), np.float32))
        f.create_dataset('omega', data=omega)
        f.create_dataset('signal_slices', data=np.arange(READOUT))
        f.attrs['grid'] = (HEIGHT, WIDTH)
    return path


def test_val_positions_never_train_and_keep_a_guard_band():
    signal = np.arange(100)

    train, val = split_slices(signal, 'train'), split_slices(signal, 'val')

    assert not set(train) & set(val)
    assert min(abs(t - v) for t in train for v in val) > BLOCK_GUARD


def test_holdout_is_a_fixed_size_subset_of_the_acquired_points():
    omega = torch.zeros(HEIGHT, WIDTH)
    omega[::2, ::2] = 1

    holdout = draw_holdout(omega, 0.25, torch.Generator().manual_seed(0))

    assert holdout.sum().item() == round(0.25 * omega.sum().item())
    assert ((holdout > 0) <= (omega > 0)).all()


def test_pool_and_holdout_partition_the_acquired_points(volume):
    dataset = SiemensSpaceDataset([volume], 'val', holdout_fraction=0.25, image_shape=(HEIGHT, WIDTH))

    sample = dataset[0]

    pool = torch.view_as_complex(sample['kspace']).abs().sum(0) > 0
    held = torch.view_as_complex(sample['holdout_kspace']).abs().sum(0) > 0
    assert not (pool & held).any()
    assert ((pool | held) == (dataset.omegas[volume] > 0)).all()
    assert (held == (sample['holdout_mask'][0] > 0)).all()
    assert (sample['kspace_support'][0][dataset.omegas[volume] > 0] == 1).all()


def test_val_holdout_repeats_and_train_holdout_does_not(volume):
    val = SiemensSpaceDataset([volume], 'val', 0.25, (HEIGHT, WIDTH))
    train = SiemensSpaceDataset([volume], 'train', 0.25, (HEIGHT, WIDTH))

    assert torch.equal(val[0]['holdout_mask'], val[0]['holdout_mask'])
    assert not torch.equal(train[0]['holdout_mask'], train[0]['holdout_mask'])


def _ssdu(recon, sense, measured, holdout_mask):
    pred = {'recon_complex': recon, 'sense_maps': sense}
    target = {'holdout_kspace': torch.view_as_real(measured), 'holdout_mask': holdout_mask}
    return SequentialSampling.ssdu_loss(pred, target)


def test_ssdu_loss_is_zero_for_a_prediction_that_reproduces_the_holdout():
    torch.manual_seed(0)
    sense = torch.randn(1, COILS, HEIGHT, WIDTH, dtype=torch.complex64)
    image = torch.randn(1, 1, HEIGHT, WIDTH, dtype=torch.complex64)
    holdout_mask = (torch.rand(1, 1, HEIGHT, WIDTH) < 0.3).float()
    measured = fft_2d_img(image * sense) * holdout_mask

    assert _ssdu(image, sense, measured, holdout_mask).item() == pytest.approx(0, abs=1e-5)


def test_ssdu_loss_of_an_empty_prediction_is_the_trivial_two():
    torch.manual_seed(0)
    sense = torch.randn(1, COILS, HEIGHT, WIDTH, dtype=torch.complex64)
    holdout_mask = (torch.rand(1, 1, HEIGHT, WIDTH) < 0.3).float()
    measured = torch.randn(1, COILS, HEIGHT, WIDTH, dtype=torch.complex64) * holdout_mask

    loss = _ssdu(torch.zeros(1, 1, HEIGHT, WIDTH, dtype=torch.complex64), sense, measured, holdout_mask)

    assert loss.item() == pytest.approx(2.0, abs=1e-5)


def test_ssdu_loss_ignores_k_space_outside_the_holdout():
    torch.manual_seed(0)
    sense = torch.randn(1, COILS, HEIGHT, WIDTH, dtype=torch.complex64)
    image = torch.randn(1, 1, HEIGHT, WIDTH, dtype=torch.complex64)
    holdout_mask = torch.zeros(1, 1, HEIGHT, WIDTH)
    holdout_mask[..., ::4, ::4] = 1
    measured = fft_2d_img(image * sense) * holdout_mask
    other = torch.randn_like(image)
    # an image that only differs where Lambda never looks cannot be told apart
    delta = fft_2d_img(other) * (1 - holdout_mask)
    perturbed = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(delta, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
    unit_sense = torch.ones_like(sense)

    exact = _ssdu(image, unit_sense, fft_2d_img(image * unit_sense) * holdout_mask, holdout_mask)
    shifted = _ssdu(image + perturbed, unit_sense, fft_2d_img(image * unit_sense) * holdout_mask, holdout_mask)

    assert shifted.item() == pytest.approx(exact.item(), abs=1e-4)


def test_varnet_regularizer_cannot_write_outside_the_k_space_support():
    torch.manual_seed(0)
    block = VarNetBlock(NormUnet(4, 2))
    support = torch.zeros(1, 1, HEIGHT, WIDTH)
    support[..., 6:26, :] = 1
    kspace = torch.randn(1, COILS, HEIGHT, WIDTH, dtype=torch.complex64) * support
    sense = torch.ones(1, COILS, HEIGHT, WIDTH, dtype=torch.complex64) / 2

    out = block(kspace, kspace, (torch.rand(1, 1, HEIGHT, WIDTH) < 0.3).float() * support, sense, None, support)
    unrestricted = block(kspace, kspace, (torch.rand(1, 1, HEIGHT, WIDTH) < 0.3).float() * support, sense, None, None)

    assert (out * (1 - support)).abs().max().item() == 0
    assert (unrestricted * (1 - support)).abs().max().item() > 0
