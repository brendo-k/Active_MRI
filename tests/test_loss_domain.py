"""The loss-domain L1 losses (complex image, coil k-space, RSS k-space) of SequentialSampling.loss_domain_l1."""
import pathlib

import pytest
import torch

from activemri.models.sequential_sampling import SequentialSampling
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img

CACHE = pathlib.Path('/export/scratch/bkadota/fastmri_complex_espirit_v2')
VAL_BRAIN = pathlib.Path('/export/scratch/bkadota/fastmri_raw/brain/multicoil_val')


def _maps(batch, coils, size, generator):
    raw = torch.randn(batch, coils, size, size, dtype=torch.complex64, generator=generator)
    smooth = ifft_2d_img(fft_2d_img(raw) * _centre(size, 6))
    return smooth / smooth.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def _centre(size, half):
    mask = torch.zeros(size, size)
    mask[size // 2 - half:size // 2 + half, size // 2 - half:size // 2 + half] = 1
    return mask


def _example(seed=0, batch=2, coils=4, size=32):
    generator = torch.Generator().manual_seed(seed)
    maps = _maps(batch, coils, size, generator)
    image = torch.randn(batch, 1, size, size, dtype=torch.complex64, generator=generator)
    support = torch.zeros(batch, 1, size, size)
    support[..., 4:-4, 4:-4] = 1
    espirit = image * support
    extent = torch.ones(batch, 1, size, size)
    return maps, espirit, support, extent


def _pred(recon, maps, extent):
    return {'recon_complex': recon, 'sense_maps': maps, 'kspace_extent': extent,
            'output': SequentialSampling._coil_rss(recon, maps, extent)}


@pytest.mark.parametrize('loss_type', ['complex_l1', 'kspace_sense_l1'])
def test_zero_when_the_reconstruction_is_the_espirit_image(loss_type):
    maps, espirit, _, extent = _example()
    pred = _pred(espirit, maps, extent)
    loss = SequentialSampling.loss_domain_l1(pred, pred['output'], espirit, loss_type)
    assert loss.item() < 1e-6


def test_rss_kspace_loss_is_zero_on_its_own_target_and_needs_no_espirit_image():
    maps, espirit, _, extent = _example()
    pred = _pred(espirit, maps, extent)
    loss = SequentialSampling.loss_domain_l1(pred, pred['output'].detach(), None, 'kspace_rss_l1')
    assert loss.item() < 1e-6


def test_complex_loss_reads_the_extent_projected_image():
    maps, espirit, _, extent = _example()
    extent[..., :, :6] = 0
    recon = espirit + 0.1 * torch.randn_like(espirit)
    pred = _pred(recon, maps, extent)
    loss = SequentialSampling.loss_domain_l1(pred, pred['output'], espirit, 'complex_l1')
    projected = (ifft_2d_img(fft_2d_img(maps * recon) * extent) * maps.conj()).sum(dim=1, keepdim=True)
    assert torch.allclose(loss, (projected - espirit).abs().mean())


def test_support_mask_drops_background_of_the_prediction_only():
    maps, espirit, support, extent = _example()
    recon = espirit + 0.5 * (1 - support)
    pred = _pred(recon, maps, extent)
    unmasked = SequentialSampling.loss_domain_l1(pred, pred['output'], espirit, 'complex_l1')
    masked = SequentialSampling.loss_domain_l1(pred, pred['output'], espirit, 'complex_l1', support_mask=True)
    assert unmasked.item() > 1e-3
    assert masked.item() < 1e-3


def test_rss_kspace_support_mask_changes_the_target_background():
    maps, espirit, support, extent = _example()
    pred = _pred(espirit, maps, extent)
    target = pred['output'].detach() + 0.2 * (1 - support)
    unmasked = SequentialSampling.loss_domain_l1(pred, target, espirit, 'kspace_rss_l1')
    masked = SequentialSampling.loss_domain_l1(pred, target, espirit, 'kspace_rss_l1', support_mask=True)
    assert unmasked.item() > 1e-3
    assert masked.item() < 1e-5


def test_sense_kspace_target_does_not_train_the_maps():
    maps, espirit, _, extent = _example()
    maps.requires_grad_(True)
    pred = _pred(torch.zeros_like(espirit), maps, extent)
    SequentialSampling.loss_domain_l1(pred, pred['output'], espirit, 'kspace_sense_l1').backward()
    assert maps.grad is None or maps.grad.abs().max().item() == 0


@pytest.mark.parametrize('loss_type', ['complex_l1', 'kspace_sense_l1', 'kspace_rss_l1'])
def test_gradients_are_finite_where_the_error_is_exactly_zero(loss_type):
    maps, espirit, support, extent = _example()
    recon = (espirit + 0.1 * support).requires_grad_(True)
    maps.requires_grad_(True)
    pred = _pred(recon, maps, extent)
    target = SequentialSampling._coil_rss(espirit, maps.detach(), extent)
    SequentialSampling.loss_domain_l1(pred, target, espirit, loss_type).backward()
    assert torch.isfinite(torch.view_as_real(recon.grad)).all()
    assert torch.isfinite(torch.view_as_real(maps.grad)).all()


def test_ground_truth_kspace_loss_is_zero_on_the_measured_data_and_needs_no_espirit_image():
    maps, espirit, _, extent = _example()
    extent[..., :, :6] = 0
    pred = _pred(espirit, maps, extent)
    measured = fft_2d_img(maps * espirit) + 5.0 * (1 - extent)
    loss = SequentialSampling.loss_domain_l1(pred, pred['output'], None, 'kspace_gt_l1', kspace=measured)
    assert loss.item() < 1e-6


def test_ground_truth_kspace_loss_trains_the_maps():
    maps, espirit, _, extent = _example()
    maps.requires_grad_(True)
    pred = _pred(espirit, maps, extent)
    measured = fft_2d_img(maps.detach() * espirit) + 0.1
    SequentialSampling.loss_domain_l1(pred, pred['output'], None, 'kspace_gt_l1', kspace=measured).backward()
    assert maps.grad.abs().max().item() > 0


def test_ground_truth_kspace_loss_refuses_the_support_mask():
    maps, espirit, _, extent = _example()
    pred = _pred(espirit, maps, extent)
    with pytest.raises(ValueError, match='kspace_gt_l1'):
        SequentialSampling.loss_domain_l1(pred, pred['output'], espirit, 'kspace_gt_l1', support_mask=True,
                                          kspace=fft_2d_img(maps * espirit))


def test_learned_maps_complex_loss_is_zero_on_the_measured_data_combined_through_the_maps():
    maps, espirit, _, extent = _example()
    extent[..., :, :6] = 0
    pred = _pred(espirit, maps, extent)
    measured = fft_2d_img(maps * espirit) + 5.0 * (1 - extent)
    loss = SequentialSampling.loss_domain_l1(pred, pred['output'], None, 'complex_maps_l1', kspace=measured)
    assert loss.item() < 1e-6


def test_learned_maps_complex_loss_ignores_a_common_phase_moved_between_maps_and_image():
    maps, espirit, _, extent = _example()
    measured = fft_2d_img(maps * espirit)
    recon = espirit + 0.1 * torch.randn_like(espirit)
    rows = torch.arange(espirit.shape[-2], dtype=torch.float32)[:, None]
    phase = torch.exp(1j * (1.7 + 0.2 * rows + 0.05 * rows.T))
    losses = [SequentialSampling.loss_domain_l1(_pred(r, m, extent), None, None, 'complex_maps_l1', kspace=measured)
              for r, m in ((recon, maps), (recon * phase.conj(), maps * phase))]
    assert torch.allclose(losses[0], losses[1], atol=1e-6)


def test_learned_maps_complex_target_does_not_train_the_maps():
    maps, espirit, _, extent = _example()
    measured = fft_2d_img(maps * espirit)
    maps.requires_grad_(True)
    pred = _pred(torch.zeros_like(espirit), maps, extent)
    SequentialSampling.loss_domain_l1(pred, pred['output'], None, 'complex_maps_l1', kspace=measured).backward()
    assert maps.grad is None or maps.grad.abs().max().item() == 0


def test_learned_maps_complex_loss_has_finite_gradients_and_needs_the_measured_kspace():
    maps, espirit, support, extent = _example()
    measured = fft_2d_img(maps * espirit)
    recon = (espirit + 0.1 * support).requires_grad_(True)
    maps.requires_grad_(True)
    pred = _pred(recon, maps, extent)
    SequentialSampling.loss_domain_l1(pred, None, None, 'complex_maps_l1', kspace=measured).backward()
    assert torch.isfinite(torch.view_as_real(recon.grad)).all()
    assert torch.isfinite(torch.view_as_real(maps.grad)).all()
    with pytest.raises(ValueError, match='complex_maps_l1'):
        SequentialSampling.loss_domain_l1(pred, None, None, 'complex_maps_l1')


def test_missing_espirit_image_raises():
    maps, espirit, _, extent = _example()
    pred = _pred(espirit, maps, extent)
    with pytest.raises(ValueError, match='complex_cache_dir'):
        SequentialSampling.loss_domain_l1(pred, pred['output'], None, 'complex_l1')


@pytest.mark.skipif(not (CACHE.exists() and VAL_BRAIN.exists()), reason='needs the local cache and raw data')
def test_dataset_serves_the_cached_image_for_its_own_slice():
    from activemri.datasets.complex_image_cache import ComplexImageCache, cache_signature
    from activemri.datasets.fastmri_data import FastMRIDataset

    name = sorted(path.name for path in VAL_BRAIN.glob('file_brain_AXFLAIR_*.h5'))[0]
    cache = ComplexImageCache(CACHE)
    dataset = FastMRIDataset(VAL_BRAIN, 'brain', [320, 320], num_coils=20, file_names=[name], readout_crop=True,
                             complex_cache=cache)
    sample = dataset[len(dataset) // 2]
    expected = cache.load('brain', name, sample['slice_id'], cache_signature('brain', [320, 320], 20, 0.2))
    assert torch.equal(torch.view_as_complex(sample['complex_target']), expected)
    inside = expected.abs() > 0
    ratio = (expected.abs()[inside] / sample['target'][inside]).median()
    assert 0.8 < ratio.item() < 1.05
