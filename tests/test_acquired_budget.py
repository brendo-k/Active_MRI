"""R counts each slice's acquired k-space (sampler.budget_over='acquired'), and the sampler's input is SENSE-combined.

Run with ``.venv/bin/python -m pytest tests/test_acquired_budget.py`` from the repo root.

FLAIR acquires ~64% of the 320 grid. With budget_over='grid' a mask still held 1/R of the grid, all of it inside the
acquired region, so FLAIR ran at R~5 of what it acquires. With 'acquired' every sampler holds 1/R of the slice's own
acquired k-space: rescaled masks through a budget per slice, the CAIPI lattice cut to the acquired region, and the
Poisson-disc mask drawn again over it.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.config import TrainConfig  # noqa: E402
from activemri.models.sequential_sampling import SequentialSampling  # noqa: E402
from activemri.utils.image_processing import fft_2d_img  # noqa: E402

SIZE = 64
SPARSITY = 0.25
PRESELECT = 4


def model(sampler_type='loupe_2d', budget_over='acquired', size=SIZE, sparsity=SPARSITY, **kwargs):
    return SequentialSampling(nn.Identity(), shape=(size, size), sparsity=sparsity, sampler_type=sampler_type,
                              num_step=1, preselect_num=PRESELECT, recon_type='varnet', budget_over=budget_over, **kwargs)


def extents(size=SIZE):
    """[full grid, FLAIR-like: 60% of the columns and 80% of the rows, centred], as [2, 1, H, W]."""
    valid = torch.ones(2, 1, size, size)
    columns, rows = round(0.6 * size), round(0.8 * size)
    flair = torch.zeros(size, size)
    flair[(size - rows) // 2:(size - rows) // 2 + rows, (size - columns) // 2:(size - columns) // 2 + columns] = 1
    valid[1, 0] = flair
    return valid


def test_full_grid_budget_matches_the_grid_convention():
    acquired, grid = model(), model(budget_over='grid')
    old = acquired._init_mask(torch.zeros(2))
    budget = acquired.mask_budget(old, torch.ones(2, 1, SIZE, SIZE))
    assert budget.shape == (2,)
    torch.testing.assert_close(budget, torch.full((2,), grid.budget, dtype=torch.float64))
    assert grid.mask_budget(old, extents()) == grid.budget


def test_rescaled_mask_holds_one_over_r_of_each_slices_acquired_kspace():
    sampler_model = model()
    valid = extents()
    old = sampler_model._init_mask(torch.zeros(2))
    unscaled = torch.rand(2, 1, SIZE, SIZE, generator=torch.Generator().manual_seed(0)) * (1 - old)
    rescaled, _ = sampler_model.sampler.select_locations(unscaled, old, sampler_model.mask_budget(old, valid), valid)
    expected = rescaled.flatten(1).sum(1) + (old * valid).flatten(1).sum(1)
    torch.testing.assert_close(expected.double(), SPARSITY * valid.flatten(1).sum(1).double(), rtol=1e-5, atol=1e-3)
    assert (rescaled * (1 - valid)).abs().max() == 0


needs_sigpy = pytest.mark.skipif(not importlib.util.find_spec('sigpy'), reason='the Poisson-disc mask comes from SigPy')


@pytest.mark.parametrize('sampler_type, tolerance', [('caipi_2d', 0.03), pytest.param('poisson_2d', 0.12, marks=needs_sigpy)])
def test_fixed_patterns_keep_their_acceleration_over_acquired_kspace(sampler_type, tolerance):
    size = 160
    sampler_model = model(sampler_type, size=size, sparsity=0.25, caipi_accel_y=2, caipi_accel_z=2)
    valid = extents(size)
    pattern = sampler_model.sampler.fixed_pattern(valid)
    assert (pattern * (1 - valid)).sum() == 0
    acceleration = valid.flatten(1).sum(1) / pattern.flatten(1).sum(1)
    torch.testing.assert_close(acceleration, torch.full((2,), 4.0), rtol=tolerance, atol=0)


@needs_sigpy
def test_full_grid_poisson_is_the_original_pattern():
    sampler_model = model('poisson_2d', size=96)
    torch.testing.assert_close(sampler_model.sampler.fixed_pattern(torch.ones(1, 1, 96, 96)),
                               sampler_model.sampler.poisson_pattern)


def test_one_budget_per_batch_samplers_refuse_mixed_extents():
    sampler_model = model(mask_train_mode='reinforce')
    old = sampler_model._init_mask(torch.zeros(2))
    valid = extents()
    with pytest.raises(ValueError, match='budget_over=grid'):
        sampler_model.sampler(None, old, sampler_model.mask_budget(old, valid), valid_mask=valid)


def test_sampler_input_is_the_sense_combination():
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(2, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    maps = torch.randn(2, 6, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    maps = maps / maps.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    combined = SequentialSampling._coil_combine(fft_2d_img(image.unsqueeze(1) * maps), maps)
    torch.testing.assert_close(combined, image, rtol=1e-4, atol=1e-4)


def test_rss_sampler_input_ignores_the_maps_common_phase():
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(2, 1, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    maps = torch.randn(2, 6, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    coil_kspace = fft_2d_img(image * maps)
    common_phase = torch.exp(1j * 2 * torch.pi * torch.rand(2, 1, SIZE, SIZE, generator=generator))
    rss_model, sense_model = model(sampler_coil_combine='rss'), model()
    rss_input = rss_model._sampler_image(coil_kspace, maps)
    torch.testing.assert_close(rss_input, rss_model._sampler_image(coil_kspace, maps * common_phase))
    torch.testing.assert_close(rss_input.real, (image * maps).abs().pow(2).sum(dim=1).sqrt(), rtol=1e-4, atol=1e-4)
    assert rss_input.imag.abs().max() == 0
    sense_input = sense_model._sampler_image(coil_kspace, maps)
    assert (sense_input - sense_model._sampler_image(coil_kspace, maps * common_phase)).abs().max() > 0.1


def test_older_runs_keep_the_grid_budget_and_new_runs_count_acquired():
    assert TrainConfig().budget_over == 'grid'
    config = OmegaConf.load(Path(__file__).resolve().parent.parent / 'configs' / 'config.yaml')
    assert config.sampler.budget_over == 'acquired'
