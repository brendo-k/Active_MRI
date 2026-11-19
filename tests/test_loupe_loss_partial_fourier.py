"""loupe_loss_partial_fourier: the complex-image loss through the model's own detached maps (`loss.loss_type=complex_l1`)
and LOUPE's point-symmetric map (`sampler.loupe_point_symmetric`).

Run with ``.venv/bin/python -m pytest tests/test_loupe_loss_partial_fourier.py`` from the repo root.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc  # noqa: E402
from activemri.samplers.loupe import ProbMask, point_reflected  # noqa: E402
from activemri.samplers.sampler import Sampler  # noqa: E402
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img  # noqa: E402
from activemri.utils.multistep import step_thresholds  # noqa: E402
from tests.test_multistep_sampling import COILS, LABELS, SIZE, SPARSITY, data, model  # noqa: E402

META = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}


def loupe_model(**kwargs):
    return model(sampler_type='loupe_2d', num_step=1, **kwargs)


def consistent_data(seed=0):
    """(image x [2, 1, H, W], unit-norm maps, coil k-space y = F(S x) inside data()'s extent, as a real view)."""
    generator = torch.Generator().manual_seed(seed)
    kspace, maps, _ = data(seed)
    image = torch.randn(2, 1, SIZE, SIZE, dtype=torch.complex64, generator=generator)
    extent = (kspace.abs().sum(dim=(1, 4)) > 0).unsqueeze(1)
    coil_kspace = fft_2d_img(image * maps) * extent
    return image, maps, torch.view_as_real(coil_kspace).contiguous()


def test_the_complex_loss_is_zero_for_a_perfect_reconstruction():
    image, maps, coil_kspace = consistent_data()
    loss = loupe_model().complex_l1(image, maps, coil_kspace)
    assert loss.item() < 1e-6 * image.abs().mean().item()


def test_the_complex_loss_ignores_the_maps_common_phase():
    image, maps, coil_kspace = consistent_data()
    reconstruction = image + 0.1 * torch.randn_like(image)
    phase = torch.exp(1j * 3 * torch.rand(2, 1, SIZE, SIZE))
    sampling_model = loupe_model()
    before = sampling_model.complex_l1(reconstruction, maps, coil_kspace)
    after = sampling_model.complex_l1(reconstruction * phase.conj(), maps * phase, coil_kspace)
    assert before.item() > 0 and after.item() == pytest.approx(before.item(), rel=1e-5)


def test_the_complex_loss_sees_a_phase_error_that_the_rss_output_cannot():
    image, maps, coil_kspace = consistent_data()
    sampling_model = loupe_model()
    extent = (coil_kspace.abs().sum(dim=(1, 4)) > 0).unsqueeze(1)
    shifted = image * torch.exp(1j * torch.linspace(0, 3, SIZE)).view(1, 1, 1, SIZE)
    rss = sampling_model._coil_rss(image, maps, extent.float())
    rss_shifted = sampling_model._coil_rss(shifted, maps, extent.float())
    assert sampling_model.complex_l1(shifted, maps, coil_kspace).item() > 10 * (rss - rss_shifted).abs().mean().item()


def test_the_learned_maps_are_rss_normalised():
    torch.manual_seed(0)
    kspace = torch.randn(2, COILS, SIZE, SIZE, dtype=torch.complex64)
    with torch.no_grad():
        maps, _ = SensetivityModel_mc(2, 2, 8)(kspace)
    power = maps.abs().pow(2).sum(dim=1)
    assert torch.allclose(power[power > 0.5], torch.ones_like(power[power > 0.5]), atol=1e-4)
    assert power.gt(0.5).float().mean() > 0.9


def test_the_complex_loss_trains_the_mask_through_the_model():
    sampling_model = loupe_model()
    sampling_model.train()
    kspace, maps, target = data()
    pred = sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps)
    loss, _ = sampling_model.loss(pred, {'target': target, 'kspace': kspace}, META, 'complex_l1')
    loss.backward()
    assert torch.isfinite(loss) and sampling_model.sampler.mask_net.mask.grad.abs().sum() > 0


def test_the_point_symmetric_map_gives_k_and_minus_k_one_probability():
    torch.manual_seed(0)
    free = ProbMask((SIZE, SIZE))
    torch.manual_seed(0)
    tied = ProbMask((SIZE, SIZE), point_symmetric=True)
    probe = torch.zeros(1, SIZE, SIZE, 1)
    with torch.no_grad():
        p_free, p_tied = free(probe), tied(probe)
    assert not torch.equal(p_free, point_reflected(p_free))
    assert torch.equal(p_tied, point_reflected(p_tied))
    assert torch.equal(tied.raw_logits(), point_reflected(tied.raw_logits(), dims=(1, 2)))
    centre = SIZE // 2
    assert torch.equal(point_reflected(torch.arange(SIZE).view(1, SIZE, 1)).flatten()[[0, centre]], torch.tensor([0, centre]))


def test_the_point_symmetric_map_keeps_the_budget():
    sampling_model = loupe_model(loupe_point_symmetric=True)
    sampling_model.eval()
    kspace, maps, target = data()
    thresholds = step_thresholds(11, (0, 1), 1, (SIZE, SIZE))
    with torch.no_grad():
        mask = sampling_model(target, kspace, anatomy_labels=LABELS, sense_maps=maps, thresholds=thresholds)['mask']
        valid = sampling_model._valid_kspace_mask(kspace)
    realised = (mask * valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))
    assert torch.allclose(realised, torch.full_like(realised, SPARSITY), atol=0.02)


def test_the_point_symmetric_map_belongs_to_loupe_2d_only():
    with pytest.raises(ValueError, match='loupe_point_symmetric'):
        Sampler(sampler_type='seqmri_2d', shape=(SIZE, SIZE), loupe_point_symmetric=True)


def test_ifft_of_fft_is_the_identity_used_by_the_loss():
    image = torch.randn(1, 1, SIZE, SIZE, dtype=torch.complex64)
    assert torch.allclose(ifft_2d_img(fft_2d_img(image)), image, atol=1e-5)
