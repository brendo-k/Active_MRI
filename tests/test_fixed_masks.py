"""Sequential_Trainer._load_fixed_masks copies each anatomy's learned mask into its slot and freezes the sampler."""
import logging
from types import SimpleNamespace

import pytest
import torch

from activemri.samplers.loupe import ProbMask
from activemri.training.trainer import Sequential_Trainer, build_optimizer

SHAPE = (8, 8)
CLASSES = ['brain', 'knee']


def save_single_mask(path, slope=10):
    mask = ProbMask(SHAPE, slope=slope).mask.detach()
    torch.save({'options': SimpleNamespace(slope=slope), 'model': {'sampler.mask_net.mask': mask}}, path)
    return mask


def fake_trainer(sampler_type='loupe_2d', anatomy_conditioning=True):
    options = SimpleNamespace(
        sampler_type=sampler_type, anatomy_conditioning=anatomy_conditioning, label_classes=CLASSES, slope=10,
    )
    sampler = SimpleNamespace(mask_net=ProbMask(SHAPE, slope=10, num_conditioning_classes=len(CLASSES)))
    return SimpleNamespace(
        options=options, model=SimpleNamespace(sampler=sampler),
        anatomy_to_idx={name: i for i, name in enumerate(CLASSES)}, logger=logging.getLogger(__name__),
    )


def test_each_anatomy_gets_its_own_mask_and_the_sampler_is_frozen(tmp_path):
    brain, knee = save_single_mask(tmp_path / 'brain.pt'), save_single_mask(tmp_path / 'knee.pt')
    trainer = fake_trainer()
    Sequential_Trainer._load_fixed_masks(trainer, {'brain': str(tmp_path / 'brain.pt'), 'knee': str(tmp_path / 'knee.pt')})
    mask = trainer.model.sampler.mask_net.mask
    assert torch.equal(mask[0], brain[0]) and torch.equal(mask[1], knee[0])
    assert not torch.equal(mask[0], mask[1])
    assert not mask.requires_grad


def test_requires_conditioned_loupe_sampler(tmp_path):
    save_single_mask(tmp_path / 'a.pt')
    with pytest.raises(ValueError, match='anatomy_conditioning'):
        Sequential_Trainer._load_fixed_masks(fake_trainer(anatomy_conditioning=False), {'brain': 'a', 'knee': 'a'})


def test_rejects_mismatched_anatomies_and_slope(tmp_path):
    save_single_mask(tmp_path / 'a.pt'), save_single_mask(tmp_path / 'slope5.pt', slope=5)
    with pytest.raises(ValueError, match='must match anatomies'):
        Sequential_Trainer._load_fixed_masks(fake_trainer(), {'brain': str(tmp_path / 'a.pt')})
    with pytest.raises(ValueError, match='slope'):
        Sequential_Trainer._load_fixed_masks(
            fake_trainer(), {'brain': str(tmp_path / 'a.pt'), 'knee': str(tmp_path / 'slope5.pt')})


def test_optimizer_with_weight_decay_leaves_the_masks_unchanged(tmp_path):
    save_single_mask(tmp_path / 'brain.pt'), save_single_mask(tmp_path / 'knee.pt')
    sampler = torch.nn.Module()
    sampler.mask_net = ProbMask(SHAPE, slope=10, num_conditioning_classes=len(CLASSES))
    model = torch.nn.Module()
    model.sampler, model.reconstructor = sampler, torch.nn.Linear(2, 2)
    trainer = fake_trainer()
    trainer.model = model
    trainer.options.mask_train_mode = 'straight_through'
    Sequential_Trainer._load_fixed_masks(trainer, {'brain': str(tmp_path / 'brain.pt'), 'knee': str(tmp_path / 'knee.pt')})
    before = sampler.mask_net.mask.detach().clone()

    optimizer = build_optimizer(1e-1, 1e-2, Sequential_Trainer._param_groups(trainer))
    for _ in range(3):
        optimizer.zero_grad()
        (model.reconstructor(torch.ones(2)).sum() + sampler.mask_net.mask.sum() * 0).backward()
        optimizer.step()

    assert torch.equal(sampler.mask_net.mask, before)
    assert sampler.mask_net.mask.grad is None
