import gc
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from activemri.training import trainer as trainer_module
from activemri.training.trainer import Sequential_Trainer


def stub_trainer(val_losses, num_epochs, val_every=1, patience=0):
    """A Sequential_Trainer whose epochs only return the given val losses, one per validated epoch."""
    t = Sequential_Trainer.__new__(Sequential_Trainer)
    t.options = SimpleNamespace(num_epochs=num_epochs, val_every=val_every, early_stop_patience=patience)
    t.start_epoch, t.end_epoch, t.epoch = 0, num_epochs, 0
    t.best_val_loss, t.validations_since_best = np.inf, 0
    t.logger = logging.getLogger('test_early_stop')
    t.scheduler = SimpleNamespace(step=lambda epoch: None)
    losses = iter(val_losses)
    t.train_epoch = lambda: (1.0, 0.0, {})
    t.val_epoch = lambda: (next(losses), 0.0, 0.0, {})
    t._plot_metrics = lambda history: None
    t.saved = []
    t.save_model = lambda is_new_best: t.saved.append((t.epoch + 1, bool(is_new_best)))
    t.save_ema_model = lambda: None
    return t


@pytest.fixture(autouse=True)
def no_wandb_and_unfreeze(monkeypatch):
    monkeypatch.setattr(trainer_module.wandb, 'finish', lambda: None)
    yield
    gc.unfreeze()


def test_patience_zero_runs_every_epoch():
    t = stub_trainer([1, 2, 3, 4, 5, 6], num_epochs=6)
    t._train_loop()
    assert [epoch for epoch, _ in t.saved] == [1, 2, 3, 4, 5, 6]


def test_stops_after_patience_validations_without_a_new_best():
    t = stub_trainer([3, 2, 2.5, 2.4, 2.6, 1, 1], num_epochs=7, patience=3)
    t._train_loop()
    assert t.saved == [(1, True), (2, True), (3, False), (4, False), (5, False)]
    assert t.best_val_loss == 2


def test_unvalidated_epochs_do_not_count():
    t = stub_trainer([3, 2, 2.5, 2.4, 1], num_epochs=10, val_every=2, patience=2)
    t._train_loop()
    assert [epoch for epoch, _ in t.saved] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_early_stop_line_matches_the_worker_done_check(caplog):
    t = stub_trainer([1, 2], num_epochs=5, patience=1)
    with caplog.at_level(logging.INFO, logger='test_early_stop'):
        t._train_loop()
    assert any(r.getMessage().startswith('Early stop after epoch 2') for r in caplog.records)
