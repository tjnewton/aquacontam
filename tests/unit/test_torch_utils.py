"""Tests for PyTorch utility classes."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aquacontam.models._torch_utils import EarlyStopping, TabularDataset, get_device  # noqa: E402


class TestGetDevice:
    """Tests for device detection."""

    def test_returns_torch_device(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_is_known(self) -> None:
        device = get_device()
        assert device.type in {"cpu", "cuda", "mps"}


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_no_stop_while_improving(self) -> None:
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            assert not es.step(loss)

    def test_stops_after_patience(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)  # best = 1.0
        es.step(1.1)  # counter = 1
        es.step(1.2)  # counter = 2
        assert es.step(1.3)  # counter = 3 => stop

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=2)
        es.step(1.0)
        es.step(1.1)  # counter = 1
        es.step(0.5)  # improvement, counter resets
        assert not es.step(0.6)  # counter = 1

    def test_min_delta(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)
        # 0.95 is not 0.1 better than 1.0 → not an improvement
        es.step(0.95)
        assert es.step(0.96)  # counter = 2 => stop

    def test_initial_state(self) -> None:
        es = EarlyStopping(patience=5)
        assert es.best_loss is None
        assert es.counter == 0
        assert not es.should_stop

    def test_saves_best_state_dict(self) -> None:
        """EarlyStopping saves model state when loss improves."""
        model = torch.nn.Linear(4, 1)
        es = EarlyStopping(patience=3)

        # First step — should save state
        es.step(1.0, model=model)
        assert es._best_state_dict is not None

        # Modify model weights
        with torch.no_grad():
            model.weight.fill_(999.0)

        # Improvement — should save new state
        es.step(0.5, model=model)
        # Verify the saved state has the 999.0 weights
        assert es._best_state_dict["weight"].mean().item() == pytest.approx(999.0)

    def test_restore_loads_best_weights(self) -> None:
        """EarlyStopping.restore() loads best weights into model."""
        model = torch.nn.Linear(4, 1)
        es = EarlyStopping(patience=2)

        # Save best state at loss=1.0
        with torch.no_grad():
            model.weight.fill_(1.0)
        es.step(1.0, model=model)

        # Overfit: loss degrades, model weights change
        with torch.no_grad():
            model.weight.fill_(42.0)
        es.step(2.0, model=model)
        es.step(3.0, model=model)  # should_stop = True

        # Restore best weights
        assert es.restore(model)
        assert model.weight.mean().item() == pytest.approx(1.0)

    def test_restore_returns_false_when_no_state(self) -> None:
        """restore() returns False when step() was never called with a model."""
        es = EarlyStopping(patience=3)
        model = torch.nn.Linear(4, 1)
        es.step(1.0)  # no model passed
        assert not es.restore(model)

    def test_backward_compatible_step_without_model(self) -> None:
        """step() still works when called without model argument."""
        es = EarlyStopping(patience=2)
        assert not es.step(1.0)
        assert not es.step(1.1)
        assert es.step(1.2)


class TestTabularDataset:
    """Tests for TabularDataset."""

    def test_length(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = rng.randn(50)
        ds = TabularDataset(X, y)
        assert len(ds) == 50

    def test_getitem_returns_tensors(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(10, 4)
        y = rng.randn(10)
        ds = TabularDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert x_item.shape == (4,)
        assert y_item.shape == ()

    def test_dtype_is_float32(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(5, 2).astype(np.float64)
        y = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        ds = TabularDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.dtype == torch.float32
        assert y_item.dtype == torch.float32
