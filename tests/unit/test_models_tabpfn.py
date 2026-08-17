"""Tests for TabPFN v2 foundation model wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# TabPFN is an optional dependency — mock it for unit tests
tabpfn_mock = MagicMock()


class _FakeTabPFN:
    """Minimal mock of tabpfn.TabPFNClassifier for testing."""

    def __init__(self, **kwargs):
        self._fitted = False
        self._X_train = None
        self._y_train = None

    def fit(self, X, y):
        self._fitted = True
        self._X_train = X
        self._y_train = y

    def predict(self, X):
        n = len(X)
        return np.zeros(n, dtype=int)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.7), np.full(n, 0.3)])


@pytest.fixture(autouse=True)
def _mock_tabpfn():
    """Patch tabpfn module so tests don't need the real package."""
    fake_module = MagicMock()
    fake_module.TabPFNClassifier = _FakeTabPFN
    with patch.dict("sys.modules", {"tabpfn": fake_module}):
        yield


def _import_tabpfn_classifier():
    """Import after mocking."""
    from aquacontam.models.tabpfn import TabPFNClassifier

    return TabPFNClassifier


class TestTabPFNClassifier:
    """Tests for TabPFNClassifier."""

    def test_name(self) -> None:
        cls = _import_tabpfn_classifier()
        model = cls()
        assert model.name == "tabpfn_classifier"

    def test_fit_predict(self) -> None:
        cls = _import_tabpfn_classifier()
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(50, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((rng.random(50) > 0.5).astype(int))
        model = cls(config={"random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_predict_proba_shape(self) -> None:
        cls = _import_tabpfn_classifier()
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4)
        y = (rng.random(30) > 0.5).astype(int)
        model = cls(config={"random_state": 42})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (30, 2)

    def test_exceeds_max_train_size(self) -> None:
        cls = _import_tabpfn_classifier()
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = (rng.random(200) > 0.5).astype(int)
        model = cls(config={"max_train_size": 100})
        with pytest.raises(ValueError, match="exceeds max_train_size"):
            model.fit(X, y)

    def test_predict_unfitted_raises(self) -> None:
        cls = _import_tabpfn_classifier()
        model = cls()
        X = np.random.default_rng(42).standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_stores_feature_names(self) -> None:
        cls = _import_tabpfn_classifier()
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(20, 3), columns=["a", "b", "c"])
        y = pd.Series((rng.random(20) > 0.5).astype(int))
        model = cls(config={"random_state": 42})
        model.fit(X, y)
        assert model.feature_names_in_ == ["a", "b", "c"]

    def test_numpy_input(self) -> None:
        cls = _import_tabpfn_classifier()
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (rng.random(40) > 0.5).astype(int)
        model = cls(config={"random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40
