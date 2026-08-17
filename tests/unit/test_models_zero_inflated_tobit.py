"""Tests for Zero-Inflated Deep Tobit (ZIDT) model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from aquacontam.models.zero_inflated_tobit import (
    ZeroInflatedTobitClassifier,
    ZeroInflatedTobitRegressor,
    _ZeroInflatedTobitNetwork,
    zero_inflated_tobit_nll,
)

FAST_CFG: dict = {
    "hidden_sizes": [4, 2],
    "epochs": 5,
    "batch_size": 32,
    "random_state": 42,
    "default_detection_limit": 0.004,
}


def _make_censored_data(
    n: int = 100, n_features: int = 5, censor_rate: float = 0.7, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic censored data for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    detection_limits = rng.uniform(1.0, 5.0, n)
    true_values = X[:, 0] * 2 + 3 + rng.randn(n) * 0.5
    censored = (rng.random(n) < censor_rate).astype(float)
    y = np.where(censored > 0.5, 0.0, true_values)
    return X, y, censored, detection_limits


class TestZeroInflatedTobitNetwork:
    """Tests for the network architecture."""

    def test_output_shapes(self) -> None:
        import torch

        net = _ZeroInflatedTobitNetwork(n_features=5, hidden_sizes=[4, 2])
        x = torch.randn(10, 5)
        pi, mu, log_sigma = net(x)
        assert pi.shape == (10,)
        assert mu.shape == (10,)
        assert log_sigma.shape == (10,)

    def test_pi_in_01(self) -> None:
        import torch

        net = _ZeroInflatedTobitNetwork(n_features=3, hidden_sizes=[4])
        x = torch.randn(20, 3) * 10  # large inputs
        pi, _, _ = net(x)
        assert torch.all(pi >= 0.0)
        assert torch.all(pi <= 1.0)

    def test_log_sigma_clamped(self) -> None:
        import torch

        net = _ZeroInflatedTobitNetwork(n_features=3, hidden_sizes=[4, 2])
        x = torch.randn(10, 3)
        _, _, log_sigma = net(x)
        assert log_sigma.min().item() >= -5.0
        assert log_sigma.max().item() <= 5.0


class TestZeroInflatedTobitNLL:
    """Tests for the loss function."""

    def test_loss_finite(self) -> None:
        import torch

        pi = torch.tensor([0.3, 0.5, 0.2])
        mu = torch.tensor([1.0, 2.0, 3.0])
        log_sigma = torch.tensor([0.0, 0.1, -0.1])
        y = torch.tensor([2.0, 0.0, 0.0])
        censored = torch.tensor([0.0, 1.0, 1.0])
        dl = torch.tensor([1.0, 1.0, 1.0])
        loss = zero_inflated_tobit_nll(pi, mu, log_sigma, y, censored, dl)
        assert torch.isfinite(loss)

    def test_all_censored(self) -> None:
        import torch

        n = 20
        pi = torch.full((n,), 0.5)
        mu = torch.zeros(n)
        log_sigma = torch.zeros(n)
        y = torch.zeros(n)
        censored = torch.ones(n)
        dl = torch.ones(n)
        loss = zero_inflated_tobit_nll(pi, mu, log_sigma, y, censored, dl)
        assert torch.isfinite(loss)

    def test_all_detected(self) -> None:
        import torch

        n = 20
        pi = torch.full((n,), 0.1)
        mu = torch.ones(n) * 5.0
        log_sigma = torch.zeros(n)
        y = torch.ones(n) * 5.0
        censored = torch.zeros(n)
        dl = torch.ones(n)
        loss = zero_inflated_tobit_nll(pi, mu, log_sigma, y, censored, dl)
        assert torch.isfinite(loss)


class TestZeroInflatedTobitClassifier:
    """Tests for ZeroInflatedTobitClassifier."""

    def test_name(self) -> None:
        model = ZeroInflatedTobitClassifier()
        assert model.name == "zi_tobit_classifier"

    def test_requires_censoring_metadata(self) -> None:
        assert ZeroInflatedTobitClassifier.requires_censoring_metadata is True

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = ZeroInflatedTobitClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, classification_data) -> None:
        X, y = classification_data
        model = ZeroInflatedTobitClassifier(config=FAST_CFG)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_with_censoring(self) -> None:
        X, y, censored, dl = _make_censored_data(n=80, censor_rate=0.7)
        model = ZeroInflatedTobitClassifier(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        proba = model.predict_proba(X, detection_limits=dl)
        assert proba.shape == (80, 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_unfitted_raises(self) -> None:
        model = ZeroInflatedTobitClassifier(config=FAST_CFG)
        X = np.random.default_rng(42).standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)


class TestZeroInflatedTobitRegressor:
    """Tests for ZeroInflatedTobitRegressor."""

    def test_name(self) -> None:
        model = ZeroInflatedTobitRegressor()
        assert model.name == "zi_tobit_regressor"

    def test_requires_censoring_metadata(self) -> None:
        assert ZeroInflatedTobitRegressor.requires_censoring_metadata is True

    def test_regressor_predict(self) -> None:
        X, y, censored, dl = _make_censored_data(n=80, censor_rate=0.5)
        model = ZeroInflatedTobitRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        preds = model.predict(X)
        assert len(preds) == 80
        assert preds.dtype in (np.float32, np.float64)
        assert np.all(np.isfinite(preds))

    def test_predict_proba_raises(self) -> None:
        X, y, censored, dl = _make_censored_data(n=40)
        model = ZeroInflatedTobitRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_log_transform(self) -> None:
        X, y, censored, dl = _make_censored_data(n=80, censor_rate=0.5)
        y = np.abs(y) + 0.01
        cfg = {**FAST_CFG, "log_transform": True}
        model = ZeroInflatedTobitRegressor(config=cfg)
        model.fit(X, y, censored=censored, detection_limits=dl)
        preds = model.predict(X)
        assert len(preds) == 80
        assert np.all(np.isfinite(preds))

    def test_without_censoring(self, regression_data) -> None:
        X, y = regression_data
        model = ZeroInflatedTobitRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.all(np.isfinite(preds))
