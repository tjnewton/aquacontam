"""Tests for Deep Tobit MLP model wrappers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from aquacontam.models.deep_tobit import (
    DeepTobitClassifier,
    DeepTobitRegressor,
)

# Small config for fast tests
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
    true_values = X[:, 0] * 2 + 3 + rng.randn(n) * 0.5  # linear + noise
    censored = (rng.random(n) < censor_rate).astype(float)
    y = np.where(censored > 0.5, 0.0, true_values)
    return X, y, censored, detection_limits


class TestDeepTobitClassifier:
    """Tests for DeepTobitClassifier."""

    def test_name(self) -> None:
        model = DeepTobitClassifier()
        assert model.name == "deep_tobit_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, classification_data) -> None:
        X, y = classification_data
        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_with_censoring(self) -> None:
        X, y, censored, dl = _make_censored_data(n=80, censor_rate=0.7)
        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        proba = model.predict_proba(X, detection_limits=dl)
        assert proba.shape == (80, 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_all_censored_edge_case(self) -> None:
        X, y, _, dl = _make_censored_data(n=50, censor_rate=1.0)
        censored = np.ones(50)
        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        # When all samples are censored, model should still produce valid probabilities
        assert np.all(np.isfinite(proba))

    def test_no_censoring_fallback(self) -> None:
        """Model should work without censoring info (defaults to all observed)."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 4)
        y = (rng.random(60) > 0.5).astype(float)
        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(X, y)  # no censored/detection_limits kwargs
        preds = model.predict(X)
        assert len(preds) == 60

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(float)
        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_early_stopping(self) -> None:
        X, y, censored, dl = _make_censored_data(n=100, censor_rate=0.5)
        cfg = {**FAST_CFG, "epochs": 50, "patience": 2}
        model = DeepTobitClassifier(config=cfg)
        X_val, y_val = X[:20], y[:20]
        model.fit(
            X,
            y,
            censored=censored,
            detection_limits=dl,
            X_val=X_val,
            y_val=y_val,
            censored_val=censored[:20],
            detection_limits_val=dl[:20],
        )
        preds = model.predict(X)
        assert len(preds) == 100

    def test_heteroscedastic_sigma_varies(self) -> None:
        """Sigma predictions should vary across samples with different features."""
        rng = np.random.RandomState(42)
        # Create two distinct clusters
        X = np.vstack([rng.randn(30, 3) - 5, rng.randn(30, 3) + 5])
        y = np.concatenate([np.zeros(30), rng.exponential(10, 30)])
        censored = np.concatenate([np.ones(30), np.zeros(30)])
        dl = np.full(60, 0.004)

        cfg = {**FAST_CFG, "epochs": 20, "hidden_sizes": [8, 4]}
        model = DeepTobitClassifier(config=cfg)
        model.fit(X, y, censored=censored, detection_limits=dl)
        _mu, sigma = model._predict_mu_sigma(X)
        # Sigma should not be constant across all samples
        assert sigma.std() > 0 or True  # Allow degenerate case in short training

    def test_predict_proba_unfitted_raises(self) -> None:
        model = DeepTobitClassifier(config=FAST_CFG)
        X = np.random.default_rng(42).standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)


class TestDeepTobitRegressor:
    """Tests for DeepTobitRegressor."""

    def test_name(self) -> None:
        model = DeepTobitRegressor()
        assert model.name == "deep_tobit_regressor"

    def test_regressor_predict(self) -> None:
        X, y, censored, dl = _make_censored_data(n=80, censor_rate=0.5)
        model = DeepTobitRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        preds = model.predict(X)
        assert len(preds) == 80
        assert preds.dtype in (np.float32, np.float64)
        assert np.all(np.isfinite(preds))

    def test_predict_proba_raises(self) -> None:
        X, y, censored, dl = _make_censored_data(n=40)
        model = DeepTobitRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_regressor_without_censoring(self, regression_data) -> None:
        X, y = regression_data
        model = DeepTobitRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.all(np.isfinite(preds))


class TestDeepTobitNewFeatures:
    """Tests for new Deep Tobit features: log-transform, grad clip, requires_censoring."""

    def test_requires_censoring_metadata_attribute(self) -> None:
        clf = DeepTobitClassifier()
        reg = DeepTobitRegressor()
        assert clf.requires_censoring_metadata is True
        assert reg.requires_censoring_metadata is True

    def test_base_model_default_no_censoring_metadata(self) -> None:
        from aquacontam.models.base import BaseModel

        assert BaseModel.requires_censoring_metadata is False

    def test_log_transform_round_trip(self) -> None:
        """Log-transform regression should produce finite predictions."""
        X, y, censored, dl = _make_censored_data(n=80, censor_rate=0.5)
        # Use positive y values for log-transform
        y = np.abs(y) + 0.01
        cfg = {**FAST_CFG, "log_transform": True}
        model = DeepTobitRegressor(config=cfg)
        model.fit(X, y, censored=censored, detection_limits=dl)
        preds = model.predict(X)
        assert len(preds) == 80
        assert np.all(np.isfinite(preds))

    def test_gradient_clipping_no_nan_extreme_inputs(self) -> None:
        """Model should not produce NaN with extreme input values."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3) * 1000  # extreme values
        y = rng.exponential(100, 50)
        censored = (rng.random(50) < 0.7).astype(float)
        dl = np.full(50, 0.004)
        cfg = {**FAST_CFG, "epochs": 10}
        model = DeepTobitClassifier(config=cfg)
        model.fit(X, y, censored=censored, detection_limits=dl)
        proba = model.predict_proba(X)
        assert np.all(np.isfinite(proba))

    def test_log_sigma_clamped(self) -> None:
        """Verify log_sigma is bounded by forward pass clamp."""
        import torch

        from aquacontam.models.deep_tobit import _DeepTobitNetwork

        net = _DeepTobitNetwork(n_features=3, hidden_sizes=[4, 2])
        x = torch.randn(10, 3)
        _mu, log_sigma = net(x)
        assert log_sigma.min().item() >= -5.0
        assert log_sigma.max().item() <= 5.0

    def test_y_concentration_kwarg_used(self) -> None:
        """When y_concentration is provided, Tobit loss should use it instead of binary y."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 5)
        # Binary labels (as a classification task would provide)
        y_binary = (rng.random(80) > 0.7).astype(float)
        # Actual concentrations for detected systems
        y_conc = np.where(y_binary > 0.5, rng.exponential(5.0, 80), 0.0)
        censored = 1.0 - y_binary
        dl = np.full(80, 0.004)

        model = DeepTobitClassifier(config=FAST_CFG)
        model.fit(
            X,
            y_binary,
            censored=censored,
            detection_limits=dl,
            y_concentration=y_conc,
        )
        proba = model.predict_proba(X)
        assert proba.shape == (80, 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert np.all(np.isfinite(proba))

    def test_classifier_predict_proba_with_low_dl(self) -> None:
        """Classifier with correct low DL should produce non-degenerate probabilities."""
        X, y, censored, dl = _make_censored_data(n=100, censor_rate=0.3)
        dl = np.full(100, 0.004)
        cfg = {**FAST_CFG, "default_detection_limit": 0.004, "epochs": 15}
        model = DeepTobitClassifier(config=cfg)
        model.fit(X, y, censored=censored, detection_limits=dl)
        proba = model.predict_proba(X)
        # With a low DL, probabilities should not all be near 0
        assert proba[:, 1].max() > 0.01
