"""Tests for Hurdle (two-part) regressor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.models.hurdle import HurdleRegressor, _resolve_model

# Small config for fast tests
FAST_CFG: dict = {
    "gate_model": "random_forest_classifier",
    "gate_config": {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": 1,
    },
    "intensity_model": "random_forest_regressor",
    "intensity_config": {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
        "n_jobs": 1,
    },
    "log_transform": True,
    "random_state": 42,
}


def _make_zero_inflated_data(
    n: int = 200, n_features: int = 5, detect_rate: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic zero-inflated concentration data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Detection driven by first feature
    p_detect = 1.0 / (1.0 + np.exp(-X[:, 0]))
    detected = rng.random(n) < (p_detect * detect_rate * 2)
    concentrations = np.where(detected, rng.exponential(5.0, n), 0.0)
    return X, concentrations


class TestHurdleRegressor:
    """Tests for HurdleRegressor."""

    def test_name(self) -> None:
        model = HurdleRegressor()
        assert model.name == "hurdle_regressor"

    def test_fit_predict_basic(self) -> None:
        X, y = _make_zero_inflated_data(n=150, detect_rate=0.2)
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 150
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0)

    def test_with_dataframe(self) -> None:
        X_np, y = _make_zero_inflated_data(n=100, detect_rate=0.2)
        X = pd.DataFrame(X_np, columns=[f"feat_{i}" for i in range(5)])
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
        assert np.all(np.isfinite(preds))

    def test_predict_proba_raises(self) -> None:
        X, y = _make_zero_inflated_data(n=80, detect_rate=0.3)
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_predict_unfitted_raises(self) -> None:
        model = HurdleRegressor(config=FAST_CFG)
        X = np.random.default_rng(42).standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_all_zeros_fallback(self) -> None:
        """When all values are 0, should still produce predictions."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = np.zeros(50)
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50
        assert np.all(np.isfinite(preds))

    def test_no_log_transform(self) -> None:
        X, y = _make_zero_inflated_data(n=100, detect_rate=0.2)
        cfg = {**FAST_CFG, "log_transform": False}
        model = HurdleRegressor(config=cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0)

    def test_feature_importances(self) -> None:
        X_np, y = _make_zero_inflated_data(n=100, detect_rate=0.2)
        X = pd.DataFrame(X_np, columns=[f"feat_{i}" for i in range(5)])
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X, y)
        imp = model.feature_importances()
        assert isinstance(imp, pd.Series)
        assert len(imp) == 5

    def test_predictions_nonnegative(self) -> None:
        X, y = _make_zero_inflated_data(n=200, detect_rate=0.15)
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(preds >= 0.0)

    def test_requires_censoring_metadata_false(self) -> None:
        model = HurdleRegressor()
        assert model.requires_censoring_metadata is False

    def test_with_validation_data(self) -> None:
        X, y = _make_zero_inflated_data(n=200, detect_rate=0.2)
        model = HurdleRegressor(config=FAST_CFG)
        model.fit(X[:150], y[:150], X_val=X[150:], y_val=y[150:])
        preds = model.predict(X)
        assert len(preds) == 200
        assert np.all(np.isfinite(preds))


class TestResolveModel:
    """Tests for model registry resolution."""

    def test_resolve_known_model(self) -> None:
        model = _resolve_model("random_forest_classifier", {"n_estimators": 5, "random_state": 42})
        assert model.name == "random_forest_classifier"

    def test_resolve_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            _resolve_model("nonexistent_model")
