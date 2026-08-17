"""Tests for XGBoost AFT survival regressor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from aquacontam.models.xgboost_aft import XGBoostAFTRegressor

FAST_CFG: dict = {
    "n_estimators": 20,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": 42,
}


def _make_censored_data(
    n: int = 100, n_features: int = 5, censor_rate: float = 0.7, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic censored data for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    detection_limits = rng.uniform(1.0, 5.0, n)
    true_values = np.abs(X[:, 0] * 2 + 3 + rng.randn(n) * 0.5)
    censored = (rng.random(n) < censor_rate).astype(float)
    y = np.where(censored > 0.5, 0.0, true_values)
    return X, y, censored, detection_limits


class TestXGBoostAFTRegressor:
    """Tests for XGBoostAFTRegressor."""

    def test_name(self) -> None:
        model = XGBoostAFTRegressor()
        assert model.name == "xgboost_aft_regressor"

    def test_requires_censoring_metadata(self) -> None:
        model = XGBoostAFTRegressor()
        assert model.requires_censoring_metadata is True

    def test_fit_predict(self) -> None:
        X, y, censored, dl = _make_censored_data(n=100)
        model = XGBoostAFTRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        preds = model.predict(X)
        assert len(preds) == 100
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0)

    def test_with_dataframe(self) -> None:
        X_np, y, censored, dl = _make_censored_data(n=80)
        X = pd.DataFrame(X_np, columns=[f"feat_{i}" for i in range(5)])
        model = XGBoostAFTRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        preds = model.predict(X)
        assert len(preds) == 80
        assert np.all(np.isfinite(preds))

    def test_without_censoring_info(self) -> None:
        """Should work without explicit censoring (defaults to all observed)."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 4)
        y = np.abs(rng.randn(60) * 3 + 5)
        model = XGBoostAFTRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60
        assert np.all(np.isfinite(preds))

    def test_predict_proba_raises(self) -> None:
        X, y, censored, dl = _make_censored_data(n=40)
        model = XGBoostAFTRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_predict_unfitted_raises(self) -> None:
        model = XGBoostAFTRegressor(config=FAST_CFG)
        X = np.random.default_rng(42).standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_feature_importances(self) -> None:
        X_np, y, censored, dl = _make_censored_data(n=100)
        X = pd.DataFrame(X_np, columns=[f"feat_{i}" for i in range(5)])
        model = XGBoostAFTRegressor(config=FAST_CFG)
        model.fit(X, y, censored=censored, detection_limits=dl)
        imp = model.feature_importances()
        assert isinstance(imp, pd.Series)
        assert len(imp) == 5

    def test_with_validation(self) -> None:
        X, y, censored, dl = _make_censored_data(n=120)
        model = XGBoostAFTRegressor(config=FAST_CFG)
        model.fit(
            X[:80],
            y[:80],
            censored=censored[:80],
            detection_limits=dl[:80],
            X_val=X[80:],
            y_val=y[80:],
            censored_val=censored[80:],
            detection_limits_val=dl[80:],
        )
        preds = model.predict(X)
        assert len(preds) == 120

    def test_encode_intervals(self) -> None:
        y = np.array([0.0, 5.0, 0.0, 10.0])
        censored = np.array([1.0, 0.0, 1.0, 0.0])
        dl = np.array([2.0, 2.0, 3.0, 3.0])
        lower, upper = XGBoostAFTRegressor._encode_intervals(y, censored, dl)
        # Detected: lower == upper == log1p(y)
        assert np.isclose(lower[1], np.log1p(5.0))
        assert np.isclose(upper[1], np.log1p(5.0))
        assert np.isclose(lower[3], np.log1p(10.0))
        # Censored: lower = eps, upper = log1p(dl)
        assert lower[0] < 0.001
        assert np.isclose(upper[0], np.log1p(2.0))
        assert lower[2] < 0.001
        assert np.isclose(upper[2], np.log1p(3.0))
