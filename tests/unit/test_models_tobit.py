"""Tests for Tobit censored regression model."""

from __future__ import annotations

import numpy as np
import pytest

from aquacontam.models.tobit import TobitRegressor


class TestTobitRegressor:
    """Tests for TobitRegressor."""

    def test_name(self) -> None:
        model = TobitRegressor()
        assert model.name == "tobit_regressor"

    def test_fit_predict_uncensored(self, regression_data) -> None:
        """Fit on fully observed data (no censoring) — falls back to OLS."""
        X, y = regression_data
        model = TobitRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds))

    def test_fit_predict_with_censoring(self, regression_data) -> None:
        """Fit with censored observations via MLE."""
        X, y = regression_data
        rng = np.random.RandomState(42)
        censored = rng.random(len(y)) < 0.3
        detection_limits = np.full(len(y), float(y.median()))

        model = TobitRegressor()
        model.fit(X, y, censored=censored, detection_limits=detection_limits)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds))

    def test_predict_proba_raises(self, regression_data) -> None:
        """predict_proba is not supported for regression models."""
        X, y = regression_data
        model = TobitRegressor()
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_numpy_input(self) -> None:
        """Accepts raw numpy arrays (no DataFrame)."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = rng.random(50) * 10
        model = TobitRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50
        assert not np.any(np.isnan(preds))

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fit raises RuntimeError."""
        model = TobitRegressor()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(np.array([[1, 2, 3]]))

    def test_all_censored(self, regression_data) -> None:
        """Fit still converges when all observations are censored."""
        X, y = regression_data
        censored = np.ones(len(y), dtype=bool)
        detection_limits = np.asarray(y, dtype=np.float64)

        model = TobitRegressor()
        model.fit(X, y, censored=censored, detection_limits=detection_limits)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_censored_none_treated_as_uncensored(self, regression_data) -> None:
        """When censored=None, behaves identically to uncensored OLS."""
        X, y = regression_data
        model_a = TobitRegressor()
        model_a.fit(X, y, censored=None)

        model_b = TobitRegressor()
        model_b.fit(X, y)

        preds_a = model_a.predict(X)
        preds_b = model_b.predict(X)
        np.testing.assert_allclose(preds_a, preds_b)

    def test_dataframe_preserves_feature_names(self, regression_data) -> None:
        """Feature names are stored when a DataFrame is passed."""
        X, y = regression_data
        model = TobitRegressor()
        model.fit(X, y)
        assert model._feature_names == list(X.columns)

    def test_converged_property_none_before_fit(self) -> None:
        model = TobitRegressor()
        assert model.converged is None

    def test_converged_true_uncensored(self, regression_data) -> None:
        """OLS path always converges."""
        X, y = regression_data
        model = TobitRegressor()
        model.fit(X, y)
        assert model.converged is True

    def test_converged_set_after_censored_fit(self) -> None:
        """Convergence flag is set after fitting with censored data."""
        rng = np.random.RandomState(42)
        n = 100
        X = rng.randn(n, 3)
        y = X @ np.array([1.0, 0.5, 0.2]) + rng.randn(n) * 0.5
        censored = y < 0.5
        dl = np.full(n, 0.5)
        y[censored] = dl[censored]

        model = TobitRegressor()
        model.fit(X, y, censored=censored, detection_limits=dl)
        assert model.converged is not None
        assert isinstance(model.converged, bool)
