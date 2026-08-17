"""Tests for AFT survival regression model."""

from __future__ import annotations

import numpy as np
import pytest

lifelines = pytest.importorskip("lifelines")

from aquacontam.models.aft import AFTRegressor  # noqa: E402


class TestAFTRegressor:
    """Tests for AFTRegressor."""

    def test_name(self) -> None:
        model = AFTRegressor()
        assert model.name == "aft_regressor"

    def test_fit_predict(self, regression_data) -> None:
        """Fit on fully observed data (no censoring)."""
        X, y = regression_data
        model = AFTRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds))

    def test_fit_predict_with_censoring(self, regression_data) -> None:
        """Fit with left-censored observations."""
        X, y = regression_data
        rng = np.random.RandomState(42)
        censored = rng.random(len(y)) < 0.3

        model = AFTRegressor()
        model.fit(X, y, censored=censored)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba_raises(self, regression_data) -> None:
        """predict_proba is not supported for regression models."""
        X, y = regression_data
        model = AFTRegressor()
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_numpy_input(self) -> None:
        """Accepts raw numpy arrays (no DataFrame)."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = rng.random(50) * 10 + 0.1  # ensure positive

        model = AFTRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fit raises RuntimeError."""
        model = AFTRegressor()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(np.array([[1, 2, 3]]))

    def test_custom_penalizer(self, regression_data) -> None:
        """Config penalizer is forwarded to WeibullAFTFitter."""
        X, y = regression_data
        model = AFTRegressor(config={"penalizer": 0.1})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_dataframe_preserves_feature_names(self, regression_data) -> None:
        """Feature names are stored when a DataFrame is passed."""
        X, y = regression_data
        model = AFTRegressor()
        model.fit(X, y)
        assert model._feature_names == list(X.columns)
