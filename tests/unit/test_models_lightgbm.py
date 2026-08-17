"""Tests for LightGBM model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from aquacontam.models.lightgbm import LightGBMClassifier, LightGBMRegressor  # noqa: E402


class TestLightGBMClassifier:
    """Tests for LightGBMClassifier."""

    def test_name(self) -> None:
        model = LightGBMClassifier()
        assert model.name == "lightgbm_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = LightGBMClassifier(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, classification_data) -> None:
        X, y = classification_data
        model = LightGBMClassifier(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importances(self, classification_data) -> None:
        X, y = classification_data
        model = LightGBMClassifier(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(int)

        model = LightGBMClassifier(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_evaluate(self, classification_data) -> None:
        X, y = classification_data
        model = LightGBMClassifier(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results

    def test_early_stopping(self, classification_data) -> None:
        X, y = classification_data
        X_val, y_val = X[:20], y[:20]

        model = LightGBMClassifier(
            config={
                "n_estimators": 100,
                "early_stopping_rounds": 5,
                "random_state": 42,
                "verbose": -1,
            }
        )
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestLightGBMRegressor:
    """Tests for LightGBMRegressor."""

    def test_name(self) -> None:
        model = LightGBMRegressor()
        assert model.name == "lightgbm_regressor"

    def test_fit_predict(self, regression_data) -> None:
        X, y = regression_data
        model = LightGBMRegressor(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_predict_proba_raises(self, regression_data) -> None:
        X, y = regression_data
        model = LightGBMRegressor(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_evaluate_regression(self, regression_data) -> None:
        X, y = regression_data
        model = LightGBMRegressor(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        results = model.evaluate(X, y, task_type="regression")
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

    def test_feature_importances(self, regression_data) -> None:
        X, y = regression_data
        model = LightGBMRegressor(config={"n_estimators": 10, "random_state": 42, "verbose": -1})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5
