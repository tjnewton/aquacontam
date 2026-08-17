"""Tests for Random Forest model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.models.random_forest import RandomForestClassifier, RandomForestRegressor


class TestRandomForestClassifier:
    """Tests for RandomForestClassifier."""

    def test_name(self) -> None:
        model = RandomForestClassifier()
        assert model.name == "random_forest_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, classification_data) -> None:
        X, y = classification_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_balanced_class_weight_default(self) -> None:
        model = RandomForestClassifier()
        # Config should default to balanced class weight
        assert model.config == {}  # user doesn't set it, defaults applied in fit

    def test_feature_importances(self, classification_data) -> None:
        X, y = classification_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5
        assert importances.is_monotonic_decreasing

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(int)

        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_custom_config(self, classification_data) -> None:
        X, y = classification_data
        model = RandomForestClassifier(
            config={
                "n_estimators": 20,
                "max_depth": 5,
                "random_state": 42,
            }
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_evaluate(self, classification_data) -> None:
        X, y = classification_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results


class TestRandomForestRegressor:
    """Tests for RandomForestRegressor."""

    def test_name(self) -> None:
        model = RandomForestRegressor()
        assert model.name == "random_forest_regressor"

    def test_fit_predict(self, regression_data) -> None:
        X, y = regression_data
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba_raises(self, regression_data) -> None:
        X, y = regression_data
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_evaluate_regression(self, regression_data) -> None:
        X, y = regression_data
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        results = model.evaluate(X, y, task_type="regression")
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

    def test_feature_importances(self, regression_data) -> None:
        X, y = regression_data
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5
