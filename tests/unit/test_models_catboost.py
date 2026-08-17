"""Tests for CatBoost model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from aquacontam.models.catboost import CatBoostClassifier, CatBoostRegressor


class TestCatBoostClassifier:
    """Tests for CatBoostClassifier."""

    def test_name(self) -> None:
        model = CatBoostClassifier()
        assert model.name == "catboost_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = CatBoostClassifier(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, classification_data) -> None:
        X, y = classification_data
        model = CatBoostClassifier(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importances(self, classification_data) -> None:
        X, y = classification_data
        model = CatBoostClassifier(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(int)

        model = CatBoostClassifier(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_evaluate(self, classification_data) -> None:
        X, y = classification_data
        model = CatBoostClassifier(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results


class TestCatBoostRegressor:
    """Tests for CatBoostRegressor."""

    def test_name(self) -> None:
        model = CatBoostRegressor()
        assert model.name == "catboost_regressor"

    def test_fit_predict(self, regression_data) -> None:
        X, y = regression_data
        model = CatBoostRegressor(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype == np.float32 or preds.dtype == np.float64

    def test_predict_proba_raises(self, regression_data) -> None:
        X, y = regression_data
        model = CatBoostRegressor(config={"iterations": 10, "random_seed": 42, "verbose": 0})
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)
