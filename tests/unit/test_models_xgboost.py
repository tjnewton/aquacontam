"""Tests for XGBoost model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.models.xgboost import XGBoostClassifier, XGBoostRegressor


class TestXGBoostClassifier:
    """Tests for XGBoostClassifier."""

    def test_name(self) -> None:
        model = XGBoostClassifier()
        assert model.name == "xgboost_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, classification_data) -> None:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_auto_scale_pos_weight(self, classification_data) -> None:
        X, y = classification_data
        model = XGBoostClassifier(
            config={
                "n_estimators": 10,
                "scale_pos_weight": "auto",
                "random_state": 42,
            }
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_early_stopping(self, classification_data) -> None:
        X, y = classification_data
        X_val, y_val = X[:20], y[:20]

        model = XGBoostClassifier(
            config={
                "n_estimators": 100,
                "early_stopping_rounds": 5,
                "random_state": 42,
            }
        )
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_feature_importances(self, classification_data) -> None:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5
        assert importances.is_monotonic_decreasing

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(int)

        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_all_same_class_auto_weight(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])

        model = XGBoostClassifier(
            config={
                "n_estimators": 10,
                "scale_pos_weight": "auto",
                "random_state": 42,
            }
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 3

    def test_evaluate_classification(self, classification_data) -> None:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results

    def test_evaluate_with_task_type(self, classification_data) -> None:
        X, y = classification_data
        model = XGBoostClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        results = model.evaluate(X, y, task_type="classification")
        assert "accuracy" in results


class TestXGBoostRegressor:
    """Tests for XGBoostRegressor."""

    def test_name(self) -> None:
        model = XGBoostRegressor()
        assert model.name == "xgboost_regressor"

    def test_fit_predict(self, regression_data) -> None:
        X, y = regression_data
        model = XGBoostRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype == np.float32 or preds.dtype == np.float64

    def test_predict_proba_raises(self, regression_data) -> None:
        X, y = regression_data
        model = XGBoostRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_evaluate_regression(self, regression_data) -> None:
        X, y = regression_data
        model = XGBoostRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        results = model.evaluate(X, y, task_type="regression")
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

    def test_feature_importances(self, regression_data) -> None:
        X, y = regression_data
        model = XGBoostRegressor(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)
        importances = model.feature_importances()
        assert isinstance(importances, pd.Series)
        assert len(importances) == 5
