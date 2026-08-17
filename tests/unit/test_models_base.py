"""Tests for the BaseModel abstract base class contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from aquacontam.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Minimal concrete subclass for testing."""

    @property
    def name(self) -> str:
        return "test_model"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        self._model = object()  # mark as fitted

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class NoProbModel(BaseModel):
    """Concrete subclass that does not support predict_proba."""

    @property
    def name(self) -> str:
        return "no_prob_model"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        self._model = object()  # mark as fitted

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError("This model does not support probability estimates")


class TestBaseModelABC:
    """Tests for BaseModel ABC enforcement."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseModel()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        model = ConcreteModel()
        assert model.name == "test_model"

    def test_default_config_is_empty_dict(self) -> None:
        model = ConcreteModel()
        assert model.config == {}

    def test_config_passed_through(self) -> None:
        model = ConcreteModel(config={"n_estimators": 100})
        assert model.config["n_estimators"] == 100


class TestBaseModelEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_accuracy(self) -> None:
        model = ConcreteModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 0])
        model.fit(X, y)
        results = model.evaluate(X, y, metrics=["accuracy"])
        assert results["accuracy"] == pytest.approx(2 / 3)
        assert "auroc" not in results

    def test_evaluate_auroc(self) -> None:
        model = ConcreteModel()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 1, 0, 0])
        model.fit(X, y)
        results = model.evaluate(X, y, metrics=["auroc"])
        assert "auroc" in results
        assert 0.0 <= results["auroc"] <= 1.0

    def test_evaluate_default_metrics(self) -> None:
        model = ConcreteModel()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 1, 0, 0])
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results

    def test_evaluate_auroc_nan_when_proba_unavailable(self) -> None:
        model = NoProbModel()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 1, 0, 0])
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert results["accuracy"] == pytest.approx(0.5)
        assert np.isnan(results["auroc"])

    def test_evaluate_custom_metrics_list(self) -> None:
        model = ConcreteModel()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])
        model.fit(X, y)
        results = model.evaluate(X, y, metrics=["accuracy"])
        assert list(results.keys()) == ["accuracy"]


class TestFeatureImportances:
    """Tests for the feature_importances() method."""

    def test_feature_importances_not_supported(self) -> None:
        model = ConcreteModel()
        model.fit(np.array([[1, 2]]), np.array([1]))
        with pytest.raises(NotImplementedError, match="does not support"):
            model.feature_importances()

    def test_feature_importances_unfitted_raises(self) -> None:
        model = ConcreteModel()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.feature_importances()
