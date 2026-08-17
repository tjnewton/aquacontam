"""Tests for logistic regression and dummy classifier baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from aquacontam.models.logistic import DummyClassifierBaseline, LogisticRegressionClassifier


@pytest.fixture()
def binary_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Simple binary classification data."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, 100),
            "f2": rng.normal(0, 1, 100),
        }
    )
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    return X, y


class TestLogisticRegressionClassifier:
    """Tests for LogisticRegressionClassifier."""

    def test_fit_predict(self, binary_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = binary_data
        model = LogisticRegressionClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, binary_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = binary_data
        model = LogisticRegressionClassifier()
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_name(self) -> None:
        model = LogisticRegressionClassifier()
        assert model.name == "logistic_regression"

    def test_not_fitted_raises(self) -> None:
        model = LogisticRegressionClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([[1, 2]]))

    def test_stores_feature_names(self, binary_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = binary_data
        model = LogisticRegressionClassifier()
        model.fit(X, y)
        assert hasattr(model, "feature_names_in_")
        assert model.feature_names_in_ == ["f1", "f2"]


class TestDummyClassifierBaseline:
    """Tests for DummyClassifierBaseline."""

    def test_fit_predict(self, binary_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = binary_data
        model = DummyClassifierBaseline()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_predict_proba(self, binary_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = binary_data
        model = DummyClassifierBaseline()
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)

    def test_name(self) -> None:
        model = DummyClassifierBaseline()
        assert model.name == "dummy_classifier"

    def test_prior_strategy(self, binary_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = binary_data
        model = DummyClassifierBaseline(config={"strategy": "prior"})
        model.fit(X, y)
        probs = model.predict_proba(X)
        # All probability predictions should be the same (class prior)
        assert np.allclose(probs[0], probs[1])
