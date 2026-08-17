"""Tests for MLP model wrappers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from aquacontam.models.mlp import MLPClassifier, MLPRegressor

# Small config for fast tests
FAST_CFG: dict = {
    "hidden_sizes": [4, 2],
    "epochs": 3,
    "batch_size": 32,
    "random_state": 42,
}


class TestMLPClassifier:
    """Tests for MLPClassifier."""

    def test_name(self) -> None:
        model = MLPClassifier()
        assert model.name == "mlp_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = MLPClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, classification_data) -> None:
        X, y = classification_data
        model = MLPClassifier(config=FAST_CFG)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_early_stopping(self, classification_data) -> None:
        X, y = classification_data
        X_val, y_val = X[:20], y[:20]
        cfg = {**FAST_CFG, "epochs": 50, "patience": 2}
        model = MLPClassifier(config=cfg)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(int)
        model = MLPClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_custom_hidden_sizes(self, classification_data) -> None:
        X, y = classification_data
        cfg = {**FAST_CFG, "hidden_sizes": [8, 4, 2]}
        model = MLPClassifier(config=cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_evaluate(self, classification_data) -> None:
        X, y = classification_data
        model = MLPClassifier(config=FAST_CFG)
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results

    def test_all_positive_class(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        y = np.ones(20, dtype=int)
        model = MLPClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20

    def test_config_defaults(self) -> None:
        model = MLPClassifier()
        assert model.config == {}

    def test_predict_proba_unfitted_raises(self) -> None:
        model = MLPClassifier(config=FAST_CFG)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_proba_between_0_and_1(self, classification_data) -> None:
        X, y = classification_data
        model = MLPClassifier(config=FAST_CFG)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


class TestMLPRegressor:
    """Tests for MLPRegressor."""

    def test_name(self) -> None:
        model = MLPRegressor()
        assert model.name == "mlp_regressor"

    def test_fit_predict(self, regression_data) -> None:
        X, y = regression_data
        model = MLPRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype in (np.float32, np.float64)

    def test_predict_proba_raises(self, regression_data) -> None:
        X, y = regression_data
        model = MLPRegressor(config=FAST_CFG)
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_evaluate_regression(self, regression_data) -> None:
        X, y = regression_data
        model = MLPRegressor(config=FAST_CFG)
        model.fit(X, y)
        results = model.evaluate(X, y, task_type="regression")
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

    def test_early_stopping(self, regression_data) -> None:
        X, y = regression_data
        X_val, y_val = X[:20], y[:20]
        cfg = {**FAST_CFG, "epochs": 50, "patience": 2}
        model = MLPRegressor(config=cfg)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert len(preds) == len(X)
