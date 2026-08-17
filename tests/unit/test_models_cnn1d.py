"""Tests for 1D-CNN model wrappers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from aquacontam.models.cnn1d import CNN1DClassifier, CNN1DRegressor

# Small config for fast tests
FAST_CFG: dict = {
    "n_filters": [4, 8],
    "kernel_size": 3,
    "epochs": 3,
    "batch_size": 32,
    "random_state": 42,
}


class TestCNN1DClassifier:
    """Tests for CNN1DClassifier."""

    def test_name(self) -> None:
        model = CNN1DClassifier()
        assert model.name == "cnn1d_classifier"

    def test_fit_predict(self, classification_data) -> None:
        X, y = classification_data
        model = CNN1DClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, classification_data) -> None:
        X, y = classification_data
        model = CNN1DClassifier(config=FAST_CFG)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_early_stopping(self, classification_data) -> None:
        X, y = classification_data
        X_val, y_val = X[:20], y[:20]
        cfg = {**FAST_CFG, "epochs": 50, "patience": 2}
        model = CNN1DClassifier(config=cfg)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_numpy_input(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (rng.random(50) > 0.5).astype(int)
        model = CNN1DClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_custom_filters(self, classification_data) -> None:
        X, y = classification_data
        cfg = {**FAST_CFG, "n_filters": [4, 8, 4]}
        model = CNN1DClassifier(config=cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_evaluate(self, classification_data) -> None:
        X, y = classification_data
        model = CNN1DClassifier(config=FAST_CFG)
        model.fit(X, y)
        results = model.evaluate(X, y)
        assert "accuracy" in results
        assert "auroc" in results

    def test_all_positive_class(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        y = np.ones(20, dtype=int)
        model = CNN1DClassifier(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20

    def test_config_defaults(self) -> None:
        model = CNN1DClassifier()
        assert model.config == {}

    def test_predict_proba_unfitted_raises(self) -> None:
        model = CNN1DClassifier(config=FAST_CFG)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_proba_between_0_and_1(self, classification_data) -> None:
        X, y = classification_data
        model = CNN1DClassifier(config=FAST_CFG)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


class TestCNN1DRegressor:
    """Tests for CNN1DRegressor."""

    def test_name(self) -> None:
        model = CNN1DRegressor()
        assert model.name == "cnn1d_regressor"

    def test_fit_predict(self, regression_data) -> None:
        X, y = regression_data
        model = CNN1DRegressor(config=FAST_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype in (np.float32, np.float64)

    def test_predict_proba_raises(self, regression_data) -> None:
        X, y = regression_data
        model = CNN1DRegressor(config=FAST_CFG)
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)

    def test_evaluate_regression(self, regression_data) -> None:
        X, y = regression_data
        model = CNN1DRegressor(config=FAST_CFG)
        model.fit(X, y)
        results = model.evaluate(X, y, task_type="regression")
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

    def test_early_stopping(self, regression_data) -> None:
        X, y = regression_data
        X_val, y_val = X[:20], y[:20]
        cfg = {**FAST_CFG, "epochs": 50, "patience": 2}
        model = CNN1DRegressor(config=cfg)
        model.fit(X, y, X_val=X_val, y_val=y_val)
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestCNN1DValColumnAlignment:
    """Tests for validation column reordering in CNN1D."""

    def test_val_columns_reordered_to_match_train(self) -> None:
        """Validation data with different column order should be aligned."""
        import pandas as pd

        rng = np.random.RandomState(42)
        cols = ["z_feat", "a_feat", "m_feat", "b_feat"]
        X_train = pd.DataFrame(rng.randn(60, 4), columns=cols)
        y_train = (rng.random(60) > 0.5).astype(int)

        # Reverse column order for validation
        X_val = pd.DataFrame(rng.randn(20, 4), columns=cols[::-1])
        y_val = (rng.random(20) > 0.5).astype(int)

        model = CNN1DClassifier(config={**FAST_CFG, "patience": 2, "epochs": 10})
        # Should not raise — columns are reordered internally
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)

    def test_val_columns_reordered_regressor(self) -> None:
        """Regressor validation data with different column order."""
        import pandas as pd

        rng = np.random.RandomState(42)
        cols = ["z_feat", "a_feat", "m_feat", "b_feat"]
        X_train = pd.DataFrame(rng.randn(60, 4), columns=cols)
        y_train = rng.randn(60)

        X_val = pd.DataFrame(rng.randn(20, 4), columns=cols[::-1])
        y_val = rng.randn(20)

        model = CNN1DRegressor(config={**FAST_CFG, "patience": 2, "epochs": 10})
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)


class TestCNN1DDomainOrdering:
    """Tests for domain-based feature ordering in CNN1D."""

    def test_domain_ordering_classifier(self, classification_data) -> None:
        X, y = classification_data
        cfg = {**FAST_CFG, "feature_order": "domain"}
        model = CNN1DClassifier(config=cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_domain_ordering_regressor(self, regression_data) -> None:
        X, y = regression_data
        cfg = {**FAST_CFG, "feature_order": "domain"}
        model = CNN1DRegressor(config=cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_domain_ordering_preserves_columns(self, classification_data) -> None:
        """Domain ordering should reorder but not lose columns."""
        X, y = classification_data
        cfg = {**FAST_CFG, "feature_order": "domain"}
        model = CNN1DClassifier(config=cfg)
        model.fit(X, y)
        assert hasattr(model, "feature_names_in_")
        assert set(model.feature_names_in_) == set(X.columns)
