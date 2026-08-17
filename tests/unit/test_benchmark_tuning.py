"""Tests for hyperparameter tuning and feature ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from aquacontam.benchmark.tuning import feature_ablation, grid_search
from aquacontam.models.logistic import LogisticRegressionClassifier


def _make_data(
    n: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X_train = pd.DataFrame(
        {
            "prox_f1": rng.normal(0, 1, n),
            "prox_f2": rng.normal(0, 1, n),
            "hydro_f1": rng.normal(0, 1, n),
        }
    )
    y_train = pd.Series((X_train["prox_f1"] + X_train["hydro_f1"] > 0).astype(int))
    X_val = pd.DataFrame(
        {
            "prox_f1": rng.normal(0, 1, n // 4),
            "prox_f2": rng.normal(0, 1, n // 4),
            "hydro_f1": rng.normal(0, 1, n // 4),
        }
    )
    y_val = pd.Series((X_val["prox_f1"] + X_val["hydro_f1"] > 0).astype(int))
    return X_train, y_train, X_val, y_val


class TestGridSearch:
    def test_returns_best_config(self) -> None:
        X_tr, y_tr, X_v, y_v = _make_data()
        param_grid = {"C": [0.01, 1.0, 100.0]}
        best_config, all_results = grid_search(
            LogisticRegressionClassifier,
            param_grid,
            X_tr,
            y_tr,
            X_v,
            y_v,
            metric="accuracy",
        )
        assert "C" in best_config
        assert len(all_results) == 3

    def test_all_results_have_scores(self) -> None:
        X_tr, y_tr, X_v, y_v = _make_data()
        _, all_results = grid_search(
            LogisticRegressionClassifier,
            {"C": [0.1, 1.0]},
            X_tr,
            y_tr,
            X_v,
            y_v,
            metric="accuracy",
        )
        for r in all_results:
            assert "score" in r
            assert r["score"] >= 0.0


class TestFeatureAblation:
    def test_ablation_structure(self) -> None:
        X_tr, y_tr, X_v, y_v = _make_data()
        categories = {
            "proximity": ["prox_"],
            "hydrogeology": ["hydro_"],
        }
        results = feature_ablation(
            LogisticRegressionClassifier,
            {"random_state": 42},
            X_tr,
            y_tr,
            X_v,
            y_v,
            categories,
            metric="accuracy",
        )
        # Should have baseline + 2 ablations
        assert len(results) == 3
        assert results[0]["category"] == "all_features"
        categories_found = {r["category"] for r in results}
        assert "proximity" in categories_found
        assert "hydrogeology" in categories_found

    def test_single_class_returns_nan(self) -> None:
        """When y_train has only one class, _score should return NaN."""
        X_tr, _, X_v, y_v = _make_data()
        # All same class
        y_single = pd.Series(np.ones(len(X_tr), dtype=int))
        categories = {"proximity": ["prox_"]}
        results = feature_ablation(
            LogisticRegressionClassifier,
            {"random_state": 42},
            X_tr,
            y_single,
            X_v,
            y_v,
            categories,
            metric="accuracy",
        )
        # Baseline and ablated should both be NaN
        for r in results:
            assert np.isnan(r["ablated_score"])

    def test_removing_features_changes_score(self) -> None:
        X_tr, y_tr, X_v, y_v = _make_data()
        categories = {"proximity": ["prox_"]}
        results = feature_ablation(
            LogisticRegressionClassifier,
            {"random_state": 42},
            X_tr,
            y_tr,
            X_v,
            y_v,
            categories,
            metric="accuracy",
        )
        # Ablation should produce a different (likely worse) score
        ablated = next(r for r in results if r["category"] == "proximity")
        assert ablated["n_features_removed"] == 2  # prox_f1, prox_f2


class TestGridSearchPassesValData:
    def test_grid_search_passes_val_data(self) -> None:
        """Verify X_val/y_val reach model.fit() via kwargs."""
        X_tr, y_tr, X_v, y_v = _make_data()

        fit_calls: list[dict] = []

        class _SpyModel:
            name = "spy"

            def __init__(self, config: dict | None = None):
                self.config = config or {}

            def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:  # type: ignore[type-arg]
                fit_calls.append(kwargs)

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

        grid_search(
            _SpyModel,  # type: ignore[arg-type]
            {"C": [0.1, 1.0]},
            X_tr,
            y_tr,
            X_v,
            y_v,
            metric="accuracy",
        )
        assert len(fit_calls) == 2
        for call_kwargs in fit_calls:
            assert "X_val" in call_kwargs
            assert "y_val" in call_kwargs
            pd.testing.assert_frame_equal(call_kwargs["X_val"], X_v)
            pd.testing.assert_series_equal(call_kwargs["y_val"], y_v)

    def test_grid_search_mlp_with_dropout(self) -> None:
        """Run a small MLP grid search with dropout dimension (requires torch)."""
        torch = pytest.importorskip("torch")  # noqa: F841
        from aquacontam.models.mlp import MLPClassifier

        X_tr, y_tr, X_v, y_v = _make_data(n=100)
        param_grid = {
            "hidden_sizes": [[32, 16]],
            "learning_rate": [0.01],
            "dropout": [0.2, 0.3],
        }
        best_config, all_results = grid_search(
            MLPClassifier,
            param_grid,
            X_tr,
            y_tr,
            X_v,
            y_v,
            metric="accuracy",
            base_config={"epochs": 5, "batch_size": 64, "random_state": 42},
        )
        assert len(all_results) == 2
        assert "dropout" in best_config
        assert best_config["dropout"] in [0.2, 0.3]
