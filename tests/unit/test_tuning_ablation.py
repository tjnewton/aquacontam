"""Tests for benchmark.tuning — feature ablation and grid search."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.tuning import feature_ablation
from aquacontam.models.random_forest import RandomForestClassifier


@pytest.fixture()
def ablation_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create synthetic train/val data for ablation tests."""
    rng = np.random.RandomState(42)
    n_train, n_val = 100, 30
    X_train = pd.DataFrame(
        {
            "dist_nearest_industrial": rng.rand(n_train),
            "dist_nearest_military": rng.rand(n_train),
            "pct_developed_5km": rng.rand(n_train),
            "pct_forest_5km": rng.rand(n_train),
            "aquifer_type_alluvial": rng.randint(0, 2, n_train),
        }
    )
    y_train = pd.Series(rng.randint(0, 2, n_train))
    X_val = pd.DataFrame(
        {
            "dist_nearest_industrial": rng.rand(n_val),
            "dist_nearest_military": rng.rand(n_val),
            "pct_developed_5km": rng.rand(n_val),
            "pct_forest_5km": rng.rand(n_val),
            "aquifer_type_alluvial": rng.randint(0, 2, n_val),
        }
    )
    y_val = pd.Series(rng.randint(0, 2, n_val))
    return X_train, y_train, X_val, y_val


class TestFeatureAblation:
    """Tests for feature_ablation()."""

    def test_returns_baseline_and_categories(
        self, ablation_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ) -> None:
        X_train, y_train, X_val, y_val = ablation_data
        categories = {
            "proximity": ["dist_nearest_"],
            "land_use": ["pct_developed_", "pct_forest_"],
        }
        results = feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            categories,
        )
        cats = [r["category"] for r in results]
        assert "all_features" in cats
        assert "proximity" in cats
        assert "land_use" in cats

    def test_zero_match_category_produces_warning_and_nan(
        self,
        ablation_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        X_train, y_train, X_val, y_val = ablation_data
        categories = {
            "nonexistent_category": ["zzz_no_match_"],
        }
        with caplog.at_level(logging.WARNING, logger="aquacontam.benchmark.tuning"):
            results = feature_ablation(
                RandomForestClassifier,
                {"n_estimators": 10, "random_state": 42},
                X_train,
                y_train,
                X_val,
                y_val,
                categories,
            )
        # Should have baseline + the zero-match category entry
        cats = {r["category"]: r for r in results}
        assert "nonexistent_category" in cats
        entry = cats["nonexistent_category"]
        assert entry["n_features_removed"] == 0
        assert np.isnan(entry["ablated_score"])
        assert entry["note"] == "no matching columns found"
        # Should have logged a warning
        assert any("no matching columns found" in msg for msg in caplog.messages)

    def test_single_class_y_train_returns_nan_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rng = np.random.RandomState(42)
        n = 50
        X_train = pd.DataFrame({"feat_a": rng.rand(n), "feat_b": rng.rand(n)})
        y_train = pd.Series(np.zeros(n, dtype=int))  # Single class
        X_val = pd.DataFrame({"feat_a": rng.rand(20), "feat_b": rng.rand(20)})
        y_val = pd.Series(np.zeros(20, dtype=int))
        categories = {"group_a": ["feat_a"]}
        with caplog.at_level(logging.WARNING, logger="aquacontam.benchmark.tuning"):
            results = feature_ablation(
                RandomForestClassifier,
                {"n_estimators": 10, "random_state": 42},
                X_train,
                y_train,
                X_val,
                y_val,
                categories,
            )
        baseline = next(r for r in results if r["category"] == "all_features")
        assert np.isnan(baseline["baseline_score"])
        assert any("Single-class" in msg for msg in caplog.messages)

    def test_single_class_y_val_returns_nan(self, caplog: pytest.LogCaptureFixture) -> None:
        """Single-class y_val should produce NaN scores with a warning."""
        rng = np.random.RandomState(42)
        n = 50
        X_train = pd.DataFrame({"feat_a": rng.rand(n), "feat_b": rng.rand(n)})
        y_train = pd.Series(rng.randint(0, 2, n))  # Two-class train
        X_val = pd.DataFrame({"feat_a": rng.rand(20), "feat_b": rng.rand(20)})
        y_val = pd.Series(np.ones(20, dtype=int))  # Single class val
        categories = {"group_a": ["feat_a"]}
        with caplog.at_level(logging.WARNING, logger="aquacontam.benchmark.tuning"):
            results = feature_ablation(
                RandomForestClassifier,
                {"n_estimators": 10, "random_state": 42},
                X_train,
                y_train,
                X_val,
                y_val,
                categories,
            )
        baseline = next(r for r in results if r["category"] == "all_features")
        assert np.isnan(baseline["baseline_score"])
        assert any("Single-class y_val" in msg for msg in caplog.messages)

    def test_all_features_removed_returns_nan(self) -> None:
        """When all columns match the category prefixes, return NaN with note."""
        rng = np.random.RandomState(42)
        n = 50
        X_train = pd.DataFrame({"feat_a": rng.rand(n), "feat_b": rng.rand(n)})
        y_train = pd.Series(rng.randint(0, 2, n))
        X_val = pd.DataFrame({"feat_a": rng.rand(20), "feat_b": rng.rand(20)})
        y_val = pd.Series(rng.randint(0, 2, 20))
        categories = {"everything": ["feat_"]}
        results = feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            categories,
        )
        entry = next(r for r in results if r["category"] == "everything")
        assert np.isnan(entry["ablated_score"])
        assert entry["note"] == "all features removed"

    def test_two_class_y_val_produces_finite_scores(
        self,
        ablation_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    ) -> None:
        """Regression guard: normal two-class case produces finite scores."""
        X_train, y_train, X_val, y_val = ablation_data
        categories = {"proximity": ["dist_nearest_"]}
        results = feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            categories,
        )
        baseline = next(r for r in results if r["category"] == "all_features")
        assert np.isfinite(baseline["baseline_score"])
        ablated = next(r for r in results if r["category"] == "proximity")
        assert np.isfinite(ablated["ablated_score"])
