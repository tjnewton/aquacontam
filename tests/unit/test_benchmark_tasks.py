"""Tests for benchmark task definitions (T1, T2, T4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.registry import TaskResult, get_task, list_tasks
from aquacontam.benchmark.tasks import run_t1, run_t2, run_t4
from aquacontam.models.random_forest import RandomForestClassifier, RandomForestRegressor


class TestTaskRegistration:
    """Tests for task registration via decorators."""

    def test_t1_registered(self) -> None:
        info, _ = get_task("T1")
        assert info.task_type == "classification"
        assert info.primary_metric == "auprc"

    def test_t2_registered(self) -> None:
        info, _ = get_task("T2")
        assert info.task_type == "regression"
        assert info.primary_metric == "rmse"

    def test_t4_registered(self) -> None:
        info, _ = get_task("T4")
        assert info.task_type == "classification"
        assert info.primary_metric == "auprc"

    def test_all_tasks_listed(self) -> None:
        tasks = list_tasks()
        names = [t.name for t in tasks]
        assert "T1" in names
        assert "T2" in names
        assert "T4" in names


class TestT1Task:
    """Tests for T1 — binary PFAS detection."""

    def test_t1_runs(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t1(
            model=model,
            data=synthetic_wq_df,
            analyte="PFOS",
        )
        assert isinstance(result, TaskResult)
        assert result.task_name == "T1"
        assert result.model_name == "random_forest_classifier"

    def test_t1_has_metrics(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t1(model=model, data=synthetic_wq_df, analyte="PFOS")
        assert "accuracy" in result.metrics or "auprc" in result.metrics

    def test_t1_has_split_metrics(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t1(model=model, data=synthetic_wq_df, analyte="PFOS")
        assert "train" in result.split_metrics

    def test_t1_threshold_optimization(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t1(model=model, data=synthetic_wq_df, analyte="PFOS")
        meta = result.metadata
        # Should have threshold optimization results
        assert "threshold_optimization" in meta
        thresh_info = meta["threshold_optimization"]
        if thresh_info:  # May be empty if val set is degenerate
            assert "val_analysis" in thresh_info
            val_analysis = thresh_info["val_analysis"]
            assert "optimal_threshold" in val_analysis
            assert 0.0 < val_analysis["optimal_threshold"] < 1.0
            # Test metrics should include optimized F1
            if result.metrics.get("f1_optimized") is not None:
                assert result.metrics["f1_optimized"] >= 0.0
                assert "optimal_threshold" in result.metrics


class TestXTestMetadata:
    """Tests for X_test stored in TaskResult metadata (for SHAP analysis)."""

    def test_t1_metadata_has_x_test(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t1(model=model, data=synthetic_wq_df, analyte="PFOS")
        meta = result.metadata
        assert "X_test" in meta, "X_test should be stored in metadata for SHAP"
        X_test = meta["X_test"]
        assert isinstance(X_test, pd.DataFrame)
        assert len(X_test) == meta["n_test"]

    def test_t2_metadata_has_x_test(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        result = run_t2(model=model, data=synthetic_wq_df, analyte="PFOS")
        meta = result.metadata
        assert "X_test" in meta, "X_test should be stored in metadata for SHAP"
        X_test = meta["X_test"]
        assert isinstance(X_test, pd.DataFrame)
        assert len(X_test) == meta["n_test"]


class TestT2Task:
    """Tests for T2 — concentration regression."""

    def test_t2_runs(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        result = run_t2(
            model=model,
            data=synthetic_wq_df,
            analyte="PFOS",
        )
        assert isinstance(result, TaskResult)
        assert result.task_name == "T2"

    def test_t2_has_regression_metrics(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        result = run_t2(model=model, data=synthetic_wq_df, analyte="PFOS")
        # Check test or val metrics exist
        assert result.metrics, "Expected non-empty metrics"
        assert "rmse" in result.metrics or "mae" in result.metrics


class TestSpatialMetadata:
    """Tests for spatial metadata stored in TaskResult.metadata."""

    def test_t1_metadata_has_spatial_arrays(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t1(model=model, data=synthetic_wq_df, analyte="PFOS")
        meta = result.metadata
        for key in ("y_true", "y_prob", "latitudes", "longitudes", "split_labels"):
            assert key in meta, f"Missing metadata key: {key}"
            assert isinstance(meta[key], list), f"{key} should be a list"
            assert len(meta[key]) > 0, f"{key} should be non-empty"
        # All arrays same length
        n = len(meta["y_true"])
        assert len(meta["y_prob"]) == n
        assert len(meta["latitudes"]) == n
        assert len(meta["longitudes"]) == n
        assert len(meta["split_labels"]) == n

    def test_t2_metadata_has_spatial_arrays(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestRegressor(config={"n_estimators": 10, "random_state": 42})
        result = run_t2(model=model, data=synthetic_wq_df, analyte="PFOS")
        meta = result.metadata
        for key in ("y_true", "y_prob", "latitudes", "longitudes", "split_labels"):
            assert key in meta, f"Missing metadata key: {key}"
            assert isinstance(meta[key], list), f"{key} should be a list"
            assert len(meta[key]) > 0, f"{key} should be non-empty"
        # All arrays same length
        n = len(meta["y_true"])
        assert len(meta["y_prob"]) == n
        assert len(meta["latitudes"]) == n
        assert len(meta["longitudes"]) == n
        assert len(meta["split_labels"]) == n


class TestT4Task:
    """Tests for T4 — heavy metal prediction."""

    def test_t4_rejects_non_lcr_analyte(self, synthetic_wq_df: pd.DataFrame) -> None:
        """T4 uses action_level target — analytes without defined levels raise."""
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        with pytest.raises(ValueError, match="No action level defined"):
            run_t4(
                model=model,
                data=synthetic_wq_df,
                analyte="PFOS",  # PFOS has no EPA action level
            )

    def test_t4_action_level_target(self) -> None:
        """T4 uses action_level target — verify both classes present with LCR-like data."""
        rng = np.random.RandomState(42)

        state_coords = {
            "CT": (41.6, -72.7),
            "NJ": (40.2, -74.7),
            "PA": (40.3, -76.9),
            "GA": (33.7, -84.4),
            "IL": (40.6, -89.4),
            "TX": (31.0, -97.0),
            "KS": (38.5, -98.8),
            "CO": (39.5, -105.0),
            "CA": (36.8, -119.4),
            "WA": (47.4, -120.5),
        }
        rows = []
        for state, (lat, lon) in state_coords.items():
            for i in range(10):
                pwsid = f"{state}{i + 1:07d}"
                # ~30% exceed lead action level (15 µg/L)
                exceeds = rng.random() < 0.3
                conc = rng.uniform(16.0, 50.0) if exceeds else rng.uniform(0.5, 14.0)
                for _ in range(3):
                    rows.append(
                        {
                            "pwsid": pwsid,
                            "analyte": "lead",
                            "concentration": conc + rng.normal(0, 1),
                            "unit": "ug/L",
                            "censored": False,  # LCR data rarely censored
                            "detection_limit": 0.5,
                            "sample_date": pd.Timestamp("2023-01-15"),
                            "latitude": lat + rng.normal(0, 0.5),
                            "longitude": lon + rng.normal(0, 0.5),
                        }
                    )

        lead_df = pd.DataFrame(rows)
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t4(model=model, data=lead_df, analyte="lead")
        assert isinstance(result, TaskResult)
        assert result.task_name == "T4"
        # Should have non-empty metrics (both classes present)
        assert result.metrics, "Expected non-empty metrics for action_level target"

    def test_t4_nonexistent_analyte(self, synthetic_wq_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t4(
            model=model,
            data=synthetic_wq_df,
            analyte="nonexistent",
        )
        assert result.metadata.get("error") == "empty training set"
