"""Tests for LORO cross-validation runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.loro import LOROResult, run_loro_classification
from aquacontam.models.random_forest import RandomForestClassifier


@pytest.fixture()
def loro_wq_df() -> pd.DataFrame:
    """Create a water quality DataFrame spanning multiple EPA regions.

    Uses proper 2-letter state code PWSID prefixes so assign_epa_region()
    can derive EPA regions.
    """
    rng = np.random.RandomState(42)

    # State codes mapped to EPA regions (one per region)
    state_region = [
        ("CT", 1, 42.0, -72.0),  # New England
        ("NJ", 2, 40.7, -74.0),  # NY/NJ
        ("PA", 3, 39.0, -76.5),  # Mid-Atlantic
        ("GA", 4, 33.7, -84.4),  # Southeast
        ("IL", 5, 41.9, -87.6),  # Great Lakes
        ("TX", 6, 30.3, -97.7),  # South Central
        ("KS", 7, 39.1, -94.6),  # Central
        ("CO", 8, 39.7, -105.0),  # Mountain
        ("CA", 9, 37.8, -122.4),  # Pacific Southwest
        ("WA", 10, 47.6, -122.3),  # Pacific Northwest
    ]

    rows = []
    for state, _region, lat, lon in state_region:
        for i in range(8):
            pwsid = f"{state}{i + 1:07d}"
            detected = rng.random() < 0.3
            for _ in range(3):
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": "PFOS",
                        "concentration": rng.uniform(5.0, 100.0) if detected else 0.0,
                        "unit": "ng/L",
                        "censored": not detected,
                        "detection_limit": 2.0,
                        "sample_date": pd.Timestamp("2023-01-15"),
                        "latitude": lat + rng.normal(0, 0.3),
                        "longitude": lon + rng.normal(0, 0.3),
                    }
                )
    return pd.DataFrame(rows)


class TestLOROResult:
    """Tests for LOROResult dataclass."""

    def test_default_values(self) -> None:
        result = LOROResult(task_name="T1", model_name="test")
        assert result.n_folds == 0
        assert result.mean_metrics == {}
        assert result.per_fold_metrics == []

    def test_summary(self) -> None:
        result = LOROResult(
            task_name="T1",
            model_name="rf",
            mean_metrics={"auroc": 0.75},
            std_metrics={"auroc": 0.05},
            fold_regions=[1, 2, 3],
            n_folds=3,
        )
        s = result.summary()
        assert s["task"] == "T1"
        assert s["model"] == "rf"
        assert s["n_folds"] == 3


class TestRunLOROClassification:
    """Tests for run_loro_classification."""

    def test_runs_and_returns_result(self, loro_wq_df: pd.DataFrame) -> None:
        result = run_loro_classification(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            loro_wq_df,
            "PFOS",
            task_name="T1",
        )
        assert isinstance(result, LOROResult)
        assert result.task_name == "T1"
        assert result.model_name == "random_forest_classifier"

    def test_produces_multiple_folds(self, loro_wq_df: pd.DataFrame) -> None:
        result = run_loro_classification(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            loro_wq_df,
            "PFOS",
        )
        # Should produce at least some folds (not all 10 may succeed)
        assert result.n_folds >= 1
        assert len(result.per_fold_metrics) == result.n_folds
        assert len(result.fold_regions) == result.n_folds

    def test_mean_metrics_computed(self, loro_wq_df: pd.DataFrame) -> None:
        result = run_loro_classification(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            loro_wq_df,
            "PFOS",
        )
        if result.n_folds > 0:
            assert "auroc" in result.mean_metrics or "accuracy" in result.mean_metrics
            assert "auroc" in result.std_metrics or "accuracy" in result.std_metrics

    def test_empty_data(self) -> None:
        empty_df = pd.DataFrame(
            columns=[
                "pwsid",
                "analyte",
                "concentration",
                "unit",
                "censored",
                "detection_limit",
                "sample_date",
                "latitude",
                "longitude",
            ]
        )
        result = run_loro_classification(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            empty_df,
            "PFOS",
        )
        assert result.n_folds == 0

    def test_nonexistent_analyte(self, loro_wq_df: pd.DataFrame) -> None:
        result = run_loro_classification(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            loro_wq_df,
            "nonexistent",
        )
        assert result.n_folds == 0

    def test_no_target_leakage(self, loro_wq_df: pd.DataFrame) -> None:
        """Regression test for the LORO target-leakage bug (referee report M6).

        In this fixture ``detected`` is assigned at random and is independent of
        every environmental feature, so a leakage-free model cannot beat chance.
        Before the fix, ``run_loro_classification`` passed the aggregation columns
        (``any_detected`` / ``detection_rate`` / ``max_concentration``) straight
        through to ``assemble_feature_matrix`` as features, so the model scored
        ~1.0 AUROC by construction. Assert the mean out-of-region AUROC is well
        below 1.0 — i.e. ``drop_leakage_columns`` is applied.
        """
        result = run_loro_classification(
            RandomForestClassifier,
            {"n_estimators": 50, "random_state": 42},
            loro_wq_df,
            "PFOS",
            task_name="T1",
        )
        assert result.n_folds >= 2, "expected multiple LORO folds to evaluate"
        assert "auroc" in result.mean_metrics, "AUROC should be computed across folds"
        assert result.mean_metrics["auroc"] < 0.95, (
            f"LORO mean AUROC {result.mean_metrics['auroc']:.3f} is implausibly high for a "
            "random target — target-leakage columns are leaking into the feature matrix"
        )
