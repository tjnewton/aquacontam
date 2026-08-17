"""Tests for T5 cross-contaminant transfer task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.registry import TaskResult, get_task, list_tasks
from aquacontam.benchmark.transfer import run_t5
from aquacontam.models.random_forest import RandomForestClassifier


@pytest.fixture()
def source_target_data():
    """Create source (PFOS) and target (PFOA) water quality DataFrames."""
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
    pwsids = []
    lats = []
    lons = []
    for state, (lat, lon) in state_coords.items():
        for i in range(10):
            pwsids.append(f"{state}{i + 1:07d}")
            lats.append(lat + rng.normal(0, 0.5))
            lons.append(lon + rng.normal(0, 0.5))

    # Source data: PFOS
    source_rows = []
    for j, pwsid in enumerate(pwsids):
        for _ in range(3):
            is_censored = rng.random() < 0.7
            dl = rng.uniform(1.0, 5.0)
            conc = 0.0 if is_censored else dl + rng.exponential(10.0)
            source_rows.append(
                {
                    "pwsid": pwsid,
                    "analyte": "PFOS",
                    "concentration": conc,
                    "unit": "ug/L",
                    "censored": is_censored,
                    "detection_limit": dl,
                    "sample_date": pd.Timestamp("2023-01-15"),
                    "latitude": lats[j],
                    "longitude": lons[j],
                }
            )
    source_df = pd.DataFrame(source_rows)

    # Target data: PFOA
    target_rows = []
    for j, pwsid in enumerate(pwsids):
        for _ in range(3):
            is_censored = rng.random() < 0.5
            dl = rng.uniform(1.0, 5.0)
            conc = 0.0 if is_censored else dl + rng.exponential(8.0)
            target_rows.append(
                {
                    "pwsid": pwsid,
                    "analyte": "PFOA",
                    "concentration": conc,
                    "unit": "ug/L",
                    "censored": is_censored,
                    "detection_limit": dl,
                    "sample_date": pd.Timestamp("2023-01-15"),
                    "latitude": lats[j],
                    "longitude": lons[j],
                }
            )
    target_df = pd.DataFrame(target_rows)

    return source_df, target_df


class TestT5Registration:
    """Tests for T5 task registration."""

    def test_t5_registered(self) -> None:
        info, _ = get_task("T5")
        assert info.task_type == "classification"
        assert info.primary_metric == "auprc"

    def test_t5_in_list_tasks(self) -> None:
        tasks = list_tasks()
        names = [t.name for t in tasks]
        assert "T5" in names

    def test_t5_description(self) -> None:
        info, _ = get_task("T5")
        assert "transfer" in info.description.lower()


class TestT5ZeroShot:
    """Tests for zero-shot transfer evaluation."""

    def test_runs_successfully(self, source_target_data) -> None:
        source_df, target_df = source_target_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=source_df,
            target_data=target_df,
            source_analyte="PFOS",
            target_analyte="PFOA",
        )
        assert isinstance(result, TaskResult)
        assert result.task_name == "T5"

    def test_has_metrics(self, source_target_data) -> None:
        source_df, target_df = source_target_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=source_df,
            target_data=target_df,
            source_analyte="PFOS",
            target_analyte="PFOA",
        )
        # Should have some metrics (from test or val split)
        assert result.metrics, "Expected non-empty metrics"
        assert any(k in result.metrics for k in ["accuracy", "auroc", "auprc"])

    def test_metadata_has_analytes(self, source_target_data) -> None:
        source_df, target_df = source_target_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=source_df,
            target_data=target_df,
            source_analyte="PFOS",
            target_analyte="PFOA",
        )
        assert result.metadata["source_analyte"] == "PFOS"
        assert result.metadata["target_analyte"] == "PFOA"

    def test_has_split_metrics(self, source_target_data) -> None:
        source_df, target_df = source_target_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=source_df,
            target_data=target_df,
            source_analyte="PFOS",
            target_analyte="PFOA",
        )
        assert "train" in result.split_metrics

    def test_default_analytes(self, source_target_data) -> None:
        """Test with default analyte pair (uses constants)."""
        source_df, target_df = source_target_data
        # Change analyte names in data to match defaults
        source_df = source_df.copy()
        source_df["analyte"] = "lead"
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=source_df,
            target_data=target_df,
            source_analyte="lead",
            target_analyte="PFOA",
        )
        assert result.metadata["source_analyte"] == "lead"


class TestT5EmptyData:
    """Tests for T5 with empty or missing data."""

    def test_empty_source_returns_error(self) -> None:
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
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=empty_df,
            target_data=empty_df,
            source_analyte="PFOS",
            target_analyte="PFOA",
        )
        assert "error" in result.metadata

    def test_single_class_source(self) -> None:
        """Source data with all detections (no censored) → single-class guard."""
        rng = np.random.RandomState(99)

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
                for _ in range(3):
                    rows.append(
                        {
                            "pwsid": pwsid,
                            "analyte": "PFOS",
                            "concentration": rng.exponential(10.0) + 5.0,
                            "unit": "ug/L",
                            "censored": False,  # All detected → single class
                            "detection_limit": 2.0,
                            "sample_date": pd.Timestamp("2023-01-15"),
                            "latitude": lat + rng.normal(0, 0.5),
                            "longitude": lon + rng.normal(0, 0.5),
                        }
                    )

        all_detected_df = pd.DataFrame(rows)
        empty_target = pd.DataFrame(
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
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=all_detected_df,
            target_data=empty_target,
            source_analyte="PFOS",
            target_analyte="PFOA",
        )
        assert result.metrics == {}
        assert result.metadata["error"] == "single-class source training set"

    def test_nonexistent_analyte(self, source_target_data) -> None:
        source_df, target_df = source_target_data
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t5(
            model=model,
            source_data=source_df,
            target_data=target_df,
            source_analyte="nonexistent",
            target_analyte="PFOA",
        )
        assert "error" in result.metadata
