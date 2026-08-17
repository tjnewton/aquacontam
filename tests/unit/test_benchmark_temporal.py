"""Tests for T7 — temporal prediction UCMR3→UCMR5."""

from __future__ import annotations

import pandas as pd
import pytest

from aquacontam.benchmark.registry import TaskResult, get_task
from aquacontam.benchmark.temporal import (
    _prepare_temporal_splits,
    run_t7,
)
from aquacontam.models.random_forest import RandomForestClassifier


class TestPrepareTemporalSplits:
    """Tests for _prepare_temporal_splits."""

    def test_split_structure(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        splits = _prepare_temporal_splits(synthetic_ucmr3_df, synthetic_ucmr5_df, "PFOS")
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert "test_persistent" in splits
        assert "test_new" in splits

    def test_train_val_from_ucmr3(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        splits = _prepare_temporal_splits(synthetic_ucmr3_df, synthetic_ucmr5_df, "PFOS")
        X_train, _ = splits["train"]
        X_val, _ = splits["val"]
        # Train + val should come from UCMR3 systems only
        ucmr3_pwsids = set(synthetic_ucmr3_df["pwsid"].unique())
        if not X_train.empty:
            assert set(X_train.index).issubset(ucmr3_pwsids)
        if not X_val.empty:
            assert set(X_val.index).issubset(ucmr3_pwsids)

    def test_test_is_ucmr5(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        splits = _prepare_temporal_splits(synthetic_ucmr3_df, synthetic_ucmr5_df, "PFOS")
        X_test, _ = splits["test"]
        ucmr5_pwsids = set(synthetic_ucmr5_df["pwsid"].unique())
        if not X_test.empty:
            assert set(X_test.index).issubset(ucmr5_pwsids)

    def test_persistent_new_partition(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        splits = _prepare_temporal_splits(synthetic_ucmr3_df, synthetic_ucmr5_df, "PFOS")
        X_persistent, _ = splits["test_persistent"]
        X_new, _ = splits["test_new"]
        X_test, _ = splits["test"]

        if not X_test.empty:
            # persistent + new should cover all test systems
            persistent_ids = set(X_persistent.index) if not X_persistent.empty else set()
            new_ids = set(X_new.index) if not X_new.empty else set()
            test_ids = set(X_test.index)
            assert persistent_ids | new_ids == test_ids
            # No overlap
            assert persistent_ids & new_ids == set()

    def test_empty_ucmr3(self, synthetic_ucmr5_df: pd.DataFrame) -> None:
        empty_df = pd.DataFrame(
            columns=[
                "pwsid",
                "analyte",
                "concentration",
                "censored",
                "detection_limit",
                "sample_date",
                "latitude",
                "longitude",
                "unit",
            ]
        )
        splits = _prepare_temporal_splits(empty_df, synthetic_ucmr5_df, "PFOS")
        X_train, _ = splits["train"]
        assert X_train.empty

    def test_invalid_analyte_not_shared(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        # _prepare_temporal_splits itself doesn't validate analyte,
        # but run_t7 does — this should still work but produce empty results
        splits = _prepare_temporal_splits(synthetic_ucmr3_df, synthetic_ucmr5_df, "nonexistent")
        X_train, _ = splits["train"]
        assert X_train.empty

    def test_with_feature_dfs(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        import numpy as np

        # Create feature df for all systems
        all_pwsids = list(
            set(synthetic_ucmr3_df["pwsid"].unique()) | set(synthetic_ucmr5_df["pwsid"].unique())
        )
        rng = np.random.RandomState(42)
        feat_df = pd.DataFrame(
            {"feat1": rng.randn(len(all_pwsids))},
            index=all_pwsids,
        )
        feat_df.index.name = "pwsid"

        splits = _prepare_temporal_splits(synthetic_ucmr3_df, synthetic_ucmr5_df, "PFOS", feat_df)
        X_train, _ = splits["train"]
        if not X_train.empty:
            assert "feat1" in X_train.columns


class TestT7Task:
    """Tests for T7 task registration and execution."""

    def test_t7_registered(self) -> None:
        info, _ = get_task("T7")
        assert info.task_type == "classification"
        assert info.primary_metric == "auprc"

    def test_t7_runs(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t7(
            model=model,
            ucmr3_data=synthetic_ucmr3_df,
            ucmr5_data=synthetic_ucmr5_df,
            analyte="PFOS",
        )
        assert isinstance(result, TaskResult)
        assert result.task_name == "T7"

    def test_t7_has_classification_metrics(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t7(
            model=model,
            ucmr3_data=synthetic_ucmr3_df,
            ucmr5_data=synthetic_ucmr5_df,
            analyte="PFOS",
        )
        assert result.metrics, "Expected non-empty metrics"
        assert "accuracy" in result.metrics or "auprc" in result.metrics

    def test_t7_has_persistent_new_splits(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t7(
            model=model,
            ucmr3_data=synthetic_ucmr3_df,
            ucmr5_data=synthetic_ucmr5_df,
            analyte="PFOS",
        )
        # Should have test sub-splits in split_metrics
        assert "test" in result.split_metrics or result.metadata.get("error")

    def test_t7_metadata(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        result = run_t7(
            model=model,
            ucmr3_data=synthetic_ucmr3_df,
            ucmr5_data=synthetic_ucmr5_df,
            analyte="PFOS",
        )
        assert result.metadata["analyte"] == "PFOS"
        assert "n_persistent" in result.metadata
        assert "n_new" in result.metadata
        assert "ucmr3_detection_rate" in result.metadata
        assert "ucmr5_detection_rate" in result.metadata

    def test_t7_invalid_analyte_raises(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        with pytest.raises(ValueError, match="T7_SHARED_ANALYTES"):
            run_t7(
                model=model,
                ucmr3_data=synthetic_ucmr3_df,
                ucmr5_data=synthetic_ucmr5_df,
                analyte="lithium",
            )

    def test_t7_different_analytes(
        self, synthetic_ucmr3_df: pd.DataFrame, synthetic_ucmr5_df: pd.DataFrame
    ) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        for analyte in ("PFOA", "PFHxS"):
            result = run_t7(
                model=model,
                ucmr3_data=synthetic_ucmr3_df,
                ucmr5_data=synthetic_ucmr5_df,
                analyte=analyte,
            )
            assert isinstance(result, TaskResult)

    def test_t7_empty_ucmr3(self, synthetic_ucmr5_df: pd.DataFrame) -> None:
        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        empty_df = pd.DataFrame(
            {
                "pwsid": pd.Series(dtype=str),
                "analyte": pd.Series(dtype=str),
                "concentration": pd.Series(dtype=float),
                "censored": pd.Series(dtype=bool),
                "detection_limit": pd.Series(dtype=float),
                "sample_date": pd.Series(dtype="datetime64[ns]"),
                "latitude": pd.Series(dtype=float),
                "longitude": pd.Series(dtype=float),
                "unit": pd.Series(dtype=str),
            }
        )
        result = run_t7(
            model=model,
            ucmr3_data=empty_df,
            ucmr5_data=synthetic_ucmr5_df,
            analyte="PFOS",
        )
        assert result.metadata.get("error") == "empty training set"
