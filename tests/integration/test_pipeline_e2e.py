"""End-to-end integration tests for the AquaContam pipeline.

These tests verify the full data → features → train → predict wiring.
Marked as slow — excluded from ``pytest tests/unit/`` runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquacontam.features.assembly import (
    aggregate_to_system_level,
    assemble_feature_matrix,
    drop_leakage_columns,
)
from aquacontam.preprocessing.splits import assign_epa_region, geographic_split


@pytest.mark.slow
class TestDataToPredictionPipeline:
    """Test the full data → features → train → predict pipeline."""

    def test_pipeline_produces_valid_predictions(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """End-to-end: aggregate → split → assemble → train → predict."""
        pytest.importorskip("xgboost")
        from aquacontam.models.xgboost import XGBoostClassifier

        # 1. Assign EPA regions and aggregate
        df = assign_epa_region(synthetic_wq_df)
        system_agg = aggregate_to_system_level(df, "PFOS")
        assert len(system_agg) > 0

        # 2. Merge EPA region and split
        regions = df.groupby("pwsid", observed=True)["epa_region"].first()
        system_agg = system_agg.join(regions)
        system_reset = system_agg.reset_index()
        train_df, _val_df, test_df = geographic_split(system_reset)

        assert len(train_df) > 0
        assert len(test_df) > 0

        # 3. Assemble features for train split
        train_indexed = train_df.set_index("pwsid")
        if "epa_region" in train_indexed.columns:
            train_indexed = train_indexed.drop(columns=["epa_region"])
        train_indexed = drop_leakage_columns(train_indexed)

        X_train, y_train, impute_stats = assemble_feature_matrix(
            train_indexed, *synthetic_feature_dfs
        )
        assert X_train.shape[0] == len(y_train)
        assert X_train.shape[1] > 0

        # 4. Assemble features for test split
        test_indexed = test_df.set_index("pwsid")
        if "epa_region" in test_indexed.columns:
            test_indexed = test_indexed.drop(columns=["epa_region"])
        test_indexed = drop_leakage_columns(test_indexed)

        X_test, y_test, _ = assemble_feature_matrix(
            test_indexed, *synthetic_feature_dfs, impute_stats=impute_stats
        )

        # 5. Train and predict
        model = XGBoostClassifier(config={"n_estimators": 10, "max_depth": 3, "random_state": 42})
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)

        # 6. Verify predictions
        assert len(preds) == len(y_test)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_feature_assembly_preserves_systems(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """Feature assembly keeps all systems from targets."""
        df = assign_epa_region(synthetic_wq_df)
        system_agg = aggregate_to_system_level(df, "PFOA")
        system_agg = drop_leakage_columns(system_agg)

        X, y, _ = assemble_feature_matrix(system_agg, *synthetic_feature_dfs)
        assert len(X) == len(system_agg)
        assert len(y) == len(system_agg)


@pytest.mark.slow
class TestBenchmarkTaskRegistration:
    """Test benchmark task discovery and registration."""

    def test_tasks_registered(self) -> None:
        import aquacontam.benchmark.tasks  # noqa: F401
        from aquacontam.benchmark.registry import list_tasks

        tasks = list_tasks()
        assert len(tasks) >= 4
        names = [t.name for t in tasks]
        assert "T1" in names
        assert "T2" in names

    def test_task_info_has_required_fields(self) -> None:
        import aquacontam.benchmark.tasks  # noqa: F401
        from aquacontam.benchmark.registry import list_tasks

        for task in list_tasks():
            assert task.name
            assert task.description
            assert task.primary_metric
            assert task.task_type in (
                "classification",
                "regression",
                "multilabel_classification",
            )


@pytest.mark.slow
class TestAdditionalPipelines:
    """Additional pipeline integration tests."""

    def test_catboost_pipeline(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """End-to-end CatBoost: aggregate → split → assemble → train → predict."""
        pytest.importorskip("catboost")
        from aquacontam.models.catboost import CatBoostClassifier

        df = assign_epa_region(synthetic_wq_df)
        system_agg = aggregate_to_system_level(df, "PFOS")
        regions = df.groupby("pwsid", observed=True)["epa_region"].first()
        system_agg = system_agg.join(regions)
        system_reset = system_agg.reset_index()
        train_df, _val_df, test_df = geographic_split(system_reset)

        train_indexed = train_df.set_index("pwsid")
        if "epa_region" in train_indexed.columns:
            train_indexed = train_indexed.drop(columns=["epa_region"])
        train_indexed = drop_leakage_columns(train_indexed)
        X_train, y_train, impute_stats = assemble_feature_matrix(
            train_indexed, *synthetic_feature_dfs
        )

        test_indexed = test_df.set_index("pwsid")
        if "epa_region" in test_indexed.columns:
            test_indexed = test_indexed.drop(columns=["epa_region"])
        test_indexed = drop_leakage_columns(test_indexed)
        X_test, y_test, _ = assemble_feature_matrix(
            test_indexed, *synthetic_feature_dfs, impute_stats=impute_stats
        )

        model = CatBoostClassifier(
            config={"iterations": 10, "depth": 3, "random_seed": 42, "verbose": 0}
        )
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)

        assert len(preds) == len(y_test)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_equity_analysis_pipeline(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs_with_demographics: list[pd.DataFrame],
    ) -> None:
        """Train XGBoost, get predictions, run equity analysis."""
        pytest.importorskip("xgboost")
        from aquacontam.analysis.equity import analyze_equity
        from aquacontam.models.xgboost import XGBoostClassifier

        feature_dfs = synthetic_feature_dfs_with_demographics
        demo_df = feature_dfs[-1]
        model_feature_dfs = feature_dfs[:-1]

        df = assign_epa_region(synthetic_wq_df)
        system_agg = aggregate_to_system_level(df, "PFOS")
        system_agg = drop_leakage_columns(system_agg)
        X, y, _ = assemble_feature_matrix(system_agg, *model_feature_dfs)

        model = XGBoostClassifier(config={"n_estimators": 10, "max_depth": 3, "random_state": 42})
        model.fit(X, y)
        y_prob_2d = model.predict_proba(X)
        y_prob = y_prob_2d[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        demographics = demo_df.loc[X.index]
        report = analyze_equity(
            y, y_pred, y_prob, demographics, "pct_people_of_color", threshold=0.5
        )

        assert isinstance(report.burden_ratio, float)
        assert report.n_high > 0
        assert report.n_low > 0

    def test_calibration_ece_computed(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """Train XGBoost, compute ECE from predicted probabilities."""
        pytest.importorskip("xgboost")
        from aquacontam.benchmark.metrics import expected_calibration_error
        from aquacontam.models.xgboost import XGBoostClassifier

        df = assign_epa_region(synthetic_wq_df)
        system_agg = aggregate_to_system_level(df, "PFOS")
        system_agg = drop_leakage_columns(system_agg)
        X, y, _ = assemble_feature_matrix(system_agg, *synthetic_feature_dfs)

        model = XGBoostClassifier(config={"n_estimators": 10, "max_depth": 3, "random_state": 42})
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]

        ece_result = expected_calibration_error(y.values, y_prob)
        assert isinstance(ece_result, dict)
        assert "ece" in ece_result
        assert 0.0 <= ece_result["ece"] <= 1.0

    def test_loro_fold_structure(
        self,
        synthetic_wq_df: pd.DataFrame,
    ) -> None:
        """LORO folds have correct count, unique test regions, no index overlap."""
        from aquacontam.preprocessing.splits import assign_epa_region, leave_one_region_out

        df = assign_epa_region(synthetic_wq_df)
        system_df = df.groupby("pwsid", observed=True).first().reset_index()

        folds = leave_one_region_out(system_df)

        # Should have one fold per present region
        assert len(folds) >= 2

        test_regions = [fold[3] for fold in folds]
        assert len(test_regions) == len(set(test_regions)), "Test regions not unique across folds"

        for train, _val, test, _region in folds:
            train_idx = set(train.index.tolist())
            test_idx = set(test.index.tolist())
            assert len(train_idx & test_idx) == 0, "Train/test index overlap"

    def test_reproduce_verify_missing_checksums(self, tmp_path: Path) -> None:
        """--verify exits with code 1 when checksums file is missing."""
        pytest.importorskip("click")
        from click.testing import CliRunner
        from scripts.reproduce import main

        runner = CliRunner()
        result = runner.invoke(main, ["--verify", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_reproduce_verify_valid_checksums(self, tmp_path: Path) -> None:
        """--verify exits with code 0 when checksums match."""
        import hashlib

        pytest.importorskip("click")
        from click.testing import CliRunner
        from scripts.reproduce import main

        dummy_file = tmp_path / "dummy_result.json"
        dummy_file.write_text('{"test": true}')
        checksum = hashlib.sha256(dummy_file.read_bytes()).hexdigest()

        checksums_file = tmp_path / "checksums.sha256"
        checksums_file.write_text(f"{checksum}  dummy_result.json\n")

        runner = CliRunner()
        result = runner.invoke(main, ["--verify", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
