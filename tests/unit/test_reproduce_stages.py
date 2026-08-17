"""Tests for reproducibility pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scripts.reproduce import (
    _export_feature_importances,
    _export_results,
    _export_zenodo_dataset,
    _generate_paper_assets,
    _preprocess_data,
    _run_equity_analysis,
    main,
)


@pytest.fixture()
def mock_wq_data() -> pd.DataFrame:
    """Minimal water quality DataFrame for reproduce tests."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(20):
        state = ["CT", "NJ", "PA", "GA", "TX"][i % 5]
        rows.append(
            {
                "pwsid": f"{state}{i + 1:07d}",
                "analyte": "PFOS",
                "concentration": rng.exponential(10.0),
                "unit": "ug/L",
                "censored": rng.random() < 0.5,
                "detection_limit": 2.0,
                "sample_date": pd.Timestamp("2023-01-01"),
                "latitude": 40.0 + rng.normal(0, 2),
                "longitude": -80.0 + rng.normal(0, 2),
            }
        )
    return pd.DataFrame(rows)


class TestPreprocessData:
    """Tests for _preprocess_data."""

    def test_uses_cached_if_exists(self, tmp_path: Path, mock_wq_data: pd.DataFrame) -> None:
        interim = tmp_path / "interim"
        interim.mkdir()
        cached = interim / "merged_wq.parquet"
        mock_wq_data.to_parquet(cached, index=False)

        result = _preprocess_data(tmp_path, {})
        assert len(result) == len(mock_wq_data)

    def test_returns_empty_if_no_sources(self, tmp_path: Path) -> None:
        result = _preprocess_data(tmp_path, {})
        assert len(result) == 0


class TestExportResults:
    """Tests for _export_results."""

    def test_creates_results_json(self, tmp_path: Path) -> None:
        results = [
            {
                "task": "T1",
                "model": "xgboost_classifier",
                "metrics": {"auroc": 0.85, "auprc": 0.75},
                "split_metrics": {},
                "metadata": {"analyte": "PFOS"},
            }
        ]
        _export_results(results, tmp_path)
        assert (tmp_path / "results.json").exists()
        data = json.loads((tmp_path / "results.json").read_text())
        assert len(data) == 1

    def test_creates_submission_jsons(self, tmp_path: Path) -> None:
        results = [
            {
                "task": "T1",
                "model": "xgboost",
                "metrics": {"auroc": 0.85},
                "split_metrics": {},
                "metadata": {},
            },
            {
                "task": "T2",
                "model": "xgboost",
                "metrics": {"rmse": 1.5},
                "split_metrics": {},
                "metadata": {},
            },
        ]
        _export_results(results, tmp_path)
        sub_dir = tmp_path / "submissions"
        assert sub_dir.exists()
        files = list(sub_dir.glob("*.json"))
        assert len(files) == 1  # One model
        sub = json.loads(files[0].read_text())
        assert sub["model_name"] == "xgboost"
        assert len(sub["results"]) == 2

    def test_multiple_models(self, tmp_path: Path) -> None:
        results = [
            {"task": "T1", "model": "xgboost", "metrics": {}, "split_metrics": {}, "metadata": {}},
            {"task": "T1", "model": "rf", "metrics": {}, "split_metrics": {}, "metadata": {}},
        ]
        _export_results(results, tmp_path)
        files = list((tmp_path / "submissions").glob("*.json"))
        assert len(files) == 2

    def test_idempotent(self, tmp_path: Path) -> None:
        results = [
            {"task": "T1", "model": "m1", "metrics": {}, "split_metrics": {}, "metadata": {}},
        ]
        _export_results(results, tmp_path)
        _export_results(results, tmp_path)
        assert (tmp_path / "results.json").exists()


class TestCLI:
    """Tests for reproduce.py CLI."""

    def test_no_stage_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "No stage selected" in result.output

    def test_t6_arsenic_is_a_recognized_stage(self) -> None:
        """`--t6-arsenic` alone must reach the T6 dispatch, not early-exit."""
        runner = CliRunner()
        with patch("aquacontam.pipeline.t6_arsenic.run_t6_experiment") as mock_t6:
            result = runner.invoke(main, ["--t6-arsenic"])
        assert "No stage selected" not in result.output
        assert mock_t6.called

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--skip-large" in result.output

    def test_download_only_flag(self) -> None:
        runner = CliRunner()
        _empty = ({}, {"new": 0, "cached": 0, "unavailable": 0})
        with patch("scripts.reproduce._download_data", return_value=_empty) as mock_dl:
            result = runner.invoke(main, ["--download-only", "--data-dir", "/tmp/test_aq"])
            assert result.exit_code == 0
            mock_dl.assert_called_once()

    def test_task_filter_parsing(self) -> None:
        runner = CliRunner()
        _empty = ({}, {"new": 0, "cached": 0, "unavailable": 0})
        with patch("scripts.reproduce._download_data", return_value=_empty):
            result = runner.invoke(
                main,
                ["--download-only", "--tasks", "T1,T3", "--data-dir", "/tmp/test_aq"],
            )
            assert result.exit_code == 0


class TestGetModelInstances:
    """Tests for _get_model_instances."""

    def test_returns_classifiers(self) -> None:
        from scripts.reproduce import _get_model_instances

        models = _get_model_instances(None, "classification", 42)
        assert len(models) >= 2  # At least xgboost + RF

    def test_returns_regressors(self) -> None:
        from scripts.reproduce import _get_model_instances

        models = _get_model_instances(None, "regression", 42)
        assert len(models) >= 2

    def test_filter_by_name(self) -> None:
        from scripts.reproduce import _get_model_instances

        models = _get_model_instances(["xgboost"], "classification", 42)
        assert len(models) == 1

    def test_unknown_model_ignored(self) -> None:
        from scripts.reproduce import _get_model_instances

        models = _get_model_instances(["nonexistent"], "classification", 42)
        assert len(models) == 0


class TestExportFeatureImportances:
    """Tests for _export_feature_importances."""

    def test_writes_importance_json(self, tmp_path: Path) -> None:
        model = MagicMock()
        model.name = "xgboost_classifier"
        model.feature_importances.return_value = pd.Series(
            {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        )
        fitted_models = {"T1": (model, "classification", 0.5)}

        result = _export_feature_importances(fitted_models, tmp_path)

        assert result is not None
        out_path = tmp_path / "feature_importance.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "feat_a" in data
        assert data["feat_a"] == 0.5

    def test_skips_if_no_models(self, tmp_path: Path) -> None:
        result = _export_feature_importances({}, tmp_path)
        assert result is None
        assert not (tmp_path / "feature_importance.json").exists()


class TestRunEquityAnalysis:
    """Tests for _run_equity_analysis."""

    def test_skips_without_demographics_column(self, tmp_path: Path) -> None:
        wq_df = pd.DataFrame({"pwsid": ["A"], "analyte": ["PFOS"]})
        feature_dfs = [pd.DataFrame({"some_feature": [1.0]}, index=["A"])]
        fitted_models = {"T1": (MagicMock(), "classification", 0.5)}

        # Should return without error — no demographics available
        _run_equity_analysis(wq_df, feature_dfs, fitted_models, tmp_path)
        assert not (tmp_path / "equity_analysis.json").exists()

    def test_runs_full_fdr_pipeline_with_mock_demographics(self, tmp_path: Path) -> None:
        """Exercise the full equity → permutation → FDR pipeline.

        Uses synthetic demographics and a lightweight trained RF so we test
        the code path that requires EJScreen data without the 6 GB download.
        """
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            assemble_feature_matrix,
            drop_leakage_columns,
        )
        from aquacontam.models.random_forest import RandomForestClassifier

        rng = np.random.RandomState(99)
        n_systems = 60
        pwsids = [f"XX{i:07d}" for i in range(n_systems)]

        # Build water-quality data — ~half detected, ~half censored
        rows = []
        for i, pid in enumerate(pwsids):
            detected = i % 2 == 0
            rows.append(
                {
                    "pwsid": pid,
                    "analyte": "PFOS",
                    "concentration": rng.exponential(10.0) if detected else 0.0,
                    "censored": not detected,
                    "detection_limit": 2.0,
                }
            )
        wq_df = pd.DataFrame(rows)

        # Demographics feature DF indexed by pwsid
        demo_df = pd.DataFrame(
            {
                "pct_people_of_color": rng.uniform(0, 1, n_systems),
                "pct_low_income": rng.uniform(0, 1, n_systems),
                "pct_limited_english": rng.uniform(0, 1, n_systems),
                "pct_less_hs_education": rng.uniform(0, 1, n_systems),
            },
            index=pwsids,
        )
        demo_df.index.name = "pwsid"

        # A numeric feature so the model has something to split on
        extra_df = pd.DataFrame(
            {"some_feature": rng.uniform(0, 10, n_systems)},
            index=pwsids,
        )
        extra_df.index.name = "pwsid"

        feature_dfs = [demo_df, extra_df]

        # Train a lightweight RF on the same feature matrix the equity
        # function will assemble, so predict/predict_proba return real arrays.
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _ = assemble_feature_matrix(sys_targets, *feature_dfs)

        model = RandomForestClassifier(config={"n_estimators": 10, "random_state": 42})
        model.fit(X, y)

        fitted_models = {"T1": (model, "classification", 0.12)}

        _run_equity_analysis(wq_df, feature_dfs, fitted_models, tmp_path)

        out_path = tmp_path / "equity_analysis.json"
        assert out_path.exists(), "equity_analysis.json was not created"

        data = json.loads(out_path.read_text())
        assert len(data) == 4, f"Expected 4 demographic groups, got {len(data)}"

        for entry in data:
            assert "p_value_fdr" in entry, f"Missing p_value_fdr in {entry['group']}"
            assert "reject_fdr" in entry, f"Missing reject_fdr in {entry['group']}"
            assert isinstance(entry["p_value_fdr"], float)
            assert isinstance(entry["reject_fdr"], bool)

    def test_skips_without_t1_model(self, tmp_path: Path) -> None:
        wq_df = pd.DataFrame({"pwsid": ["A"], "analyte": ["PFOS"]})
        feature_dfs = [
            pd.DataFrame({"pct_people_of_color": [0.5]}, index=["A"]),
        ]
        # No T1 model
        fitted_models = {"T2": (MagicMock(), "regression", 0.5)}

        _run_equity_analysis(wq_df, feature_dfs, fitted_models, tmp_path)
        assert not (tmp_path / "equity_analysis.json").exists()


class TestGeneratePaperAssets:
    """Tests for _generate_paper_assets."""

    def test_skips_if_matplotlib_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.dict("sys.modules", {"paper.generate_figures": None}),
            patch(
                "builtins.__import__",
                side_effect=_selective_import_error("paper.generate_figures"),
            ),
        ):
            _generate_paper_assets(tmp_path, tmp_path)
        assert any("matplotlib" in r.message or "figure" in r.message for r in caplog.records)

    def test_invokes_generators(self, tmp_path: Path) -> None:
        mock_fig_main = MagicMock()
        mock_tbl_main = MagicMock()
        mock_runner = MagicMock()
        mock_runner.invoke.return_value = MagicMock(exit_code=0, output="ok")

        fig_mod = MagicMock(main=mock_fig_main)
        tbl_mod = MagicMock(main=mock_tbl_main)

        with (
            patch.dict(
                "sys.modules",
                {"paper.generate_figures": fig_mod, "paper.generate_tables": tbl_mod},
            ),
            patch("click.testing.CliRunner", return_value=mock_runner),
        ):
            _generate_paper_assets(tmp_path, tmp_path)

        assert mock_runner.invoke.call_count == 2


class TestExportZenodoDataset:
    """Tests for _export_zenodo_dataset."""

    def test_calls_export_dataset(self, tmp_path: Path) -> None:
        wq_df = pd.DataFrame({"pwsid": ["A"], "analyte": ["PFOS"]})
        mock_export = MagicMock()

        with patch("aquacontam.export.dataset.export_dataset", mock_export):
            _export_zenodo_dataset(wq_df, [], tmp_path, tmp_path)

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs[1]["output_dir"] == tmp_path / "zenodo-dataset"

    def test_writes_metadata_files(self, tmp_path: Path) -> None:
        wq_df = pd.DataFrame({"pwsid": ["A", "B"], "analyte": ["PFOS", "PFOA"]})
        mock_export = MagicMock()
        mock_write_meta = MagicMock(return_value=[])

        with (
            patch("aquacontam.export.dataset.export_dataset", mock_export),
            patch("aquacontam.export.metadata.write_metadata_files", mock_write_meta),
        ):
            _export_zenodo_dataset(wq_df, [], tmp_path, tmp_path)

        mock_write_meta.assert_called_once()
        call_kwargs = mock_write_meta.call_args
        assert call_kwargs[0][0] == tmp_path / "zenodo-dataset"
        assert call_kwargs[1]["n_systems"] == 2
        assert call_kwargs[1]["n_samples"] == 2

    def test_skips_if_module_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        wq_df = pd.DataFrame({"pwsid": ["A"], "analyte": ["PFOS"]})

        with (
            patch.dict("sys.modules", {"aquacontam.export.dataset": None}),
            patch(
                "builtins.__import__",
                side_effect=_selective_import_error("aquacontam.export.dataset"),
            ),
        ):
            _export_zenodo_dataset(wq_df, [], tmp_path, tmp_path)

        assert not (tmp_path / "zenodo-dataset").exists()


def _selective_import_error(blocked_module: str):
    """Return an import function that blocks *only* ``blocked_module``."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _import(name: str, *args: Any, **kwargs: Any):
        if name == blocked_module:
            raise ImportError(f"mocked: {blocked_module}")
        return real_import(name, *args, **kwargs)

    return _import
