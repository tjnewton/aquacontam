"""Tests for aquacontam.pipeline core modules.

Covers download, preprocess, features, assembly, models, training, and export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# download.py
# ---------------------------------------------------------------------------


class TestDownloadData:
    """Tests for aquacontam.pipeline.download.download_data."""

    @pytest.mark.slow
    def test_cached_files_skip_download(self, tmp_path: Path) -> None:
        """Pre-existing parquets in processed_dir are returned without calling source.run()."""
        from aquacontam.pipeline.download import download_data

        processed = tmp_path / "processed"
        processed.mkdir(parents=True)
        interim = tmp_path / "interim"
        interim.mkdir(parents=True)

        # Create cached parquet files for two DataSource loaders
        for name in ("ucmr5", "ucmr3"):
            df = pd.DataFrame({"pwsid": ["A"], "val": [1]})
            df.to_parquet(processed / f"{name}.parquet")

        # Also create cached FRS, TRI, DoD, NJ private wells to avoid real downloads
        for name in ("epa_frs", "tri_pfas", "dod_pfas", "nj_private_wells"):
            df = pd.DataFrame({"id": [1]})
            df.to_parquet(interim / f"{name}.parquet")

        # Also create ejscreen cached
        pd.DataFrame({"id": [1]}).to_parquet(interim / "ejscreen.parquet")

        # Mock config to avoid needing real YAML
        mock_config = {}

        # Mock out the function-based downloaders (ZCTA, NLCD, aquifers) to avoid
        # network I/O, plus the data config loader
        with (
            patch(
                "aquacontam._config.load_data_config",
                return_value=mock_config,
            ),
            patch(
                "aquacontam.geo.geocoding.download_zcta_gazetteer",
                return_value=tmp_path / "raw" / "zcta.csv",
            ),
            patch(
                "aquacontam.features.land_use.download_nlcd",
                return_value=tmp_path / "raw" / "nlcd.tif",
            ),
            patch(
                "aquacontam.features.hydrogeology.download_principal_aquifers",
                return_value=tmp_path / "raw" / "aquifers.shp",
            ),
        ):
            downloaded, stats = download_data(tmp_path, skip_large=True)

        # ucmr5/ucmr3 should come from cache, no new downloads for them
        assert "ucmr5" in downloaded
        assert "ucmr3" in downloaded
        assert stats["cached"] >= 2

    def test_unavailable_source_counted(self, tmp_path: Path) -> None:
        """A source marked unavailable in config increments stats['unavailable']."""
        from aquacontam.pipeline.download import download_data

        processed = tmp_path / "processed"
        interim = tmp_path / "interim"
        for d in (tmp_path / "raw", interim, processed):
            d.mkdir(parents=True, exist_ok=True)

        # Mark ucmr5 as unavailable in config; cache everything else so that
        # no real downloads occur and only ucmr5 exercises the unavailable path.
        mock_config = {
            "ucmr5": {"unavailable": True, "unavailable_reason": "test reason"},
        }

        # Create cached parquets for every other DataSource loader
        other_sources = [
            "ucmr3",
            "sdwis",
            "mi_mpart",
            "ca_geotracker",
            "nj_dep",
            "nc_deq",
            "wqp",
            "mo_dnr",
            "tx_tceq",
            "oh_epa",
            "wa_doh",
            "mn_mdh",
        ]
        for name in other_sources:
            pd.DataFrame({"id": [1]}).to_parquet(processed / f"{name}.parquet")

        # Cache function-based sources in interim
        for name in ("epa_frs", "ejscreen", "nj_private_wells", "tri_pfas", "dod_pfas"):
            pd.DataFrame({"id": [1]}).to_parquet(interim / f"{name}.parquet")

        with (
            patch("aquacontam._config.load_data_config", return_value=mock_config),
            patch(
                "aquacontam.geo.geocoding.download_zcta_gazetteer",
                return_value=tmp_path / "raw" / "zcta.csv",
            ),
            patch(
                "aquacontam.features.hydrogeology.download_principal_aquifers",
                return_value=tmp_path / "raw" / "aquifers.shp",
            ),
        ):
            downloaded, stats = download_data(tmp_path, skip_large=True)

        # ucmr5 must be counted as unavailable
        assert stats["unavailable"] >= 1
        # ucmr5 must NOT be in downloaded
        assert "ucmr5" not in downloaded

    def test_strict_reraises(self, tmp_path: Path) -> None:
        """With strict=True, a failing source raises instead of being swallowed."""
        import importlib as _importlib

        from aquacontam.pipeline._strict import set_strict
        from aquacontam.pipeline.download import download_data

        for d in ("raw", "interim", "processed"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        mock_config = {}

        _real_import_module = _importlib.import_module

        def _fail_ucmr5(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "aquacontam.data.ucmr5":
                raise RuntimeError("deliberate failure")
            return _real_import_module(name, *args, **kwargs)

        set_strict(True)
        try:
            with (
                patch("aquacontam._config.load_data_config", return_value=mock_config),
                patch("importlib.import_module", side_effect=_fail_ucmr5),
                pytest.raises(RuntimeError, match="deliberate failure"),
            ):
                download_data(tmp_path, skip_large=True)
        finally:
            set_strict(False)


# ---------------------------------------------------------------------------
# preprocess.py
# ---------------------------------------------------------------------------


class TestPreprocessData:
    """Tests for aquacontam.pipeline.preprocess.preprocess_data."""

    def test_uses_cached_if_exists(self, tmp_path: Path) -> None:
        """If merged_wq.parquet exists in interim/, it is loaded without merging."""
        from aquacontam.pipeline.preprocess import preprocess_data

        interim = tmp_path / "interim"
        interim.mkdir()
        cached = interim / "merged_wq.parquet"
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["PFOS"],
                "concentration": [10.0],
                "latitude": [34.0],
                "longitude": [-118.0],
            }
        )
        df.to_parquet(cached, index=False)

        result = preprocess_data(tmp_path, {})
        assert len(result) == 1
        assert result["pwsid"].iloc[0] == "CA0101001"

    def test_returns_empty_if_no_sources(self, tmp_path: Path) -> None:
        """No downloaded WQ sources produces an empty DataFrame."""
        from aquacontam.pipeline.preprocess import preprocess_data

        (tmp_path / "interim").mkdir(parents=True, exist_ok=True)
        result = preprocess_data(tmp_path, {})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for aquacontam.pipeline.features.extract_features."""

    def test_cached_features_loaded(self, tmp_path: Path) -> None:
        """Pre-existing parquets in features/ dir are loaded without extraction."""
        from aquacontam.pipeline.features import extract_features

        features_dir = tmp_path / "interim" / "features"
        features_dir.mkdir(parents=True)

        # Create cached proximity parquet
        prox_df = pd.DataFrame(
            {"nearest_industrial_km": [5.0]}, index=pd.Index(["CA0101001"], name="pwsid")
        )
        prox_df.to_parquet(features_dir / "proximity.parquet")

        # Create cached hydrogeology parquet
        hydro_df = pd.DataFrame(
            {"aquifer_type": ["sandstone"]}, index=pd.Index(["CA0101001"], name="pwsid")
        )
        hydro_df.to_parquet(features_dir / "hydrogeology.parquet")

        # Minimal wq_df (non-empty so it doesn't short-circuit)
        wq_df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "latitude": [34.0],
                "longitude": [-118.0],
                "analyte": ["PFOS"],
            }
        )

        # Mock build_system_geodataframe to avoid geopandas dependency chain.
        # The function is imported inside extract_features from
        # aquacontam.features.assembly, so patch at the source.
        mock_gdf = MagicMock()
        with patch(
            "aquacontam.features.assembly.build_system_geodataframe",
            return_value=mock_gdf,
        ):
            result = extract_features(tmp_path, wq_df, {}, skip_large=True)

        # Should have loaded both cached feature DFs
        assert len(result) >= 2

    def test_empty_wq_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty wq DataFrame produces an empty feature list."""
        from aquacontam.pipeline.features import extract_features

        (tmp_path / "interim" / "features").mkdir(parents=True)
        result = extract_features(tmp_path, pd.DataFrame(), {})
        assert result == []


# ---------------------------------------------------------------------------
# assembly.py
# ---------------------------------------------------------------------------


class TestAssembleWithSplitImputation:
    """Tests for aquacontam.pipeline.assembly.assemble_with_split_imputation."""

    def test_returns_four_items(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """Return value is a 4-tuple (X, y, impute_stats, regions)."""
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )
        from aquacontam.pipeline.assembly import assemble_with_split_imputation

        sys_targets = aggregate_to_system_level(synthetic_wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)

        result = assemble_with_split_imputation(sys_targets, synthetic_feature_dfs)

        assert len(result) == 4
        X, y, impute_stats, regions = result
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(impute_stats, dict)
        assert isinstance(regions, pd.Series)
        # X and y should have aligned indices
        assert list(X.index) == list(y.index)

    def test_imputation_uses_train_only(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """Impute stats are computed from train regions only, not full data."""
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )
        from aquacontam.pipeline.assembly import assemble_with_split_imputation

        sys_targets = aggregate_to_system_level(synthetic_wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)

        calls: list[dict[str, Any]] = []
        real_assemble = None

        # Import the real function so we can track calls
        from aquacontam.features.assembly import assemble_feature_matrix as _real

        real_assemble = _real

        def _tracking_assemble(sys_tgt: Any, *feature_dfs: Any, **kwargs: Any) -> Any:
            calls.append({"n_systems": len(sys_tgt), "kwargs": kwargs})
            return real_assemble(sys_tgt, *feature_dfs, **kwargs)

        with patch(
            "aquacontam.features.assembly.assemble_feature_matrix",
            side_effect=_tracking_assemble,
        ):
            _X, _y, _impute_stats, _regions = assemble_with_split_imputation(
                sys_targets, synthetic_feature_dfs
            )

        # First call = train split (no impute_stats kwarg)
        # Second+ calls = val/test splits (with impute_stats kwarg)
        assert len(calls) >= 2
        # First call should NOT have impute_stats
        assert "impute_stats" not in calls[0]["kwargs"]
        # Subsequent calls SHOULD have impute_stats
        for call in calls[1:]:
            assert "impute_stats" in call["kwargs"]

    def test_column_alignment_rare_categories(
        self,
        synthetic_wq_df: pd.DataFrame,
        synthetic_feature_dfs: list[pd.DataFrame],
    ) -> None:
        """Val/test splits are column-aligned to training despite rare categories."""
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )
        from aquacontam.pipeline.assembly import assemble_with_split_imputation

        sys_targets = aggregate_to_system_level(synthetic_wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)

        # Inject a rare category only in val region (NJ = EPA region 2)
        feature_dfs = [df.copy() for df in synthetic_feature_dfs]
        aquifer_df = feature_dfs[2]  # aquifer features
        nj_mask = aquifer_df.index.str.startswith("NJ")
        aquifer_df.loc[nj_mask, "aquifer_type"] = "volcanic"
        feature_dfs[2] = aquifer_df

        X, _y, _impute_stats, regions = assemble_with_split_imputation(sys_targets, feature_dfs)

        # No object-dtype columns (the bool→int8 + reindex fix)
        obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
        assert obj_cols == [], f"Object columns found: {obj_cols}"

        # All rows have the same columns (no NaN from concat mismatch)
        train_idx = X.index[regions.isin((1, 3, 4, 5, 6))]
        val_idx = X.index[regions.isin((2, 7))]
        assert set(X.loc[train_idx].columns) == set(X.loc[val_idx].columns)

        # Val-only category "volcanic" should NOT appear as a column
        volcanic_cols = [c for c in X.columns if "volcanic" in c]
        assert volcanic_cols == [], f"Val-only category leaked: {volcanic_cols}"


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------


class TestLoadModelConfigs:
    """Tests for aquacontam.pipeline.models.load_model_configs."""

    def test_returns_dict(self) -> None:
        """load_model_configs returns a dict."""
        from aquacontam.pipeline.models import load_model_configs

        result = load_model_configs()
        assert isinstance(result, dict)

    def test_file_not_found_returns_empty(self) -> None:
        """When experiment.yaml is missing, returns empty dict."""
        from aquacontam.pipeline.models import load_model_configs

        with patch(
            "aquacontam._config.load_experiment_config",
            side_effect=FileNotFoundError("no file"),
        ):
            result = load_model_configs()

        assert result == {}


class TestWithNameOverride:
    """Tests for aquacontam.pipeline.models.with_name_override."""

    def test_overrides_name(self) -> None:
        """The model's name property is replaced with the override."""
        from aquacontam.pipeline.models import with_name_override

        model = MagicMock()
        model.name = "original_name"
        # with_name_override patches __class__, so we need a real object
        # Use a lightweight real model
        from aquacontam.models.logistic import DummyClassifierBaseline

        model = DummyClassifierBaseline(config={"random_state": 42})
        assert model.name != "custom_name"

        result = with_name_override(model, "custom_name")
        assert result.name == "custom_name"
        # Should be the same object
        assert result is model


class TestGetModelInstances:
    """Tests for aquacontam.pipeline.models.get_model_instances."""

    def test_classifiers_returned(self) -> None:
        """Requesting 'classification' returns at least xgboost + RF."""
        from aquacontam.pipeline.models import get_model_instances

        models = get_model_instances(None, "classification", 42)
        names = [m.name for m in models]
        # At minimum: dummy, logistic, xgboost, xgboost_default, random_forest
        assert len(models) >= 4
        assert any("xgboost" in n.lower() for n in names)
        assert any("random_forest" in n.lower() for n in names)

    def test_filter_by_name(self) -> None:
        """Filtering by name returns only the specified models."""
        from aquacontam.pipeline.models import get_model_instances

        models = get_model_instances(["xgboost"], "classification", 42)
        assert len(models) == 1

    def test_unknown_model_returns_empty(self) -> None:
        """An unknown model name returns an empty list."""
        from aquacontam.pipeline.models import get_model_instances

        models = get_model_instances(["nonexistent_model"], "classification", 42)
        assert len(models) == 0

    def test_regressors_returned(self) -> None:
        """Requesting 'regression' returns regressor instances."""
        from aquacontam.pipeline.models import get_model_instances

        models = get_model_instances(None, "regression", 42)
        assert len(models) >= 2


# ---------------------------------------------------------------------------
# training.py
# ---------------------------------------------------------------------------


class TestBuildTaskKwargs:
    """Tests for aquacontam.pipeline.training.build_task_kwargs."""

    def test_t1_standard_kwargs(self) -> None:
        """T1 produces standard kwargs with 'data', 'model', 'feature_dfs'."""
        from aquacontam.pipeline.training import build_task_kwargs

        model = MagicMock()
        wq_df = pd.DataFrame({"pwsid": ["A"]})
        feature_dfs = [pd.DataFrame({"f1": [1.0]}, index=["A"])]
        downloaded: dict[str, Path] = {}

        result = build_task_kwargs("T1", model, wq_df, feature_dfs, downloaded)

        assert "model" in result
        assert "data" in result
        assert "feature_dfs" in result
        assert result["model"] is model
        assert result["data"] is wq_df
        # T1 should NOT have source_data/target_data/well_data
        assert "source_data" not in result
        assert "target_data" not in result
        assert "well_data" not in result

    def test_t5_has_source_and_target(self, tmp_path: Path) -> None:
        """T5 adds source_data and target_data to kwargs."""
        from aquacontam.pipeline.training import build_task_kwargs

        model = MagicMock()
        wq_df = pd.DataFrame({"pwsid": ["A"]})
        feature_dfs: list[pd.DataFrame] = []

        # Create fake downloaded parquets for sdwis and ucmr5
        sdwis_path = tmp_path / "sdwis.parquet"
        ucmr5_path = tmp_path / "ucmr5.parquet"
        pd.DataFrame({"pwsid": ["B"], "source": ["sdwis"]}).to_parquet(sdwis_path)
        pd.DataFrame({"pwsid": ["C"], "source": ["ucmr5"]}).to_parquet(ucmr5_path)

        downloaded = {"sdwis": sdwis_path, "ucmr5": ucmr5_path}

        result = build_task_kwargs("T5", model, wq_df, feature_dfs, downloaded)

        assert "source_data" in result
        assert "target_data" in result
        assert "data" not in result  # T5 does NOT use generic 'data'

    def test_t6_without_private_wells(self) -> None:
        """T6 without nj_private_wells still returns well_data (empty DF)."""
        from aquacontam.pipeline.training import build_task_kwargs

        model = MagicMock()
        wq_df = pd.DataFrame({"pwsid": ["A"]})
        feature_dfs: list[pd.DataFrame] = []
        downloaded: dict[str, Path] = {}

        result = build_task_kwargs("T6", model, wq_df, feature_dfs, downloaded)

        assert "well_data" in result
        assert "data" in result
        assert len(result["well_data"]) == 0  # Empty DataFrame


class TestTrainAndEvaluate:
    """Tests for aquacontam.pipeline.training.train_and_evaluate."""

    def test_skips_t6_without_private_wells(self, tmp_path: Path) -> None:
        """T6 is skipped entirely when nj_private_wells is not in downloaded."""
        from aquacontam.pipeline.training import train_and_evaluate

        wq_df = pd.DataFrame({"pwsid": ["A"]})
        feature_dfs: list[pd.DataFrame] = []
        downloaded: dict[str, Path] = {}  # No nj_private_wells

        # Mock list_tasks to return only T6
        mock_task_info = MagicMock()
        mock_task_info.name = "T6"
        mock_task_info.task_type = "classification"

        mock_task_fn = MagicMock()

        with (
            patch(
                "aquacontam.benchmark.registry.list_tasks",
                return_value=[mock_task_info],
            ),
            patch(
                "aquacontam.benchmark.registry.get_task",
                return_value=(mock_task_info, mock_task_fn),
            ),
        ):
            results, fitted, _ = train_and_evaluate(
                wq_df,
                feature_dfs,
                downloaded,
                tmp_path,
                task_filter=["T6"],
            )

        # T6 should be skipped — task function never called
        mock_task_fn.assert_not_called()
        assert results == []
        assert fitted == {}


# ---------------------------------------------------------------------------
# export.py
# ---------------------------------------------------------------------------


class TestExportResults:
    """Tests for aquacontam.pipeline.export.export_results."""

    def test_creates_results_json(self, tmp_path: Path) -> None:
        """export_results writes results.json to output_dir."""
        from aquacontam.pipeline.export import export_results

        results = [
            {
                "task": "T1",
                "model": "xgboost_classifier",
                "metrics": {"auroc": 0.85},
                "split_metrics": {},
                "metadata": {},
            }
        ]
        export_results(results, tmp_path)

        path = tmp_path / "results.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["task"] == "T1"

    def test_creates_submission_jsons(self, tmp_path: Path) -> None:
        """export_results creates per-model submission JSONs."""
        from aquacontam.pipeline.export import export_results

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
        export_results(results, tmp_path)

        sub_dir = tmp_path / "submissions"
        assert sub_dir.exists()
        files = list(sub_dir.glob("*.json"))
        assert len(files) == 1  # One model
        sub = json.loads(files[0].read_text())
        assert sub["model_name"] == "xgboost"
        assert len(sub["results"]) == 2


class TestExportFeatureImportances:
    """Tests for aquacontam.pipeline.export.export_feature_importances."""

    def test_writes_json(self, tmp_path: Path) -> None:
        """Model with feature_importances() produces feature_importance.json."""
        from aquacontam.pipeline.export import export_feature_importances

        model = MagicMock()
        model.name = "xgboost_classifier"
        model.feature_importances.return_value = pd.Series(
            {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        )
        fitted_models = {"T1": (model, "classification", 0.5)}

        result = export_feature_importances(fitted_models, tmp_path)

        assert result is not None
        out_path = tmp_path / "feature_importance.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "feat_a" in data
        assert data["feat_a"] == 0.5

    def test_skips_unsupported_model(self, tmp_path: Path) -> None:
        """Model raising NotImplementedError is skipped gracefully."""
        from aquacontam.pipeline.export import export_feature_importances

        model = MagicMock()
        model.name = "dummy_classifier"
        model.feature_importances.side_effect = NotImplementedError("not supported")
        fitted_models = {"T1": (model, "classification", 0.5)}

        result = export_feature_importances(fitted_models, tmp_path)

        assert result is None
        assert not (tmp_path / "feature_importance.json").exists()


class TestExportIcpDiagnostics:
    """Tests for aquacontam.pipeline.export.export_icp_diagnostics."""

    def test_no_icp_results_skips(self, tmp_path: Path) -> None:
        """When no ICP results exist, skip gracefully without writing."""
        from aquacontam.pipeline.export import export_icp_diagnostics

        results = [
            {"task": "T1", "model": "xgboost", "metrics": {"auroc": 0.85}},
        ]
        fitted_models: dict[str, tuple[Any, str, float]] = {}

        export_icp_diagnostics(results, fitted_models, tmp_path)

        assert not (tmp_path / "icp_diagnostics.json").exists()

    def test_icp_results_written(self, tmp_path: Path) -> None:
        """ICP results are written to icp_diagnostics.json."""
        from aquacontam.pipeline.export import export_icp_diagnostics

        results = [
            {
                "task": "T1",
                "model": "icp_classifier",
                "metrics": {"auroc": 0.80},
            },
        ]
        model = MagicMock()
        model.name = "icp_classifier"
        fitted_models = {"T1": (model, "classification", 0.5)}

        export_icp_diagnostics(results, fitted_models, tmp_path)

        out_path = tmp_path / "icp_diagnostics.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "results" in data
        assert len(data["results"]) == 1

    def test_history_from_icp_models_channel(self, tmp_path: Path) -> None:
        """ICP per-epoch history is persisted from the icp_models side-channel.

        Regression guard: fitted_models[T1] holds the tree SHAP
        model (no ICP diagnostics); the ICP's training history must reach
        icp_diagnostics.json via the dedicated icp_models channel from
        train_and_evaluate — not be silently dropped.
        """
        from aquacontam.pipeline.export import export_icp_diagnostics

        results = [
            {"task": "T1", "model": "icp_classifier", "metrics": {"auroc": 0.731}},
        ]
        # fitted_models holds a NON-ICP tree model (as in the real pipeline).
        tree = MagicMock()
        tree.name = "xgboost_classifier"
        fitted_models = {"T1": (tree, "classification", 0.5)}
        # The ICP arrives via its own channel.
        icp = MagicMock()
        icp.name = "icp_classifier"
        icp.get_training_history.return_value = [
            {"epoch": 0, "task_loss": 0.45, "adv_loss": 1.05, "lambda_adv": 0.0},
            {"epoch": 1, "task_loss": 0.40, "adv_loss": 1.03, "lambda_adv": 0.0},
        ]
        icp.get_adversary_r2.return_value = 0.012

        export_icp_diagnostics(results, fitted_models, tmp_path, icp_models={"T1": icp})

        data = json.loads((tmp_path / "icp_diagnostics.json").read_text())
        assert data["history"]["T1"], "ICP training history not persisted"
        assert len(data["history"]["T1"]) == 2
        assert data["adversary_r2"]["T1"] == 0.012


class TestGeneratePaperAssets:
    """Tests for aquacontam.pipeline.export.generate_paper_assets."""

    def test_skips_if_matplotlib_missing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When paper.generate_figures can't be imported, logs warning and returns."""
        from aquacontam.pipeline.export import generate_paper_assets

        real_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _block_figures(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "paper.generate_figures":
                raise ImportError("mocked: no matplotlib")
            return real_import(name, *args, **kwargs)

        with (
            patch.dict("sys.modules", {"paper.generate_figures": None}),
            patch("builtins.__import__", side_effect=_block_figures),
        ):
            generate_paper_assets(tmp_path, tmp_path)

        assert any("matplotlib" in r.message or "figure" in r.message for r in caplog.records)


class TestExportZenodoDataset:
    """Tests for aquacontam.pipeline.export.export_zenodo_dataset."""

    def test_calls_export_and_metadata(self, tmp_path: Path) -> None:
        """export_zenodo_dataset calls export_dataset and write_metadata_files."""
        from aquacontam.pipeline.export import export_zenodo_dataset

        wq_df = pd.DataFrame({"pwsid": ["A", "B"], "analyte": ["PFOS", "PFOA"]})
        mock_export = MagicMock()
        mock_write_meta = MagicMock(return_value=[])

        with (
            patch("aquacontam.export.dataset.export_dataset", mock_export),
            patch("aquacontam.export.metadata.write_metadata_files", mock_write_meta),
        ):
            export_zenodo_dataset(wq_df, [], tmp_path, tmp_path)

        mock_export.assert_called_once()
        mock_write_meta.assert_called_once()
        # Verify output dir
        call_kwargs = mock_export.call_args
        assert call_kwargs[1]["output_dir"] == tmp_path / "zenodo-dataset"
