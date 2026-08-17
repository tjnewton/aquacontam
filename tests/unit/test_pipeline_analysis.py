"""Tests for pipeline analysis functions (aquacontam.pipeline.analysis).

Each analysis function follows a pattern: take inputs, compute (mocked), write
JSON/CSV to output_dir, and handle errors via try/except with is_strict().
We mock the expensive inner functions and verify the output files are written.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from aquacontam.pipeline._strict import set_strict

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_strict():
    """Ensure strict mode is off around each test."""
    set_strict(False)
    yield
    set_strict(False)


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Create and return a temporary output directory."""
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture()
def classification_results() -> list[dict[str, Any]]:
    """Minimal results list with metadata for classification analysis functions."""
    rng = np.random.RandomState(0)
    n = 100
    y_true = rng.randint(0, 2, n).tolist()
    y_prob = rng.uniform(0, 1, n).tolist()
    y_pred = [int(p >= 0.5) for p in y_prob]
    latitudes = rng.uniform(30, 48, n).tolist()
    longitudes = rng.uniform(-120, -70, n).tolist()
    split_labels = ["train"] * 50 + ["val"] * 20 + ["test"] * 30

    return [
        {
            "task": "T1",
            "model": "xgboost_classifier",
            "metrics": {"auroc": 0.85, "auprc": 0.70},
            "metadata": {
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_pred,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "split_labels": split_labels,
                "X_test": pd.DataFrame(
                    rng.randn(30, 5),
                    columns=[f"feat_{i}" for i in range(5)],
                ),
            },
        },
        {
            "task": "T4",
            "model": "catboost_classifier",
            "metrics": {"auroc": 0.90},
            "metadata": {
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_pred,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "split_labels": split_labels,
            },
        },
    ]


@pytest.fixture()
def dummy_fitted_models() -> dict[str, tuple[Any, str, float]]:
    """Fitted models dict with a mock T1 model."""
    model = MagicMock()
    model.name = "xgboost_classifier"
    model.predict_proba.return_value = np.column_stack(
        [np.zeros(30), np.random.RandomState(0).uniform(0, 1, 30)]
    )
    model.predict.return_value = np.zeros(30, dtype=int)
    model.feature_names = [f"feat_{i}" for i in range(5)]
    return {"T1": (model, "classification", 0.5)}


# ---------------------------------------------------------------------------
# _run_equity_analysis
# ---------------------------------------------------------------------------


class TestRunEquityAnalysis:
    """Tests for _run_equity_analysis."""

    def test_skips_without_demographics(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_equity_analysis

        # feature_dfs with no pct_people_of_color column
        feature_dfs = [pd.DataFrame({"x": [1, 2]}, index=["A", "B"])]
        _run_equity_analysis(
            wq_df=pd.DataFrame(),
            feature_dfs=feature_dfs,
            fitted_models={},
            output_dir=output_dir,
        )
        assert not (output_dir / "equity_analysis.json").exists()

    def test_skips_without_t1_model(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_equity_analysis

        demo_df = pd.DataFrame(
            {"pct_people_of_color": [0.3]}, index=pd.Index(["SYS001"], name="pwsid")
        )
        _run_equity_analysis(
            wq_df=pd.DataFrame(),
            feature_dfs=[demo_df],
            fitted_models={},  # no T1
            output_dir=output_dir,
        )
        assert not (output_dir / "equity_analysis.json").exists()


# ---------------------------------------------------------------------------
# _run_spatial_autocorrelation
# ---------------------------------------------------------------------------


class TestRunSpatialAutocorrelation:
    """Tests for _run_spatial_autocorrelation."""

    @patch("aquacontam.pipeline.analysis._evaluation.analyze_spatial_autocorrelation", create=True)
    def test_writes_json(self, mock_sa: MagicMock, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_spatial_autocorrelation

        # Build a mock MoranResult
        mock_result = MagicMock()
        mock_result.statistic = 0.05
        mock_result.expected = -0.01
        mock_result.z_score = 2.1
        mock_result.p_value = 0.035
        mock_result.n = 100

        # analyze_spatial_autocorrelation is imported inside the function,
        # so we patch it at the module where it's used
        with patch(
            "aquacontam.analysis.spatial_autocorrelation.analyze_spatial_autocorrelation"
        ) as mock_fn:
            mock_fn.return_value = {
                "overall": mock_result,
                "splits": {},
                "multi_threshold": [],
            }
            # Re-import to get the patched version
            # The function does a top-level import, so we patch the module namespace
            _run_spatial_autocorrelation(
                results=[
                    {
                        "task": "T1",
                        "model": "xgb",
                        "metadata": {
                            "y_true": [0, 1, 0, 1],
                            "y_prob": [0.1, 0.9, 0.2, 0.8],
                            "latitudes": [40.0, 41.0, 42.0, 43.0],
                            "longitudes": [-75.0, -76.0, -77.0, -78.0],
                            "split_labels": None,
                        },
                    }
                ],
                output_dir=output_dir,
            )

        out_path = output_dir / "spatial_autocorrelation.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) == 1
        assert data[0]["task"] == "T1"

    def test_skips_without_metadata(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_spatial_autocorrelation

        # No y_true/y_prob in metadata
        _run_spatial_autocorrelation(
            results=[{"task": "T1", "model": "xgb", "metadata": {}}],
            output_dir=output_dir,
        )
        assert not (output_dir / "spatial_autocorrelation.json").exists()


# ---------------------------------------------------------------------------
# _run_calibration_analysis
# ---------------------------------------------------------------------------


class TestRunCalibrationAnalysis:
    """Tests for _run_calibration_analysis."""

    def test_writes_json(
        self, classification_results: list[dict[str, Any]], output_dir: Path
    ) -> None:
        from aquacontam.pipeline.analysis import _run_calibration_analysis

        _run_calibration_analysis(classification_results, output_dir)

        out_path = output_dir / "calibration_analysis.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) >= 1
        assert "ece" in data[0]
        assert "brier_score" in data[0]

    def test_skips_t2(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_calibration_analysis

        results: list[dict[str, Any]] = [
            {
                "task": "T2",
                "model": "xgb",
                "metadata": {
                    "y_true": [1.0, 2.0],
                    "y_prob": [1.1, 2.1],
                },
            }
        ]
        _run_calibration_analysis(results, output_dir)
        assert not (output_dir / "calibration_analysis.json").exists()


# ---------------------------------------------------------------------------
# _run_loro_cv
# ---------------------------------------------------------------------------


class TestRunLoroCv:
    """Tests for _run_loro_cv."""

    @patch("aquacontam.pipeline.analysis._robustness.assemble_with_split_imputation")
    @patch("aquacontam.pipeline.analysis._robustness._load_model_configs")
    @patch("aquacontam.features.assembly.drop_leakage_columns")
    @patch("aquacontam.features.assembly.aggregate_to_system_level")
    def test_writes_json(
        self,
        mock_agg: MagicMock,
        mock_drop: MagicMock,
        mock_configs: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_cv

        mock_configs.return_value = {"xgboost_classifier": {"n_estimators": 10}}

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)
        mock_agg.return_value = pd.DataFrame()
        mock_drop.return_value = pd.DataFrame()

        # Mock the models to avoid real training
        with patch("aquacontam.models.xgboost.XGBoostClassifier") as MockXGB:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.zeros(10, dtype=int)
            mock_model.predict_proba.return_value = np.column_stack(
                [np.ones(10) * 0.5, np.ones(10) * 0.5]
            )
            MockXGB.return_value = mock_model

            _run_loro_cv(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "loro_cv.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert isinstance(data, dict)

    @patch("aquacontam.pipeline.analysis._robustness.assemble_with_split_imputation")
    @patch("aquacontam.pipeline.analysis._robustness._load_model_configs")
    @patch("aquacontam.features.assembly.drop_leakage_columns")
    @patch("aquacontam.features.assembly.aggregate_to_system_level")
    def test_region5_fold_persists_oof_predictions(
        self,
        mock_agg: MagicMock,
        mock_drop: MagicMock,
        mock_configs: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        """The region-5 fold gains aligned per-system OOF (for the MN held-out read)."""
        from aquacontam.pipeline.analysis import _run_loro_cv

        mock_configs.return_value = {"xgboost_classifier": {"n_estimators": 10}}

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)
        mock_agg.return_value = pd.DataFrame()
        mock_drop.return_value = pd.DataFrame()

        with patch("aquacontam.models.xgboost.XGBoostClassifier") as MockXGB:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.zeros(10, dtype=int)
            mock_model.predict_proba.return_value = np.column_stack(
                [np.ones(10) * 0.4, np.ones(10) * 0.6]
            )
            MockXGB.return_value = mock_model
            _run_loro_cv(pd.DataFrame(), [], output_dir, seed=42)

        data = json.loads((output_dir / "loro_cv.json").read_text())
        found = False
        for models in data.values():
            if not isinstance(models, dict):
                continue
            for res in models.values():
                for fold in res.get("folds", []) if isinstance(res, dict) else []:
                    if fold.get("test_region") != 5:
                        assert "oof_predictions" not in fold  # only region 5 captures OOF
                        continue
                    oof = fold["oof_predictions"]
                    n_test = fold["n_test"]
                    assert len(oof["pwsid"]) == n_test
                    assert len(oof["y_true"]) == n_test
                    assert len(oof["y_prob"]) == n_test
                    found = True
        assert found, "no region-5 fold carried oof_predictions"


# ---------------------------------------------------------------------------
# _run_multi_seed_analysis
# ---------------------------------------------------------------------------


class TestRunMultiSeedAnalysis:
    """Tests for _run_multi_seed_analysis."""

    @patch("aquacontam.pipeline.analysis._robustness.assemble_with_split_imputation")
    @patch("aquacontam.pipeline.analysis._robustness._load_model_configs")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_configs: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_multi_seed_analysis

        mock_configs.return_value = {"xgboost_classifier": {"n_estimators": 10}}

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        # Regions: train=1,3,4,5,6 val=2,7 test=8,9,10
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        with patch("aquacontam.models.xgboost.XGBoostClassifier") as MockXGB:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.zeros(30, dtype=int)
            mock_model.predict_proba.return_value = np.column_stack(
                [np.ones(30) * 0.4, np.ones(30) * 0.6]
            )
            MockXGB.return_value = mock_model

            with patch(
                "aquacontam.pipeline.analysis.load_experiment_config",
                side_effect=FileNotFoundError,
                create=True,
            ):
                _run_multi_seed_analysis(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "multi_seed_stability.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# _run_region_heterogeneity
# ---------------------------------------------------------------------------


class TestRunRegionHeterogeneity:
    """Tests for _run_region_heterogeneity."""

    @patch("aquacontam.pipeline.analysis._robustness.assemble_with_split_imputation")
    @patch("aquacontam.analysis.regional.analyze_regional_heterogeneity")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_analyze: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_region_heterogeneity

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)
        mock_analyze.return_value = {"region_stats": {}, "kruskal_p_value": 0.04}

        _run_region_heterogeneity(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "region_heterogeneity.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "kruskal_p_value" in data


# ---------------------------------------------------------------------------
# _run_bootstrap_cis
# ---------------------------------------------------------------------------


class TestRunBootstrapCis:
    """Tests for _run_bootstrap_cis."""

    def test_updates_results_with_cis(
        self, classification_results: list[dict[str, Any]], output_dir: Path
    ) -> None:
        from aquacontam.pipeline.analysis import _run_bootstrap_cis

        updated = _run_bootstrap_cis(classification_results, output_dir, n_bootstrap=50, seed=42)

        # At least one result should have bootstrap_ci in metadata
        has_ci = any("bootstrap_ci" in r.get("metadata", {}) for r in updated)
        assert has_ci

        # Check the summary file was written
        ci_path = output_dir / "bootstrap_ci.json"
        assert ci_path.exists()
        data = json.loads(ci_path.read_text())
        assert len(data) > 0
        assert "ci_lower" in data[0]
        assert "ci_upper" in data[0]

    def test_derives_y_pred_from_y_prob_when_absent(self, output_dir: Path) -> None:
        """A result with only y_true + y_prob (no y_pred) still gets a CI.

        The T4 retrain produces rows that store y_prob but not y_pred;
        _run_bootstrap_cis must derive y_pred = (y_prob >= 0.5) so the
        AUROC/AUPRC bootstrap CIs are computed rather than skipped. See PR #41.
        """
        from aquacontam.pipeline.analysis import _run_bootstrap_cis

        rng = np.random.RandomState(0)
        n = 100
        results = [
            {
                "task": "T4",
                "model": "logistic_regression",
                "metrics": {"auroc": 0.80, "auprc": 0.60},
                "metadata": {
                    "y_true": rng.randint(0, 2, n).tolist(),
                    "y_prob": rng.uniform(0, 1, n).tolist(),
                    # NB: no "y_pred" key — must be derived from y_prob.
                },
            }
        ]

        updated = _run_bootstrap_cis(results, output_dir, n_bootstrap=50, seed=42)

        assert "bootstrap_ci" in updated[0]["metadata"], (
            "CI must be computed from y_prob when y_pred is absent"
        )
        ci = updated[0]["metadata"]["bootstrap_ci"]
        assert "auroc" in ci and "auprc" in ci


# ---------------------------------------------------------------------------
# _run_model_comparison
# ---------------------------------------------------------------------------


class TestRunModelComparison:
    """Tests for _run_model_comparison."""

    def test_writes_json(
        self, classification_results: list[dict[str, Any]], output_dir: Path
    ) -> None:
        from aquacontam.pipeline.analysis import _run_model_comparison

        # Need >=2 models on same task; modify results so both are T1
        results = classification_results.copy()
        results[1] = {**results[1], "task": "T1", "model": "random_forest"}

        with patch("aquacontam.benchmark.metrics.compare_models_pairwise") as mock_cmp:
            mock_df = pd.DataFrame(
                {
                    "model_a": ["xgboost_classifier"],
                    "model_b": ["random_forest"],
                    "auroc_a": [0.85],
                    "auroc_b": [0.80],
                    "p_value": [0.03],
                    "significant": [True],
                }
            )
            mock_cmp.return_value = mock_df

            _run_model_comparison(results, output_dir)

        out_path = output_dir / "model_comparison.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "T1" in data

    def test_skips_single_model(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_model_comparison

        results: list[dict[str, Any]] = [
            {
                "task": "T1",
                "model": "xgb",
                "metadata": {
                    "y_true": [0, 1],
                    "y_prob": [0.1, 0.9],
                },
            }
        ]
        _run_model_comparison(results, output_dir)
        # Only 1 model on T1, so no comparison
        assert not (output_dir / "model_comparison.json").exists()


# ---------------------------------------------------------------------------
# _run_feature_ablation
# ---------------------------------------------------------------------------


class TestRunFeatureAblation:
    """Tests for _run_feature_ablation."""

    @patch("aquacontam.pipeline.analysis._robustness.assemble_with_split_imputation")
    @patch("aquacontam.analysis.ablation.run_feature_ablation")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_ablation: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_feature_ablation

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(
            rng.randn(n, 6),
            columns=[
                "nearest_industrial_km",
                "frac_developed_1km",
                "aquifer_depth_m",
                "pop_served",
                "n_samples",
                "mean_dl",
            ],
        )
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        # Mock ablation results
        mock_ar = MagicMock()
        mock_ar.category = "proximity"
        mock_ar.n_features_dropped = 1
        mock_ar.n_features_remaining = 5
        mock_ar.metrics = {"auroc": 0.82, "auprc": 0.65}
        mock_ar.significance = {"auroc": {"p_value": 0.01}}
        mock_ablation.return_value = [mock_ar]

        # Provide config with ablation categories
        yaml_content = {
            "ablation": {
                "feature_categories": {
                    "proximity": ["nearest_"],
                    "land_use": ["frac_"],
                }
            }
        }
        with (
            patch("builtins.open", create=True),
            patch("yaml.safe_load", return_value=yaml_content),
            patch("pathlib.Path.exists", return_value=True),
        ):
            _run_feature_ablation(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "feature_ablation.json"
        assert out_path.exists()

    def test_skips_without_categories(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_feature_ablation

        # No config file -> no categories -> skip
        with patch("pathlib.Path.exists", return_value=False):
            _run_feature_ablation(pd.DataFrame(), [], output_dir, seed=42)

        assert not (output_dir / "feature_ablation.json").exists()


# ---------------------------------------------------------------------------
# _run_shap_analysis
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not importlib.util.find_spec("shap"),
    reason="shap not installed",
)
class TestRunShapAnalysis:
    """Tests for _run_shap_analysis."""

    def test_writes_json(
        self,
        classification_results: list[dict[str, Any]],
        dummy_fitted_models: dict[str, tuple[Any, str, float]],
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_shap_analysis

        with patch("aquacontam.analysis.interpretability.compute_shap_values") as mock_shap:
            mock_shap.return_value = pd.DataFrame(
                np.random.RandomState(0).randn(30, 5),
                columns=[f"feat_{i}" for i in range(5)],
            )

            _run_shap_analysis(classification_results, dummy_fitted_models, output_dir)

        out_path = output_dir / "shap_T1.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["task"] == "T1"
        assert "mean_abs_shap" in data

    def test_skips_trivial_model(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_shap_analysis

        model = MagicMock()
        model.name = "dummy_classifier"
        fitted = {"T1": (model, "classification", 0.5)}

        _run_shap_analysis([], fitted, output_dir)
        assert not (output_dir / "shap_T1.json").exists()


# ---------------------------------------------------------------------------
# _run_conformal_analysis
# ---------------------------------------------------------------------------


class TestRunConformalAnalysis:
    """Tests for _run_conformal_analysis."""

    def test_writes_json(
        self, classification_results: list[dict[str, Any]], output_dir: Path
    ) -> None:
        from aquacontam.pipeline.analysis import _run_conformal_analysis

        _run_conformal_analysis(classification_results, output_dir)

        out_path = output_dir / "conformal_results.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) > 0
        assert "coverage" in data[0]
        assert "alpha" in data[0]

    def test_skips_without_split_labels(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_conformal_analysis

        results: list[dict[str, Any]] = [
            {
                "task": "T1",
                "model": "xgb",
                "metadata": {
                    "y_true": [0, 1],
                    "y_prob": [0.1, 0.9],
                    # no split_labels
                },
            }
        ]
        _run_conformal_analysis(results, output_dir)
        assert not (output_dir / "conformal_results.json").exists()


# ---------------------------------------------------------------------------
# _run_group_conformal_analysis
# ---------------------------------------------------------------------------


class TestRunGroupConformalAnalysis:
    """Tests for _run_group_conformal_analysis."""

    @patch("aquacontam.pipeline.analysis._equity_transfer.assemble_with_split_imputation")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_group_conformal_analysis

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        mock_gcc = MagicMock()
        mock_gcc.coverage_and_set_size.return_value = {
            "coverage": 0.95,
            "avg_set_size": 1.2,
            "singleton_frac": 0.8,
            "per_group": {8: {"coverage": 0.94}, 9: {"coverage": 0.96}},
        }

        with (
            patch(
                "aquacontam.calibration.conformal.GroupConformalClassifier",
                return_value=mock_gcc,
            ),
            patch("aquacontam.models.xgboost.XGBoostClassifier") as MockXGB,
            patch(
                "aquacontam._config.load_experiment_config",
                return_value={
                    "split": {
                        "train_regions": [1, 3, 4, 5, 6],
                        "val_regions": [2, 7],
                        "test_regions": [8, 9, 10],
                    },
                    "models": {"xgboost_classifier": {}},
                },
            ),
        ):
            mock_model = MagicMock()
            MockXGB.return_value = mock_model
            _run_group_conformal_analysis(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "group_conformal_results.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) == 3  # 3 alpha levels
        assert data[0]["coverage"] == 0.95


# ---------------------------------------------------------------------------
# _run_transfer_ablation
# ---------------------------------------------------------------------------


class TestRunTransferAblation:
    """Tests for _run_transfer_ablation."""

    @patch("aquacontam.benchmark.transfer.run_t5_ablated_transfer")
    @patch("aquacontam.pipeline.analysis._equity_transfer._load_model_configs")
    def test_writes_json(
        self,
        mock_configs: MagicMock,
        mock_transfer: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_transfer_ablation

        mock_configs.return_value = {"xgboost_classifier": {"n_estimators": 10}}

        full_result = MagicMock()
        full_result.metrics = {"auroc": 0.80}
        full_result.metadata = {"n_train": 500}

        abl_result = MagicMock()
        abl_result.metrics = {"auroc": 0.75}
        abl_result.metadata = {"n_train": 500}

        mock_transfer.return_value = {
            "full": full_result,
            "no_system_chars": abl_result,
        }

        wq_df = pd.DataFrame({"pwsid": ["A", "B"], "analyte": ["PFOS", "lead"]})
        _run_transfer_ablation(wq_df, [], downloaded={}, output_dir=output_dir, seed=42)

        out_path = output_dir / "transfer_ablation.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "full" in data
        assert "no_system_chars" in data


# ---------------------------------------------------------------------------
# _run_dedup_comparison
# ---------------------------------------------------------------------------


class TestRunDedupComparison:
    """Tests for _run_dedup_comparison."""

    @patch("aquacontam.preprocessing.splits.geographic_split")
    @patch("aquacontam.preprocessing.splits.assign_epa_region")
    @patch("aquacontam.features.assembly.assemble_feature_matrix")
    @patch("aquacontam.features.assembly.drop_leakage_columns")
    @patch("aquacontam.features.assembly.aggregate_to_system_level")
    def test_writes_json(
        self,
        mock_agg: MagicMock,
        mock_drop: MagicMock,
        mock_assemble: MagicMock,
        mock_assign: MagicMock,
        mock_split: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_dedup_comparison

        rng = np.random.RandomState(42)
        n = 60
        sys_df = pd.DataFrame(
            {
                "detected": rng.randint(0, 2, n),
                "n_samples": rng.randint(1, 10, n),
            },
            index=pd.Index([f"SYS{i:04d}" for i in range(n)], name="pwsid"),
        )
        mock_agg.return_value = sys_df
        mock_drop.return_value = sys_df

        # assign_epa_region returns df with epa_region
        df_with_region = sys_df.reset_index()
        df_with_region["epa_region"] = np.repeat(np.arange(1, 11), 6)
        mock_assign.return_value = df_with_region

        # geographic_split returns train/val/test slices
        mock_split.return_value = (
            df_with_region.iloc[:30],
            df_with_region.iloc[30:40],
            df_with_region.iloc[40:],
        )

        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        mock_assemble.return_value = (X.iloc[:30], y.iloc[:30], {})

        with patch("aquacontam.models.xgboost.XGBoostClassifier") as MockXGB:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.zeros(20, dtype=int)
            mock_model.predict_proba.return_value = np.column_stack(
                [np.ones(20) * 0.5, np.ones(20) * 0.5]
            )
            MockXGB.return_value = mock_model

            with patch(
                "aquacontam.benchmark.metrics.compute_classification_metrics",
                return_value={"auroc": 0.80, "auprc": 0.65},
            ):
                _run_dedup_comparison(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "dedup_comparison.json"
        assert out_path.exists()


# ---------------------------------------------------------------------------
# _run_split_comparison
# ---------------------------------------------------------------------------


class TestRunSplitComparison:
    """Tests for _run_split_comparison."""

    @patch("aquacontam.analysis.split_strategy_comparison.run_split_comparison_matrix")
    @patch("aquacontam.analysis.split_strategy_comparison.run_split_ablation")
    def test_writes_json(
        self,
        mock_simple: MagicMock,
        mock_matrix: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_split_comparison

        mock_simple.return_value = {"geographic": 0.85, "random": 0.87}
        mock_matrix.return_value = {"xgb_geo": 0.85, "xgb_rand": 0.87}

        _run_split_comparison(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "split_comparison.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "simple" in data
        assert "matrix" in data


# ---------------------------------------------------------------------------
# _run_lift_analysis
# ---------------------------------------------------------------------------


class TestRunLiftAnalysis:
    """Tests for _run_lift_analysis."""

    @patch("aquacontam.pipeline.analysis._equity_transfer.assemble_with_split_imputation")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_assemble: MagicMock,
        dummy_fitted_models: dict[str, tuple[Any, str, float]],
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_lift_analysis

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(
            rng.randn(n, 5),
            columns=[f"feat_{i}" for i in range(5)],
        )
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        # Adjust the model to return correctly shaped predictions for test set
        model = dummy_fitted_models["T1"][0]
        test_size = int((regions.isin((8, 9, 10))).sum())
        model.predict_proba.return_value = np.column_stack(
            [np.ones(test_size) * 0.3, np.ones(test_size) * 0.7]
        )

        with (
            patch(
                "aquacontam.benchmark.metrics.compute_lift_analysis",
                return_value={"top_decile_lift": 4.5, "deciles": []},
            ),
            patch(
                "aquacontam.analysis.monitoring_bias.compute_monitoring_dependence",
                return_value={
                    "prediction_shift": 0.01,
                    "max_shift": 0.05,
                    "correlation": 0.1,
                },
            ),
        ):
            _run_lift_analysis(
                pd.DataFrame(),
                [],
                dummy_fitted_models,
                output_dir,
                seed=42,
            )

        out_path = output_dir / "lift_analysis.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "full_model" in data
        assert "monitoring_dependence" in data

    def test_skips_without_t1(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_lift_analysis

        _run_lift_analysis(pd.DataFrame(), [], {}, output_dir, seed=42)
        assert not (output_dir / "lift_analysis.json").exists()


# ---------------------------------------------------------------------------
# _run_coordinate_sensitivity
# ---------------------------------------------------------------------------


class TestRunCoordinateSensitivity:
    """Tests for _run_coordinate_sensitivity."""

    @patch("aquacontam.analysis.coordinate_sensitivity.coordinate_sensitivity_summary")
    @patch("aquacontam.analysis.coordinate_sensitivity.run_coordinate_sensitivity")
    def test_writes_json(
        self,
        mock_run: MagicMock,
        mock_summary: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_coordinate_sensitivity

        mock_result = MagicMock()
        mock_result.magnitude_km = 1.0
        mock_result.seed = 42
        mock_result.metrics = {"auroc": 0.84}
        mock_result.n_systems = 100
        mock_result.mean_displacement_km = 0.5

        mock_run.return_value = [mock_result]
        mock_summary.return_value = pd.DataFrame({"magnitude_km": [1.0], "mean_auroc": [0.84]})

        _run_coordinate_sensitivity(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "coordinate_sensitivity.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "summary" in data
        assert "raw_results" in data


# ---------------------------------------------------------------------------
# _run_fair_tuning
# ---------------------------------------------------------------------------


class TestRunFairTuning:
    """Tests for _run_fair_tuning."""

    @patch("aquacontam.pipeline.analysis._tuning.assemble_with_split_imputation")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_fair_tuning

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        mock_gs = MagicMock()
        mock_gs.return_value = (
            {"n_estimators": 100, "max_depth": 6, "random_state": 42},
            [{"config": {}, "score": 0.70}],
        )
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(30, dtype=int)
        mock_model.predict_proba.return_value = np.column_stack(
            [np.ones(30) * 0.4, np.ones(30) * 0.6]
        )

        with (
            patch(
                "aquacontam._config.load_experiment_config",
                return_value={
                    "tuning": {
                        "xgboost_classifier": {
                            "n_estimators": [50, 100],
                            "max_depth": [3, 6],
                        }
                    },
                    "models": {"xgboost_classifier": {"n_estimators": 100}},
                },
            ),
            patch("aquacontam.benchmark.tuning.grid_search", mock_gs),
            patch(
                "aquacontam.models.xgboost.XGBoostClassifier",
                return_value=mock_model,
            ),
            patch(
                "aquacontam.benchmark.metrics.compute_classification_metrics",
                return_value={"auroc": 0.85, "auprc": 0.70},
            ),
        ):
            _run_fair_tuning(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "tuning_comparison.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) > 0


# ---------------------------------------------------------------------------
# _run_optuna_tuning
# ---------------------------------------------------------------------------


class TestRunOptunaTuning:
    """Tests for _run_optuna_tuning."""

    def test_writes_json(self, output_dir: Path) -> None:
        from aquacontam.benchmark._prep_data import TaskData
        from aquacontam.pipeline.analysis import _run_optuna_tuning

        rng = np.random.RandomState(42)
        n = 80
        X_train = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y_train = pd.Series(rng.randint(0, 2, n))
        X_val = pd.DataFrame(rng.randn(20, 3), columns=["a", "b", "c"])
        y_val = pd.Series(rng.randint(0, 2, 20))
        fake_data = TaskData(X_train, y_train, X_val, y_val, fit_extra={})

        mock_study = MagicMock()
        mock_study.trials = [MagicMock()] * 5
        mock_study.best_value = 0.72
        mock_study.best_params = {"n_estimators": 200}

        with (
            patch(
                "aquacontam.benchmark.optuna_hpo.optuna_search",
                return_value=({"n_estimators": 200, "random_state": 42}, mock_study),
            ),
            patch.dict(
                "aquacontam.benchmark._prep_data.PREP_FUNCTIONS",
                {"T1": lambda *a, **k: fake_data},
            ),
        ):
            _run_optuna_tuning(
                pd.DataFrame(),
                [],
                output_dir,
                seed=42,
                n_trials=5,
                tasks=["T1"],
                models=["xgboost_classifier"],
            )

        out_path = output_dir / "optuna_tuning.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert len(data) > 0
        assert data[0]["model"] == "xgboost_classifier"
        assert data[0]["task"] == "T1"


# ---------------------------------------------------------------------------
# _run_causal_deconfounding
# ---------------------------------------------------------------------------


class TestRunCausalDeconfounding:
    """Tests for _run_causal_deconfounding."""

    @patch("aquacontam.pipeline.analysis._equity_transfer.assemble_with_split_imputation")
    @patch("aquacontam.analysis.causal_deconfounding.train_deconfounded_model")
    @patch("aquacontam.analysis.causal_deconfounding.causal_feature_analysis")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_causal: MagicMock,
        mock_deconf: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_causal_deconfounding

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        mock_causal.return_value = pd.DataFrame(
            {
                "feature": ["a", "b"],
                "causal_effect": [0.05, -0.03],
                "shap_importance": [0.01, 0.02],
            }
        )
        mock_deconf.return_value = {
            "deconfounded_auroc": 0.80,
            "original_auroc": 0.85,
        }

        with patch(
            "aquacontam._config.load_experiment_config",
            return_value={
                "ablation": {
                    "feature_categories": {"demographics": ["a"]},
                    "causal_analysis": {
                        "confounders": ["n_samples"],
                        "n_folds": 3,
                    },
                }
            },
        ):
            _run_causal_deconfounding(pd.DataFrame(), [], output_dir, seed=42)

        causal_path = output_dir / "causal_deconfounding.json"
        deconf_path = output_dir / "deconfounded_auroc.json"
        rank_path = output_dir / "dml_shap_rank_comparison.json"
        assert causal_path.exists()
        assert deconf_path.exists()
        assert rank_path.exists()

        deconf_data = json.loads(deconf_path.read_text())
        assert deconf_data["deconfounded_auroc"] == 0.80

        rank_data = json.loads(rank_path.read_text())
        assert "spearman_rho" in rank_data
        assert rank_data["n_features"] == 2


# ---------------------------------------------------------------------------
# _run_hyperparam_sensitivity
# ---------------------------------------------------------------------------


class TestRunHyperparamSensitivity:
    """Tests for _run_hyperparam_sensitivity."""

    @patch("aquacontam.pipeline.analysis._tuning.assemble_with_split_imputation")
    @patch("aquacontam.analysis.sensitivity.hyperparameter_sensitivity")
    @patch("aquacontam.features.assembly.drop_leakage_columns", return_value=pd.DataFrame())
    @patch("aquacontam.features.assembly.aggregate_to_system_level", return_value=pd.DataFrame())
    def test_writes_json(
        self,
        _mock_agg: MagicMock,
        _mock_drop: MagicMock,
        mock_sensitivity: MagicMock,
        mock_assemble: MagicMock,
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_hyperparam_sensitivity

        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, n))
        regions = pd.Series(np.repeat(np.arange(1, 11), 10))
        mock_assemble.return_value = (X, y, {}, regions)

        mock_sensitivity.return_value = [{"config": {"n_estimators": 100}, "auroc": 0.85}]

        with (
            patch("aquacontam.pipeline.analysis._tuning._load_model_configs", return_value={}),
            patch(
                "aquacontam._config.load_experiment_config",
                return_value={
                    "tuning": {
                        "xgboost_classifier": {
                            "n_estimators": [50, 100],
                            "max_depth": [3, 6],
                        }
                    }
                },
            ),
            patch(
                "aquacontam.analysis.sensitivity.generate_config_variants",
                return_value=[{"n_estimators": 100, "random_state": 42}],
            ),
        ):
            _run_hyperparam_sensitivity(pd.DataFrame(), [], output_dir, seed=42)

        out_path = output_dir / "hyperparam_sensitivity.json"
        assert out_path.exists()


# ---------------------------------------------------------------------------
# _run_external_validation
# ---------------------------------------------------------------------------


class TestRunExternalValidation:
    """Tests for _run_external_validation."""

    @patch("aquacontam.benchmark.external_validation.run_external_validation")
    def test_writes_json(
        self,
        mock_ext: MagicMock,
        dummy_fitted_models: dict[str, tuple[Any, str, float]],
        output_dir: Path,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_external_validation

        mock_ext.return_value = {"mi_mpart": {"auroc": 0.78}}

        # Create a fake parquet path
        downloaded: dict[str, Path] = {}
        _run_external_validation(
            pd.DataFrame(),
            [],
            dummy_fitted_models,
            downloaded,
            output_dir,
        )
        # No state data available with empty downloaded, so it should skip
        assert not (output_dir / "external_validation.json").exists()

    def test_skips_without_t1(self, output_dir: Path) -> None:
        from aquacontam.pipeline.analysis import _run_external_validation

        _run_external_validation(pd.DataFrame(), [], {}, {}, output_dir)
        assert not (output_dir / "external_validation.json").exists()


# ---------------------------------------------------------------------------
# _run_cnn1d_feature_ordering
# ---------------------------------------------------------------------------


class TestRunCnn1dFeatureOrdering:
    """Tests for _run_cnn1d_feature_ordering."""

    def test_skips_without_torch(self, output_dir: Path) -> None:
        """When CNN1DClassifier cannot be imported, the function returns early."""
        import builtins

        from aquacontam.pipeline.analysis import _run_cnn1d_feature_ordering

        _real_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "aquacontam.models.cnn1d":
                raise ImportError("no torch")
            return _real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            _run_cnn1d_feature_ordering(pd.DataFrame(), [], output_dir, seed=42)

        assert not (output_dir / "cnn1d_feature_ordering.json").exists()
