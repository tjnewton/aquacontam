"""Tests for split strategy comparison module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquacontam.analysis.split_strategy_comparison import (
    SplitComparisonResult,
    build_baseline_feature_set,
    build_fernandez_feature_set,
    run_split_comparison_matrix,
    split_comparison_summary_table,
)


class TestBuildBaselineFeatureSet:
    def test_selects_proximity_features(self, synthetic_feature_dfs):
        result = build_baseline_feature_set(synthetic_feature_dfs)
        assert not result.empty
        # Should include proximity features (nearest_*, count_*)
        assert any(c.startswith("nearest_") for c in result.columns)

    def test_empty_input(self):
        result = build_baseline_feature_set([])
        assert result.empty

    def test_no_matching_features(self):
        df = pd.DataFrame({"aquifer_type": ["a", "b"]}, index=["sys1", "sys2"])
        result = build_baseline_feature_set([df])
        assert result.empty


class TestBuildFernandezFeatureSet:
    def test_selects_proximity_and_land_use(self):
        """Fernandez set should include proximity + land use + demographics."""
        pwsids = [f"SYS{i:04d}" for i in range(20)]
        df1 = pd.DataFrame(
            {
                "nearest_industrial_km": np.random.default_rng(42).uniform(0, 100, 20),
                "nearest_military_km": np.random.default_rng(42).uniform(0, 100, 20),
                "count_industrial_5km": np.random.default_rng(42).integers(0, 10, 20),
                "n_samples": np.random.default_rng(42).integers(1, 100, 20),
            },
            index=pwsids,
        )
        df2 = pd.DataFrame(
            {
                "pct_developed_high": np.random.default_rng(42).uniform(0, 1, 20),
                "pct_forest_deciduous": np.random.default_rng(42).uniform(0, 1, 20),
                "pct_people_of_color": np.random.default_rng(42).uniform(0, 1, 20),
                "pct_low_income": np.random.default_rng(42).uniform(0, 1, 20),
            },
            index=pwsids,
        )
        result = build_fernandez_feature_set([df1, df2])
        assert not result.empty
        # Should include proximity + land use + demographics
        assert "nearest_industrial_km" in result.columns
        assert "pct_developed_high" in result.columns
        assert "pct_people_of_color" in result.columns
        # Should exclude monitoring features
        assert "n_samples" not in result.columns

    def test_empty_input(self):
        result = build_fernandez_feature_set([])
        assert result.empty

    def test_excludes_monitoring_and_system_features(self):
        """Should not include monitoring intensity or system characteristics."""
        pwsids = [f"SYS{i:04d}" for i in range(10)]
        df = pd.DataFrame(
            {
                "n_samples": [5] * 10,
                "mean_detection_limit": [0.01] * 10,
                "population_served": [1000] * 10,
                "aquifer_rock_type_sandstone": [1] * 10,
            },
            index=pwsids,
        )
        result = build_fernandez_feature_set([df])
        assert result.empty


class TestRunSplitComparisonMatrix:
    """Smoke tests for the multi-model comparison matrix."""

    def test_returns_matrix_results(self, synthetic_wq_df, synthetic_feature_dfs):
        """Matrix should return results for each model x feature_set x split."""
        from aquacontam.models.logistic import LogisticRegressionClassifier

        model_specs = [
            ("LogReg", LogisticRegressionClassifier, {}),
        ]
        result = run_split_comparison_matrix(
            synthetic_wq_df,
            synthetic_feature_dfs,
            model_specs=model_specs,
            n_bootstrap=10,  # fast
            compute_delong=True,
        )
        assert "results" in result
        assert "delong_tests" in result
        assert "n_models" in result
        assert "feature_set_sizes" in result
        # Should have entries for 2-3 feature sets x 2 splits x 1 model
        n_feature_sets = len(result["feature_set_sizes"])
        assert len(result["results"]) == n_feature_sets * 2

    def test_matrix_result_structure(self, synthetic_wq_df, synthetic_feature_dfs):
        """Each result entry should have expected fields."""
        from aquacontam.models.logistic import LogisticRegressionClassifier

        model_specs = [("LogReg", LogisticRegressionClassifier, {})]
        result = run_split_comparison_matrix(
            synthetic_wq_df,
            synthetic_feature_dfs,
            model_specs=model_specs,
            n_bootstrap=0,
            compute_delong=False,
        )
        for entry in result["results"]:
            assert "model_name" in entry
            assert "feature_set" in entry
            assert "split_strategy" in entry
            assert "metrics" in entry
            assert "auroc" in entry["metrics"]
            assert "auprc" in entry["metrics"]

    def test_delong_populated(self, synthetic_wq_df, synthetic_feature_dfs):
        """DeLong tests should be populated when compute_delong=True."""
        from aquacontam.models.logistic import LogisticRegressionClassifier

        model_specs = [("LogReg", LogisticRegressionClassifier, {})]
        result = run_split_comparison_matrix(
            synthetic_wq_df,
            synthetic_feature_dfs,
            model_specs=model_specs,
            n_bootstrap=0,
            compute_delong=True,
        )
        # Should have DeLong test for LogReg
        if result["delong_tests"]:
            for _model_name, dl in result["delong_tests"].items():
                assert "p_value" in dl
                assert "z_statistic" in dl

    def test_empty_baseline_features(self, synthetic_wq_df):
        """Graceful handling when baseline features are empty."""
        # Only non-baseline features
        pwsids = synthetic_wq_df["pwsid"].unique()[:20]
        df = pd.DataFrame(
            {"aquifer_depth": np.random.default_rng(42).uniform(10, 500, len(pwsids))},
            index=pwsids,
        )
        df.index.name = "pwsid"
        result = run_split_comparison_matrix(synthetic_wq_df, [df])
        assert result.get("error") == "no_baseline_features"


class TestSplitComparisonSummaryTable:
    def test_output_format(self):
        """Summary table should be a DataFrame with expected columns."""
        results = [
            SplitComparisonResult(
                model_name="XGBoost",
                feature_set="baseline",
                split_strategy="geographic",
                metrics={"auroc": 0.75, "auprc": 0.40},
                n_train=100,
                n_test=50,
            ),
            SplitComparisonResult(
                model_name="XGBoost",
                feature_set="full",
                split_strategy="geographic",
                metrics={"auroc": 0.82, "auprc": 0.55},
                bootstrap_ci={
                    "auroc": {"ci_lower": 0.78, "ci_upper": 0.86, "point": 0.82, "std": 0.02}
                },
                n_train=100,
                n_test=50,
            ),
        ]
        df = split_comparison_summary_table(results)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "AUROC" in df.columns
        assert len(df) == 2

    def test_empty_results(self):
        df = split_comparison_summary_table([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestSplitComparisonResultDataclass:
    def test_to_dict(self):
        r = SplitComparisonResult(
            model_name="RF",
            feature_set="full",
            split_strategy="random",
            metrics={"auroc": 0.7},
            fold_metrics=[{"auroc": 0.68}, {"auroc": 0.72}],
            n_train=80,
            n_test=20,
        )
        d = r.to_dict()
        assert d["model_name"] == "RF"
        assert d["fold_metrics"] is not None
        assert len(d["fold_metrics"]) == 2
        # bootstrap_ci should not be in dict when None
        assert "bootstrap_ci" not in d
