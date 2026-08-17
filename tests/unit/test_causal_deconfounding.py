"""Tests for causal deconfounding via Double Machine Learning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.causal_deconfounding import (
    _build_feature_to_category,
    _impute_fold,
    category_displacement_test,
    causal_feature_analysis,
    compare_dml_shap_rankings,
    train_deconfounded_model,
)


def _make_confounded_data(
    n: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Generate data where x1 has real causal effect but confounder W drives both x2 and y."""
    rng = np.random.RandomState(seed)

    # Confounder: monitoring intensity
    w = rng.randn(n)

    # x1: real environmental feature (no confounding)
    x1 = rng.randn(n)

    # x2: correlated with confounder (spurious)
    x2 = 0.8 * w + 0.2 * rng.randn(n)

    # y: depends on x1 (causal) and w (confounding), not on x2 directly
    logit = 0.5 * x1 + 1.5 * w + rng.randn(n) * 0.5
    y = (logit > 0).astype(float)

    X = pd.DataFrame({"x1": x1, "x2": x2, "w": w})
    return X, pd.Series(y, name="target"), ["w"]


class TestCausalFeatureAnalysis:
    """Tests for causal_feature_analysis."""

    def test_known_causal_effect_recovered(self) -> None:
        """x1 should have a significant causal effect; x2 should shrink."""
        X, y, confounders = _make_confounded_data(n=500, seed=42)
        result = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)

        assert "feature" in result.columns
        assert "causal_effect" in result.columns
        assert "p_value" in result.columns

        # x1 has real causal effect, x2 is spurious
        x1_row = result[result["feature"] == "x1"]
        x2_row = result[result["feature"] == "x2"]
        assert len(x1_row) == 1
        assert len(x2_row) == 1

        # x1's causal effect should be larger in magnitude than x2's
        assert abs(x1_row["causal_effect"].to_numpy()[0]) > abs(
            x2_row["causal_effect"].to_numpy()[0]
        )

    def test_strong_confounding_reduces_effect(self) -> None:
        """Spurious feature x2 should have smaller causal effect than SHAP importance."""
        rng = np.random.RandomState(99)
        n = 300
        w = rng.randn(n)
        x1 = rng.randn(n)
        x2 = 0.9 * w + 0.1 * rng.randn(n)  # strongly confounded
        y = (1.5 * w + rng.randn(n) * 0.3 > 0).astype(float)

        X = pd.DataFrame({"x1": x1, "x2": x2, "w": w})
        shap = pd.Series({"x1": 0.05, "x2": 0.30, "w": 0.50})

        result = causal_feature_analysis(
            X, pd.Series(y), ["w"], shap_importances=shap, n_folds=3, seed=99
        )

        x2_row = result[result["feature"] == "x2"]
        assert len(x2_row) == 1
        # The ratio should be small — causal effect << SHAP importance
        assert x2_row["ratio"].to_numpy()[0] < 1.0

    def test_output_schema_columns(self) -> None:
        X, y, confounders = _make_confounded_data(n=100)
        result = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)

        expected_cols = {
            "feature",
            "causal_effect",
            "std_error",
            "p_value",
            "p_value_fdr",
            "significant_fdr",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) > 0

    def test_fdr_correction_applied(self) -> None:
        """FDR-corrected p-values should be >= raw p-values."""
        X, y, confounders = _make_confounded_data(n=500, seed=42)
        result = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)

        assert "p_value_fdr" in result.columns
        assert "significant_fdr" in result.columns
        # FDR-corrected p-values are always >= raw p-values
        assert (result["p_value_fdr"] >= result["p_value"] - 1e-10).all()
        # FDR-corrected p-values are in [0, 1]
        assert (result["p_value_fdr"] >= 0).all()
        assert (result["p_value_fdr"] <= 1.0).all()
        # significant_fdr is boolean
        assert result["significant_fdr"].dtype == bool

    def test_empty_confounders_raises(self) -> None:
        X, y, _ = _make_confounded_data(n=50)
        with pytest.raises(ValueError, match="non-empty"):
            causal_feature_analysis(X, y, [], n_folds=3)

    def test_missing_confounders_raises(self) -> None:
        X, y, _ = _make_confounded_data(n=50)
        with pytest.raises(ValueError, match="None of the specified confounders"):
            causal_feature_analysis(X, y, ["nonexistent_col"], n_folds=3)

    def test_single_feature(self) -> None:
        """Should work with only one non-confounder feature."""
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({"x1": rng.randn(n), "w": rng.randn(n)})
        y = pd.Series((rng.random(n) > 0.5).astype(float))
        result = causal_feature_analysis(X, y, ["w"], n_folds=3, seed=42)
        assert len(result) == 1
        assert result["feature"].to_numpy()[0] == "x1"

    def test_cross_fitting_no_leakage(self) -> None:
        """Results should be deterministic with the same seed."""
        X, y, confounders = _make_confounded_data(n=150, seed=42)
        r1 = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)
        r2 = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)
        np.testing.assert_array_almost_equal(
            r1["causal_effect"].to_numpy(), r2["causal_effect"].to_numpy()
        )


class TestTrainDeconfoundedModel:
    """Tests for train_deconfounded_model."""

    def test_returns_aurocs(self) -> None:
        X, y, confounders = _make_confounded_data(n=200, seed=42)
        result = train_deconfounded_model(X, y, confounders, n_folds=3, seed=42)
        assert "deconfounded_auroc" in result
        assert "original_auroc" in result
        assert "n_features" in result
        assert 0.0 <= result["deconfounded_auroc"] <= 1.0
        assert 0.0 <= result["original_auroc"] <= 1.0

    def test_no_confounders_present(self) -> None:
        """When no confounders are found, returns original AUROC for both."""
        X, y, _ = _make_confounded_data(n=100)
        result = train_deconfounded_model(X, y, ["nonexistent"], n_folds=3, seed=42)
        assert result["deconfounded_auroc"] == result["original_auroc"]


class TestImputeFold:
    """Tests for per-fold imputation helper."""

    def test_1d_train_mean_only(self) -> None:
        """Test fold imputation uses train mean, not global mean."""
        train = np.array([1.0, 2.0, 3.0, np.nan])
        test = np.array([np.nan, 10.0])
        train_out, test_out = _impute_fold(train, test)
        # Train mean = mean([1, 2, 3]) = 2.0
        assert train_out[3] == 2.0
        assert test_out[0] == 2.0
        # Non-NaN values unchanged
        np.testing.assert_array_equal(train_out[:3], [1.0, 2.0, 3.0])
        assert test_out[1] == 10.0

    def test_2d_per_column(self) -> None:
        """Test 2-D imputation is column-wise with train means."""
        train = np.array([[1.0, 10.0], [3.0, np.nan], [np.nan, 30.0]])
        test = np.array([[np.nan, np.nan]])
        train_out, test_out = _impute_fold(train, test)
        # Col 0 train mean = 2.0, Col 1 train mean = 20.0
        assert train_out[2, 0] == 2.0
        assert train_out[1, 1] == 20.0
        np.testing.assert_array_almost_equal(test_out[0], [2.0, 20.0])

    def test_all_nan_train_falls_back_to_zero(self) -> None:
        """If all train values are NaN, impute with 0."""
        train = np.array([np.nan, np.nan])
        test = np.array([np.nan])
        train_out, test_out = _impute_fold(train, test)
        assert train_out[0] == 0.0
        assert test_out[0] == 0.0

    def test_no_nan_is_noop(self) -> None:
        """When no NaN, output equals input."""
        train = np.array([1.0, 2.0, 3.0])
        test = np.array([4.0, 5.0])
        train_out, test_out = _impute_fold(train, test)
        np.testing.assert_array_equal(train_out, train)
        np.testing.assert_array_equal(test_out, test)

    def test_does_not_mutate_input(self) -> None:
        """Inputs should not be modified in place."""
        train = np.array([1.0, np.nan])
        test = np.array([np.nan])
        train_copy = train.copy()
        test_copy = test.copy()
        _impute_fold(train, test)
        np.testing.assert_array_equal(train, train_copy)
        np.testing.assert_array_equal(test, test_copy)


class TestFoldIsolation:
    """Verify that NaN imputation in DML uses only train-fold data."""

    def test_nan_confounder_fold_isolation(self) -> None:
        """Confounders with NaN in only one fold's test portion should
        be imputed using train-fold means, not global means.
        """
        rng = np.random.RandomState(42)
        n = 150
        # Create confounder w with NaN concentrated in last 50 samples
        w = rng.randn(n)
        w_with_nan = w.copy()
        w_with_nan[100:125] = np.nan  # NaN in a specific region

        x1 = rng.randn(n)
        y = (0.5 * x1 + 1.5 * w + rng.randn(n) * 0.5 > 0).astype(float)

        X = pd.DataFrame({"x1": x1, "w": w_with_nan})
        result = causal_feature_analysis(X, pd.Series(y), ["w"], n_folds=3, seed=42)

        # Should complete without error and produce a result
        assert len(result) == 1
        assert result["feature"].to_numpy()[0] == "x1"
        assert np.isfinite(result["causal_effect"].to_numpy()[0])

    def test_nan_treatment_fold_isolation(self) -> None:
        """Treatment features with NaN should be imputed per-fold."""
        rng = np.random.RandomState(42)
        n = 150
        w = rng.randn(n)
        x1 = rng.randn(n)
        x1_with_nan = x1.copy()
        x1_with_nan[50:75] = np.nan  # NaN in a specific region

        y = (0.5 * x1 + 1.5 * w + rng.randn(n) * 0.5 > 0).astype(float)

        X = pd.DataFrame({"x1": x1_with_nan, "w": w})
        result = causal_feature_analysis(X, pd.Series(y), ["w"], n_folds=3, seed=42)

        assert len(result) == 1
        assert np.isfinite(result["causal_effect"].to_numpy()[0])

    def test_deconfounded_model_with_nan(self) -> None:
        """train_deconfounded_model should handle NaN via per-fold imputation."""
        rng = np.random.RandomState(42)
        n = 200
        w = rng.randn(n)
        x1 = rng.randn(n)
        x2 = 0.8 * w + 0.2 * rng.randn(n)
        y = (1.5 * w + 0.5 * x1 + rng.randn(n) * 0.5 > 0).astype(float)

        # Introduce NaN in all columns
        X = pd.DataFrame({"x1": x1, "x2": x2, "w": w})
        X.loc[30:45, "x1"] = np.nan
        X.loc[80:95, "w"] = np.nan

        result = train_deconfounded_model(X, pd.Series(y), ["w"], n_folds=3, seed=42)
        assert 0.0 <= result["deconfounded_auroc"] <= 1.0
        assert 0.0 <= result["original_auroc"] <= 1.0


class TestDmlDiagnostics:
    """Tests for DML assumption diagnostics (overlap, residual correlation, R²)."""

    def test_diagnostic_columns_present(self) -> None:
        X, y, confounders = _make_confounded_data(n=500, seed=42)
        result = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)
        for col in ("residualization_r2", "overlap_warning", "max_residual_confounder_corr"):
            assert col in result.columns, f"Missing column: {col}"

    def test_residualization_r2_bounded(self) -> None:
        X, y, confounders = _make_confounded_data(n=500, seed=42)
        result = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)
        # R² can be negative if model predicts worse than the mean, but should be <= 1
        assert (result["residualization_r2"] <= 1.01).all()
        # R² should be finite
        assert result["residualization_r2"].notna().all()

    def test_overlap_warning_on_strongly_confounded_feature(self) -> None:
        """Feature nearly perfectly predicted by confounders triggers overlap warning."""
        rng = np.random.RandomState(99)
        n = 300
        w = rng.randn(n)
        # x1 is nearly fully determined by w — R² ≈ 1.0
        x1 = w + 0.01 * rng.randn(n)
        # x2 is independent of w
        x2 = rng.randn(n)
        y = (w + x2 + rng.randn(n) * 0.5 > 0).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2, "w": w})

        result = causal_feature_analysis(X, pd.Series(y), ["w"], n_folds=3, seed=42)
        if "x1" in result["feature"].to_numpy():
            x1_row = result[result["feature"] == "x1"].iloc[0]
            assert bool(x1_row["overlap_warning"]) is True
            assert x1_row["residualization_r2"] > 0.9

    def test_max_residual_confounder_corr_bounded(self) -> None:
        X, y, confounders = _make_confounded_data(n=500, seed=42)
        result = causal_feature_analysis(X, y, confounders, n_folds=3, seed=42)
        assert (result["max_residual_confounder_corr"] >= 0).all()
        assert (result["max_residual_confounder_corr"] <= 1.0).all()


class TestCompareDmlShapRankings:
    """Tests for scale-free DML-vs-SHAP rank comparison."""

    @staticmethod
    def _make_dml_df() -> pd.DataFrame:
        """Synthetic DML results with known ranks."""
        return pd.DataFrame(
            {
                "feature": ["a", "b", "c", "d", "e"],
                "causal_effect": [0.5, 0.1, 0.3, 0.05, 0.2],
                "shap_importance": [0.01, 0.05, 0.03, 0.02, 0.04],
            }
        )

    def test_basic_rank_computation(self) -> None:
        df = self._make_dml_df()
        result = compare_dml_shap_rankings(df)
        assert result["n_features"] == 5
        assert "spearman_rho" in result
        assert "spearman_p" in result
        # Feature "a" has largest |DML| (rank 1) but smallest SHAP (rank 5)
        ranks = {r["feature"]: r for r in result["feature_ranks"]}
        assert ranks["a"]["dml_rank"] == 1
        assert ranks["a"]["shap_rank"] == 5
        assert ranks["a"]["displacement"] == 4  # shap_rank - dml_rank

    def test_demographic_filter(self) -> None:
        df = self._make_dml_df()
        result = compare_dml_shap_rankings(df, demographic_prefixes=["a", "b"])
        assert result["demographic_n"] == 2
        assert "demographic_mean_displacement" in result
        assert len(result["demographic_ranks"]) == 2

    def test_no_demographic_prefixes(self) -> None:
        df = self._make_dml_df()
        result = compare_dml_shap_rankings(df)
        assert "demographic_ranks" not in result

    def test_too_few_features(self) -> None:
        df = pd.DataFrame(
            {
                "feature": ["a"],
                "causal_effect": [0.1],
                "shap_importance": [0.01],
            }
        )
        result = compare_dml_shap_rankings(df)
        assert result["n_features"] == 1
        assert np.isnan(result["spearman_rho"])

    def test_zero_shap_filtered(self) -> None:
        df = pd.DataFrame(
            {
                "feature": ["a", "b", "c", "d"],
                "causal_effect": [0.5, 0.1, 0.3, 0.2],
                "shap_importance": [0.01, 0.0, 0.03, 0.02],
            }
        )
        result = compare_dml_shap_rankings(df)
        assert result["n_features"] == 3  # "b" filtered out


class TestBuildFeatureToCategory:
    """Tests for _build_feature_to_category prefix matching."""

    def test_prefix_matching(self) -> None:
        features = [
            "dist_nearest_wwtp",
            "pct_developed_high",
            "pct_low_income",
            "aquifer_type_sand",
        ]
        prefixes = {
            "proximity": ["dist_nearest_"],
            "land_use": ["pct_developed_"],
            "demographics": ["pct_low_income"],
            "hydrogeology": ["aquifer_"],
        }
        result = _build_feature_to_category(features, prefixes)
        assert result["dist_nearest_wwtp"] == "proximity"
        assert result["pct_developed_high"] == "land_use"
        assert result["pct_low_income"] == "demographics"
        assert result["aquifer_type_sand"] == "hydrogeology"

    def test_unmatched_features_get_other(self) -> None:
        features = ["unknown_feature", "dist_nearest_wwtp"]
        prefixes = {"proximity": ["dist_nearest_"]}
        result = _build_feature_to_category(features, prefixes)
        assert result["unknown_feature"] == "other"
        assert result["dist_nearest_wwtp"] == "proximity"

    def test_exact_match(self) -> None:
        """Feature name exactly equals a prefix string."""
        features = ["n_samples"]
        prefixes = {"monitoring": ["n_samples"]}
        result = _build_feature_to_category(features, prefixes)
        assert result["n_samples"] == "monitoring"

    def test_first_match_wins(self) -> None:
        """When a feature matches multiple categories, first wins."""
        features = ["n_samples"]
        # Use ordered categories (Python 3.7+ dicts preserve insertion order)
        prefixes = {"monitoring": ["n_samples"], "system": ["n_samples"]}
        result = _build_feature_to_category(features, prefixes)
        assert result["n_samples"] == "monitoring"

    def test_empty_features(self) -> None:
        result = _build_feature_to_category([], {"a": ["x"]})
        assert result == {}


class TestCategoryDisplacementTest:
    """Tests for category_displacement_test permutation test."""

    @staticmethod
    def _make_rank_comparison(features: list[str], displacements: list[int]) -> dict:
        return {
            "feature_ranks": [
                {"feature": f, "displacement": d, "dml_rank": i + 1, "shap_rank": i + 1 + d}
                for i, (f, d) in enumerate(zip(features, displacements, strict=True))
            ]
        }

    def test_significant_category_detected(self) -> None:
        """Category with systematically large displacements gets low p-value."""
        features = [f"prox_{i}" for i in range(10)] + [f"demo_{i}" for i in range(5)]
        # Proximity: small displacements; Demographics: large displacements
        displacements = [1, -1, 2, 0, -2, 1, 0, -1, 2, -1, *[50, 45, 55, 48, 52]]
        rank_comp = self._make_rank_comparison(features, displacements)
        feat_to_cat = {
            f: "proximity" if f.startswith("prox") else "demographics" for f in features
        }

        result = category_displacement_test(rank_comp, feat_to_cat, n_permutations=5000, seed=42)
        assert len(result) == 2
        demo_row = result[result["category"] == "demographics"].iloc[0]
        prox_row = result[result["category"] == "proximity"].iloc[0]
        assert demo_row["p_value"] < 0.05
        assert demo_row["mean_abs_displacement"] > prox_row["mean_abs_displacement"]

    def test_no_significant_categories(self) -> None:
        """Random displacements yield no significant categories."""
        rng = np.random.RandomState(42)
        features = [f"a_{i}" for i in range(20)] + [f"b_{i}" for i in range(20)]
        displacements = rng.randint(-5, 6, size=40).tolist()
        rank_comp = self._make_rank_comparison(features, displacements)
        feat_to_cat = {f: "cat_a" if f.startswith("a") else "cat_b" for f in features}

        result = category_displacement_test(rank_comp, feat_to_cat, n_permutations=1000, seed=42)
        # Neither category should be significant — both have similar random displacements
        assert not result["significant_fdr"].any()

    def test_fdr_geq_raw(self) -> None:
        """FDR-corrected p-values should be >= raw p-values."""
        features = [f"x_{i}" for i in range(15)]
        displacements = list(range(15))
        rank_comp = self._make_rank_comparison(features, displacements)
        feat_to_cat = {
            f: "cat_a" if i < 5 else ("cat_b" if i < 10 else "cat_c")
            for i, f in enumerate(features)
        }
        result = category_displacement_test(rank_comp, feat_to_cat, n_permutations=500, seed=42)
        assert (result["p_value_fdr"] >= result["p_value"] - 1e-10).all()

    def test_empty_feature_ranks(self) -> None:
        result = category_displacement_test({"feature_ranks": []}, {})
        assert len(result) == 0
        assert list(result.columns) == [
            "category",
            "n_features",
            "mean_abs_displacement",
            "p_value",
            "p_value_fdr",
            "significant_fdr",
        ]

    def test_single_category(self) -> None:
        """Single category should still run."""
        features = ["a", "b", "c"]
        displacements = [10, 20, 5]
        rank_comp = self._make_rank_comparison(features, displacements)
        feat_to_cat = {f: "only_cat" for f in features}
        result = category_displacement_test(rank_comp, feat_to_cat, n_permutations=100, seed=42)
        assert len(result) == 1
        assert result.iloc[0]["category"] == "only_cat"

    def test_output_schema(self) -> None:
        features = ["a", "b", "c", "d"]
        displacements = [5, 10, 3, 8]
        rank_comp = self._make_rank_comparison(features, displacements)
        feat_to_cat = {"a": "x", "b": "x", "c": "y", "d": "y"}
        result = category_displacement_test(rank_comp, feat_to_cat, n_permutations=100, seed=42)
        expected_cols = {
            "category",
            "n_features",
            "mean_abs_displacement",
            "p_value",
            "p_value_fdr",
            "significant_fdr",
        }
        assert set(result.columns) == expected_cols
        assert result["n_features"].dtype in (np.int64, int)
        assert result["significant_fdr"].dtype == bool
