"""Tests for per-region heterogeneity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.regional import (
    analyze_regional_heterogeneity,
    compare_feature_distributions,
    compute_missing_data_patterns,
    compute_region_detection_rates,
)


@pytest.fixture()
def regional_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create synthetic data with region assignments."""
    rng = np.random.RandomState(42)
    n = 200

    X = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.normal(5, 2, n),
            "feat_c": rng.exponential(1, n),
            "feat_d": rng.uniform(0, 1, n),
            "feat_e": rng.normal(0, 1, n),
        },
        index=[f"PWS{i:05d}" for i in range(n)],
    )

    # Inject some NaN values
    X.iloc[0:10, 2] = np.nan  # feat_c missing in first 10 rows
    X.iloc[50:55, 4] = np.nan  # feat_e missing in rows 50-55

    y = pd.Series(rng.randint(0, 2, n), index=X.index, name="detected")

    # Assign regions 1-10, with different sizes
    region_values = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n)
    regions = pd.Series(region_values, index=X.index, name="epa_region")

    return X, y, regions


class TestComputeRegionDetectionRates:
    """Tests for compute_region_detection_rates."""

    def test_basic(self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
        _, y, regions = regional_data
        result = compute_region_detection_rates(y, regions)

        # Should have entries for each unique region
        assert len(result) > 0
        for region, info in result.items():
            assert isinstance(region, int)
            assert "n_samples" in info
            assert "n_positive" in info
            assert "detection_rate" in info
            assert 0.0 <= info["detection_rate"] <= 1.0
            assert info["n_positive"] <= info["n_samples"]

    def test_all_positive(self) -> None:
        y = pd.Series([1, 1, 1, 1], index=["a", "b", "c", "d"])
        regions = pd.Series([1, 1, 2, 2], index=["a", "b", "c", "d"])
        result = compute_region_detection_rates(y, regions)
        assert result[1]["detection_rate"] == 1.0
        assert result[2]["detection_rate"] == 1.0

    def test_all_negative(self) -> None:
        y = pd.Series([0, 0, 0], index=["a", "b", "c"])
        regions = pd.Series([1, 1, 2], index=["a", "b", "c"])
        result = compute_region_detection_rates(y, regions)
        assert result[1]["detection_rate"] == 0.0
        assert result[2]["detection_rate"] == 0.0

    def test_nan_regions_skipped(self) -> None:
        y = pd.Series([1, 0, 1], index=["a", "b", "c"])
        regions = pd.Series([1, np.nan, 2], index=["a", "b", "c"])
        result = compute_region_detection_rates(y, regions)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result


class TestCompareFeatureDistributions:
    """Tests for compare_feature_distributions."""

    def test_basic(self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
        X, _, regions = regional_data
        result = compare_feature_distributions(
            X, regions, ["feat_a", "feat_b"], best_region=3, worst_region=4
        )
        assert len(result) == 2
        for entry in result:
            assert "feature" in entry
            assert "ks_statistic" in entry
            assert "p_value" in entry
            assert "mean_best" in entry
            assert "mean_worst" in entry
            assert "std_best" in entry
            assert "std_worst" in entry
            assert 0.0 <= entry["ks_statistic"] <= 1.0

    def test_missing_feature_skipped(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, _, regions = regional_data
        result = compare_feature_distributions(
            X, regions, ["feat_a", "nonexistent"], best_region=3, worst_region=4
        )
        assert len(result) == 1
        assert result[0]["feature"] == "feat_a"

    def test_empty_region_handled(self) -> None:
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
        regions = pd.Series([1, 1, 2], index=["a", "b", "c"])
        # Region 99 has no data - should produce empty result
        result = compare_feature_distributions(X, regions, ["f1"], best_region=1, worst_region=99)
        assert len(result) == 0

    def test_identical_distributions(self) -> None:
        rng = np.random.RandomState(0)
        vals = rng.normal(0, 1, 100)
        X = pd.DataFrame({"f1": np.tile(vals, 2)}, index=range(200))
        regions = pd.Series([1] * 100 + [2] * 100, index=range(200))
        result = compare_feature_distributions(X, regions, ["f1"], best_region=1, worst_region=2)
        assert len(result) == 1
        # KS test p-value should be high for identical distributions
        assert result[0]["p_value"] > 0.05


class TestComputeMissingDataPatterns:
    """Tests for compute_missing_data_patterns."""

    def test_basic(self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
        X, _, regions = regional_data
        result = compute_missing_data_patterns(X, regions)
        assert len(result) > 0
        for region, info in result.items():
            assert isinstance(region, int)
            assert "overall_missing_rate" in info
            assert 0.0 <= info["overall_missing_rate"] <= 1.0

    def test_no_missing(self) -> None:
        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]}, index=["a", "b", "c"])
        regions = pd.Series([1, 1, 2], index=["a", "b", "c"])
        result = compute_missing_data_patterns(X, regions)
        assert result[1]["overall_missing_rate"] == 0.0
        assert result[2]["overall_missing_rate"] == 0.0

    def test_all_missing(self) -> None:
        X = pd.DataFrame({"f1": [np.nan, np.nan], "f2": [np.nan, np.nan]}, index=["a", "b"])
        regions = pd.Series([1, 1], index=["a", "b"])
        result = compute_missing_data_patterns(X, regions)
        assert result[1]["overall_missing_rate"] == 1.0

    def test_top5_missing_features(self) -> None:
        # Create data with varying missing rates
        X = pd.DataFrame(
            {
                "f1": [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "f2": [np.nan, np.nan, 2, 3, 4, 5, 6, 7, 8, 9],
                "f3": [np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, 8, 9],
                "f4": [np.nan, np.nan, np.nan, np.nan, 4, 5, 6, 7, 8, 9],
                "f5": [np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9],
                "f6": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, 7, 8, 9],
                "f7": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # no missing
            },
            index=range(10),
        )
        regions = pd.Series([1] * 10, index=range(10))
        result = compute_missing_data_patterns(X, regions)
        # Should report top 5 most-missing features (f6, f5, f4, f3, f2)
        # f7 has no missing so shouldn't appear; f1 may or may not be in top 5
        info = result[1]
        assert "overall_missing_rate" in info
        # Top 5 should include the most-missing features
        feature_keys = [k for k in info if k != "overall_missing_rate"]
        assert len(feature_keys) <= 5


class TestAnalyzeRegionalHeterogeneity:
    """Tests for the main entry point."""

    def test_basic(self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
        X, y, regions = regional_data
        result = analyze_regional_heterogeneity(X, y, regions)
        assert "detection_rates" in result
        assert "feature_distributions" in result
        assert "missing_data" in result
        assert "best_region" in result
        assert "worst_region" in result
        assert "top_features" in result

    def test_custom_features(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, y, regions = regional_data
        result = analyze_regional_heterogeneity(X, y, regions, top_features=["feat_a", "feat_b"])
        assert result["top_features"] == ["feat_a", "feat_b"]

    def test_custom_regions(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, y, regions = regional_data
        result = analyze_regional_heterogeneity(X, y, regions, best_region=1, worst_region=2)
        assert result["best_region"] == 1
        assert result["worst_region"] == 2

    def test_default_top_features_by_variance(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, y, regions = regional_data
        result = analyze_regional_heterogeneity(X, y, regions)
        # Should pick top 5 by variance
        assert len(result["top_features"]) == 5

    def test_invalid_features_fallback(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, y, regions = regional_data
        result = analyze_regional_heterogeneity(
            X, y, regions, top_features=["nonexistent_1", "nonexistent_2"]
        )
        # Should fallback to variance-based features
        assert len(result["top_features"]) == 5
        # All features should exist in X
        for f in result["top_features"]:
            assert f in X.columns

    def test_json_serializable(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        """Result should be JSON-serializable (important for reproduce.py)."""
        import json

        X, y, regions = regional_data
        result = analyze_regional_heterogeneity(X, y, regions)
        # Should not raise
        json.dumps(result, default=str)


class TestFdrCorrection:
    """Tests for FDR correction in compare_feature_distributions."""

    def test_fdr_columns_present(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, _, regions = regional_data
        features = list(X.columns[:3])
        results = compare_feature_distributions(X, regions, features, 3, 4)
        assert len(results) > 0
        for r in results:
            assert "p_value_fdr" in r
            assert "significant_fdr" in r

    def test_fdr_corrected_ge_raw(
        self, regional_data: tuple[pd.DataFrame, pd.Series, pd.Series]
    ) -> None:
        X, _, regions = regional_data
        features = list(X.columns)
        results = compare_feature_distributions(X, regions, features, 3, 4)
        for r in results:
            if not np.isnan(r["p_value"]):
                assert r["p_value_fdr"] >= r["p_value"] - 1e-10
