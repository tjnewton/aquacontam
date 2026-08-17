"""Tests for feature ablation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.ablation import (
    DETECTION_ONLY_SOURCES,
    AblationResult,
    _columns_matching_prefixes,
    ablation_summary,
    evaluate_excluding_detection_only,
    evaluate_loro_fold_excluding_detection_only,
    run_feature_ablation,
)
from aquacontam.models.random_forest import RandomForestClassifier


@pytest.fixture()
def ablation_data() -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """Create synthetic data with named feature categories."""
    rng = np.random.RandomState(42)
    n_train, n_val, n_test = 100, 30, 50

    def _make_split(n: int) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame(
            {
                "nearest_industrial_km": rng.uniform(0, 50, n),
                "nearest_military_km": rng.uniform(0, 100, n),
                "count_industrial_10km": rng.randint(0, 10, n),
                "aquifer_type_Glacial": rng.randint(0, 2, n),
                "aquifer_type_Basalt": rng.randint(0, 2, n),
                "aquifer_lithology_sand": rng.randint(0, 2, n),
                "frac_developed_low": rng.uniform(0, 1, n),
                "pct_people_of_color": rng.uniform(0, 100, n),
                "generic_feature": rng.uniform(0, 1, n),
            }
        )
        y = pd.Series(rng.randint(0, 2, n), name="target")
        return X, y

    X_train, y_train = _make_split(n_train)
    X_val, y_val = _make_split(n_val)
    X_test, y_test = _make_split(n_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestColumnsMatchingPrefixes:
    """Tests for _columns_matching_prefixes."""

    def test_matches_prefix(self) -> None:
        df = pd.DataFrame({"aquifer_type_A": [1], "aquifer_type_B": [1], "other": [1]})
        matched = _columns_matching_prefixes(df, ["aquifer_type_"])
        assert set(matched) == {"aquifer_type_A", "aquifer_type_B"}

    def test_no_match(self) -> None:
        df = pd.DataFrame({"col_a": [1], "col_b": [1]})
        matched = _columns_matching_prefixes(df, ["zzz_"])
        assert matched == []

    def test_exact_match(self) -> None:
        df = pd.DataFrame({"target": [1], "feature": [1]})
        matched = _columns_matching_prefixes(df, ["target"])
        assert matched == ["target"]


class TestRunFeatureAblation:
    """Tests for run_feature_ablation."""

    def test_returns_full_baseline(self, ablation_data: tuple) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test = ablation_data
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={},
        )
        assert len(results) == 1
        assert results[0].category == "full"
        assert results[0].n_features_dropped == 0
        assert "auroc" in results[0].metrics or "accuracy" in results[0].metrics

    def test_drops_category(self, ablation_data: tuple) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test = ablation_data
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={
                "aquifer_type_only": ["aquifer_type_"],
            },
        )
        assert len(results) == 2
        ablation_result = results[1]
        assert ablation_result.category == "aquifer_type_only"
        assert ablation_result.n_features_dropped == 2  # Glacial + Basalt
        assert ablation_result.n_features_remaining == 7
        assert "auroc" in ablation_result.metrics or "accuracy" in ablation_result.metrics

    def test_multiple_categories(self, ablation_data: tuple) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test = ablation_data
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={
                "proximity": ["nearest_", "count_"],
                "aquifer_type_only": ["aquifer_type_"],
            },
        )
        # full + proximity + aquifer_type_only
        assert len(results) == 3
        cats = [r.category for r in results]
        assert "full" in cats
        assert "proximity" in cats
        assert "aquifer_type_only" in cats

    def test_single_class_returns_empty(self, ablation_data: tuple) -> None:
        """When y_train has only one class, run_feature_ablation returns empty."""
        X_train, _, X_val, y_val, X_test, y_test = ablation_data
        y_single = pd.Series(np.ones(len(X_train), dtype=int), name="target")
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_single,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={"proximity": ["nearest_"]},
        )
        assert len(results) == 0

    def test_nonexistent_category_skipped(self, ablation_data: tuple) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test = ablation_data
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={"nonexistent": ["zzz_"]},
        )
        # Only full baseline
        assert len(results) == 1

    def test_significance_tests_present(self, ablation_data: tuple) -> None:
        """Ablated results should include paired bootstrap significance tests."""
        X_train, y_train, X_val, y_val, X_test, y_test = ablation_data
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={"proximity": ["nearest_", "count_"]},
            n_bootstrap=100,
            seed=42,
        )
        assert len(results) == 2
        # Full baseline has no significance (nothing to compare against)
        assert results[0].category == "full"
        assert results[0].significance == {}
        # Ablated result has significance for auroc and auprc
        ablated = results[1]
        assert "auroc" in ablated.significance
        assert "auprc" in ablated.significance
        for metric_name in ("auroc", "auprc"):
            sig = ablated.significance[metric_name]
            assert "delta" in sig
            assert "p_value" in sig
            assert "ci_lower" in sig
            assert "ci_upper" in sig
            assert 0.0 <= sig["p_value"] <= 1.0

    def test_significance_in_summary(self, ablation_data: tuple) -> None:
        """ablation_summary should include p-value and CI columns."""
        X_train, y_train, X_val, y_val, X_test, y_test = ablation_data
        results = run_feature_ablation(
            RandomForestClassifier,
            {"n_estimators": 10, "random_state": 42},
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_categories={"proximity": ["nearest_", "count_"]},
            n_bootstrap=100,
            seed=42,
        )
        df = ablation_summary(results)
        assert "auroc_p_value" in df.columns
        assert "auroc_ci_lower" in df.columns
        assert "auroc_ci_upper" in df.columns


class TestAblationSummary:
    """Tests for ablation_summary."""

    def test_creates_dataframe(self) -> None:
        results = [
            AblationResult(
                category="full",
                n_features_dropped=0,
                metrics={"auroc": 0.8, "auprc": 0.6},
                n_features_remaining=10,
            ),
            AblationResult(
                category="aquifer_type",
                n_features_dropped=3,
                metrics={"auroc": 0.75, "auprc": 0.55},
                n_features_remaining=7,
            ),
        ]
        df = ablation_summary(results)
        assert len(df) == 2
        assert "category" in df.columns
        assert "auroc" in df.columns
        assert "auroc_delta" in df.columns

    def test_deltas_computed(self) -> None:
        results = [
            AblationResult(
                category="full",
                metrics={"auroc": 0.8},
                n_features_remaining=10,
            ),
            AblationResult(
                category="test",
                n_features_dropped=2,
                metrics={"auroc": 0.7},
                n_features_remaining=8,
            ),
        ]
        df = ablation_summary(results)
        assert df.loc[df["category"] == "full", "auroc_delta"].to_numpy()[0] == pytest.approx(0.0)
        assert df.loc[df["category"] == "test", "auroc_delta"].to_numpy()[0] == pytest.approx(-0.1)

    def test_empty_results(self) -> None:
        df = ablation_summary([])
        assert df.empty

    def test_fdr_correction_columns_present(self) -> None:
        """ablation_summary should include FDR-corrected p-value columns."""
        results = [
            AblationResult(
                category="full",
                metrics={"auroc": 0.85, "auprc": 0.65},
                n_features_remaining=10,
            ),
            AblationResult(
                category="prox",
                n_features_dropped=3,
                metrics={"auroc": 0.80, "auprc": 0.60},
                n_features_remaining=7,
                significance={
                    "auroc": {"p_value": 0.01, "ci_lower": -0.08, "ci_upper": -0.02},
                    "auprc": {"p_value": 0.03, "ci_lower": -0.08, "ci_upper": -0.01},
                },
            ),
            AblationResult(
                category="hydro",
                n_features_dropped=2,
                metrics={"auroc": 0.82, "auprc": 0.62},
                n_features_remaining=8,
                significance={
                    "auroc": {"p_value": 0.04, "ci_lower": -0.05, "ci_upper": 0.01},
                    "auprc": {"p_value": 0.06, "ci_lower": -0.05, "ci_upper": 0.02},
                },
            ),
        ]
        df = ablation_summary(results)
        assert "auroc_p_value_fdr" in df.columns
        assert "auroc_significant_fdr" in df.columns
        assert "auprc_p_value_fdr" in df.columns
        assert "auprc_significant_fdr" in df.columns

        # FDR-corrected p-values should be >= raw p-values
        non_full = df[df["category"] != "full"]
        for metric in ("auroc", "auprc"):
            raw = non_full[f"{metric}_p_value"].to_numpy()
            fdr = non_full[f"{metric}_p_value_fdr"].to_numpy()
            for r, f in zip(raw, fdr, strict=True):
                if not np.isnan(r):
                    assert f >= r - 1e-10

        # Full row should have NaN FDR
        full_row = df[df["category"] == "full"]
        assert np.isnan(full_row["auroc_p_value_fdr"].to_numpy()[0])


class TestDetectionOnlyAblation:
    """Tests for evaluate_excluding_detection_only."""

    def test_basic_exclusion(self) -> None:
        """Excluding detection-only sources produces valid metrics."""
        rng = np.random.RandomState(42)
        n = 200
        y_true = pd.Series(rng.randint(0, 2, n))
        y_prob = pd.Series(rng.uniform(0, 1, n))
        sources = pd.Series(rng.choice(["ucmr5", "ucmr3", "wa_doh", "sdwis", "mi_mpart"], n))

        result = evaluate_excluding_detection_only(y_true, y_prob, sources)

        assert "all_sources" in result
        assert "excluding_detection_only" in result
        assert "delta" in result
        assert result["n_filtered"] < result["n_all"]
        assert result["n_excluded"] > 0
        # Metrics should have AUROC key
        assert "auroc" in result["all_sources"]
        assert "auroc" in result["excluding_detection_only"]

    def test_no_detection_only_sources(self) -> None:
        """When no detection-only sources are present, results should match."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = pd.Series(rng.randint(0, 2, n))
        y_prob = pd.Series(rng.uniform(0, 1, n))
        sources = pd.Series(["ucmr5"] * n)

        result = evaluate_excluding_detection_only(y_true, y_prob, sources)

        assert result["n_filtered"] == result["n_all"]
        assert result["n_excluded"] == 0

    def test_all_detection_only_returns_empty(self) -> None:
        """When all sources are detection-only, filtered metrics should be empty."""
        rng = np.random.RandomState(42)
        n = 50
        y_true = pd.Series(rng.randint(0, 2, n))
        y_prob = pd.Series(rng.uniform(0, 1, n))
        sources = pd.Series(["wa_doh"] * n)

        result = evaluate_excluding_detection_only(y_true, y_prob, sources)

        assert result["excluding_detection_only"] == {}

    def test_detection_only_sources_constant(self) -> None:
        """DETECTION_ONLY_SOURCES should contain the known detection-only sources."""
        assert {"mi_mpart", "sdwis", "wa_doh"} == DETECTION_ONLY_SOURCES


class TestLORODetectionOnlyAblation:
    """Tests for evaluate_loro_fold_excluding_detection_only."""

    def test_basic_fold_with_detection_only(self) -> None:
        """Fold containing detection-only sources shows reduced n_filtered."""
        rng = np.random.RandomState(42)
        n = 200
        y_true = pd.Series(rng.randint(0, 2, n))
        y_prob = pd.Series(rng.uniform(0, 1, n))
        sources = pd.Series(rng.choice(["ucmr5", "wa_doh"], n, p=[0.7, 0.3]))

        result = evaluate_loro_fold_excluding_detection_only(
            y_true, y_prob, sources, test_region=10
        )

        assert result["test_region"] == 10
        assert result["n_excluded"] > 0
        assert "wa_doh" in result["excluded_sources"]
        assert "auroc" in result["all_sources"]
        assert "auroc" in result["excluding_detection_only"]
        assert "wa_doh" in result["n_detection_only_by_source"]
        assert result["n_detection_only_by_source"]["wa_doh"] > 0

    def test_fold_without_detection_only(self) -> None:
        """Fold with no detection-only sources has identical metrics."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = pd.Series(rng.randint(0, 2, n))
        y_prob = pd.Series(rng.uniform(0, 1, n))
        sources = pd.Series(["ucmr5"] * n)

        result = evaluate_loro_fold_excluding_detection_only(
            y_true, y_prob, sources, test_region=3
        )

        assert result["test_region"] == 3
        assert result["n_excluded"] == 0
        assert result["n_filtered"] == result["n_all"]
        assert result["n_detection_only_by_source"] == {}

    def test_single_class_after_exclusion(self) -> None:
        """If only one class remains after exclusion, filtered metrics are empty."""
        # All positive samples from wa_doh, negatives from ucmr5
        y_true = pd.Series([1] * 50 + [0] * 50)
        y_prob = pd.Series([0.9] * 50 + [0.1] * 50)
        sources = pd.Series(["wa_doh"] * 50 + ["ucmr5"] * 50)

        result = evaluate_loro_fold_excluding_detection_only(
            y_true, y_prob, sources, test_region=10
        )

        assert result["test_region"] == 10
        assert result["excluding_detection_only"] == {}
        assert result["n_detection_only_by_source"]["wa_doh"] == 50

    def test_region_metadata_present(self) -> None:
        """Result contains test_region and n_detection_only_by_source."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = pd.Series(rng.randint(0, 2, n))
        y_prob = pd.Series(rng.uniform(0, 1, n))
        sources = pd.Series(["ucmr5"] * 60 + ["wa_doh"] * 30 + ["sdwis"] * 10)

        result = evaluate_loro_fold_excluding_detection_only(
            y_true, y_prob, sources, test_region=10
        )

        assert "test_region" in result
        assert "n_detection_only_by_source" in result
        assert "sdwis" in result["n_detection_only_by_source"]
        assert result["n_detection_only_by_source"]["sdwis"] == 10
        assert result["n_detection_only_by_source"]["wa_doh"] == 30
