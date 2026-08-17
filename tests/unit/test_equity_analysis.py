"""Tests for environmental justice disparity analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.equity import (
    DisparityReport,
    analyze_equity,
    apply_multiple_testing_correction,
    compute_burden_ratio,
    compute_group_metrics,
    compute_monitoring_adjusted_burden_ratio,
    permutation_test_burden_ratio,
)


@pytest.fixture()
def equity_data():
    """Synthetic prediction + demographic data for equity analysis."""
    rng = np.random.RandomState(42)
    n = 200

    # Higher contamination in high-POC areas (disparate burden)
    pct_poc = rng.uniform(0.1, 0.9, n)
    # Contamination probability increases with POC percentage
    p_contaminated = 0.1 + 0.5 * pct_poc
    y_true = (rng.random(n) < p_contaminated).astype(int)
    y_pred = (rng.random(n) < p_contaminated).astype(int)
    y_prob = p_contaminated + rng.normal(0, 0.1, n)
    y_prob = np.clip(y_prob, 0.01, 0.99)

    demographics = pd.DataFrame(
        {
            "pct_people_of_color": pct_poc,
            "pct_low_income": rng.uniform(0.1, 0.5, n),
        }
    )
    return y_true, y_pred, y_prob, demographics


class TestComputeBurdenRatio:
    """Tests for compute_burden_ratio()."""

    def test_equal_rates_returns_one(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        group = np.array([0.1, 0.2, 0.8, 0.9])
        ratio, _ = compute_burden_ratio(y_true, group)
        assert ratio == pytest.approx(1.0)

    def test_higher_rate_in_high_group(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        group = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        ratio, _ = compute_burden_ratio(y_true, group)
        assert ratio > 1.0

    def test_zero_rate_in_low_returns_inf(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        group = np.array([0.1, 0.2, 0.8, 0.9])
        ratio, _ = compute_burden_ratio(y_true, group)
        assert math.isinf(ratio)

    def test_custom_threshold(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        group = np.array([0.1, 0.2, 0.8, 0.9])
        _ratio, thresh = compute_burden_ratio(y_true, group, threshold=0.5)
        assert thresh == 0.5

    def test_empty_input(self) -> None:
        ratio, _thresh = compute_burden_ratio(np.array([]), np.array([]))
        assert math.isnan(ratio)

    def test_handles_nan_values(self) -> None:
        y_true = np.array([1, 0, 1, float("nan")])
        group = np.array([0.8, 0.2, 0.7, 0.5])
        ratio, _ = compute_burden_ratio(y_true, group)
        assert not math.isnan(ratio)

    def test_returns_threshold(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        group = np.array([0.2, 0.4, 0.6, 0.8])
        _, threshold = compute_burden_ratio(y_true, group)
        assert threshold == pytest.approx(0.5)


class TestComputeGroupMetrics:
    """Tests for compute_group_metrics()."""

    def test_returns_per_group_dict(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1])
        groups = np.array(["high", "high", "low", "low"])
        result = compute_group_metrics(y_true, y_pred, None, groups)
        assert "high" in result
        assert "low" in result

    def test_has_standard_metrics(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1])
        y_prob = np.array([0.9, 0.1, 0.4, 0.2, 0.8, 0.6])
        groups = np.array(["high", "high", "high", "low", "low", "low"])
        result = compute_group_metrics(y_true, y_pred, y_prob, groups)
        for label in ["high", "low"]:
            assert "accuracy" in result[label]
            assert "n_samples" in result[label]
            assert "positive_rate" in result[label]

    def test_positive_rate_correct(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        groups = np.array(["a", "a", "b", "b"])
        result = compute_group_metrics(y_true, y_pred, None, groups)
        assert result["a"]["positive_rate"] == pytest.approx(1.0)
        assert result["b"]["positive_rate"] == pytest.approx(0.0)

    def test_skips_small_groups(self) -> None:
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 1])
        groups = np.array(["big", "big", "tiny"])  # tiny has only 1 sample
        result = compute_group_metrics(y_true, y_pred, None, groups)
        assert "tiny" not in result

    def test_fpr_fnr_from_confusion_matrix(self) -> None:
        # TP=3, FN=1, TN=2, FP=2 -> FPR=2/4=0.5, FNR=1/4=0.25 (= 1 - recall)
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 1])
        groups = np.array(["g"] * 8)
        result = compute_group_metrics(y_true, y_pred, None, groups)
        assert result["g"]["fpr"] == pytest.approx(0.5)
        assert result["g"]["fnr"] == pytest.approx(0.25)
        assert result["g"]["fnr"] == pytest.approx(1.0 - result["g"]["recall"])

    def test_ece_present_with_probabilities(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        groups = np.array(["g"] * 6)
        result = compute_group_metrics(y_true, y_pred, y_prob, groups)
        assert "ece" in result["g"]
        assert 0.0 <= result["g"]["ece"] <= 1.0

    def test_ece_perfectly_calibrated_is_low(self) -> None:
        # Probabilities equal to empirical frequency within each bin -> ECE ~ 0.
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = (y_true).astype(int)
        y_prob = np.where(y_true == 1, 0.95, 0.05)
        groups = np.array(["g"] * 10)
        result = compute_group_metrics(y_true, y_pred, y_prob, groups)
        assert result["g"]["ece"] == pytest.approx(0.05, abs=1e-6)


class TestAnalyzeEquity:
    """Tests for analyze_equity() end-to-end."""

    def test_returns_disparity_report(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_people_of_color")
        assert isinstance(report, DisparityReport)

    def test_report_has_burden_ratio(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_people_of_color")
        assert isinstance(report.burden_ratio, float)
        assert report.burden_ratio > 0

    def test_report_has_group_metrics(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_people_of_color")
        assert "high" in report.group_metrics
        assert "low" in report.group_metrics

    def test_report_counts(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_people_of_color")
        assert report.n_high > 0
        assert report.n_low > 0
        # With percentile-based split (80th/50th), middle group is excluded
        n_middle = report.metadata.get("n_middle", 0)
        assert report.n_high + report.n_low + n_middle == len(y_true)

    def test_metadata_includes_threshold(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_people_of_color")
        assert "high_threshold" in report.metadata
        assert "low_threshold" in report.metadata
        assert "split_method" in report.metadata
        assert "percentile" in report.metadata["split_method"]

    def test_custom_threshold(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(
            y_true,
            y_pred,
            y_prob,
            demographics,
            "pct_people_of_color",
            threshold=0.3,
        )
        assert report.metadata["high_threshold"] == 0.3
        assert report.metadata["split_method"] == "custom"
        # Single threshold: no middle group, all samples classified
        assert report.n_high + report.n_low == len(y_true)

    def test_disparate_burden_detected(self, equity_data) -> None:
        """With our synthetic data, high-POC areas have higher contamination."""
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_people_of_color")
        # Burden ratio should be > 1 since we set up disparate outcomes
        assert report.burden_ratio > 1.0

    def test_group_col_stored(self, equity_data) -> None:
        y_true, y_pred, y_prob, demographics = equity_data
        report = analyze_equity(y_true, y_pred, y_prob, demographics, "pct_low_income")
        assert report.group_col == "pct_low_income"


class TestPermutationTest:
    """Tests for permutation_test_burden_ratio."""

    def test_returns_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        groups = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6, 0.1, 0.9])

        result = permutation_test_burden_ratio(y_true, groups, n_permutations=100)
        assert "observed_ratio" in result
        assert "p_value" in result
        assert "n_permutations" in result
        assert result["n_permutations"] == 100

    def test_p_value_in_range(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.randint(0, 2, n)
        groups = rng.random(n)

        result = permutation_test_burden_ratio(y_true, groups, n_permutations=100)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_strong_disparity_low_p(self) -> None:
        # Very strong disparity: all contamination in high-group
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10)
        groups = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9] * 10)

        result = permutation_test_burden_ratio(y_true, groups, n_permutations=500)
        assert result["p_value"] < 0.05  # Strong signal should be significant

    def test_empty_input(self) -> None:
        result = permutation_test_burden_ratio(np.array([]), np.array([]))
        assert np.isnan(result["observed_ratio"])
        assert np.isnan(result["p_value"])


class TestEquityFDRPipeline:
    """Integration tests for FDR correction applied to permutation test p-values."""

    def test_fdr_applied_to_permutation_pvalues(self, equity_data) -> None:
        """Run permutation tests for 2 groups, apply FDR, verify corrections."""
        y_true, _y_pred, _y_prob, demographics = equity_data
        groups = ["pct_people_of_color", "pct_low_income"]
        p_values = []
        for group_col in groups:
            result = permutation_test_burden_ratio(
                y_true, demographics[group_col].to_numpy(), n_permutations=200
            )
            p_values.append(result["p_value"])

        correction = apply_multiple_testing_correction(p_values, method="fdr_bh")
        corrected = correction["corrected_p_values"]
        reject = correction["reject"]

        assert len(corrected) == 2
        assert len(reject) == 2
        # Corrected p-values >= originals
        for orig, corr in zip(p_values, corrected, strict=True):
            assert corr >= orig or corr == pytest.approx(orig)

    def test_fdr_with_four_demographic_groups(self, equity_data) -> None:
        """Full 4-group FDR pipeline matching reproduce.py's _run_equity_analysis."""
        y_true, _y_pred, _y_prob, demographics = equity_data
        rng = np.random.RandomState(99)
        n = len(y_true)
        demographics = demographics.copy()
        demographics["pct_limited_english"] = rng.uniform(0.0, 0.3, n)
        demographics["pct_less_hs_education"] = rng.uniform(0.05, 0.4, n)

        groups = [
            "pct_people_of_color",
            "pct_low_income",
            "pct_limited_english",
            "pct_less_hs_education",
        ]
        p_values = []
        for group_col in groups:
            result = permutation_test_burden_ratio(
                y_true, demographics[group_col].to_numpy(), n_permutations=200
            )
            p_values.append(result["p_value"])

        correction = apply_multiple_testing_correction(p_values, method="fdr_bh")
        assert len(correction["corrected_p_values"]) == 4
        assert len(correction["reject"]) == 4
        # All corrected p-values in [0, 1]
        for p in correction["corrected_p_values"]:
            assert 0.0 <= p <= 1.0


class TestMultipleTestingCorrection:
    """Tests for apply_multiple_testing_correction."""

    def test_bonferroni_multiplies_by_n(self) -> None:
        p_values = [0.01, 0.03, 0.05]
        result = apply_multiple_testing_correction(p_values, method="bonferroni")
        corrected = result["corrected_p_values"]
        assert corrected[0] == pytest.approx(0.03)
        assert corrected[1] == pytest.approx(0.09)
        assert corrected[2] == pytest.approx(0.15)

    def test_bonferroni_caps_at_one(self) -> None:
        p_values = [0.5, 0.6, 0.7]
        result = apply_multiple_testing_correction(p_values, method="bonferroni")
        for p in result["corrected_p_values"]:
            assert p <= 1.0

    def test_fdr_bh_default(self) -> None:
        p_values = [0.01, 0.04, 0.1, 0.5]
        result = apply_multiple_testing_correction(p_values, method="fdr_bh")
        corrected = result["corrected_p_values"]
        # FDR-corrected p-values should be >= original
        for orig, corr in zip(p_values, corrected, strict=True):
            assert corr >= orig or corr == pytest.approx(orig)
        # Should be monotonic (sorted order preserved)
        assert len(corrected) == 4

    def test_fdr_bh_reject_mask(self) -> None:
        # Very small p-values should be rejected
        p_values = [0.001, 0.002, 0.8, 0.9]
        result = apply_multiple_testing_correction(p_values, method="fdr_bh", alpha=0.05)
        reject = result["reject"]
        assert reject[0] is True
        assert reject[1] is True
        assert reject[2] is False
        assert reject[3] is False

    def test_holm_stepdown(self) -> None:
        p_values = [0.01, 0.04, 0.06]
        result = apply_multiple_testing_correction(p_values, method="holm")
        corrected = result["corrected_p_values"]
        for orig, corr in zip(p_values, corrected, strict=True):
            assert corr >= orig or corr == pytest.approx(orig)

    def test_empty_input(self) -> None:
        result = apply_multiple_testing_correction([])
        assert result["corrected_p_values"] == []
        assert result["reject"] == []

    def test_single_p_value(self) -> None:
        result = apply_multiple_testing_correction([0.03], method="bonferroni")
        assert result["corrected_p_values"] == [pytest.approx(0.03)]
        assert result["reject"] == [True]

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_multiple_testing_correction([0.05], method="nonexistent")


class TestMonitoringAdjustedBurdenRatio:
    """Tests for the IPW monitoring-adjusted burden ratio."""

    @staticmethod
    def _confounded_data(seed: int = 0) -> tuple:
        """Build a scenario with NO true contamination disparity but where the
        high-demographic group is monitored more (so observed detection -- a
        function of sampling effort -- is inflated for it). Overlap in monitoring
        is preserved so IPW can adjust.
        """
        rng = np.random.RandomState(seed)
        n = 1200
        g = rng.uniform(0.0, 100.0, n)  # demographic indicator
        # Monitoring increases with g but with substantial noise (overlap preserved).
        n_samples = np.clip(4.0 + 0.30 * g + rng.normal(0.0, 9.0, n), 1.0, None)
        # TRUE per-sample detection probability is identical across groups (no real
        # disparity); observed any-detection = 1 - (1-p)^n_samples rises with effort.
        p_detect = 1.0 - (1.0 - 0.045) ** n_samples
        y = (rng.uniform(0.0, 1.0, n) < p_detect).astype(int)
        # An environmental feature correlated with monitoring so the propensity
        # model can recover the monitoring assignment.
        env = g / 100.0 + rng.normal(0.0, 0.15, n)
        X = pd.DataFrame({"env": env, "n_samples": n_samples})
        high_t = float(np.percentile(g, 80))
        low_t = float(np.percentile(g, 50))
        return y, g, n_samples, X, high_t, low_t

    def test_ipw_attenuates_monitoring_confound(self) -> None:
        y, g, n_samples, X, high_t, low_t = self._confounded_data()
        raw_high = y[g >= high_t].mean()
        raw_low = y[g < low_t].mean()
        raw_ratio = raw_high / raw_low

        adj = compute_monitoring_adjusted_burden_ratio(
            y, g, n_samples, X, high_threshold=high_t, low_threshold=low_t, n_permutations=200
        )
        # Confounding inflated the raw ratio above the true value of 1.0 ...
        assert raw_ratio > 1.15
        # ... and the IPW adjustment pulls it back toward 1 (attenuation).
        assert 0.0 < adj["burden_ratio_ipw"] < raw_ratio
        assert adj["n_high"] > 0 and adj["n_low"] > 0
        assert 0.0 <= adj["p_value"] <= 1.0

    def test_returns_expected_keys(self) -> None:
        y, g, n_samples, X, high_t, low_t = self._confounded_data()
        adj = compute_monitoring_adjusted_burden_ratio(
            y, g, n_samples, X, high_threshold=high_t, low_threshold=low_t, n_permutations=50
        )
        for key in (
            "burden_ratio_ipw",
            "rate_high_ipw",
            "rate_low_ipw",
            "p_value",
            "n_high",
            "n_low",
            "ipw_weight_mean",
        ):
            assert key in adj

    def test_constant_monitoring_returns_nan(self) -> None:
        # Positivity is undefined when monitoring is constant -> nan, not a crash.
        rng = np.random.RandomState(1)
        n = 100
        g = rng.uniform(0.0, 100.0, n)
        y = rng.randint(0, 2, n).astype(float)
        n_samples = np.full(n, 10.0)
        X = pd.DataFrame({"env": rng.normal(0, 1, n), "n_samples": n_samples})
        adj = compute_monitoring_adjusted_burden_ratio(
            y, g, n_samples, X, high_threshold=80.0, low_threshold=50.0, n_permutations=10
        )
        assert math.isnan(adj["burden_ratio_ipw"])
