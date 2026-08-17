"""Tests for benchmark evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from aquacontam.benchmark.metrics import (
    bootstrap_classification_metrics,
    calibration_curve_data,
    compare_models_pairwise,
    compute_censoring_aware_metrics,
    compute_classification_metrics,
    compute_lift_analysis,
    compute_regression_metrics,
    delong_test,
    delong_test_vs_chance,
    expected_calibration_error,
    optimal_threshold_analysis,
    paired_bootstrap_test,
)


class TestClassificationMetrics:
    """Tests for compute_classification_metrics."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        result = compute_classification_metrics(y_true, y_pred, y_prob)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["balanced_accuracy"] == 1.0
        assert result["auroc"] == pytest.approx(1.0)
        assert result["auprc"] == pytest.approx(1.0)

    def test_worst_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 0.0
        assert result["f1"] == 0.0
        assert result["recall"] == 0.0

    def test_no_positive_class(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert np.isnan(result["auroc"])
        assert np.isnan(result["auprc"])

    def test_single_class_nan_fallback(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7])

        result = compute_classification_metrics(y_true, y_pred, y_prob)
        assert np.isnan(result["auroc"])
        assert np.isnan(result["auprc"])

    def test_no_prob_provided(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        result = compute_classification_metrics(y_true, y_pred, y_prob=None)
        assert np.isnan(result["auroc"])
        assert np.isnan(result["auprc"])
        assert result["accuracy"] == pytest.approx(0.75)

    def test_custom_metrics_subset(self) -> None:
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1, 1])

        result = compute_classification_metrics(y_true, y_pred, metrics=["accuracy", "f1"])
        assert set(result.keys()) == {"accuracy", "f1"}

    def test_unknown_metric_returns_nan(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])

        result = compute_classification_metrics(y_true, y_pred, metrics=["nonexistent"])
        assert np.isnan(result["nonexistent"])

    def test_imbalanced_classification(self) -> None:
        # 90% negative, 10% positive — reflects water quality data
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 90 + [1] * 10)
        y_prob = rng.random(100)
        y_pred = (y_prob > 0.5).astype(int)

        result = compute_classification_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= result["auroc"] <= 1.0
        assert 0.0 <= result["auprc"] <= 1.0
        assert 0.0 <= result["balanced_accuracy"] <= 1.0

    def test_pandas_series_input(self) -> None:
        import pandas as pd

        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])

        result = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in result

    def test_zero_division_precision(self) -> None:
        # No positive predictions → precision zero_division=0
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])

        result = compute_classification_metrics(y_true, y_pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


class TestRegressionMetrics:
    """Tests for compute_regression_metrics."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = compute_regression_metrics(y_true, y_pred)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)
        assert result["explained_variance"] == pytest.approx(1.0)
        assert result["median_ae"] == pytest.approx(0.0)

    def test_constant_offset(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])

        result = compute_regression_metrics(y_true, y_pred)
        assert result["rmse"] == pytest.approx(1.0)
        assert result["mae"] == pytest.approx(1.0)

    def test_custom_metrics_subset(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])

        result = compute_regression_metrics(y_true, y_pred, metrics=["rmse"])
        assert set(result.keys()) == {"rmse"}

    def test_unknown_metric_returns_nan(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])

        result = compute_regression_metrics(y_true, y_pred, metrics=["nonexistent"])
        assert np.isnan(result["nonexistent"])

    def test_negative_r2(self) -> None:
        # Very bad predictions → negative R²
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([100.0, 200.0, 300.0])

        result = compute_regression_metrics(y_true, y_pred)
        assert result["r2"] < 0.0

    def test_pandas_input(self) -> None:
        import pandas as pd

        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 2.1, 3.1])

        result = compute_regression_metrics(y_true, y_pred)
        assert result["mae"] == pytest.approx(0.1, abs=1e-10)


class TestCensoringAwareMetrics:
    """Tests for compute_censoring_aware_metrics."""

    def test_all_detected(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        censored = np.array([False, False, False])
        dl = np.array([0.5, 0.5, 0.5])

        result = compute_censoring_aware_metrics(y_true, y_pred, censored, dl)
        assert result["detected_rmse"] > 0
        assert np.isnan(result["censored_rmse"])
        assert 0.0 <= result["concordance_index"] <= 1.0

    def test_all_censored(self) -> None:
        y_true = np.array([0.5, 0.5, 0.5])
        y_pred = np.array([0.1, 0.3, 0.8])
        censored = np.array([True, True, True])
        dl = np.array([0.5, 0.5, 0.5])

        result = compute_censoring_aware_metrics(y_true, y_pred, censored, dl)
        assert np.isnan(result["detected_rmse"])
        assert result["censored_rmse"] >= 0.0

    def test_tobit_no_penalty_below_dl(self) -> None:
        # Predictions below DL for censored rows → no penalty
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.1, 0.2])
        censored = np.array([True, True])
        dl = np.array([0.5, 0.5])

        result = compute_censoring_aware_metrics(y_true, y_pred, censored, dl)
        assert result["censored_rmse"] == pytest.approx(0.0)

    def test_tobit_penalty_above_dl(self) -> None:
        # Predictions above DL for censored rows → penalized
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        censored = np.array([True, True])
        dl = np.array([0.5, 0.5])

        result = compute_censoring_aware_metrics(y_true, y_pred, censored, dl)
        assert result["censored_rmse"] > 0.0

    def test_concordance_perfect(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        censored = np.array([False, False, False, False])
        dl = np.array([0.5, 0.5, 0.5, 0.5])

        result = compute_censoring_aware_metrics(y_true, y_pred, censored, dl)
        assert result["concordance_index"] == pytest.approx(1.0)

    def test_concordance_single_sample(self) -> None:
        result = compute_censoring_aware_metrics(
            np.array([1.0]),
            np.array([1.0]),
            np.array([False]),
            np.array([0.5]),
        )
        assert np.isnan(result["concordance_index"])

    def test_mixed_censored_detected(self) -> None:
        y_true = np.array([0.0, 1.5, 0.0, 3.0])
        y_pred = np.array([0.2, 1.4, 0.8, 2.9])
        censored = np.array([True, False, True, False])
        dl = np.array([0.5, 0.5, 0.5, 0.5])

        result = compute_censoring_aware_metrics(y_true, y_pred, censored, dl)
        assert result["detected_rmse"] >= 0.0
        assert result["censored_rmse"] >= 0.0
        assert "concordance_index" in result


class TestBootstrapClassification:
    """Tests for bootstrap_classification_metrics."""

    def test_returns_ci_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.6, 0.7])

        result = bootstrap_classification_metrics(
            y_true, y_pred, y_prob, metrics=["auroc", "f1"], n_bootstrap=100
        )
        for m in ["auroc", "f1"]:
            assert "point" in result[m]
            assert "ci_lower" in result[m]
            assert "ci_upper" in result[m]
            assert "std" in result[m]

    def test_ci_bounds_order(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.randint(0, 2, n)
        y_pred = rng.randint(0, 2, n)
        y_prob = rng.random(n)

        result = bootstrap_classification_metrics(
            y_true, y_pred, y_prob, metrics=["accuracy"], n_bootstrap=200
        )
        acc = result["accuracy"]
        assert acc["ci_lower"] <= acc["point"] or np.isnan(acc["ci_lower"])
        assert acc["ci_upper"] >= acc["point"] or np.isnan(acc["ci_upper"])

    def test_perfect_predictions_narrow_ci(self) -> None:
        y_true = np.array([0, 0, 1, 1] * 10)
        y_pred = np.array([0, 0, 1, 1] * 10)
        y_prob = np.array([0.0, 0.0, 1.0, 1.0] * 10)

        result = bootstrap_classification_metrics(
            y_true, y_pred, y_prob, metrics=["accuracy"], n_bootstrap=100
        )
        assert result["accuracy"]["point"] == 1.0
        assert result["accuracy"]["ci_lower"] >= 0.9


class TestPairedBootstrapTest:
    """Tests for paired_bootstrap_test."""

    def test_identical_models_delta_zero(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.randint(0, 2, n)
        y_prob = rng.random(n)

        result = paired_bootstrap_test(y_true, y_prob, y_prob, n_iterations=200)
        assert result["delta"] == pytest.approx(0.0)
        assert result["p_value"] >= 0.5  # not significant

    def test_better_model_positive_delta(self) -> None:
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, 2, n)
        # Model A: nearly perfect probabilities
        y_prob_a = y_true * 0.8 + (1 - y_true) * 0.2
        # Model B: random probabilities
        y_prob_b = rng.random(n)

        result = paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n_iterations=500)
        assert result["delta"] > 0
        assert result["p_value"] < 0.05

    def test_returns_required_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob_a = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        y_prob_b = np.array([0.4, 0.3, 0.6, 0.5, 0.4, 0.6])

        result = paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n_iterations=100)
        assert "delta" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_auprc_metric(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.randint(0, 2, n)
        y_prob_a = rng.random(n)
        y_prob_b = rng.random(n)

        result = paired_bootstrap_test(
            y_true, y_prob_a, y_prob_b, metric_fn="auprc", n_iterations=100
        )
        assert "delta" in result
        assert not np.isnan(result["delta"])

    def test_single_class_returns_nan(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_prob_a = np.array([0.1, 0.2, 0.3, 0.4])
        y_prob_b = np.array([0.2, 0.3, 0.4, 0.5])

        result = paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n_iterations=100)
        assert np.isnan(result["delta"])


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error."""

    def test_perfectly_calibrated(self) -> None:
        # Probabilities match empirical frequencies
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        result = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert result["ece"] == pytest.approx(0.0, abs=0.05)

    def test_worst_calibrated(self) -> None:
        # All probabilities = 0.5 but all labels are 0
        y_true = np.zeros(100)
        y_prob = np.full(100, 0.5)

        result = expected_calibration_error(y_true, y_prob)
        assert result["ece"] == pytest.approx(0.5, abs=0.01)
        assert result["mce"] == pytest.approx(0.5, abs=0.01)

    def test_returns_required_keys(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        result = expected_calibration_error(y_true, y_prob)
        assert "ece" in result
        assert "mce" in result
        assert "mean_predicted_prob" in result
        assert "mean_observed_freq" in result

    def test_empty_input_returns_nan(self) -> None:
        result = expected_calibration_error(np.array([]), np.array([]))
        assert np.isnan(result["ece"])

    def test_quantile_strategy(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_prob = rng.random(100)

        result = expected_calibration_error(y_true, y_prob, strategy="quantile")
        assert 0.0 <= result["ece"] <= 1.0

    def test_ece_bounded(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.random(200)

        result = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= result["ece"] <= 1.0
        assert 0.0 <= result["mce"] <= 1.0


class TestCalibrationCurveData:
    """Tests for calibration_curve_data."""

    def test_returns_list_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

        result = calibration_curve_data(y_true, y_prob, n_bins=5)
        assert isinstance(result["mean_predicted"], list)
        assert isinstance(result["fraction_positive"], list)
        assert isinstance(result["bin_counts"], list)

    def test_single_class_returns_empty(self) -> None:
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])

        result = calibration_curve_data(y_true, y_prob)
        assert result["mean_predicted"] == []
        assert result["fraction_positive"] == []

    def test_fraction_positive_bounded(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.random(200)

        result = calibration_curve_data(y_true, y_prob, n_bins=10)
        for frac in result["fraction_positive"]:
            assert 0.0 <= frac <= 1.0


class TestOptimalThresholdAnalysis:
    """Tests for optimal_threshold_analysis."""

    def test_imbalanced_finds_better_threshold(self) -> None:
        # 90% negative, 10% positive with informative probabilities
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 90 + [1] * 10)
        # Give positive class higher probabilities
        y_prob = np.concatenate([rng.uniform(0.0, 0.3, 90), rng.uniform(0.3, 0.8, 10)])

        result = optimal_threshold_analysis(y_true, y_prob)
        assert result["optimal_threshold"] < 0.5
        assert result["f1_at_optimal"] > result["default_threshold_f1"]

    def test_balanced_threshold_near_half(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.concatenate([rng.uniform(0.0, 0.5, 50), rng.uniform(0.5, 1.0, 50)])

        result = optimal_threshold_analysis(y_true, y_prob)
        assert 0.3 <= result["optimal_threshold"] <= 0.7

    def test_returns_required_keys(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        result = optimal_threshold_analysis(y_true, y_prob)
        assert "optimal_threshold" in result
        assert "optimal_metric_value" in result
        assert "f1_at_optimal" in result
        assert "precision_at_optimal" in result
        assert "recall_at_optimal" in result
        assert "default_threshold_f1" in result

    def test_empty_input_returns_nan(self) -> None:
        result = optimal_threshold_analysis(np.array([]), np.array([]))
        assert np.isnan(result["optimal_threshold"])

    def test_single_class_returns_nan(self) -> None:
        result = optimal_threshold_analysis(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        assert np.isnan(result["optimal_threshold"])

    def test_youden_metric(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 80 + [1] * 20)
        y_prob = np.concatenate([rng.uniform(0.0, 0.4, 80), rng.uniform(0.3, 0.9, 20)])

        result = optimal_threshold_analysis(y_true, y_prob, metric="youden")
        assert 0.0 < result["optimal_threshold"] < 1.0

    def test_balanced_accuracy_metric(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 80 + [1] * 20)
        y_prob = np.concatenate([rng.uniform(0.0, 0.4, 80), rng.uniform(0.3, 0.9, 20)])

        result = optimal_threshold_analysis(y_true, y_prob, metric="balanced_accuracy")
        assert result["optimal_metric_value"] > 0.5

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            optimal_threshold_analysis(np.array([0, 1]), np.array([0.3, 0.7]), metric="unknown")


class TestDeLongTest:
    """Tests for the DeLong AUROC comparison test."""

    def test_identical_models(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = rng.uniform(0.0, 1.0, 100)
        result = delong_test(y_true, y_prob, y_prob)
        assert result["delta"] == pytest.approx(0.0, abs=1e-10)
        assert result["p_value"] == pytest.approx(1.0, abs=1e-5)

    def test_clearly_different_models(self) -> None:
        y_true = np.array([0] * 100 + [1] * 100)
        # Model A: perfect separation
        y_prob_a = np.array([0.1] * 100 + [0.9] * 100)
        # Model B: random
        rng = np.random.RandomState(42)
        y_prob_b = rng.uniform(0.0, 1.0, 200)
        result = delong_test(y_true, y_prob_a, y_prob_b)
        assert result["delta"] > 0
        assert result["p_value"] < 0.05

    def test_single_class_returns_nan(self) -> None:
        y_true = np.array([0, 0, 0])
        y_prob_a = np.array([0.1, 0.2, 0.3])
        y_prob_b = np.array([0.4, 0.5, 0.6])
        result = delong_test(y_true, y_prob_a, y_prob_b)
        assert np.isnan(result["p_value"])

    def test_auroc_values_reasonable(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 80 + [1] * 20)
        y_prob_a = np.concatenate([rng.uniform(0.0, 0.5, 80), rng.uniform(0.4, 1.0, 20)])
        y_prob_b = rng.uniform(0.0, 1.0, 100)
        result = delong_test(y_true, y_prob_a, y_prob_b)
        assert 0.0 <= result["auroc_a"] <= 1.0
        assert 0.0 <= result["auroc_b"] <= 1.0
        assert -1.0 <= result["delta"] <= 1.0

    def test_output_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob_a = np.array([0.1, 0.2, 0.8, 0.9])
        y_prob_b = np.array([0.3, 0.4, 0.6, 0.7])
        result = delong_test(y_true, y_prob_a, y_prob_b)
        expected_keys = {"auroc_a", "auroc_b", "delta", "z_statistic", "p_value"}
        assert set(result.keys()) == expected_keys


class TestDeLongTestVsChance:
    """Tests for one-sample DeLong test (AUROC vs 0.5)."""

    def test_perfect_separation_significant(self) -> None:
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.array([0.1] * 50 + [0.9] * 50)
        result = delong_test_vs_chance(y_true, y_prob)
        assert result["auroc"] == pytest.approx(1.0)
        assert result["p_value"] < 0.05

    def test_random_predictions_not_significant(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)
        result = delong_test_vs_chance(y_true, y_prob)
        assert result["p_value"] > 0.05

    def test_single_class_returns_nan(self) -> None:
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.5, 0.6, 0.7, 0.8])
        result = delong_test_vs_chance(y_true, y_prob)
        assert np.isnan(result["p_value"])

    def test_output_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = delong_test_vs_chance(y_true, y_prob)
        assert set(result.keys()) == {"auroc", "variance", "z_statistic", "p_value"}

    def test_auroc_matches_delong_pair(self) -> None:
        """AUROC from vs-chance should match the paired DeLong function."""
        rng = np.random.RandomState(123)
        y_true = np.array([0] * 60 + [1] * 40)
        y_prob = np.concatenate([rng.uniform(0.0, 0.6, 60), rng.uniform(0.3, 1.0, 40)])
        result_vs_chance = delong_test_vs_chance(y_true, y_prob)
        result_pair = delong_test(y_true, y_prob, y_prob)
        assert result_vs_chance["auroc"] == pytest.approx(result_pair["auroc_a"], abs=1e-10)

    def test_small_sample_near_chance(self) -> None:
        """Simulate CA GeoTracker-like scenario: small n, AUROC near 0.5."""
        rng = np.random.RandomState(99)
        n = 81
        y_true = rng.choice([0, 1], size=n, p=[0.41, 0.59])
        # Weakly informative predictions
        y_prob = rng.uniform(0.3, 0.7, size=n)
        result = delong_test_vs_chance(y_true, y_prob)
        # With such weak signal and small n, should not be significant
        assert result["p_value"] > 0.05
        assert result["variance"] > 0


class TestCompareModelsPairwise:
    """Tests for pairwise model comparison with FDR correction."""

    def test_three_models(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 100 + [1] * 100)
        model_probs = {
            "perfect": np.array([0.1] * 100 + [0.9] * 100),
            "good": np.concatenate([rng.uniform(0.0, 0.4, 100), rng.uniform(0.6, 1.0, 100)]),
            "random": rng.uniform(0.0, 1.0, 200),
        }
        df = compare_models_pairwise(y_true, model_probs)
        assert len(df) == 3  # C(3,2) = 3 pairs
        assert "p_value_fdr" in df.columns
        assert "significant" in df.columns

    def test_empty_models(self) -> None:
        y_true = np.array([0, 1])
        df = compare_models_pairwise(y_true, {})
        assert len(df) == 0

    def test_fdr_correction_applied(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 100 + [1] * 100)
        model_probs = {
            "a": np.array([0.05] * 100 + [0.95] * 100),
            "b": rng.uniform(0.0, 1.0, 200),
            "c": rng.uniform(0.0, 1.0, 200),
        }
        df = compare_models_pairwise(y_true, model_probs)
        # FDR-adjusted p-values should be >= raw p-values
        assert (df["p_value_fdr"] >= df["p_value"] - 1e-10).all()

    def test_bootstrap_method(self) -> None:
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        model_probs = {
            "a": np.array([0.1] * 50 + [0.9] * 50),
            "b": rng.uniform(0.0, 1.0, 100),
        }
        df = compare_models_pairwise(y_true, model_probs, method="bootstrap")
        assert len(df) == 1

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            compare_models_pairwise(
                np.array([0, 1]),
                {"a": np.array([0.5, 0.5]), "b": np.array([0.3, 0.7])},
                method="bad",
            )


class TestComputeLiftAnalysis:
    """Tests for compute_lift_analysis."""

    def test_perfect_model_high_lift(self) -> None:
        """A perfect model should rank all positives in the top deciles."""
        y_true = np.array([0] * 80 + [1] * 20)
        y_prob = np.array([0.1] * 80 + [0.9] * 20)
        result = compute_lift_analysis(y_true, y_prob)

        assert result["overall_detection_rate"] == pytest.approx(0.2)
        # Top decile should have ~100% detection rate
        assert result["decile_detection_rates"][0] == pytest.approx(1.0)
        # Top decile lift should be 5x (1.0 / 0.2)
        assert result["top_decile_lift"] == pytest.approx(5.0)
        # Top quintile should capture all detections
        assert result["top_quintile_capture"] == pytest.approx(1.0)

    def test_random_model_lift_near_one(self) -> None:
        """A random model should have lift near 1.0."""
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.3, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        result = compute_lift_analysis(y_true, y_prob)

        # Lift should be close to 1.0 for random
        assert 0.5 < result["top_decile_lift"] < 2.0
        assert len(result["decile_detection_rates"]) == 10
        assert len(result["cumulative_lift"]) == 10

    def test_empty_input(self) -> None:
        result = compute_lift_analysis(np.array([]), np.array([]))
        assert np.isnan(result["top_decile_lift"])
        assert result["decile_detection_rates"] == []

    def test_all_negative(self) -> None:
        """All-negative data should have zero detection rates."""
        y_true = np.zeros(100)
        y_prob = np.random.default_rng(42).uniform(0, 1, 100)
        result = compute_lift_analysis(y_true, y_prob)
        assert result["overall_detection_rate"] == 0.0
        assert all(r == 0.0 for r in result["decile_detection_rates"])

    def test_reference_prevalence_reanchors_lift(self) -> None:
        """A lower reference prevalence than the in-sample rate yields a HIGHER
        reference-anchored lift; in-sample lift is unchanged and both are reported."""
        rng = np.random.default_rng(0)
        # Enriched in-sample prevalence ~0.5; a perfectly-ranked top decile.
        y_true = np.concatenate([np.ones(50), np.zeros(50)])
        y_prob = np.concatenate([rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.4, 50)])
        result = compute_lift_analysis(y_true, y_prob, reference_prevalence=0.1)
        assert result["reference_prevalence"] == 0.1
        # in-sample lift unchanged and present
        assert "top_decile_lift" in result
        # vs a 0.1 base rate the lift is larger than vs the ~0.5 in-sample rate
        assert result["top_decile_lift_vs_reference"] > result["top_decile_lift"]
        assert len(result["cumulative_lift_vs_reference"]) == len(result["cumulative_lift"])

    def test_no_reference_prevalence_omits_fields(self) -> None:
        y_true = np.concatenate([np.ones(20), np.zeros(20)])
        y_prob = np.concatenate([np.full(20, 0.9), np.full(20, 0.1)])
        result = compute_lift_analysis(y_true, y_prob)
        assert "reference_prevalence" not in result
        assert "top_decile_lift_vs_reference" not in result
