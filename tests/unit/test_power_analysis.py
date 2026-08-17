"""Tests for post-hoc power analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquacontam.analysis.power_analysis import (
    compute_auroc_comparison_power,
    compute_proportion_test_power,
    compute_regression_coefficient_power,
    summarize_power,
)


class TestAurocComparisonPower:
    """Tests for compute_auroc_comparison_power."""

    def test_equal_aurocs_power_near_alpha(self) -> None:
        """When AUROCs are equal, power should be approximately alpha."""
        result = compute_auroc_comparison_power(0.80, 0.80, 1000, 1000)
        assert result["effect_size"] == pytest.approx(0.0)
        assert result["power"] == pytest.approx(0.05, abs=0.001)

    def test_large_effect_high_power(self) -> None:
        """Large AUROC difference with big n should yield high power."""
        result = compute_auroc_comparison_power(0.90, 0.70, 5000, 5000)
        assert result["power"] > 0.99

    def test_small_effect_low_power(self) -> None:
        """Small AUROC difference with small n should yield low power."""
        result = compute_auroc_comparison_power(0.80, 0.79, 50, 50)
        assert result["power"] < 0.20

    def test_very_large_n_power_approaches_one(self) -> None:
        """Even a modest effect should be detected with enormous n."""
        result = compute_auroc_comparison_power(0.82, 0.80, 100_000, 100_000)
        assert result["power"] > 0.99

    def test_return_keys(self) -> None:
        """Verify all expected keys are present."""
        result = compute_auroc_comparison_power(0.85, 0.80, 500, 500)
        assert set(result.keys()) == {"effect_size", "pooled_se", "z_statistic", "power", "alpha"}

    def test_symmetric_in_auroc_order(self) -> None:
        """Power should be the same regardless of which AUROC is larger."""
        r1 = compute_auroc_comparison_power(0.85, 0.75, 300, 300)
        r2 = compute_auroc_comparison_power(0.75, 0.85, 300, 300)
        assert r1["power"] == pytest.approx(r2["power"])


class TestProportionTestPower:
    """Tests for compute_proportion_test_power."""

    def test_equal_proportions_power_near_alpha(self) -> None:
        """When proportions are equal, power should be approximately alpha."""
        result = compute_proportion_test_power(0.15, 0.15, 500, 500)
        assert result["effect_size"] == pytest.approx(0.0)
        assert result["power"] == pytest.approx(0.05, abs=0.001)

    def test_large_difference_high_power(self) -> None:
        """Large proportion difference should yield high power."""
        # 25% vs 6% with reasonable n — should be very well powered
        result = compute_proportion_test_power(0.252, 0.062, 325, 805)
        assert result["power"] > 0.99

    def test_small_difference_small_n(self) -> None:
        """Small proportion difference with small n has low power."""
        result = compute_proportion_test_power(0.50, 0.48, 30, 30)
        assert result["power"] < 0.10

    def test_return_keys(self) -> None:
        """Verify all expected keys are present."""
        result = compute_proportion_test_power(0.3, 0.2, 100, 100)
        assert set(result.keys()) == {"effect_size", "pooled_se", "z_statistic", "power", "alpha"}


class TestRegressionCoefficientPower:
    """Tests for compute_regression_coefficient_power."""

    def test_significant_coefficient_high_power(self) -> None:
        """A coefficient well above its SE should have high power."""
        # beta/se = 0.038/0.015 ~ 2.53, with n=14215 -> good power
        result = compute_regression_coefficient_power(0.038, 0.015, 14215)
        assert result["power"] > 0.70

    def test_tiny_coefficient_low_power(self) -> None:
        """Coefficient near zero relative to SE should have low power."""
        result = compute_regression_coefficient_power(0.001, 0.05, 100)
        assert result["power"] < 0.10

    def test_large_n_boosts_power(self) -> None:
        """With fixed effect/SE ratio, more data means more power (via df)."""
        r_small = compute_regression_coefficient_power(0.5, 0.2, 30)
        r_large = compute_regression_coefficient_power(0.5, 0.2, 10000)
        # Both should have high power with t=2.5, but large-n should be >= small-n
        assert r_large["power"] >= r_small["power"] - 0.01

    def test_return_keys(self) -> None:
        """Verify all expected keys including df."""
        result = compute_regression_coefficient_power(1.0, 0.5, 200)
        assert set(result.keys()) == {"effect_size", "se", "t_statistic", "power", "alpha", "df"}
        assert result["df"] == 198

    def test_negative_beta_uses_absolute(self) -> None:
        """Negative beta should produce same power as positive."""
        r_pos = compute_regression_coefficient_power(0.5, 0.2, 100)
        r_neg = compute_regression_coefficient_power(-0.5, 0.2, 100)
        assert r_pos["power"] == pytest.approx(r_neg["power"])
        assert r_neg["effect_size"] == pytest.approx(0.5)


class TestSummarizePower:
    """Tests for summarize_power."""

    def test_defaults_no_results_dir(self) -> None:
        """With no results_dir, uses hardcoded defaults and returns all keys."""
        result = summarize_power()
        assert set(result.keys()) == {"delong_auroc", "ej_burden_ratio", "dml_coefficient"}
        # Check that each sub-dict has the expected structure
        assert "power" in result["delong_auroc"]
        assert "power" in result["ej_burden_ratio"]
        assert "power" in result["dml_coefficient"]

    def test_defaults_produce_reasonable_power(self) -> None:
        """Default pipeline values should produce sensible power estimates."""
        result = summarize_power()
        # EJ burden ratio (25.2% vs 6.2%) should be very well powered
        assert result["ej_burden_ratio"]["power"] > 0.99
        # DeLong (top-two T1 AUROCs ~0.864 vs ~0.856) — a tiny difference, low power
        assert 0.0 < result["delong_auroc"]["power"] < 1.0
        # DML coefficient (beta=0.038, se=0.015, n=14215)
        assert result["dml_coefficient"]["power"] > 0.50

    def test_loads_delong_pair_from_results_json(self, tmp_path: Path) -> None:
        """R5 m3: the DeLong pair is read from the benchmark results.json (top two T1 AUROCs)."""
        results = [
            {
                "task": "T1",
                "model": "a",
                "metrics": {"auroc": 0.95},
                "metadata": {"n_test": 10000},
            },
            {
                "task": "T1",
                "model": "b",
                "metrics": {"auroc": 0.50},
                "metadata": {"n_test": 10000},
            },
            {
                "task": "T1",
                "model": "c",
                "metrics": {"auroc": 0.70},
                "metadata": {"n_test": 10000},
            },
        ]
        (tmp_path / "results.json").write_text(json.dumps(results))
        result = summarize_power(results_dir=tmp_path)
        # top two are 0.95 and 0.70 -> effect 0.25, and n=10000 -> high power
        assert result["delong_auroc"]["effect_size"] == pytest.approx(0.25)
        assert result["delong_auroc"]["power"] > 0.99

    def test_missing_results_json_fails_loud(self, tmp_path: Path) -> None:
        """R5 m3: a missing benchmark is an error, not a licence to use hardcoded AUROCs."""
        with pytest.raises(FileNotFoundError):
            summarize_power(results_dir=tmp_path)
