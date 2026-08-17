"""Tests for compute_inflation_across_seeds."""

from __future__ import annotations

from aquacontam.analysis.split_strategy_comparison import compute_inflation_across_seeds


class TestInflationAcrossSeeds:
    def test_basic_positive_inflation(self) -> None:
        result = compute_inflation_across_seeds(
            random_aurocs=[0.90, 0.91, 0.89, 0.92, 0.88],
            geographic_auroc=0.85,
        )
        assert result["inflation_mean"] > 0
        assert result["all_positive_inflation"]
        assert result["n_seeds"] == 5
        assert abs(result["random_auroc_mean"] - 0.90) < 0.01

    def test_no_inflation(self) -> None:
        result = compute_inflation_across_seeds(
            random_aurocs=[0.85, 0.85, 0.85],
            geographic_auroc=0.85,
        )
        assert abs(result["inflation_mean"]) < 1e-10
        assert not result["all_positive_inflation"]  # 0 is not > 0

    def test_negative_inflation(self) -> None:
        result = compute_inflation_across_seeds(
            random_aurocs=[0.80, 0.81],
            geographic_auroc=0.85,
        )
        assert result["inflation_mean"] < 0
        assert not result["all_positive_inflation"]

    def test_single_seed(self) -> None:
        result = compute_inflation_across_seeds(
            random_aurocs=[0.90],
            geographic_auroc=0.85,
        )
        assert result["n_seeds"] == 1
        assert abs(result["inflation_mean"] - 0.05) < 1e-10
        assert result["inflation_std"] == 0.0
