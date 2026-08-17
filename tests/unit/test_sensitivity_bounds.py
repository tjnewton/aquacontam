"""Tests for Rosenbaum-style sensitivity bounds for unmeasured confounding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.causal_deconfounding import (
    compute_sensitivity_bounds,
    sensitivity_analysis,
)


class TestComputeSensitivityBounds:
    """Tests for compute_sensitivity_bounds."""

    def test_gamma_one_reproduces_original(self) -> None:
        """At gamma=1 (no unmeasured confounding), bounds match the original effect."""
        effect = 0.5
        se = 0.1
        result = compute_sensitivity_bounds(effect, se, gamma_range=(1.0, 1.0), n_steps=1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["gamma"] == pytest.approx(1.0)
        # log(1) = 0, so shift is zero — bounds are just the standard CI
        assert row["effect_lower"] == pytest.approx(effect - 1.96 * se, abs=0.01)
        assert row["effect_upper"] == pytest.approx(effect + 1.96 * se, abs=0.01)
        # p-value bound at gamma=1 should match the original z-test p-value
        from scipy import stats

        expected_p = 2.0 * (1.0 - stats.norm.cdf(abs(effect) / se))
        assert row["p_value_bound"] == pytest.approx(expected_p, abs=1e-6)
        assert bool(row["significant"]) is True

    def test_large_gamma_makes_effects_nonsignificant(self) -> None:
        """A moderate effect should become non-significant at large enough gamma."""
        effect = 0.3
        se = 0.1
        result = compute_sensitivity_bounds(effect, se, gamma_range=(1.0, 10.0), n_steps=51)

        # At gamma=1, should be significant (z = 3.0)
        first_row = result.iloc[0]
        assert bool(first_row["significant"]) is True

        # At gamma=10, shift = log(10)*0.1 ≈ 0.23, so z_gamma ≈ (0.3-0.23)/0.1 = 0.7
        # which is not significant
        last_row = result.iloc[-1]
        assert bool(last_row["significant"]) is False

    def test_very_large_effect_stays_significant(self) -> None:
        """An extremely large effect should remain significant even at high gamma."""
        effect = 10.0
        se = 0.1
        result = compute_sensitivity_bounds(effect, se, gamma_range=(1.0, 3.0), n_steps=21)

        # z at gamma=3: (10 - log(3)*0.1) / 0.1 ≈ (10 - 0.11) / 0.1 ≈ 98.9
        # Definitely significant at all gamma values in [1, 3]
        assert result["significant"].all()

    def test_tipping_point_for_moderate_effect(self) -> None:
        """Tipping point Gamma* should be between 1.0 and max gamma for moderate effects."""
        effect = 0.3
        se = 0.1
        result = compute_sensitivity_bounds(effect, se, gamma_range=(1.0, 5.0), n_steps=101)

        sig = result["significant"]
        # First row (gamma=1, z=3) should be significant
        assert bool(sig.iloc[0]) is True
        # Last row (gamma=5, shift=log(5)*0.1≈0.16, z≈(0.3-0.16)/0.1=1.4) not significant
        assert bool(sig.iloc[-1]) is False

        # Find tipping point: first gamma where significant switches to False
        first_nonsig = result.loc[~sig, "gamma"].iloc[0]
        assert 1.0 < first_nonsig < 5.0

    def test_negative_effect(self) -> None:
        """Sensitivity bounds should work correctly for negative causal effects."""
        effect = -0.5
        se = 0.1
        result = compute_sensitivity_bounds(effect, se, gamma_range=(1.0, 3.0), n_steps=11)

        # All bounds should be finite
        assert result["effect_lower"].notna().all()
        assert result["effect_upper"].notna().all()
        # Lower bound should always be <= upper bound
        assert (result["effect_lower"] <= result["effect_upper"]).all()
        # At gamma=1, the effect is highly significant (z=5)
        assert bool(result.iloc[0]["significant"]) is True

    def test_output_columns(self) -> None:
        """Output DataFrame should have the expected columns."""
        result = compute_sensitivity_bounds(0.5, 0.1)
        expected_cols = {"gamma", "effect_lower", "effect_upper", "p_value_bound", "significant"}
        assert set(result.columns) == expected_cols

    def test_n_steps(self) -> None:
        """Number of rows should match n_steps."""
        result = compute_sensitivity_bounds(0.5, 0.1, n_steps=15)
        assert len(result) == 15

    def test_p_value_bound_increases_with_gamma(self) -> None:
        """p-value bound should be non-decreasing as gamma increases."""
        result = compute_sensitivity_bounds(0.5, 0.1, gamma_range=(1.0, 5.0), n_steps=51)
        p_values = result["p_value_bound"].to_numpy()
        # Allow small numerical tolerance for monotonicity
        assert np.all(np.diff(p_values) >= -1e-10)

    def test_zero_std_error_raises(self) -> None:
        """std_error <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="std_error must be positive"):
            compute_sensitivity_bounds(0.5, 0.0)
        with pytest.raises(ValueError, match="std_error must be positive"):
            compute_sensitivity_bounds(0.5, -0.1)

    def test_gamma_below_one_raises(self) -> None:
        """gamma_range lower bound < 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match=r"gamma_range lower bound must be >= 1\.0"):
            compute_sensitivity_bounds(0.5, 0.1, gamma_range=(0.5, 3.0))


class TestSensitivityAnalysis:
    """Tests for sensitivity_analysis wrapper."""

    @pytest.fixture()
    def mock_dml_results(self) -> list[dict]:
        """Mock DML output with three features of varying strength."""
        return [
            {"feature": "frs_proximity", "causal_effect": 0.8, "std_error": 0.1},
            {"feature": "land_use_industrial", "causal_effect": 0.3, "std_error": 0.1},
            {"feature": "population_density", "causal_effect": 0.05, "std_error": 0.1},
        ]

    def test_all_features_analyzed(self, mock_dml_results: list[dict]) -> None:
        """Without feature filter, all features should be analyzed."""
        result = sensitivity_analysis(mock_dml_results)
        assert set(result.keys()) == {"frs_proximity", "land_use_industrial", "population_density"}
        for df in result.values():
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 21  # default n_steps

    def test_feature_filter(self, mock_dml_results: list[dict]) -> None:
        """Only specified features should be analyzed."""
        result = sensitivity_analysis(
            mock_dml_results, features=["frs_proximity", "population_density"]
        )
        assert set(result.keys()) == {"frs_proximity", "population_density"}

    def test_strong_feature_more_robust(self, mock_dml_results: list[dict]) -> None:
        """Feature with larger effect should remain significant at higher gamma."""
        result = sensitivity_analysis(mock_dml_results, gamma_range=(1.0, 5.0))

        # frs_proximity (effect=0.8) should be significant at gamma where
        # population_density (effect=0.05) is not
        strong = result["frs_proximity"]
        weak = result["population_density"]

        # Count how many gamma levels each stays significant
        strong_count = strong["significant"].sum()
        weak_count = weak["significant"].sum()
        assert strong_count > weak_count

    def test_custom_gamma_range(self, mock_dml_results: list[dict]) -> None:
        """Custom gamma range should be passed through."""
        result = sensitivity_analysis(mock_dml_results, gamma_range=(1.0, 10.0))
        df = result["frs_proximity"]
        assert df["gamma"].iloc[0] == pytest.approx(1.0)
        assert df["gamma"].iloc[-1] == pytest.approx(10.0)

    def test_empty_results(self) -> None:
        """Empty DML results should produce empty output."""
        result = sensitivity_analysis([])
        assert result == {}

    def test_skips_zero_se(self) -> None:
        """Features with zero std_error should be skipped with a warning."""
        dml = [{"feature": "bad_feat", "causal_effect": 0.5, "std_error": 0.0}]
        result = sensitivity_analysis(dml)
        assert result == {}
