"""Tests for new supplementary analyses (S18-S22 + seed inflation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# E-value computation
# ---------------------------------------------------------------------------


class TestEValue:
    """Tests for _compute_e_value."""

    def test_positive_beta(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _compute_e_value

        # VanderWeele & Ding (2017) approximate E-value for a standardized effect:
        # RR = exp(0.91 * |d|), E = RR + sqrt(RR * (RR - 1)).
        e = _compute_e_value(0.5)
        rr = np.exp(0.91 * 0.5)
        expected = rr + np.sqrt(rr * (rr - 1.0))
        assert abs(e - expected) < 1e-6

    def test_zero_beta(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _compute_e_value

        assert _compute_e_value(0.0) == 1.0

    def test_negative_beta(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _compute_e_value

        # Should use abs(beta)
        e_neg = _compute_e_value(-0.5)
        e_pos = _compute_e_value(0.5)
        assert abs(e_neg - e_pos) < 1e-6

    def test_large_beta(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _compute_e_value

        e = _compute_e_value(2.0)
        assert e > 10.0  # Large effect → large E-value


# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------


class TestClassifyFeature:
    """Tests for _classify_feature."""

    def test_spatial(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _classify_feature

        assert _classify_feature("dist_nearest_industrial") == "spatial"
        assert _classify_feature("count_industrial_10km") == "spatial"
        assert _classify_feature("kernel_density_500") == "spatial"
        assert _classify_feature("land_use_developed") == "spatial"
        assert _classify_feature("nlcd_majority") == "spatial"

    def test_system(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _classify_feature

        assert _classify_feature("population_served") == "system"
        assert _classify_feature("system_type_CWS") == "system"
        assert _classify_feature("owner_type_private") == "system"

    def test_demographic(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _classify_feature

        assert _classify_feature("pct_people_of_color") == "demographic"
        assert _classify_feature("median_income") == "demographic"

    def test_other(self) -> None:
        from aquacontam.pipeline.analysis._evaluation import _classify_feature

        assert _classify_feature("n_samples") == "other"
        assert _classify_feature("mean_detection_limit") == "other"


# ---------------------------------------------------------------------------
# Power analysis pipeline function
# ---------------------------------------------------------------------------


class TestRunPowerAnalysis:
    """Tests for _run_power_analysis."""

    def test_writes_json(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_power_analysis

        _run_power_analysis(tmp_path)
        out = tmp_path / "power_analysis.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "core_tests" in data

    def test_reads_ablation_json(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_power_analysis

        # Create mock ablation results
        ablation_data = [
            {
                "category": "proximity",
                "task": "T1",
                "n_test": 200,
                "significance": {
                    "auroc": {
                        "delta": -0.05,
                        "ci_lower": -0.08,
                        "ci_upper": -0.02,
                        "p_value": 0.001,
                    }
                },
            }
        ]
        (tmp_path / "feature_ablation.json").write_text(json.dumps(ablation_data))
        _run_power_analysis(tmp_path)
        data = json.loads((tmp_path / "power_analysis.json").read_text())
        assert len(data["ablation_tests"]) == 1
        assert data["ablation_tests"][0]["category"] == "proximity"

    def test_reads_loro_json(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_power_analysis

        loro_data = {
            "T1": {
                "xgboost": {
                    "summary": {"mean_auroc": 0.85, "std_auroc": 0.03, "n_folds": 10},
                    "per_fold": {},
                }
            }
        }
        (tmp_path / "loro_cv.json").write_text(json.dumps(loro_data))
        _run_power_analysis(tmp_path)
        data = json.loads((tmp_path / "power_analysis.json").read_text())
        assert len(data["loro_tests"]) == 1
        assert data["loro_tests"][0]["power_vs_chance"] > 0.9


# ---------------------------------------------------------------------------
# Sensitivity bounds pipeline function
# ---------------------------------------------------------------------------


class TestRunSensitivityBounds:
    """Tests for _run_sensitivity_bounds."""

    def test_skips_when_no_dml(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_sensitivity_bounds

        _run_sensitivity_bounds(tmp_path)
        assert not (tmp_path / "sensitivity_bounds.json").exists()

    def test_computes_bounds(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_sensitivity_bounds

        dml_data = {
            "per_feature": [
                {"feature": "dist_nearest_industrial", "causal_effect": 0.05, "std_error": 0.01},
                {"feature": "population_served", "causal_effect": 0.002, "std_error": 0.003},
            ]
        }
        (tmp_path / "causal_deconfounding.json").write_text(json.dumps(dml_data))
        _run_sensitivity_bounds(tmp_path)
        out = tmp_path / "sensitivity_bounds.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "dist_nearest_industrial" in data
        assert "e_value" in data["dist_nearest_industrial"]
        assert data["dist_nearest_industrial"]["e_value"] > 1.0
        assert "gamma_star" in data["dist_nearest_industrial"]
        assert "bounds_table" in data["dist_nearest_industrial"]


# ---------------------------------------------------------------------------
# Detection-only ablation pipeline function
# ---------------------------------------------------------------------------


class TestRunDetectionOnlyAblation:
    """Tests for _run_detection_only_ablation."""

    def test_skips_without_source_column(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_detection_only_ablation

        wq_df = pd.DataFrame({"pwsid": ["A", "B"], "analyte": ["PFOS", "PFOS"]})
        _run_detection_only_ablation(wq_df, [], tmp_path)
        assert not (tmp_path / "detection_only_ablation.json").exists()

    def test_computes_ablation(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_detection_only_ablation

        rng = np.random.RandomState(42)
        n = 200
        pwsids = [f"PWS{i:04d}" for i in range(n)]
        sources = rng.choice(["ucmr5", "mi_mpart", "sdwis", "ca_geotracker"], n)
        wq_df = pd.DataFrame({"pwsid": pwsids, "source": sources})

        y_true = rng.randint(0, 2, n).tolist()
        y_prob = rng.uniform(0, 1, n).tolist()

        results = [
            {
                "task": "T1",
                "model": "xgboost",
                "metadata": {
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "pwsids": pwsids,
                },
            }
        ]

        _run_detection_only_ablation(wq_df, results, tmp_path)
        out = tmp_path / "detection_only_ablation.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert "all_sources" in data[0]
        assert "excluding_detection_only" in data[0]


# ---------------------------------------------------------------------------
# Seed inflation check
# ---------------------------------------------------------------------------


class TestRunSeedInflationCheck:
    """Tests for _run_seed_inflation_check."""

    def test_skips_without_files(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_seed_inflation_check

        _run_seed_inflation_check(tmp_path)
        assert not (tmp_path / "seed_inflation_check.json").exists()

    def test_computes_inflation(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_seed_inflation_check

        ms_data = {
            "T1": {
                "xgboost": {
                    "summary": {"mean_auroc": 0.870, "std_auroc": 0.005},
                    "per_seed": {
                        "42": {"auroc": 0.873},
                        "123": {"auroc": 0.868},
                        "456": {"auroc": 0.871},
                        "789": {"auroc": 0.867},
                        "2024": {"auroc": 0.872},
                    },
                }
            }
        }
        results_data = [{"task": "T1", "model": "xgboost", "metrics": {"auroc": 0.873}}]

        (tmp_path / "multi_seed_stability.json").write_text(json.dumps(ms_data))
        (tmp_path / "results.json").write_text(json.dumps(results_data))

        _run_seed_inflation_check(tmp_path)
        out = tmp_path / "seed_inflation_check.json"
        assert out.exists()

        data = json.loads(out.read_text())
        assert "summary" in data
        assert data["summary"]["any_inflation_detected"] is False
        xgb = data["T1"]["xgboost"]
        assert xgb["reported_auroc"] == 0.873
        assert abs(xgb["mean_auroc_multi_seed"] - 0.870) < 0.01
        assert not xgb["is_inflated"]

    def test_detects_inflation(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._evaluation import _run_seed_inflation_check

        # Simulate a cherry-picked seed that's >2 SD above mean
        ms_data = {
            "T1": {
                "xgboost": {
                    "summary": {"mean_auroc": 0.850, "std_auroc": 0.005},
                    "per_seed": {
                        "42": {"auroc": 0.870},
                        "123": {"auroc": 0.848},
                        "456": {"auroc": 0.851},
                        "789": {"auroc": 0.847},
                        "2024": {"auroc": 0.852},
                    },
                }
            }
        }
        results_data = [{"task": "T1", "model": "xgboost", "metrics": {"auroc": 0.870}}]

        (tmp_path / "multi_seed_stability.json").write_text(json.dumps(ms_data))
        (tmp_path / "results.json").write_text(json.dumps(results_data))

        _run_seed_inflation_check(tmp_path)
        data = json.loads((tmp_path / "seed_inflation_check.json").read_text())
        assert data["summary"]["any_inflation_detected"] is True
        assert data["T1"]["xgboost"]["is_inflated"] is True
        assert data["T1"]["xgboost"]["inflation_z_score"] > 2.0


# ---------------------------------------------------------------------------
# MCL exceedance analysis
# ---------------------------------------------------------------------------


class TestRunMclExceedanceAnalysis:
    """Tests for _run_mcl_exceedance_analysis."""

    def test_skips_with_insufficient_data(self, tmp_path: Path) -> None:
        from aquacontam.pipeline.analysis._robustness import _run_mcl_exceedance_analysis

        wq_df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "analyte": ["OTHER"],
                "concentration": [0.1],
                "censored": [False],
            }
        )
        _run_mcl_exceedance_analysis(wq_df, [], tmp_path, seed=42)
        # Should warn about too few systems
        assert not (tmp_path / "mcl_exceedance_analysis.json").exists()


# ---------------------------------------------------------------------------
# MCL threshold module (core function)
# ---------------------------------------------------------------------------


class TestMclExceedance:
    """Tests for the core compute_mcl_exceedance function."""

    def test_basic_exceedance(self) -> None:
        from aquacontam.analysis.mcl_threshold import compute_mcl_exceedance

        wq_df = pd.DataFrame(
            {
                "pwsid": ["A", "A", "B", "B"],
                "analyte": ["PFOS", "PFOA", "PFOS", "PFOA"],
                "concentration": [0.010, 0.001, 0.001, 0.001],
                "censored": [False, False, True, True],
            }
        )
        result = compute_mcl_exceedance(wq_df)
        assert len(result) == 2
        a_row = result[result["pwsid"] == "A"].iloc[0]
        assert a_row["mcl_exceedance"] == 1
        assert a_row["n_analytes_exceeding"] >= 1
        b_row = result[result["pwsid"] == "B"].iloc[0]
        assert b_row["mcl_exceedance"] == 0

    def test_all_censored(self) -> None:
        from aquacontam.analysis.mcl_threshold import compute_mcl_exceedance

        wq_df = pd.DataFrame(
            {
                "pwsid": ["C", "C"],
                "analyte": ["PFOS", "PFOA"],
                "concentration": [0.001, 0.001],
                "censored": [True, True],
            }
        )
        result = compute_mcl_exceedance(wq_df)
        assert result.iloc[0]["mcl_exceedance"] == 0

    def test_no_regulated_analytes(self) -> None:
        from aquacontam.analysis.mcl_threshold import compute_mcl_exceedance

        wq_df = pd.DataFrame(
            {
                "pwsid": ["D"],
                "analyte": ["UNKNOWN_PFAS"],
                "concentration": [100.0],
                "censored": [False],
            }
        )
        result = compute_mcl_exceedance(wq_df)
        assert result.empty


# ---------------------------------------------------------------------------
# Power analysis core functions
# ---------------------------------------------------------------------------


class TestPowerAnalysisCore:
    """Tests for core power analysis functions."""

    def test_auroc_power_high(self) -> None:
        from aquacontam.analysis.power_analysis import compute_auroc_comparison_power

        result = compute_auroc_comparison_power(0.90, 0.70, 2000, 2000)
        assert result["power"] > 0.99

    def test_auroc_power_low(self) -> None:
        from aquacontam.analysis.power_analysis import compute_auroc_comparison_power

        result = compute_auroc_comparison_power(0.85, 0.84, 100, 100)
        assert result["power"] < 0.3

    def test_proportion_power(self) -> None:
        from aquacontam.analysis.power_analysis import compute_proportion_test_power

        result = compute_proportion_test_power(0.25, 0.06, 300, 800)
        assert result["power"] > 0.99

    def test_regression_power(self) -> None:
        from aquacontam.analysis.power_analysis import compute_regression_coefficient_power

        result = compute_regression_coefficient_power(0.04, 0.015, 14000)
        assert result["power"] > 0.70


# ---------------------------------------------------------------------------
# Rosenbaum sensitivity bounds
# ---------------------------------------------------------------------------


class TestSensitivityBoundsCore:
    """Tests for Rosenbaum sensitivity bounds."""

    def test_gamma_1_preserves_significance(self) -> None:
        from aquacontam.analysis.causal_deconfounding import compute_sensitivity_bounds

        bounds = compute_sensitivity_bounds(0.05, 0.01)
        # At gamma=1, no confounding — should remain significant
        row_1 = bounds[bounds["gamma"] == 1.0].iloc[0]
        assert bool(row_1["significant"]) is True

    def test_large_gamma_overturns(self) -> None:
        from aquacontam.analysis.causal_deconfounding import compute_sensitivity_bounds

        # Weak effect (z=2.5) should be overturned at high gamma
        bounds = compute_sensitivity_bounds(0.025, 0.01, gamma_range=(1.0, 5.0))
        last_row = bounds.iloc[-1]
        assert bool(last_row["significant"]) is False

    def test_invalid_se(self) -> None:
        from aquacontam.analysis.causal_deconfounding import compute_sensitivity_bounds

        with pytest.raises(ValueError, match="std_error must be positive"):
            compute_sensitivity_bounds(0.05, 0.0)
