"""Tests for MCL-threshold exceedance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.mcl_threshold import (
    DEFAULT_MCL_THRESHOLDS,
    compute_mcl_exceedance,
    mcl_summary_statistics,
)


@pytest.fixture()
def sample_wq_data() -> pd.DataFrame:
    """Create synthetic water quality data with known MCL exceedances."""
    rows = [
        # System A: PFOS above MCL (0.010 > 0.004)
        {"pwsid": "AA0000001", "analyte": "PFOS", "concentration": 0.010, "censored": False},
        {"pwsid": "AA0000001", "analyte": "PFOA", "concentration": 0.002, "censored": False},
        # System B: all below MCL
        {"pwsid": "BB0000002", "analyte": "PFOS", "concentration": 0.001, "censored": False},
        {"pwsid": "BB0000002", "analyte": "PFOA", "concentration": 0.001, "censored": False},
        # System C: censored only (non-detect)
        {"pwsid": "CC0000003", "analyte": "PFOS", "concentration": np.nan, "censored": True},
        # System D: PFOA above MCL, PFOS censored
        {"pwsid": "DD0000004", "analyte": "PFOS", "concentration": np.nan, "censored": True},
        {"pwsid": "DD0000004", "analyte": "PFOA", "concentration": 0.050, "censored": False},
        # System E: non-regulated analyte only (should be excluded)
        {"pwsid": "EE0000005", "analyte": "lithium", "concentration": 5.0, "censored": False},
    ]
    return pd.DataFrame(rows)


class TestComputeMclExceedance:
    def test_basic_exceedance(self, sample_wq_data: pd.DataFrame) -> None:
        result = compute_mcl_exceedance(sample_wq_data)
        # Should have 4 systems (A, B, C, D — E excluded since lithium not regulated)
        assert len(result) == 4
        # System A exceeds MCL (PFOS = 0.010 > 0.004)
        sys_a = result[result["pwsid"] == "AA0000001"].iloc[0]
        assert sys_a["mcl_exceedance"] == 1
        assert sys_a["n_analytes_exceeding"] == 1

    def test_below_mcl(self, sample_wq_data: pd.DataFrame) -> None:
        result = compute_mcl_exceedance(sample_wq_data)
        sys_b = result[result["pwsid"] == "BB0000002"].iloc[0]
        assert sys_b["mcl_exceedance"] == 0
        assert sys_b["n_analytes_exceeding"] == 0

    def test_censored_system(self, sample_wq_data: pd.DataFrame) -> None:
        result = compute_mcl_exceedance(sample_wq_data)
        sys_c = result[result["pwsid"] == "CC0000003"].iloc[0]
        assert sys_c["mcl_exceedance"] == 0  # censored = non-detect = below MCL

    def test_mixed_censored_exceed(self, sample_wq_data: pd.DataFrame) -> None:
        result = compute_mcl_exceedance(sample_wq_data)
        sys_d = result[result["pwsid"] == "DD0000004"].iloc[0]
        assert sys_d["mcl_exceedance"] == 1  # PFOA exceeds
        assert sys_d["n_analytes_exceeding"] == 1

    def test_max_ratio(self, sample_wq_data: pd.DataFrame) -> None:
        result = compute_mcl_exceedance(sample_wq_data)
        sys_a = result[result["pwsid"] == "AA0000001"].iloc[0]
        # PFOS: 0.010 / 0.004 = 2.5; PFOA: 0.002 / 0.004 = 0.5
        assert abs(sys_a["max_pfas_ratio"] - 2.5) < 0.01

    def test_custom_thresholds(self, sample_wq_data: pd.DataFrame) -> None:
        # Very high threshold — nothing should exceed
        result = compute_mcl_exceedance(
            sample_wq_data, mcl_thresholds={"PFOS": 100.0, "PFOA": 100.0}
        )
        assert result["mcl_exceedance"].sum() == 0

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"pwsid": ["A"], "analyte": ["PFOS"]})
        with pytest.raises(ValueError, match="Missing columns"):
            compute_mcl_exceedance(df)

    def test_empty_regulated(self) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "analyte": ["lithium"],
                "concentration": [5.0],
                "censored": [False],
            }
        )
        result = compute_mcl_exceedance(df)
        assert len(result) == 0


class TestMclSummaryStatistics:
    def test_basic_summary(self, sample_wq_data: pd.DataFrame) -> None:
        mcl_df = compute_mcl_exceedance(sample_wq_data)
        summary = mcl_summary_statistics(mcl_df)
        assert summary["n_systems"] == 4
        assert summary["n_exceeding"] == 2  # A and D
        assert summary["exceedance_rate"] == 0.5

    def test_empty_dataframe(self) -> None:
        summary = mcl_summary_statistics(pd.DataFrame())
        assert summary["n_systems"] == 0


class TestDefaultThresholds:
    def test_pfos_mcl(self) -> None:
        assert DEFAULT_MCL_THRESHOLDS["PFOS"] == 0.004

    def test_pfoa_mcl(self) -> None:
        assert DEFAULT_MCL_THRESHOLDS["PFOA"] == 0.004

    def test_five_regulated_analytes(self) -> None:
        assert len(DEFAULT_MCL_THRESHOLDS) == 5
