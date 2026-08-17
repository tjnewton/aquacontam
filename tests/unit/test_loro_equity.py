"""Tests for LORO cross-validated equity analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def loro_equity_wq_df() -> pd.DataFrame:
    """Synthetic water quality data spanning 4 EPA regions with demographic signal."""
    rng = np.random.RandomState(42)

    # 4 regions with enough systems each (15 per region = 60 total)
    state_region = [
        ("CT", 1, 42.0, -72.0),
        ("NJ", 2, 40.7, -74.0),
        ("PA", 3, 39.0, -76.5),
        ("GA", 4, 33.7, -84.4),
    ]

    rows = []
    for state, _region, lat, lon in state_region:
        for i in range(15):
            pwsid = f"{state}{i + 1:07d}"
            detected = rng.random() < 0.3
            for _ in range(3):
                rows.append(
                    {
                        "pwsid": pwsid,
                        "analyte": "PFOS",
                        "concentration": rng.uniform(5.0, 100.0) if detected else 0.0,
                        "unit": "ng/L",
                        "censored": not detected,
                        "detection_limit": 2.0,
                        "sample_date": pd.Timestamp("2023-01-15"),
                        "latitude": lat + rng.normal(0, 0.3),
                        "longitude": lon + rng.normal(0, 0.3),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def loro_equity_demo_df(loro_equity_wq_df: pd.DataFrame) -> pd.DataFrame:
    """Demographics DataFrame aligned with water quality systems."""
    rng = np.random.RandomState(123)
    pws = loro_equity_wq_df["pwsid"].unique()
    return pd.DataFrame(
        {
            "pct_people_of_color": rng.uniform(0.1, 0.9, len(pws)),
            "pct_low_income": rng.uniform(0.1, 0.5, len(pws)),
            "pct_limited_english": rng.uniform(0.0, 0.3, len(pws)),
            "pct_less_hs_education": rng.uniform(0.05, 0.4, len(pws)),
        },
        index=pd.Index(pws, name="pwsid"),
    )


class TestLOROEquityAnalysis:
    """Tests for _run_loro_equity_analysis pipeline function."""

    def test_output_structure(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
        loro_equity_demo_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [loro_equity_demo_df],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=5,
        )

        out_path = tmp_path / "loro_equity_analysis.json"
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert "per_region" in data
        assert "summary" in data
        assert "metadata" in data
        assert isinstance(data["per_region"], list)
        assert isinstance(data["summary"], dict)

    def test_per_region_entries_have_required_keys(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
        loro_equity_demo_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [loro_equity_demo_df],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=5,
        )

        data = json.loads((tmp_path / "loro_equity_analysis.json").read_text())
        if data["per_region"]:
            entry = data["per_region"][0]
            assert "region" in entry
            assert "group" in entry
            assert "burden_ratio" in entry
            assert "p_value" in entry
            assert "p_value_fdr" in entry
            assert "n_high" in entry
            assert "n_low" in entry
            assert "n_systems" in entry

    def test_skips_small_regions(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
        loro_equity_demo_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        # Set min_systems very high so all regions are skipped
        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [loro_equity_demo_df],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=999,
        )

        # Should not produce output when all regions skipped
        out_path = tmp_path / "loro_equity_analysis.json"
        assert not out_path.exists()

    def test_fdr_correction_applied(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
        loro_equity_demo_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [loro_equity_demo_df],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=5,
        )

        out_path = tmp_path / "loro_equity_analysis.json"
        if not out_path.exists():
            pytest.skip("No regions produced results")

        data = json.loads(out_path.read_text())
        for entry in data["per_region"]:
            raw = entry["p_value"]
            fdr = entry["p_value_fdr"]
            # FDR-corrected p-values >= raw p-values
            if raw is not None and fdr is not None:
                assert fdr >= raw - 1e-10

    def test_skips_without_demographics(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        # Pass feature_dfs without demographics
        no_demo = pd.DataFrame(
            {"some_feature": [1.0] * len(loro_equity_wq_df["pwsid"].unique())},
            index=pd.Index(loro_equity_wq_df["pwsid"].unique(), name="pwsid"),
        )

        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [no_demo],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=5,
        )

        assert not (tmp_path / "loro_equity_analysis.json").exists()

    def test_summary_statistics(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
        loro_equity_demo_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [loro_equity_demo_df],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=5,
        )

        out_path = tmp_path / "loro_equity_analysis.json"
        if not out_path.exists():
            pytest.skip("No regions produced results")

        data = json.loads(out_path.read_text())
        summary = data["summary"]

        # Should have summary for at least one demographic group
        assert len(summary) > 0

        for _group, stats in summary.items():
            assert "n_regions_analyzed" in stats
            assert "median_burden_ratio" in stats
            assert "mean_burden_ratio" in stats
            assert "n_regions_significant_fdr" in stats
            assert stats["n_regions_analyzed"] >= 1

    def test_metadata_correct(
        self,
        tmp_path: Path,
        loro_equity_wq_df: pd.DataFrame,
        loro_equity_demo_df: pd.DataFrame,
    ) -> None:
        from aquacontam.pipeline.analysis import _run_loro_equity_analysis

        _run_loro_equity_analysis(
            loro_equity_wq_df,
            [loro_equity_demo_df],
            tmp_path,
            seed=42,
            n_permutations=10,
            min_systems=5,
        )

        out_path = tmp_path / "loro_equity_analysis.json"
        if not out_path.exists():
            pytest.skip("No regions produced results")

        data = json.loads(out_path.read_text())
        meta = data["metadata"]
        assert meta["n_permutations"] == 10
        assert meta["min_systems"] == 5
        assert meta["fdr_method"] == "fdr_bh"
        assert meta["seed"] == 42
        assert meta["model"] == "XGBoostClassifier"
