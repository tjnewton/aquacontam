"""Tests for the national monitoring-inequity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquacontam.analysis.monitoring_equity import analyze_monitoring_inequity


class TestMonitoringInequity:
    def test_detects_inequity(self) -> None:
        # High-people-of-colour systems are monitored substantially more.
        rng = np.random.RandomState(0)
        n = 800
        poc = rng.uniform(0.0, 100.0, n)
        n_samples = np.clip(5.0 + 0.4 * poc + rng.normal(0.0, 5.0, n), 1.0, None)
        demo = pd.DataFrame({"pct_people_of_color": poc})

        res = analyze_monitoring_inequity(
            n_samples, demo, group_cols=("pct_people_of_color",), n_permutations=300
        )
        r = res["pct_people_of_color"]
        assert r["monitoring_ratio"] > 1.2  # high group sampled more
        assert r["mean_high"] > r["mean_low"]
        assert r["p_value"] < 0.05
        assert r["n_high"] > 0 and r["n_low"] > 0

    def test_null_no_inequity(self) -> None:
        # Monitoring independent of demographics -> ratio near parity.
        rng = np.random.RandomState(2)
        n = 800
        poc = rng.uniform(0.0, 100.0, n)
        n_samples = np.clip(rng.normal(20.0, 5.0, n), 1.0, None)
        demo = pd.DataFrame({"pct_people_of_color": poc})

        res = analyze_monitoring_inequity(
            n_samples, demo, group_cols=("pct_people_of_color",), n_permutations=300
        )
        r = res["pct_people_of_color"]
        assert 0.85 < r["monitoring_ratio"] < 1.18

    def test_size_adjustment_reported(self) -> None:
        rng = np.random.RandomState(0)
        n = 500
        poc = rng.uniform(0.0, 100.0, n)
        pop = np.clip(1000.0 + 50.0 * poc + rng.normal(0.0, 2000.0, n), 100.0, None)
        # Monitoring is driven by population (size), not demographics directly.
        n_samples = np.clip(5.0 + 0.01 * pop + rng.normal(0.0, 3.0, n), 1.0, None)
        demo = pd.DataFrame({"pct_people_of_color": poc})

        res = analyze_monitoring_inequity(
            n_samples,
            demo,
            population=pop,
            group_cols=("pct_people_of_color",),
            n_permutations=100,
        )
        r = res["pct_people_of_color"]
        assert r["adjusted_for_population"] is True
        assert "size_adjusted_coef" in r and "size_adjusted_p" in r
        assert np.isfinite(r["size_adjusted_coef"])

    def test_missing_column_skipped(self) -> None:
        demo = pd.DataFrame({"other": list(range(20))})
        n_samples = np.arange(1, 21).astype(float)
        res = analyze_monitoring_inequity(
            n_samples, demo, group_cols=("pct_people_of_color",), n_permutations=10
        )
        assert res == {}
