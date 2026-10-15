"""Tests for the monitoring-design decomposition estimators (EST-D2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aquacontam.analysis.equity import analyze_equity
from aquacontam.analysis.monitoring_design import (
    PRIMARY_SPEC,
    REGULATED_PFAS,
    bootstrap_burden_ratio,
    burden_estimate,
    burden_ratio,
    common_reporting_limits,
    composition_share,
    detection_labels,
    first_sample_dates,
    group_masks,
    is_stable,
    mantel_haenszel_risk_ratio,
    percentile_cutoffs,
    risk_ratio,
    spec_grid,
    weighted_mean,
)


def _records(rows: list[tuple]) -> pd.DataFrame:
    cols = ["pwsid", "analyte", "source", "censored", "concentration", "detection_limit"]
    return pd.DataFrame(rows, columns=cols)


class TestBurdenRatioMatchesAnalyzeEquity:
    def test_same_value_and_groups(self) -> None:
        rng = np.random.RandomState(0)
        n = 600
        poc = rng.uniform(0.0, 100.0, n)
        poc[rng.choice(n, 40, replace=False)] = np.nan  # unknown-demographics systems
        y = (rng.uniform(size=n) < 0.15 + 0.004 * np.nan_to_num(poc)).astype(int)
        demo = pd.DataFrame({"pct_people_of_color": poc})

        report = analyze_equity(y, np.zeros(n, dtype=int), None, demo, "pct_people_of_color")
        est = burden_estimate(y, poc, percentile_cutoffs(poc))

        assert est.burden_ratio == pytest.approx(report.burden_ratio, rel=1e-12)
        assert est.n_high == report.n_high
        assert est.n_low == report.n_low
        assert est.n_with_demographics == n - 40

    def test_zero_low_rate_conventions(self) -> None:
        high = np.array([True, True, False, False])
        low = np.array([False, False, True, True])
        assert burden_ratio([1, 0, 0, 0], high, low) == math.inf
        assert burden_ratio([0, 0, 0, 0], high, low) == 1.0

    def test_missing_values_in_neither_group(self) -> None:
        high, low = group_masks([np.nan, 10.0, 90.0], (80.0, 50.0))
        assert not high[0] and not low[0]
        assert low[1] and high[2]


class TestDetectionLabels:
    def test_union_of_designs_reproduces_pooled_label(self) -> None:
        rec = _records(
            [
                ("A", "PFOS", "ucmr5", True, np.nan, 0.004),
                ("A", "PFOS", "ca_geotracker", False, 0.003, 0.002),
                ("B", "PFOS", "ucmr5", False, 0.010, 0.004),
                ("C", "PFOS", "ca_geotracker", True, np.nan, 0.002),
            ]
        )
        pooled = detection_labels(rec, analytes=["PFOS"])
        fed = detection_labels(rec, analytes=["PFOS"], sources=["ucmr5"])
        state = detection_labels(rec, analytes=["PFOS"], sources=["ca_geotracker"])
        union = pd.concat([fed, state], axis=1).max(axis=1).astype(int)
        pd.testing.assert_series_equal(union.sort_index(), pooled.sort_index(), check_names=False)

    def test_common_rl_recensors_and_drops_uninformative(self) -> None:
        rec = _records(
            [
                ("A", "PFOS", "ca_geotracker", False, 0.003, 0.002),  # below 0.004 -> non-detect
                ("B", "PFOS", "ca_geotracker", False, 0.005, 0.002),  # stays a detection
                ("C", "PFOS", "ca_geotracker", True, np.nan, 0.010),  # ND with high RL -> dropped
                ("D", "PFOS", "ca_geotracker", True, np.nan, 0.002),  # informative non-detect
                (
                    "E",
                    "PFOS",
                    "ca_geotracker",
                    False,
                    np.nan,
                    0.002,
                ),  # detect, no value -> dropped
            ]
        )
        labels = detection_labels(rec, analytes=["PFOS"], common_rl={"PFOS": 0.004})
        assert labels.to_dict() == {"A": 0, "B": 1, "D": 0}

    def test_multi_analyte_any_detection(self) -> None:
        rec = _records(
            [
                ("A", "PFOA", "ucmr5", False, 0.006, 0.004),
                ("A", "PFOS", "ucmr5", True, np.nan, 0.004),
                ("B", "PFHxS", "ucmr5", True, np.nan, 0.003),
            ]
        )
        labels = detection_labels(rec, analytes=REGULATED_PFAS)
        assert labels.to_dict() == {"A": 1, "B": 0}

    def test_common_limits_take_the_higher_native_limit(self) -> None:
        rec = _records(
            [
                ("A", "PFOS", "ucmr5", True, np.nan, 0.004),
                ("B", "PFOS", "ucmr5", True, np.nan, 0.004),
                ("C", "PFOS", "ca_geotracker", True, np.nan, 0.002),
                ("D", "HFPO-DA", "ucmr5", True, np.nan, 0.005),
            ]
        )
        limits = common_reporting_limits(rec, ["PFOS", "HFPO-DA"])
        assert limits == {"PFOS": pytest.approx(0.004), "HFPO-DA": pytest.approx(0.005)}

    def test_first_sample_dates(self) -> None:
        rec = _records([("A", "PFOS", "ca_geotracker", True, np.nan, 0.002)] * 2)
        rec["sample_date"] = pd.to_datetime(["2021-05-01", "2019-03-02"])
        first = first_sample_dates(rec, sources=["ca_geotracker"], analytes=["PFOS"])
        assert first.loc["A"] == pd.Timestamp("2019-03-02")


class TestCompositionShare:
    def _groups(self) -> tuple[np.ndarray, np.ndarray]:
        high = np.array([True] * 4 + [False] * 4)
        low = ~high
        return high, low

    def test_equal_mix_explains_nothing(self) -> None:
        high, low = self._groups()
        strata = ["F", "S", "F", "S"] * 2
        y = [1, 1, 1, 0, 1, 0, 0, 0]
        out = composition_share(y, high, low, strata)
        assert out["share"] == pytest.approx(0.0, abs=1e-12)

    def test_mix_only_disparity_is_fully_explained(self) -> None:
        # Within each stratum the high and low rates are equal; only the mix differs.
        high, low = self._groups()
        strata = ["S", "S", "S", "F", "F", "F", "F", "S"]
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        out = composition_share(y, high, low, strata)
        assert out["burden_ratio"] == pytest.approx(3.0)
        assert out["share"] == pytest.approx(1.0)

    def test_undefined_when_stratum_missing_from_high(self) -> None:
        high, low = self._groups()
        strata = ["F"] * 4 + ["F", "S", "F", "S"]
        out = composition_share([1, 0, 1, 0, 1, 0, 1, 0], high, low, strata)
        assert math.isnan(out["share"])


class TestRiskRatios:
    def test_crude_risk_ratio(self) -> None:
        exposed = [True] * 10 + [False] * 20
        outcome = [1] * 4 + [0] * 6 + [1] * 2 + [0] * 18
        assert risk_ratio(exposed, outcome) == pytest.approx(0.4 / 0.1)

    def test_mantel_haenszel_hand_computed(self) -> None:
        # Stratum 1: a=4/10 exposed, c=2/20 unexposed; stratum 2: a=3/5, c=3/15.
        exposed = [True] * 10 + [False] * 20 + [True] * 5 + [False] * 15
        outcome = [1] * 4 + [0] * 6 + [1] * 2 + [0] * 18 + [1] * 3 + [0] * 2 + [1] * 3 + [0] * 12
        strata = [1] * 30 + [2] * 20
        num = 4 * 20 / 30 + 3 * 15 / 20
        den = 2 * 10 / 30 + 3 * 5 / 20
        assert mantel_haenszel_risk_ratio(exposed, outcome, strata) == pytest.approx(num / den)

    def test_undefined_cases(self) -> None:
        assert math.isnan(risk_ratio([True, True], [1, 0]))
        assert math.isnan(risk_ratio([True, False], [1, 0]))


class TestInferenceHelpers:
    def test_bootstrap_is_seeded_and_brackets_point(self) -> None:
        rng = np.random.RandomState(1)
        n = 500
        poc = rng.uniform(0.0, 100.0, n)
        y = (rng.uniform(size=n) < 0.1 + 0.005 * poc).astype(int)
        cut = percentile_cutoffs(poc)
        a = bootstrap_burden_ratio(y, poc, cut, n_boot=300, seed=42)
        b = bootstrap_burden_ratio(y, poc, cut, n_boot=300, seed=42)
        assert a == b
        point = burden_estimate(y, poc, cut).burden_ratio
        assert a[0] < point < a[1]

    def test_stratified_bootstrap_runs(self) -> None:
        rng = np.random.RandomState(3)
        poc = rng.uniform(0.0, 100.0, 300)
        y = (rng.uniform(size=300) < 0.3).astype(int)
        lo, hi = bootstrap_burden_ratio(
            y, poc, percentile_cutoffs(poc), n_boot=100, seed=0, strata=[1, 2, 3] * 100
        )
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi

    def test_weighted_mean_ignores_non_finite(self) -> None:
        assert weighted_mean([1.0, math.inf, 3.0], [1, 5, 1]) == pytest.approx(2.0)

    def test_stability_flags(self) -> None:
        stable = burden_estimate(
            [1] * 20 + [0] * 20 + [1] * 10 + [0] * 50, [90.0] * 40 + [10.0] * 60, (80.0, 50.0)
        )
        assert is_stable(stable)
        few = burden_estimate(
            [1] * 5 + [0] * 35 + [1] * 10 + [0] * 50, [90.0] * 40 + [10.0] * 60, (80.0, 50.0)
        )
        assert not is_stable(few)


class TestSpecGrid:
    def test_thirty_six_unique_cells_one_primary(self) -> None:
        cells = spec_grid()
        keys = {(c["outcome"], c["rl"], c["targeted"], c["population"]) for c in cells}
        assert len(cells) == 36 and len(keys) == 36
        primaries = [c for c in cells if c["primary"]]
        assert len(primaries) == 1
        assert {k: primaries[0][k] for k in PRIMARY_SPEC} == PRIMARY_SPEC
