"""Coordinate-rescue regression tests for the split-strategy comparison.

Regression guard for the harness bug where ``run_split_ablation`` /
``run_split_comparison_matrix`` / ``run_coordinate_sensitivity`` assigned EPA
regions on already-aggregated targets (no lat/lon), so the coordinate fallback
could not fire and systems with non-state PWSID prefixes (e.g. WQP synthetic
IDs) were silently dropped from the geographic split — the same bug class
fixed for the other analysis harnesses in commit b17c46a, which left
``split_comparison.json`` on the pre-fix 2,241-system test population.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import aquacontam.features.assembly as feature_assembly
from aquacontam.features.assembly import (
    aggregate_to_system_level,
    derive_system_epa_regions,
    drop_leakage_columns,
)
from aquacontam.preprocessing.splits import assign_epa_region

# Patch target: assign_epa_region imports latlon_to_epa_region from here.
_LATLON = "aquacontam.preprocessing._region_lookup.latlon_to_epa_region"

N_WQP = 6
_TEST_REGIONS = (8, 9, 10)


def _with_wqp_systems(
    wq_df: pd.DataFrame, feature_dfs: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """Clone CA donor systems into WQP_* synthetic-ID systems (coords kept)."""
    wqp_ids = [f"WQP_{i:07d}" for i in range(N_WQP)]
    donor_ids = [p for p in wq_df["pwsid"].unique() if p.startswith("CA")][:N_WQP]
    extra = []
    for wqp, donor in zip(wqp_ids, donor_ids, strict=True):
        rows = wq_df[wq_df["pwsid"] == donor].copy()
        rows["pwsid"] = wqp
        extra.append(rows)
    wq_out = pd.concat([wq_df, *extra], ignore_index=True)

    fdfs_out = []
    for df in feature_dfs:
        donors_present = [d for d in donor_ids if d in df.index]
        donor_rows = df.loc[donors_present].copy()
        donor_rows.index = pd.Index(wqp_ids[: len(donor_rows)], name="pwsid")
        fdfs_out.append(pd.concat([df, donor_rows]))
    return wq_out, fdfs_out, wqp_ids


def test_derive_system_epa_regions_rescues_coordinates() -> None:
    wq_df = pd.DataFrame(
        {
            "pwsid": ["TX1234567", "CA0101001", "WQP_0000001"],
            "latitude": [31.0, 34.05, 39.74],
            "longitude": [-99.0, -118.24, -104.99],
        }
    )
    with patch(_LATLON, return_value=9):
        regions = derive_system_epa_regions(wq_df)

    assert regions.loc["TX1234567"] == 6  # prefix path, not overwritten
    assert regions.loc["WQP_0000001"] == 9  # rescued via coordinates
    aligned = derive_system_epa_regions
    with patch(_LATLON, return_value=9):
        idx = pd.Index(["WQP_0000001", "ZZ9999999"], name="pwsid")
        out = aligned(wq_df, idx)
    assert out.loc["WQP_0000001"] == 9
    assert pd.isna(out.loc["ZZ9999999"])  # unknown system stays NaN


def test_matrix_geographic_arm_includes_rescued_systems(
    synthetic_wq_df, synthetic_feature_dfs
) -> None:
    from aquacontam.analysis.split_strategy_comparison import run_split_comparison_matrix
    from aquacontam.models.logistic import LogisticRegressionClassifier

    wq_df, fdfs, _wqp_ids = _with_wqp_systems(synthetic_wq_df, synthetic_feature_dfs)

    with patch(_LATLON, return_value=9):
        expected_test = int(derive_system_epa_regions(wq_df).isin(_TEST_REGIONS).sum())
        result = run_split_comparison_matrix(
            wq_df,
            fdfs,
            model_specs=[("LogReg", LogisticRegressionClassifier, {})],
            n_bootstrap=0,
            compute_delong=False,
        )

    geo_rows = [r for r in result["results"] if r["split_strategy"] == "geographic"]
    assert geo_rows, "matrix produced no geographic rows"
    # CO + CA + WA donors (30) plus the rescued WQP systems.
    assert expected_test == 30 + N_WQP
    assert all(r["n_test"] == expected_test for r in geo_rows)


def test_rescued_population_exceeds_legacy_derivation(
    synthetic_wq_df, synthetic_feature_dfs
) -> None:
    """The aggregated-targets derivation (the old bug) drops WQP systems."""
    wq_df, _fdfs, wqp_ids = _with_wqp_systems(synthetic_wq_df, synthetic_feature_dfs)
    sys_targets = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS"))

    legacy = assign_epa_region(sys_targets.reset_index()).set_index("pwsid")["epa_region"]
    legacy = pd.to_numeric(legacy, errors="coerce")
    with patch(_LATLON, return_value=9):
        rescued = derive_system_epa_regions(wq_df, sys_targets.index)

    assert legacy.loc[wqp_ids].isna().all()  # old path: dropped
    assert rescued.loc[wqp_ids].notna().all()  # new path: rescued
    assert int(rescued.isin(_TEST_REGIONS).sum()) == int(legacy.isin(_TEST_REGIONS).sum()) + N_WQP


def test_size_matched_arm_train_size_equals_geographic(
    synthetic_wq_df, synthetic_feature_dfs
) -> None:
    from aquacontam.analysis.split_strategy_comparison import run_split_comparison_matrix
    from aquacontam.models.logistic import LogisticRegressionClassifier

    with patch(_LATLON, return_value=9):
        result = run_split_comparison_matrix(
            synthetic_wq_df,
            synthetic_feature_dfs,
            model_specs=[("LogReg", LogisticRegressionClassifier, {})],
            n_bootstrap=0,
            compute_delong=False,
            size_matched=True,
        )

    by_split: dict[str, list[dict]] = {}
    for r in result["results"]:
        by_split.setdefault(r["split_strategy"], []).append(r)
    assert "random_sizematched" in by_split
    geo_train = {r["feature_set"]: r["n_train"] for r in by_split["geographic"]}
    for r in by_split["random_sizematched"]:
        assert r["n_train"] == geo_train[r["feature_set"]]
        assert len(r["fold_metrics"]) == 5


def test_split_functions_consume_wq_df(
    synthetic_wq_df, synthetic_feature_dfs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threading guard: both entry points must derive regions from wq_df."""
    from aquacontam.analysis.split_strategy_comparison import (
        run_split_ablation,
        run_split_comparison_matrix,
    )
    from aquacontam.models.logistic import LogisticRegressionClassifier

    calls: list[int] = []
    real = feature_assembly.derive_system_epa_regions

    def spy(wq_df: pd.DataFrame, index: pd.Index | None = None) -> pd.Series:
        calls.append(len(wq_df))
        return real(wq_df, index)

    monkeypatch.setattr(feature_assembly, "derive_system_epa_regions", spy)

    run_split_ablation(synthetic_wq_df, synthetic_feature_dfs, seed=42)
    assert len(calls) == 1
    run_split_comparison_matrix(
        synthetic_wq_df,
        synthetic_feature_dfs,
        model_specs=[("LogReg", LogisticRegressionClassifier, {})],
        n_bootstrap=0,
        compute_delong=False,
    )
    assert len(calls) == 2
    # Both calls received the sample-level frame, not the aggregated targets.
    assert all(n == len(synthetic_wq_df) for n in calls)


def test_coordinate_sensitivity_includes_rescued_systems(
    synthetic_wq_df, synthetic_feature_dfs, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aquacontam.preprocessing.splits as splits_mod
    from aquacontam.analysis.coordinate_sensitivity import run_coordinate_sensitivity
    from aquacontam.models.logistic import LogisticRegressionClassifier

    wq_df, fdfs, wqp_ids = _with_wqp_systems(synthetic_wq_df, synthetic_feature_dfs)

    captured: list[pd.DataFrame] = []
    real_split = splits_mod.geographic_split

    def spy(df: pd.DataFrame, *args, **kwargs):
        captured.append(df)
        return real_split(df, *args, **kwargs)

    monkeypatch.setattr(splits_mod, "geographic_split", spy)

    with patch(_LATLON, return_value=9):
        results = run_coordinate_sensitivity(
            LogisticRegressionClassifier,
            {},
            wq_df,
            feature_dfs=fdfs,
            magnitudes_km=(0.0,),
            n_seeds=1,
        )

    assert results, "coordinate sensitivity produced no results"
    assert captured, "geographic_split was not called"
    frame = captured[0].set_index("pwsid")
    regions = pd.to_numeric(frame.loc[wqp_ids, "epa_region"], errors="coerce")
    assert regions.notna().all()  # rescued systems enter the split
