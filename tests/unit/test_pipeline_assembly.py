"""Tests for ``assemble_with_split_imputation`` coordinate-rescued regions.

Regression guard for the harness bug where region assignment on
already-aggregated targets (no lat/lon) silently dropped geocoded systems whose
PWSID prefix is not a state code (e.g. WQP synthetic IDs), making the analysis
harnesses (LORO, multi-seed, equity, DML) evaluate a smaller test population
than the canonical Table 2 benchmark.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from aquacontam.pipeline.assembly import assemble_with_split_imputation

# Patch target: assign_epa_region imports latlon_to_epa_region from here.
_LATLON = "aquacontam.preprocessing._region_lookup.latlon_to_epa_region"


def _make_inputs() -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    # TX -> region 6 (train), CA -> region 9 (test), WQP synthetic ID whose
    # prefix "WQ" is not a state code (rescued only via coordinates).
    pwsids = ["TX1234567", "CA0101001", "WQP_0000001"]
    sys_targets = pd.DataFrame({"target": [1, 0, 1]}, index=pd.Index(pwsids, name="pwsid"))
    feat = pd.DataFrame(
        {"feat_a": [0.3, 0.5, 0.7], "feat_b": [1.0, 1.5, 2.0]},
        index=pd.Index(pwsids, name="pwsid"),
    )
    wq_df = pd.DataFrame(
        {
            "pwsid": pwsids,
            "latitude": [31.0, 34.05, 39.74],
            "longitude": [-99.0, -118.24, -104.99],
        }
    )
    return sys_targets, [feat], wq_df


def test_wq_df_rescues_non_state_pwsid_via_coordinates() -> None:
    sys_targets, fdfs, wq_df = _make_inputs()

    # latlon_to_epa_region is exercised separately; patch it so this test
    # isolates the assemble-side rescue logic and stays CI-safe. Rescue the WQP
    # system to a train region so the feature matrix is non-degenerate.
    with patch(_LATLON, return_value=1):
        _X, _y, _stats, regions = assemble_with_split_imputation(
            sys_targets.copy(), fdfs, wq_df=wq_df
        )

    reg = pd.to_numeric(regions, errors="coerce")
    assert reg.notna().all()  # all three systems retain a region
    assert not pd.isna(reg.get("WQP_0000001"))  # rescued, not dropped


def test_without_wq_df_non_state_pwsid_is_dropped() -> None:
    sys_targets, fdfs, _wq_df = _make_inputs()

    # No coordinate fallback without wq_df → the WQP system gets NaN region.
    _X, _y, _stats, regions = assemble_with_split_imputation(sys_targets.copy(), fdfs)

    reg = pd.to_numeric(regions, errors="coerce")
    assert pd.isna(reg.get("WQP_0000001"))
    # The state-prefixed systems are still assigned.
    assert not pd.isna(reg.get("TX1234567"))


def test_wq_df_path_includes_more_systems_than_legacy() -> None:
    sys_targets, fdfs, wq_df = _make_inputs()

    with patch(_LATLON, return_value=1):
        X_new, _y, _s, _r = assemble_with_split_imputation(sys_targets.copy(), fdfs, wq_df=wq_df)
    X_old, _y2, _s2, _r2 = assemble_with_split_imputation(sys_targets.copy(), fdfs)

    assert "WQP_0000001" in X_new.index
    assert "WQP_0000001" not in X_old.index
    assert len(X_new) > len(X_old)
