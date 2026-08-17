"""Tests for the paper-side polygon region-mismatch flag (``paper/_geo_flags.py``).

The flag drives the Extended Data Fig. 1 split-map exclusion and the
``misgeocode_sensitivity_v2.json`` compute. The polygon rule replaced a loose
bounding-box rule that falsely flagged correctly-located border systems
(band regression cases below); the legacy rule is retained as the frozen-v1
reproduction witness and is pinned here on a small fixture.

The city matrix doubles as a lon/lat-order-swap detector: a swapped call
misclassifies every case at once.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("shapely")

from paper._geo_flags import (
    _NEAREST_MAX_DEG,
    _load_state_polygons,
    coord_epa_region_series,
    figure_plot_regions,
    region_mismatch_mask,
    region_mismatch_mask_bbox_legacy,
)


def _coords(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["pwsid", "latitude", "longitude"])


# ---------------------------------------------------------------------------
# Asset integrity
# ---------------------------------------------------------------------------


class TestStatePolygonAsset:
    def test_loads_56_features_all_valid(self) -> None:
        tree, stusps, regions = _load_state_polygons()
        assert len(stusps) == 56
        assert len(regions) == 56
        assert all(g.is_valid for g in tree.geometries)

    def test_states_map_to_regions_territories_may_skip(self) -> None:
        _tree, stusps, regions = _load_state_polygons()
        by_code = dict(zip(stusps, regions, strict=True))
        # All 50 states + DC must map; AS/MP are outside EPA_REGIONS and stay None.
        assert by_code["CA"] == 9
        assert by_code["MO"] == 7
        assert by_code["DC"] == 3
        assert by_code["AS"] is None
        assert by_code["MP"] is None
        n_mapped = sum(1 for r in regions if r is not None)
        assert n_mapped == 54  # 50 states + DC + PR + VI + GU


# ---------------------------------------------------------------------------
# Point -> region classification (the bbox rule provably fails these)
# ---------------------------------------------------------------------------


class TestCoordRegionClassification:
    @pytest.mark.parametrize(
        ("lat", "lon", "region"),
        [
            (38.63, -90.20, 7),  # St. Louis, MO (R7) - inside IL's bbox
            (38.62, -90.15, 5),  # East St. Louis, IL (R5) - across the river
            (39.10, -84.51, 5),  # Cincinnati, OH (R5) - inside KY-ish bboxes
            (39.08, -84.51, 4),  # Covington, KY (R4) - across the Ohio River
            (39.95, -75.16, 3),  # Philadelphia, PA (R3) - inside NJ's bbox
            (39.93, -75.12, 2),  # Camden, NJ (R2) - across the Delaware
            (30.27, -97.74, 6),  # Austin, TX (R6)
            (47.61, -122.33, 10),  # Seattle, WA (R10)
        ],
    )
    def test_city_matrix(self, lat: float, lon: float, region: int) -> None:
        got = coord_epa_region_series(_coords([("XX000001", lat, lon)]))
        assert got.iloc[0] == region

    def test_offshore_resolves_to_nearest_state(self) -> None:
        # ~15 km off the NJ shore in the Atlantic: no polygon covers it.
        got = coord_epa_region_series(_coords([("XX000001", 39.30, -73.90)]))
        assert got.iloc[0] == 2  # NJ, R2

    def test_far_offshore_is_nan(self) -> None:
        # Mid-Atlantic, farther than _NEAREST_MAX_DEG from every state.
        got = coord_epa_region_series(_coords([("XX000001", 33.0, -65.0)]))
        assert pd.isna(got.iloc[0])
        assert _NEAREST_MAX_DEG < 5.0  # sanity: the point above is ~7 deg out


# ---------------------------------------------------------------------------
# The mismatch flag
# ---------------------------------------------------------------------------


class TestRegionMismatchMask:
    def test_border_system_in_own_state_not_flagged(self) -> None:
        """Band regression: bbox rule flagged St. Louis MO systems via IL's box."""
        df = _coords([("MO1234567", 38.63, -90.20)])
        assert not region_mismatch_mask(df).iloc[0]
        # ... and the legacy rule demonstrably got this wrong (IL box wins).
        assert region_mismatch_mask_bbox_legacy(df).iloc[0]

    def test_true_misgeocode_flagged(self) -> None:
        df = _coords([("MO1234567", 41.88, -87.63)])  # Chicago, IL (R5)
        assert region_mismatch_mask(df).iloc[0]

    def test_water_jittered_foreign_system_flagged(self) -> None:
        """Water-leak regression: mid-Lake-Michigan point must still resolve."""
        df = _coords([("CA1234567", 43.5, -87.0)])  # open Lake Michigan
        assert region_mismatch_mask(df).iloc[0]

    def test_coastal_jitter_own_state_not_flagged(self) -> None:
        df = _coords([("NJ1234567", 39.30, -73.90)])  # off the NJ shore
        assert not region_mismatch_mask(df).iloc[0]

    def test_far_offshore_never_flagged(self) -> None:
        df = _coords([("CA1234567", 33.0, -65.0)])  # beyond the tolerance
        assert not region_mismatch_mask(df).iloc[0]

    def test_non_state_prefix_never_flagged(self) -> None:
        df = _coords([("WQP_00001", 41.88, -87.63), ("09123456", 38.63, -90.20)])
        assert not region_mismatch_mask(df).any()

    def test_same_region_cross_state_not_flagged(self) -> None:
        df = _coords([("KS1234567", 39.10, -94.58)])  # Kansas City, MO side (both R7)
        assert not region_mismatch_mask(df).iloc[0]


# ---------------------------------------------------------------------------
# Legacy witness: pinned behavior on a fixture (gate-B contract)
# ---------------------------------------------------------------------------


class TestLegacyBboxWitness:
    def test_legacy_reproduces_bbox_semantics(self) -> None:
        df = _coords(
            [
                ("MO0000001", 38.63, -90.20),  # in IL's bbox -> flagged (false positive)
                ("IL0000001", 41.88, -87.63),  # Chicago, own state -> not flagged
                ("MO0000002", 41.88, -87.63),  # true misgeocode -> flagged
                ("WQP_00001", 38.63, -90.20),  # no prefix region -> never flagged
                ("CA0000001", 33.0, -65.0),  # outside every bbox -> never flagged
            ]
        )
        got = region_mismatch_mask_bbox_legacy(df)
        assert got.tolist() == [True, False, True, False, False]


# ---------------------------------------------------------------------------
# Figure coloring helper
# ---------------------------------------------------------------------------


class TestFigurePlotRegions:
    def test_prefix_wins_over_coordinates(self) -> None:
        # A mis-geocoded MO system is still colored R7 (split truth) --
        # the exclusion mask, not recoloring, handles these on the map.
        df = _coords([("MO1234567", 41.88, -87.63)])
        assert figure_plot_regions(df).iloc[0] == 7

    def test_non_state_prefix_colored_by_polygon(self) -> None:
        df = _coords(
            [
                ("WQP_00001", 38.63, -90.20),  # St. Louis -> R7 (bbox said R5)
                ("09123456", 34.05, -118.24),  # LA -> R9
                ("XX000001", 33.0, -65.0),  # far offshore -> NaN (unplotted)
            ]
        )
        got = figure_plot_regions(df)
        assert got.iloc[0] == 7
        assert got.iloc[1] == 9
        assert pd.isna(got.iloc[2])
