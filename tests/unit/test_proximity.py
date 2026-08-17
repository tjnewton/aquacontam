"""Tests for features.proximity — Tier 1 proximity features."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE
from aquacontam.features.proximity import (
    compute_proximity_features,
    extract_all_proximity_features,
)


def _make_facilities_gdf(
    lons: list[float],
    lats: list[float],
    sic_codes: list[str] | None = None,
    naics_codes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)]
    data: dict[str, list[str] | list[float]] = {
        "facility_name": [f"Facility_{i}" for i in range(len(lons))],
        "latitude": lats,
        "longitude": lons,
    }
    if sic_codes is not None:
        data["sic_codes"] = sic_codes
    if naics_codes is not None:
        data["naics_codes"] = naics_codes
    return gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_STORAGE)


class TestComputeProximityFeatures:
    def test_output_columns_match_pattern(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        facilities = _make_facilities_gdf([-97.73], [30.27])
        result = compute_proximity_features(
            systems, facilities, "industrial", radii_m=(1000.0, 5000.0)
        )
        expected_cols = {
            "dist_nearest_industrial",
            "log1p_dist_nearest_industrial",
            "n_industrial_1000m",
            "n_industrial_5000m",
        }
        assert set(result.columns) == expected_cols

    def test_nearest_distance_is_positive(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        facilities = _make_facilities_gdf([-97.73, -97.72], [30.27, 30.28])
        result = compute_proximity_features(systems, facilities, "test")
        assert result["dist_nearest_test"].iloc[0] > 0

    def test_counts_monotonically_increase(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        facilities = _make_facilities_gdf(
            [-97.735, -97.72, -97.65],
            [30.27, 30.27, 30.27],
        )
        result = compute_proximity_features(
            systems, facilities, "test", radii_m=(1000.0, 5000.0, 50000.0)
        )
        c1 = result["n_test_1000m"].iloc[0]
        c5 = result["n_test_5000m"].iloc[0]
        c50 = result["n_test_50000m"].iloc[0]
        assert c1 <= c5 <= c50

    def test_empty_facilities(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        facilities = _make_facilities_gdf([], [])
        result = compute_proximity_features(systems, facilities, "empty")
        assert result["dist_nearest_empty"].iloc[0] == np.inf
        assert result["n_empty_1000m"].iloc[0] == 0

    def test_multiple_systems(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(
            ["TX0000001", "CA0000001"],
            [-97.74, -118.24],
            [30.27, 34.05],
        )
        # Facility near TX only
        facilities = _make_facilities_gdf([-97.73], [30.27])
        result = compute_proximity_features(systems, facilities, "test")
        assert result["dist_nearest_test"].iloc[0] < result["dist_nearest_test"].iloc[1]

    def test_index_is_pwsid(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001", "CA0000001"], [-97.74, -118.24], [30.27, 34.05])
        facilities = _make_facilities_gdf([-97.73], [30.27])
        result = compute_proximity_features(systems, facilities, "test")
        assert result.index.name == "pwsid"
        assert list(result.index) == ["TX0000001", "CA0000001"]


class TestExtractAllProximityFeatures:
    def test_produces_columns_for_each_type(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        # Create FRS data with SIC codes for multiple types
        facilities = _make_facilities_gdf(
            [-97.73, -97.72, -97.71, -97.70, -97.69],
            [30.27, 30.28, 30.29, 30.30, 30.31],
            sic_codes=["4952", "4953", "4512", "9711", "4952"],
        )
        result = extract_all_proximity_features(systems, facilities)
        # Should have columns for each configured type
        assert any("wwtp" in c for c in result.columns)
        assert any("landfill" in c for c in result.columns)
        assert any("airport" in c for c in result.columns)
        assert any("military" in c for c in result.columns)

    def test_empty_frs_returns_empty_columns(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        facilities = _make_facilities_gdf([], [], sic_codes=[])
        result = extract_all_proximity_features(systems, facilities)
        assert len(result) == 1

    def test_result_indexed_by_pwsid(self, make_systems_gdf) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        facilities = _make_facilities_gdf([-97.73], [30.27], sic_codes=["4952"])
        result = extract_all_proximity_features(systems, facilities)
        assert result.index.name == "pwsid"

    def test_facility_type_filtering_produces_subsets(self, make_systems_gdf) -> None:
        """Verify that different facility types get different subsets."""
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        # Only WWTPs — airports should have inf distance
        facilities = _make_facilities_gdf(
            [-97.735],
            [30.27],
            sic_codes=["4952"],
        )
        result = extract_all_proximity_features(systems, facilities)
        # WWTP should be close
        if "dist_nearest_wwtp" in result.columns:
            assert result["dist_nearest_wwtp"].iloc[0] < 5000
        # Airport should be far (inf or very large) since no airports in data
        if "dist_nearest_airport" in result.columns:
            assert result["dist_nearest_airport"].iloc[0] == np.inf
