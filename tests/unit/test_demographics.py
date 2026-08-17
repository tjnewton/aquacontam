"""Tests for demographic feature extraction from EJScreen data."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE, EJSCREEN_DEMOGRAPHIC_COLS, EJSCREEN_EJ_INDEX_COLS
from aquacontam.features.demographics import extract_demographic_features


@pytest.fixture()
def mock_systems_gdf() -> gpd.GeoDataFrame:
    """Small set of water systems for testing."""
    return gpd.GeoDataFrame(
        {
            "pwsid": ["NJ0100001", "NJ0100002", "NJ0100003", "NJ0100004"],
        },
        geometry=[
            Point(-74.0, 40.0),
            Point(-74.1, 40.1),
            Point(-74.2, 40.2),
            Point(-74.3, 40.3),
        ],
        crs=CRS_STORAGE,
    )


@pytest.fixture()
def mock_ejscreen_gdf() -> gpd.GeoDataFrame:
    """Synthetic EJScreen block group data."""
    n = 20
    rng = np.random.RandomState(42)
    data = {
        "block_group_id": [f"3400100{i:04d}" for i in range(n)],
        "pct_people_of_color": rng.uniform(0.1, 0.9, n),
        "pct_low_income": rng.uniform(0.1, 0.5, n),
        "pct_less_hs_education": rng.uniform(0.05, 0.3, n),
        "pct_limited_english": rng.uniform(0.01, 0.15, n),
        "pct_under_5": rng.uniform(0.04, 0.08, n),
        "pct_over_64": rng.uniform(0.10, 0.20, n),
        "ej_index_water": rng.uniform(0, 100, n),
        "ej_supplemental_index": rng.uniform(0, 1, n),
        "prox_wastewater_discharge": rng.uniform(0, 100, n),
        "prox_superfund": rng.uniform(0, 100, n),
        "prox_hazardous_waste": rng.uniform(0, 100, n),
        "prox_rmp_facility": rng.uniform(0, 100, n),
    }
    # Spread block group centroids around the system locations
    lons = -74.0 + rng.uniform(-0.2, 0.4, n)
    lats = 40.0 + rng.uniform(-0.2, 0.4, n)
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)]
    return gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_STORAGE)


class TestExtractDemographicFeatures:
    """Tests for extract_demographic_features()."""

    def test_output_shape(
        self, mock_systems_gdf: gpd.GeoDataFrame, mock_ejscreen_gdf: gpd.GeoDataFrame
    ) -> None:
        result = extract_demographic_features(mock_systems_gdf, mock_ejscreen_gdf)
        assert len(result) == len(mock_systems_gdf)

    def test_indexed_by_pwsid(
        self, mock_systems_gdf: gpd.GeoDataFrame, mock_ejscreen_gdf: gpd.GeoDataFrame
    ) -> None:
        result = extract_demographic_features(mock_systems_gdf, mock_ejscreen_gdf)
        assert result.index.name == "pwsid"
        assert set(result.index) == set(mock_systems_gdf["pwsid"])

    def test_has_demographic_columns(
        self, mock_systems_gdf: gpd.GeoDataFrame, mock_ejscreen_gdf: gpd.GeoDataFrame
    ) -> None:
        result = extract_demographic_features(mock_systems_gdf, mock_ejscreen_gdf)
        for col in EJSCREEN_DEMOGRAPHIC_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_has_ej_index_columns(
        self, mock_systems_gdf: gpd.GeoDataFrame, mock_ejscreen_gdf: gpd.GeoDataFrame
    ) -> None:
        result = extract_demographic_features(mock_systems_gdf, mock_ejscreen_gdf)
        found = [c for c in EJSCREEN_EJ_INDEX_COLS if c in result.columns]
        assert len(found) > 0

    def test_values_are_numeric(
        self, mock_systems_gdf: gpd.GeoDataFrame, mock_ejscreen_gdf: gpd.GeoDataFrame
    ) -> None:
        result = extract_demographic_features(mock_systems_gdf, mock_ejscreen_gdf)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} not numeric"

    def test_empty_systems(self, mock_ejscreen_gdf: gpd.GeoDataFrame) -> None:
        empty = gpd.GeoDataFrame(
            {"pwsid": pd.Series(dtype=str)},
            geometry=[],
            crs=CRS_STORAGE,
        )
        result = extract_demographic_features(empty, mock_ejscreen_gdf)
        assert len(result) == 0

    def test_empty_ejscreen(self, mock_systems_gdf: gpd.GeoDataFrame) -> None:
        empty_ej = gpd.GeoDataFrame(
            {"block_group_id": pd.Series(dtype=str)},
            geometry=[],
            crs=CRS_STORAGE,
        )
        result = extract_demographic_features(mock_systems_gdf, empty_ej)
        assert len(result) == len(mock_systems_gdf)
        # All values should be NaN (no match)
        assert result.isna().all().all()

    def test_max_distance_filtering(self, mock_systems_gdf: gpd.GeoDataFrame) -> None:
        """Systems far from any block group should get NaN values."""
        # Put EJScreen data very far from systems
        far_ej = gpd.GeoDataFrame(
            {
                "block_group_id": ["999999"],
                "pct_people_of_color": [0.5],
                "pct_low_income": [0.3],
            },
            geometry=[Point(-120.0, 35.0)],  # California, far from NJ
            crs=CRS_STORAGE,
        )
        result = extract_demographic_features(mock_systems_gdf, far_ej, max_distance_m=10_000.0)
        # All should be unmatched (NaN)
        assert result.isna().all().all()

    def test_deduplicates_per_pwsid(self, mock_ejscreen_gdf: gpd.GeoDataFrame) -> None:
        """Each pwsid should appear exactly once in output."""
        systems = gpd.GeoDataFrame(
            {"pwsid": ["NJ0100001", "NJ0100001"]},
            geometry=[Point(-74.0, 40.0), Point(-74.0, 40.0)],
            crs=CRS_STORAGE,
        )
        result = extract_demographic_features(systems, mock_ejscreen_gdf)
        assert not result.index.duplicated().any()

    def test_closest_block_group_matched(self) -> None:
        """Should match to the nearest block group centroid."""
        systems = gpd.GeoDataFrame(
            {"pwsid": ["SYS001"]},
            geometry=[Point(-74.0, 40.0)],
            crs=CRS_STORAGE,
        )
        # Two block groups: one nearby, one far
        ejscreen = gpd.GeoDataFrame(
            {
                "block_group_id": ["BG_NEAR", "BG_FAR"],
                "pct_people_of_color": [0.8, 0.1],
            },
            geometry=[Point(-74.001, 40.001), Point(-75.0, 41.0)],
            crs=CRS_STORAGE,
        )
        result = extract_demographic_features(systems, ejscreen)
        assert result.loc["SYS001", "pct_people_of_color"] == pytest.approx(0.8)

    def test_returns_dataframe_not_geodataframe(
        self, mock_systems_gdf: gpd.GeoDataFrame, mock_ejscreen_gdf: gpd.GeoDataFrame
    ) -> None:
        result = extract_demographic_features(mock_systems_gdf, mock_ejscreen_gdf)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, gpd.GeoDataFrame)
