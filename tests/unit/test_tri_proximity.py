"""Tests for TRI PFAS proximity features."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE
from aquacontam.features.tri_proximity import (
    _aggregate_tri_facilities,
    extract_tri_features,
)


@pytest.fixture()
def sample_systems() -> gpd.GeoDataFrame:
    """Create sample water system GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "pwsid": ["OH0000001", "OH0000002"],
            "latitude": [41.0, 39.0],
            "longitude": [-83.0, -84.0],
        },
        geometry=[Point(-83.0, 41.0), Point(-84.0, 39.0)],
        crs=CRS_STORAGE,
    )


@pytest.fixture()
def sample_tri() -> pd.DataFrame:
    """Create sample TRI PFAS facility data."""
    return pd.DataFrame(
        {
            "frs_id": ["1", "1", "2"],
            "facility_name": ["FACTORY A", "FACTORY A", "FACTORY B"],
            "latitude": [41.01, 41.01, 42.0],
            "longitude": [-83.01, -83.01, -85.0],
            "chemical": ["PFOA", "PFOS", "PFOA"],
            "year": [2023, 2023, 2023],
            "onsite_release_lb": [100.0, 50.0, 200.0],
            "water_release_lb": [10.0, 5.0, 20.0],
            "potw_transfer_lb": [0.0, 0.0, 0.0],
        }
    )


class TestAggregateFacilities:
    """Tests for _aggregate_tri_facilities."""

    def test_aggregates_by_location(self, sample_tri: pd.DataFrame) -> None:
        result = _aggregate_tri_facilities(sample_tri)
        # 2 unique locations
        assert len(result) == 2

    def test_sums_releases(self, sample_tri: pd.DataFrame) -> None:
        result = _aggregate_tri_facilities(sample_tri)
        # Factory A: 100 + 50 = 150 total onsite
        factory_a = result[result["facility_name"] == "FACTORY A"]
        assert len(factory_a) == 1
        assert factory_a.iloc[0]["total_onsite_lb"] == 150.0

    def test_counts_chemicals(self, sample_tri: pd.DataFrame) -> None:
        result = _aggregate_tri_facilities(sample_tri)
        factory_a = result[result["facility_name"] == "FACTORY A"]
        assert factory_a.iloc[0]["n_chemicals"] == 2


class TestExtractTriFeatures:
    """Tests for extract_tri_features."""

    def test_produces_expected_columns(
        self, sample_systems: gpd.GeoDataFrame, sample_tri: pd.DataFrame
    ) -> None:
        features = extract_tri_features(sample_systems, sample_tri)
        assert "nearest_tri_pfas_m" in features.columns
        assert "count_tri_pfas_5km" in features.columns
        assert "release_weighted_tri_5km" in features.columns

    def test_nearest_distance(
        self, sample_systems: gpd.GeoDataFrame, sample_tri: pd.DataFrame
    ) -> None:
        features = extract_tri_features(sample_systems, sample_tri)
        # System 1 (41, -83) is very close to Factory A (41.01, -83.01)
        assert features.loc["OH0000001", "nearest_tri_pfas_m"] < 5000

    def test_empty_tri(self, sample_systems: gpd.GeoDataFrame) -> None:
        """Empty TRI data should produce NaN features."""
        empty = pd.DataFrame(
            columns=[
                "frs_id",
                "facility_name",
                "latitude",
                "longitude",
                "chemical",
                "year",
                "onsite_release_lb",
                "water_release_lb",
                "potw_transfer_lb",
            ]
        )
        features = extract_tri_features(sample_systems, empty)
        assert len(features) == 2
        assert features["nearest_tri_pfas_m"].isna().all()

    def test_release_weighted_positive(
        self, sample_systems: gpd.GeoDataFrame, sample_tri: pd.DataFrame
    ) -> None:
        """Systems near TRI facilities should have positive release-weighted values."""
        features = extract_tri_features(sample_systems, sample_tri, radii_m=(100000.0,))
        col = "release_weighted_tri_100km"
        assert col in features.columns
        # At least one system should have non-zero weighted proximity
        assert (features[col] > 0).any()

    def test_index_is_pwsid(
        self, sample_systems: gpd.GeoDataFrame, sample_tri: pd.DataFrame
    ) -> None:
        features = extract_tri_features(sample_systems, sample_tri)
        assert features.index.name == "pwsid"
        assert list(features.index) == list(sample_systems["pwsid"])
