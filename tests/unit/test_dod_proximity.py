"""Tests for DoD PFAS site proximity features."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE
from aquacontam.features.dod_proximity import extract_dod_features


@pytest.fixture()
def sample_systems() -> gpd.GeoDataFrame:
    """Create sample water system GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "pwsid": ["OH0000001", "NC0000002"],
            "latitude": [39.82, 35.14],
            "longitude": [-84.05, -79.00],
        },
        geometry=[Point(-84.05, 39.82), Point(-79.00, 35.14)],
        crs=CRS_STORAGE,
    )


@pytest.fixture()
def sample_dod_sites() -> pd.DataFrame:
    """Create sample DoD PFAS site data."""
    return pd.DataFrame(
        {
            "site_name": ["WRIGHT-PATTERSON AFB", "FORT BRAGG", "CAMP LEJEUNE"],
            "agency": ["Air Force", "Army", "Navy"],
            "state": ["OH", "NC", "NC"],
            "latitude": [39.83, 35.14, 34.62],
            "longitude": [-84.05, -79.00, -77.36],
            "pfas_presence": ["Known Detection", "Known Detection", "Known Detection"],
            "is_dod": [True, True, True],
        }
    )


class TestExtractDodFeatures:
    """Tests for extract_dod_features."""

    def test_produces_expected_columns(
        self, sample_systems: gpd.GeoDataFrame, sample_dod_sites: pd.DataFrame
    ) -> None:
        features = extract_dod_features(sample_systems, sample_dod_sites)
        assert "nearest_dod_pfas_m" in features.columns
        assert "log1p_nearest_dod_pfas" in features.columns
        assert "count_dod_pfas_5km" in features.columns
        assert "count_dod_pfas_10km" in features.columns
        assert "count_dod_pfas_25km" in features.columns

    def test_nearest_distance(
        self, sample_systems: gpd.GeoDataFrame, sample_dod_sites: pd.DataFrame
    ) -> None:
        features = extract_dod_features(sample_systems, sample_dod_sites)
        # System 1 (39.82, -84.05) is very close to Wright-Patterson (39.83, -84.05)
        assert features.loc["OH0000001", "nearest_dod_pfas_m"] < 5000

    def test_log1p_consistent(
        self, sample_systems: gpd.GeoDataFrame, sample_dod_sites: pd.DataFrame
    ) -> None:
        features = extract_dod_features(sample_systems, sample_dod_sites)
        np.testing.assert_allclose(
            features["log1p_nearest_dod_pfas"].values,
            np.log1p(features["nearest_dod_pfas_m"].values),
        )

    def test_empty_sites(self, sample_systems: gpd.GeoDataFrame) -> None:
        """Empty DoD data should produce NaN features."""
        empty = pd.DataFrame(
            columns=[
                "site_name",
                "agency",
                "state",
                "latitude",
                "longitude",
                "pfas_presence",
                "is_dod",
            ]
        )
        features = extract_dod_features(sample_systems, empty)
        assert len(features) == 2
        assert features["nearest_dod_pfas_m"].isna().all()

    def test_count_within_radius(
        self, sample_systems: gpd.GeoDataFrame, sample_dod_sites: pd.DataFrame
    ) -> None:
        """Systems near DoD sites should have non-zero counts at large radius."""
        features = extract_dod_features(sample_systems, sample_dod_sites, radii_m=(100000.0,))
        col = "count_dod_pfas_100km"
        assert col in features.columns
        assert (features[col] > 0).any()

    def test_index_is_pwsid(
        self, sample_systems: gpd.GeoDataFrame, sample_dod_sites: pd.DataFrame
    ) -> None:
        features = extract_dod_features(sample_systems, sample_dod_sites)
        assert features.index.name == "pwsid"
        assert list(features.index) == list(sample_systems["pwsid"])
