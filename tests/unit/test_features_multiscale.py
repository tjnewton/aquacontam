"""Unit tests for multi-scale spatial features."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE


@pytest.fixture()
def systems_gdf() -> gpd.GeoDataFrame:
    """Small GeoDataFrame of water systems."""
    data = {
        "pwsid": ["SYS001", "SYS002", "SYS003"],
        "geometry": [
            Point(-90.0, 38.0),
            Point(-89.5, 38.5),
            Point(-91.0, 37.5),
        ],
    }
    return gpd.GeoDataFrame(data, crs=CRS_STORAGE)


@pytest.fixture()
def facilities_gdf() -> gpd.GeoDataFrame:
    """Small GeoDataFrame of facilities."""
    data = {
        "geometry": [
            Point(-90.01, 38.01),
            Point(-89.99, 37.99),
            Point(-89.5, 38.5),
            Point(-90.5, 37.8),
            Point(-91.1, 37.6),
        ],
        "primary_sic": ["2819", "2911", "4952", "4953", "2869"],
        "interest_types": [
            "CERCLIS",
            "RCRAINFO",
            "NPDES",
            "RCRAINFO",
            "TRIS",
        ],
    }
    return gpd.GeoDataFrame(data, crs=CRS_STORAGE)


class TestKernelDensityFeatures:
    def test_shape_and_index(self, systems_gdf, facilities_gdf):
        from aquacontam.features.multiscale import extract_kernel_density_features

        result = extract_kernel_density_features(
            systems_gdf,
            facilities_gdf,
            bandwidths=(5000.0, 10000.0),
            facility_types=("industrial",),
        )
        assert result.index.name == "pwsid"
        assert len(result) == 3
        # 2 bandwidths x 1 facility type = 2 columns
        assert len(result.columns) == 2
        assert all(c.startswith("kd_industrial_") for c in result.columns)

    def test_density_nonnegative(self, systems_gdf, facilities_gdf):
        from aquacontam.features.multiscale import extract_kernel_density_features

        result = extract_kernel_density_features(
            systems_gdf,
            facilities_gdf,
            bandwidths=(5000.0,),
            facility_types=("industrial",),
        )
        assert (result >= 0).all().all()

    def test_empty_facilities(self, systems_gdf):
        from aquacontam.features.multiscale import extract_kernel_density_features

        empty = gpd.GeoDataFrame(
            {"geometry": [], "primary_sic": [], "interest_types": []},
            crs=CRS_STORAGE,
        )
        result = extract_kernel_density_features(
            systems_gdf,
            empty,
            bandwidths=(5000.0,),
            facility_types=("industrial",),
        )
        assert len(result) == 3


class TestKNNFeatures:
    def test_shape_and_columns(self, systems_gdf, facilities_gdf):
        from aquacontam.features.multiscale import extract_knn_features

        result = extract_knn_features(
            systems_gdf,
            facilities_gdf,
            k_values=(1, 3),
            facility_types=("industrial",),
        )
        assert result.index.name == "pwsid"
        assert len(result) == 3
        # 2 k-values x 3 stats (mean, std, max) x 1 type = 6
        assert len(result.columns) == 6

    def test_knn_distances_positive(self, systems_gdf, facilities_gdf):
        from aquacontam.features.multiscale import extract_knn_features

        result = extract_knn_features(
            systems_gdf,
            facilities_gdf,
            k_values=(1,),
            facility_types=("industrial",),
        )
        # Mean distance should be positive
        mean_col = next(c for c in result.columns if "mean_dist" in c)
        assert (result[mean_col] >= 0).all()

    def test_empty_facilities_knn(self, systems_gdf):
        from aquacontam.features.multiscale import extract_knn_features

        empty = gpd.GeoDataFrame(
            {"geometry": [], "primary_sic": [], "interest_types": []},
            crs=CRS_STORAGE,
        )
        result = extract_knn_features(
            systems_gdf, empty, k_values=(1,), facility_types=("industrial",)
        )
        assert len(result) == 3


class TestExtractAllMultiscale:
    def test_combines_kd_and_knn(self, systems_gdf, facilities_gdf):
        from aquacontam.features.multiscale import extract_all_multiscale_features

        result = extract_all_multiscale_features(systems_gdf, facilities_gdf)
        assert result.index.name == "pwsid"
        assert len(result) == 3
        # Should have both KD and KNN columns
        kd_cols = [c for c in result.columns if c.startswith("kd_")]
        knn_cols = [c for c in result.columns if c.startswith("knn")]
        assert len(kd_cols) > 0
        assert len(knn_cols) > 0
