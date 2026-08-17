"""Tests for features.hydrogeology — USGS aquifer features.

Uses synthetic aquifer polygons for deterministic testing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from aquacontam._constants import CRS_STORAGE
from aquacontam.features.hydrogeology import extract_aquifer_features


def _make_aquifer_gdf() -> gpd.GeoDataFrame:
    """Create synthetic aquifer polygons.

    Two aquifers:
    - Aquifer A: covers Austin TX area
    - Aquifer B: covers Dallas TX area
    """
    return gpd.GeoDataFrame(
        {
            "AQ_NAME": ["Edwards Aquifer", "Trinity Aquifer"],
            "ROCK_TYPE": ["Carbonate", "Sandstone"],
            "AQ_TYPE": ["Confined", "Unconfined"],
        },
        geometry=[
            box(-98.5, 29.5, -97.0, 31.0),  # Austin area
            box(-97.5, 32.0, -96.0, 33.5),  # Dallas area
        ],
        crs=CRS_STORAGE,
    )


@pytest.fixture()
def aquifer_shapefile(tmp_path: Path) -> Path:
    """Write synthetic aquifers to a shapefile."""
    gdf = _make_aquifer_gdf()
    shp_path = tmp_path / "aquifers.shp"
    gdf.to_file(shp_path)
    return shp_path


class TestExtractAquiferFeatures:
    def test_point_inside_aquifer_gets_attributes(
        self, make_systems_gdf, aquifer_shapefile: Path
    ) -> None:
        # Point inside Edwards Aquifer (Austin)
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        with patch("aquacontam.features.hydrogeology.load_data_config") as mock_cfg:
            mock_cfg.return_value = {
                "usgs_aquifers": {
                    "column_map": {
                        "AQ_NAME": "aquifer_type",
                        "ROCK_TYPE": "aquifer_lithology",
                        "AQ_TYPE": "aquifer_confinement",
                    }
                }
            }
            result = extract_aquifer_features(systems, aquifer_shapefile)
        assert result["aquifer_type"].iloc[0] == "Edwards Aquifer"
        assert result["aquifer_lithology"].iloc[0] == "Carbonate"
        assert result["aquifer_confinement"].iloc[0] == "Confined"

    def test_point_outside_aquifers_gets_nan(
        self, make_systems_gdf, aquifer_shapefile: Path
    ) -> None:
        # Point far from any aquifer (e.g., middle of ocean)
        systems = make_systems_gdf(["XX0000001"], [-80.0], [25.0])
        with patch("aquacontam.features.hydrogeology.load_data_config") as mock_cfg:
            mock_cfg.return_value = {
                "usgs_aquifers": {
                    "column_map": {
                        "AQ_NAME": "aquifer_type",
                        "ROCK_TYPE": "aquifer_lithology",
                        "AQ_TYPE": "aquifer_confinement",
                    }
                }
            }
            result = extract_aquifer_features(systems, aquifer_shapefile)
        assert pd.isna(result["aquifer_type"].iloc[0])

    def test_multiple_systems_mixed_results(
        self, make_systems_gdf, aquifer_shapefile: Path
    ) -> None:
        # One inside, one outside
        systems = make_systems_gdf(
            ["TX0000001", "XX0000001"],
            [-97.74, -80.0],
            [30.27, 25.0],
        )
        with patch("aquacontam.features.hydrogeology.load_data_config") as mock_cfg:
            mock_cfg.return_value = {
                "usgs_aquifers": {
                    "column_map": {
                        "AQ_NAME": "aquifer_type",
                        "ROCK_TYPE": "aquifer_lithology",
                        "AQ_TYPE": "aquifer_confinement",
                    }
                }
            }
            result = extract_aquifer_features(systems, aquifer_shapefile)
        assert result["aquifer_type"].iloc[0] == "Edwards Aquifer"
        assert pd.isna(result["aquifer_type"].iloc[1])

    def test_correct_aquifer_assignment(self, make_systems_gdf, aquifer_shapefile: Path) -> None:
        # Point inside Trinity Aquifer (Dallas area)
        systems = make_systems_gdf(["TX0000002"], [-96.80], [32.78])
        with patch("aquacontam.features.hydrogeology.load_data_config") as mock_cfg:
            mock_cfg.return_value = {
                "usgs_aquifers": {
                    "column_map": {
                        "AQ_NAME": "aquifer_type",
                        "ROCK_TYPE": "aquifer_lithology",
                        "AQ_TYPE": "aquifer_confinement",
                    }
                }
            }
            result = extract_aquifer_features(systems, aquifer_shapefile)
        assert result["aquifer_type"].iloc[0] == "Trinity Aquifer"
        assert result["aquifer_lithology"].iloc[0] == "Sandstone"

    def test_output_indexed_by_pwsid(self, make_systems_gdf, aquifer_shapefile: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        with patch("aquacontam.features.hydrogeology.load_data_config") as mock_cfg:
            mock_cfg.return_value = {
                "usgs_aquifers": {
                    "column_map": {
                        "AQ_NAME": "aquifer_type",
                        "ROCK_TYPE": "aquifer_lithology",
                        "AQ_TYPE": "aquifer_confinement",
                    }
                }
            }
            result = extract_aquifer_features(systems, aquifer_shapefile)
        assert result.index.name == "pwsid"
        assert result.index[0] == "TX0000001"

    def test_has_expected_columns(self, make_systems_gdf, aquifer_shapefile: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.74], [30.27])
        with patch("aquacontam.features.hydrogeology.load_data_config") as mock_cfg:
            mock_cfg.return_value = {
                "usgs_aquifers": {
                    "column_map": {
                        "AQ_NAME": "aquifer_type",
                        "ROCK_TYPE": "aquifer_lithology",
                        "AQ_TYPE": "aquifer_confinement",
                    }
                }
            }
            result = extract_aquifer_features(systems, aquifer_shapefile)
        assert set(result.columns) == {"aquifer_type", "aquifer_lithology", "aquifer_confinement"}
