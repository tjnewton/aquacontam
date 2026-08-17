"""Tests for geo.raster — raster extraction helpers.

Uses a synthetic 10x10 GeoTIFF for deterministic testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from aquacontam.geo.raster import (
    _buffer_and_match_raster_crs,
    categorical_fractions,
    zonal_stats_for_points,
)


@pytest.fixture()
def synthetic_raster(tmp_path: Path) -> Path:
    """Create a tiny 10x10 synthetic GeoTIFF.

    Layout (in EPSG:4326, small area near Austin TX):
    - Left half (cols 0-4): pixel value 24 (developed)
    - Right half (cols 5-9): pixel value 41 (forest)
    """
    raster_path = tmp_path / "test_raster.tif"
    west, south, east, north = -97.80, 30.20, -97.70, 30.30
    width, height = 10, 10
    transform = from_bounds(west, south, east, north, width, height)

    data = np.zeros((height, width), dtype=np.uint8)
    data[:, :5] = 24  # Left half: developed high intensity
    data[:, 5:] = 41  # Right half: deciduous forest

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return raster_path


class TestBufferAndMatchRasterCrs:
    def test_returns_polygon_geometries(self, make_systems_gdf, synthetic_raster: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        buffered = _buffer_and_match_raster_crs(systems, synthetic_raster, 1000.0)
        assert buffered.geometry.iloc[0].geom_type == "Polygon"

    def test_output_crs_matches_raster(self, make_systems_gdf, synthetic_raster: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        buffered = _buffer_and_match_raster_crs(systems, synthetic_raster, 1000.0)
        with rasterio.open(str(synthetic_raster)) as src:
            raster_crs = src.crs
        assert buffered.crs == raster_crs


class TestZonalStatsForPoints:
    def test_returns_dataframe_with_stats(self, make_systems_gdf, synthetic_raster: Path) -> None:
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = zonal_stats_for_points(systems, synthetic_raster, 5000.0, ["mean", "min", "max"])
        assert "mean" in result.columns
        assert "min" in result.columns
        assert "max" in result.columns
        assert len(result) == 1

    def test_indexed_by_pwsid(self, make_systems_gdf, synthetic_raster: Path) -> None:
        systems = make_systems_gdf(["TX0000001", "TX0000002"], [-97.75, -97.72], [30.25, 30.25])
        result = zonal_stats_for_points(systems, synthetic_raster, 5000.0, ["mean"])
        assert list(result.index) == ["TX0000001", "TX0000002"]

    def test_stat_values_within_expected_range(
        self, make_systems_gdf, synthetic_raster: Path
    ) -> None:
        # Center point covers both halves — mean should be between 24 and 41
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = zonal_stats_for_points(systems, synthetic_raster, 5000.0, ["mean"])
        mean_val = result["mean"].iloc[0]
        assert 24 <= mean_val <= 41


class TestCategoricalFractions:
    def test_fractions_sum_to_approximately_one(
        self, make_systems_gdf, synthetic_raster: Path
    ) -> None:
        class_map = {24: "developed", 41: "forest"}
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = categorical_fractions(systems, synthetic_raster, 5000.0, class_map)
        total = result.iloc[0].sum()
        # May not be exactly 1.0 due to buffer extending beyond small raster
        assert 0.5 <= total <= 1.01

    def test_has_expected_columns(self, make_systems_gdf, synthetic_raster: Path) -> None:
        class_map = {24: "developed", 41: "forest"}
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = categorical_fractions(systems, synthetic_raster, 5000.0, class_map)
        assert "developed" in result.columns
        assert "forest" in result.columns

    def test_left_side_mostly_developed(self, make_systems_gdf, synthetic_raster: Path) -> None:
        class_map = {24: "developed", 41: "forest"}
        # Point on the left side of the raster
        systems = make_systems_gdf(["TX0000001"], [-97.78], [30.25])
        result = categorical_fractions(systems, synthetic_raster, 1000.0, class_map)
        assert result["developed"].iloc[0] > result["forest"].iloc[0]

    def test_grouped_classes(self, make_systems_gdf, synthetic_raster: Path) -> None:
        # Map both classes to same group
        class_map = {24: "any_land", 41: "any_land"}
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = categorical_fractions(systems, synthetic_raster, 5000.0, class_map)
        # Should be all one category
        assert "any_land" in result.columns
        assert 0.5 <= result["any_land"].iloc[0] <= 1.01

    def test_indexed_by_pwsid(self, make_systems_gdf, synthetic_raster: Path) -> None:
        class_map = {24: "developed", 41: "forest"}
        systems = make_systems_gdf(["TX0000001"], [-97.75], [30.25])
        result = categorical_fractions(systems, synthetic_raster, 5000.0, class_map)
        assert result.index[0] == "TX0000001"
