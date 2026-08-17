"""Tests for CONUS grid generation."""

from __future__ import annotations

import geopandas as gpd

from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
)
from aquacontam.features.grid import generate_conus_grid


class TestGenerateConusGrid:
    """Tests for generate_conus_grid()."""

    def test_returns_geodataframe(self) -> None:
        grid = generate_conus_grid(resolution_km=200.0)
        assert isinstance(grid, gpd.GeoDataFrame)

    def test_has_grid_id(self) -> None:
        grid = generate_conus_grid(resolution_km=200.0)
        assert "grid_id" in grid.columns

    def test_crs_is_wgs84(self) -> None:
        grid = generate_conus_grid(resolution_km=200.0)
        assert grid.crs is not None
        assert grid.crs.to_epsg() == 4326

    def test_points_within_conus(self) -> None:
        grid = generate_conus_grid(resolution_km=200.0)
        bounds = grid.total_bounds  # [minx, miny, maxx, maxy]
        # Grid should approximately cover CONUS (allow some margin for projection)
        assert bounds[0] > CONUS_LON_MIN - 5  # minx (longitude)
        assert bounds[2] < CONUS_LON_MAX + 5  # maxx
        assert bounds[1] > CONUS_LAT_MIN - 5  # miny (latitude)
        assert bounds[3] < CONUS_LAT_MAX + 5  # maxy

    def test_higher_resolution_more_points(self) -> None:
        coarse = generate_conus_grid(resolution_km=500.0)
        fine = generate_conus_grid(resolution_km=200.0)
        assert len(fine) > len(coarse)

    def test_grid_ids_unique(self) -> None:
        grid = generate_conus_grid(resolution_km=200.0)
        assert grid["grid_id"].nunique() == len(grid)

    def test_default_resolution(self) -> None:
        # Just make sure it works without args (10km -> many points)
        # Use a large resolution to keep test fast
        grid = generate_conus_grid(resolution_km=500.0)
        assert len(grid) > 10

    def test_geometry_is_point(self) -> None:
        grid = generate_conus_grid(resolution_km=500.0)
        assert all(geom.geom_type == "Point" for geom in grid.geometry)
