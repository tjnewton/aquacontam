"""Tests for geo.crs — CRS transformation helpers."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from aquacontam._constants import CRS_DISTANCE, CRS_STORAGE
from aquacontam.geo.crs import ensure_crs, to_conus_albers, to_wgs84


def _make_gdf(
    lons: list[float],
    lats: list[float],
    crs: str = CRS_STORAGE,
) -> gpd.GeoDataFrame:
    """Helper: create a point GeoDataFrame."""
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)]
    return gpd.GeoDataFrame({"id": range(len(lons))}, geometry=geometry, crs=crs)


class TestToConusAlbers:
    def test_reprojects_to_5070(self) -> None:
        gdf = _make_gdf([-118.24], [34.05])
        result = to_conus_albers(gdf)
        assert result.crs is not None
        assert result.crs.to_epsg() == 5070

    def test_coordinates_change_after_reprojection(self) -> None:
        gdf = _make_gdf([-118.24], [34.05])
        result = to_conus_albers(gdf)
        # Albers coords are in meters, very different from lon/lat
        assert abs(result.geometry.iloc[0].x) > 1000

    def test_no_op_when_already_5070(self) -> None:
        gdf = _make_gdf([-118.24], [34.05])
        projected = to_conus_albers(gdf)
        result = to_conus_albers(projected)
        assert result.crs.to_epsg() == 5070
        # Coordinates should be identical
        np.testing.assert_allclose(
            result.geometry.iloc[0].x,
            projected.geometry.iloc[0].x,
            atol=0.01,
        )


class TestToWgs84:
    def test_reprojects_to_4326(self) -> None:
        gdf = _make_gdf([-118.24], [34.05])
        projected = to_conus_albers(gdf)
        result = to_wgs84(projected)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_round_trip_preserves_coordinates(self) -> None:
        lon, lat = -97.74, 30.27
        gdf = _make_gdf([lon], [lat])
        result = to_wgs84(to_conus_albers(gdf))
        np.testing.assert_allclose(result.geometry.iloc[0].x, lon, atol=1e-6)
        np.testing.assert_allclose(result.geometry.iloc[0].y, lat, atol=1e-6)


class TestEnsureCrs:
    def test_no_op_same_crs(self) -> None:
        gdf = _make_gdf([-83.05], [42.33])
        result = ensure_crs(gdf, CRS_STORAGE)
        assert result.crs.to_epsg() == 4326
        np.testing.assert_allclose(result.geometry.iloc[0].x, -83.05, atol=1e-6)

    def test_reprojects_when_different(self) -> None:
        gdf = _make_gdf([-83.05], [42.33])
        result = ensure_crs(gdf, CRS_DISTANCE)
        assert result.crs.to_epsg() == 5070

    def test_empty_geodataframe(self) -> None:
        gdf = gpd.GeoDataFrame(geometry=[], crs=CRS_STORAGE)
        result = ensure_crs(gdf, CRS_DISTANCE)
        assert result.crs.to_epsg() == 5070
        assert len(result) == 0

    def test_raises_on_no_crs(self) -> None:
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
        with pytest.raises(ValueError, match="no CRS set"):
            ensure_crs(gdf, CRS_STORAGE)

    def test_multiple_points(self) -> None:
        lons = [-118.24, -97.74, -83.05]
        lats = [34.05, 30.27, 42.33]
        gdf = _make_gdf(lons, lats)
        result = to_conus_albers(gdf)
        assert len(result) == 3
        assert result.crs.to_epsg() == 5070
