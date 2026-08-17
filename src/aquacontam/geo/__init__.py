"""Geospatial utilities — CRS transforms, distance calculations, raster extraction."""

from aquacontam.geo.crs import ensure_crs, to_conus_albers, to_wgs84
from aquacontam.geo.distance import (
    build_spatial_index,
    count_within_radius,
    nearest_distances,
)
from aquacontam.geo.raster import categorical_fractions, zonal_stats_for_points

__all__ = [
    "build_spatial_index",
    "categorical_fractions",
    "count_within_radius",
    "ensure_crs",
    "nearest_distances",
    "to_conus_albers",
    "to_wgs84",
    "zonal_stats_for_points",
]
