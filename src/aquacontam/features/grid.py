"""CONUS grid generation for national risk surface mapping.

Generates a regular grid of points across the contiguous US in EPSG:5070
(Conus Albers Equal Area), then reprojects to EPSG:4326 for feature extraction.

Typical usage::

    from aquacontam.features.grid import generate_conus_grid

    grid = generate_conus_grid(resolution_km=10.0)
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

from aquacontam._constants import (
    CONUS_GRID_RESOLUTION_KM,
    CRS_DISTANCE,
    CRS_STORAGE,
)

logger = logging.getLogger(__name__)

# Approximate CONUS bounding box in EPSG:5070 (meters)
_CONUS_5070_X_MIN: float = -2_356_000.0
_CONUS_5070_X_MAX: float = 2_258_000.0
_CONUS_5070_Y_MIN: float = 272_000.0
_CONUS_5070_Y_MAX: float = 3_172_000.0


def generate_conus_grid(
    resolution_km: float = CONUS_GRID_RESOLUTION_KM,
) -> gpd.GeoDataFrame:
    """Generate a regular point grid covering the contiguous US.

    Parameters
    ----------
    resolution_km : float
        Grid spacing in kilometers. Default is 10 km.

    Returns
    -------
    GeoDataFrame
        Grid points with ``grid_id`` column and point geometry in EPSG:4326.
    """
    spacing_m = resolution_km * 1000.0

    x_coords = np.arange(_CONUS_5070_X_MIN, _CONUS_5070_X_MAX, spacing_m)
    y_coords = np.arange(_CONUS_5070_Y_MIN, _CONUS_5070_Y_MAX, spacing_m)

    logger.info(
        "Generating CONUS grid: %.0f km spacing, %d x %d = %d points",
        resolution_km,
        len(x_coords),
        len(y_coords),
        len(x_coords) * len(y_coords),
    )

    xx, yy = np.meshgrid(x_coords, y_coords)
    points = gpd.points_from_xy(xx.ravel(), yy.ravel())
    grid_ids = [f"G{i:07d}" for i in range(len(points))]

    gdf = gpd.GeoDataFrame(
        {"grid_id": grid_ids},
        geometry=points,
        crs=CRS_DISTANCE,
    )

    # Reproject to WGS84 for feature extraction
    gdf = gdf.to_crs(CRS_STORAGE)

    logger.info("Generated %d grid points", len(gdf))
    return gdf
