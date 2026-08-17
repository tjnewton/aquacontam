"""CRS transformation helpers.

Provides simple wrappers for reprojecting GeoDataFrames between the
project's two standard CRS values: EPSG:4326 (storage) and EPSG:5070
(Conus Albers, for distance calculations in meters).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pyproj

from aquacontam._constants import CRS_DISTANCE, CRS_STORAGE

logger = logging.getLogger(__name__)


def to_conus_albers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to EPSG:5070 (Conus Albers Equal Area).

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data in any CRS.

    Returns
    -------
    GeoDataFrame
        Reprojected to EPSG:5070.
    """
    return ensure_crs(gdf, CRS_DISTANCE)


def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to EPSG:4326 (WGS 84).

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data in any CRS.

    Returns
    -------
    GeoDataFrame
        Reprojected to EPSG:4326.
    """
    return ensure_crs(gdf, CRS_STORAGE)


def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject *gdf* to *target_crs* if needed (no-op when already correct).

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data. Must have a CRS set (``gdf.crs`` must not be ``None``).
    target_crs : str
        Target CRS string (e.g. ``"EPSG:4326"``).

    Returns
    -------
    GeoDataFrame
        Reprojected copy, or the original if CRS already matches.

    Raises
    ------
    ValueError
        If the input GeoDataFrame has no CRS set.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS set; cannot reproject.")
    if len(gdf) == 0:
        logger.debug("Empty GeoDataFrame, setting CRS to %s", target_crs)
        return gdf.to_crs(target_crs)
    if gdf.crs.to_authority() == pyproj.CRS(target_crs).to_authority():
        logger.debug("GeoDataFrame already in %s, skipping reprojection", target_crs)
        return gdf
    logger.debug("Reprojecting %d geometries from %s to %s", len(gdf), gdf.crs, target_crs)
    return gdf.to_crs(target_crs)
