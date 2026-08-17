"""Raster extraction helpers for point-buffered zonal statistics.

Wraps ``rasterstats.zonal_stats`` with CRS handling and error recovery.
All buffers are specified in meters and applied in EPSG:5070.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from aquacontam.geo.crs import ensure_crs, to_conus_albers

logger = logging.getLogger(__name__)


def _get_pwsids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract pwsid values from a GeoDataFrame (column or index)."""
    if "pwsid" in gdf.columns:
        return np.asarray(gdf["pwsid"].to_numpy())
    if gdf.index.name == "pwsid":
        return np.asarray(gdf.index.to_numpy())
    raise KeyError("GeoDataFrame must have 'pwsid' as a column or index")


def _buffer_and_match_raster_crs(
    points_gdf: gpd.GeoDataFrame,
    raster_path: Path | str,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Buffer points in EPSG:5070 then reproject to match raster CRS.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Point geometries.
    raster_path : Path or str
        Raster file to read CRS from.
    buffer_m : float
        Buffer radius in meters.

    Returns
    -------
    GeoDataFrame
        Buffered polygons in the raster's CRS.
    """
    projected = to_conus_albers(points_gdf)
    buffered = projected.copy()
    buffered["geometry"] = projected.geometry.buffer(buffer_m)

    # Read raster CRS and reproject buffered geometries to match
    with rasterio.open(str(raster_path)) as src:
        raster_crs = str(src.crs)

    return ensure_crs(buffered, raster_crs)


def zonal_stats_for_points(
    points_gdf: gpd.GeoDataFrame,
    raster_path: Path | str,
    buffer_m: float,
    stats: list[str],
) -> pd.DataFrame:
    """Extract raster zonal statistics within a buffer around each point.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Point geometries with ``pwsid`` column.
    raster_path : Path or str
        Path to a GeoTIFF raster file.
    buffer_m : float
        Buffer radius in meters (applied in EPSG:5070).
    stats : list[str]
        Statistics to compute (e.g. ``["mean", "median", "std"]``).

    Returns
    -------
    pd.DataFrame
        One row per point, indexed by ``pwsid``, with a column per stat.
    """
    buffered = _buffer_and_match_raster_crs(points_gdf, raster_path, buffer_m)
    pwsids = _get_pwsids(points_gdf)

    results = zonal_stats(
        buffered,
        str(raster_path),
        stats=stats,
        nodata=np.nan,
    )

    df = pd.DataFrame(results, index=pwsids)
    return cast(pd.DataFrame, df)


def categorical_fractions(
    points_gdf: gpd.GeoDataFrame,
    raster_path: Path | str,
    buffer_m: float,
    class_map: dict[int, str],
) -> pd.DataFrame:
    """Compute land cover category fractions within a buffer around each point.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Point geometries with ``pwsid`` column.
    raster_path : Path or str
        Path to a categorical raster (e.g. NLCD GeoTIFF).
    buffer_m : float
        Buffer radius in meters.
    class_map : dict[int, str]
        Maps raster integer class codes to category names. Codes mapping
        to the same name are aggregated (e.g. ``{21: "developed", 22: "developed"}``).

    Returns
    -------
    pd.DataFrame
        One row per point, indexed by ``pwsid``. Column per category name,
        values are fractions summing to <=1.0.
    """
    buffered = _buffer_and_match_raster_crs(points_gdf, raster_path, buffer_m)
    pwsids = _get_pwsids(points_gdf)

    results = zonal_stats(
        buffered,
        str(raster_path),
        categorical=True,
        all_touched=True,
    )

    rows = []
    for result_dict in results:
        total = sum(result_dict.values()) if result_dict else 0
        category_counts: dict[str, float] = {}
        for code, count in result_dict.items():
            cat_name = class_map.get(int(code))
            if cat_name is not None:
                category_counts[cat_name] = category_counts.get(cat_name, 0.0) + count
        # Convert counts to fractions
        if total > 0:
            category_fracs = {k: v / total for k, v in category_counts.items()}
        else:
            category_fracs = {k: 0.0 for k in set(class_map.values())}
        rows.append(category_fracs)

    # Ensure all category columns exist
    all_categories = sorted(set(class_map.values()))
    df = pd.DataFrame(rows, index=pwsids)
    for cat in all_categories:
        if cat not in df.columns:
            df[cat] = 0.0
    result: pd.DataFrame = df[all_categories].fillna(0.0)

    return result
