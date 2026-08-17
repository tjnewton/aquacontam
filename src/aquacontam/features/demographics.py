"""Demographic feature extraction from EJScreen block group data.

Spatially joins EJScreen block group centroids to water system locations,
producing demographic and environmental justice features per system.

Typical usage::

    from aquacontam.features.demographics import extract_demographic_features

    features = extract_demographic_features(systems_gdf, ejscreen_gdf)
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import pandas as pd

from aquacontam._constants import (
    CRS_DISTANCE,
    EJSCREEN_DEMOGRAPHIC_COLS,
    EJSCREEN_EJ_INDEX_COLS,
)

logger = logging.getLogger(__name__)

# Maximum distance (meters) for sjoin_nearest fallback
_MAX_JOIN_DISTANCE_M: float = 10_000.0

# Columns to transfer from EJScreen to output
_FEATURE_COLS: list[str] = list(EJSCREEN_DEMOGRAPHIC_COLS) + list(EJSCREEN_EJ_INDEX_COLS)


def extract_demographic_features(
    systems: gpd.GeoDataFrame,
    ejscreen: gpd.GeoDataFrame,
    *,
    max_distance_m: float = _MAX_JOIN_DISTANCE_M,
) -> pd.DataFrame:
    """Extract demographic features for water systems from EJScreen data.

    Uses spatial nearest-neighbor join to match each water system to
    the closest EJScreen block group centroid.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column and point geometry.
        Must have a CRS set.
    ejscreen : GeoDataFrame
        EJScreen block group centroids with demographic columns.
        Must have a CRS set.
    max_distance_m : float
        Maximum join distance in meters. Systems further than this from
        any block group centroid will have NaN demographic values.

    Returns
    -------
    pd.DataFrame
        Demographic features indexed by ``pwsid``.
    """
    # Ensure pwsid is a column (it may be the index)
    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()

    if systems.empty:
        logger.warning("Empty systems GeoDataFrame — returning empty features")
        return cast(pd.DataFrame, pd.DataFrame(columns=_FEATURE_COLS))

    if ejscreen.empty:
        logger.warning("Empty EJScreen data — returning NaN features for all systems")
        result = pd.DataFrame(index=systems["pwsid"], columns=_FEATURE_COLS, dtype=float)
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    # Project to Conus Albers for distance-based join
    systems_proj = systems.to_crs(CRS_DISTANCE)
    ejscreen_proj = ejscreen.to_crs(CRS_DISTANCE)

    # Nearest-neighbor spatial join
    joined = gpd.sjoin_nearest(
        systems_proj[["pwsid", "geometry"]],
        ejscreen_proj,
        how="left",
        max_distance=max_distance_m,
        distance_col="_join_dist",
    )

    # Deduplicate — keep closest match per pwsid
    joined = joined.sort_values("_join_dist").drop_duplicates(subset="pwsid", keep="first")

    # Extract feature columns
    available_cols = [c for c in _FEATURE_COLS if c in joined.columns]
    result = joined.set_index("pwsid")[available_cols].copy()
    result.index.name = "pwsid"

    n_matched = result.notna().any(axis=1).sum()
    n_total = len(systems)
    logger.info(
        "Matched %d / %d systems to EJScreen block groups (max_distance=%.0fm)",
        n_matched,
        n_total,
        max_distance_m,
    )

    return result  # type: ignore[no-any-return]
