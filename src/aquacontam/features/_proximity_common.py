"""Shared utilities for proximity-based feature extraction.

Provides common functions used by ``proximity.py``, ``tri_proximity.py``,
and ``dod_proximity.py`` to avoid code duplication.
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from aquacontam._constants import CRS_STORAGE
from aquacontam.geo.distance import count_within_radius, nearest_distances

logger = logging.getLogger(__name__)


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a DataFrame with lat/lon columns to a GeoDataFrame in EPSG:4326.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``latitude`` and ``longitude`` columns.

    Returns
    -------
    gpd.GeoDataFrame
        Point geometries in WGS 84, with rows missing coordinates dropped.
    """
    valid = df.dropna(subset=["latitude", "longitude"])
    return gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
        crs=CRS_STORAGE,
    )


def ensure_pwsid_column(systems: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure ``pwsid`` is a column (not just the index).

    Parameters
    ----------
    systems : GeoDataFrame
        Water systems GeoDataFrame.

    Returns
    -------
    GeoDataFrame
        With ``pwsid`` as a column.
    """
    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()
    return systems


def compute_proximity_features(
    systems: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    radii: tuple[float, ...],
    prefix: str,
) -> pd.DataFrame:
    """Compute nearest-distance and count-within-radius features.

    Parameters
    ----------
    systems : GeoDataFrame
        Water systems with geometry (EPSG:4326) and ``pwsid`` column.
    targets : GeoDataFrame
        Target facilities/sites with point geometry.
    radii : tuple[float, ...]
        Radii in meters for count features.
    prefix : str
        Column name prefix (e.g., ``"dod_pfas"``, ``"tri_pfas"``).

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with columns:
        - ``nearest_{prefix}_m``: distance to nearest target (meters)
        - ``count_{prefix}_{r/1000:.0f}km``: count within each radius
    """
    systems = ensure_pwsid_column(systems)
    pwsid_index = pd.Index(systems["pwsid"].to_numpy(), name="pwsid")

    if len(targets) == 0:
        result = pd.DataFrame(
            {f"nearest_{prefix}_m": np.full(len(systems), np.nan)},
            index=pwsid_index,
        )
        for r in radii:
            result[f"count_{prefix}_{r / 1000:.0f}km"] = 0
        return cast(pd.DataFrame, result)

    dists = nearest_distances(systems, targets)
    result = pd.DataFrame(
        {f"nearest_{prefix}_m": dists.to_numpy()},
        index=pwsid_index,
    )

    for r in radii:
        counts = count_within_radius(systems, targets, r)
        result[f"count_{prefix}_{r / 1000:.0f}km"] = counts.to_numpy()

    return cast(pd.DataFrame, result)
