"""TRI PFAS release-weighted proximity features.

Computes proximity features for water systems based on TRI PFAS facility
locations and release quantities. Unlike binary FRS proximity features,
these use release-weighted distances (inverse distance x release quantity).

Typical usage::

    from aquacontam.features.tri_proximity import extract_tri_features
    from aquacontam.data.tri import load_tri_pfas

    tri = load_tri_pfas(Path("data/raw"))
    features = extract_tri_features(systems_gdf, tri)
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from aquacontam._constants import CRS_DISTANCE
from aquacontam.features._proximity_common import (
    compute_proximity_features,
    to_geodataframe,
)

logger = logging.getLogger(__name__)


def _aggregate_tri_facilities(tri: pd.DataFrame) -> pd.DataFrame:
    """Aggregate TRI records to unique facility locations.

    Sums release quantities across years and chemicals per facility.

    Returns
    -------
    pd.DataFrame
        One row per unique facility location with columns:
        ``latitude``, ``longitude``, ``total_onsite_lb``,
        ``total_water_lb``, ``total_potw_lb``, ``n_chemicals``.
    """
    # Group by facility location (use lat/lon as key since FRS ID may be missing)
    grouped = tri.groupby(["latitude", "longitude"], as_index=False).agg(
        facility_name=("facility_name", "first"),
        total_onsite_lb=("onsite_release_lb", "sum"),
        total_water_lb=("water_release_lb", "sum"),
        total_potw_lb=("potw_transfer_lb", "sum"),
        n_chemicals=("chemical", "nunique"),
        n_years=("year", "nunique"),
    )
    return grouped


def extract_tri_features(
    systems: gpd.GeoDataFrame,
    tri: pd.DataFrame,
    radii_m: tuple[float, ...] = (5000.0, 10000.0, 25000.0),
) -> pd.DataFrame:
    """Compute TRI PFAS proximity features for water systems.

    Features produced:
    - ``nearest_tri_pfas_m``: Distance to nearest TRI PFAS facility (meters)
    - ``count_tri_pfas_{radius}m``: Number of TRI PFAS facilities within radius
    - ``release_weighted_tri_{radius}m``: Sum of (release_lb / distance_m) for
      all TRI PFAS facilities within radius (higher = more exposure risk)

    Parameters
    ----------
    systems : GeoDataFrame
        Water systems with geometry (EPSG:4326).
    tri : pd.DataFrame
        TRI PFAS facility data from ``load_tri_pfas()``.
    radii_m : tuple[float, ...]
        Radii in meters for count and release-weighted features.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame indexed by system index with TRI proximity columns.
    """
    if len(tri) == 0:
        logger.warning("No TRI PFAS facilities — returning NaN features")
        cols = ["nearest_tri_pfas_m"]
        for r in radii_m:
            r_km = int(r / 1000)
            cols.extend([f"count_tri_pfas_{r_km}km", f"release_weighted_tri_{r_km}km"])
        return cast(pd.DataFrame, pd.DataFrame(np.nan, index=systems.index, columns=cols))

    # Aggregate TRI to unique facility locations
    tri_agg = _aggregate_tri_facilities(tri)
    tri_gdf = to_geodataframe(tri_agg)

    if len(tri_gdf) == 0:
        logger.warning("No TRI facilities with valid coordinates")
        return cast(pd.DataFrame, pd.DataFrame(index=systems.index))

    # Use shared proximity helper for nearest distance + counts
    result = compute_proximity_features(systems, tri_gdf, radii_m, prefix="tri_pfas")

    # Project for release-weighted calculations (TRI-specific)
    from aquacontam.features._proximity_common import ensure_pwsid_column

    systems = ensure_pwsid_column(systems)
    systems_proj = systems.to_crs(CRS_DISTANCE)
    tri_proj = tri_gdf.to_crs(CRS_DISTANCE)

    # Release-weighted proximity
    sys_coords = np.column_stack([systems_proj.geometry.x, systems_proj.geometry.y])
    tri_coords = np.column_stack([tri_proj.geometry.x, tri_proj.geometry.y])
    tri_releases = tri_agg.loc[tri_gdf.index, "total_onsite_lb"].to_numpy().astype(float)

    from scipy.spatial import cKDTree

    tri_tree = cKDTree(tri_coords)

    for radius in radii_m:
        r_km = int(radius / 1000)
        weighted = np.zeros(len(systems_proj))
        neighbor_lists = tri_tree.query_ball_point(sys_coords, r=radius)
        for i, neighbors in enumerate(neighbor_lists):
            if neighbors:
                nbr_idx = np.asarray(neighbors)
                diffs = tri_coords[nbr_idx] - sys_coords[i]
                distances = np.sqrt((diffs**2).sum(axis=1))
                # Avoid division by zero — use max(distance, 100m)
                safe_dists = np.maximum(distances, 100.0)
                weighted[i] = (tri_releases[nbr_idx] / safe_dists).sum()
        result[f"release_weighted_tri_{r_km}km"] = weighted

    logger.info(
        "Computed TRI proximity features: %d systems x %d facilities",
        len(systems),
        len(tri_gdf),
    )
    return result
