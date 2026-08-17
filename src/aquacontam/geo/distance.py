"""Spatial distance calculations using KD-trees.

All distance calculations are performed in EPSG:5070 (Conus Albers Equal
Area) so that results are in meters. Input GeoDataFrames are automatically
reprojected if needed.
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aquacontam.geo.crs import to_conus_albers

logger = logging.getLogger(__name__)


def _extract_coords(gdf: gpd.GeoDataFrame) -> NDArray[np.float64]:
    """Extract (x, y) coordinate array from geometry column."""
    return np.column_stack([gdf.geometry.x, gdf.geometry.y])


# Module-level cache for KD-trees. Keyed by (id(gdf), len(gdf)) to avoid
# rebuilding the same tree ~50 times per pipeline run (once per bandwidth,
# radius, or K value). Safe within a single pipeline run because the same
# GeoDataFrame object is reused for all queries against a given facility set.
_tree_cache: dict[tuple[int, int], cKDTree] = {}


def build_spatial_index(points_gdf: gpd.GeoDataFrame) -> cKDTree:
    """Build (or retrieve cached) KD-tree spatial index.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Point geometries. Automatically reprojected to EPSG:5070.

    Returns
    -------
    cKDTree
        Spatial index for efficient nearest-neighbor queries.
    """
    cache_key = (id(points_gdf), len(points_gdf))
    if cache_key in _tree_cache:
        return _tree_cache[cache_key]
    projected = to_conus_albers(points_gdf)
    coords = _extract_coords(projected)
    logger.debug("Building KD-tree with %d points (cached)", len(coords))
    tree = cKDTree(coords)
    _tree_cache[cache_key] = tree
    return tree


def clear_tree_cache() -> None:
    """Clear the KD-tree cache. Call between pipeline runs."""
    _tree_cache.clear()


def nearest_distances(
    systems: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
) -> pd.Series:
    """Compute distance from each system to the nearest target point.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations (must have ``pwsid`` column).
    targets : GeoDataFrame
        Target point locations (e.g. facilities).

    Returns
    -------
    pd.Series
        Distance in meters, indexed by ``pwsid``. If *targets* is empty,
        all distances are ``np.inf``.
    """
    sys_proj = to_conus_albers(systems)

    if len(targets) == 0:
        return cast(
            pd.Series,
            pd.Series(
                np.inf,
                index=sys_proj["pwsid"],
                dtype=np.float64,
                name="dist_nearest",
            ),
        )

    tree = build_spatial_index(targets)
    sys_coords = _extract_coords(sys_proj)
    distances, _ = tree.query(sys_coords, k=1)

    return cast(
        pd.Series,
        pd.Series(
            distances,
            index=sys_proj["pwsid"].values,
            dtype=np.float64,
            name="dist_nearest",
        ),
    )


def query_k_nearest(
    systems: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    k: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Query k-nearest target points for each system.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations (must have ``pwsid`` column).
    targets : GeoDataFrame
        Target point locations (e.g. facilities).
    k : int
        Number of nearest neighbors to find.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.int64]]
        ``(distances, indices)`` arrays of shape ``(n_systems, k)``.
        If *targets* has fewer than *k* points, *k* is clamped.
        If *targets* is empty, distances are ``np.inf`` and indices are ``-1``.
    """
    sys_proj = to_conus_albers(systems)
    n_systems = len(sys_proj)

    if len(targets) == 0:
        return (
            np.full((n_systems, k), np.inf, dtype=np.float64),
            np.full((n_systems, k), -1, dtype=np.int64),
        )

    k_actual = min(k, len(targets))
    tree = build_spatial_index(targets)
    sys_coords = _extract_coords(sys_proj)
    distances, indices = tree.query(sys_coords, k=k_actual)

    # Ensure 2D shape even when k_actual == 1
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    # Pad if targets < k
    if k_actual < k:
        pad_d = np.full((n_systems, k - k_actual), np.inf, dtype=np.float64)
        pad_i = np.full((n_systems, k - k_actual), -1, dtype=np.int64)
        distances = np.hstack([distances, pad_d])
        indices = np.hstack([indices, pad_i])

    return distances.astype(np.float64), indices.astype(np.int64)


def kernel_density_at_points(
    systems: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    bandwidth_m: float,
) -> pd.Series:
    """Compute Gaussian kernel density of targets at each system location.

    Uses a Gaussian kernel: K(d) = exp(-0.5 * (d / bandwidth)^2).
    The result is the sum of kernel contributions from all targets within
    3 * bandwidth (effectively zero beyond that).

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations (must have ``pwsid`` column).
    targets : GeoDataFrame
        Target point locations (e.g. facilities).
    bandwidth_m : float
        Kernel bandwidth in meters.

    Returns
    -------
    pd.Series
        Kernel density estimate, indexed by ``pwsid``.
    """
    sys_proj = to_conus_albers(systems)

    if len(targets) == 0:
        return cast(
            pd.Series,
            pd.Series(
                0.0,
                index=sys_proj["pwsid"] if "pwsid" in sys_proj.columns else sys_proj.index,
                dtype=np.float64,
                name="kernel_density",
            ),
        )

    tree = build_spatial_index(targets)
    sys_coords = _extract_coords(sys_proj)
    cutoff = 3.0 * bandwidth_m

    # Query all targets within cutoff radius
    neighbors = tree.query_ball_point(sys_coords, r=cutoff)
    target_coords = _extract_coords(to_conus_albers(targets))

    densities = np.zeros(len(sys_coords), dtype=np.float64)
    for i, nbr_indices in enumerate(neighbors):
        if nbr_indices:
            dists = np.sqrt(np.sum((target_coords[nbr_indices] - sys_coords[i]) ** 2, axis=1))
            densities[i] = np.sum(np.exp(-0.5 * (dists / bandwidth_m) ** 2))

    pwsid_vals = (
        sys_proj["pwsid"].to_numpy() if "pwsid" in sys_proj.columns else sys_proj.index.to_numpy()
    )
    return cast(
        pd.Series,
        pd.Series(
            densities,
            index=pwsid_vals,
            dtype=np.float64,
            name="kernel_density",
        ),
    )


def count_within_radius(
    systems: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """Count target points within a given radius of each system.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations (must have ``pwsid`` column).
    targets : GeoDataFrame
        Target point locations (e.g. facilities).
    radius_m : float
        Search radius in meters.

    Returns
    -------
    pd.Series
        Integer count, indexed by ``pwsid``.
    """
    sys_proj = to_conus_albers(systems)

    if len(targets) == 0:
        return cast(
            pd.Series,
            pd.Series(
                0,
                index=sys_proj["pwsid"],
                dtype=np.int64,
                name="count",
            ),
        )

    tree = build_spatial_index(targets)
    sys_coords = _extract_coords(sys_proj)
    counts = tree.query_ball_point(sys_coords, r=radius_m, return_length=True)

    return cast(
        pd.Series,
        pd.Series(
            counts,
            index=sys_proj["pwsid"].values,
            dtype=np.int64,
            name="count",
        ),
    )
