"""Leakage-free spatial neighborhood features.

Computes neighborhood-based features using ONLY training set labels to
prevent information leakage. For validation/test systems, the detection
prevalence of k-nearest training-set neighbors provides a spatial prior
without leaking the target variable.

Critical constraint: all label-based features are computed per-split,
using only the training set's labels for all splits.
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from aquacontam.geo.crs import to_conus_albers

logger = logging.getLogger(__name__)


def _load_spatial_graph_config() -> dict[str, object]:
    """Load spatial graph config from experiment.yaml (if available)."""
    try:
        from aquacontam._config import load_experiment_config

        cfg = load_experiment_config(section="feature_extraction")
        result: dict[str, object] = cfg.get("spatial_graph", {})  # type: ignore[assignment]
        return result
    except (FileNotFoundError, KeyError, TypeError):
        return {}


_cfg = _load_spatial_graph_config()

# K values for neighbor detection prevalence
NEIGHBOR_K_VALUES: tuple[int, ...] = tuple(
    _cfg.get("neighbor_k_values", (5, 10, 20))  # type: ignore[arg-type]
)


def _build_coords_array(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract EPSG:5070 (x, y) coordinates from a GeoDataFrame."""
    proj = to_conus_albers(gdf)
    return np.column_stack([proj.geometry.x, proj.geometry.y])


def compute_neighbor_prevalence(
    query_systems: gpd.GeoDataFrame,
    train_systems: gpd.GeoDataFrame,
    train_labels: pd.Series,
    k_values: tuple[int, ...] = NEIGHBOR_K_VALUES,
) -> pd.DataFrame:
    """Compute detection prevalence among k-nearest training neighbors.

    For each query system, finds the k nearest systems in the training set
    and computes the fraction with positive labels.

    Parameters
    ----------
    query_systems : GeoDataFrame
        Systems to compute features for (any split).
    train_systems : GeoDataFrame
        Training set systems (used as reference neighbors).
    train_labels : pd.Series
        Binary labels for training systems, indexed by ``pwsid``.
    k_values : tuple[int, ...]
        K values for neighbor queries.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with columns like ``nbr_prev_k5``,
        ``nbr_prev_k10``, ``nbr_prev_k20``.
    """
    if "pwsid" not in query_systems.columns and query_systems.index.name == "pwsid":
        query_systems = query_systems.reset_index()
    if "pwsid" not in train_systems.columns and train_systems.index.name == "pwsid":
        train_systems = train_systems.reset_index()

    query_pwsids = query_systems["pwsid"].to_numpy()
    n_query = len(query_systems)

    if len(train_systems) == 0 or len(train_labels) == 0:
        result = pd.DataFrame(
            {f"nbr_prev_k{k}": np.full(n_query, np.nan) for k in k_values},
            index=query_pwsids,
        )
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    # Build KD-tree from training systems
    train_coords = _build_coords_array(train_systems)
    query_coords = _build_coords_array(query_systems)
    tree = cKDTree(train_coords)

    # Align labels with training system order
    train_pwsids = train_systems["pwsid"].to_numpy()
    labels_aligned = train_labels.reindex(train_pwsids).fillna(0).to_numpy().astype(float)

    max_k = min(max(k_values), len(train_systems))
    _, indices = tree.query(query_coords, k=max_k)

    # Ensure 2D
    if indices.ndim == 1:
        indices = indices[:, np.newaxis]

    features: dict[str, np.ndarray] = {}
    for k in k_values:
        k_actual = min(k, max_k)
        k_indices = indices[:, :k_actual]
        k_labels = labels_aligned[k_indices]
        features[f"nbr_prev_k{k}"] = np.mean(k_labels, axis=1)

    result = pd.DataFrame(features, index=query_pwsids)
    result.index.name = "pwsid"
    return cast(pd.DataFrame, result)


def compute_neighbor_characteristics(
    query_systems: gpd.GeoDataFrame,
    train_systems: gpd.GeoDataFrame,
    train_features: pd.DataFrame,
    k: int = 10,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute average characteristics of k-nearest training neighbors.

    Parameters
    ----------
    query_systems : GeoDataFrame
        Systems to compute features for.
    train_systems : GeoDataFrame
        Training set systems.
    train_features : pd.DataFrame
        Feature DataFrame for training systems, indexed by ``pwsid``.
    k : int
        Number of nearest neighbors.
    feature_columns : list[str] or None
        Columns to average. Defaults to numeric columns available.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with ``nbr_mean_{col}`` columns.
    """
    if "pwsid" not in query_systems.columns and query_systems.index.name == "pwsid":
        query_systems = query_systems.reset_index()
    if "pwsid" not in train_systems.columns and train_systems.index.name == "pwsid":
        train_systems = train_systems.reset_index()

    query_pwsids = query_systems["pwsid"].to_numpy()
    n_query = len(query_systems)

    if feature_columns is None:
        feature_columns = train_features.select_dtypes(include=[np.number]).columns.tolist()

    if len(train_systems) == 0 or not feature_columns:
        result = pd.DataFrame(
            {f"nbr_mean_{c}": np.full(n_query, np.nan) for c in feature_columns},
            index=query_pwsids,
        )
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    train_coords = _build_coords_array(train_systems)
    query_coords = _build_coords_array(query_systems)
    tree = cKDTree(train_coords)

    k_actual = min(k, len(train_systems))
    _, indices = tree.query(query_coords, k=k_actual)
    if indices.ndim == 1:
        indices = indices[:, np.newaxis]

    # Align features with training system order
    train_pwsids = train_systems["pwsid"].to_numpy()
    train_feat_aligned = train_features.reindex(train_pwsids)

    features: dict[str, np.ndarray] = {}
    for col in feature_columns:
        col_vals = train_feat_aligned[col].fillna(0).to_numpy().astype(float)
        k_vals = col_vals[indices]
        features[f"nbr_mean_{col}"] = np.mean(k_vals, axis=1)

    result = pd.DataFrame(features, index=query_pwsids)
    result.index.name = "pwsid"
    return cast(pd.DataFrame, result)


def extract_spatial_graph_features(
    query_systems: gpd.GeoDataFrame,
    train_systems: gpd.GeoDataFrame,
    train_labels: pd.Series,
    k_values: tuple[int, ...] = NEIGHBOR_K_VALUES,
) -> pd.DataFrame:
    """Extract all leakage-free spatial graph features.

    Parameters
    ----------
    query_systems : GeoDataFrame
        Systems to compute features for (train, val, or test split).
    train_systems : GeoDataFrame
        Training set systems only.
    train_labels : pd.Series
        Binary labels for training set, indexed by ``pwsid``.
    k_values : tuple[int, ...]
        K values for neighbor prevalence computation.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with spatial neighborhood features.
    """
    result = compute_neighbor_prevalence(query_systems, train_systems, train_labels, k_values)
    logger.info("Extracted %d spatial graph features", len(result.columns))
    return result
