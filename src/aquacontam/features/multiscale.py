"""Multi-scale spatial features for contamination prediction.

Extracts richer geospatial signal from existing EPA FRS, NLCD, and USGS
data by computing:
- Distance-decay kernel densities at multiple bandwidths
- K-nearest-neighbor distance statistics per facility type
- Land use ring gradients (outer - inner fractions)
- Land use Shannon diversity per buffer
- Facility x land use interaction features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from aquacontam._constants import FRS_SIC_CODES, NLCD_CLASS_MAP
from aquacontam.data.epa_frs import filter_by_type
from aquacontam.geo.distance import (
    kernel_density_at_points,
    query_k_nearest,
)
from aquacontam.geo.raster import categorical_fractions

logger = logging.getLogger(__name__)


def _load_feature_config() -> dict[str, Any]:
    """Load multiscale feature config from experiment.yaml (if available)."""
    try:
        from aquacontam._config import load_experiment_config

        cfg = load_experiment_config(section="feature_extraction")
        result: dict[str, Any] = cfg.get("multiscale", {})
        return result
    except (FileNotFoundError, KeyError, TypeError):
        return {}


_cfg = _load_feature_config()

# Kernel bandwidths (meters) for distance-decay density features
KERNEL_BANDWIDTHS: tuple[float, ...] = tuple(
    _cfg.get("kernel_bandwidths", (500.0, 2000.0, 5000.0, 10000.0, 25000.0))
)

# Facility types to compute multi-scale features for
FACILITY_TYPES: tuple[str, ...] = (
    "industrial",
    "military",
    "airport",
    "wwtp",
    "landfill",
)

# K values for KNN statistics
KNN_K_VALUES: tuple[int, ...] = tuple(_cfg.get("knn_k_values", (1, 3, 5, 10)))

# Land use ring pairs (inner_m, outer_m) for gradient features
_default_ring_pairs = ((500.0, 2000.0), (1000.0, 5000.0), (2000.0, 10000.0), (5000.0, 25000.0))
RING_PAIRS: tuple[tuple[float, float], ...] = tuple(
    tuple(p) for p in _cfg.get("ring_pairs", _default_ring_pairs)
)

# Land use categories for ring gradients
GRADIENT_CATEGORIES: tuple[str, ...] = (
    "developed",
    "agriculture",
    "forest",
    "wetland",
)

# Buffer radii for Shannon diversity
DIVERSITY_BUFFERS: tuple[float, ...] = tuple(
    _cfg.get("diversity_buffers", (1000.0, 5000.0, 10000.0))
)


def extract_kernel_density_features(
    systems: gpd.GeoDataFrame,
    frs_gdf: gpd.GeoDataFrame,
    bandwidths: tuple[float, ...] = KERNEL_BANDWIDTHS,
    facility_types: tuple[str, ...] = FACILITY_TYPES,
) -> pd.DataFrame:
    """Compute kernel density features at multiple bandwidths per facility type.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column.
    frs_gdf : GeoDataFrame
        Full FRS facility GeoDataFrame (unfiltered).
    bandwidths : tuple[float, ...]
        Kernel bandwidths in meters.
    facility_types : tuple[str, ...]
        Facility types to compute densities for.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with kernel density columns.
    """
    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()

    all_features: list[pd.DataFrame] = []

    for ftype in facility_types:
        if ftype not in FRS_SIC_CODES:
            continue
        filtered = filter_by_type(frs_gdf, [ftype])
        if filtered.empty:
            continue

        for bw in bandwidths:
            density = kernel_density_at_points(systems, filtered, bw)
            col_name = f"kd_{ftype}_{int(bw)}m"
            df = pd.DataFrame(
                {col_name: density.to_numpy()},
                index=systems["pwsid"].to_numpy(),
            )
            df.index.name = "pwsid"
            all_features.append(df)

    if not all_features:
        result = pd.DataFrame(index=systems["pwsid"].to_numpy())
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"
    logger.info("Extracted %d kernel density features", len(result.columns))
    return result


def extract_knn_features(
    systems: gpd.GeoDataFrame,
    frs_gdf: gpd.GeoDataFrame,
    k_values: tuple[int, ...] = KNN_K_VALUES,
    facility_types: tuple[str, ...] = FACILITY_TYPES,
) -> pd.DataFrame:
    """Compute k-nearest-neighbor distance statistics per facility type.

    For each facility type and k value, computes:
    - Mean distance to k nearest facilities
    - Std deviation of distances to k nearest
    - Distance to k-th nearest facility

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column.
    frs_gdf : GeoDataFrame
        Full FRS facility GeoDataFrame (unfiltered).
    k_values : tuple[int, ...]
        K values for KNN queries.
    facility_types : tuple[str, ...]
        Facility types to compute KNN stats for.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with KNN statistic columns.
    """
    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()

    max_k = max(k_values)
    all_features: list[pd.DataFrame] = []

    for ftype in facility_types:
        if ftype not in FRS_SIC_CODES:
            continue
        filtered = filter_by_type(frs_gdf, [ftype])

        dists, _ = query_k_nearest(systems, filtered, k=max_k)

        for k in k_values:
            k_dists = dists[:, :k]
            prefix = f"knn{k}_{ftype}"

            # Replace inf with NaN for stats computation
            k_dists_safe = np.where(np.isinf(k_dists), np.nan, k_dists)

            df = pd.DataFrame(
                {
                    f"{prefix}_mean_dist": np.nanmean(k_dists_safe, axis=1),
                    f"{prefix}_std_dist": np.nanstd(k_dists_safe, axis=1),
                    f"{prefix}_max_dist": k_dists[:, k - 1],  # k-th nearest
                },
                index=systems["pwsid"].to_numpy(),
            )
            df.index.name = "pwsid"
            all_features.append(df)

    if not all_features:
        result = pd.DataFrame(index=systems["pwsid"].to_numpy())
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"
    logger.info("Extracted %d KNN features", len(result.columns))
    return result


def extract_land_use_ring_gradients(
    systems: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    ring_pairs: tuple[tuple[float, float], ...] = RING_PAIRS,
    categories: tuple[str, ...] = GRADIENT_CATEGORIES,
) -> pd.DataFrame:
    """Compute land use ring gradients (outer fraction - inner fraction).

    Captures urban edge effects and contamination gradients by comparing
    land use composition at different distances from the system.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column.
    nlcd_path : Path or str
        Path to NLCD GeoTIFF raster.
    ring_pairs : tuple[tuple[float, float], ...]
        Pairs of (inner_radius_m, outer_radius_m).
    categories : tuple[str, ...]
        Land use categories to compute gradients for.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with gradient columns.
    """
    # Pre-compute fractions at all needed radii
    all_radii = sorted({r for pair in ring_pairs for r in pair})
    radius_fracs: dict[float, pd.DataFrame] = {}
    for radius in all_radii:
        radius_fracs[radius] = categorical_fractions(systems, nlcd_path, radius, NLCD_CLASS_MAP)

    all_features: list[pd.DataFrame] = []
    for inner_m, outer_m in ring_pairs:
        inner_fracs = radius_fracs[inner_m]
        outer_fracs = radius_fracs[outer_m]

        for cat in categories:
            inner_col = inner_fracs.get(cat, pd.Series(0.0, index=inner_fracs.index))
            outer_col = outer_fracs.get(cat, pd.Series(0.0, index=outer_fracs.index))
            gradient = outer_col - inner_col
            col_name = f"lu_grad_{cat}_{int(inner_m)}_{int(outer_m)}m"
            df = pd.DataFrame({col_name: gradient.to_numpy()}, index=inner_fracs.index)
            df.index.name = "pwsid"
            all_features.append(df)

    if not all_features:
        result = pd.DataFrame(
            index=systems["pwsid"].to_numpy() if "pwsid" in systems.columns else systems.index
        )
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"
    logger.info("Extracted %d land use gradient features", len(result.columns))
    return result


def extract_land_use_diversity(
    systems: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    buffers_m: tuple[float, ...] = DIVERSITY_BUFFERS,
) -> pd.DataFrame:
    """Compute Shannon diversity of land use within buffers.

    Urban-rural transition zones with high diversity correlate with
    legacy contamination patterns.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column.
    nlcd_path : Path or str
        Path to NLCD GeoTIFF raster.
    buffers_m : tuple[float, ...]
        Buffer radii in meters.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with Shannon diversity columns.
    """
    all_features: list[pd.DataFrame] = []

    for buf in buffers_m:
        fracs = categorical_fractions(systems, nlcd_path, buf, NLCD_CLASS_MAP)
        # Shannon diversity: H = -sum(p * log(p)) for p > 0
        fracs_arr = fracs.to_numpy()
        mask = fracs_arr > 0
        log_fracs = np.zeros_like(fracs_arr)
        log_fracs[mask] = np.log(fracs_arr[mask])
        diversity = -np.sum(fracs_arr * log_fracs, axis=1)

        col_name = f"lu_diversity_{int(buf / 1000)}km"
        df = pd.DataFrame(
            {col_name: diversity},
            index=fracs.index,
        )
        df.index.name = "pwsid"
        all_features.append(df)

    if not all_features:
        result = pd.DataFrame(
            index=systems["pwsid"].to_numpy() if "pwsid" in systems.columns else systems.index
        )
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"
    logger.info("Extracted %d land use diversity features", len(result.columns))
    return result


def extract_facility_landuse_interactions(
    systems: gpd.GeoDataFrame,
    frs_gdf: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    buffer_m: float = 5000.0,
    facility_types: tuple[str, ...] = FACILITY_TYPES,
    landuse_categories: tuple[str, ...] = GRADIENT_CATEGORIES,
) -> pd.DataFrame:
    """Compute facility count x land use fraction interaction features.

    An industrial facility in a developed area carries different risk
    than one in a forested area.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column.
    frs_gdf : GeoDataFrame
        Full FRS facility GeoDataFrame.
    nlcd_path : Path or str
        Path to NLCD GeoTIFF raster.
    buffer_m : float
        Buffer radius in meters for both counts and land use.
    facility_types : tuple[str, ...]
        Facility types to compute interactions for.
    landuse_categories : tuple[str, ...]
        Land use categories to interact with.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with interaction columns.
    """
    from aquacontam.geo.distance import count_within_radius

    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()

    # Get land use fractions
    lu_fracs = categorical_fractions(systems, nlcd_path, buffer_m, NLCD_CLASS_MAP)

    all_features: list[pd.DataFrame] = []

    for ftype in facility_types:
        if ftype not in FRS_SIC_CODES:
            continue
        filtered = filter_by_type(frs_gdf, [ftype])
        counts = count_within_radius(systems, filtered, buffer_m)

        for cat in landuse_categories:
            lu_col = lu_fracs.get(cat, pd.Series(0.0, index=lu_fracs.index))
            interaction = counts.to_numpy() * lu_col.to_numpy()
            col_name = f"ix_{ftype}_{cat}_{int(buffer_m / 1000)}km"
            df = pd.DataFrame(
                {col_name: interaction},
                index=systems["pwsid"].to_numpy(),
            )
            df.index.name = "pwsid"
            all_features.append(df)

    if not all_features:
        result = pd.DataFrame(index=systems["pwsid"].to_numpy())
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"
    logger.info("Extracted %d facilityxlanduse interaction features", len(result.columns))
    return result


def extract_all_multiscale_features(
    systems: gpd.GeoDataFrame,
    frs_gdf: gpd.GeoDataFrame,
    nlcd_path: Path | str | None = None,
) -> pd.DataFrame:
    """Extract all multi-scale spatial features.

    Convenience function that combines kernel densities, KNN stats,
    and optionally land use gradients/diversity/interactions.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column.
    frs_gdf : GeoDataFrame
        Full FRS facility GeoDataFrame.
    nlcd_path : Path or str or None
        Path to NLCD GeoTIFF. If None, land use features are skipped.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with all multi-scale feature columns.
    """
    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()

    parts: list[pd.DataFrame] = []

    # Kernel density features
    kd = extract_kernel_density_features(systems, frs_gdf)
    parts.append(kd)

    # KNN features
    knn = extract_knn_features(systems, frs_gdf)
    parts.append(knn)

    # Land use features (require NLCD raster)
    if nlcd_path is not None:
        gradients = extract_land_use_ring_gradients(systems, nlcd_path)
        parts.append(gradients)

        diversity = extract_land_use_diversity(systems, nlcd_path)
        parts.append(diversity)

        interactions = extract_facility_landuse_interactions(systems, frs_gdf, nlcd_path)
        parts.append(interactions)

    if not parts:
        result = pd.DataFrame(index=systems["pwsid"].to_numpy())
        result.index.name = "pwsid"
        return cast(pd.DataFrame, result)

    result = pd.concat(parts, axis=1)
    result.index.name = "pwsid"
    logger.info("Total multi-scale features: %d columns", len(result.columns))
    return result
