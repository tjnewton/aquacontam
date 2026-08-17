"""Tier 1 proximity features — distance to nearest facilities and counts within radii.

Computes proximity features for water systems based on EPA FRS facility
locations. Uses KD-tree spatial indexing for efficient nearest-neighbor
queries in EPSG:5070 (Conus Albers, meters).

Typical usage::

    from aquacontam.features.proximity import extract_all_proximity_features
    from aquacontam.data.epa_frs import load_frs

    facilities = load_frs(Path("data/raw"))
    features = extract_all_proximity_features(systems_gdf, facilities)
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from aquacontam._constants import FRS_SIC_CODES
from aquacontam.data.epa_frs import filter_by_type
from aquacontam.geo.distance import count_within_radius, nearest_distances

logger = logging.getLogger(__name__)

# Facility type configurations: (type_name, label, radii_for_counts)
FACILITY_CONFIGS: list[tuple[str, str, tuple[float, ...]]] = [
    ("industrial", "industrial", (1000.0, 5000.0, 10000.0)),
    ("military", "military", (5000.0,)),
    ("airport", "airport", (10000.0,)),
    ("wwtp", "wwtp", (5000.0,)),
    ("landfill", "landfill", (10000.0,)),
]


def compute_proximity_features(
    systems: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    facility_label: str,
    radii_m: tuple[float, ...] = (1000.0, 5000.0, 10000.0),
) -> pd.DataFrame:
    """Compute distance and count features for one facility type.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` column and point geometry.
    facilities : GeoDataFrame
        Facility locations (point geometry).
    facility_label : str
        Label used in output column names (e.g. ``"industrial"``).
    radii_m : tuple[float, ...]
        Radii in meters for count-within-radius features.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with columns:
        - ``dist_nearest_{label}`` — distance to nearest facility (meters)
        - ``log1p_dist_nearest_{label}`` — log1p of distance (for distance-decay)
        - ``n_{label}_{r}m`` — count within each radius
    """
    # Ensure pwsid is a column (it may be the index)
    if "pwsid" not in systems.columns and systems.index.name == "pwsid":
        systems = systems.reset_index()
    result = pd.DataFrame(index=systems["pwsid"].values)
    result.index.name = "pwsid"

    # Nearest distance (raw meters + log1p for distance-decay modeling)
    dist = nearest_distances(systems, facilities)
    result[f"dist_nearest_{facility_label}"] = dist.to_numpy()
    result[f"log1p_dist_nearest_{facility_label}"] = np.log1p(dist.to_numpy())

    # Count within each radius
    for radius in radii_m:
        count = count_within_radius(systems, facilities, radius)
        col_name = f"n_{facility_label}_{int(radius)}m"
        result[col_name] = count.to_numpy()

    logger.info(
        "Computed %d proximity features for %s (%d facilities)",
        len(result.columns),
        facility_label,
        len(facilities),
    )
    return cast(pd.DataFrame, result)


def extract_all_proximity_features(
    systems: gpd.GeoDataFrame,
    frs_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Extract all proximity features for all configured facility types.

    Filters the FRS data by facility type and computes proximity features
    for each type, then merges all results into a single DataFrame.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system locations with ``pwsid`` and point geometry.
    frs_gdf : GeoDataFrame
        Full FRS facility GeoDataFrame (unfiltered).

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with all proximity feature columns.
    """
    all_features: list[pd.DataFrame] = []

    for ftype, label, radii in FACILITY_CONFIGS:
        if ftype not in FRS_SIC_CODES:
            logger.warning("No SIC codes defined for facility type: %s", ftype)
            continue

        filtered = filter_by_type(frs_gdf, [ftype])
        logger.info("Filtered %d %s facilities from FRS", len(filtered), ftype)

        features = compute_proximity_features(systems, filtered, label, radii)
        all_features.append(features)

    if not all_features:
        return cast(pd.DataFrame, pd.DataFrame(index=systems["pwsid"].values))

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"
    logger.info("Total proximity features: %d columns", len(result.columns))
    return result
