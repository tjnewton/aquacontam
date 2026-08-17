"""DoD PFAS site proximity features.

Computes proximity features for water systems based on DoD/federal PFAS
contamination sites from EPA PFAS Analytic Tools. These sites represent
known or suspected PFAS contamination, primarily from AFFF (aqueous
film-forming foam) use at military installations.

Typical usage::

    from aquacontam.features.dod_proximity import extract_dod_features
    from aquacontam.data.dod_pfas import load_dod_pfas

    sites = load_dod_pfas(Path("data/raw"))
    features = extract_dod_features(systems_gdf, sites)
"""

from __future__ import annotations

import logging
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd

from aquacontam.features._proximity_common import (
    compute_proximity_features,
    to_geodataframe,
)

logger = logging.getLogger(__name__)


def extract_dod_features(
    systems: gpd.GeoDataFrame,
    dod_sites: pd.DataFrame,
    radii_m: tuple[float, ...] = (5000.0, 10000.0, 25000.0),
) -> pd.DataFrame:
    """Compute DoD PFAS site proximity features for water systems.

    Features produced:

    - ``nearest_dod_pfas_m``: Distance to nearest DoD PFAS site (meters)
    - ``log1p_nearest_dod_pfas``: log1p of nearest distance
    - ``count_dod_pfas_{radius}km``: Number of DoD PFAS sites within radius

    Parameters
    ----------
    systems : GeoDataFrame
        Water systems with geometry (EPSG:4326) and ``pwsid`` column.
    dod_sites : pd.DataFrame
        DoD PFAS site data from ``load_dod_pfas()``.
    radii_m : tuple[float, ...]
        Radii in meters for count features.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame indexed by system index with DoD proximity columns.
    """
    dod_gdf = to_geodataframe(dod_sites) if len(dod_sites) > 0 else dod_sites

    if len(dod_gdf) == 0:
        logger.warning("No DoD PFAS sites with valid coordinates — returning NaN features")
        cols = ["nearest_dod_pfas_m", "log1p_nearest_dod_pfas"]
        for r in radii_m:
            cols.append(f"count_dod_pfas_{r / 1000:.0f}km")
        return cast(pd.DataFrame, pd.DataFrame(np.nan, index=systems.index, columns=cols))

    # Use shared proximity helper for nearest distance + counts
    result = compute_proximity_features(systems, dod_gdf, radii_m, prefix="dod_pfas")

    # Add DoD-specific log1p feature
    result["log1p_nearest_dod_pfas"] = np.log1p(result["nearest_dod_pfas_m"])

    logger.info(
        "Computed DoD PFAS proximity features: %d systems x %d sites",
        len(systems),
        len(dod_gdf),
    )
    return result
