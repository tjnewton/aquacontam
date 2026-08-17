"""Tier 2 hydrogeology features — USGS principal aquifer attributes.

Extracts aquifer name and rock-type lithology for each water system via spatial
join with USGS principal aquifer polygons. The USGS release carries no
confinement attribute (see ``configs/data.yaml``), so no confinement signal is
available.

Typical usage::

    from aquacontam.features.hydrogeology import (
        download_principal_aquifers,
        extract_aquifer_features,
    )

    aquifer_path = download_principal_aquifers(Path("data/raw"))
    features = extract_aquifer_features(systems_gdf, aquifer_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import DOWNLOAD_TIMEOUT_LARGE
from aquacontam.data._download import download_and_extract
from aquacontam.geo.crs import to_conus_albers

logger = logging.getLogger(__name__)


def download_principal_aquifers(raw_dir: Path, *, force: bool = False) -> Path:
    """Download USGS principal aquifer shapefile.

    Parameters
    ----------
    raw_dir : Path
        Directory to save downloaded files.
    force : bool
        Re-download even if files already exist.

    Returns
    -------
    Path
        Path to the extracted shapefile (.shp).
    """
    config = load_data_config()["usgs_aquifers"]
    url: str = config["url"]
    expected_files: list[str] = config["expected_files"]

    paths = download_and_extract(
        url,
        raw_dir,
        expected_files,
        force=force,
        timeout=DOWNLOAD_TIMEOUT_LARGE,
        progress_desc="Aquifer download",
    )
    return paths[0]


def extract_aquifer_features(
    systems: gpd.GeoDataFrame,
    aquifer_path: Path | str,
) -> pd.DataFrame:
    """Extract aquifer attributes for each water system via spatial join.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations with ``pwsid`` column.
    aquifer_path : Path or str
        Path to the USGS principal aquifer shapefile or GeoPackage.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with columns:
        - ``aquifer_type`` — principal-aquifer name (from ``AQ_NAME``)
        - ``aquifer_lithology`` — rock-type lithology class (from ``ROCK_TYPE``)
        - ``aquifer_confinement`` — always missing: the USGS release has no
          confinement attribute (``AQ_TYPE`` absent), so this column is entirely
          ``NaN`` and survives only as a constant missingness indicator.

        Points outside all aquifer polygons have ``NaN`` values.
    """
    config = load_data_config()["usgs_aquifers"]
    col_map = config["column_map"]

    aquifers = gpd.read_file(aquifer_path)
    logger.info("Loaded %d aquifer polygons", len(aquifers))

    # Ensure both are in the same CRS for spatial join
    systems_proj = to_conus_albers(systems)
    # Ensure pwsid is a column (it may be the index)
    if "pwsid" not in systems_proj.columns and systems_proj.index.name == "pwsid":
        systems_proj = systems_proj.reset_index()
    aquifers_proj = to_conus_albers(aquifers)

    # Spatial join: points within polygons
    joined = gpd.sjoin(
        systems_proj[["pwsid", "geometry"]],
        aquifers_proj,
        how="left",
        predicate="within",
    )

    # Deduplicate — if a system falls within multiple aquifer polygons
    # (overlapping boundaries), keep the first match
    joined = joined.drop_duplicates(subset="pwsid", keep="first")

    # Extract and rename columns
    result = pd.DataFrame(index=joined["pwsid"].values)
    result.index.name = "pwsid"

    for src_col, dst_col in col_map.items():
        if src_col in joined.columns:
            result[dst_col] = joined[src_col].to_numpy()
        else:
            logger.warning("Column %s not found in aquifer data", src_col)
            result[dst_col] = pd.NA

    logger.info(
        "Extracted aquifer features for %d systems (%d with aquifer data)",
        len(result),
        result["aquifer_type"].notna().sum() if "aquifer_type" in result.columns else 0,
    )
    return cast(pd.DataFrame, result)
