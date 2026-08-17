"""EPA Facility Registry Service (FRS) data loader.

Downloads and parses the national FRS dataset to produce a GeoDataFrame of
facility locations. This is NOT a ``DataSource`` subclass — FRS facilities
are auxiliary geospatial data, not water quality samples.

Typical usage::

    from aquacontam.data.epa_frs import download_frs, load_frs

    download_frs(Path("data/raw"))
    facilities = load_frs(Path("data/raw"))
    wwtps = load_frs(Path("data/raw"), facility_types=["wwtp"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from aquacontam._config import load_data_config
from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    CRS_STORAGE,
    DOWNLOAD_TIMEOUT_LARGE,
    FRS_INTEREST_TYPES,
    FRS_NAICS_CODES,
    FRS_SIC_CODES,
)
from aquacontam.data._download import download_and_extract

logger = logging.getLogger(__name__)


def download_frs(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the national FRS CSV from EPA.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the downloaded file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded/extracted CSV file.
    """
    config = load_data_config()["epa_frs"]
    url: str = config["url"]
    expected_files: list[str] = config["expected_files"]

    paths = download_and_extract(
        url,
        raw_dir,
        expected_files,
        force=force,
        timeout=DOWNLOAD_TIMEOUT_LARGE,
        progress_desc="FRS download",
    )
    return paths[0]


def load_frs(
    raw_dir: Path,
    facility_types: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load and parse FRS facility data into a GeoDataFrame.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the FRS CSV file.
    facility_types : list[str] or None
        If provided, filter to these facility types. Valid values:
        ``"wwtp"``, ``"landfill"``, ``"airport"``, ``"military"``,
        ``"industrial"``. If ``None``, return all facilities.

    Returns
    -------
    GeoDataFrame
        Facility locations with columns: ``facility_name``, ``latitude``,
        ``longitude``, ``sic_codes``, ``naics_codes``, ``state_code``,
        ``registry_id``.
    """
    config = load_data_config()["epa_frs"]
    expected_file: str = config["expected_files"][0]
    filepath = raw_dir / expected_file

    if not filepath.exists():
        raise FileNotFoundError(f"FRS data file not found: {filepath}. Run download_frs() first.")

    col_map: dict[str, str] = config["column_map"]
    encoding: str = config.get("format", {}).get("encoding", "latin-1")

    logger.info("Reading FRS data from %s …", filepath)
    df = pd.read_csv(
        filepath,
        encoding=encoding,
        dtype=str,
        low_memory=False,
    )
    logger.info("Read %d rows from FRS", len(df))

    # Rename columns per config
    rename = {raw: std for raw, std in col_map.items() if raw in df.columns}
    df = df.rename(columns=rename)

    # Coerce coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows without valid coordinates
    valid_coords = df["latitude"].notna() & df["longitude"].notna()
    n_dropped = (~valid_coords).sum()
    if n_dropped > 0:
        logger.info("Dropping %d facilities without valid coordinates", n_dropped)
    df = df[valid_coords].copy()

    # Filter to CONUS bounding box
    conus_mask = (
        (df["latitude"] >= CONUS_LAT_MIN)
        & (df["latitude"] <= CONUS_LAT_MAX)
        & (df["longitude"] >= CONUS_LON_MIN)
        & (df["longitude"] <= CONUS_LON_MAX)
    )
    n_outside = (~conus_mask).sum()
    if n_outside > 0:
        logger.info("Dropping %d facilities outside CONUS bounding box", n_outside)
    df = df[conus_mask].copy()

    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"], strict=True)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_STORAGE)

    # Filter by facility type if requested
    if facility_types is not None:
        gdf = filter_by_type(gdf, facility_types)

    logger.info("Loaded %d FRS facilities", len(gdf))
    return gdf


def filter_by_type(
    gdf: gpd.GeoDataFrame,
    facility_types: list[str],
) -> gpd.GeoDataFrame:
    """Filter facilities by SIC/NAICS codes matching requested types.

    Parameters
    ----------
    gdf : GeoDataFrame
        Full FRS facility data.
    facility_types : list[str]
        Facility types to keep (e.g. ``["wwtp", "landfill"]``).

    Returns
    -------
    GeoDataFrame
        Filtered subset.
    """
    if len(gdf) == 0:
        return gdf

    masks = []
    for ftype in facility_types:
        ftype = ftype.lower()
        mask = pd.Series(False, index=gdf.index)

        # Match by SIC codes
        if ftype in FRS_SIC_CODES and "sic_codes" in gdf.columns:
            sic_col = gdf["sic_codes"].astype(str).fillna("")
            for code in FRS_SIC_CODES[ftype]:
                mask = mask | sic_col.str.contains(code, na=False)

        # Match by NAICS codes
        if ftype in FRS_NAICS_CODES and "naics_codes" in gdf.columns:
            naics_col = gdf["naics_codes"].astype(str).fillna("")
            for code in FRS_NAICS_CODES[ftype]:
                mask = mask | naics_col.str.contains(code, na=False)

        # Match by interest types
        if ftype in FRS_INTEREST_TYPES and "interest_types" in gdf.columns:
            interest_col = gdf["interest_types"].astype(str).fillna("")
            for itype in FRS_INTEREST_TYPES[ftype]:
                mask = mask | interest_col.str.contains(itype, na=False)

        masks.append(mask)

    if not masks:
        return gdf

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    return gdf[combined].copy()
