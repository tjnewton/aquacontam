"""NJ Private Well Testing Act (PWTA) data loader.

Downloads and parses the NJ PWTA private well testing dataset. This is NOT
a ``DataSource`` subclass — private well data has a different structure
(individual wells, not public water systems with PWSIDs).

Typical usage::

    from aquacontam.data.nj_private_wells import download_nj_private_wells, load_nj_private_wells

    download_nj_private_wells(Path("data/raw"))
    wells = load_nj_private_wells(Path("data/raw"))
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    CRS_STORAGE,
    DOWNLOAD_TIMEOUT_LARGE,
)
from aquacontam.data._download import download_and_extract

logger = logging.getLogger(__name__)


def download_nj_private_wells(raw_dir: Path, *, force: bool = False) -> Path:
    """Download NJ PWTA private well testing data.

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
    config = load_data_config()["nj_private_wells"]
    url: str = config["url"]
    expected_files: list[str] = config["expected_files"]
    primary = raw_dir / expected_files[0]

    if not force and primary.exists():
        return primary

    if config.get("unavailable", False):
        reason = config.get("unavailable_reason", "source unavailable")
        raise RuntimeError(f"NJ private wells download skipped: {reason}")

    paths = download_and_extract(
        url,
        raw_dir,
        expected_files,
        force=force,
        timeout=DOWNLOAD_TIMEOUT_LARGE,
        progress_desc="NJ PWTA download",
    )
    return paths[0]


def load_nj_private_wells(
    raw_dir: Path,
    *,
    analytes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load and parse NJ PWTA private well data into a GeoDataFrame.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the PWTA CSV file.
    analytes : list[str] | None
        If provided, filter to these analytes. If None, return all.

    Returns
    -------
    GeoDataFrame
        Private well test results with columns: ``sample_id``, ``analyte``,
        ``concentration``, ``detection_limit``, ``censored``, ``sample_date``.
        CRS is EPSG:4326.
    """
    config = load_data_config()["nj_private_wells"]
    expected_file: str = config["expected_files"][0]
    filepath = raw_dir / expected_file

    if not filepath.exists():
        raise FileNotFoundError(
            f"NJ PWTA data file not found: {filepath}. Run download_nj_private_wells() first."
        )

    col_map: dict[str, str] = config["column_map"]
    encoding: str = config.get("format", {}).get("encoding", "utf-8")

    logger.info("Reading NJ PWTA data from %s …", filepath)
    df = pd.read_csv(filepath, encoding=encoding, dtype=str, low_memory=False)
    logger.info("Read %d rows from NJ PWTA", len(df))

    # Rename columns per config
    rename = {raw: std for raw, std in col_map.items() if raw in df.columns}
    df = df.rename(columns=rename)

    # Coerce numeric columns
    for col in ("concentration", "detection_limit", "latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse sample date
    if "sample_date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")

    # Derive censored column from result qualifier
    if "result_qualifier" in df.columns:
        df["censored"] = df["result_qualifier"].str.strip().str.upper().isin(["U", "ND", "<"])
    else:
        df["censored"] = df["concentration"].isna()

    # Drop rows without valid coordinates
    valid_coords = df["latitude"].notna() & df["longitude"].notna()
    n_dropped = int((~valid_coords).sum())
    if n_dropped > 0:
        logger.info("Dropping %d rows without valid coordinates", n_dropped)
    df = df[valid_coords].copy()

    # Filter to CONUS bounding box (NJ should be fully within)
    conus_mask = (
        (df["latitude"] >= CONUS_LAT_MIN)
        & (df["latitude"] <= CONUS_LAT_MAX)
        & (df["longitude"] >= CONUS_LON_MIN)
        & (df["longitude"] <= CONUS_LON_MAX)
    )
    df = df[conus_mask].copy()

    # Filter to requested analytes
    if analytes is not None and "analyte" in df.columns:
        df = df[df["analyte"].isin(analytes)].copy()

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=CRS_STORAGE,
    )

    logger.info("Loaded %d NJ private well test results", len(gdf))
    return gdf
