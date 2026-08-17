"""DoD / federal PFAS contamination sites from EPA PFAS Analytic Tools.

Downloads the ``PAT_Federal_Sites`` layer from EPA's PFAS Analytic Tools
ArcGIS FeatureServer. Contains 761 federal sites (724 DoD/military) with
known or suspected PFAS contamination.  This is a feature source (like
EPA FRS and TRI), not a ``DataSource`` subclass.

Typical usage::

    from aquacontam.data.dod_pfas import download_dod_pfas, load_dod_pfas

    download_dod_pfas(Path("data/raw"))
    sites = load_dod_pfas(Path("data/raw"))
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from aquacontam._config import load_data_config
from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    DOWNLOAD_TIMEOUT_DEFAULT,
)

logger = logging.getLogger(__name__)

# Agencies that are part of the Department of Defense or military
_DOD_AGENCIES: frozenset[str] = frozenset({"Air Force", "Army", "Navy", "DLA", "FUDS", "USCG"})

# Fields to request from the ArcGIS layer
_OUT_FIELDS = (
    "F_Site_Name,Federal_Agency,State,Latitude,Longitude,"
    "PFAS_Presence,Property_Type,DoD_Reported_Cleanup_Status"
)


def download_dod_pfas(raw_dir: Path, *, force: bool = False) -> Path:
    """Download DoD / federal PFAS site data from EPA PFAS Analytic Tools.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the CSV file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded CSV.
    """
    config = load_data_config()["dod_pfas"]
    expected_file: str = config["expected_files"][0]
    dest = raw_dir / expected_file

    if not force and dest.exists():
        logger.info("DoD PFAS data already exists: %s", dest)
        return dest

    base_url: str = config["arcgis_url"]
    query_url = f"{base_url}/query"

    all_features: list[dict] = []
    offset = 0
    page_size = 2000

    while True:
        params: dict[str, str | int] = {
            "where": "1=1",
            "outFields": _OUT_FIELDS,
            "returnGeometry": "false",
            "f": "json",
            "resultRecordCount": page_size,
            "resultOffset": offset,
        }
        logger.info("Fetching DoD PFAS sites (offset=%d)…", offset)
        try:
            resp = requests.get(query_url, params=params, timeout=DOWNLOAD_TIMEOUT_DEFAULT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Failed to fetch DoD PFAS sites: %s", e)
            break

        data = resp.json()
        if "error" in data:
            logger.warning("ArcGIS error: %s", data["error"].get("message", ""))
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        if len(features) < page_size:
            break
        offset += page_size

    if not all_features:
        logger.warning("No DoD PFAS sites retrieved")
        pd.DataFrame(
            columns=[
                "site_name",
                "agency",
                "state",
                "latitude",
                "longitude",
                "pfas_presence",
                "property_type",
                "cleanup_status",
                "is_dod",
            ]
        ).to_csv(dest, index=False)
        return dest

    rows = [f["attributes"] for f in all_features]
    df = pd.DataFrame(rows)

    result = pd.DataFrame()
    result["site_name"] = df["F_Site_Name"].astype(str).str.strip()
    result["agency"] = df["Federal_Agency"].astype(str).str.strip()
    result["state"] = df["State"].astype(str).str.strip()
    result["latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    result["longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    result["pfas_presence"] = df["PFAS_Presence"].astype(str).str.strip()
    result["property_type"] = df.get("Property_Type", "").astype(str).str.strip()  # type: ignore[union-attr]
    result["cleanup_status"] = df.get("DoD_Reported_Cleanup_Status", "").astype(str).str.strip()  # type: ignore[union-attr]
    result["is_dod"] = result["agency"].isin(_DOD_AGENCIES)

    raw_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest, index=False)
    logger.info("Saved %d DoD/federal PFAS sites to %s", len(result), dest)
    return dest


def load_dod_pfas(raw_dir: Path, *, dod_only: bool = True) -> pd.DataFrame:
    """Load DoD / federal PFAS site data.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the DoD PFAS CSV.
    dod_only : bool
        If True (default), return only DoD/military agencies.
        If False, return all federal sites including DOE, NASA, etc.

    Returns
    -------
    pd.DataFrame
        DoD PFAS site records with columns: ``site_name``, ``agency``,
        ``state``, ``latitude``, ``longitude``, ``pfas_presence``,
        ``property_type``, ``cleanup_status``, ``is_dod``.
    """
    config = load_data_config()["dod_pfas"]
    expected_file: str = config["expected_files"][0]
    filepath = raw_dir / expected_file

    if not filepath.exists():
        raise FileNotFoundError(
            f"DoD PFAS data not found: {filepath}. Run download_dod_pfas() first."
        )

    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Loaded %d federal PFAS site records", len(df))

    # Filter to CONUS bounding box
    valid_coords = df["latitude"].notna() & df["longitude"].notna()
    conus_mask = (
        valid_coords
        & (df["latitude"] >= CONUS_LAT_MIN)
        & (df["latitude"] <= CONUS_LAT_MAX)
        & (df["longitude"] >= CONUS_LON_MIN)
        & (df["longitude"] <= CONUS_LON_MAX)
    )
    n_outside = int(valid_coords.sum() - conus_mask.sum())
    if n_outside > 0:
        logger.info("Dropping %d sites outside CONUS", n_outside)
    df = df[conus_mask].copy()

    if dod_only:
        n_before = len(df)
        df = df[df["is_dod"]].copy()
        logger.info("Filtered to %d DoD/military sites (from %d CONUS)", len(df), n_before)

    return df  # type: ignore[no-any-return]
