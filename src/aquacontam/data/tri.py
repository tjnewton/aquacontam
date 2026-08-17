"""EPA Toxics Release Inventory (TRI) — PFAS release data.

Downloads TRI Basic Data Files, filters for PFAS chemicals, and extracts
facility coordinates and release quantities. This is a feature source
(like EPA FRS), not a ``DataSource`` subclass.

Typical usage::

    from aquacontam.data.tri import download_tri_pfas, load_tri_pfas

    download_tri_pfas(Path("data/raw"))
    facilities = load_tri_pfas(Path("data/raw"))
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from aquacontam._config import load_data_config
from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    DOWNLOAD_TIMEOUT_DEFAULT,
)

logger = logging.getLogger(__name__)

# Column names in TRI Basic Data Files (numbered prefix format)
_COL_YEAR = "1. YEAR"
_COL_FACILITY = "4. FACILITY NAME"
_COL_CITY = "6. CITY"
_COL_STATE = "8. ST"
_COL_LAT = "12. LATITUDE"
_COL_LON = "13. LONGITUDE"
_COL_CHEMICAL = "37. CHEMICAL"
_COL_CAS = "40. CAS#"
_COL_PFAS = "48. PFAS"
_COL_UNIT = "50. UNIT OF MEASURE"
_COL_WATER_RELEASE = "53. 5.3 - WATER"
_COL_ONSITE_TOTAL = "65. ON-SITE RELEASE TOTAL"
_COL_POTW_TOTAL = "68. POTW - TOTAL TRANSFERS"
_COL_FRS_ID = "3. FRS ID"


def download_tri_pfas(
    raw_dir: Path,
    *,
    force: bool = False,
    years: list[int] | None = None,
) -> Path:
    """Download TRI Basic Data Files and extract PFAS facility records.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the processed CSV file.
    force : bool
        Re-download even if the file already exists.
    years : list[int] | None
        Years to download. If None, uses config defaults.

    Returns
    -------
    Path
        Path to the PFAS-filtered facility CSV.
    """
    config = load_data_config()["tri"]
    url_template: str = config["url_template"]
    expected_file: str = config["expected_files"][0]
    dest = raw_dir / expected_file

    if not force and dest.exists():
        logger.info("TRI PFAS data already exists: %s", dest)
        return dest

    if years is None:
        years = config.get("years", [2023])

    all_pfas: list[pd.DataFrame] = []
    for year in years:
        url = url_template.format(year=year)
        logger.info("Downloading TRI %d data from %s…", year, url)
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_DEFAULT, stream=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Failed to download TRI %d: %s", year, e)
            continue

        total = int(resp.headers.get("content-length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"TRI {year}") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                buf.write(chunk)
                pbar.update(len(chunk))
        buf.seek(0)

        df = pd.read_csv(buf, dtype=str, low_memory=False)
        # Filter to PFAS records
        pfas_col = _COL_PFAS if _COL_PFAS in df.columns else None
        if pfas_col is None:
            # Try case-insensitive search
            for c in df.columns:
                if "pfas" in c.lower():
                    pfas_col = c
                    break
        if pfas_col is None:
            logger.warning("No PFAS column found in TRI %d data", year)
            continue

        pfas_df = df[df[pfas_col] == "YES"].copy()
        if len(pfas_df) == 0:
            logger.info("No PFAS records in TRI %d", year)
            continue

        logger.info("Found %d PFAS records in TRI %d", len(pfas_df), year)
        all_pfas.append(pfas_df)

    if not all_pfas:
        logger.warning("No PFAS records found across all TRI years")
        # Write empty file with expected columns
        pd.DataFrame(
            columns=[
                "frs_id",
                "facility_name",
                "latitude",
                "longitude",
                "chemical",
                "year",
                "onsite_release_lb",
                "water_release_lb",
                "potw_transfer_lb",
            ]
        ).to_csv(dest, index=False)
        return dest

    combined = pd.concat(all_pfas, ignore_index=True)

    # Extract and standardize columns
    result = pd.DataFrame()
    result["frs_id"] = combined.get(_COL_FRS_ID, "").astype(str).str.strip()  # type: ignore[union-attr]
    result["facility_name"] = combined.get(_COL_FACILITY, "").astype(str).str.strip()  # type: ignore[union-attr]
    result["city"] = combined.get(_COL_CITY, "").astype(str).str.strip()  # type: ignore[union-attr]
    result["state"] = combined.get(_COL_STATE, "").astype(str).str.strip()  # type: ignore[union-attr]
    result["latitude"] = pd.to_numeric(combined.get(_COL_LAT, ""), errors="coerce")
    result["longitude"] = pd.to_numeric(combined.get(_COL_LON, ""), errors="coerce")
    result["chemical"] = combined.get(_COL_CHEMICAL, "").astype(str).str.strip()  # type: ignore[union-attr]
    result["cas_number"] = combined.get(_COL_CAS, "").astype(str).str.strip()  # type: ignore[union-attr]
    result["year"] = pd.to_numeric(combined.get(_COL_YEAR, ""), errors="coerce")
    result["onsite_release_lb"] = pd.to_numeric(
        combined.get(_COL_ONSITE_TOTAL, ""), errors="coerce"
    ).fillna(0.0)  # type: ignore[union-attr]
    result["water_release_lb"] = pd.to_numeric(
        combined.get(_COL_WATER_RELEASE, ""), errors="coerce"
    ).fillna(0.0)  # type: ignore[union-attr]
    result["potw_transfer_lb"] = pd.to_numeric(
        combined.get(_COL_POTW_TOTAL, ""), errors="coerce"
    ).fillna(0.0)  # type: ignore[union-attr]

    raw_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest, index=False)
    logger.info("Saved %d TRI PFAS records to %s", len(result), dest)
    return dest


def load_tri_pfas(raw_dir: Path) -> pd.DataFrame:
    """Load TRI PFAS facility data.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the TRI PFAS CSV.

    Returns
    -------
    pd.DataFrame
        TRI PFAS facility records with columns: ``frs_id``, ``facility_name``,
        ``latitude``, ``longitude``, ``chemical``, ``year``,
        ``onsite_release_lb``, ``water_release_lb``, ``potw_transfer_lb``.
    """
    config = load_data_config()["tri"]
    expected_file: str = config["expected_files"][0]
    filepath = raw_dir / expected_file

    if not filepath.exists():
        raise FileNotFoundError(
            f"TRI PFAS data not found: {filepath}. Run download_tri_pfas() first."
        )

    df = pd.read_csv(filepath, dtype={"frs_id": str}, low_memory=False)
    logger.info("Loaded %d TRI PFAS records", len(df))

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
        logger.info("Dropping %d TRI facilities outside CONUS", n_outside)
    df = df[conus_mask].copy()

    logger.info("Loaded %d TRI PFAS CONUS facilities", len(df))
    return df  # type: ignore[no-any-return]
