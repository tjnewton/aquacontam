"""USGS National Groundwater Aggregation (NGA) arsenic data loader.

Loads measured arsenic concentrations in U.S. groundwater wells from the USGS
data release *"Arsenic, manganese, and pH groundwater quality data, selected
well construction characteristics, and aquifer assignments for wells in the
conterminous U.S."* (Erickson, Hill & Wilson, 2020; DOI 10.5066/P9JMUAPY;
CC0 public domain).

This source backs benchmark task **T6** (private-well risk extrapolation). It is
*not* a :class:`DataSource` subclass — like other well-level sources it exposes
``download_*``/``load_*`` free functions. Each row is one well sample, keyed by
a USGS ``SITE_ID`` rather than a 9-character PWSID.

Why this source: the NJ Private Well Testing Act bulk data is confidential by
statute and unrecoverable, and public PWTA summaries are aggregated (no per-well
points). The NGA release provides *measured*, point-level arsenic with a
``WATER_USE`` code that distinguishes **domestic** (private) wells from
**public supply** wells — enabling a construct-valid, openly reproducible
public-supply → domestic extrapolation. Arsenic is geogenic (aquifer-driven),
so non-domestic observation wells are not contamination-biased the way a
point-source contaminant (e.g. PFAS) would be.

Typical usage::

    from aquacontam.data.nga_arsenic import download_nga_arsenic, load_nga_arsenic

    download_nga_arsenic(Path("data/raw"))
    domestic = load_nga_arsenic(Path("data/raw"), water_use="Domestic")
    public = load_nga_arsenic(Path("data/raw"), water_use="Public supply")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    CRS_DISTANCE,
    CRS_STORAGE,
)

logger = logging.getLogger(__name__)

# ScienceBase file-download URLs for the two CSVs of item 5f36baa782cee144fb3873e8
# (recovered from the ScienceBase manifest). Content-addressed, stable paths.
_SB_ITEM = "5f36baa782cee144fb3873e8"
_NGA_FILES: dict[str, str] = {
    "Water_Chemistry.csv": (
        "https://www.sciencebase.gov/catalog/file/get/5f36baa782cee144fb3873e8"
        "?f=__disk__c6%2F03%2Fdb%2Fc603db7bd8ca1d5698e6aa2fe0a09a709e9e612e"
    ),
    "Site_Information.csv": (
        "https://www.sciencebase.gov/catalog/file/get/5f36baa782cee144fb3873e8"
        "?f=__disk__43%2Fb0%2Fe6%2F43b0e621b57328ec30ee9bc564b4f715780be8f5"
    ),
}
# Expected byte sizes from the manifest — used to detect truncated downloads.
_NGA_EXPECTED_BYTES: dict[str, int] = {
    "Water_Chemistry.csv": 24806534,
    "Site_Information.csv": 10274735,
}

# Censoring remark codes (As_Remark) that indicate a left-censored (non-detect)
# result. "E" (estimated) and "V" (blank-contaminated) are treated as detected.
_CENSOR_REMARKS = {"<", "<?"}

# Arsenic EPA Maximum Contaminant Level (µg/L) — the exceedance threshold.
ARSENIC_MCL_UGL = 10.0


def download_nga_arsenic(raw_dir: Path, *, force: bool = False) -> list[Path]:
    """Download the NGA arsenic CSVs from ScienceBase.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the downloaded files (a ``nga_arsenic`` subdirectory
        is used).
    force : bool
        Re-download even if the files already exist.

    Returns
    -------
    list[Path]
        Paths to ``Site_Information.csv`` and ``Water_Chemistry.csv``.

    Notes
    -----
    ScienceBase serves these files only through its own proxy (no public S3
    URL). If the host is unreachable (it has intermittent outages) this raises
    ``RuntimeError``; the files are CC0 and can be fetched manually from
    https://doi.org/10.5066/P9JMUAPY and placed in ``raw_dir/nga_arsenic/``.
    """
    import requests

    dest_dir = raw_dir / "nga_arsenic"
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for name, url in _NGA_FILES.items():
        dest = dest_dir / name
        expected = _NGA_EXPECTED_BYTES[name]
        if not force and dest.exists() and dest.stat().st_size == expected:
            paths.append(dest)
            continue
        logger.info("Downloading NGA %s from ScienceBase …", name)
        try:
            resp = requests.get(
                url, timeout=900, stream=True, headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"NGA arsenic download failed for {name} ({exc}). ScienceBase "
                "may be down; fetch manually from https://doi.org/10.5066/P9JMUAPY "
                f"into {dest_dir}."
            ) from exc
        if dest.stat().st_size != expected:
            raise RuntimeError(
                f"NGA {name} download size {dest.stat().st_size} != expected {expected}"
            )
        paths.append(dest)

    # Return in a stable order: site info first, then chemistry.
    return [dest_dir / "Site_Information.csv", dest_dir / "Water_Chemistry.csv"]


def load_nga_arsenic(
    raw_dir: Path,
    *,
    water_use: str | None = None,
) -> gpd.GeoDataFrame:
    """Load NGA arsenic samples into the standard water-quality schema.

    Joins ``Water_Chemistry.csv`` (one row per sample) to
    ``Site_Information.csv`` (one row per well) on ``SITE_ID``, reprojects the
    EPSG:5070 well coordinates to EPSG:4326, and emits one row per arsenic
    sample in the project's long-form schema.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the ``nga_arsenic`` subdirectory.
    water_use : str | None
        If given, keep only wells whose ``WATER_USE`` matches this value
        (case-insensitive), e.g. ``"Domestic"`` or ``"Public supply"``.

    Returns
    -------
    GeoDataFrame
        Arsenic samples with columns ``pwsid`` (= NGA ``SITE_ID``), ``analyte``
        (``"arsenic"``), ``concentration`` (µg/L; 0 for non-detects),
        ``unit``, ``censored``, ``detection_limit``, ``sample_date``,
        ``latitude``, ``longitude``, plus ``water_use``, ``data_source``,
        ``well_depth_ft`` and ``aquifer``. CRS is EPSG:4326.
    """
    dest_dir = raw_dir / "nga_arsenic"
    site_path = dest_dir / "Site_Information.csv"
    chem_path = dest_dir / "Water_Chemistry.csv"
    for p in (site_path, chem_path):
        if not p.exists():
            raise FileNotFoundError(f"NGA file not found: {p}. Run download_nga_arsenic() first.")

    logger.info("Reading NGA site information from %s …", site_path)
    site = pd.read_csv(site_path, dtype=str, low_memory=False)
    logger.info("Reading NGA water chemistry from %s …", chem_path)
    chem = pd.read_csv(chem_path, dtype=str, low_memory=False)

    # --- Sites: reproject EPSG:5070 → EPSG:4326 (once, on unique wells) ---
    site = site.copy()
    site["XCoord"] = pd.to_numeric(site["XCoord"], errors="coerce")
    site["YCoord"] = pd.to_numeric(site["YCoord"], errors="coerce")
    site = site.dropna(subset=["XCoord", "YCoord"])
    site_pts = gpd.GeoDataFrame(
        site,
        geometry=gpd.points_from_xy(site["XCoord"], site["YCoord"]),
        crs=CRS_DISTANCE,  # EPSG:5070 (NGA stores Albers Equal Area)
    ).to_crs(CRS_STORAGE)
    site["longitude"] = site_pts.geometry.x
    site["latitude"] = site_pts.geometry.y
    site["well_depth_ft"] = pd.to_numeric(site["Depth_Value"], errors="coerce")
    site["water_use"] = site["WATER_USE"].astype(str).str.strip()
    site["data_source"] = site["DataSource"].astype(str).str.strip()
    site = site.rename(columns={"Aquifer": "aquifer"})

    if water_use is not None:
        mask = site["water_use"].str.lower() == water_use.strip().lower()
        site = site[mask].copy()
        logger.info("Filtered to WATER_USE=%r: %d wells", water_use, len(site))

    site_cols = [
        "SITE_ID",
        "latitude",
        "longitude",
        "water_use",
        "data_source",
        "well_depth_ft",
        "aquifer",
    ]
    site_small = site[site_cols].drop_duplicates(subset=["SITE_ID"])

    # --- Chemistry: keep arsenic measurements only ---
    chem = chem.copy()
    chem["As_Value"] = pd.to_numeric(chem["As_Value"], errors="coerce")
    chem = chem.dropna(subset=["As_Value"])
    remark = chem["As_Remark"].astype(str).str.strip()
    chem["censored"] = remark.isin(_CENSOR_REMARKS)
    # For non-detects the reported value is the detection level; concentration 0.
    chem["detection_limit"] = chem["As_Value"].where(chem["censored"], other=pd.NA)
    chem["concentration"] = chem["As_Value"].where(~chem["censored"], other=0.0)
    chem["detection_limit"] = pd.to_numeric(chem["detection_limit"], errors="coerce")
    chem["sample_date"] = pd.to_datetime(chem["Sample_Date"], errors="coerce")
    chem["analyte"] = "arsenic"
    chem["unit"] = "ug/L"

    # --- Join sample → site ---
    merged = chem.merge(site_small, on="SITE_ID", how="inner")
    merged = merged.rename(columns={"SITE_ID": "pwsid"})

    # Valid CONUS coordinates only
    valid = merged["latitude"].between(CONUS_LAT_MIN, CONUS_LAT_MAX) & merged["longitude"].between(
        CONUS_LON_MIN, CONUS_LON_MAX
    )
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.info("Dropping %d NGA arsenic samples outside CONUS bbox", n_dropped)
    merged = merged[valid].copy()

    out_cols = [
        "pwsid",
        "analyte",
        "concentration",
        "unit",
        "censored",
        "detection_limit",
        "sample_date",
        "latitude",
        "longitude",
        "water_use",
        "data_source",
        "well_depth_ft",
        "aquifer",
    ]
    merged = merged[out_cols]

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["longitude"], merged["latitude"]),
        crs=CRS_STORAGE,
    )
    logger.info(
        "Loaded %d NGA arsenic samples across %d wells",
        len(gdf),
        gdf["pwsid"].nunique(),
    )
    return gdf


def nga_metadata() -> dict[str, Any]:
    """Return provenance metadata for the NGA arsenic source."""
    return {
        "source": "USGS National Groundwater Aggregation (Erickson, Hill & Wilson 2020)",
        "doi": "10.5066/P9JMUAPY",
        "sciencebase_item": _SB_ITEM,
        "license": "CC0 1.0 Universal",
        "analyte": "arsenic",
        "mcl_ugl": ARSENIC_MCL_UGL,
    }
