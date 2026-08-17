"""ZIP code centroid geocoding via US Census ZCTA gazetteer.

Maps ZIP codes to approximate coordinates (~5 km accuracy) using
Census ZCTA (ZIP Code Tabulation Area) centroid locations.

Typical usage::

    from aquacontam.geo.geocoding import load_zcta_centroids, geocode_by_zipcode

    centroids = load_zcta_centroids(Path("data/raw"))
    df = geocode_by_zipcode(df, centroids, zip_col="raw_ZipCode")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import DOWNLOAD_TIMEOUT_DEFAULT
from aquacontam.data._download import download_and_extract

logger = logging.getLogger(__name__)


def download_zcta_gazetteer(raw_dir: Path, *, force: bool = False) -> Path:
    """Download the Census ZCTA gazetteer file.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the downloaded file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the extracted gazetteer text file.
    """
    config = load_data_config()["zcta_gazetteer"]
    paths = download_and_extract(
        config["url"],
        raw_dir,
        config["expected_files"],
        force=force,
        timeout=DOWNLOAD_TIMEOUT_DEFAULT,
        progress_desc="ZCTA gazetteer",
    )
    return paths[0]


def load_zcta_centroids(raw_dir: Path) -> pd.DataFrame:
    """Load ZCTA centroid lookup table.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the gazetteer text file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``zipcode``, ``latitude``, ``longitude``
        indexed by ``zipcode`` (string, zero-padded to 5 digits).
    """
    config = load_data_config()["zcta_gazetteer"]
    filepath = raw_dir / config["expected_files"][0]

    if not filepath.exists():
        filepath = download_zcta_gazetteer(raw_dir)

    fmt = config.get("format", {})
    col_map = config.get("column_map", {})

    df = pd.read_csv(
        filepath,
        sep=fmt.get("delimiter", "\t"),
        encoding=fmt.get("encoding", "utf-8"),
        dtype={"GEOID": str},
    )

    # Strip whitespace from column names (Census gazetteer has trailing spaces)
    df.columns = df.columns.str.strip()

    # Rename columns
    rename = {raw: std for raw, std in col_map.items() if raw in df.columns}
    df = df.rename(columns=rename)

    # Ensure zipcode is zero-padded 5 digits
    df["zipcode"] = df["zipcode"].str.strip().str.zfill(5)

    # Keep only needed columns
    df = df[["zipcode", "latitude", "longitude"]].copy()
    df = df.set_index("zipcode")

    logger.info("Loaded %d ZCTA centroids", len(df))
    return df  # type: ignore[no-any-return]


def geocode_by_zipcode(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    zip_col: str = "raw_ZipCode",
) -> pd.DataFrame:
    """Add latitude/longitude from ZIP code centroid lookup.

    Fills in ``latitude`` and ``longitude`` columns for rows that
    currently have NaN coordinates, using ZCTA centroid lookup.
    Existing non-NaN coordinates are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame. Must have a column named *zip_col*
        containing ZIP code strings.
    centroids : pd.DataFrame
        ZCTA centroids indexed by zipcode (from ``load_zcta_centroids``).
    zip_col : str
        Name of the ZIP code column in *df*.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``latitude``, ``longitude`` filled where possible.
        Adds ``coord_source`` column: ``"zip_centroid"`` for geocoded,
        ``"original"`` for pre-existing coordinates.
    """
    if zip_col not in df.columns:
        logger.warning("ZIP code column %r not found; skipping geocoding", zip_col)
        return df

    result = df.copy()

    # Normalize ZIP codes to 5-digit strings
    zips = result[zip_col].astype(str).str.strip().str.split("-").str[0].str.zfill(5)
    # Some ZIP codes may be "nan" or empty after conversion
    zips = zips.replace({"nan": "", "00nan": "", "00000": ""})

    # Track which rows already have coordinates
    has_lat = (
        result["latitude"].notna()
        if "latitude" in result.columns
        else pd.Series(False, index=result.index)
    )
    has_lon = (
        result["longitude"].notna()
        if "longitude" in result.columns
        else pd.Series(False, index=result.index)
    )
    has_coords = has_lat & has_lon

    # Initialize coord_source
    if "coord_source" not in result.columns:
        result["coord_source"] = ""
    result.loc[has_coords, "coord_source"] = "original"

    # Look up centroids for rows missing coordinates
    needs_geocoding = ~has_coords & (zips.str.len() == 5) & (zips != "")

    if needs_geocoding.any():
        lookup = zips[needs_geocoding].map(centroids["latitude"])
        lookup_lon = zips[needs_geocoding].map(centroids["longitude"])

        # Ensure latitude/longitude columns exist
        if "latitude" not in result.columns:
            result["latitude"] = float("nan")
        if "longitude" not in result.columns:
            result["longitude"] = float("nan")

        matched = lookup.notna()
        result.loc[needs_geocoding & matched, "latitude"] = lookup[matched].to_numpy()
        result.loc[needs_geocoding & matched, "longitude"] = lookup_lon[matched].to_numpy()
        result.loc[needs_geocoding & matched, "coord_source"] = "zip_centroid"

        n_geocoded = int(matched.sum())
        n_failed = int((~matched).sum())
        logger.info(
            "Geocoded %d rows via ZIP centroid (%d unmatched)",
            n_geocoded,
            n_failed,
        )
    else:
        logger.info("No rows need ZIP centroid geocoding")

    return result  # type: ignore[no-any-return]


def _haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Vectorized haversine distance in kilometres.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : pd.Series
        Coordinates in decimal degrees (WGS84).

    Returns
    -------
    pd.Series
        Great-circle distance in km.
    """
    import numpy as np

    r = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    result: pd.Series[Any] = 2 * r * np.arcsin(np.sqrt(a))
    return result


def compute_coord_source_stats(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    zip_col: str = "raw_ZipCode",
    pwsid_col: str = "pwsid",
) -> dict:
    """Compute breakdown of coordinate sources and centroid imprecision.

    Parameters
    ----------
    df : pd.DataFrame
        Geocoded DataFrame with ``coord_source``, ``latitude``, ``longitude``,
        and a ZIP code column.
    centroids : pd.DataFrame
        ZCTA centroids indexed by ZIP string with ``INTPTLAT`` and
        ``INTPTLONG`` columns.
    zip_col : str
        Column name for ZIP codes in *df*.
    pwsid_col : str
        Column name for system identifiers.

    Returns
    -------
    dict
        Keys: ``system_counts`` (per coord_source), ``row_counts``,
        ``centroid_imprecision_km`` (distance stats for original-coord systems).
    """
    import numpy as np

    result: dict = {"system_counts": {}, "row_counts": {}, "centroid_imprecision_km": {}}

    if "coord_source" not in df.columns:
        return result

    # --- Per-system and per-row counts ---
    for source in ("original", "zip_centroid"):
        mask = df["coord_source"] == source
        result["row_counts"][source] = int(mask.sum())
        if pwsid_col in df.columns:
            result["system_counts"][source] = int(df.loc[mask, pwsid_col].nunique())

    missing_mask = df["coord_source"].isna() | (df["coord_source"] == "")
    result["row_counts"]["missing"] = int(missing_mask.sum())
    if pwsid_col in df.columns:
        result["system_counts"]["missing"] = int(df.loc[missing_mask, pwsid_col].nunique())

    count_keys = [k for k in result["system_counts"] if not k.endswith("_pct")]
    total_systems = sum(result["system_counts"][k] for k in count_keys)
    if total_systems > 0:
        for source in count_keys:
            result["system_counts"][f"{source}_pct"] = round(
                result["system_counts"][source] / total_systems * 100, 1
            )

    # --- Centroid imprecision for systems with original coords ---
    orig_mask = df["coord_source"] == "original"
    has_zip = df[zip_col].notna() & (df[zip_col] != "")
    eligible = df.loc[orig_mask & has_zip].copy()

    if len(eligible) > 0 and len(centroids) > 0:
        zips = eligible[zip_col].astype(str).str.strip().str.split("-").str[0].str.zfill(5)
        # Centroids may have "INTPTLAT"/"INTPTLONG" (raw) or "latitude"/"longitude" (loaded)
        lat_col = "INTPTLAT" if "INTPTLAT" in centroids.columns else "latitude"
        lon_col = "INTPTLONG" if "INTPTLONG" in centroids.columns else "longitude"
        lat_centroid = zips.map(centroids[lat_col])
        lon_centroid = zips.map(centroids[lon_col])

        valid = lat_centroid.notna() & lon_centroid.notna()
        if valid.sum() > 0:
            distances = _haversine_km(
                eligible.loc[valid, "latitude"].astype(float),
                eligible.loc[valid, "longitude"].astype(float),
                lat_centroid[valid].astype(float),
                lon_centroid[valid].astype(float),
            )
            result["centroid_imprecision_km"] = {
                "n_systems": int(valid.sum()),
                "median": round(float(np.nanmedian(distances)), 2),
                "mean": round(float(np.nanmean(distances)), 2),
                "p25": round(float(np.nanpercentile(distances, 25)), 2),
                "p75": round(float(np.nanpercentile(distances, 75)), 2),
                "p95": round(float(np.nanpercentile(distances, 95)), 2),
                "max": round(float(np.nanmax(distances)), 2),
            }

    return result
