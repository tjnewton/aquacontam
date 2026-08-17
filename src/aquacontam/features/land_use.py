"""Tier 1 NLCD land use fraction features.

Extracts land cover class fractions from NLCD 30m raster data within
configurable buffers around water system locations. Produces features
like ``pct_developed_5km``, ``pct_agriculture_5km``, etc.

Typical usage::

    from aquacontam.features.land_use import extract_land_use_features

    features = extract_land_use_features(systems_gdf, Path("data/raw/nlcd_2021.tif"))
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from aquacontam._config import load_data_config
from aquacontam._constants import (
    DEFAULT_LAND_USE_BUFFERS,
    DOWNLOAD_TIMEOUT_LARGE,
    NLCD_CLASS_MAP,
)
from aquacontam.data._download import download_and_extract
from aquacontam.geo.raster import categorical_fractions

logger = logging.getLogger(__name__)

# Subset of NLCD_CLASS_MAP for the "developed_high" (class 24 only) feature
_HIGH_INTENSITY_MAP: dict[int, str] = {24: "urban_high"}


def download_nlcd(raw_dir: Path, year: int = 2021, *, force: bool = False) -> Path:
    """Download NLCD land cover GeoTIFF from MRLC.

    Parameters
    ----------
    raw_dir : Path
        Directory to save the downloaded file.
    year : int
        NLCD dataset year. Currently ignored — the URL is read from
        ``configs/data.yaml`` which points to NLCD 2021.
    force : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the downloaded GeoTIFF file.
    """
    if year != 2021:
        warnings.warn(
            f"The 'year' parameter is currently ignored; the NLCD URL in "
            f"configs/data.yaml always points to 2021. Got year={year}.",
            DeprecationWarning,
            stacklevel=2,
        )
    config = load_data_config()["nlcd"]

    if config.get("unavailable"):
        # Search for manually downloaded files before raising.
        # Use rglob to find files in subdirectories (e.g.
        # Annual_NLCD_LndCov_2021_CU_C1V1/Annual_NLCD_LndCov_2021_CU_C1V1.tif).
        search_patterns = [
            "*NLCD*LndCov*.[ti][im][fg]",
            "nlcd_2021*.[ti][im][fg]",
            "*nlcd*2021*.[ti][im][fg]",
        ]
        for pattern in search_patterns:
            # Skip AppleDouble/resource-fork junk (``._*``) that can shadow the
            # real raster on macOS/network-shared filesystems.
            matches = [
                p for p in raw_dir.rglob(pattern) if p.is_file() and not p.name.startswith("._")
            ]
            if matches:
                logger.info("Found manually downloaded NLCD file: %s", matches[0])
                return matches[0]

        reason = config.get("unavailable_reason", "no reason given")
        raise RuntimeError(
            f"NLCD data source is marked unavailable: {reason}\n"
            f"Manual download: visit https://www.sciencebase.gov/catalog/item/"
            f"655ceb8ad34ee4b6e05cc51a and place the file in {raw_dir}"
        )

    url: str = config["url"]
    expected_files: list[str] = config["expected_files"]

    paths = download_and_extract(
        url,
        raw_dir,
        expected_files,
        force=force,
        timeout=DOWNLOAD_TIMEOUT_LARGE,
        progress_desc="NLCD download",
    )
    logger.info("NLCD saved to %s", paths[0])
    return paths[0]


def compute_land_use_fractions(
    systems: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    buffer_m: float = 5000.0,
) -> pd.DataFrame:
    """Compute land cover category fractions within a buffer.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations with ``pwsid`` column.
    nlcd_path : Path or str
        Path to NLCD GeoTIFF raster.
    buffer_m : float
        Buffer radius in meters.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with columns for each land cover category
        fraction (e.g. ``developed``, ``forest``, ``agriculture``).
    """
    fracs = categorical_fractions(systems, nlcd_path, buffer_m, NLCD_CLASS_MAP)
    suffix = f"_{int(buffer_m / 1000)}km"
    result: pd.DataFrame = fracs.rename(columns={c: f"pct_{c}{suffix}" for c in fracs.columns})
    return result


def _compute_high_intensity_fraction(
    systems: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    buffer_m: float,
) -> pd.DataFrame:
    """Compute fraction of high-intensity urban (NLCD class 24) only."""
    fracs = categorical_fractions(systems, nlcd_path, buffer_m, _HIGH_INTENSITY_MAP)
    suffix = f"_{int(buffer_m / 1000)}km"
    result: pd.DataFrame = fracs.rename(columns={c: f"pct_{c}{suffix}" for c in fracs.columns})
    return result


def _compute_majority_class(
    systems: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    buffer_m: float = 5000.0,
) -> pd.Series:
    """Determine the majority NLCD class within the buffer."""
    fracs = categorical_fractions(systems, nlcd_path, buffer_m, NLCD_CLASS_MAP)
    majority = fracs.idxmax(axis=1)
    majority.name = "nlcd_majority_class"
    return majority


def extract_land_use_features(
    systems: gpd.GeoDataFrame,
    nlcd_path: Path | str,
    buffers_m: tuple[float, ...] = DEFAULT_LAND_USE_BUFFERS,
) -> pd.DataFrame:
    """Extract all land use features at multiple buffer radii.

    Parameters
    ----------
    systems : GeoDataFrame
        Water system point locations with ``pwsid`` column.
    nlcd_path : Path or str
        Path to NLCD GeoTIFF raster.
    buffers_m : tuple[float, ...]
        Buffer radii in meters to compute fractions at.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with land use feature columns including:
        - ``pct_developed_{r}km`` for each radius
        - ``pct_agriculture_{r}km``, ``pct_forest_{r}km``, etc.
        - ``pct_urban_high_1km`` (class 24 only)
        - ``nlcd_majority_class``
    """
    all_features: list[pd.DataFrame | pd.Series] = []

    for buf in buffers_m:
        fracs = compute_land_use_fractions(systems, nlcd_path, buf)
        all_features.append(fracs)

    # High-intensity urban at 1km
    if 1000.0 in buffers_m:
        high_int = _compute_high_intensity_fraction(systems, nlcd_path, 1000.0)
        all_features.append(high_int)

    # Majority class at largest buffer
    majority = _compute_majority_class(systems, nlcd_path, max(buffers_m))
    all_features.append(majority)

    result = pd.concat(all_features, axis=1)
    result.index.name = "pwsid"

    logger.info("Extracted %d land use features", len(result.columns))
    return result
