"""Shared utilities for UCMR3 and UCMR5 data ingestion.

Both UCMR datasets use the same tab-delimited format from EPA. This
module provides the common download and parsing logic so that the
individual ``UCMR3Source`` and ``UCMR5Source`` classes stay thin.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aquacontam.data._download import download_and_extract

logger = logging.getLogger(__name__)


def download_and_extract_ucmr_zip(
    url: str,
    raw_dir: Path,
    expected_files: list[str],
    *,
    force: bool = False,
    chunk_size: int = 8192,
) -> list[Path]:
    """Download a UCMR ZIP archive from EPA and extract it.

    Parameters
    ----------
    url : str
        URL of the ZIP file.
    raw_dir : Path
        Directory to extract files into.
    expected_files : list[str]
        Filenames expected inside the ZIP.
    force : bool
        If ``True``, re-download even if files already exist.
    chunk_size : int
        Deprecated — ignored. Kept for backward compatibility.

    Returns
    -------
    list[Path]
        Paths to the extracted files.
    """
    if chunk_size != 8192:
        warnings.warn(
            "The 'chunk_size' parameter is deprecated and ignored. "
            "Download chunking is handled by the download_and_extract utility.",
            DeprecationWarning,
            stacklevel=2,
        )
    return download_and_extract(
        url,
        raw_dir,
        expected_files,
        force=force,
        progress_desc="UCMR download",
    )


def parse_ucmr_txt(
    filepath: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Parse a UCMR tab-delimited occurrence file into standardized columns.

    Parameters
    ----------
    filepath : Path
        Path to the ``.txt`` file (e.g. ``UCMR5_All.txt``).
    config : dict[str, Any]
        The source-specific config section (from ``configs/data.yaml``).
        Must contain ``format``, ``column_map``, ``derived_columns``,
        and ``extra_columns``.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard schema columns plus ``raw_*`` extras.
    """
    fmt = config["format"]
    column_map: dict[str, str] = config["column_map"]
    derived = config["derived_columns"]
    extra_cols: list[str] = config.get("extra_columns", [])

    logger.info("Reading %s …", filepath)
    df = pd.read_csv(
        filepath,
        sep=fmt["delimiter"],
        encoding=fmt["encoding"],
        dtype=fmt.get("dtype_overrides", {}),
        low_memory=False,
    )
    logger.info("Read %d rows, %d columns", len(df), len(df.columns))

    # --- Derive censored boolean from sign column ---
    censored_cfg = derived["censored"]
    sign_col = censored_cfg["source_column"]
    condition = censored_cfg["condition"]
    if sign_col in df.columns:
        df["censored"] = df[sign_col].fillna("").str.strip() == condition
    else:
        logger.warning("Sign column '%s' not found; defaulting censored=False", sign_col)
        df["censored"] = False

    # --- Rename mapped columns ---
    rename = {raw: std for raw, std in column_map.items() if raw in df.columns}
    df = df.rename(columns=rename)

    # --- Set unit ---
    df["unit"] = derived["unit"]

    # --- Parse dates ---
    if "sample_date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")

    # --- Coerce numeric columns ---
    for col in ("concentration", "detection_limit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # For censored rows with NaN concentration, fill with 0
    if "concentration" in df.columns:
        df.loc[df["censored"] & df["concentration"].isna(), "concentration"] = 0.0

    # --- Set lat/lon as NaN (geocoded downstream via ZCTA centroids) ---
    df["latitude"] = float("nan")
    df["longitude"] = float("nan")

    # --- Prefix extra columns with raw_ ---
    # Map original extra column names to their potentially-renamed versions
    for orig_col in extra_cols:
        if orig_col in df.columns:
            df = df.rename(columns={orig_col: f"raw_{orig_col}"})

    # --- Select final columns ---
    standard_cols = [
        "pwsid",
        "analyte",
        "concentration",
        "unit",
        "censored",
        "detection_limit",
        "sample_date",
        "latitude",
        "longitude",
    ]
    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    keep = [c for c in standard_cols if c in df.columns] + raw_cols
    df = df[keep]

    # Deterministic row ordering for reproducibility across filesystems
    sort_cols = [c for c in ["pwsid", "analyte", "sample_date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return cast(pd.DataFrame, df)
