"""Data cleaning pipeline — deduplication, unit harmonization, and merging.

Provides functions for combining multiple water quality DataFrames into
a single clean dataset ready for feature engineering and modeling.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from aquacontam._constants import MGL_TO_UGL, PPT_TO_UGL

logger = logging.getLogger(__name__)


def deduplicate_samples(
    df: pd.DataFrame,
    keys: tuple[str, ...] = ("pwsid", "analyte", "sample_date"),
    strategy: str = "max",
) -> pd.DataFrame:
    """Deduplicate samples keeping one row per key combination.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame with ``concentration`` column.
    keys : tuple[str, ...]
        Columns defining a unique sample.
    strategy : str
        How to resolve duplicates:
        - ``"max"``: keep the row with highest concentration (conservative
          for risk prediction).
        - ``"mean"``: replace concentration with mean of duplicates.
        - ``"first"``: keep the first occurrence.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.

    Raises
    ------
    ValueError
        If strategy is not recognized or required columns are missing.
    """
    valid_strategies = ("max", "mean", "first")
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy!r}")

    missing_keys = [k for k in keys if k not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns: {missing_keys}")

    if "concentration" not in df.columns:
        raise ValueError("DataFrame must contain a 'concentration' column")

    n_before = len(df)
    n_dups = int(df.duplicated(subset=list(keys)).sum())

    if n_dups == 0:
        logger.info("No duplicates found")
        return df

    if strategy == "max":
        idx = df.groupby(list(keys), sort=False)["concentration"].idxmax()
        result = df.loc[idx].reset_index(drop=True)
    elif strategy == "mean":
        # Keep first row's metadata, replace concentration with mean
        grouped = df.groupby(list(keys), sort=False)
        result = grouped.first().reset_index()
        means = grouped["concentration"].mean()
        result = result.set_index(list(keys))
        result["concentration"] = means
        result = result.reset_index()
    else:  # first
        result = df.drop_duplicates(subset=list(keys), keep="first").reset_index(drop=True)

    logger.info(
        "Deduplicated %d → %d rows (%d duplicates removed, strategy=%s)",
        n_before,
        len(result),
        n_before - len(result),
        strategy,
    )
    return cast(pd.DataFrame, result)


def harmonize_units(
    df: pd.DataFrame,
    target_unit: str = "ug/L",
) -> pd.DataFrame:
    """Convert all concentration values to a common unit.

    Handles PPT (ng/L), mg/L, and ug/L conversions. Adds an
    ``original_unit`` column preserving the pre-conversion unit.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``concentration``, ``unit``, and ``detection_limit``.
    target_unit : str
        Target unit (default ``"ug/L"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with concentrations and detection limits converted
        to ``target_unit``.
    """
    df = df.copy()

    if "concentration" not in df.columns:
        raise ValueError("DataFrame must contain a 'concentration' column")
    if "unit" not in df.columns:
        raise ValueError("DataFrame must contain a 'unit' column")

    df["original_unit"] = df["unit"].copy()
    unit_upper = df["unit"].str.strip().str.upper()

    # PPT / ng/L → ug/L
    ppt_mask = unit_upper.isin(["PPT", "NG/L"])
    if ppt_mask.any():
        df.loc[ppt_mask, "concentration"] = df.loc[ppt_mask, "concentration"] * PPT_TO_UGL
        if "detection_limit" in df.columns:
            dl_ppt = ppt_mask & df["detection_limit"].notna()
            df.loc[dl_ppt, "detection_limit"] = df.loc[dl_ppt, "detection_limit"] * PPT_TO_UGL
        n_ppt = int(ppt_mask.sum())
        logger.info("Converted %d rows from PPT/ng/L → ug/L", n_ppt)

    # mg/L → ug/L
    mg_mask = unit_upper.isin(["MG/L"])
    if mg_mask.any():
        df.loc[mg_mask, "concentration"] = df.loc[mg_mask, "concentration"] * MGL_TO_UGL
        if "detection_limit" in df.columns:
            dl_mg = mg_mask & df["detection_limit"].notna()
            df.loc[dl_mg, "detection_limit"] = df.loc[dl_mg, "detection_limit"] * MGL_TO_UGL
        n_mg = int(mg_mask.sum())
        logger.info("Converted %d rows from mg/L → ug/L", n_mg)

    # Already ug/L — no conversion needed
    ugl_mask = unit_upper.isin(["UG/L", "µG/L", "PPB"])
    n_already = int(ugl_mask.sum())
    if n_already:
        logger.debug("%d rows already in ug/L", n_already)

    # Unknown units — warn but don't convert
    known = ppt_mask | mg_mask | ugl_mask
    unknown_mask = ~known
    n_unknown = int(unknown_mask.sum())
    if n_unknown:
        unknown_units = df.loc[unknown_mask, "unit"].unique().tolist()
        logger.warning(
            "%d rows with unrecognized unit(s): %s (not converted)", n_unknown, unknown_units
        )

    df.loc[known, "unit"] = target_unit
    return df


def merge_datasets(
    *datasets: pd.DataFrame,
    source_column: str = "source",
) -> pd.DataFrame:
    """Concatenate multiple water quality DataFrames with source tracking.

    Each DataFrame must already have a ``source`` column (or one will be
    generated from position index). Missing coordinates are flagged with
    a ``has_native_coordinates`` boolean column.

    Parameters
    ----------
    *datasets : pd.DataFrame
        One or more DataFrames to merge.
    source_column : str
        Name of the source provenance column.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with ``source_column`` and
        ``has_native_coordinates`` columns.
    """
    if not datasets:
        raise ValueError("At least one dataset is required")

    parts = []
    for i, df in enumerate(datasets):
        df = df.copy()
        if source_column not in df.columns:
            df[source_column] = f"dataset_{i}"
        parts.append(df)

    result = pd.concat(parts, ignore_index=True)

    # Flag rows with valid coordinates
    if "latitude" in result.columns and "longitude" in result.columns:
        result["has_native_coordinates"] = result["latitude"].notna() & result["longitude"].notna()
    else:
        result["has_native_coordinates"] = False

    logger.info(
        "Merged %d datasets → %d total rows (%d with coordinates)",
        len(datasets),
        len(result),
        int(result["has_native_coordinates"].sum()),
    )

    return cast(pd.DataFrame, result)
