"""Schema validation for standardized water quality DataFrames.

Every ``DataSource.validate()`` delegates to the functions here. Validators
return boolean masks so the caller can decide whether to warn, drop, or raise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aquacontam._constants import (
    CONUS_LAT_MAX,
    CONUS_LAT_MIN,
    CONUS_LON_MAX,
    CONUS_LON_MIN,
    PWSID_LENGTH,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)

# Canonical dtypes for the 9 required columns.
SCHEMA_DTYPES: dict[str, str] = {
    "pwsid": "str",
    "analyte": "str",
    "concentration": "float64",
    "unit": "str",
    "censored": "bool",
    "detection_limit": "float64",
    "sample_date": "datetime64[ns]",
    "latitude": "float64",
    "longitude": "float64",
}


def _coerce_dtype(series: pd.Series, target: str) -> pd.Series:  # type: ignore[type-arg]
    """Best-effort coercion of *series* to *target* dtype."""
    if target == "str":
        return series.astype("str")  # type: ignore[call-overload]
    if target == "bool":
        return series.astype("bool")
    if target.startswith("datetime64"):
        return pd.to_datetime(series, errors="coerce")  # type: ignore[return-value]
    return series.astype(target)  # type: ignore[call-overload, no-any-return]


def validate_pwsid(series: pd.Series) -> pd.Series:
    """Return a boolean mask — ``True`` where the PWSID is valid.

    Valid means: non-null string of exactly ``PWSID_LENGTH`` characters.

    Parameters
    ----------
    series : pd.Series
        The ``pwsid`` column.

    Returns
    -------
    pd.Series
        Boolean mask (True = valid).
    """
    is_str = series.apply(lambda v: isinstance(v, str))
    correct_len = series.str.len() == PWSID_LENGTH
    return is_str & correct_len


def validate_concentrations(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask — ``True`` where concentration values are valid.

    Checks:
    - ``concentration >= 0``
    - ``detection_limit > 0``
    - For censored rows: ``concentration <= detection_limit``

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``concentration``, ``detection_limit``, and ``censored``.

    Returns
    -------
    pd.Series
        Boolean mask (True = valid).
    """
    conc_ok = df["concentration"] >= 0
    # Detection limit must be > 0 for censored rows; non-censored rows may
    # have unknown (NaN) or zero detection limits.
    dl_ok = ~df["censored"] | (df["detection_limit"] > 0)
    censored_ok = ~df["censored"] | (df["concentration"] <= df["detection_limit"])
    return conc_ok & dl_ok & censored_ok


def validate_coordinates(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask — ``True`` where lat/lon are within CONUS.

    NaN coordinates are considered *invalid* by this check. Callers that
    expect missing coordinates (e.g. UCMR raw data) should use
    ``skip_coord_check=True`` in :func:`validate_schema`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``latitude`` and ``longitude``.

    Returns
    -------
    pd.Series
        Boolean mask (True = valid).
    """
    lat = df["latitude"]
    lon = df["longitude"]
    lat_ok = (lat >= CONUS_LAT_MIN) & (lat <= CONUS_LAT_MAX)
    lon_ok = (lon >= CONUS_LON_MIN) & (lon <= CONUS_LON_MAX)
    return lat_ok & lon_ok


def validate_schema(
    df: pd.DataFrame,
    *,
    require_all_columns: bool = True,
    coerce_dtypes: bool = True,
    drop_invalid_rows: bool = False,
    skip_coord_check: bool = False,
    skip_pwsid_check: bool = False,
) -> pd.DataFrame:
    """Validate and optionally coerce a water quality DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    require_all_columns : bool
        If ``True`` (default), raise ``ValueError`` when required columns
        are missing.
    coerce_dtypes : bool
        If ``True`` (default), attempt to coerce columns to canonical dtypes.
    drop_invalid_rows : bool
        If ``True``, silently drop rows that fail row-level checks.
        Otherwise log warnings but keep all rows.
    skip_coord_check : bool
        If ``True``, skip coordinate validation (useful for datasets
        without lat/lon, like raw UCMR data).
    skip_pwsid_check : bool
        If ``True``, skip PWSID format validation (useful for data
        sources like WQP where identifiers are not public water system IDs).

    Returns
    -------
    pd.DataFrame
        The (possibly coerced/filtered) DataFrame.

    Raises
    ------
    ValueError
        If ``require_all_columns`` is ``True`` and columns are missing.
    """
    # --- Column presence ---
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        if require_all_columns:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        logger.warning("Missing columns (non-fatal): %s", sorted(missing))

    # --- Empty DataFrame shortcut (after column check) ---
    if len(df) == 0:
        return df.copy()  # type: ignore[no-any-return]

    # --- Dtype coercion ---
    if coerce_dtypes:
        for col, dtype in SCHEMA_DTYPES.items():
            if col in df.columns:
                try:
                    df[col] = _coerce_dtype(df[col], dtype)
                except (ValueError, TypeError) as exc:
                    logger.warning("Could not coerce column '%s' to %s: %s", col, dtype, exc)

    # --- Row-level checks ---
    mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)

    # Track per-validator drop counts for detailed reporting
    validator_drops: dict[str, int] = {}

    if "pwsid" in df.columns and not skip_pwsid_check:
        pwsid_ok = validate_pwsid(df["pwsid"])
        n_bad = int((~pwsid_ok).sum())
        if n_bad:
            bad_examples = df.loc[~pwsid_ok, "pwsid"].head(5).tolist()
            logger.warning("%d rows with invalid PWSID (examples: %s)", n_bad, bad_examples)
            validator_drops["pwsid"] = n_bad
        mask &= pwsid_ok

    conc_cols = {"concentration", "detection_limit", "censored"}
    if conc_cols.issubset(df.columns):
        conc_ok = validate_concentrations(df)
        n_bad = int((~conc_ok).sum())
        if n_bad:
            logger.warning("%d rows with invalid concentration / detection limit", n_bad)
            validator_drops["concentration"] = n_bad
        mask &= conc_ok

    if not skip_coord_check and {"latitude", "longitude"}.issubset(df.columns):
        coord_ok = validate_coordinates(df)
        n_bad = int((~coord_ok).sum())
        if n_bad:
            logger.warning("%d rows with coordinates outside CONUS", n_bad)
            validator_drops["coordinates"] = n_bad
        mask &= coord_ok

    if drop_invalid_rows:
        n_dropped = int((~mask).sum())
        if n_dropped:
            detail = ", ".join(f"{k}: {v}" for k, v in validator_drops.items())
            logger.info(
                "Dropping %d invalid rows (%d remain). Breakdown: %s",
                n_dropped,
                int(mask.sum()),
                detail or "multiple validators",
            )
        df = df.loc[mask.to_numpy()].reset_index(drop=True)

    return df
