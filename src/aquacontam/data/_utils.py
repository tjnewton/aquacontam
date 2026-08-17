"""Shared helpers for data source modules.

Consolidates column-lookup, PWSID normalization, and unit-conversion logic
that was previously duplicated across multiple loaders.
"""

from __future__ import annotations

import pandas as pd

from aquacontam._constants import MGL_TO_UGL, PPT_TO_UGL

PGL_TO_UGL: float = 1e-6
"""1 pg/L = 1e-6 ug/L."""


def find_col(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    """Return the first column name from *candidates* that exists in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search.
    candidates : list[str]
        Possible column names in priority order.
    required : bool
        If ``True`` (default), raise ``ValueError`` when no match is found.

    Returns
    -------
    str or None
        Matched column name, or ``None`` if not required and not found.

    Raises
    ------
    ValueError
        If *required* is ``True`` and no candidate is found.
    """
    # Exact match first
    for name in candidates:
        if name in df.columns:
            return name
    # Case-insensitive fallback
    col_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        actual = col_lower.get(name.lower())
        if actual is not None:
            return actual
    if required:
        raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")
    return None


def normalize_pwsid(raw: str, state_prefix: str) -> str:
    """Pad a raw PWSID to 9 characters with the given state prefix.

    If the raw value already starts with the state prefix (case-insensitive),
    only the numeric suffix is zero-padded. Otherwise the entire raw string
    is zero-padded and the prefix is prepended.

    Parameters
    ----------
    raw : str
        Raw PWSID string (may be shorter than 9 characters).
    state_prefix : str
        Two-letter state abbreviation, e.g. ``"NJ"``.

    Returns
    -------
    str
        A 9-character PWSID string.
    """
    raw = raw.strip()
    if len(raw) >= 9:
        return raw
    prefix = state_prefix.upper()
    if raw.upper().startswith(prefix):
        return prefix + raw[len(prefix) :].zfill(7)
    return prefix + raw.zfill(7)


def convert_to_ugl(value: float, unit: str) -> float:
    """Convert a concentration value to micrograms per liter (ug/L).

    Parameters
    ----------
    value : float
        Concentration value in the original unit.
    unit : str
        Unit string (case-insensitive). Supported: ``"UG/L"``, ``"PPT"``,
        ``"NG/L"``, ``"PG/L"``, ``"MG/L"``.

    Returns
    -------
    float
        Value in ug/L.

    Raises
    ------
    ValueError
        If the unit is not recognized.
    """
    unit_upper = unit.strip().upper()
    if unit_upper in ("UG/L", ""):
        return value
    if unit_upper in ("PPT", "NG/L"):
        return value * PPT_TO_UGL
    if unit_upper == "PG/L":
        return value * PGL_TO_UGL
    if unit_upper == "MG/L":
        return value * MGL_TO_UGL
    raise ValueError(f"Unknown unit: {unit!r}")
