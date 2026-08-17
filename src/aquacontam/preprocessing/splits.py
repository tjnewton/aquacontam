"""Geographic stratification for train/val/test splits.

Splits water quality data by EPA region to prevent spatial leakage.
The default assignment produces roughly 60/15/25 train/val/test
based on the population served by water systems in each region.

Also provides ``random_split`` for ablation studies comparing
geographic vs. random split strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam._constants import STATE_TO_EPA_REGION

logger = logging.getLogger(__name__)


def assign_epa_region(df: pd.DataFrame) -> pd.DataFrame:
    """Derive EPA region from PWSID state code (first 2 chars).

    The ``epa_region`` column is always derived from the PWSID prefix
    first. If derivation fails (unknown state code), ``raw_Region``
    or ``raw_epa_region`` columns are used as fallbacks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``pwsid`` column.

    Returns
    -------
    pd.DataFrame
        Copy with an ``epa_region`` column (int or NaN).
    """
    df = df.copy()

    if "pwsid" not in df.columns:
        raise ValueError("DataFrame must contain a 'pwsid' column")

    # Extract 2-letter state code from PWSID prefix
    state_codes = df["pwsid"].astype(str).str[:2].str.upper()
    df["epa_region"] = state_codes.map(STATE_TO_EPA_REGION)

    # Fallback: use existing region column if derivation failed
    na_mask = df["epa_region"].isna()
    if na_mask.any():
        for fallback_col in ("raw_Region", "raw_epa_region"):
            if fallback_col in df.columns:
                fallback_vals = pd.to_numeric(df.loc[na_mask, fallback_col], errors="coerce")
                df.loc[na_mask, "epa_region"] = fallback_vals
                na_mask = df["epa_region"].isna()
                if not na_mask.any():
                    break

    # Coordinate-based fallback for systems with non-standard PWSID prefixes
    # (e.g. WQP synthetic IDs "WQP_XXXXXXX" whose prefix "WQ" is not a state code).
    if na_mask.any() and "latitude" in df.columns and "longitude" in df.columns:
        from aquacontam.preprocessing._region_lookup import latlon_to_epa_region

        coord_mask = na_mask & df["latitude"].notna() & df["longitude"].notna()
        if coord_mask.any():
            regions_from_coords = df.loc[coord_mask].apply(
                lambda r: latlon_to_epa_region(r["latitude"], r["longitude"]), axis=1
            )
            df.loc[coord_mask, "epa_region"] = regions_from_coords
            n_coord_assigned = int(regions_from_coords.notna().sum())
            if n_coord_assigned:
                logger.info(
                    "Assigned EPA region to %d rows via coordinate fallback", n_coord_assigned
                )
            na_mask = df["epa_region"].isna()

    n_missing = int(na_mask.sum())
    if n_missing:
        logger.warning(
            "%d rows could not be assigned an EPA region (unknown state code)", n_missing
        )

    return df


def geographic_split(
    df: pd.DataFrame,
    *,
    region_column: str = "epa_region",
    train_regions: tuple[int, ...] = (1, 3, 4, 5, 6),
    val_regions: tuple[int, ...] = (2, 7),
    test_regions: tuple[int, ...] = (8, 9, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by EPA region to prevent spatial leakage.

    Default region assignments produce ~60/15/25 train/val/test by
    population served:
    - Train: Regions 1, 3, 4, 5, 6 (Northeast, Mid-Atlantic, Southeast, Great Lakes, South Central)
    - Val:   Regions 2, 7 (NY/NJ, Plains)
    - Test:  Regions 8, 9, 10 (Mountain, Pacific, Northwest)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the ``region_column``.
    region_column : str
        Column containing integer EPA region numbers.
    train_regions : tuple[int, ...]
        EPA regions assigned to the training set.
    val_regions : tuple[int, ...]
        EPA regions assigned to the validation set.
    test_regions : tuple[int, ...]
        EPA regions assigned to the test set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames.

    Raises
    ------
    ValueError
        If region sets overlap or ``region_column`` is missing.
    """
    if region_column not in df.columns:
        raise ValueError(f"Column '{region_column}' not found in DataFrame")

    # Validate no overlap
    all_sets = [set(train_regions), set(val_regions), set(test_regions)]
    for i, s1 in enumerate(all_sets):
        for j, s2 in enumerate(all_sets):
            if i < j and s1 & s2:
                raise ValueError(f"Overlapping regions in splits: {s1 & s2}")

    regions = pd.to_numeric(df[region_column], errors="coerce")

    train = df[regions.isin(train_regions)].reset_index(drop=True)
    val = df[regions.isin(val_regions)].reset_index(drop=True)
    test = df[regions.isin(test_regions)].reset_index(drop=True)

    # Rows with NaN region are dropped (logged as warning)
    n_dropped = len(df) - len(train) - len(val) - len(test)
    if n_dropped:
        logger.warning("%d rows dropped (region not in any split set or NaN)", n_dropped)

    logger.info(
        "Geographic split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train),
        100 * len(train) / max(len(df), 1),
        len(val),
        100 * len(val) / max(len(df), 1),
        len(test),
        100 * len(test) / max(len(df), 1),
    )

    return train, val, test


def random_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame randomly (no geographic stratification).

    Used as an ablation baseline to quantify the benefit of geographic
    stratification. **Not recommended** for production evaluation — see
    ``geographic_split`` instead.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    train_frac : float
        Fraction of data for training (default 0.60).
    val_frac : float
        Fraction of data for validation (default 0.15).
        Test fraction is ``1 - train_frac - val_frac``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames.

    Raises
    ------
    ValueError
        If fractions don't sum to <= 1.0.
    """
    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac + val_frac must be <= 1.0, got {train_frac + val_frac:.2f}")

    rng = np.random.RandomState(seed)
    n = len(df)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[indices[:n_train]].reset_index(drop=True)
    val = df.iloc[indices[n_train : n_train + n_val]].reset_index(drop=True)
    test = df.iloc[indices[n_train + n_val :]].reset_index(drop=True)

    logger.info(
        "Random split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train),
        100 * len(train) / max(n, 1),
        len(val),
        100 * len(val) / max(n, 1),
        len(test),
        100 * len(test) / max(n, 1),
    )

    return train, val, test


def leave_one_region_out(
    df: pd.DataFrame,
    *,
    region_column: str = "epa_region",
    all_regions: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    val_fraction: float = 0.2,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]]:
    """Generate Leave-One-Region-Out (LORO) cross-validation folds.

    For each EPA region, hold it out as the test set. The remaining
    regions are split into train/val by randomly holding out
    ``val_fraction`` of the remaining regions for validation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the ``region_column``.
    region_column : str
        Column containing integer EPA region numbers.
    all_regions : tuple[int, ...]
        All EPA region numbers to iterate over.
    val_fraction : float
        Approximate fraction of non-test regions to use for validation
        (default 0.2, i.e. ~2 of 9 remaining regions).
    seed : int
        Random seed for reproducible val region selection.

    Yields
    ------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]
        ``(train, val, test, test_region)`` for each fold.

    Notes
    -----
    Produces 10 folds (one per EPA region). Each fold guarantees
    complete geographic separation between train/val/test. Fold
    sizes vary because EPA regions differ in population and number
    of water systems.
    """
    if region_column not in df.columns:
        raise ValueError(f"Column '{region_column}' not found in DataFrame")

    regions = pd.to_numeric(df[region_column], errors="coerce")
    rng = np.random.RandomState(seed)
    present_regions = sorted(set(all_regions) & set(regions.dropna().unique().astype(int)))

    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]] = []

    for test_region in present_regions:
        remaining = [r for r in present_regions if r != test_region]
        n_val = max(1, round(len(remaining) * val_fraction))
        val_regions = sorted(rng.choice(remaining, size=n_val, replace=False).tolist())
        train_regions = [r for r in remaining if r not in val_regions]

        test = df[regions == test_region]
        val = df[regions.isin(val_regions)]
        train = df[regions.isin(train_regions)]

        logger.info(
            "LORO fold (test region %d): train=%d (regions %s), val=%d (regions %s), test=%d",
            test_region,
            len(train),
            train_regions,
            len(val),
            val_regions,
            len(test),
        )
        folds.append((train, val, test, test_region))

    return folds


def split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Compute summary statistics for a geographic split.

    Parameters
    ----------
    train : pd.DataFrame
        Training set.
    val : pd.DataFrame
        Validation set.
    test : pd.DataFrame
        Test set.

    Returns
    -------
    dict[str, Any]
        Summary containing sizes, region coverage, and censoring rates
        per split.
    """
    total = len(train) + len(val) + len(test)

    def _stats(name: str, df: pd.DataFrame) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": name,
            "n_samples": len(df),
            "fraction": len(df) / max(total, 1),
        }
        if "epa_region" in df.columns:
            info["regions"] = sorted(df["epa_region"].dropna().unique().tolist())
        if "censored" in df.columns:
            info["censoring_rate"] = float(df["censored"].mean()) if len(df) > 0 else 0.0
        if "analyte" in df.columns:
            info["n_analytes"] = int(df["analyte"].nunique())
        if "pwsid" in df.columns:
            info["n_systems"] = int(df["pwsid"].nunique())
        return info

    summary: dict[str, Any] = {
        "total_samples": total,
        "train": _stats("train", train),
        "val": _stats("val", val),
        "test": _stats("test", test),
    }

    return summary
