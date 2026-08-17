"""Detection limit handling for left-censored water quality data.

Water quality measurements below the method reporting limit (MRL) are
left-censored: we know the true value is between 0 and the detection limit,
but not the exact value. This module provides substitution methods,
summary statistics, and a nonparametric Kaplan-Meier mean estimator.

Note
----
The main pipeline uses **binary detection targets** (any PFAS detected at a
system, via ``aggregate_to_system_level(..., target="detected")``), so
``substitute_non_detects()`` is not called in the standard pipeline flow.
Concentration substitution is available for sensitivity analyses and for
models that require continuous concentration values (e.g., Deep Tobit).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import norm

from aquacontam._constants import DL_SUBSTITUTION_FACTOR_HALF, DL_SUBSTITUTION_FACTOR_SQRT2

logger = logging.getLogger(__name__)


class SubstitutionMethod(Enum):
    """Methods for substituting non-detect (censored) values."""

    HALF_DL = "half_dl"
    """Replace with DL / 2 (simple, widely used)."""

    SQRT2_DL = "sqrt2_dl"
    """Replace with DL / sqrt(2) (Hornung & Reed, 1990)."""

    ZERO = "zero"
    """Replace with 0 (biased low, but transparent)."""

    DL = "dl"
    """Replace with the detection limit itself (biased high)."""

    LOGNORMAL_ROS = "lognormal_ros"
    """Regression on Order Statistics (Helsel, 2005)."""


def _lognormal_ros_impute(
    concentrations: np.ndarray,
    censored: np.ndarray,
    detection_limits: np.ndarray,
) -> np.ndarray:
    """Impute censored values using lognormal Regression on Order Statistics.

    Parameters
    ----------
    concentrations : np.ndarray
        Observed concentrations (may be 0 for censored rows).
    censored : np.ndarray
        Boolean array — ``True`` for left-censored (non-detect) observations.
    detection_limits : np.ndarray
        Detection limits for each observation.

    Returns
    -------
    np.ndarray
        Concentrations with censored values replaced by ROS-imputed values.

    References
    ----------
    Helsel, D.R. (2005). *Nondetects and Data Analysis*. Wiley.
    """
    result: np.ndarray = concentrations.copy()
    n = len(concentrations)
    n_censored = int(censored.sum())
    n_detected = n - n_censored

    # Edge case: no censoring — return as-is
    if n_censored == 0:
        return result

    # Edge case: all censored or ≤1 detected — fall back to DL/2
    if n_detected <= 1:
        result[censored] = detection_limits[censored] * DL_SUBSTITUTION_FACTOR_HALF
        return result

    # Extract detected values; guard against non-positive values that would
    # produce -inf under log().  Re-classify them as censored (DL/2 fallback).
    detected_vals = concentrations[~censored]
    nonpositive = detected_vals <= 0
    if nonpositive.any():
        # Treat non-positive "detected" values as censored
        result[~censored] = np.where(
            nonpositive,
            detection_limits[~censored] * DL_SUBSTITUTION_FACTOR_HALF,
            detected_vals,
        )
        detected_vals = detected_vals[~nonpositive]
        n_detected = len(detected_vals)
        n_censored = n - n_detected
        if n_detected <= 1:
            result[censored] = detection_limits[censored] * DL_SUBSTITUTION_FACTOR_HALF
            return result
    detected_sorted = np.sort(detected_vals)

    # Test lognormality assumption before ROS imputation
    if n_detected >= 3:
        from scipy.stats import shapiro

        log_detected = np.log(detected_vals)
        try:
            # Shapiro-Wilk on log-transformed values tests lognormality
            if n_detected <= 5000:
                _, sw_p = shapiro(log_detected)
            else:
                from scipy.stats import normaltest

                _, sw_p = normaltest(log_detected)
            if sw_p < 0.05:
                logger.warning(
                    "Lognormal assumption rejected (p=%.4f) for ROS imputation "
                    "with %d detected values. Consider using half_dl or sqrt2_dl "
                    "substitution instead.",
                    sw_p,
                    n_detected,
                )
        except (ValueError, RuntimeError):
            pass  # Don't fail ROS over a diagnostic test

    # Proportion censored
    p_censored = n_censored / n

    # Plotting positions for detected values (Blom formula adjusted for censoring)
    i_vals = np.arange(1, n_detected + 1, dtype=np.float64)
    pp_detected = p_censored + (1.0 - p_censored) * (i_vals - 0.375) / (n_detected + 0.25)

    # Normal quantiles for detected values
    z_detected = norm.ppf(pp_detected)

    # OLS fit: log(detected) = intercept + slope * z
    log_detected = np.log(detected_sorted)
    slope, intercept = np.polyfit(z_detected, log_detected, 1)

    # Plotting positions for censored values (uniformly below p_censored)
    j_vals = np.arange(1, n_censored + 1, dtype=np.float64)
    pp_censored = j_vals * p_censored / (n_censored + 1)

    # Predict imputed values
    z_censored = norm.ppf(pp_censored)
    log_imputed = intercept + slope * z_censored
    imputed = np.exp(log_imputed)

    # Clip to [0, detection_limit] and assign
    dl_censored = detection_limits[censored]
    imputed = np.clip(imputed, 0.0, dl_censored)

    result[censored] = imputed
    return result


def substitute_non_detects(
    df: pd.DataFrame,
    method: SubstitutionMethod = SubstitutionMethod.HALF_DL,
    *,
    concentration_col: str = "concentration",
    censored_col: str = "censored",
    detection_limit_col: str = "detection_limit",
) -> pd.DataFrame:
    """Apply substitution to censored rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *concentration_col*, *censored_col*, and
        *detection_limit_col*.
    method : SubstitutionMethod
        Which substitution strategy to apply.
    concentration_col, censored_col, detection_limit_col : str
        Column names (defaults match the standard schema).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with substituted concentrations and a new
        ``substitution_method`` column recording which method was used.
    """
    df = df.copy()
    mask = df[censored_col]
    dl = df[detection_limit_col]

    if method is SubstitutionMethod.LOGNORMAL_ROS:
        imputed = _lognormal_ros_impute(
            df[concentration_col].to_numpy(dtype=np.float64),
            mask.to_numpy(dtype=bool),
            dl.to_numpy(dtype=np.float64),
        )
        df[concentration_col] = imputed
    elif method is SubstitutionMethod.HALF_DL:
        df.loc[mask, concentration_col] = dl[mask] * DL_SUBSTITUTION_FACTOR_HALF
    elif method is SubstitutionMethod.SQRT2_DL:
        df.loc[mask, concentration_col] = dl[mask] * DL_SUBSTITUTION_FACTOR_SQRT2
    elif method is SubstitutionMethod.ZERO:
        df.loc[mask, concentration_col] = 0.0
    elif method is SubstitutionMethod.DL:
        df.loc[mask, concentration_col] = dl[mask]

    df["substitution_method"] = ""
    df.loc[mask, "substitution_method"] = method.value
    return df


def compute_censoring_summary(
    df: pd.DataFrame,
    *,
    group_by: str | list[str] | None = None,
    censored_col: str = "censored",
) -> pd.DataFrame:
    """Compute censoring statistics, optionally grouped.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *censored_col*.
    group_by : str | list[str] | None
        Column(s) to group by (e.g. ``"analyte"``).
    censored_col : str
        Name of the boolean censored column.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_total``, ``n_censored``, ``pct_censored``.
        Indexed by *group_by* columns if provided.
    """
    if group_by is not None:
        grouped = df.groupby(group_by, observed=True)[censored_col]
        result = grouped.agg(
            n_total="count",
            n_censored="sum",
        )
        summary = pd.DataFrame(result)
    else:
        summary = pd.DataFrame(
            {
                "n_total": [len(df)],
                "n_censored": [int(df[censored_col].sum())],
            }
        )

    summary["pct_censored"] = (summary["n_censored"] / summary["n_total"]) * 100
    return cast(pd.DataFrame, summary)


def add_binary_detection_column(
    df: pd.DataFrame,
    *,
    censored_col: str = "censored",
) -> pd.DataFrame:
    """Add a ``detected`` boolean column (inverse of ``censored``).

    This is the target variable for benchmark task T1 (binary PFAS
    detection).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *censored_col*.
    censored_col : str
        Name of the boolean censored column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``detected`` column.
    """
    df = df.copy()
    df["detected"] = ~df[censored_col]
    return df


def kaplan_meier_mean(
    concentrations: np.ndarray | pd.Series,
    censored: np.ndarray | pd.Series,
) -> float:
    """Nonparametric Kaplan-Meier estimator for the mean of left-censored data.

    Uses the "flipped" Kaplan-Meier approach where left-censored
    observations are treated analogously to right-censored survival data
    by reversing the order.

    Parameters
    ----------
    concentrations : array-like
        Observed values (for censored rows, this is the detection limit).
    censored : array-like
        Boolean array — ``True`` for left-censored (non-detect) observations.

    Returns
    -------
    float
        Estimated mean concentration.

    References
    ----------
    Helsel, D.R. (2005). *Nondetects and Data Analysis*. Wiley.
    """
    conc = np.asarray(concentrations, dtype=np.float64)
    cens = np.asarray(censored, dtype=bool)

    if len(conc) == 0:
        return np.nan

    # If all values are censored, we can only bound the mean
    if cens.all():
        return float(conc.max()) / 2.0

    # If no values are censored, simple mean
    if not cens.any():
        return float(conc.mean())

    # Sort by concentration (ascending)
    order = np.argsort(conc)
    conc_sorted = conc[order]
    cens_sorted = cens[order]
    n = len(conc_sorted)

    # Reversed KM: treat left-censored as right-censored by flipping.
    # Reverse the data so the largest values come first.
    conc_rev = conc_sorted[::-1]
    cens_rev = cens_sorted[::-1]

    # Compute the KM survival function on reversed data
    # At each step i, n_at_risk = n - i (number remaining)
    # Event = not censored (detected); censored = censored
    survival = 1.0
    km_mean = 0.0
    prev_val = conc_rev[0]

    for i in range(n):
        n_at_risk = n - i
        if not cens_rev[i]:
            # "Event" in reversed KM = detected value
            survival *= (n_at_risk - 1) / n_at_risk

        next_val = conc_rev[i + 1] if i + 1 < n else 0.0

        # Area under the KM curve between current and next value
        km_mean += (prev_val - next_val) * survival
        prev_val = next_val

    return float(km_mean)
