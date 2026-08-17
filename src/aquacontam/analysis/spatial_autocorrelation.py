"""Spatial autocorrelation analysis for model residuals.

Computes Moran's I statistic on model residuals to assess whether
geographic stratification adequately addresses spatial autocorrelation
in prediction errors.

Typical usage::

    from aquacontam.analysis.spatial_autocorrelation import (
        analyze_spatial_autocorrelation,
    )

    report = analyze_spatial_autocorrelation(
        y_true=y_test,
        y_prob_or_pred=probs,
        latitude=lat_array,
        longitude=lon_array,
        split_labels=split_array,
    )
    print(report["overall"].statistic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class MoranResult:
    """Results of Moran's I spatial autocorrelation test.

    Parameters
    ----------
    statistic : float
        Moran's I statistic. Positive values indicate positive spatial
        autocorrelation (similar residuals cluster spatially).
    expected : float
        Expected I under the null hypothesis of no spatial autocorrelation.
    variance : float
        Variance of I under the null hypothesis.
    z_score : float
        Standardized test statistic.
    p_value : float
        Two-sided p-value.
    n : int
        Number of observations.
    """

    statistic: float
    expected: float
    variance: float
    z_score: float
    p_value: float
    n: int


def _build_spatial_weights(
    coords: np.ndarray,
    threshold_km: float,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Build binary spatial weights from a KD-tree with distance threshold.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape ``(n, 2)`` with projected coordinates (meters).
    threshold_km : float
        Distance threshold in kilometres. Pairs within this distance
        are assigned weight 1; all others 0.

    Returns
    -------
    tuple[dict[int, list[int]], dict[int, list[float]]]
        ``(neighbors, weights)`` where ``neighbors[i]`` is the list of
        neighbor indices for observation *i* and ``weights[i]`` is a
        list of corresponding weights (all 1.0 for binary weights).

    Notes
    -----
    Uses ``scipy.spatial.cKDTree`` for efficient neighbor queries.
    """
    threshold_m = threshold_km * 1000.0
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=threshold_m, output_type="set")

    n = len(coords)
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    weights: dict[int, list[float]] = {i: [] for i in range(n)}

    for i, j in pairs:
        neighbors[i].append(j)
        weights[i].append(1.0)
        neighbors[j].append(i)
        weights[j].append(1.0)

    return neighbors, weights


def morans_i(
    residuals: np.ndarray,
    coords: np.ndarray,
    *,
    threshold_km: float = 50.0,
) -> MoranResult:
    """Compute Moran's I spatial autocorrelation statistic.

    Uses the randomization assumption for inference (Cliff & Ord, 1981).

    Parameters
    ----------
    residuals : np.ndarray
        1-D array of model residuals (length *n*).
    coords : np.ndarray
        Array of shape ``(n, 2)`` with projected coordinates (metres).
    threshold_km : float
        Distance threshold in kilometres for the binary spatial weights
        matrix (default 50 km).

    Returns
    -------
    MoranResult
        Moran's I statistic with inference results.

    Raises
    ------
    ValueError
        If fewer than 3 observations are provided, or if all residuals
        are identical (zero variance).

    Notes
    -----
    The Moran's I statistic is computed as:

    .. math::

        I = \\frac{N}{W} \\cdot
            \\frac{\\sum_i \\sum_j w_{ij}(x_i - \\bar{x})(x_j - \\bar{x})}
                 {\\sum_i (x_i - \\bar{x})^2}

    where *N* is the number of observations, *W* is the sum of all
    spatial weights, and :math:`w_{ij}` are the binary weights.

    Inference uses the randomization assumption (Cliff & Ord, 1981) to
    derive the variance of *I* under the null hypothesis.
    """
    residuals = np.asarray(residuals, dtype=float)
    coords = np.asarray(coords, dtype=float)

    n = len(residuals)
    if n < 3:
        raise ValueError(f"Moran's I requires at least 3 observations, got {n}.")

    x = residuals
    xbar = x.mean()
    z = x - xbar
    ss = float((z**2).sum())

    if ss == 0.0:
        raise ValueError("All residuals are identical (zero variance); Moran's I is undefined.")

    # Build spatial weights
    neighbors, weights = _build_spatial_weights(coords, threshold_km)

    # Compute W (sum of all weights) and the numerator
    W = 0.0
    numerator = 0.0
    for i in range(n):
        for j_idx, j in enumerate(neighbors[i]):
            w_ij = weights[i][j_idx]
            W += w_ij
            numerator += w_ij * z[i] * z[j]

    if W == 0.0:
        logger.warning(
            "No spatial neighbors found within %.1f km; "
            "Moran's I is undefined. Consider increasing threshold_km.",
            threshold_km,
        )
        return MoranResult(
            statistic=float("nan"),
            expected=-1.0 / (n - 1),
            variance=float("nan"),
            z_score=float("nan"),
            p_value=float("nan"),
            n=n,
        )

    i_stat = (n / W) * (numerator / ss)
    expected_i = -1.0 / (n - 1)

    # --- Randomization assumption variance (Cliff & Ord, 1981) ---
    # S1 = 0.5 * sum_i sum_j (w_ij + w_ji)^2
    # S2 = sum_i (sum_j w_ij + sum_j w_ji)^2
    # For symmetric binary weights: w_ij = w_ji, so
    #   S1 = 2 * sum_i sum_j w_ij^2
    #   S2 = 4 * sum_i (sum_j w_ij)^2
    S1 = 0.0
    S2 = 0.0
    for i in range(n):
        row_sum = sum(weights[i])
        # For symmetric weights, col_sum == row_sum
        S2 += (2.0 * row_sum) ** 2
        for w_ij in weights[i]:
            # (w_ij + w_ji)^2 = (2 * w_ij)^2 for symmetric weights
            S1 += (2.0 * w_ij) ** 2
    S1 *= 0.5  # the formula has 0.5 factor

    b2 = float(n * (z**4).sum() / ss**2)  # kurtosis

    # Variance under randomization (Cliff & Ord eq. 5.8)
    A = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * W**2)
    B = b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * W**2)
    C = (n - 1) * (n - 2) * (n - 3) * W**2

    if C == 0.0:
        # Randomization variance requires n >= 4 (C contains (n-3) factor).
        # Return the statistic but without inference.
        logger.warning(
            "Cannot compute randomization variance with n=%d (need n >= 4); "
            "z-score and p-value set to NaN.",
            n,
        )
        return MoranResult(
            statistic=i_stat,
            expected=expected_i,
            variance=float("nan"),
            z_score=float("nan"),
            p_value=float("nan"),
            n=n,
        )

    ei2 = (A - B) / C
    var_i = ei2 - expected_i**2

    # Guard against numerical issues yielding negative variance
    if var_i <= 0.0:
        logger.warning(
            "Computed variance of Moran's I is non-positive (%.6e); "
            "setting z-score and p-value to NaN.",
            var_i,
        )
        return MoranResult(
            statistic=i_stat,
            expected=expected_i,
            variance=var_i,
            z_score=float("nan"),
            p_value=float("nan"),
            n=n,
        )

    z_score = (i_stat - expected_i) / np.sqrt(var_i)
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))

    return MoranResult(
        statistic=i_stat,
        expected=expected_i,
        variance=var_i,
        z_score=z_score,
        p_value=p_value,
        n=n,
    )


def _latlon_to_projected(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Convert lat/lon (EPSG:4326) to EPSG:5070 (CONUS Albers) metres.

    Uses ``pyproj`` for proper projection when available, falling back
    to an equirectangular approximation otherwise.

    Parameters
    ----------
    latitude : np.ndarray
        Latitude values in decimal degrees.
    longitude : np.ndarray
        Longitude values in decimal degrees.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 2)`` with projected coordinates in
        metres (easting, northing).
    """
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)
        return np.column_stack([easting, northing])
    except ImportError:
        logger.warning(
            "pyproj not available; falling back to equirectangular approximation "
            "for spatial weight coordinates. Install pyproj for proper EPSG:5070 projection."
        )
        lat_rad = np.radians(latitude)
        mean_lat_rad = np.mean(lat_rad)
        cos_lat = np.cos(mean_lat_rad)

        easting = longitude * 111_000.0 * cos_lat
        northing = latitude * 111_000.0

        return np.column_stack([easting, northing])


def multi_threshold_morans_i(
    residuals: np.ndarray,
    coords: np.ndarray,
    *,
    thresholds_km: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0),
) -> list[dict[str, Any]]:
    """Compute Moran's I at multiple distance thresholds.

    Shows the spatial autocorrelation decay profile — how residual
    clustering decreases with distance.

    Parameters
    ----------
    residuals : np.ndarray
        1-D array of model residuals.
    coords : np.ndarray
        Array of shape ``(n, 2)`` with projected coordinates (metres).
    thresholds_km : tuple[float, ...]
        Distance thresholds in kilometres.

    Returns
    -------
    list[dict[str, Any]]
        One entry per threshold with keys ``threshold_km``, ``statistic``,
        ``z_score``, ``p_value``, ``n``.
    """
    results: list[dict[str, Any]] = []
    for km in thresholds_km:
        try:
            result = morans_i(residuals, coords, threshold_km=km)
            results.append(
                {
                    "threshold_km": km,
                    "statistic": result.statistic,
                    "expected": result.expected,
                    "z_score": result.z_score,
                    "p_value": result.p_value,
                    "n": result.n,
                }
            )
        except ValueError as exc:
            logger.warning("Moran's I at %d km failed: %s", km, exc)
            results.append(
                {
                    "threshold_km": km,
                    "statistic": float("nan"),
                    "expected": float("nan"),
                    "z_score": float("nan"),
                    "p_value": float("nan"),
                    "n": len(residuals),
                }
            )

    # Apply FDR correction across thresholds
    if results:
        from aquacontam.analysis.equity import apply_multiple_testing_correction

        raw_p = [r["p_value"] for r in results]
        valid_p = [p for p in raw_p if not np.isnan(p)]
        if valid_p:
            correction = apply_multiple_testing_correction(valid_p)
            j = 0
            for r, p in zip(results, raw_p, strict=True):
                if np.isnan(p):
                    r["p_value_fdr"] = float("nan")
                    r["significant_fdr"] = False
                else:
                    r["p_value_fdr"] = correction["corrected_p_values"][j]
                    r["significant_fdr"] = correction["reject"][j]
                    j += 1
        else:
            for r in results:
                r["p_value_fdr"] = float("nan")
                r["significant_fdr"] = False

    return results


def analyze_spatial_autocorrelation(
    y_true: np.ndarray,
    y_prob_or_pred: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    split_labels: np.ndarray | None = None,
    threshold_km: float = 50.0,
) -> dict[str, Any]:
    """Compute spatial autocorrelation of model residuals.

    Calculates Moran's I on residuals (``y_true - y_prob_or_pred``)
    overall and per data-split partition, to diagnose whether the
    geographic train/test split adequately removes spatial dependence
    in prediction errors.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (binary labels or continuous).
    y_prob_or_pred : np.ndarray
        Predicted probabilities (classification) or predicted values
        (regression). Residuals are computed as ``y_true - y_prob_or_pred``.
    latitude : np.ndarray
        Latitude in decimal degrees (EPSG:4326).
    longitude : np.ndarray
        Longitude in decimal degrees (EPSG:4326).
    split_labels : np.ndarray | None
        Optional array of split identifiers (e.g. ``"train"``,
        ``"test"``). When provided, Moran's I is also computed
        separately for each partition.
    threshold_km : float
        Distance threshold in kilometres for the binary spatial weights
        matrix (default 50 km).

    Returns
    -------
    dict[str, Any]
        Dictionary with key ``"overall"`` mapping to a
        :class:`MoranResult`, and optionally ``"splits"`` mapping to
        a ``dict[str, MoranResult]`` if *split_labels* was provided.

    Notes
    -----
    Coordinates are converted from WGS-84 (EPSG:4326) to CONUS Albers
    (EPSG:5070) metres using ``pyproj`` when available, or an
    equirectangular approximation as fallback.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob_or_pred = np.asarray(y_prob_or_pred, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)

    residuals = y_true - y_prob_or_pred

    # Drop NaN entries
    valid = ~np.isnan(residuals) & ~np.isnan(latitude) & ~np.isnan(longitude)
    residuals = residuals[valid]
    latitude = latitude[valid]
    longitude = longitude[valid]

    if split_labels is not None:
        split_labels = np.asarray(split_labels)[valid]

    # Project to approximate metres
    coords = _latlon_to_projected(latitude, longitude)

    # Overall Moran's I
    result: dict[str, Any] = {}
    try:
        result["overall"] = morans_i(residuals, coords, threshold_km=threshold_km)
    except ValueError as exc:
        logger.warning("Could not compute overall Moran's I: %s", exc)
        result["overall"] = None

    # Per-split Moran's I
    if split_labels is not None:
        split_results: dict[str, MoranResult | None] = {}
        for label in np.unique(split_labels):
            mask = split_labels == label
            n_split = int(mask.sum())
            if n_split < 3:
                logger.warning(
                    "Split '%s' has %d observations (need >= 3); skipping.",
                    label,
                    n_split,
                )
                split_results[str(label)] = None
                continue
            try:
                split_results[str(label)] = morans_i(
                    residuals[mask],
                    coords[mask],
                    threshold_km=threshold_km,
                )
            except ValueError as exc:
                logger.warning(
                    "Could not compute Moran's I for split '%s': %s",
                    label,
                    exc,
                )
                split_results[str(label)] = None
        result["splits"] = split_results

    # Multi-threshold decay profile
    result["multi_threshold"] = multi_threshold_morans_i(residuals, coords)

    return result
