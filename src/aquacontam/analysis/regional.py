"""Per-region heterogeneity analysis.

Investigates region-level differences in detection rates, feature
distributions, and missing data patterns to explain performance
variation observed in leave-one-region-out cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_region_detection_rates(
    y: pd.Series,
    regions: pd.Series,
) -> dict[int, dict[str, Any]]:
    """Compute detection rates per EPA region.

    Parameters
    ----------
    y : pd.Series
        Binary target variable (0/1).
    regions : pd.Series
        EPA region assignments, aligned to ``y``.

    Returns
    -------
    dict[int, dict[str, Any]]
        Mapping from region number to a dict with keys ``n_samples``,
        ``n_positive``, and ``detection_rate``.
    """
    result: dict[int, dict[str, Any]] = {}
    aligned_regions = regions.loc[y.index]

    for region in sorted(aligned_regions.dropna().unique()):
        mask = aligned_regions == region
        y_region = y.loc[mask]
        n_samples = len(y_region)
        if n_samples == 0:
            continue
        n_positive = int(y_region.sum())
        detection_rate = n_positive / n_samples
        result[int(region)] = {
            "n_samples": n_samples,
            "n_positive": n_positive,
            "detection_rate": round(detection_rate, 4),
        }

    return result


def compare_feature_distributions(
    X: pd.DataFrame,
    regions: pd.Series,
    top_features: list[str],
    best_region: int,
    worst_region: int,
) -> list[dict[str, Any]]:
    """Compare feature distributions between the best and worst performing regions.

    Runs a two-sample Kolmogorov-Smirnov test for each feature to identify
    distributional differences that may explain performance gaps.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    regions : pd.Series
        EPA region assignments, aligned to ``X``.
    top_features : list[str]
        Feature names to compare.
    best_region : int
        Region with highest LORO CV performance.
    worst_region : int
        Region with lowest LORO CV performance.

    Returns
    -------
    list[dict[str, Any]]
        One dict per feature with keys: ``feature``, ``ks_statistic``,
        ``p_value``, ``mean_best``, ``mean_worst``, ``std_best``,
        ``std_worst``.
    """
    aligned_regions = regions.loc[X.index]
    best_mask = aligned_regions == best_region
    worst_mask = aligned_regions == worst_region

    results: list[dict[str, Any]] = []

    for feat in top_features:
        if feat not in X.columns:
            logger.warning("Feature %r not in X, skipping", feat)
            continue

        vals_best = X.loc[best_mask, feat].dropna()
        vals_worst = X.loc[worst_mask, feat].dropna()

        if len(vals_best) < 2 or len(vals_worst) < 2:
            logger.warning(
                "Insufficient data for KS test on %r (best=%d, worst=%d)",
                feat,
                len(vals_best),
                len(vals_worst),
            )
            continue

        ks_stat, p_value = stats.ks_2samp(vals_best, vals_worst)

        results.append(
            {
                "feature": feat,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": float(p_value),
                "mean_best": round(float(vals_best.mean()), 4),
                "mean_worst": round(float(vals_worst.mean()), 4),
                "std_best": round(float(vals_best.std()), 4),
                "std_worst": round(float(vals_worst.std()), 4),
            }
        )

    # Apply FDR correction across features
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

    return results


def compute_missing_data_patterns(
    X: pd.DataFrame,
    regions: pd.Series,
) -> dict[int, dict[str, float]]:
    """Compute missing data rates per region.

    For each region reports the overall missing rate and per-feature
    missing rates for the top 5 most-missing features within that region.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    regions : pd.Series
        EPA region assignments, aligned to ``X``.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from region number to a dict with ``overall_missing_rate``
        and per-feature keys for the top 5 most-missing features.
    """
    aligned_regions = regions.loc[X.index]
    result: dict[int, dict[str, float]] = {}

    for region in sorted(aligned_regions.dropna().unique()):
        mask = aligned_regions == region
        X_region = X.loc[mask]

        if X_region.empty:
            continue

        total_cells = X_region.size
        total_missing = int(X_region.isna().sum().sum())
        overall_rate = total_missing / total_cells if total_cells > 0 else 0.0

        region_dict: dict[str, float] = {
            "overall_missing_rate": round(overall_rate, 4),
        }

        # Top 5 most-missing features in this region
        per_col_missing = X_region.isna().mean().sort_values(ascending=False)
        for col_name, rate in per_col_missing.head(5).items():
            if rate > 0:
                region_dict[str(col_name)] = round(float(rate), 4)

        result[int(region)] = region_dict

    return result


def analyze_regional_heterogeneity(
    X: pd.DataFrame,
    y: pd.Series,
    regions: pd.Series,
    top_features: list[str] | None = None,
    best_region: int = 3,
    worst_region: int = 4,
) -> dict[str, Any]:
    """Analyze per-region heterogeneity in features, targets, and missingness.

    Main entry point that orchestrates detection rate computation,
    feature distribution comparison, and missing data pattern analysis
    across EPA regions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, indexed by pwsid.
    y : pd.Series
        Binary target, indexed by pwsid.
    regions : pd.Series
        EPA region assignments, indexed by pwsid.
    top_features : list[str] or None
        Features to compare between best/worst regions. If ``None``,
        defaults to the top 5 features by variance.
    best_region : int
        EPA region with highest LORO CV AUROC (default 3).
    worst_region : int
        EPA region with lowest LORO CV AUROC (default 4).

    Returns
    -------
    dict[str, Any]
        Combined results with keys ``detection_rates``,
        ``feature_distributions``, ``missing_data``, ``best_region``,
        ``worst_region``, and ``top_features``.
    """
    # Default to top 5 features by variance if not provided
    if top_features is None:
        variances = X.var(numeric_only=True).sort_values(ascending=False)
        top_features = list(variances.head(5).index)
        logger.info("Using top 5 features by variance: %s", top_features)

    # Filter to features that actually exist in X
    valid_features = [f for f in top_features if f in X.columns]
    if not valid_features:
        logger.warning("No valid features found in X, using top 5 by variance")
        variances = X.var(numeric_only=True).sort_values(ascending=False)
        valid_features = list(variances.head(5).index)

    logger.info(
        "Analyzing regional heterogeneity: best=%d, worst=%d, %d features",
        best_region,
        worst_region,
        len(valid_features),
    )

    detection_rates = compute_region_detection_rates(y, regions)
    feature_dists = compare_feature_distributions(
        X, regions, valid_features, best_region, worst_region
    )
    missing_patterns = compute_missing_data_patterns(X, regions)

    return {
        "best_region": best_region,
        "worst_region": worst_region,
        "top_features": valid_features,
        "detection_rates": detection_rates,
        "feature_distributions": feature_dists,
        "missing_data": missing_patterns,
    }
