"""Environmental justice disparity analysis.

Computes burden ratios, group-stratified performance metrics, and
disparity indices to assess whether contamination predictions and
actual contamination rates differ across demographic subgroups.

Typical usage::

    from aquacontam.analysis.equity import analyze_equity

    report = analyze_equity(
        y_true=y_test,
        y_pred=preds,
        y_prob=probs,
        demographics=demo_df,
        group_col="pct_people_of_color",
    )
    print(report.burden_ratio)
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class DisparityReport:
    """Results of an equity/disparity analysis.

    Parameters
    ----------
    group_col : str
        Column used for grouping (e.g. ``"pct_people_of_color"``).
    burden_ratio : float
        Ratio of contamination rate in high-vulnerability (>80th pct)
        vs reference (<50th pct) group.
    prediction_ratio : float
        Ratio of positive prediction rate in high vs low group.
    group_metrics : dict[str, dict[str, float]]
        Per-group model performance metrics (AUROC, F1, recall, etc.).
    n_high : int
        Number of samples in high-vulnerability group.
    n_low : int
        Number of samples in low-vulnerability group.
    metadata : dict[str, Any]
        Additional info (threshold, split method, etc.).
    """

    group_col: str
    burden_ratio: float
    prediction_ratio: float
    group_metrics: dict[str, dict[str, float]]
    n_high: int
    n_low: int
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_burden_ratio(
    y_true: np.ndarray | pd.Series,
    group_values: np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
) -> tuple[float, float]:
    """Compute the burden ratio between high and low vulnerability groups.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0/1).
    group_values : array-like
        Continuous demographic values used for splitting (e.g. % people of color).
    threshold : float | None
        Split threshold. If None, uses median of ``group_values``.

    Returns
    -------
    tuple[float, float]
        ``(burden_ratio, threshold_used)``. Burden ratio is the ratio of
        contamination rate (mean of y_true) in the high group vs the low group.
        Returns ``float('inf')`` if the low group rate is 0.
    """
    y_arr = np.asarray(y_true, dtype=float)
    g_arr = np.asarray(group_values, dtype=float)

    # Drop NaN values from both
    valid = ~(np.isnan(y_arr) | np.isnan(g_arr))
    y_arr = y_arr[valid]
    g_arr = g_arr[valid]

    if len(y_arr) == 0:
        return float("nan"), float("nan")

    if threshold is None:
        threshold = float(np.median(g_arr))

    high_mask = g_arr >= threshold
    low_mask = ~high_mask

    rate_high = float(y_arr[high_mask].mean()) if high_mask.any() else 0.0
    rate_low = float(y_arr[low_mask].mean()) if low_mask.any() else 0.0

    ratio = (float("inf") if rate_high > 0.0 else 1.0) if rate_low == 0.0 else rate_high / rate_low

    return ratio, threshold


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (ECE) with equal-width probability bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|, the weighted mean gap
    between predicted confidence and empirical accuracy across ``n_bins``
    equal-width bins on [0, 1]. Lower is better-calibrated.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(yt)
    for lo, hi in itertools.pairwise(edges):
        # Last bin is closed on the right so prob == 1.0 is included.
        in_bin = (yp >= lo) & (yp < hi) if hi < 1.0 else (yp >= lo) & (yp <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        conf = float(yp[in_bin].mean())
        acc = float(yt[in_bin].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def compute_group_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None,
    group_labels: np.ndarray | pd.Series,
) -> dict[str, dict[str, float]]:
    """Compute classification metrics stratified by group.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like | None
        Predicted probabilities for positive class (for AUROC).
    group_labels : array-like
        Group label for each sample (e.g. ``"high"``/``"low"``).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of group label → metric dict (accuracy, f1, recall,
        precision, auroc, positive_rate, n_samples).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    group_arr = np.asarray(group_labels)

    y_prob_arr: np.ndarray | None = None
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)

    results: dict[str, dict[str, float]] = {}

    for label in np.unique(group_arr):
        mask = group_arr == label
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]

        if len(yt) < 2:
            continue

        metrics: dict[str, float] = {
            "n_samples": float(len(yt)),
            "positive_rate": float(yt.mean()),
            "accuracy": float(accuracy_score(yt, yp)),
        }

        # Only compute F1/precision/recall if both classes present in true labels
        if len(np.unique(yt)) > 1:
            metrics["f1"] = float(f1_score(yt, yp, zero_division=0))
            metrics["precision"] = float(precision_score(yt, yp, zero_division=0))
            metrics["recall"] = float(recall_score(yt, yp, zero_division=0))

            # Confusion-matrix error rates: deployment-relevant asymmetry
            # (a false negative = an unflagged contaminated system). FNR is
            # 1 - recall; FPR is the complement of specificity.
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
            metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")

            if y_prob_arr is not None:
                yp_prob = y_prob_arr[mask]
                with contextlib.suppress(ValueError):
                    metrics["auroc"] = float(roc_auc_score(yt, yp_prob))
                metrics["ece"] = _expected_calibration_error(yt, yp_prob)
        else:
            metrics["f1"] = float("nan")
            metrics["precision"] = float("nan")
            metrics["recall"] = float("nan")
            metrics["fpr"] = float("nan")
            metrics["fnr"] = float("nan")

        results[str(label)] = metrics

    return results


def permutation_test_burden_ratio(
    y_true: np.ndarray | pd.Series,
    group_values: np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Test statistical significance of burden ratio via permutation.

    Shuffles group assignments to build a null distribution of burden
    ratios, then computes a two-sided p-value.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0/1).
    group_values : array-like
        Continuous demographic values for splitting.
    threshold : float | None
        Split threshold. If None, uses median.
    n_permutations : int
        Number of permutations (default 10,000).
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float]
        Keys: ``"observed_ratio"``, ``"p_value"``, ``"n_permutations"``.
    """
    y_arr = np.asarray(y_true, dtype=float)
    g_arr = np.asarray(group_values, dtype=float)

    valid = ~(np.isnan(y_arr) | np.isnan(g_arr))
    y_arr = y_arr[valid]
    g_arr = g_arr[valid]

    if len(y_arr) == 0:
        return {"observed_ratio": float("nan"), "p_value": float("nan"), "n_permutations": 0}

    observed_ratio, used_threshold = compute_burden_ratio(y_arr, g_arr, threshold=threshold)

    rng = np.random.RandomState(seed)
    count_extreme = 0

    for _ in range(n_permutations):
        shuffled_groups = rng.permutation(g_arr)
        perm_ratio, _ = compute_burden_ratio(y_arr, shuffled_groups, threshold=used_threshold)
        if (
            not np.isnan(perm_ratio)
            and not np.isnan(observed_ratio)
            and abs(perm_ratio - 1.0) >= abs(observed_ratio - 1.0)
        ):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "observed_ratio": observed_ratio,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def compute_monitoring_adjusted_burden_ratio(
    y_true: np.ndarray | pd.Series,
    group_values: np.ndarray | pd.Series,
    n_samples: np.ndarray | pd.Series,
    X_causal: pd.DataFrame,
    *,
    high_threshold: float,
    low_threshold: float,
    n_permutations: int = 10_000,
    seed: int = 42,
    random_state: int = 42,
) -> dict[str, float]:
    """Monitoring-adjusted (IPW) burden ratio.

    The raw burden ratio compares observed detection rates between a high-burden
    demographic group (``>= high_threshold``) and a reference group
    (``< low_threshold``). Observed detection is *monitoring-dependent* (more
    samples -> higher P(any detection)), and monitoring intensity is itself
    correlated with demographics, so the raw ratio confounds differential
    contamination with differential ascertainment. This estimator reweights each
    system by a stabilized inverse-propensity weight for high-vs-low monitoring
    (propensity estimated from environmental/causal features only, via
    :func:`aquacontam.models._propensity.compute_propensity_scores`), standardizing
    the monitoring-intensity distribution across demographic groups before
    comparing detection rates. Significance is assessed with a matched permutation
    test (the same IPW weights, shuffling only the demographic group assignment).

    Parameters
    ----------
    y_true, group_values, n_samples : array-like
        Binary detection, the demographic indicator, and monitoring intensity
        (samples per system), all aligned to the rows of ``X_causal``.
    X_causal : pd.DataFrame
        Environmental/causal feature matrix used to estimate the monitoring
        propensity (monitoring columns are dropped internally). Must be imputed
        (no NaN); a defensive zero-fill is applied.
    high_threshold, low_threshold : float
        The high-burden / reference cut points (typically the 80th / 50th
        percentiles, matching :func:`analyze_equity`) so the adjusted ratio is
        directly comparable to the raw one.

    Returns
    -------
    dict[str, float]
        ``burden_ratio_ipw``, ``rate_high_ipw``, ``rate_low_ipw``, ``p_value``,
        ``n_high``, ``n_low``, ``n_permutations``, ``ipw_weight_mean``,
        ``ipw_weight_max``.
    """
    from aquacontam.models._propensity import compute_ipw_weights, compute_propensity_scores

    y = np.asarray(y_true, dtype=float)
    g = np.asarray(group_values, dtype=float)
    ns = np.asarray(n_samples, dtype=float)
    X_df = (
        X_causal.reset_index(drop=True)
        if isinstance(X_causal, pd.DataFrame)
        else pd.DataFrame(np.asarray(X_causal))
    )

    valid = ~(np.isnan(y) | np.isnan(g) | np.isnan(ns))
    nan_result = {
        "burden_ratio_ipw": float("nan"),
        "rate_high_ipw": float("nan"),
        "rate_low_ipw": float("nan"),
        "p_value": float("nan"),
        "n_high": 0,
        "n_low": 0,
        "n_permutations": 0,
        "ipw_weight_mean": float("nan"),
        "ipw_weight_max": float("nan"),
    }
    if valid.sum() == 0 or np.unique(ns[valid]).size < 2:
        # No data, or monitoring is constant -> IPW is undefined/degenerate.
        return nan_result

    y, g, ns = y[valid], g[valid], ns[valid]
    X_valid = X_df.loc[valid].fillna(0.0)

    propensity = compute_propensity_scores(X_valid, ns, random_state=random_state)
    ipw = compute_ipw_weights(propensity, ns)

    def _weighted_ratio(gg: np.ndarray) -> float:
        hm = gg >= high_threshold
        lm = gg < low_threshold
        rh = float(np.average(y[hm], weights=ipw[hm])) if hm.any() and ipw[hm].sum() > 0 else 0.0
        rl = float(np.average(y[lm], weights=ipw[lm])) if lm.any() and ipw[lm].sum() > 0 else 0.0
        if rl == 0.0:
            return float("inf") if rh > 0.0 else 1.0
        return rh / rl

    ratio = _weighted_ratio(g)
    high_mask = g >= high_threshold
    low_mask = g < low_threshold
    rate_high = (
        float(np.average(y[high_mask], weights=ipw[high_mask]))
        if high_mask.any() and ipw[high_mask].sum() > 0
        else 0.0
    )
    rate_low = (
        float(np.average(y[low_mask], weights=ipw[low_mask]))
        if low_mask.any() and ipw[low_mask].sum() > 0
        else 0.0
    )

    rng = np.random.RandomState(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        perm_ratio = _weighted_ratio(rng.permutation(g))
        if (
            not np.isnan(perm_ratio)
            and not np.isnan(ratio)
            and abs(perm_ratio - 1.0) >= abs(ratio - 1.0)
        ):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "burden_ratio_ipw": float(ratio),
        "rate_high_ipw": rate_high,
        "rate_low_ipw": rate_low,
        "p_value": float(p_value),
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
        "n_permutations": n_permutations,
        "ipw_weight_mean": float(np.mean(ipw)),
        "ipw_weight_max": float(np.max(ipw)),
    }


def apply_multiple_testing_correction(
    p_values: list[float],
    *,
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> dict[str, list[float] | list[bool]]:
    """Apply multiple testing correction to a set of p-values.

    Uses Benjamini-Hochberg FDR correction by default, appropriate for
    testing multiple demographic dimensions (e.g., 4 EJScreen indicators)
    or multiple analytes (e.g., 5 PFAS in T3).

    Parameters
    ----------
    p_values : list[float]
        Raw p-values from individual hypothesis tests.
    method : str
        Correction method: ``"fdr_bh"`` (Benjamini-Hochberg FDR),
        ``"bonferroni"``, or ``"holm"`` (Holm-Bonferroni).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict[str, list[float] | list[bool]]
        Keys: ``"corrected_p_values"``, ``"reject"`` (boolean mask
        indicating which hypotheses are rejected after correction).
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)

    if n == 0:
        return {"corrected_p_values": [], "reject": []}

    if method == "bonferroni":
        corrected = np.minimum(p_arr * n, 1.0)
        reject = corrected <= alpha

    elif method == "holm":
        # Holm-Bonferroni step-down procedure
        order = np.argsort(p_arr)
        corrected = np.empty(n, dtype=float)
        for rank, idx in enumerate(order):
            corrected[idx] = min(p_arr[idx] * (n - rank), 1.0)
        # Enforce monotonicity
        sorted_corrected = corrected[order]
        for i in range(1, n):
            sorted_corrected[i] = max(sorted_corrected[i], sorted_corrected[i - 1])
        corrected[order] = sorted_corrected
        reject = corrected <= alpha

    elif method == "fdr_bh":
        # Benjamini-Hochberg procedure
        order = np.argsort(p_arr)
        corrected = np.empty(n, dtype=float)
        for rank_idx, orig_idx in enumerate(order):
            bh_rank = rank_idx + 1
            corrected[orig_idx] = p_arr[orig_idx] * n / bh_rank
        # Enforce monotonicity (step up from largest)
        sorted_corrected = corrected[order]
        for i in range(n - 2, -1, -1):
            sorted_corrected[i] = min(sorted_corrected[i], sorted_corrected[i + 1])
        corrected[order] = sorted_corrected
        corrected = np.minimum(corrected, 1.0)
        reject = corrected <= alpha

    else:
        raise ValueError(f"Unknown correction method: {method!r}")

    return {
        "corrected_p_values": corrected.tolist(),
        "reject": reject.tolist(),
    }


def analyze_equity(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None,
    demographics: pd.DataFrame,
    group_col: str,
    *,
    threshold: float | None = None,
    high_percentile: float = 80,
    low_percentile: float = 50,
) -> DisparityReport:
    """Run full equity analysis on model predictions.

    Splits samples into high-burden and reference groups by percentiles
    of a demographic column, then computes burden ratios and per-group
    model performance. Samples between the low and high percentile
    thresholds are excluded from comparison.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like | None
        Predicted probabilities for positive class.
    demographics : pd.DataFrame
        Demographic data aligned with y_true (same index/order).
    group_col : str
        Column in ``demographics`` to split on (e.g. ``"pct_people_of_color"``).
    threshold : float | None
        Explicit split threshold (overrides percentile-based splitting).
        When provided, uses single-threshold split (all samples classified).
    high_percentile : float
        Percentile above which samples are classified as high-burden
        (default 80). Only used when ``threshold`` is None.
    low_percentile : float
        Percentile below which samples are classified as reference/low
        (default 50). Only used when ``threshold`` is None.

    Returns
    -------
    DisparityReport
        Complete disparity analysis results.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    group_values = np.asarray(demographics[group_col], dtype=float)

    valid = ~np.isnan(group_values)
    valid_values = group_values[valid]

    if threshold is not None:
        # Legacy single-threshold mode: all samples classified
        high_thresh = threshold
        low_thresh = threshold
        split_method = "custom"
    else:
        # Percentile-based split (paper default: 80th / 50th)
        high_thresh = (
            float(np.percentile(valid_values, high_percentile)) if len(valid_values) else 0.0
        )
        low_thresh = (
            float(np.percentile(valid_values, low_percentile)) if len(valid_values) else 0.0
        )
        split_method = f"percentile_{high_percentile}/{low_percentile}"

    # Create group labels
    group_labels = np.full(len(group_values), "middle", dtype=object)
    group_labels[group_values >= high_thresh] = "high"
    group_labels[group_values < low_thresh] = "low"
    group_labels[~valid] = "unknown"

    # For single-threshold mode, there is no middle group
    if threshold is not None:
        group_labels[group_values < high_thresh] = "low"

    high_mask = group_labels == "high"
    low_mask = group_labels == "low"

    # Burden ratio: contamination rate in high vs low
    rate_high = float(y_true_arr[high_mask].mean()) if high_mask.any() else 0.0
    rate_low = float(y_true_arr[low_mask].mean()) if low_mask.any() else 0.0

    if rate_low == 0.0:
        burden_ratio = float("inf") if rate_high > 0.0 else 1.0
    else:
        burden_ratio = rate_high / rate_low

    # Prediction ratio
    pred_rate_high = float(y_pred_arr[high_mask].mean()) if high_mask.any() else 0.0
    pred_rate_low = float(y_pred_arr[low_mask].mean()) if low_mask.any() else 0.0

    if pred_rate_low == 0.0:
        prediction_ratio = float("inf") if pred_rate_high > 0.0 else 1.0
    else:
        prediction_ratio = pred_rate_high / pred_rate_low

    # Group metrics (high, low, and middle if present)
    group_metrics = compute_group_metrics(y_true_arr, y_pred_arr, y_prob, group_labels)

    return DisparityReport(
        group_col=group_col,
        burden_ratio=burden_ratio,
        prediction_ratio=prediction_ratio,
        group_metrics=group_metrics,
        n_high=int(high_mask.sum()),
        n_low=int(low_mask.sum()),
        metadata={
            "high_threshold": high_thresh,
            "low_threshold": low_thresh,
            "split_method": split_method,
            "n_middle": int((group_labels == "middle").sum()),
        },
    )
