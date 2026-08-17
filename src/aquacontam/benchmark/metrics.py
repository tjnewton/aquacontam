"""Evaluation metrics for benchmark tasks.

Wraps sklearn metrics with graceful fallbacks for edge cases common
in water quality data (single-class splits, missing probabilities).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam._constants import CLASSIFICATION_METRICS, MULTILABEL_METRICS, REGRESSION_METRICS

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None = None,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like or None
        Predicted probabilities for the positive class.
        Required for ``auroc`` and ``auprc``.
    metrics : tuple or list of str, optional
        Which metrics to compute. Defaults to all ``CLASSIFICATION_METRICS``.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to value. Returns ``nan`` for metrics
        that cannot be computed (e.g., AUROC with single-class y_true).
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if metrics is None:
        metrics = list(CLASSIFICATION_METRICS)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)

    results: dict[str, float] = {}
    n_classes = len(np.unique(y_true))

    for m in metrics:
        try:
            if m == "auroc":
                if y_prob is None or n_classes < 2:
                    results[m] = float("nan")
                else:
                    results[m] = float(roc_auc_score(y_true, y_prob))
            elif m == "auprc":
                if y_prob is None or n_classes < 2:
                    results[m] = float("nan")
                else:
                    results[m] = float(average_precision_score(y_true, y_prob))
            elif m == "f1":
                results[m] = float(f1_score(y_true, y_pred, zero_division=0.0))
            elif m == "precision":
                results[m] = float(precision_score(y_true, y_pred, zero_division=0.0))
            elif m == "recall":
                results[m] = float(recall_score(y_true, y_pred, zero_division=0.0))
            elif m == "accuracy":
                results[m] = float(accuracy_score(y_true, y_pred))
            elif m == "balanced_accuracy":
                results[m] = float(balanced_accuracy_score(y_true, y_pred))
            elif m == "brier_score":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    from sklearn.metrics import brier_score_loss

                    results[m] = float(brier_score_loss(y_true, y_prob))
            elif m == "ece":
                if y_prob is None or n_classes < 2:
                    results[m] = float("nan")
                else:
                    ece_result = expected_calibration_error(y_true, y_prob)
                    results[m] = float(ece_result["ece"])
            else:
                logger.warning("Unknown classification metric: %s", m)
                results[m] = float("nan")
        except (ValueError, ZeroDivisionError):
            logger.warning("Failed to compute metric %s", m, exc_info=True)
            results[m] = float("nan")

    return results


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Compute regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    metrics : tuple or list of str, optional
        Which metrics to compute. Defaults to all ``REGRESSION_METRICS``.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to value.
    """
    from sklearn.metrics import (
        explained_variance_score,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    if metrics is None:
        metrics = list(REGRESSION_METRICS)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    results: dict[str, float] = {}

    for m in metrics:
        try:
            if m == "rmse":
                results[m] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif m == "mae":
                results[m] = float(mean_absolute_error(y_true, y_pred))
            elif m == "r2":
                results[m] = float(r2_score(y_true, y_pred))
            elif m == "explained_variance":
                results[m] = float(explained_variance_score(y_true, y_pred))
            elif m == "median_ae":
                results[m] = float(median_absolute_error(y_true, y_pred))
            else:
                logger.warning("Unknown regression metric: %s", m)
                results[m] = float("nan")
        except (ValueError, ZeroDivisionError):
            logger.warning("Failed to compute metric %s", m, exc_info=True)
            results[m] = float("nan")

    return results


def compute_multilabel_metrics(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    y_prob: np.ndarray | pd.DataFrame | None = None,
    label_names: list[str] | None = None,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Compute multilabel classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, n_labels)
        Ground truth binary label matrix.
    y_pred : array-like, shape (n_samples, n_labels)
        Predicted binary label matrix.
    y_prob : array-like or None, shape (n_samples, n_labels)
        Predicted probabilities per label.
    label_names : list[str] or None
        Label names for per-label metric keys.
    metrics : tuple or list of str, optional
        Which metrics to compute. Defaults to all ``MULTILABEL_METRICS``.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to value. When ``label_names`` is provided,
        also includes per-label AUROC/AUPRC keys like ``auroc_PFOS``.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        hamming_loss,
        label_ranking_average_precision_score,
        roc_auc_score,
    )

    if metrics is None:
        metrics = list(MULTILABEL_METRICS)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)

    _n_samples, n_labels = y_true.shape
    results: dict[str, float] = {}

    for m in metrics:
        try:
            if m == "subset_accuracy":
                results[m] = float(accuracy_score(y_true, y_pred))
            elif m == "hamming_loss":
                results[m] = float(hamming_loss(y_true, y_pred))
            elif m == "macro_auroc":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    results[m] = float(
                        roc_auc_score(y_true, y_prob, average="macro", multi_class="raise")
                    )
            elif m == "micro_auroc":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    results[m] = float(
                        roc_auc_score(y_true, y_prob, average="micro", multi_class="raise")
                    )
            elif m == "macro_auprc":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    results[m] = float(average_precision_score(y_true, y_prob, average="macro"))
            elif m == "micro_auprc":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    results[m] = float(average_precision_score(y_true, y_prob, average="micro"))
            elif m == "macro_f1":
                results[m] = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
            elif m == "micro_f1":
                results[m] = float(f1_score(y_true, y_pred, average="micro", zero_division=0.0))
            elif m == "label_ranking_avg_precision":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    results[m] = float(label_ranking_average_precision_score(y_true, y_prob))
            elif m == "mean_per_label_auroc":
                if y_prob is None:
                    results[m] = float("nan")
                else:
                    per_label = []
                    for j in range(n_labels):
                        n_classes = len(np.unique(y_true[:, j]))
                        if n_classes < 2:
                            continue
                        per_label.append(float(roc_auc_score(y_true[:, j], y_prob[:, j])))
                    results[m] = float(np.mean(per_label)) if per_label else float("nan")
            else:
                logger.warning("Unknown multilabel metric: %s", m)
                results[m] = float("nan")
        except (ValueError, ZeroDivisionError):
            logger.warning("Failed to compute multilabel metric %s", m, exc_info=True)
            results[m] = float("nan")

    # Per-label AUROC/AUPRC when label_names provided
    if label_names is not None and y_prob is not None:
        for j, name in enumerate(label_names):
            n_classes = len(np.unique(y_true[:, j]))
            if n_classes < 2:
                results[f"auroc_{name}"] = float("nan")
                results[f"auprc_{name}"] = float("nan")
                continue
            try:
                results[f"auroc_{name}"] = float(roc_auc_score(y_true[:, j], y_prob[:, j]))
                results[f"auprc_{name}"] = float(
                    average_precision_score(y_true[:, j], y_prob[:, j])
                )
            except (ValueError, ZeroDivisionError):
                results[f"auroc_{name}"] = float("nan")
                results[f"auprc_{name}"] = float("nan")

    return results


def bootstrap_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None = None,
    metrics: tuple[str, ...] | list[str] | None = None,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute classification metrics with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like or None
        Predicted probabilities for the positive class.
    metrics : tuple or list of str, optional
        Which metrics to compute. Defaults to all ``CLASSIFICATION_METRICS``.
    n_bootstrap : int
        Number of bootstrap iterations (default 1000).
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of metric name to dict with keys ``"point"``, ``"ci_lower"``,
        ``"ci_upper"``, ``"std"``.
    """
    if metrics is None:
        metrics = list(CLASSIFICATION_METRICS)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)

    # Point estimates
    point_estimates = compute_classification_metrics(y_true, y_pred, y_prob, metrics=list(metrics))

    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = 1 - confidence

    # Bootstrap
    boot_results: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_prob[idx] if y_prob is not None else None

        boot_metrics = compute_classification_metrics(y_t, y_p, y_pr, metrics=list(metrics))
        for m in metrics:
            val = boot_metrics.get(m, float("nan"))
            if not np.isnan(val):
                boot_results[m].append(val)

    results: dict[str, dict[str, float]] = {}
    for m in metrics:
        vals = np.array(boot_results[m])
        if len(vals) < 10:
            results[m] = {
                "point": point_estimates.get(m, float("nan")),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "std": float("nan"),
            }
        else:
            results[m] = {
                "point": point_estimates.get(m, float("nan")),
                "ci_lower": float(np.percentile(vals, 100 * alpha / 2)),
                "ci_upper": float(np.percentile(vals, 100 * (1 - alpha / 2))),
                "std": float(np.std(vals)),
            }

    return results


def paired_bootstrap_test(
    y_true: np.ndarray | pd.Series,
    y_prob_a: np.ndarray | pd.Series,
    y_prob_b: np.ndarray | pd.Series,
    metric_fn: str = "auroc",
    *,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for comparing two models.

    Tests whether model A significantly outperforms model B on the same
    test set using a paired bootstrap procedure.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob_a : array-like
        Predicted probabilities from model A.
    y_prob_b : array-like
        Predicted probabilities from model B.
    metric_fn : str
        Metric to compare: ``"auroc"`` or ``"auprc"``.
    n_iterations : int
        Number of bootstrap iterations (default 1000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Keys: ``"delta"`` (A - B point estimate), ``"p_value"``
        (two-sided), ``"ci_lower"``, ``"ci_upper"`` (95% CI of delta).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    n = len(y_true)

    if metric_fn == "auroc":
        score_fn = roc_auc_score
    elif metric_fn == "auprc":
        score_fn = average_precision_score
    else:
        raise ValueError(f"Unknown metric: {metric_fn}")

    # Point estimate
    try:
        observed_delta = float(score_fn(y_true, y_prob_a) - score_fn(y_true, y_prob_b))
    except ValueError:
        return {
            "delta": float("nan"),
            "p_value": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    rng = np.random.RandomState(seed)
    boot_deltas: list[float] = []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        if len(np.unique(y_t)) < 2:
            continue
        try:
            delta = float(score_fn(y_t, y_prob_a[idx]) - score_fn(y_t, y_prob_b[idx]))
            boot_deltas.append(delta)
        except ValueError:
            continue

    if len(boot_deltas) < 10:
        return {
            "delta": observed_delta,
            "p_value": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    deltas = np.array(boot_deltas)
    # Two-sided p-value: fraction of bootstrap deltas with opposite sign
    p_value = float(np.mean(deltas <= 0)) if observed_delta > 0 else float(np.mean(deltas >= 0))
    p_value = min(2 * p_value, 1.0)  # two-sided

    return {
        "delta": observed_delta,
        "p_value": p_value,
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    Critical for public health applications where risk scores are used
    for decision-making.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins for calibration (default 10).
    strategy : str
        Binning strategy: ``"uniform"`` (equal-width bins) or
        ``"quantile"`` (equal-count bins).

    Returns
    -------
    dict[str, float]
        Keys: ``"ece"`` (expected calibration error),
        ``"mce"`` (maximum calibration error),
        ``"mean_predicted_prob"``, ``"mean_observed_freq"``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "mean_predicted_prob": float("nan"),
            "mean_observed_freq": float("nan"),
        }

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        gap = abs(avg_accuracy - avg_confidence)

        ece += (n_bin / n) * gap
        mce = max(mce, gap)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "mean_predicted_prob": float(y_prob.mean()),
        "mean_observed_freq": float(y_true.mean()),
    }


def calibration_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, list[float]]:
    """Compute data for a reliability diagram (calibration curve).

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins (default 10).
    strategy : str
        Binning strategy: ``"uniform"`` or ``"quantile"``.

    Returns
    -------
    dict[str, list[float]]
        Keys: ``"mean_predicted"`` (bin centres), ``"fraction_positive"``
        (observed frequency), ``"bin_counts"`` (samples per bin).
    """
    from sklearn.calibration import calibration_curve as _sklearn_cal

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return {
            "mean_predicted": [],
            "fraction_positive": [],
            "bin_counts": [],
        }

    fraction_pos, mean_pred = _sklearn_cal(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    # Compute bin counts
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_counts: list[float] = []
    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_counts.append(float(mask.sum()))

    return {
        "mean_predicted": mean_pred.tolist(),
        "fraction_positive": fraction_pos.tolist(),
        "bin_counts": bin_counts,
    }


def optimal_threshold_analysis(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> dict[str, float]:
    """Find the optimal classification threshold for a given metric.

    Addresses the F1=0 problem where models never predict the positive
    class at the default 0.5 threshold under severe class imbalance.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    metric : str
        Metric to optimize: ``"f1"``, ``"balanced_accuracy"``,
        ``"youden"`` (Youden's J = sensitivity + specificity - 1).
    n_thresholds : int
        Number of threshold candidates to evaluate (default 100).

    Returns
    -------
    dict[str, float]
        Keys: ``"optimal_threshold"``, ``"optimal_metric_value"``,
        ``"f1_at_optimal"``, ``"precision_at_optimal"``,
        ``"recall_at_optimal"``, ``"default_threshold_f1"``.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "optimal_threshold": float("nan"),
            "optimal_metric_value": float("nan"),
            "f1_at_optimal": float("nan"),
            "precision_at_optimal": float("nan"),
            "recall_at_optimal": float("nan"),
            "default_threshold_f1": float("nan"),
        }

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        if metric == "f1":
            score = float(f1_score(y_true, y_pred_t, zero_division=0.0))
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, y_pred_t))
        elif metric == "youden":
            tp = float(((y_pred_t == 1) & (y_true == 1)).sum())
            fn = float(((y_pred_t == 0) & (y_true == 1)).sum())
            fp = float(((y_pred_t == 1) & (y_true == 0)).sum())
            tn = float(((y_pred_t == 0) & (y_true == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sens + spec - 1.0
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    # Metrics at optimal threshold
    y_opt = (y_prob >= best_thresh).astype(int)
    y_default = (y_prob >= 0.5).astype(int)

    return {
        "optimal_threshold": best_thresh,
        "optimal_metric_value": best_score,
        "f1_at_optimal": float(f1_score(y_true, y_opt, zero_division=0.0)),
        "precision_at_optimal": float(precision_score(y_true, y_opt, zero_division=0.0)),
        "recall_at_optimal": float(recall_score(y_true, y_opt, zero_division=0.0)),
        "default_threshold_f1": float(f1_score(y_true, y_default, zero_division=0.0)),
    }


def compute_censoring_aware_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    censored: np.ndarray | pd.Series,
    detection_limits: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute censoring-aware regression metrics.

    For censored (non-detect) observations, predictions below the
    detection limit are not penalized (Tobit-style). Only predictions
    above the DL for censored rows incur error.

    Parameters
    ----------
    y_true : array-like
        Ground truth concentrations (DL for censored rows).
    y_pred : array-like
        Predicted concentrations.
    censored : array-like
        Boolean array — True for censored (non-detect) observations.
    detection_limits : array-like
        Detection limits for each observation.

    Returns
    -------
    dict[str, float]
        Keys: ``detected_rmse``, ``censored_rmse``, ``concordance_index``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    censored = np.asarray(censored, dtype=bool)
    dl = np.asarray(detection_limits, dtype=np.float64)

    results: dict[str, float] = {}

    # RMSE on detected samples only
    detected_mask = ~censored
    if detected_mask.any():
        residuals = y_true[detected_mask] - y_pred[detected_mask]
        results["detected_rmse"] = float(np.sqrt(np.mean(residuals**2)))
    else:
        results["detected_rmse"] = float("nan")

    # Tobit-style RMSE on censored: only penalize if prediction > DL
    if censored.any():
        pred_censored = y_pred[censored]
        dl_censored = dl[censored]
        # Clamp: error is max(0, pred - DL)
        excess = np.maximum(0.0, pred_censored - dl_censored)
        results["censored_rmse"] = float(np.sqrt(np.mean(excess**2)))
    else:
        results["censored_rmse"] = float("nan")

    # Concordance index: fraction of pairs where ordering is correct
    # Vectorized with numpy pairwise broadcasting (detected pairs only)
    det_idx = np.where(detected_mask)[0]
    n_det = len(det_idx)
    if n_det < 2:
        results["concordance_index"] = float("nan")
    else:
        y_true_det = y_true[det_idx]
        y_pred_det = y_pred[det_idx]
        # Pairwise differences (upper triangle only)
        diff_true = y_true_det[:, None] - y_true_det[None, :]
        diff_pred = y_pred_det[:, None] - y_pred_det[None, :]
        upper = np.triu(np.ones((n_det, n_det), dtype=bool), k=1)
        # Count concordant/discordant/tied for ordered pairs
        gt = (diff_true > 0) & upper
        lt = (diff_true < 0) & upper
        concordant = int(np.sum((diff_pred > 0) & gt)) + int(np.sum((diff_pred < 0) & lt))
        discordant = int(np.sum((diff_pred < 0) & gt)) + int(np.sum((diff_pred > 0) & lt))
        tied = int(np.sum((diff_pred == 0) & (gt | lt)))
        total_pairs = concordant + discordant + tied
        if total_pairs > 0:
            results["concordance_index"] = float((concordant + 0.5 * tied) / total_pairs)
        else:
            results["concordance_index"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# DeLong test for comparing two AUROC values
# ---------------------------------------------------------------------------


def _fast_delong_placements(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute structural components (V10, V01) for the DeLong test.

    Uses the efficient Mann-Whitney placement formulation from
    Sun & Xu (2014, "Fast Implementation of DeLong's Algorithm").

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (V10, V01) placement vectors of shape (n_pos,) and (n_neg,).
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    scores_pos = y_prob[pos_idx]
    scores_neg = y_prob[neg_idx]

    # V10[i] = fraction of negatives with score < scores_pos[i] + 0.5 * ties
    sorted_neg = np.sort(scores_neg)
    left = np.searchsorted(sorted_neg, scores_pos, side="left").astype(float)
    right = np.searchsorted(sorted_neg, scores_pos, side="right").astype(float)
    v10 = (left + right) / 2.0 / n_neg

    # V01[j] = fraction of positives with score > scores_neg[j] + 0.5 * ties
    sorted_pos = np.sort(scores_pos)
    left_pos = np.searchsorted(sorted_pos, scores_neg, side="left").astype(float)
    right_pos = np.searchsorted(sorted_pos, scores_neg, side="right").astype(float)
    v01 = 1.0 - (left_pos + right_pos) / 2.0 / n_pos

    return v10, v01


def delong_test(
    y_true: np.ndarray | pd.Series,
    y_prob_a: np.ndarray | pd.Series,
    y_prob_b: np.ndarray | pd.Series,
) -> dict[str, float]:
    """DeLong test for comparing two ROC AUC values.

    Tests the null hypothesis that two classifiers have equal AUROC on
    the same test set. More efficient and better calibrated than
    bootstrap for AUROC comparisons.

    Based on DeLong et al. (1988) as efficiently implemented by
    Sun & Xu (2014).

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    y_prob_a : array-like
        Predicted probabilities from model A.
    y_prob_b : array-like
        Predicted probabilities from model B.

    Returns
    -------
    dict[str, float]
        Keys: ``"auroc_a"``, ``"auroc_b"``, ``"delta"``,
        ``"z_statistic"``, ``"p_value"`` (two-sided).
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=int)
    y_prob_a = np.asarray(y_prob_a, dtype=float)
    y_prob_b = np.asarray(y_prob_b, dtype=float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos < 1 or n_neg < 1:
        return {
            "auroc_a": float("nan"),
            "auroc_b": float("nan"),
            "delta": float("nan"),
            "z_statistic": float("nan"),
            "p_value": float("nan"),
        }

    v10_a, v01_a = _fast_delong_placements(y_true, y_prob_a)
    v10_b, v01_b = _fast_delong_placements(y_true, y_prob_b)

    auroc_a = float(v10_a.mean())
    auroc_b = float(v10_b.mean())

    # Covariance matrix of (AUC_A, AUC_B)
    # S10 = covariance among positive placements
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    # S01 = covariance among negative placements
    s01 = np.cov(np.vstack([v01_a, v01_b]))

    # Handle scalar returns for single-sample edge cases
    if s10.ndim == 0:
        s10 = np.array([[float(s10), 0.0], [0.0, float(s10)]])
    if s01.ndim == 0:
        s01 = np.array([[float(s01), 0.0], [0.0, float(s01)]])

    # Variance of the difference AUC_A - AUC_B
    # contrast = [1, -1]
    cov_matrix = s10 / n_pos + s01 / n_neg
    var_diff = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]

    if var_diff <= 0:
        # Zero variance means models are identical (or nearly so)
        delta = auroc_a - auroc_b
        return {
            "auroc_a": auroc_a,
            "auroc_b": auroc_b,
            "delta": delta,
            "z_statistic": 0.0 if abs(delta) < 1e-10 else float("nan"),
            "p_value": 1.0 if abs(delta) < 1e-10 else float("nan"),
        }

    z = (auroc_a - auroc_b) / np.sqrt(var_diff)
    p_value = 2.0 * float(stats.norm.sf(abs(z)))

    return {
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "delta": auroc_a - auroc_b,
        "z_statistic": float(z),
        "p_value": p_value,
    }


def delong_test_vs_chance(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float]:
    """DeLong one-sample test of AUROC against chance (0.5).

    Uses the DeLong variance estimate to test H0: AUROC = 0.5.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict[str, float]
        Keys: ``"auroc"``, ``"variance"``, ``"z_statistic"``, ``"p_value"``
        (two-sided).
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos < 1 or n_neg < 1:
        return {
            "auroc": float("nan"),
            "variance": float("nan"),
            "z_statistic": float("nan"),
            "p_value": float("nan"),
        }

    v10, v01 = _fast_delong_placements(y_true, y_prob)
    auroc = float(v10.mean())

    # DeLong variance of single AUC
    var_auroc = float(np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg)

    if var_auroc <= 0:
        # Zero variance: perfect separation (AUROC=1) or degenerate case
        if abs(auroc - 0.5) < 1e-10:
            return {"auroc": auroc, "variance": 0.0, "z_statistic": 0.0, "p_value": 1.0}
        # AUROC != 0.5 with zero variance → deterministically different from chance
        z_sign = 1.0 if auroc > 0.5 else -1.0
        return {
            "auroc": auroc,
            "variance": 0.0,
            "z_statistic": z_sign * float("inf"),
            "p_value": 0.0,
        }

    z = (auroc - 0.5) / np.sqrt(var_auroc)
    p_value = 2.0 * float(stats.norm.sf(abs(z)))

    return {
        "auroc": auroc,
        "variance": var_auroc,
        "z_statistic": float(z),
        "p_value": p_value,
    }


def compute_lift_analysis(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
    *,
    reference_prevalence: float | None = None,
) -> dict[str, Any]:
    """Compute decile-based lift analysis.

    Ranks systems by predicted risk and measures detection rates per decile
    to quantify how much better model-guided prioritization is vs random.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins (default 10 for deciles).
    reference_prevalence : float | None
        Optional representative base rate to anchor lift against (e.g. the
        national UCMR5 PFOS detection rate). The in-sample evaluation prevalence
        can be ENRICHED relative to the national rate because detection-enriched
        sources are pooled in, which makes the in-sample lift understate the
        deployment-relevant prioritization gain. When provided, additional
        ``*_vs_reference`` fields report lift relative to this base rate; the
        default ``top_decile_lift`` remains anchored to the in-sample prevalence.
        Both are reported so the reader can see the prevalence each is computed
        against (``overall_detection_rate`` vs ``reference_prevalence``).

    Returns
    -------
    dict[str, Any]
        decile_detection_rates : list[float]
            Detection rate per decile (highest risk first).
        cumulative_lift : list[float]
            Lift vs random at each cumulative cutoff (in-sample prevalence).
        top_decile_lift : float
            Detection rate in top 10% / overall (in-sample) rate.
        top_quintile_capture : float
            Fraction of all detections in top 20%.
        overall_detection_rate : float
        reference_prevalence, top_decile_lift_vs_reference,
        cumulative_lift_vs_reference : present only when ``reference_prevalence``
            is supplied.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        empty: dict[str, Any] = {
            "decile_detection_rates": [],
            "cumulative_lift": [],
            "top_decile_lift": float("nan"),
            "top_quintile_capture": float("nan"),
            "overall_detection_rate": float("nan"),
        }
        if reference_prevalence is not None:
            empty["reference_prevalence"] = float(reference_prevalence)
            empty["top_decile_lift_vs_reference"] = float("nan")
            empty["cumulative_lift_vs_reference"] = []
        return empty

    overall_rate = float(y_true.mean())
    total_positives = float(y_true.sum())

    # Sort by predicted risk (descending)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    # Split into bins
    bin_size = len(y_sorted) // n_bins
    remainder = len(y_sorted) % n_bins

    decile_rates: list[float] = []
    cumulative_lift: list[float] = []
    cumulative_positives = 0.0
    cumulative_total = 0

    for i in range(n_bins):
        # Distribute remainder across first bins
        size = bin_size + (1 if i < remainder else 0)
        start = cumulative_total
        end = start + size
        bin_labels = y_sorted[start:end]

        rate = float(bin_labels.mean()) if size > 0 else 0.0
        decile_rates.append(rate)

        cumulative_positives += float(bin_labels.sum())
        cumulative_total = end

        # Lift: cumulative detection rate / overall rate
        cum_rate = cumulative_positives / cumulative_total if cumulative_total > 0 else 0.0
        lift = cum_rate / overall_rate if overall_rate > 0 else float("nan")
        cumulative_lift.append(lift)

    # Top decile lift
    top_decile_lift = decile_rates[0] / overall_rate if overall_rate > 0 else float("nan")

    # Top quintile capture: fraction of detections in top 20%
    n_top_quintile = len(y_sorted) // 5
    if n_top_quintile > 0 and total_positives > 0:
        top_quintile_capture = float(y_sorted[:n_top_quintile].sum() / total_positives)
    else:
        top_quintile_capture = float("nan")

    result: dict[str, Any] = {
        "decile_detection_rates": decile_rates,
        "cumulative_lift": cumulative_lift,
        "top_decile_lift": top_decile_lift,
        "top_quintile_capture": top_quintile_capture,
        "overall_detection_rate": overall_rate,
    }

    if reference_prevalence is not None and reference_prevalence > 0:
        # Re-anchor lift to a representative (e.g. national UCMR5) base rate so the
        # number is not conflated with the enriched in-sample prevalence.
        result["reference_prevalence"] = float(reference_prevalence)
        result["top_decile_lift_vs_reference"] = decile_rates[0] / reference_prevalence
        result["cumulative_lift_vs_reference"] = [
            cl * overall_rate / reference_prevalence for cl in cumulative_lift
        ]

    return result


def compare_models_pairwise(
    y_true: np.ndarray | pd.Series,
    model_probs: dict[str, np.ndarray | pd.Series],
    *,
    method: str = "delong",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise model comparison tests with FDR correction.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0/1).
    model_probs : dict[str, array-like]
        Mapping of model name → predicted probabilities.
    method : str
        ``"delong"`` for DeLong test or ``"bootstrap"`` for paired
        bootstrap test.
    alpha : float
        Significance level for FDR correction (default 0.05).

    Returns
    -------
    pd.DataFrame
        Columns: ``model_a``, ``model_b``, ``auroc_a``, ``auroc_b``,
        ``delta``, ``z_statistic`` (DeLong only), ``p_value``,
        ``p_value_fdr``, ``significant``.
    """
    names = sorted(model_probs.keys())
    rows: list[dict[str, object]] = []

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            if method == "delong":
                result = delong_test(y_true, model_probs[name_a], model_probs[name_b])
            elif method == "bootstrap":
                result = paired_bootstrap_test(y_true, model_probs[name_a], model_probs[name_b])
                result["auroc_a"] = float("nan")
                result["auroc_b"] = float("nan")
                result["z_statistic"] = float("nan")
            else:
                raise ValueError(f"Unknown method: {method!r}")

            rows.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "auroc_a": result.get("auroc_a", float("nan")),
                    "auroc_b": result.get("auroc_b", float("nan")),
                    "delta": result["delta"],
                    "z_statistic": result.get("z_statistic", float("nan")),
                    "p_value": result["p_value"],
                }
            )

    if not rows:
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                columns=[
                    "model_a",
                    "model_b",
                    "auroc_a",
                    "auroc_b",
                    "delta",
                    "z_statistic",
                    "p_value",
                    "p_value_fdr",
                    "significant",
                ]
            ),
        )

    df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction
    p_values = df["p_value"].to_numpy()
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, n_tests + 1)
    fdr_adjusted = np.minimum(p_values * n_tests / ranks, 1.0)
    # Enforce monotonicity (from largest rank down)
    sorted_fdr = fdr_adjusted[sorted_idx]
    for j in range(n_tests - 2, -1, -1):
        sorted_fdr[j] = min(sorted_fdr[j], sorted_fdr[j + 1])
    fdr_adjusted[sorted_idx] = sorted_fdr

    df["p_value_fdr"] = fdr_adjusted
    df["significant"] = df["p_value_fdr"] < alpha

    return cast(pd.DataFrame, df)
