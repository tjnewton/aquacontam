"""Feature ablation studies.

Evaluates model performance with feature categories removed to quantify
their contribution and test for spatial leakage (e.g. aquifer type).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam.benchmark._utils import fit_with_val, get_proba
from aquacontam.benchmark.metrics import compute_classification_metrics, paired_bootstrap_test
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result from a single ablation experiment.

    Attributes
    ----------
    category : str
        Name of the dropped feature category (or "full" for baseline).
    n_features_dropped : int
        Number of columns dropped.
    dropped_columns : list[str]
        Column names that were dropped.
    metrics : dict[str, float]
        Evaluation metrics.
    n_features_remaining : int
        Number of features used in the model.
    significance : dict[str, dict[str, float]]
        Bootstrap significance tests comparing full vs. ablated model.
        Maps metric name (e.g. ``"auroc"``) to dict with keys
        ``"delta"``, ``"p_value"``, ``"ci_lower"``, ``"ci_upper"``.
        Empty for the ``"full"`` baseline row.
    """

    category: str
    n_features_dropped: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    n_features_remaining: int = 0
    significance: dict[str, dict[str, float]] = field(default_factory=dict)


def _columns_matching_prefixes(
    df: pd.DataFrame,
    prefixes: list[str],
) -> list[str]:
    """Find columns matching any of the given prefixes."""
    matched: list[str] = []
    for col in df.columns:
        for prefix in prefixes:
            if col.startswith(prefix) or col == prefix:
                matched.append(col)
                break
    return matched


def run_feature_ablation(
    model_cls: type[BaseModel],
    model_config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_categories: dict[str, list[str]],
    *,
    metrics: list[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[AblationResult]:
    """Run feature ablation by dropping one category at a time.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate.
    model_config : dict
        Model configuration.
    X_train, y_train, X_val, y_val, X_test, y_test
        Train/val/test splits.
    feature_categories : dict[str, list[str]]
        Mapping of category name → list of column prefixes to drop.
        Loaded from ``experiment.yaml`` ablation config.
    metrics : list[str] or None
        Classification metrics to compute.
    n_bootstrap : int
        Number of bootstrap iterations for significance testing (default 1000).
    seed : int
        Random seed for bootstrap reproducibility.

    Returns
    -------
    list[AblationResult]
        One result per category (including "full" baseline).
        Each ablated result includes ``significance`` with paired bootstrap
        tests comparing full vs. ablated model (AUROC and AUPRC).
    """
    results: list[AblationResult] = []

    # Full model baseline
    if len(np.unique(y_train)) < 2:
        logger.warning("Ablation: single-class y_train, returning empty results")
        return results

    model_full = model_cls(config=model_config)
    fit_with_val(model_full, X_train, y_train, X_val, y_val)
    preds_full = model_full.predict(X_test)
    probs_full = get_proba(model_full, X_test)
    metrics_full = compute_classification_metrics(y_test, preds_full, probs_full, metrics=metrics)
    results.append(
        AblationResult(
            category="full",
            n_features_dropped=0,
            dropped_columns=[],
            metrics=metrics_full,
            n_features_remaining=X_train.shape[1],
        )
    )
    logger.info(
        "Ablation [full]: %d features, AUROC=%.3f, AUPRC=%.3f",
        X_train.shape[1],
        metrics_full.get("auroc", float("nan")),
        metrics_full.get("auprc", float("nan")),
    )

    # Drop each category
    for category, prefixes in feature_categories.items():
        drop_cols = _columns_matching_prefixes(X_train, prefixes)

        if not drop_cols:
            logger.info("Ablation [-%s]: no matching columns found, skipping", category)
            continue

        X_train_abl = X_train.drop(columns=drop_cols, errors="ignore")
        X_val_abl = X_val.drop(columns=drop_cols, errors="ignore")
        X_test_abl = X_test.drop(columns=drop_cols, errors="ignore")

        if X_train_abl.empty:
            logger.warning("Ablation [-%s]: all columns dropped, skipping", category)
            continue

        if len(np.unique(y_train)) < 2:
            logger.warning("Ablation [-%s]: single-class y_train, skipping", category)
            continue

        model_abl = model_cls(config=model_config)
        fit_with_val(model_abl, X_train_abl, y_train, X_val_abl, y_val)
        preds_abl = model_abl.predict(X_test_abl)
        probs_abl = get_proba(model_abl, X_test_abl)
        metrics_abl = compute_classification_metrics(y_test, preds_abl, probs_abl, metrics=metrics)

        # Paired bootstrap significance tests (full vs. ablated)
        sig: dict[str, dict[str, float]] = {}
        if probs_full is not None and probs_abl is not None:
            for metric_name in ("auroc", "auprc"):
                sig[metric_name] = paired_bootstrap_test(
                    y_test,
                    probs_full,
                    probs_abl,
                    metric_fn=metric_name,
                    n_iterations=n_bootstrap,
                    seed=seed,
                )

        result = AblationResult(
            category=category,
            n_features_dropped=len(drop_cols),
            dropped_columns=drop_cols,
            metrics=metrics_abl,
            n_features_remaining=X_train_abl.shape[1],
            significance=sig,
        )
        results.append(result)

        # Compute delta from full model
        delta_auroc = metrics_abl.get("auroc", 0.0) - metrics_full.get("auroc", 0.0)
        delta_auprc = metrics_abl.get("auprc", 0.0) - metrics_full.get("auprc", 0.0)
        p_str = ""
        if "auroc" in sig:
            p_str = f", p={sig['auroc'].get('p_value', float('nan')):.3f}"
        logger.info(
            "Ablation [-%s]: dropped %d cols (%d remaining), "
            "AUROC=%.3f (delta=%.3f%s), AUPRC=%.3f (delta=%.3f)",
            category,
            len(drop_cols),
            X_train_abl.shape[1],
            metrics_abl.get("auroc", float("nan")),
            delta_auroc,
            p_str,
            metrics_abl.get("auprc", float("nan")),
            delta_auprc,
        )

    return results


def ablation_summary(results: list[AblationResult]) -> pd.DataFrame:
    """Convert ablation results to a summary DataFrame.

    Parameters
    ----------
    results : list[AblationResult]
        Results from ``run_feature_ablation``.

    Returns
    -------
    pd.DataFrame
        Summary with columns for each metric plus deltas from full model.
    """
    if not results:
        return cast(pd.DataFrame, pd.DataFrame())

    # Find full model metrics as baseline
    full_result = next((r for r in results if r.category == "full"), None)
    full_metrics = full_result.metrics if full_result else {}

    rows: list[dict[str, object]] = []
    for r in results:
        row: dict[str, object] = {
            "category": r.category,
            "n_dropped": r.n_features_dropped,
            "n_remaining": r.n_features_remaining,
        }
        for key, val in r.metrics.items():
            row[key] = val
            if full_metrics:
                row[f"{key}_delta"] = val - full_metrics.get(key, 0.0)
        # Add bootstrap significance columns
        for metric_name, sig_dict in r.significance.items():
            row[f"{metric_name}_p_value"] = sig_dict.get("p_value", np.nan)
            row[f"{metric_name}_ci_lower"] = sig_dict.get("ci_lower", np.nan)
            row[f"{metric_name}_ci_upper"] = sig_dict.get("ci_upper", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Apply FDR correction across ablated categories for each metric
    if len(df) > 1:
        from aquacontam.analysis.equity import apply_multiple_testing_correction

        non_full_mask = df["category"] != "full"
        for metric_name in ("auroc", "auprc"):
            p_col = f"{metric_name}_p_value"
            if p_col not in df.columns:
                continue
            raw_p = df.loc[non_full_mask, p_col].tolist()
            valid_p = [p for p in raw_p if not np.isnan(p)]
            fdr_col = f"{metric_name}_p_value_fdr"
            sig_col = f"{metric_name}_significant_fdr"
            if not valid_p:
                df[fdr_col] = np.nan
                df[sig_col] = False
                continue
            correction = apply_multiple_testing_correction(valid_p)
            corrected = correction["corrected_p_values"]
            reject = correction["reject"]
            # Initialize columns with correct dtypes first
            df[fdr_col] = np.nan
            df[sig_col] = False
            # Map corrected values back, preserving NaN for full row
            j = 0
            non_full_idx = df.index[non_full_mask]
            for idx, p in zip(non_full_idx, raw_p, strict=True):
                if not np.isnan(p):
                    df.loc[idx, fdr_col] = corrected[j]
                    df.loc[idx, sig_col] = reject[j]
                    j += 1

    return cast(pd.DataFrame, df)


# ---------------------------------------------------------------------------
# Detection-only source ablation
# ---------------------------------------------------------------------------

#: Sources that report only detected samples (0% censoring).
DETECTION_ONLY_SOURCES = frozenset({"mi_mpart", "sdwis", "wa_doh"})


def evaluate_excluding_detection_only(
    y_true: pd.Series,
    y_prob: pd.Series | np.ndarray,
    source_labels: pd.Series,
    *,
    detection_only: frozenset[str] = DETECTION_ONLY_SOURCES,
) -> dict[str, Any]:
    """Compare classification metrics with and without detection-only sources.

    This is an evaluation-side analysis: the model is NOT retrained. Instead,
    we subset the test set to exclude systems originating from detection-only
    sources (which contribute only positive examples and may inflate metrics).

    Parameters
    ----------
    y_true : pd.Series
        True labels (0/1) indexed by PWSID or integer.
    y_prob : pd.Series or np.ndarray
        Predicted probabilities (positive class).
    source_labels : pd.Series
        Source name for each test sample (e.g. "ucmr5", "wa_doh").
        Must be aligned with *y_true*.
    detection_only : frozenset[str]
        Source names to exclude (default: MI MPART, SDWIS, WA DOH).

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"all_sources": {metrics}, "excluding_detection_only": {metrics},
           "delta": {metric_name: ablated - full}}``
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred_all = (y_prob_arr >= 0.5).astype(int)

    metrics_all = compute_classification_metrics(y_true_arr, y_pred_all, y_prob_arr)

    # Filter to mixed-censoring sources only
    mask = np.asarray(~source_labels.isin(detection_only))
    if mask.sum() == 0:
        logger.warning("No test samples remain after excluding detection-only sources")
        return {"all_sources": metrics_all, "excluding_detection_only": {}, "delta": {}}

    y_true_filt = y_true_arr[mask]
    y_prob_filt = y_prob_arr[mask]
    y_pred_filt = (y_prob_filt >= 0.5).astype(int)

    if len(np.unique(y_true_filt)) < 2:
        logger.warning(
            "Single class after excluding detection-only sources — cannot compute AUROC"
        )
        return {"all_sources": metrics_all, "excluding_detection_only": {}, "delta": {}}

    metrics_filt = compute_classification_metrics(y_true_filt, y_pred_filt, y_prob_filt)

    delta = {
        k: metrics_filt.get(k, 0.0) - metrics_all.get(k, 0.0)
        for k in metrics_all
        if isinstance(metrics_all.get(k), (int, float))
    }

    return {
        "all_sources": metrics_all,
        "excluding_detection_only": metrics_filt,
        "delta": delta,
        "n_all": len(y_true_arr),
        "n_filtered": int(mask.sum()),
        "n_excluded": int((~mask).sum()),
        "excluded_sources": sorted(detection_only & set(source_labels.unique())),
    }


def evaluate_loro_fold_excluding_detection_only(
    y_true: pd.Series,
    y_prob: pd.Series | np.ndarray,
    source_labels: pd.Series,
    *,
    test_region: int,
    detection_only: frozenset[str] = DETECTION_ONLY_SOURCES,
) -> dict[str, Any]:
    """Evaluate a single LORO fold with and without detection-only sources.

    Thin wrapper around :func:`evaluate_excluding_detection_only` that adds
    LORO-specific metadata (test region, per-source exclusion counts).

    Parameters
    ----------
    y_true : pd.Series
        True labels (0/1) for the test fold.
    y_prob : pd.Series or np.ndarray
        Predicted probabilities (positive class) for the test fold.
    source_labels : pd.Series
        Source name for each test sample, aligned with *y_true*.
    test_region : int
        EPA region number for this LORO fold.
    detection_only : frozenset[str]
        Source names to exclude (default: MI MPART, SDWIS, WA DOH).

    Returns
    -------
    dict[str, Any]
        Same structure as :func:`evaluate_excluding_detection_only` plus
        ``"test_region"`` and ``"n_detection_only_by_source"``.
    """
    result = evaluate_excluding_detection_only(
        y_true, y_prob, source_labels, detection_only=detection_only
    )
    result["test_region"] = test_region

    # Per-source breakdown of excluded systems
    present = detection_only & set(source_labels.unique())
    result["n_detection_only_by_source"] = {
        src: int((source_labels == src).sum()) for src in sorted(present)
    }
    return result
