"""Monitoring bias diagnostics for the ICP model.

Provides analysis functions to verify that the ICP model is truly
monitoring-invariant:
- Adversary R² tracking during training
- SHAP comparison between standard and ICP models
- Monitoring-free ablation table
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_monitoring_dependence(
    model: Any,
    X: pd.DataFrame,
    monitoring_columns: list[str] | None = None,
    impute_medians: dict[str, float] | None = None,
) -> dict[str, float]:
    """Measure how much a model's predictions depend on monitoring features.

    Computes the change in predictions when monitoring features are
    imputed to their median values.

    Parameters
    ----------
    model : BaseModel
        Fitted model with ``predict_proba`` method.
    X : pd.DataFrame
        Feature matrix.
    monitoring_columns : list[str], optional
        Columns to impute. Defaults to ``["n_samples", "mean_detection_limit"]``.
    impute_medians : dict[str, float], optional
        Pre-computed medians (from training set) for monitoring columns.
        If ``None``, medians are computed from ``X`` (which may include
        test data — use with caution).

    Returns
    -------
    dict[str, float]
        ``prediction_shift``: mean absolute change in P(positive) when
        monitoring features are imputed to median.
        ``max_shift``: maximum absolute change.
        ``correlation``: Pearson correlation between original and imputed predictions.
    """
    if monitoring_columns is None:
        monitoring_columns = ["n_samples", "mean_detection_limit"]

    cols_present = [c for c in monitoring_columns if c in X.columns]
    if not cols_present:
        return {"prediction_shift": 0.0, "max_shift": 0.0, "correlation": 1.0}

    # Original predictions
    proba_orig = model.predict_proba(X)
    if proba_orig.ndim == 2 and proba_orig.shape[1] == 2:
        p_orig = proba_orig[:, 1]
    else:
        p_orig = proba_orig.ravel()

    # Impute monitoring features to median
    X_imputed = X.copy()
    if impute_medians is None:
        logger.debug("Computing monitoring medians from input data (may include test data)")
    for col in cols_present:
        if impute_medians is not None and col in impute_medians:
            X_imputed[col] = impute_medians[col]
        else:
            X_imputed[col] = X[col].median()

    proba_imputed = model.predict_proba(X_imputed)
    if proba_imputed.ndim == 2 and proba_imputed.shape[1] == 2:
        p_imputed = proba_imputed[:, 1]
    else:
        p_imputed = proba_imputed.ravel()

    shift = np.abs(p_orig - p_imputed)
    corr = float(np.corrcoef(p_orig, p_imputed)[0, 1]) if len(p_orig) > 1 else 1.0

    return {
        "prediction_shift": float(shift.mean()),
        "max_shift": float(shift.max()),
        "correlation": corr,
    }


def build_comparison_table(
    results: dict[str, dict[str, float]],
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a comparison table of model performance metrics.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Mapping of model name → metrics dict.
    model_names : list[str], optional
        Order of models. Defaults to dict key order.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows, metrics as columns.
    """
    if model_names is None:
        model_names = list(results.keys())

    rows = []
    for name in model_names:
        if name in results:
            rows.append({"model": name, **results[name]})

    df: pd.DataFrame = pd.DataFrame(rows).set_index("model")
    return df
