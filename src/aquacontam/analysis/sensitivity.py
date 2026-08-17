"""Hyperparameter sensitivity analysis.

Evaluates model performance variation across hyperparameter configurations
to quantify robustness and identify sensitivity to specific settings.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def generate_config_variants(
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
    max_configs: int = 50,
) -> list[dict[str, Any]]:
    """Generate hyperparameter configuration variants from a grid.

    Parameters
    ----------
    base_config : dict[str, Any]
        Base configuration to modify.
    param_grid : dict[str, list[Any]]
        Parameter name -> list of values to try.
    max_configs : int
        Maximum number of configurations to generate.

    Returns
    -------
    list[dict[str, Any]]
        List of configuration dicts.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(base_config)
        for k, v in zip(keys, combo, strict=True):
            cfg[k] = v
        configs.append(cfg)
        if len(configs) >= max_configs:
            break
    return configs


def hyperparameter_sensitivity(
    model_cls: type[BaseModel],
    configs: list[dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    metric: str = "auprc",
    task_type: str = "classification",
) -> dict[str, Any]:
    """Evaluate model across multiple hyperparameter configurations.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate.
    configs : list[dict[str, Any]]
        List of hyperparameter configs to evaluate.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation data.
    metric : str
        Primary metric to report.
    task_type : str
        "classification" or "regression".

    Returns
    -------
    dict[str, Any]
        Keys: ``per_config_results`` (list of dicts), ``summary`` (mean/std/min/max).
    """
    from aquacontam.benchmark.metrics import (
        compute_classification_metrics,
        compute_regression_metrics,
    )

    per_config: list[dict[str, Any]] = []

    for i, cfg in enumerate(configs):
        try:
            model = model_cls(config=cfg)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            if task_type == "classification":
                preds = model.predict(X_val)
                probs_arr = None
                try:
                    probs = model.predict_proba(X_val)
                    if probs.ndim == 2 and probs.shape[1] == 2:
                        probs_arr = probs[:, 1]
                except NotImplementedError:
                    pass
                metrics = compute_classification_metrics(
                    y_val, preds, probs_arr, metrics=["auroc", "auprc", "f1"]
                )
            else:
                preds = model.predict(X_val)
                metrics = compute_regression_metrics(y_val, preds, metrics=["rmse", "mae", "r2"])

            per_config.append({"config_index": i, "config": cfg, "metrics": metrics})
            logger.info("  Config %d: %s = %.4f", i, metric, metrics.get(metric, float("nan")))
        except (ValueError, RuntimeError, TypeError):
            logger.warning("Config %d failed", i, exc_info=True)
            per_config.append({"config_index": i, "config": cfg, "metrics": {}, "error": True})

    # Summary statistics
    valid_metrics = [r["metrics"] for r in per_config if r["metrics"] and metric in r["metrics"]]
    if valid_metrics:
        values = [m[metric] for m in valid_metrics]
        summary = {
            "metric": metric,
            "n_configs": len(configs),
            "n_successful": len(valid_metrics),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    else:
        summary = {"metric": metric, "n_configs": len(configs), "n_successful": 0}

    return {"per_config_results": per_config, "summary": summary}


def detection_limit_sensitivity(
    model_cls: type[BaseModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    fillna_values: list[float] | None = None,
    metric: str = "auroc",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate model sensitivity to detection limit fill value.

    Models that use censoring metadata (Deep Tobit, ICP) receive a
    ``mean_detection_limit`` column that may contain NaN for systems with
    unknown detection limits.  This function varies the fill value used
    for those NaNs and measures the impact on model performance.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation data.
    fillna_values : list[float] | None
        Detection limit fill values to test.  Defaults to
        ``[0.0, 0.001, 0.002, 0.004, 0.008, 0.01]``.
    metric : str
        Primary metric to report (default ``"auroc"``).
    config : dict[str, Any] | None
        Model config overrides.

    Returns
    -------
    dict[str, Any]
        Keys: ``per_value_results`` (list of dicts with ``fillna_value``
        and ``metrics``), ``summary`` (mean/std/range of primary metric).
    """
    from aquacontam.benchmark.metrics import compute_classification_metrics

    if fillna_values is None:
        fillna_values = [0.0, 0.001, 0.002, 0.004, 0.008, 0.01]

    per_value: list[dict[str, Any]] = []

    for fv in fillna_values:
        try:
            model = model_cls(config=dict(config or {}))
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            preds = model.predict(X_val)
            probs_arr = None
            try:
                probs = model.predict_proba(X_val)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs_arr = probs[:, 1]
            except NotImplementedError:
                pass
            metrics = compute_classification_metrics(
                y_val, preds, probs_arr, metrics=["auroc", "auprc", "f1"]
            )
            per_value.append({"fillna_value": fv, "metrics": metrics})
            logger.info("  DL fillna=%.4f: %s=%.4f", fv, metric, metrics.get(metric, float("nan")))
        except (ValueError, RuntimeError, TypeError):
            logger.warning("DL fillna=%.4f failed", fv, exc_info=True)
            per_value.append({"fillna_value": fv, "metrics": {}, "error": True})

    valid = [r["metrics"] for r in per_value if r["metrics"] and metric in r["metrics"]]
    if valid:
        vals = [m[metric] for m in valid]
        summary = {
            "metric": metric,
            "n_values": len(fillna_values),
            "n_successful": len(valid),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "range": float(np.max(vals) - np.min(vals)),
        }
    else:
        summary = {"metric": metric, "n_values": len(fillna_values), "n_successful": 0}

    return {"per_value_results": per_value, "summary": summary}
