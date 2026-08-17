"""Hyperparameter tuning and feature ablation utilities.

Provides grid search for XGBoost (or any model) on a validation set,
and feature ablation by category.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.benchmark.metrics import compute_classification_metrics
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def grid_search(
    model_cls: type[BaseModel],
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    metric: str = "auprc",
    base_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grid search over hyperparameter combinations.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate.
    param_grid : dict[str, list[Any]]
        Parameter name → list of values to try.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation data for scoring.
    metric : str
        Metric to optimize (default ``"auprc"``).
    base_config : dict[str, Any] | None
        Base config merged with each grid point.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        ``(best_config, all_results)`` — best config and all results.
    """
    if base_config is None:
        base_config = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    all_results: list[dict[str, Any]] = []
    best_score = -np.inf
    best_config: dict[str, Any] = base_config.copy()

    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo, strict=False)))

        model = model_cls(config=config)
        try:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            preds = model.predict(X_val)
            probs = None
            try:
                probs_full = model.predict_proba(X_val)
                if probs_full.ndim == 2 and probs_full.shape[1] == 2:
                    probs = probs_full[:, 1]
                else:
                    probs = probs_full
            except NotImplementedError:
                pass

            metrics = compute_classification_metrics(y_val, preds, probs, metrics=[metric])
            score = metrics.get(metric, float("-inf"))

            result = {"config": config, "metric": metric, "score": score}
            all_results.append(result)

            if score > best_score:
                best_score = score
                best_config = config.copy()

            logger.info(
                "  %s=%s → %s=%.4f",
                dict(zip(keys, combo, strict=False)),
                "",
                metric,
                score,
            )
        except (ValueError, RuntimeError, TypeError):
            logger.warning("Failed config: %s", config, exc_info=True)
            all_results.append({"config": config, "metric": metric, "score": float("nan")})

    logger.info("Best config: %s (%.4f)", best_config, best_score)
    return best_config, all_results


def feature_ablation(
    model_cls: type[BaseModel],
    model_config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_categories: dict[str, list[str]],
    *,
    metric: str = "auprc",
) -> list[dict[str, Any]]:
    """Run feature ablation study.

    For each feature category, trains a model with that category removed
    and reports the change in the target metric.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate.
    model_config : dict[str, Any]
        Model configuration.
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation data.
    feature_categories : dict[str, list[str]]
        Category name → list of column name prefixes belonging to that
        category.
    metric : str
        Metric to evaluate.

    Returns
    -------
    list[dict[str, Any]]
        Per-category results: category, n_features_removed, baseline_score,
        ablated_score, delta.
    """

    def _score(X_tr: pd.DataFrame, X_v: pd.DataFrame) -> float:
        if len(np.unique(y_train)) < 2:
            logger.warning(
                "Single-class y_train (%d unique) — returning NaN for %s",
                len(np.unique(y_train)),
                metric,
            )
            return float("nan")
        if len(np.unique(y_val)) < 2:
            logger.warning(
                "Single-class y_val (%d unique) — returning NaN for %s",
                len(np.unique(y_val)),
                metric,
            )
            return float("nan")
        model = model_cls(config=model_config.copy())
        model.fit(X_tr, y_train)
        preds = model.predict(X_v)
        probs = None
        try:
            p = model.predict_proba(X_v)
            probs = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p
        except NotImplementedError:
            pass
        result = compute_classification_metrics(y_val, preds, probs, metrics=[metric])
        return result.get(metric, float("nan"))

    # Baseline (all features)
    baseline_score = _score(X_train, X_val)
    logger.info("Baseline %s: %.4f", metric, baseline_score)

    results: list[dict[str, Any]] = [
        {
            "category": "all_features",
            "n_features_removed": 0,
            "baseline_score": baseline_score,
            "ablated_score": baseline_score,
            "delta": 0.0,
        }
    ]

    for category, prefixes in feature_categories.items():
        # Find columns matching any prefix in this category
        cols_to_drop = [c for c in X_train.columns if any(c.startswith(p) for p in prefixes)]
        if not cols_to_drop:
            logger.warning(
                "  %s: no matching columns found (0/%d prefixes matched)", category, len(prefixes)
            )
            results.append(
                {
                    "category": category,
                    "n_features_removed": 0,
                    "baseline_score": baseline_score,
                    "ablated_score": float("nan"),
                    "delta": float("nan"),
                    "note": "no matching columns found",
                }
            )
            continue

        X_tr_ablated = X_train.drop(columns=cols_to_drop, errors="ignore")
        X_v_ablated = X_val.drop(columns=cols_to_drop, errors="ignore")

        if X_tr_ablated.shape[1] == 0:
            logger.warning("  %s: all features removed — returning NaN", category)
            results.append(
                {
                    "category": category,
                    "n_features_removed": len(cols_to_drop),
                    "baseline_score": baseline_score,
                    "ablated_score": float("nan"),
                    "delta": float("nan"),
                    "note": "all features removed",
                }
            )
            continue

        ablated_score = _score(X_tr_ablated, X_v_ablated)
        delta = ablated_score - baseline_score

        results.append(
            {
                "category": category,
                "n_features_removed": len(cols_to_drop),
                "baseline_score": baseline_score,
                "ablated_score": ablated_score,
                "delta": delta,
            }
        )
        logger.info(
            "  -%s (%d cols): %s=%.4f (Δ=%.4f)",
            category,
            len(cols_to_drop),
            metric,
            ablated_score,
            delta,
        )

    return results
