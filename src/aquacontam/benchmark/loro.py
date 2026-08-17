"""Leave-One-Region-Out (LORO) cross-validation for benchmark tasks.

Evaluates classification models across 10 EPA region folds to provide
robust performance estimates that are less sensitive to a single
train/test split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aquacontam.benchmark._utils import fit_with_val, get_proba
from aquacontam.benchmark.metrics import compute_classification_metrics
from aquacontam.features.assembly import (
    aggregate_to_system_level,
    assemble_feature_matrix,
    drop_leakage_columns,
)
from aquacontam.models.base import BaseModel
from aquacontam.preprocessing.splits import assign_epa_region, leave_one_region_out

logger = logging.getLogger(__name__)


@dataclass
class LOROResult:
    """Results from Leave-One-Region-Out cross-validation.

    Attributes
    ----------
    task_name : str
        Name of the benchmark task (e.g. "T1").
    model_name : str
        Name of the model evaluated.
    mean_metrics : dict[str, float]
        Mean metrics across folds.
    std_metrics : dict[str, float]
        Standard deviation of metrics across folds.
    per_fold_metrics : list[dict[str, float]]
        Metrics for each fold (indexed by fold number).
    fold_regions : list[int]
        EPA region number for each fold's test set.
    n_folds : int
        Number of folds evaluated.
    """

    task_name: str
    model_name: str
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    per_fold_metrics: list[dict[str, float]] = field(default_factory=list)
    fold_regions: list[int] = field(default_factory=list)
    n_folds: int = 0

    def summary(self) -> dict[str, object]:
        """Return a summary dict suitable for JSON serialization."""
        return {
            "task": self.task_name,
            "model": self.model_name,
            "n_folds": self.n_folds,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "fold_regions": self.fold_regions,
        }


def run_loro_classification(
    model_cls: type[BaseModel],
    model_config: dict,
    data: pd.DataFrame,
    analyte: str,
    *,
    task_name: str = "T1",
    feature_dfs: list[pd.DataFrame] | None = None,
    categorical_columns: list[str] | None = None,
    target: str = "detected",
    metrics: list[str] | None = None,
) -> LOROResult:
    """Run LORO cross-validation for a classification task.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to instantiate for each fold (fresh model per fold).
    model_config : dict
        Configuration dict passed to ``model_cls(config=...)``.
    data : pd.DataFrame
        Water quality DataFrame with standard schema.
    analyte : str
        Target analyte (e.g. "PFOS", "lead").
    task_name : str
        Task identifier for the result.
    feature_dfs : list[pd.DataFrame] or None
        Feature DataFrames indexed by pwsid.
    categorical_columns : list[str] or None
        Columns to one-hot encode.
    target : str
        Target column name (default "detected").
    metrics : list[str] or None
        Classification metrics to compute (default: all standard).

    Returns
    -------
    LOROResult
        Aggregated results with per-fold breakdown.
    """
    if feature_dfs is None:
        feature_dfs = []

    # Aggregate to system level, then drop target-leaking aggregation columns
    # (any_detected / detection_rate / max_concentration). Without this, those
    # columns flow into assemble_feature_matrix() below as features and the model
    # scores ~1.0 by construction. Mirrors pipeline/analysis/_robustness.py.
    sys_agg = aggregate_to_system_level(data, analyte, target=target)
    sys_agg = drop_leakage_columns(sys_agg)
    if sys_agg.empty:
        logger.warning("Empty system aggregation for LORO %s / %s", task_name, analyte)
        model_instance = model_cls(config=model_config)
        return LOROResult(task_name=task_name, model_name=model_instance.name)

    # Assign EPA regions
    df_with_region = assign_epa_region(data)
    sys_regions = df_with_region.groupby("pwsid", observed=True)["epa_region"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    )
    sys_agg = sys_agg.join(sys_regions)
    sys_agg = sys_agg.dropna(subset=["epa_region"])
    sys_agg = sys_agg.reset_index()

    # Generate LORO folds
    folds = leave_one_region_out(sys_agg)

    per_fold_metrics: list[dict[str, float]] = []
    fold_regions: list[int] = []

    for train_df, val_df, test_df, test_region in folds:
        if test_df.empty or train_df.empty:
            logger.warning("Skipping LORO fold %d: empty split", test_region)
            continue

        # Assemble features for each split
        train_idx = train_df.set_index("pwsid")
        val_idx = val_df.set_index("pwsid") if not val_df.empty else pd.DataFrame()
        test_idx = test_df.set_index("pwsid")

        # Check for target variability
        y_train_raw = train_idx["target"]
        if len(y_train_raw.unique()) < 2:
            logger.warning("Skipping LORO fold %d: single-class training set", test_region)
            continue

        X_train, y_train, impute_stats = assemble_feature_matrix(
            train_idx,
            *feature_dfs,
            categorical_columns=categorical_columns,
        )

        if not val_idx.empty:
            X_val, y_val, _ = assemble_feature_matrix(
                val_idx,
                *feature_dfs,
                categorical_columns=categorical_columns,
                impute_stats=impute_stats,
            )
        else:
            X_val = pd.DataFrame()
            y_val = pd.Series(dtype=float)

        X_test, y_test, _ = assemble_feature_matrix(
            test_idx,
            *feature_dfs,
            categorical_columns=categorical_columns,
            impute_stats=impute_stats,
        )

        if X_train.empty or X_test.empty:
            logger.warning("Skipping LORO fold %d: empty feature matrix", test_region)
            continue

        # Fresh model for each fold
        model = model_cls(config=model_config)
        fit_with_val(model, X_train, y_train, X_val, y_val)

        preds = model.predict(X_test)
        probs = get_proba(model, X_test)

        fold_metrics = compute_classification_metrics(y_test, preds, probs, metrics=metrics)
        per_fold_metrics.append(fold_metrics)
        fold_regions.append(test_region)

        logger.info(
            "LORO fold %d: n_test=%d, AUROC=%.3f, AUPRC=%.3f",
            test_region,
            len(X_test),
            fold_metrics.get("auroc", float("nan")),
            fold_metrics.get("auprc", float("nan")),
        )

    if not per_fold_metrics:
        model_instance = model_cls(config=model_config)
        return LOROResult(task_name=task_name, model_name=model_instance.name)

    # Aggregate metrics across folds
    all_metric_keys: set[str] = set()
    for fm in per_fold_metrics:
        all_metric_keys.update(fm.keys())

    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for key in sorted(all_metric_keys):
        values = [fm[key] for fm in per_fold_metrics if key in fm and not np.isnan(fm[key])]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))
        else:
            mean_metrics[key] = float("nan")
            std_metrics[key] = float("nan")

    model_instance = model_cls(config=model_config)
    return LOROResult(
        task_name=task_name,
        model_name=model_instance.name,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        per_fold_metrics=per_fold_metrics,
        fold_regions=fold_regions,
        n_folds=len(per_fold_metrics),
    )
