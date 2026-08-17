"""T5: Cross-contaminant transfer learning task.

Evaluates whether a model trained on one contaminant class (e.g. heavy metals)
can predict contamination for a different class (e.g. PFAS) — and vice versa.

Two evaluation modes:
1. **zero-shot**: train on source analyte, evaluate on target analyte directly
2. **scratch**: train on target analyte only (baseline for comparison)

The ``transfer_ratio`` in TaskResult.metadata quantifies transfer effectiveness:
zero-shot AUROC / scratch AUROC.  A ratio near 1.0 means source knowledge
transfers well; near 0.5 means no better than random.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from aquacontam._constants import (
    CLASSIFICATION_METRICS,
    LCR_ACTION_LEVELS,
    T5_TRANSFER_DIRECTIONS,
)
from aquacontam.benchmark._utils import (
    accumulate_spatial_metadata,
    fit_with_val,
    get_proba,
    get_system_coordinates,
)
from aquacontam.benchmark.metrics import compute_classification_metrics
from aquacontam.benchmark.registry import TaskResult, register_task
from aquacontam.features.assembly import prepare_train_val_test
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _align_features(
    target_X: pd.DataFrame,
    source_columns: list[str],
) -> pd.DataFrame:
    """Align target feature columns to match source training columns.

    Missing columns are filled with 0; extra columns are dropped.

    Parameters
    ----------
    target_X : pd.DataFrame
        Target feature matrix.
    source_columns : list[str]
        Column names from source training set.

    Returns
    -------
    pd.DataFrame
        Aligned feature matrix with same columns as source.
    """
    aligned = pd.DataFrame(index=target_X.index, columns=source_columns, dtype=float)
    for col in source_columns:
        if col in target_X.columns:
            aligned[col] = target_X[col].to_numpy()
        else:
            aligned[col] = 0.0
    return cast(pd.DataFrame, aligned)


@register_task(
    name="T5",
    description="Cross-contaminant transfer learning",
    task_type="classification",
    primary_metric="auprc",
    analytes=("PFOS", "lead"),
)
def run_t5(
    *,
    model: BaseModel,
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    source_analyte: str | None = None,
    target_analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Run T5: cross-contaminant transfer learning.

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    source_data : pd.DataFrame
        Water quality data for source contaminant.
    target_data : pd.DataFrame
        Water quality data for target contaminant.
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    source_analyte : str, optional
        Source analyte for training. Defaults to first in T5_TRANSFER_DIRECTIONS.
    target_analyte : str, optional
        Target analyte for evaluation. Defaults to first in T5_TRANSFER_DIRECTIONS.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.
    exclude_features : list[str], optional
        Column names to exclude from feature matrices.

    Returns
    -------
    TaskResult
    """
    if feature_dfs is None:
        feature_dfs = []
    if source_analyte is None:
        source_analyte = T5_TRANSFER_DIRECTIONS[0][0]
    if target_analyte is None:
        target_analyte = T5_TRANSFER_DIRECTIONS[0][1]

    # --- Prepare source splits ---
    # Use action_level target for LCR analytes (lead/copper) since SDWIS LCR
    # data has near-100% detection rates making "detected" single-class.
    source_target = "action_level" if source_analyte in LCR_ACTION_LEVELS else "detected"
    source_splits = prepare_train_val_test(
        source_data,
        source_analyte,
        *feature_dfs,
        target=source_target,
        categorical_columns=categorical_columns,
        exclude_features=exclude_features,
    )

    X_src_train, y_src_train = source_splits["train"]

    if X_src_train.empty:
        logger.warning("Empty source training set for T5 (%s)", source_analyte)
        return TaskResult(
            task_name="T5",
            model_name=model.name,
            metrics={},
            metadata={
                "source_analyte": source_analyte,
                "target_analyte": target_analyte,
                "error": "empty source training set",
            },
        )

    unique_classes = y_src_train.unique()
    if len(unique_classes) < 2:
        logger.warning(
            "Single-class source training set for T5 (%s): only class(es) %s",
            source_analyte,
            unique_classes.tolist(),
        )
        return TaskResult(
            task_name="T5",
            model_name=model.name,
            metrics={},
            metadata={
                "source_analyte": source_analyte,
                "target_analyte": target_analyte,
                "error": "single-class source training set",
            },
        )

    source_columns = list(X_src_train.columns)

    # Fit with optional validation set
    X_src_val, y_src_val = source_splits.get("val", (pd.DataFrame(), pd.Series(dtype=float)))
    fit_with_val(model, X_src_train, y_src_train, X_src_val, y_src_val)

    # --- Prepare target data ---
    target_splits = prepare_train_val_test(
        target_data,
        target_analyte,
        *feature_dfs,
        target="detected",
        categorical_columns=categorical_columns,
        exclude_features=exclude_features,
    )

    # --- Zero-shot evaluation on target ---
    sys_coords = get_system_coordinates(target_data)
    spatial_acc: dict[str, list] = {
        "y_true": [],
        "y_prob": [],
        "latitudes": [],
        "longitudes": [],
        "split_labels": [],
    }

    zero_shot_metrics: dict[str, dict[str, float]] = {}
    for split_name, (X_tgt, y_tgt) in target_splits.items():
        if X_tgt.empty:
            continue
        X_aligned = _align_features(X_tgt, source_columns)
        preds = model.predict(X_aligned)
        probs = get_proba(model, X_aligned)

        zero_shot_metrics[split_name] = compute_classification_metrics(
            y_tgt, preds, probs, metrics=list(CLASSIFICATION_METRICS)
        )

        accumulate_spatial_metadata(y_tgt, probs, X_tgt.index, sys_coords, split_name, spatial_acc)

    primary_zs = zero_shot_metrics.get("test", zero_shot_metrics.get("val", {}))

    metadata: dict[str, object] = {
        "source_analyte": source_analyte,
        "target_analyte": target_analyte,
        "mode": "zero_shot",
        "n_source_train": len(X_src_train),
    }
    if spatial_acc["y_true"]:
        metadata["y_true"] = spatial_acc["y_true"]
        metadata["y_prob"] = spatial_acc["y_prob"]
        metadata["latitudes"] = spatial_acc["latitudes"]
        metadata["longitudes"] = spatial_acc["longitudes"]
        metadata["split_labels"] = spatial_acc["split_labels"]

    return TaskResult(
        task_name="T5",
        model_name=model.name,
        metrics=primary_zs,
        split_metrics=zero_shot_metrics,
        metadata=metadata,
    )


# System characteristics to exclude for ablated transfer experiments
_SYSTEM_CHAR_FEATURES = [
    "n_samples",
    "population_served",
    "log_population_served",
    "mean_detection_limit",
]


def run_t5_ablated_transfer(
    *,
    model: BaseModel,
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    source_analyte: str | None = None,
    target_analyte: str | None = None,
    categorical_columns: list[str] | None = None,
) -> dict[str, TaskResult]:
    """Run T5 transfer with and without system characteristics.

    Validates whether cross-contaminant transfer is driven by genuine
    environmental signal or shared system characteristics (n_samples,
    population_served, etc.).

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    source_data, target_data : pd.DataFrame
        Water quality data for source and target contaminants.
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    source_analyte, target_analyte : str, optional
        Source/target analytes.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.

    Returns
    -------
    dict[str, TaskResult]
        Keys ``"full"`` and ``"no_system_chars"`` with TaskResults.
    """
    import copy

    # Full transfer (baseline)
    model_full = copy.deepcopy(model)
    full_result = run_t5(
        model=model_full,
        source_data=source_data,
        target_data=target_data,
        feature_dfs=feature_dfs,
        source_analyte=source_analyte,
        target_analyte=target_analyte,
        categorical_columns=categorical_columns,
    )

    # Ablated transfer (no system characteristics)
    model_ablated = copy.deepcopy(model)
    ablated_result = run_t5(
        model=model_ablated,
        source_data=source_data,
        target_data=target_data,
        feature_dfs=feature_dfs,
        source_analyte=source_analyte,
        target_analyte=target_analyte,
        categorical_columns=categorical_columns,
        exclude_features=_SYSTEM_CHAR_FEATURES,
    )
    ablated_result.metadata["mode"] = "zero_shot_no_system_chars"
    ablated_result.metadata["excluded_features"] = _SYSTEM_CHAR_FEATURES

    return {"full": full_result, "no_system_chars": ablated_result}
