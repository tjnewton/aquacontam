"""T6: Private-well risk extrapolation (geogenic arsenic).

Trains on **public-supply** groundwater wells and evaluates **zero-shot** on
independent **domestic** (private) wells, using measured arsenic from the USGS
National Groundwater Aggregation (Erickson, Hill & Wilson 2020; CC0; see
``data/nga_arsenic.py``). The target is EPA arsenic-MCL exceedance
(``max concentration >= 10 µg/L``).

Why this design: the NJ Private Well Testing Act bulk data is confidential by
statute and unrecoverable, and public PWTA summaries are aggregated (no per-well
points). The NGA release provides measured, point-level arsenic with a water-use
code distinguishing domestic from public-supply wells — a construct-valid,
openly reproducible substitute. Arsenic is geogenic (aquifer-driven), so
non-domestic wells are not contamination-biased the way a point-source
contaminant would be.

Both populations pass through the **same** environmental feature pipeline, so
the public→domestic comparison is a clean transfer with no feature-space
mismatch. System-metadata columns (monitoring intensity) are excluded via
``exclude_features`` so the transfer reflects environmental signal, not
sampling artifacts.
"""

from __future__ import annotations

import logging

import pandas as pd

from aquacontam._constants import (
    CLASSIFICATION_METRICS,
    T6_DEFAULT_ANALYTES,
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

# Monitoring-intensity / system-metadata columns excluded from the transfer so
# the model relies on environmental signal rather than sampling artifacts that
# differ between public-supply and domestic wells. Mirrors the T5 ablation set.
T6_EXCLUDE_FEATURES: tuple[str, ...] = (
    "n_samples",
    "mean_detection_limit",
    "population_served",
    "log_population_served",
)


@register_task(
    name="T6",
    description="Private-well risk extrapolation (public-supply → domestic arsenic)",
    task_type="classification",
    primary_metric="auprc",
    analytes=T6_DEFAULT_ANALYTES,
)
def run_t6(
    *,
    model: BaseModel,
    data: pd.DataFrame,
    well_data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    analyte: str | None = None,
    target: str = "action_level",
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> TaskResult:
    """Run T6: train on public-supply wells, evaluate zero-shot on domestic wells.

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    data : pd.DataFrame
        **Public-supply** well samples (standard schema) — the training
        population. Geographically split into train/val/test.
    well_data : pd.DataFrame
        **Domestic** (private) well samples (standard schema) — the held-out
        zero-shot evaluation population.
    feature_dfs : list[pd.DataFrame], optional
        Environmental feature DataFrames indexed by well id (``pwsid``),
        covering both populations.
    analyte : str, optional
        Analyte to predict. Defaults to ``"arsenic"``.
    target : str
        Target type. Defaults to ``"action_level"`` (arsenic MCL exceedance,
        ≥10 µg/L).
    split_strategy : str
        ``"geographic"`` (default; honest EPA-region holdout) or ``"random"``
        (leaky ablation used to quantify geographic-leakage inflation).
    exclude_features : list[str], optional
        Feature columns to exclude. Defaults to :data:`T6_EXCLUDE_FEATURES`
        (monitoring-intensity / system-metadata columns).
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.

    Returns
    -------
    TaskResult
        ``split_metrics`` carries ``train``/``val``/``test`` (in-distribution
        public-supply) and ``holdout`` (zero-shot domestic). The primary
        ``metrics`` are the domestic holdout — the task's headline.
    """
    if feature_dfs is None:
        feature_dfs = []
    if analyte is None:
        analyte = T6_DEFAULT_ANALYTES[0]
    if exclude_features is None:
        exclude_features = list(T6_EXCLUDE_FEATURES)

    # Train on public-supply wells; domestic wells become the "holdout" split,
    # assembled in the training feature space (train-only imputation).
    splits = prepare_train_val_test(
        data,
        analyte,
        *feature_dfs,
        target=target,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
        categorical_columns=categorical_columns,
        holdout_df=well_data,
    )

    X_train, y_train = splits["train"]
    if X_train.empty:
        logger.warning("Empty training set for T6 (%s)", analyte)
        return TaskResult(
            task_name="T6",
            model_name=model.name,
            metrics={},
            metadata={"analyte": analyte, "error": "empty training set"},
        )

    X_val, y_val = splits.get("val", (pd.DataFrame(), pd.Series(dtype=float)))
    fit_with_val(model, X_train, y_train, X_val, y_val)

    # Coordinates for both populations (for spatial-autocorrelation metadata).
    sys_coords = get_system_coordinates(pd.concat([data, well_data], ignore_index=True))
    spatial_acc: dict[str, list] = {
        "y_true": [],
        "y_prob": [],
        "latitudes": [],
        "longitudes": [],
        "split_labels": [],
    }

    split_metrics: dict[str, dict[str, float]] = {}
    for split_name, (X, y) in splits.items():
        if X.empty:
            continue
        preds = model.predict(X)
        probs = get_proba(model, X)
        split_metrics[split_name] = compute_classification_metrics(
            y, preds, probs, metrics=list(CLASSIFICATION_METRICS)
        )
        # Accumulate spatial metadata for the public test and the domestic holdout.
        if split_name in ("test", "holdout"):
            accumulate_spatial_metadata(y, probs, X.index, sys_coords, split_name, spatial_acc)

    # Headline = domestic zero-shot holdout; fall back to public test.
    primary = split_metrics.get("holdout") or split_metrics.get(
        "test", split_metrics.get("val", {})
    )

    n_holdout = len(splits.get("holdout", (pd.DataFrame(),))[0])
    metadata: dict[str, object] = {
        "analyte": analyte,
        "target": target,
        "split_strategy": split_strategy,
        "n_train": len(X_train),
        "n_domestic_holdout": n_holdout,
    }
    if spatial_acc["y_true"]:
        metadata["y_true"] = spatial_acc["y_true"]
        metadata["y_prob"] = spatial_acc["y_prob"]
        metadata["latitudes"] = spatial_acc["latitudes"]
        metadata["longitudes"] = spatial_acc["longitudes"]
        metadata["split_labels"] = spatial_acc["split_labels"]

    return TaskResult(
        task_name="T6",
        model_name=model.name,
        metrics=primary,
        split_metrics=split_metrics,
        metadata=metadata,
    )
