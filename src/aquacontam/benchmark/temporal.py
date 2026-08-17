"""T7: Temporal prediction — train on UCMR3 (2013-2015), predict UCMR5 (2023-2025).

Evaluates whether contamination patterns from UCMR3 monitoring (2013-2015)
can predict detection outcomes in UCMR5 (2023-2025). Also reports separate
metrics for *persistent* systems (present in both eras) vs *new* systems
(UCMR5-only).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam._constants import (
    CLASSIFICATION_METRICS,
    T7_DEFAULT_ANALYTES,
    T7_SHARED_ANALYTES,
)
from aquacontam.benchmark._utils import (
    _safe_mode,
    accumulate_spatial_metadata,
    fit_with_val,
    get_proba,
    get_system_coordinates,
)
from aquacontam.benchmark.metrics import compute_classification_metrics
from aquacontam.benchmark.registry import TaskResult, register_task
from aquacontam.benchmark.transfer import _align_features
from aquacontam.features.assembly import (
    aggregate_to_system_level,
    assemble_feature_matrix,
    drop_leakage_columns,
)
from aquacontam.models.base import BaseModel
from aquacontam.preprocessing.splits import assign_epa_region, geographic_split

logger = logging.getLogger(__name__)


def _prepare_temporal_splits(
    ucmr3_data: pd.DataFrame,
    ucmr5_data: pd.DataFrame,
    analyte: str,
    *feature_dfs: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    drop_na_threshold: float = 0.5,
    exclude_features: list[str] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Prepare temporal train/val/test splits for UCMR3→UCMR5 prediction.

    Parameters
    ----------
    ucmr3_data : pd.DataFrame
        UCMR3-era water quality data (training source).
    ucmr5_data : pd.DataFrame
        UCMR5-era water quality data (test target).
    analyte : str
        Analyte to predict.
    *feature_dfs : pd.DataFrame
        Feature DataFrames indexed by pwsid.
    categorical_columns : list[str] or None
        Columns to one-hot encode.
    drop_na_threshold : float
        Threshold for dropping high-NaN columns.
    exclude_features : list[str] or None
        Column names to exclude from feature matrices (e.g. for ablation).

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.Series]]
        Keys: ``"train"``, ``"val"`` (from UCMR3), ``"test"`` (all UCMR5),
        ``"test_persistent"`` (systems in both eras),
        ``"test_new"`` (UCMR5-only systems).
    """
    # --- UCMR3: train/val via geographic split ---
    ucmr3_with_region = assign_epa_region(ucmr3_data)
    ucmr3_agg = aggregate_to_system_level(ucmr3_with_region, analyte, target="detected")

    if ucmr3_agg.empty:
        empty_xy = (pd.DataFrame(), pd.Series(dtype=float))
        return {
            "train": empty_xy,
            "val": empty_xy,
            "test": empty_xy,
            "test_persistent": empty_xy,
            "test_new": empty_xy,
        }

    ucmr3_regions = ucmr3_with_region.groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
    ucmr3_agg = ucmr3_agg.join(ucmr3_regions)
    ucmr3_reset = ucmr3_agg.reset_index()

    train_sys, val_sys, _test_sys = geographic_split(ucmr3_reset)

    # Assemble train/val from UCMR3
    splits: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    impute_stats = None

    for split_name, split_df in [("train", train_sys), ("val", val_sys)]:
        if split_df.empty:
            splits[split_name] = (pd.DataFrame(), pd.Series(dtype=float))
            continue

        split_indexed = split_df.set_index("pwsid")
        if "epa_region" in split_indexed.columns:
            split_indexed = split_indexed.drop(columns=["epa_region"])
        split_indexed = drop_leakage_columns(split_indexed)

        X, y, stats = assemble_feature_matrix(
            split_indexed,
            *feature_dfs,
            categorical_columns=categorical_columns,
            drop_na_threshold=drop_na_threshold,
            impute_stats=impute_stats,
        )
        if exclude_features:
            drop_cols = [c for c in X.columns if c in exclude_features]
            if drop_cols:
                logger.info(
                    "Excluding %d feature(s) from %s split: %s",
                    len(drop_cols),
                    split_name,
                    drop_cols,
                )
                X = X.drop(columns=drop_cols)
        if split_name == "train":
            impute_stats = stats
        splits[split_name] = (X, y)

    train_columns = list(splits["train"][0].columns) if not splits["train"][0].empty else []

    # --- UCMR5: test splits ---
    ucmr5_agg = aggregate_to_system_level(ucmr5_data, analyte, target="detected")

    if ucmr5_agg.empty:
        empty_xy = (pd.DataFrame(), pd.Series(dtype=float))
        splits["test"] = empty_xy
        splits["test_persistent"] = empty_xy
        splits["test_new"] = empty_xy
        return splits

    # Partition UCMR5 into persistent (in both) and new (UCMR5-only)
    ucmr3_pwsids = set(ucmr3_agg.index)
    ucmr5_pwsids = set(ucmr5_agg.index)
    persistent_pwsids = ucmr3_pwsids & ucmr5_pwsids
    new_pwsids = ucmr5_pwsids - ucmr3_pwsids

    def _assemble_test_split(
        agg_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if agg_df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        agg_df = drop_leakage_columns(agg_df)
        X, y, _ = assemble_feature_matrix(
            agg_df,
            *feature_dfs,
            categorical_columns=categorical_columns,
            drop_na_threshold=drop_na_threshold,
            impute_stats=impute_stats,
        )
        if exclude_features:
            drop_cols = [c for c in X.columns if c in exclude_features]
            if drop_cols:
                X = X.drop(columns=drop_cols)
        # Align to training columns
        if train_columns:
            X = _align_features(X, train_columns)
        return X, y

    # Full test set (all UCMR5)
    splits["test"] = _assemble_test_split(ucmr5_agg)

    # Persistent sub-split
    persistent_agg = ucmr5_agg.loc[ucmr5_agg.index.isin(persistent_pwsids)]
    splits["test_persistent"] = _assemble_test_split(persistent_agg)

    # New sub-split
    new_agg = ucmr5_agg.loc[ucmr5_agg.index.isin(new_pwsids)]
    splits["test_new"] = _assemble_test_split(new_agg)

    return splits


def diagnose_temporal_shift(
    ucmr3_data: pd.DataFrame,
    ucmr5_data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Diagnose distribution shift between UCMR3 and UCMR5 eras.

    Computes per-analyte detection rate changes, detection limit distribution
    shifts (KS test), and feature distribution changes between eras.

    Parameters
    ----------
    ucmr3_data : pd.DataFrame
        UCMR3-era water quality data.
    ucmr5_data : pd.DataFrame
        UCMR5-era water quality data.
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames for distribution shift analysis.

    Returns
    -------
    dict[str, Any]
        Diagnostic results with keys: ``per_analyte``, ``system_overlap``,
        ``feature_shift``.
    """
    from scipy.stats import ks_2samp

    results: dict[str, Any] = {}

    # Per-analyte analysis
    per_analyte: list[dict[str, Any]] = []
    for analyte in T7_SHARED_ANALYTES:
        u3 = ucmr3_data[ucmr3_data["analyte"] == analyte]
        u5 = ucmr5_data[ucmr5_data["analyte"] == analyte]

        entry: dict[str, Any] = {"analyte": analyte}

        # Detection rates
        if len(u3) > 0:
            entry["ucmr3_detection_rate"] = float((~u3["censored"]).mean())
            entry["ucmr3_n_samples"] = len(u3)
        if len(u5) > 0:
            entry["ucmr5_detection_rate"] = float((~u5["censored"]).mean())
            entry["ucmr5_n_samples"] = len(u5)

        # Detection limit distribution shift (KS test)
        dl3 = u3["detection_limit"].dropna()
        dl5 = u5["detection_limit"].dropna()
        if len(dl3) > 5 and len(dl5) > 5:
            ks_stat, ks_p = ks_2samp(dl3, dl5)
            entry["dl_ks_statistic"] = float(ks_stat)
            entry["dl_ks_pvalue"] = float(ks_p)
            entry["ucmr3_mean_dl"] = float(dl3.mean())
            entry["ucmr5_mean_dl"] = float(dl5.mean())

        per_analyte.append(entry)

    results["per_analyte"] = per_analyte

    # System overlap statistics
    ucmr3_pwsids = set(ucmr3_data["pwsid"].unique())
    ucmr5_pwsids = set(ucmr5_data["pwsid"].unique())
    persistent = ucmr3_pwsids & ucmr5_pwsids
    results["system_overlap"] = {
        "ucmr3_systems": len(ucmr3_pwsids),
        "ucmr5_systems": len(ucmr5_pwsids),
        "persistent": len(persistent),
        "new_in_ucmr5": len(ucmr5_pwsids - ucmr3_pwsids),
        "dropped_from_ucmr3": len(ucmr3_pwsids - ucmr5_pwsids),
    }

    # Feature distribution shift (Earth Mover's Distance approximation)
    if feature_dfs:
        feature_shift: list[dict[str, Any]] = []
        for df in feature_dfs:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                vals_u3 = df.loc[df.index.isin(ucmr3_pwsids), col].dropna()
                vals_u5 = df.loc[df.index.isin(ucmr5_pwsids), col].dropna()
                if len(vals_u3) > 5 and len(vals_u5) > 5:
                    ks_stat, ks_p = ks_2samp(vals_u3, vals_u5)
                    feature_shift.append(
                        {
                            "feature": col,
                            "ucmr3_mean": float(vals_u3.mean()),
                            "ucmr5_mean": float(vals_u5.mean()),
                            "ucmr3_std": float(vals_u3.std()),
                            "ucmr5_std": float(vals_u5.std()),
                            "ks_statistic": float(ks_stat),
                            "ks_pvalue": float(ks_p),
                        }
                    )
        results["feature_shift"] = feature_shift

    return results


@register_task(
    name="T7",
    description="Temporal prediction UCMR3→UCMR5",
    task_type="classification",
    primary_metric="auprc",
    analytes=T7_DEFAULT_ANALYTES,
)
def run_t7(
    *,
    model: BaseModel,
    ucmr3_data: pd.DataFrame,
    ucmr5_data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Run T7: temporal prediction UCMR3→UCMR5.

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    ucmr3_data : pd.DataFrame
        UCMR3-era water quality data.
    ucmr5_data : pd.DataFrame
        UCMR5-era water quality data.
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    analyte : str, optional
        Which analyte to predict. Must be in ``T7_SHARED_ANALYTES``.
        Defaults to ``"PFOS"``.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.
    exclude_features : list[str], optional
        Column names to exclude from feature matrices.

    Returns
    -------
    TaskResult

    Raises
    ------
    ValueError
        If ``analyte`` is not in ``T7_SHARED_ANALYTES``.
    """
    if feature_dfs is None:
        feature_dfs = []
    if analyte is None:
        analyte = T7_DEFAULT_ANALYTES[0]

    if analyte not in T7_SHARED_ANALYTES:
        raise ValueError(f"Analyte {analyte!r} not in T7_SHARED_ANALYTES: {T7_SHARED_ANALYTES}")

    splits = _prepare_temporal_splits(
        ucmr3_data,
        ucmr5_data,
        analyte,
        *feature_dfs,
        categorical_columns=categorical_columns,
        exclude_features=exclude_features,
    )

    X_train, y_train = splits["train"]

    if X_train.empty:
        logger.warning("Empty UCMR3 training set for T7 (%s)", analyte)
        return TaskResult(
            task_name="T7",
            model_name=model.name,
            metrics={},
            metadata={"analyte": analyte, "error": "empty training set"},
        )

    # Fit with optional validation set
    X_val, y_val = splits.get("val", (pd.DataFrame(), pd.Series(dtype=float)))
    fit_with_val(model, X_train, y_train, X_val, y_val)

    # Coordinates for spatial metadata (from UCMR5 target data)
    sys_coords = get_system_coordinates(ucmr5_data)
    # Also include UCMR3 coordinates for train/val splits
    ucmr3_coords = get_system_coordinates(ucmr3_data)
    combined_coords = pd.concat([ucmr3_coords, sys_coords])
    combined_coords = combined_coords[~combined_coords.index.duplicated(keep="last")]
    spatial_acc: dict[str, list] = {
        "y_true": [],
        "y_prob": [],
        "latitudes": [],
        "longitudes": [],
        "split_labels": [],
    }

    # Evaluate on all splits
    split_metrics: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test", "test_persistent", "test_new"):
        X, y = splits.get(split_name, (pd.DataFrame(), pd.Series(dtype=float)))
        if X.empty:
            continue

        preds = model.predict(X)
        probs = get_proba(model, X)

        split_metrics[split_name] = compute_classification_metrics(
            y, preds, probs, metrics=list(CLASSIFICATION_METRICS)
        )

        accumulate_spatial_metadata(y, probs, X.index, combined_coords, split_name, spatial_acc)

    primary = split_metrics.get("test", split_metrics.get("val", {}))

    # Detection rates per era
    ucmr3_det_rate = float("nan")
    ucmr5_det_rate = float("nan")
    ucmr3_filtered = ucmr3_data[ucmr3_data["analyte"] == analyte]
    ucmr5_filtered = ucmr5_data[ucmr5_data["analyte"] == analyte]
    if len(ucmr3_filtered) > 0:
        ucmr3_det_rate = float((~ucmr3_filtered["censored"]).mean())
    if len(ucmr5_filtered) > 0:
        ucmr5_det_rate = float((~ucmr5_filtered["censored"]).mean())

    # Count persistent vs new
    ucmr3_pwsids = set(ucmr3_data[ucmr3_data["analyte"] == analyte]["pwsid"].unique())
    ucmr5_pwsids = set(ucmr5_data[ucmr5_data["analyte"] == analyte]["pwsid"].unique())

    metadata: dict[str, object] = {
        "analyte": analyte,
        "n_train": len(X_train),
        "n_test": len(splits.get("test", (pd.DataFrame(),))[0]),
        "n_persistent": len(ucmr3_pwsids & ucmr5_pwsids),
        "n_new": len(ucmr5_pwsids - ucmr3_pwsids),
        "ucmr3_detection_rate": ucmr3_det_rate,
        "ucmr5_detection_rate": ucmr5_det_rate,
    }
    if spatial_acc["y_true"]:
        metadata["y_true"] = spatial_acc["y_true"]
        metadata["y_prob"] = spatial_acc["y_prob"]
        metadata["latitudes"] = spatial_acc["latitudes"]
        metadata["longitudes"] = spatial_acc["longitudes"]
        metadata["split_labels"] = spatial_acc["split_labels"]

    return TaskResult(
        task_name="T7",
        model_name=model.name,
        metrics=primary,
        split_metrics=split_metrics,
        metadata=metadata,
    )
