"""Benchmark task definitions — T1 (PFAS detection), T2 (concentration), T4 (heavy metals).

Each task registers itself via ``@register_task`` and implements the
full pipeline: filter → aggregate → split → assemble → fit → evaluate.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam._constants import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    T1_DEFAULT_ANALYTES,
    T2_DEFAULT_ANALYTES,
    T4_DEFAULT_ANALYTES,
)
from aquacontam.benchmark._utils import (
    accumulate_spatial_metadata,
    fit_with_val,
    get_proba,
    get_system_coordinates,
)
from aquacontam.benchmark.metrics import (
    bootstrap_classification_metrics,
    compute_censoring_aware_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    optimal_threshold_analysis,
)
from aquacontam.benchmark.registry import TaskResult, register_task
from aquacontam.features.assembly import (
    aggregate_to_system_level,
    prepare_train_val_test,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# Default detection limit fill value (µg/L) for systems with unknown DL.
# Used as fallback when mean_detection_limit is NaN in censoring metadata
# passed to Deep Tobit and ICP models.  Approximate median PFAS reporting
# limit across data sources.
DEFAULT_DL_FILLNA: float = 0.004


def _run_classification_task(
    model: BaseModel,
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    analyte: str,
    task_name: str,
    categorical_columns: list[str] | None = None,
    target: str = "detected",
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Shared logic for classification benchmark tasks."""
    splits = prepare_train_val_test(
        data,
        analyte,
        *feature_dfs,
        target=target,
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, _y_test = splits["test"]

    if X_train.empty:
        logger.warning("Empty training set for %s / %s", task_name, analyte)
        return TaskResult(
            task_name=task_name,
            model_name=model.name,
            metrics={},
            metadata={"analyte": analyte, "error": "empty training set"},
        )

    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        logger.warning(
            "Single-class target for %s / %s: only class(es) %s in training data",
            task_name,
            analyte,
            unique_classes.tolist(),
        )
        return TaskResult(
            task_name=task_name,
            model_name=model.name,
            metrics={},
            metadata={"analyte": analyte, "error": "single-class target"},
        )

    # Coordinates for spatial metadata (and GNN models)
    sys_coords = get_system_coordinates(data)

    # Build EPSG:5070 coordinate arrays for GNN models.
    # Not all systems have geocoded coordinates, so we subset to the
    # intersection when needed (GNN requires coords for every sample).
    _fit_extra: dict[str, Any] = {}

    # Pass censoring metadata for models that require it (e.g. Deep Tobit).
    # Extract separately from aggregate_to_system_level to avoid leakage
    # (these columns are dropped from the feature matrix by drop_leakage_columns).
    if getattr(model, "requires_censoring_metadata", False):
        sys_agg_cens = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_cens.empty and "any_detected" in sys_agg_cens.columns:
            # Align censoring metadata with train split
            train_cens = sys_agg_cens.reindex(X_train.index)
            _fit_extra["censored"] = (~train_cens["any_detected"].astype(bool)).to_numpy(
                dtype=np.float64,
                na_value=0.0,
            )
            _fit_extra["detection_limits"] = (
                train_cens["mean_detection_limit"]
                .fillna(DEFAULT_DL_FILLNA)
                .to_numpy(dtype=np.float64)
            )
            # Pass actual concentrations so the Tobit loss sees continuous
            # values instead of binary 0/1 labels for classification tasks.
            _fit_extra["y_concentration"] = (
                train_cens["target"].fillna(0.0).to_numpy(dtype=np.float64)
            )

            # Align with val split
            if not X_val.empty:
                val_cens = sys_agg_cens.reindex(X_val.index)
                _fit_extra["censored_val"] = (~val_cens["any_detected"].astype(bool)).to_numpy(
                    dtype=np.float64,
                    na_value=0.0,
                )
                _fit_extra["detection_limits_val"] = (
                    val_cens["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64)
                )
                _fit_extra["y_concentration_val"] = (
                    val_cens["target"].fillna(0.0).to_numpy(dtype=np.float64)
                )
    # Pass confounder targets for ICP models (monitoring-invariant prediction).
    # Extract monitoring intensity metadata BEFORE leakage columns are dropped.
    if getattr(model, "requires_confounder_targets", False):
        sys_agg_conf = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_conf.empty:
            train_conf = sys_agg_conf.reindex(X_train.index)
            # Confounder targets: [log1p(n_samples), mean_detection_limit]
            conf_train = np.column_stack(
                [
                    np.log1p(train_conf["n_samples"].fillna(0).to_numpy(dtype=np.float64)),
                    train_conf["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64),
                ]
            )
            _fit_extra["confounder_targets"] = conf_train

            # EPA region groups for Group DRO
            from aquacontam.preprocessing.splits import assign_epa_region

            df_with_region = assign_epa_region(data)
            sys_regions = df_with_region.groupby("pwsid", observed=True)["epa_region"].first()
            train_regions = sys_regions.reindex(X_train.index).fillna(0).astype(int).to_numpy()
            _fit_extra["groups"] = train_regions

            if not X_val.empty:
                val_conf = sys_agg_conf.reindex(X_val.index)
                conf_val = np.column_stack(
                    [
                        np.log1p(val_conf["n_samples"].fillna(0).to_numpy(dtype=np.float64)),
                        val_conf["mean_detection_limit"]
                        .fillna(DEFAULT_DL_FILLNA)
                        .to_numpy(dtype=np.float64),
                    ]
                )
                _fit_extra["confounder_targets_val"] = conf_val
                val_regions = sys_regions.reindex(X_val.index).fillna(0).astype(int).to_numpy()
                _fit_extra["groups_val"] = val_regions

    _X_fit, _y_fit = X_train, y_train
    _X_val_fit, _y_val_fit = X_val, y_val
    # Only models that consume coordinates (e.g. GNNs) get the coordinate
    # alignment/subsetting. For all other models this block is skipped so they
    # train on the FULL split — subsetting them to coordinate-covered systems
    # needlessly discards training data and can severely depress metrics.
    if (
        getattr(model, "requires_coords", False)
        and sys_coords is not None
        and not sys_coords.empty
    ):
        from pyproj import Transformer

        _proj = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        _all_x, _all_y = _proj.transform(
            sys_coords["longitude"].values, sys_coords["latitude"].values
        )
        _coords_5070 = pd.DataFrame({"x": _all_x, "y": _all_y}, index=sys_coords.index)

        # Align coords with train split
        _train_aligned = _coords_5070.reindex(X_train.index).dropna()
        if len(_train_aligned) == len(X_train):
            # Full coverage — pass coords as-is
            _fit_extra["coords"] = _train_aligned.to_numpy()
        elif len(_train_aligned) >= len(X_train) * 0.5:
            # Partial coverage — subset train data to coordinated systems
            _keep = _train_aligned.index
            _X_fit = X_train.loc[_keep]
            _y_fit = y_train.loc[_keep]
            _fit_extra["coords"] = _train_aligned.to_numpy()
            logger.info(
                "GNN coord subsetting: %d/%d train systems have coordinates",
                len(_keep),
                len(X_train),
            )

        # Align coords with val split
        if "coords" in _fit_extra and not X_val.empty:
            _val_aligned = _coords_5070.reindex(X_val.index).dropna()
            if len(_val_aligned) == len(X_val):
                _fit_extra["coords_val"] = _val_aligned.to_numpy()
            elif len(_val_aligned) >= len(X_val) * 0.5:
                _X_val_fit = X_val.loc[_val_aligned.index]
                _y_val_fit = y_val.loc[_val_aligned.index]
                _fit_extra["coords_val"] = _val_aligned.to_numpy()

    # Fit with optional validation set for early stopping
    fit_with_val(model, _X_fit, _y_fit, _X_val_fit, _y_val_fit, **_fit_extra)
    spatial_acc: dict[str, list] = {
        "y_true": [],
        "y_prob": [],
        "latitudes": [],
        "longitudes": [],
        "split_labels": [],
    }

    # Evaluate per split
    split_metrics: dict[str, dict[str, float]] = {}
    for split_name, (X, y) in splits.items():
        if X.empty:
            continue
        preds = model.predict(X)
        probs = get_proba(model, X)

        split_metrics[split_name] = compute_classification_metrics(
            y, preds, probs, metrics=list(CLASSIFICATION_METRICS)
        )

        accumulate_spatial_metadata(y, probs, X.index, sys_coords, split_name, spatial_acc)

    # Use test metrics as primary (or val if test is empty)
    primary = split_metrics.get("test", split_metrics.get("val", {}))

    # Threshold optimization: find optimal threshold on val, apply to test.
    # Addresses F1=0 under severe class imbalance (e.g. 97% censored).
    threshold_results: dict[str, object] = {}
    if not X_val.empty and len(y_val.unique()) >= 2:
        val_probs = get_proba(model, X_val)
        if val_probs is not None:
            val_thresh = optimal_threshold_analysis(y_val, val_probs, metric="f1")
            threshold_results["val_analysis"] = val_thresh
            opt_t = val_thresh["optimal_threshold"]

            # Apply val-tuned threshold to test set
            if not X_test.empty:
                test_probs_t = get_proba(model, X_test)
                if test_probs_t is not None:
                    test_preds_opt = (np.asarray(test_probs_t) >= opt_t).astype(int)
                    from sklearn.metrics import f1_score, precision_score, recall_score

                    y_t = splits["test"][1]
                    primary["f1_optimized"] = float(
                        f1_score(y_t, test_preds_opt, zero_division=0.0)
                    )
                    primary["precision_optimized"] = float(
                        precision_score(y_t, test_preds_opt, zero_division=0.0)
                    )
                    primary["recall_optimized"] = float(
                        recall_score(y_t, test_preds_opt, zero_division=0.0)
                    )
                    primary["optimal_threshold"] = opt_t

    # Bootstrap confidence intervals on test set
    bootstrap_ci: dict[str, dict[str, float]] = {}
    test_data = splits.get("test", (pd.DataFrame(), pd.Series(dtype=float)))
    X_test_ci, y_test_ci = test_data
    if not X_test_ci.empty and len(y_test_ci.unique()) >= 2:
        test_preds = model.predict(X_test_ci)
        test_probs = get_proba(model, X_test_ci)
        bootstrap_ci = bootstrap_classification_metrics(
            y_test_ci,
            test_preds,
            test_probs,
            metrics=["auroc", "auprc"],
            n_bootstrap=1000,
            confidence=0.95,
            seed=42,
        )

    metadata: dict[str, object] = {
        "analyte": analyte,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "bootstrap_ci": bootstrap_ci,
        "threshold_optimization": threshold_results,
        "X_test": X_test,
    }
    if spatial_acc["y_true"]:
        metadata["y_true"] = spatial_acc["y_true"]
        metadata["y_prob"] = spatial_acc["y_prob"]
        metadata["latitudes"] = spatial_acc["latitudes"]
        metadata["longitudes"] = spatial_acc["longitudes"]
        metadata["split_labels"] = spatial_acc["split_labels"]

    return TaskResult(
        task_name=task_name,
        model_name=model.name,
        metrics=primary,
        split_metrics=split_metrics,
        metadata=metadata,
    )


def _run_regression_task(
    model: BaseModel,
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    analyte: str,
    task_name: str,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Shared logic for regression benchmark tasks."""
    splits = prepare_train_val_test(
        data,
        analyte,
        *feature_dfs,
        target="max_concentration",
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, _y_test = splits["test"]

    if X_train.empty:
        logger.warning("Empty training set for %s / %s", task_name, analyte)
        return TaskResult(
            task_name=task_name,
            model_name=model.name,
            metrics={},
            metadata={"analyte": analyte, "error": "empty training set"},
        )

    # Pass censoring metadata for models that require it (e.g. Deep Tobit).
    _reg_extra: dict[str, Any] = {}
    if getattr(model, "requires_censoring_metadata", False):
        sys_agg_cens = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_cens.empty and "any_detected" in sys_agg_cens.columns:
            train_cens = sys_agg_cens.reindex(X_train.index)
            _reg_extra["censored"] = (~train_cens["any_detected"].astype(bool)).to_numpy(
                dtype=np.float64,
                na_value=0.0,
            )
            _reg_extra["detection_limits"] = (
                train_cens["mean_detection_limit"]
                .fillna(DEFAULT_DL_FILLNA)
                .to_numpy(dtype=np.float64)
            )

            if not X_val.empty:
                val_cens = sys_agg_cens.reindex(X_val.index)
                _reg_extra["censored_val"] = (~val_cens["any_detected"].astype(bool)).to_numpy(
                    dtype=np.float64,
                    na_value=0.0,
                )
                _reg_extra["detection_limits_val"] = (
                    val_cens["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64)
                )

    # Pass confounder targets for ICP models
    if getattr(model, "requires_confounder_targets", False):
        sys_agg_conf_r = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_conf_r.empty:
            train_conf_r = sys_agg_conf_r.reindex(X_train.index)
            conf_train_r = np.column_stack(
                [
                    np.log1p(train_conf_r["n_samples"].fillna(0).to_numpy(dtype=np.float64)),
                    train_conf_r["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64),
                ]
            )
            _reg_extra["confounder_targets"] = conf_train_r

            from aquacontam.preprocessing.splits import assign_epa_region

            df_with_region_r = assign_epa_region(data)
            sys_regions_r = df_with_region_r.groupby("pwsid", observed=True)["epa_region"].first()
            _reg_extra["groups"] = (
                sys_regions_r.reindex(X_train.index).fillna(0).astype(int).to_numpy()
            )

            if not X_val.empty:
                val_conf_r = sys_agg_conf_r.reindex(X_val.index)
                conf_val_r = np.column_stack(
                    [
                        np.log1p(val_conf_r["n_samples"].fillna(0).to_numpy(dtype=np.float64)),
                        val_conf_r["mean_detection_limit"]
                        .fillna(DEFAULT_DL_FILLNA)
                        .to_numpy(dtype=np.float64),
                    ]
                )
                _reg_extra["confounder_targets_val"] = conf_val_r
                _reg_extra["groups_val"] = (
                    sys_regions_r.reindex(X_val.index).fillna(0).astype(int).to_numpy()
                )

    fit_with_val(model, X_train, y_train, X_val, y_val, **_reg_extra)

    # Coordinates for spatial metadata
    sys_coords = get_system_coordinates(data)
    spatial_acc: dict[str, list] = {
        "y_true": [],
        "y_prob": [],
        "latitudes": [],
        "longitudes": [],
        "split_labels": [],
    }

    # Evaluate per split
    split_metrics: dict[str, dict[str, float]] = {}
    for split_name, (X, y) in splits.items():
        if X.empty:
            continue
        preds = model.predict(X)
        split_metrics[split_name] = compute_regression_metrics(
            y, preds, metrics=list(REGRESSION_METRICS)
        )

        accumulate_spatial_metadata(y, preds, X.index, sys_coords, split_name, spatial_acc)

    primary = split_metrics.get("test", split_metrics.get("val", {}))

    # Censoring-aware metrics on test set
    censoring_metrics: dict[str, float] = {}
    if not X_test.empty:
        try:
            # Recover censoring info from original data for test systems
            sys_agg = aggregate_to_system_level(data, analyte, target="max_concentration")
            test_pwsids = X_test.index
            test_sys = sys_agg.reindex(test_pwsids).dropna(subset=["target"])
            if not test_sys.empty and "any_detected" in test_sys.columns:
                censored = ~test_sys["any_detected"].astype(bool)
                dl = test_sys["mean_detection_limit"].fillna(0.0)
                y_test_censor = test_sys["target"]
                preds_test = model.predict(X_test.reindex(test_sys.index))
                censoring_metrics = compute_censoring_aware_metrics(
                    y_test_censor, preds_test, censored, dl
                )
                # Merge into primary metrics
                primary.update(censoring_metrics)
        except (KeyError, ValueError, TypeError):
            logger.debug("Could not compute censoring-aware metrics", exc_info=True)

    # Detected-only secondary metrics on test set
    detected_only_metrics: dict[str, float] = {}
    if not X_test.empty:
        try:
            sys_agg_det = aggregate_to_system_level(data, analyte, target="max_concentration")
            test_sys_det = sys_agg_det.reindex(X_test.index).dropna(subset=["target"])
            if not test_sys_det.empty and "any_detected" in test_sys_det.columns:
                detected_mask = test_sys_det["any_detected"].astype(bool)
                detected_sys = test_sys_det[detected_mask]
                if len(detected_sys) > 1:
                    preds_det = model.predict(X_test.reindex(detected_sys.index))
                    detected_only_metrics = compute_regression_metrics(
                        detected_sys["target"], preds_det, metrics=list(REGRESSION_METRICS)
                    )
        except (KeyError, ValueError, IndexError, TypeError):
            logger.debug("Could not compute detected-only metrics", exc_info=True)

    metadata: dict[str, object] = {
        "analyte": analyte,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "censoring_metrics": censoring_metrics,
        "detected_only_metrics": detected_only_metrics,
        "X_test": X_test,
    }
    if spatial_acc["y_true"]:
        metadata["y_true"] = spatial_acc["y_true"]
        metadata["y_prob"] = spatial_acc["y_prob"]
        metadata["latitudes"] = spatial_acc["latitudes"]
        metadata["longitudes"] = spatial_acc["longitudes"]
        metadata["split_labels"] = spatial_acc["split_labels"]

    return TaskResult(
        task_name=task_name,
        model_name=model.name,
        metrics=primary,
        split_metrics=split_metrics,
        metadata=metadata,
    )


@register_task(
    name="T1",
    description="Binary PFAS detection prediction",
    task_type="classification",
    primary_metric="auprc",
    analytes=T1_DEFAULT_ANALYTES,
)
def run_t1(
    *,
    model: BaseModel,
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Run T1: binary PFAS detection classification.

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    data : pd.DataFrame
        Water quality DataFrame (standard schema).
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    analyte : str, optional
        Which PFAS analyte to predict. Defaults to ``"PFOS"``.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.
    split_strategy : str
        ``"geographic"`` (default) or ``"random"`` for ablation.
    exclude_features : list[str], optional
        Column names to exclude from feature matrices.

    Returns
    -------
    TaskResult
    """
    if feature_dfs is None:
        feature_dfs = []
    if analyte is None:
        analyte = T1_DEFAULT_ANALYTES[0]

    return _run_classification_task(
        model,
        data,
        feature_dfs,
        analyte,
        "T1",
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )


@register_task(
    name="T2",
    description="PFAS concentration regression",
    task_type="regression",
    primary_metric="rmse",
    analytes=T2_DEFAULT_ANALYTES,
)
def run_t2(
    *,
    model: BaseModel,
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Run T2: PFAS concentration regression.

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    data : pd.DataFrame
        Water quality DataFrame (standard schema).
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    analyte : str, optional
        Which PFAS analyte to predict. Defaults to ``"PFOS"``.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.
    split_strategy : str
        ``"geographic"`` (default) or ``"random"`` for ablation.
    exclude_features : list[str], optional
        Column names to exclude from feature matrices.

    Returns
    -------
    TaskResult
    """
    if feature_dfs is None:
        feature_dfs = []
    if analyte is None:
        analyte = T2_DEFAULT_ANALYTES[0]

    return _run_regression_task(
        model,
        data,
        feature_dfs,
        analyte,
        "T2",
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )


@register_task(
    name="T4",
    description="Heavy metal action level exceedance prediction",
    task_type="classification",
    primary_metric="auprc",
    analytes=T4_DEFAULT_ANALYTES,
)
def run_t4(
    *,
    model: BaseModel,
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskResult:
    """Run T4: heavy metal action level exceedance classification.

    Predicts whether a water system's 90th-percentile lead/copper
    concentration exceeds the EPA action level (15 µg/L for lead,
    1300 µg/L for copper).

    Parameters
    ----------
    model : BaseModel
        Model to train and evaluate.
    data : pd.DataFrame
        Water quality DataFrame (standard schema).
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    analyte : str, optional
        Which heavy metal to predict. Defaults to ``"lead"``.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.
    split_strategy : str
        ``"geographic"`` (default) or ``"random"`` for ablation.
    exclude_features : list[str], optional
        Column names to exclude from feature matrices.

    Returns
    -------
    TaskResult
    """
    if feature_dfs is None:
        feature_dfs = []
    if analyte is None:
        analyte = T4_DEFAULT_ANALYTES[0]

    return _run_classification_task(
        model,
        data,
        feature_dfs,
        analyte,
        "T4",
        categorical_columns=categorical_columns,
        target="action_level",
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )
