"""Task-specific data-prep helpers for HPO drivers.

These functions extract the (X_train, y_train, X_val, y_val, fit_extra)
tuple needed to train one trial. Built so the HPO loop can prep data
*once per (task, model)* and reuse it across N trials, avoiding redundant
aggregation work.

Each helper accepts ``model_cls`` (or a template instance) and inspects
``requires_censoring_metadata`` / ``requires_confounder_targets`` flags to
populate the appropriate ``fit_extra`` keys.

T1, T2, T4, T6 are implemented (T6 loads the USGS NGA arsenic source and tunes
on the public-supply train/val splits). T3, T5, T7 raise ``NotImplementedError``
with a clear message — they can be wired in later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aquacontam._constants import (
    T1_DEFAULT_ANALYTES,
    T2_DEFAULT_ANALYTES,
    T4_DEFAULT_ANALYTES,
)
from aquacontam.benchmark._utils import get_system_coordinates
from aquacontam.features.assembly import (
    aggregate_to_system_level,
    prepare_train_val_test,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# Same default DL fallback as ``benchmark.tasks._run_classification_task``.
DEFAULT_DL_FILLNA: float = 0.004


@dataclass
class TaskData:
    """Train/val tensors plus model-specific fit_extra for one (task, model)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    fit_extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: extract censoring + confounder metadata for a given model class
# ---------------------------------------------------------------------------


def _extract_classification_extras(
    template: BaseModel,
    data: pd.DataFrame,
    analyte: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> dict[str, Any]:
    """Censoring + confounder + GNN coords metadata. Mirrors tasks.py logic."""
    fit_extra: dict[str, Any] = {}

    if getattr(template, "requires_censoring_metadata", False):
        sys_agg_cens = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_cens.empty and "any_detected" in sys_agg_cens.columns:
            train_cens = sys_agg_cens.reindex(X_train.index)
            fit_extra["censored"] = (~train_cens["any_detected"].astype(bool)).to_numpy(
                dtype=np.float64,
                na_value=0.0,
            )
            fit_extra["detection_limits"] = (
                train_cens["mean_detection_limit"]
                .fillna(DEFAULT_DL_FILLNA)
                .to_numpy(dtype=np.float64)
            )
            fit_extra["y_concentration"] = (
                train_cens["target"].fillna(0.0).to_numpy(dtype=np.float64)
            )
            if not X_val.empty:
                val_cens = sys_agg_cens.reindex(X_val.index)
                fit_extra["censored_val"] = (~val_cens["any_detected"].astype(bool)).to_numpy(
                    dtype=np.float64,
                    na_value=0.0,
                )
                fit_extra["detection_limits_val"] = (
                    val_cens["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64)
                )
                fit_extra["y_concentration_val"] = (
                    val_cens["target"].fillna(0.0).to_numpy(dtype=np.float64)
                )

    if getattr(template, "requires_confounder_targets", False):
        sys_agg_conf = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_conf.empty:
            train_conf = sys_agg_conf.reindex(X_train.index)
            conf_train = np.column_stack(
                [
                    np.log1p(train_conf["n_samples"].fillna(0).to_numpy(dtype=np.float64)),
                    train_conf["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64),
                ]
            )
            fit_extra["confounder_targets"] = conf_train

            from aquacontam.preprocessing.splits import assign_epa_region

            df_with_region = assign_epa_region(data)
            sys_regions = df_with_region.groupby("pwsid", observed=True)["epa_region"].first()
            fit_extra["groups"] = (
                sys_regions.reindex(X_train.index).fillna(0).astype(int).to_numpy()
            )

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
                fit_extra["confounder_targets_val"] = conf_val
                fit_extra["groups_val"] = (
                    sys_regions.reindex(X_val.index).fillna(0).astype(int).to_numpy()
                )

    return fit_extra


def _add_gnn_coords(
    fit_extra: dict[str, Any],
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """If coordinate coverage is good, attach EPSG:5070 coords to fit_extra.

    May subset the train/val matrices if coverage is partial (>= 50%).
    Returns the (possibly subset) train/val matrices.
    """
    sys_coords = get_system_coordinates(data)
    if sys_coords is None or sys_coords.empty:
        return X_train, y_train, X_val, y_val

    from pyproj import Transformer

    proj = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    all_x, all_y = proj.transform(sys_coords["longitude"].values, sys_coords["latitude"].values)
    coords_5070 = pd.DataFrame({"x": all_x, "y": all_y}, index=sys_coords.index)

    train_aligned = coords_5070.reindex(X_train.index).dropna()
    if len(train_aligned) == len(X_train):
        fit_extra["coords"] = train_aligned.to_numpy()
    elif len(train_aligned) >= len(X_train) * 0.5:
        keep = train_aligned.index
        X_train = X_train.loc[keep]
        y_train = y_train.loc[keep]
        fit_extra["coords"] = train_aligned.to_numpy()

    if "coords" in fit_extra and not X_val.empty:
        val_aligned = coords_5070.reindex(X_val.index).dropna()
        if len(val_aligned) == len(X_val):
            fit_extra["coords_val"] = val_aligned.to_numpy()
        elif len(val_aligned) >= len(X_val) * 0.5:
            X_val = X_val.loc[val_aligned.index]
            y_val = y_val.loc[val_aligned.index]
            fit_extra["coords_val"] = val_aligned.to_numpy()

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Public prep helpers
# ---------------------------------------------------------------------------


def prepare_t1_data(
    model_cls: type[BaseModel],
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskData | None:
    """Prep T1 (binary detection) train/val data for HPO.

    Returns ``None`` if the prep fails (empty train, single-class target).
    """
    if analyte is None:
        analyte = T1_DEFAULT_ANALYTES[0]

    splits = prepare_train_val_test(
        data,
        analyte,
        *feature_dfs,
        target="detected",
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    if X_train.empty:
        logger.warning("T1 prep: empty training set for analyte=%s", analyte)
        return None
    if y_train.nunique() < 2:
        logger.warning("T1 prep: single-class target for analyte=%s", analyte)
        return None

    template = model_cls(config={})
    fit_extra = _extract_classification_extras(template, data, analyte, X_train, X_val)
    X_train, y_train, X_val, y_val = _add_gnn_coords(
        fit_extra, data, X_train, y_train, X_val, y_val
    )

    return TaskData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_extra=fit_extra,
        metadata={"task": "T1", "analyte": analyte},
    )


def prepare_t2_data(
    model_cls: type[BaseModel],
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskData | None:
    """Prep T2 (concentration regression) train/val data for HPO."""
    if analyte is None:
        analyte = T2_DEFAULT_ANALYTES[0]

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

    if X_train.empty:
        logger.warning("T2 prep: empty training set for analyte=%s", analyte)
        return None

    template = model_cls(config={})
    fit_extra: dict[str, Any] = {}

    # Censoring metadata (Deep Tobit, AFT, XGBoost-AFT, ZIT, Hurdle).
    if getattr(template, "requires_censoring_metadata", False):
        sys_agg_cens = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_cens.empty and "any_detected" in sys_agg_cens.columns:
            train_cens = sys_agg_cens.reindex(X_train.index)
            fit_extra["censored"] = (~train_cens["any_detected"].astype(bool)).to_numpy(
                dtype=np.float64,
                na_value=0.0,
            )
            fit_extra["detection_limits"] = (
                train_cens["mean_detection_limit"]
                .fillna(DEFAULT_DL_FILLNA)
                .to_numpy(dtype=np.float64)
            )
            if not X_val.empty:
                val_cens = sys_agg_cens.reindex(X_val.index)
                fit_extra["censored_val"] = (~val_cens["any_detected"].astype(bool)).to_numpy(
                    dtype=np.float64,
                    na_value=0.0,
                )
                fit_extra["detection_limits_val"] = (
                    val_cens["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64)
                )

    # Confounder targets (ICP regressors).
    if getattr(template, "requires_confounder_targets", False):
        sys_agg_conf = aggregate_to_system_level(data, analyte, target="max_concentration")
        if not sys_agg_conf.empty:
            train_conf = sys_agg_conf.reindex(X_train.index)
            conf_train = np.column_stack(
                [
                    np.log1p(train_conf["n_samples"].fillna(0).to_numpy(dtype=np.float64)),
                    train_conf["mean_detection_limit"]
                    .fillna(DEFAULT_DL_FILLNA)
                    .to_numpy(dtype=np.float64),
                ]
            )
            fit_extra["confounder_targets"] = conf_train

            from aquacontam.preprocessing.splits import assign_epa_region

            df_with_region = assign_epa_region(data)
            sys_regions = df_with_region.groupby("pwsid", observed=True)["epa_region"].first()
            fit_extra["groups"] = (
                sys_regions.reindex(X_train.index).fillna(0).astype(int).to_numpy()
            )

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
                fit_extra["confounder_targets_val"] = conf_val
                fit_extra["groups_val"] = (
                    sys_regions.reindex(X_val.index).fillna(0).astype(int).to_numpy()
                )

    return TaskData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_extra=fit_extra,
        metadata={"task": "T2", "analyte": analyte},
    )


def prepare_t4_data(
    model_cls: type[BaseModel],
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskData | None:
    """Prep T4 (heavy-metal action-level exceedance classification) data for HPO.

    T4 reuses the T1 classification flow but with a different default analyte.
    """
    if analyte is None:
        analyte = T4_DEFAULT_ANALYTES[0]

    splits = prepare_train_val_test(
        data,
        analyte,
        *feature_dfs,
        target="action_level",
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features,
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    if X_train.empty:
        logger.warning("T4 prep: empty training set for analyte=%s", analyte)
        return None
    if y_train.nunique() < 2:
        logger.warning("T4 prep: single-class target for analyte=%s", analyte)
        return None

    template = model_cls(config={})
    fit_extra = _extract_classification_extras(template, data, analyte, X_train, X_val)
    X_train, y_train, X_val, y_val = _add_gnn_coords(
        fit_extra, data, X_train, y_train, X_val, y_val
    )

    return TaskData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_extra=fit_extra,
        metadata={"task": "T4", "analyte": analyte},
    )


def prepare_t3_data(*args: Any, **kwargs: Any) -> TaskData | None:
    """Prep T3 (multilabel multi-PFAS) data — not yet implemented for HPO.

    T3 needs a multilabel y matrix wrapper; deferred until the T1/T2/T4 HPO
    path is validated. See ``benchmark/multilabel.py:run_t3``.
    """
    raise NotImplementedError(
        "T3 HPO prep is not yet implemented. T1/T2/T4 HPO is supported. "
        "T3 multilabel prep requires factoring out from benchmark/multilabel.py."
    )


def prepare_t5_data(*args: Any, **kwargs: Any) -> TaskData | None:
    """Prep T5 (cross-contam transfer) data — not yet implemented for HPO."""
    raise NotImplementedError("T5 HPO prep is not yet implemented. See benchmark/transfer.py.")


def prepare_t6_data(
    model_cls: type[BaseModel],
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    analyte: str | None = None,
    categorical_columns: list[str] | None = None,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
) -> TaskData | None:
    """Prep T6 (arsenic public-supply → domestic transfer) data for HPO.

    Unlike T1/T2/T4, T6 does not use the pipeline ``data``/``feature_dfs`` — it
    loads the USGS NGA arsenic source (``data/nga_arsenic.py``) and builds its own
    environmental features. HPO tunes on the **public-supply** train/val splits
    (arsenic MCL-exceedance, env-only); the domestic holdout stays the downstream
    eval (no tune-on-test leakage). Returns ``None`` when the NGA data is absent
    (e.g. in CI) so the task is skipped gracefully. Imports are local to avoid
    import cycles.
    """
    from pathlib import Path

    from aquacontam.benchmark.private_wells import T6_EXCLUDE_FEATURES

    if analyte is None:
        analyte = "arsenic"

    raw_dir = Path("data") / "raw"
    try:
        from aquacontam.data.nga_arsenic import load_nga_arsenic
        from aquacontam.pipeline.t6_arsenic import build_nga_feature_dfs

        pub = load_nga_arsenic(raw_dir, water_use="Public supply")
        dom = load_nga_arsenic(raw_dir, water_use="Domestic")
    except (FileNotFoundError, OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.warning("T6 prep: NGA arsenic data unavailable (%s) — skipping", exc)
        return None
    if pub.empty:
        logger.warning("T6 prep: no public-supply arsenic wells — skipping")
        return None

    # Build features for pub+dom so the ``features_nga`` cache stays consistent
    # with the standalone runner (pipeline/t6_arsenic.py), which uses both.
    all_wells = pd.concat([pub, dom], ignore_index=True)
    nga_feats = build_nga_feature_dfs(Path("data"), all_wells)

    splits = prepare_train_val_test(
        pub,
        analyte,
        *nga_feats,
        target="action_level",
        categorical_columns=categorical_columns,
        split_strategy=split_strategy,
        exclude_features=exclude_features or list(T6_EXCLUDE_FEATURES),
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits.get("val", (pd.DataFrame(), pd.Series(dtype=float)))

    if X_train.empty:
        logger.warning("T6 prep: empty training set")
        return None
    if y_train.nunique() < 2:
        logger.warning("T6 prep: single-class target")
        return None

    template = model_cls(config={})
    fit_extra = _extract_classification_extras(template, pub, analyte, X_train, X_val)
    X_train, y_train, X_val, y_val = _add_gnn_coords(
        fit_extra, pub, X_train, y_train, X_val, y_val
    )

    return TaskData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fit_extra=fit_extra,
        metadata={"task": "T6", "analyte": analyte},
    )


def prepare_t7_data(*args: Any, **kwargs: Any) -> TaskData | None:
    """Prep T7 (UCMR3→UCMR5 temporal) data — not yet implemented for HPO."""
    raise NotImplementedError("T7 HPO prep is not yet implemented. See benchmark/temporal.py.")


PREP_FUNCTIONS: dict[str, Any] = {
    "T1": prepare_t1_data,
    "T2": prepare_t2_data,
    "T3": prepare_t3_data,
    "T4": prepare_t4_data,
    "T5": prepare_t5_data,
    "T6": prepare_t6_data,
    "T7": prepare_t7_data,
}
