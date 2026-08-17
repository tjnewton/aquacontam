"""External validation with state databases and WQP data.

Trains models on UCMR5 geographic train split and evaluates on independent
state database systems and WQP monitoring locations to measure cross-dataset
generalization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.benchmark.metrics import (
    bootstrap_classification_metrics,
    compute_classification_metrics,
    delong_test_vs_chance,
)
from aquacontam.benchmark.transfer import _align_features
from aquacontam.features.assembly import (
    aggregate_to_system_level,
    assemble_feature_matrix,
    drop_leakage_columns,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# State DB -> EPA region mapping for overlap detection
_STATE_REGIONS: dict[str, int] = {
    "mi_mpart": 5,  # Michigan -> Region 5 (TRAIN)
    "ca_geotracker": 9,  # California -> Region 9 (TEST)
    "nj_dep": 2,  # New Jersey -> Region 2 (VAL)
    "nc_deq": 4,  # North Carolina -> Region 4 (TRAIN)
    "mo_dnr": 7,  # Missouri -> Region 7 (VAL)
    "tx_tceq": 6,  # Texas -> Region 6 (TRAIN)
    "oh_epa": 5,  # Ohio -> Region 5 (TRAIN)
    "wa_doh": 10,  # Washington -> Region 10 (TEST)
}


def run_external_validation(
    model: BaseModel,
    ucmr5_data: pd.DataFrame,
    state_dfs: dict[str, pd.DataFrame],
    feature_dfs: list[pd.DataFrame],
    analyte: str = "PFOS",
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Run external validation on state databases.

    Parameters
    ----------
    model : BaseModel
        Fitted model (trained on UCMR5 geographic train split).
    ucmr5_data : pd.DataFrame
        UCMR5 water quality data (for identifying overlapping PWSIDs).
    state_dfs : dict[str, pd.DataFrame]
        State database DataFrames keyed by source name.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.
    analyte : str
        Analyte to predict.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Per-state metrics and metadata.
    """
    ucmr5_pwsids = set(ucmr5_data["pwsid"].unique()) if not ucmr5_data.empty else set()
    results: dict[str, Any] = {}

    # Get training feature columns from fitted model
    train_cols: list[str] | None = None
    if hasattr(model, "feature_names_in_"):
        train_cols = list(model.feature_names_in_)
    elif hasattr(model, "_model") and hasattr(model._model, "feature_names_in_"):
        train_cols = list(model._model.feature_names_in_)

    for state_name, state_df in state_dfs.items():
        if state_df.empty:
            continue

        try:
            # Filter to analyte
            state_analyte = state_df[state_df["analyte"] == analyte]
            if state_analyte.empty:
                logger.info("No %s data in %s -- skipping", analyte, state_name)
                continue

            # Aggregate to system level
            sys_targets = aggregate_to_system_level(state_analyte, analyte, target="detected")
            sys_targets = drop_leakage_columns(sys_targets)

            if sys_targets.empty:
                logger.info("No system-level data for %s in %s", analyte, state_name)
                continue

            if len(sys_targets["target"].unique()) < 2:
                # All systems have same class (e.g. 100% detection).
                # Record metadata but skip metric computation.
                n_sys = len(sys_targets)
                det_rate = float((sys_targets["target"] == 1).mean())
                region = _STATE_REGIONS.get(state_name)
                results[state_name] = {
                    "metrics": {},
                    "n_systems": n_sys,
                    "n_overlap_excluded": 0,
                    "epa_region": region,
                    "detection_rate": det_rate,
                    "note": "single_class",
                }
                logger.info(
                    "Single-class data for %s in %s: %d systems, detection_rate=%.3f",
                    analyte,
                    state_name,
                    n_sys,
                    det_rate,
                )
                continue

            # Exclude PWSIDs that appear in UCMR5 (for train-region states)
            region = _STATE_REGIONS.get(state_name)
            state_pwsids = set(sys_targets.index)
            overlap = state_pwsids & ucmr5_pwsids
            if region in (1, 3, 4, 5, 6):  # TRAIN regions
                # Exclude overlapping systems
                sys_targets = sys_targets.loc[~sys_targets.index.isin(overlap)]
                if sys_targets.empty:
                    continue

            # Assemble features
            X, y, _ = assemble_feature_matrix(sys_targets, *feature_dfs)

            if X.empty or len(y.unique()) < 2:
                continue

            # Align features to training columns
            if train_cols:
                X = _align_features(X, train_cols)

            # Predict and evaluate
            preds = model.predict(X)
            probs = None
            try:
                probs_raw = model.predict_proba(X)
                if probs_raw.ndim == 2 and probs_raw.shape[1] == 2:
                    probs = probs_raw[:, 1]
            except NotImplementedError:
                pass

            metrics = compute_classification_metrics(
                y, preds, probs, metrics=["auroc", "auprc", "f1", "accuracy"]
            )

            # Bootstrap 95% CIs for AUROC and AUPRC
            boot_ci = bootstrap_classification_metrics(
                y,
                preds,
                probs,
                metrics=["auroc", "auprc"],
                n_bootstrap=1000,
                seed=seed,
            )

            # DeLong one-sample test: AUROC vs chance (0.5)
            chance_test = None
            if probs is not None:
                chance_test = delong_test_vs_chance(y, probs)

            results[state_name] = {
                "metrics": metrics,
                "bootstrap_ci": boot_ci,
                "chance_test": chance_test,
                "n_systems": len(X),
                "n_overlap_excluded": len(overlap) if region in (1, 3, 4, 5, 6) else 0,
                "epa_region": region,
                "detection_rate": float((y == 1).mean()),
            }
            logger.info(
                "External validation %s: n=%d, AUROC=%.4f, AUPRC=%.4f",
                state_name,
                len(X),
                metrics.get("auroc", float("nan")),
                metrics.get("auprc", float("nan")),
            )
        except (KeyError, ValueError, OSError, TypeError):
            logger.warning("Failed external validation for %s", state_name, exc_info=True)

    return results


# EPA region -> state FIPS mapping for WQP filtering
_TRAIN_REGIONS = {1, 3, 4, 5, 6}

# State abbreviation -> EPA region (subset for CONUS)
_STATE_EPA_REGION: dict[str, int] = {
    "CT": 1,
    "ME": 1,
    "MA": 1,
    "NH": 1,
    "RI": 1,
    "VT": 1,
    "NJ": 2,
    "NY": 2,
    "PR": 2,
    "VI": 2,
    "DE": 3,
    "DC": 3,
    "MD": 3,
    "PA": 3,
    "VA": 3,
    "WV": 3,
    "AL": 4,
    "FL": 4,
    "GA": 4,
    "KY": 4,
    "MS": 4,
    "NC": 4,
    "SC": 4,
    "TN": 4,
    "IL": 5,
    "IN": 5,
    "MI": 5,
    "MN": 5,
    "OH": 5,
    "WI": 5,
    "AR": 6,
    "LA": 6,
    "NM": 6,
    "OK": 6,
    "TX": 6,
    "IA": 7,
    "KS": 7,
    "MO": 7,
    "NE": 7,
    "CO": 8,
    "MT": 8,
    "ND": 8,
    "SD": 8,
    "UT": 8,
    "WY": 8,
    "AZ": 9,
    "CA": 9,
    "HI": 9,
    "NV": 9,
    "AK": 10,
    "ID": 10,
    "OR": 10,
    "WA": 10,
}


def run_wqp_regional_validation(
    model: BaseModel,
    wqp_data: pd.DataFrame,
    ucmr5_data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    analyte: str = "PFOS",
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate model on WQP data from training-region states.

    Filters WQP data to training EPA regions (1, 3, 4, 5, 6), excludes
    any IDs that overlap with UCMR5 training data, and evaluates the
    fitted model per region.

    Parameters
    ----------
    model : BaseModel
        Fitted model (trained on UCMR5 geographic train split).
    wqp_data : pd.DataFrame
        WQP water quality data.
    ucmr5_data : pd.DataFrame
        UCMR5 data for overlap detection.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.
    analyte : str
        Analyte to predict.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Per-region metrics, overall metrics, and metadata. Includes a
        caveat field noting WQP sites are ambient monitoring locations,
        not public water systems.
    """
    if wqp_data.empty:
        return {"error": "empty_wqp_data"}

    ucmr5_pwsids = set(ucmr5_data["pwsid"].unique()) if not ucmr5_data.empty else set()

    # Get training feature columns from fitted model
    train_cols: list[str] | None = None
    if hasattr(model, "feature_names_in_"):
        train_cols = list(model.feature_names_in_)
    elif hasattr(model, "_model") and hasattr(model._model, "feature_names_in_"):
        train_cols = list(model._model.feature_names_in_)

    # Filter to analyte
    wqp_analyte = (
        wqp_data[wqp_data["analyte"] == analyte] if "analyte" in wqp_data.columns else wqp_data
    )
    if wqp_analyte.empty:
        return {"error": f"no_{analyte}_in_wqp"}

    # Determine EPA region per WQP system.  WQP IDs are synthetic
    # (``WQP_XXXXXXX``), so the 2-char prefix is "WQ" — not a state code.
    # Use ``assign_epa_region`` which falls back to coordinate lookup.
    from aquacontam.preprocessing.splits import assign_epa_region

    wqp_analyte = wqp_analyte.copy()
    if "state_code" in wqp_analyte.columns:
        wqp_analyte["epa_region"] = wqp_analyte["state_code"].map(_STATE_EPA_REGION)
    else:
        wqp_analyte = assign_epa_region(wqp_analyte)

    # Filter to training regions only
    train_mask = wqp_analyte["epa_region"].isin(_TRAIN_REGIONS)
    wqp_train = wqp_analyte[train_mask]

    if wqp_train.empty:
        return {"error": "no_training_region_data"}

    # Aggregate to system level
    try:
        sys_targets = aggregate_to_system_level(wqp_train, analyte, target="detected")
    except (KeyError, ValueError, TypeError):
        logger.warning("Failed to aggregate WQP data", exc_info=True)
        return {"error": "aggregation_failed"}

    sys_targets = drop_leakage_columns(sys_targets)

    # Exclude overlapping systems
    overlap = set(sys_targets.index) & ucmr5_pwsids
    sys_targets = sys_targets.loc[~sys_targets.index.isin(overlap)]

    if sys_targets.empty or len(sys_targets["target"].unique()) < 2:
        return {
            "error": "insufficient_data_after_overlap_removal",
            "n_overlap_excluded": len(overlap),
            "n_remaining": len(sys_targets),
        }

    # Assemble features
    X, y, _ = assemble_feature_matrix(sys_targets, *feature_dfs)

    if X.empty or len(y.unique()) < 2:
        return {"error": "insufficient_features"}

    # Align features
    if train_cols:
        X = _align_features(X, train_cols)

    # Overall evaluation
    preds = model.predict(X)
    probs = None
    try:
        probs_raw = model.predict_proba(X)
        if probs_raw.ndim == 2 and probs_raw.shape[1] == 2:
            probs = probs_raw[:, 1]
    except NotImplementedError:
        pass

    overall_metrics = compute_classification_metrics(
        y, preds, probs, metrics=["auroc", "auprc", "f1", "accuracy"]
    )

    # Per-region breakdown — deduplicate to one region per system
    region_col = (
        wqp_analyte.drop_duplicates(subset="pwsid")
        .set_index("pwsid")["epa_region"]
        .reindex(X.index)
    )
    per_region: dict[int, dict[str, Any]] = {}
    for region in sorted(_TRAIN_REGIONS):
        region_mask = region_col == region
        if region_mask.sum() < 5:
            continue
        y_r = y[region_mask]
        if len(y_r.unique()) < 2:
            per_region[region] = {
                "n_systems": int(region_mask.sum()),
                "detection_rate": float((y_r == 1).mean()),
                "note": "single_class",
            }
            continue

        probs_r = probs[np.asarray(region_mask)] if probs is not None else None
        preds_r = preds[np.asarray(region_mask)]
        region_metrics = compute_classification_metrics(
            y_r, preds_r, probs_r, metrics=["auroc", "auprc"]
        )
        per_region[region] = {
            "metrics": region_metrics,
            "n_systems": int(region_mask.sum()),
            "detection_rate": float((y_r == 1).mean()),
        }

    result: dict[str, Any] = {
        "overall_metrics": overall_metrics,
        "per_region": per_region,
        "n_systems": len(X),
        "n_overlap_excluded": len(overlap),
        "detection_rate": float((y == 1).mean()),
        "caveat": (
            "WQP monitoring locations are ambient sites (groundwater wells, "
            "surface water), not public water systems. This validation shows "
            "whether environmental risk patterns generalize, not PWS-specific prediction."
        ),
    }

    logger.info(
        "WQP training-region validation: n=%d, AUROC=%.4f, AUPRC=%.4f",
        len(X),
        overall_metrics.get("auroc", float("nan")),
        overall_metrics.get("auprc", float("nan")),
    )
    return result
