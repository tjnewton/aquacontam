"""Feature assembly with split-aware imputation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def assemble_with_split_imputation(
    sys_targets: Any,
    feature_dfs: list[Any],
    *,
    wq_df: Any = None,
    train_regions: tuple[int, ...] = (1, 3, 4, 5, 6),
    val_regions: tuple[int, ...] = (2, 7),
    test_regions: tuple[int, ...] = (8, 9, 10),
    drop_na_threshold: float = 0.5,
) -> tuple[Any, Any, Any, Any]:
    """Assemble features with train-only imputation to prevent leakage.

    Unlike calling ``assemble_feature_matrix()`` on the full dataset (which
    computes imputation medians from all data including test), this helper
    splits first, computes imputation stats on train only, then applies them
    to val/test before concatenating back.

    Parameters
    ----------
    sys_targets : pd.DataFrame
        System-level targets (from ``aggregate_to_system_level`` with leakage
        columns already dropped). Must have ``epa_region`` column.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by ``pwsid``.
    wq_df : pd.DataFrame or None
        Sample-level water-quality DataFrame (with ``latitude``/``longitude``).
        When provided and ``sys_targets`` lacks an ``epa_region`` column, EPA
        regions are derived from ``wq_df`` so the coordinate-based fallback in
        ``assign_epa_region`` can rescue systems whose PWSID prefix is not a
        state code (e.g. WQP synthetic IDs). Without it, region assignment on
        already-aggregated targets silently drops those geocoded systems —
        matching the canonical ``prepare_train_val_test`` test population.
    train_regions, val_regions, test_regions : tuple[int, ...]
        EPA regions for each split.
    drop_na_threshold : float
        Threshold for dropping high-NaN columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, dict, pd.Series]
        ``(X, y, impute_stats, regions)`` — combined feature matrix, targets,
        train imputation stats, and region series aligned with X index.
    """
    import pandas as pd

    from aquacontam.features.assembly import assemble_feature_matrix
    from aquacontam.preprocessing.splits import assign_epa_region

    if "epa_region" not in sys_targets.columns:
        if wq_df is not None:
            # Coordinate-rescued region assignment (mirrors prepare_train_val_test):
            # assign regions on the sample-level wq_df, which HAS lat/lon, then
            # aggregate the most-common region per system. This rescues systems
            # with non-state-code PWSIDs via the coordinate fallback in
            # assign_epa_region — without it, ~600 geocoded T1 systems (WQP
            # synthetic IDs) get region=NaN and are silently dropped.
            from aquacontam.features.assembly import _safe_mode

            dfr = assign_epa_region(wq_df)
            sys_region = dfr.groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
            sys_targets = sys_targets.copy()
            sys_targets["epa_region"] = sys_targets.index.map(sys_region)
        else:
            tmp = assign_epa_region(sys_targets.reset_index()).set_index("pwsid")
            sys_targets = sys_targets.copy()
            sys_targets["epa_region"] = tmp["epa_region"]

    regions = pd.to_numeric(sys_targets["epa_region"], errors="coerce")
    train_mask = regions.isin(train_regions)
    val_mask = regions.isin(val_regions)
    test_mask = regions.isin(test_regions)

    train_tgt = sys_targets.loc[train_mask].drop(columns=["epa_region"])
    val_tgt = sys_targets.loc[val_mask].drop(columns=["epa_region"])
    test_tgt = sys_targets.loc[test_mask].drop(columns=["epa_region"])

    if train_tgt.empty:
        logger.warning("Train split is empty — returning empty feature matrix")
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        return empty_X, empty_y, {}, regions

    # Imputation stats are computed from train split ONLY (impute_stats=None
    # triggers median computation inside assemble_feature_matrix). Val/test
    # splits reuse these stats to prevent information leakage.
    X_train, y_train, impute_stats = assemble_feature_matrix(
        train_tgt, *feature_dfs, drop_na_threshold=drop_na_threshold
    )
    parts_X = [X_train]
    parts_y = [y_train]

    for split_tgt in (val_tgt, test_tgt):
        if not split_tgt.empty:
            X_s, y_s, _ = assemble_feature_matrix(
                split_tgt, *feature_dfs, impute_stats=impute_stats
            )
            # Align columns with training set — per-split get_dummies can
            # produce different dummy column sets for rare categories.
            # Reindex drops unseen columns and fills missing dummies with 0.
            X_s = X_s.reindex(columns=X_train.columns, fill_value=0)
            parts_X.append(X_s)
            parts_y.append(y_s)

    X = pd.concat(parts_X)
    y = pd.concat(parts_y)

    return X, y, impute_stats, regions.reindex(X.index)
