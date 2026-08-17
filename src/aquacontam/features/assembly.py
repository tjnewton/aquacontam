"""Feature matrix assembly — bridge data loaders and model training.

Aggregates long-form water quality samples to system-level rows,
builds GeoDataFrames from coordinates, and assembles feature matrices
with proper imputation (train-set stats only applied to val/test).
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from aquacontam._constants import CRS_STORAGE
from aquacontam.preprocessing.splits import assign_epa_region, geographic_split, random_split

logger = logging.getLogger(__name__)

# Columns derived from monitoring results that would leak the target variable.
# These are produced by aggregate_to_system_level() and must be dropped before
# passing system-level data into assemble_feature_matrix().
_LEAKAGE_COLUMNS = ("any_detected", "detection_rate", "max_concentration")

# Feature names/patterns that encode the data-source / monitoring PROCESS rather
# than the environment. Pass as ``exclude_features=PROVENANCE_FREE_EXCLUDE`` (via
# ``reproduce.py --provenance-free``) to obtain an honest "environmental-signal"
# model. The ``*_nan`` one-hot missingness indicators (produced by
# ``pd.get_dummies(dummy_na=True)``) in particular leak which data source a system
# came from -- e.g. state PFAS programs lack the SDWIS owner_type/source_water_type
# metadata, so ``owner_type_nan`` becomes a near-perfect source proxy and was the
# highest-magnitude SHAP feature in the with-provenance model.
#
# NOTE: the categorical *values* ``source_water_type_GW``/``source_water_type_SW``
# (groundwater vs surface water -- hydrology) and ``system_type_*`` (community vs
# non-transient -- exposure population) are deliberately RETAINED: they carry
# genuine environmental/exposure signal, not data-source provenance.
PROVENANCE_FREE_EXCLUDE: tuple[str, ...] = (
    "n_samples",  # monitoring intensity
    "mean_detection_limit",  # monitoring artifact
    "population_served",  # administrative / system size
    "log_population_served",
    "*_nan",  # one-hot missingness indicators -> data-source provenance
)


def resolve_excluded_columns(columns: Iterable[str], exclude_features: Iterable[str]) -> list[str]:
    """Expand exact names and glob patterns against actual feature columns.

    An entry containing a glob metacharacter (``*``, ``?``, ``[``) is matched
    against ``columns`` via :func:`fnmatch.filter` (e.g. ``"*_nan"`` matches every
    one-hot missingness indicator); any other entry is treated as an exact name.
    Order-preserving and de-duplicated. Backward compatible with plain name lists
    (e.g. ``["n_samples", "mean_detection_limit"]``).

    Parameters
    ----------
    columns : Iterable[str]
        The actual feature-matrix column names.
    exclude_features : Iterable[str]
        Exact names and/or glob patterns to exclude.

    Returns
    -------
    list[str]
        The subset of ``columns`` selected for exclusion, in column order.
    """
    cols = list(columns)
    drop: set[str] = set()
    for pat in exclude_features:
        if any(ch in pat for ch in "*?["):
            drop.update(fnmatch.filter(cols, pat))
        elif pat in cols:
            drop.add(pat)
    return [c for c in cols if c in drop]


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target-leaking columns from a system-level DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        System-level DataFrame (e.g. from ``aggregate_to_system_level``).

    Returns
    -------
    pd.DataFrame
        DataFrame with leakage columns removed (if present).
    """
    to_drop = [c for c in _LEAKAGE_COLUMNS if c in df.columns]
    if to_drop:
        logger.info("Dropping target-leakage columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def aggregate_to_system_level(
    df: pd.DataFrame,
    analyte: str,
    *,
    target: str = "detected",
    deduplicate_strategy: str | None = None,
) -> pd.DataFrame:
    """Aggregate long-form samples to one row per water system.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame with columns: ``pwsid``, ``analyte``,
        ``concentration``, ``censored``, ``detection_limit``.
    analyte : str
        Analyte to filter to before aggregating.
    target : str
        Target variable type:
        - ``"detected"``: binary — any detection for this system.
        - ``"max_concentration"``: continuous — maximum measured value.
        - ``"action_level"``: binary — max concentration >= EPA action level
          (only for analytes in ``ACTION_LEVELS_UGL``: lead, copper, arsenic).
    deduplicate_strategy : str or None
        If not None, deduplicate samples before aggregation using
        ``preprocessing.cleaning.deduplicate_samples()``. Valid strategies:
        ``"max"``, ``"mean"``, ``"first"``. Default ``None`` preserves
        backward-compatible behavior (no deduplication).

    Returns
    -------
    pd.DataFrame
        One row per ``pwsid`` with columns: target, ``n_samples``,
        ``detection_rate``, ``mean_detection_limit``, ``max_concentration``,
        ``any_detected``. WARNING: ``any_detected``, ``detection_rate`` and
        ``max_concentration`` are derived from the monitoring results and leak
        the target — drop them with :func:`drop_leakage_columns` before passing
        the frame to :func:`assemble_feature_matrix` or any model training.

    Raises
    ------
    ValueError
        If ``target`` is not recognized or required columns are missing.
    """
    valid_targets = ("detected", "max_concentration", "action_level")
    if target not in valid_targets:
        raise ValueError(f"target must be one of {valid_targets}, got {target!r}")

    if deduplicate_strategy is not None:
        valid_strategies = ("max", "mean", "first")
        if deduplicate_strategy not in valid_strategies:
            raise ValueError(
                f"deduplicate_strategy must be one of {valid_strategies}, "
                f"got {deduplicate_strategy!r}"
            )

    required = {"pwsid", "analyte", "concentration", "censored", "detection_limit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter to analyte
    filtered = df[df["analyte"] == analyte].copy()

    # Optionally deduplicate before aggregation
    if deduplicate_strategy is not None and not filtered.empty:
        from aquacontam.preprocessing.cleaning import deduplicate_samples

        n_before = len(filtered)
        filtered = deduplicate_samples(filtered, strategy=deduplicate_strategy)
        n_removed = n_before - len(filtered)
        if n_removed > 0:
            logger.info(
                "Deduplication removed %d rows (%s strategy) before aggregation",
                n_removed,
                deduplicate_strategy,
            )
    if filtered.empty:
        logger.warning("No samples found for analyte %r", analyte)
        empty: pd.DataFrame = pd.DataFrame(
            columns=[
                "pwsid",
                "target",
                "n_samples",
                "detection_rate",
                "mean_detection_limit",
                "max_concentration",
                "any_detected",
            ]
        ).set_index("pwsid")
        return empty

    # Aggregate per system
    grouped = filtered.groupby("pwsid", observed=True)

    agg = pd.DataFrame(
        {
            "max_concentration": grouped["concentration"].max(),
            "any_detected": grouped["censored"].apply(lambda s: bool((~s).any())),
            "detection_rate": grouped["censored"].apply(lambda s: float((~s).mean())),
            "n_samples": grouped["concentration"].count(),
            "mean_detection_limit": grouped["detection_limit"].mean(),
        }
    )

    # Set target column
    if target == "detected":
        agg["target"] = agg["any_detected"].astype(int)
    elif target == "action_level":
        from aquacontam._constants import ACTION_LEVELS_UGL

        threshold = ACTION_LEVELS_UGL.get(analyte)
        if threshold is None:
            raise ValueError(f"No action level defined for analyte {analyte!r}")
        agg["target"] = (agg["max_concentration"] >= threshold).astype(int)
    else:
        agg["target"] = agg["max_concentration"]

    return cast(pd.DataFrame, agg)


def build_system_geodataframe(
    df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of water systems with point geometries.

    Computes median lat/lon per ``pwsid`` and filters out systems
    without valid coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pwsid``, ``latitude``, ``longitude``.

    Returns
    -------
    gpd.GeoDataFrame
        Point geometry in EPSG:4326, indexed by ``pwsid``.
    """
    required = {"pwsid", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Median coordinates per system
    coords = df.groupby("pwsid", observed=True)[["latitude", "longitude"]].median()

    # Filter out systems without coordinates
    valid = coords.dropna(subset=["latitude", "longitude"])
    n_dropped = len(coords) - len(valid)
    if n_dropped:
        logger.info(
            "%d systems dropped (missing coordinates), %d retained",
            n_dropped,
            len(valid),
        )

    if valid.empty:
        return gpd.GeoDataFrame(
            {"pwsid": pd.Series(dtype=str)},
            geometry=[],
            crs=CRS_STORAGE,
        ).set_index("pwsid")

    geometry = gpd.points_from_xy(valid["longitude"], valid["latitude"])
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=geometry,
        crs=CRS_STORAGE,
    )
    # pwsid is already the index from groupby
    return gdf


def assemble_feature_matrix(
    system_targets: pd.DataFrame,
    *feature_dfs: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    drop_na_threshold: float = 0.5,
    impute_stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Assemble feature matrix from system targets and feature DataFrames.

    Parameters
    ----------
    system_targets : pd.DataFrame
        Indexed by ``pwsid`` with a ``target`` column (from
        ``aggregate_to_system_level``). Should be **leakage-free**: call
        :func:`drop_leakage_columns` on it first to remove ``any_detected`` /
        ``detection_rate`` / ``max_concentration`` (derived from the monitoring
        results) — otherwise they enter the feature matrix and leak the target.
        A warning is logged if they are present.
    *feature_dfs : pd.DataFrame
        Feature DataFrames indexed by ``pwsid``.
    categorical_columns : list[str] or None
        Columns to one-hot encode.
    drop_na_threshold : float
        Drop columns with more than this fraction of NaN values.
    impute_stats : dict[str, Any] or None
        Pre-computed imputation medians (from training set). If None,
        medians are computed from the input data. Also carries
        ``_high_na_cols`` and ``_train_columns`` for cross-split alignment.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, dict[str, Any]]
        ``(X, y, impute_stats)`` — features, target, and imputation
        stats to reuse for val/test.
    """
    if not 0.0 <= drop_na_threshold <= 1.0:
        raise ValueError(f"drop_na_threshold must be in [0, 1], got {drop_na_threshold}")

    if categorical_columns is None:
        categorical_columns = []

    # Guard the public-API footgun: leakage columns left in ``system_targets`` would
    # flow in as features and inflate performance. The pipeline drops them upstream;
    # warn (don't auto-drop, to preserve reproducibility) so external callers notice.
    leakage_present = [c for c in _LEAKAGE_COLUMNS if c in system_targets.columns]
    if leakage_present:
        logger.warning(
            "assemble_feature_matrix received target-leakage columns %s in system_targets; "
            "these would be used as FEATURES and can inflate performance. Call "
            "drop_leakage_columns() on the aggregated frame before assembling.",
            leakage_present,
        )

    y = system_targets["target"].copy()

    # Left-join feature DataFrames
    X = system_targets.drop(columns=["target"]).copy()
    for fdf in feature_dfs:
        overlapping = set(X.columns) & set(fdf.columns)
        if overlapping:
            raise ValueError(
                f"Duplicate columns in feature DataFrames: {sorted(overlapping)}. "
                "Rename or drop conflicting columns before assembly."
            )
        X = X.join(fdf, how="left")

    # One-hot encode categorical columns (explicit + auto-detected string/object)
    cat_cols_present = [c for c in categorical_columns if c in X.columns]
    auto_cat = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    all_cat = sorted(set(cat_cols_present + auto_cat))
    if all_cat:
        X = pd.get_dummies(X, columns=all_cat, dummy_na=True)
        # Deduplicate columns — pd.get_dummies(dummy_na=True) can produce
        # duplicate "<col>_nan" names when the original data contains the
        # string "nan" alongside real NaN values.
        if not X.columns.is_unique:
            X = X.loc[:, ~X.columns.duplicated()]
        # Pandas 2.x get_dummies returns bool dtype; convert to int8 so that
        # concat/reindex with fill_value=0 doesn't produce object dtype.
        bool_cols = X.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype("int8")

    # Drop columns with too many NaNs
    if impute_stats is not None and "_high_na_cols" in impute_stats:
        # Val/test: use training set's high-NaN column list
        high_na_cols = [c for c in impute_stats["_high_na_cols"] if c in X.columns]
    else:
        # Training: compute from data
        na_fractions = X.isna().mean()
        high_na_cols = na_fractions[na_fractions > drop_na_threshold].index.tolist()
    if high_na_cols:
        log_fn = logger.warning if len(high_na_cols) > 5 else logger.info
        log_fn(
            "Dropping %d columns with >%.0f%% NaN: %s",
            len(high_na_cols),
            drop_na_threshold * 100,
            high_na_cols,
        )
        X = X.drop(columns=high_na_cols)

    # Impute remaining NaNs with medians
    if impute_stats is None:
        impute_stats = {"_high_na_cols": high_na_cols}
        for col in X.columns:
            if X[col].isna().any():
                median_val = float(X[col].median())
                impute_stats[col] = median_val
    else:
        impute_stats = dict(impute_stats)

    fillna_dict = {
        col: fill_val
        for col, fill_val in impute_stats.items()
        if isinstance(fill_val, (int, float)) and col in X.columns
    }
    if fillna_dict:
        X = X.fillna(fillna_dict)

    # Validate no non-numeric columns remain
    non_numeric_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Non-numeric columns found after encoding: {non_numeric_cols}. "
            "Add them to `categorical_columns` or remove before assembly."
        )

    # Ensure all numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill any remaining NaNs (from to_numeric coercion) with 0
    X = X.fillna(0.0)

    # Align columns to training set
    if "_train_columns" not in impute_stats:
        # Training: record final column order
        impute_stats["_train_columns"] = X.columns.tolist()
    else:
        # Val/test: align to training columns
        train_cols = impute_stats["_train_columns"]
        for col in train_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[[c for c in train_cols if c in X.columns]]

    return X, y, impute_stats


def _safe_mode(x: pd.Series) -> Any:
    """Return the mode of a Series, or NaN if empty."""
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else np.nan


def derive_system_epa_regions(wq_df: pd.DataFrame, index: pd.Index | None = None) -> pd.Series:
    """Per-system EPA region derived from sample-level rows with coordinate rescue.

    Region assignment must run on the sample-level frame (which carries
    latitude/longitude) so that ``assign_epa_region``'s coordinate fallback can
    fire for systems whose PWSID prefix is not a state code (e.g. WQP synthetic
    IDs). Assigning regions on already-aggregated targets silently drops those
    systems from geographic splits.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Sample-level water quality frame with ``pwsid`` and coordinates.
    index : pd.Index or None
        If given, return regions aligned to this pwsid index (NaN where
        unknown); otherwise return the per-system region Series.

    Returns
    -------
    pd.Series
        Numeric EPA region per pwsid.
    """
    sys_region = (
        assign_epa_region(wq_df).groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
    )
    sys_region = pd.to_numeric(sys_region, errors="coerce")
    if index is not None:
        aligned: pd.Series = sys_region.reindex(index)
        aligned.name = "epa_region"
        return aligned
    return sys_region


def prepare_train_val_test(
    df: pd.DataFrame,
    analyte: str,
    *feature_dfs: pd.DataFrame,
    target: str = "detected",
    categorical_columns: list[str] | None = None,
    drop_na_threshold: float = 0.5,
    split_strategy: str = "geographic",
    exclude_features: list[str] | None = None,
    deduplicate_strategy: str | None = None,
    holdout_df: pd.DataFrame | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """End-to-end pipeline: aggregate, split, assemble features.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame with standard schema.
    analyte : str
        Analyte to model.
    *feature_dfs : pd.DataFrame
        Feature DataFrames indexed by ``pwsid``.
    target : str
        Target type (``"detected"``, ``"max_concentration"``, or
        ``"action_level"``).
    categorical_columns : list[str] or None
        Columns to one-hot encode.
    drop_na_threshold : float
        Threshold for dropping high-NaN columns.
    split_strategy : str
        Split strategy: ``"geographic"`` (default) for EPA region-based
        stratification, or ``"random"`` for ablation comparison.
    exclude_features : list[str] or None
        Column names to exclude from the feature matrix after assembly.
        Used by ``--no-monitoring-features`` to drop monitoring intensity
        columns (e.g. ``n_samples``, ``mean_detection_limit``).
    deduplicate_strategy : str or None
        If not None, deduplicate samples before aggregation. See
        ``aggregate_to_system_level`` for valid strategies.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.Series]]
        Keys ``"train"``, ``"val"``, ``"test"`` mapping to
        ``(X, y)`` tuples.
    """
    # Assign EPA region for splitting
    df_with_region = assign_epa_region(df)

    # Aggregate to system level
    system_agg = aggregate_to_system_level(
        df_with_region, analyte, target=target, deduplicate_strategy=deduplicate_strategy
    )

    # Merge EPA region info (most common region per system)
    system_regions = df_with_region.groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
    system_agg = system_agg.join(system_regions)

    # Split: geographic (default) or random (ablation)
    system_reset = system_agg.reset_index()
    if split_strategy == "random":
        train_sys, val_sys, test_sys = random_split(system_reset)
    else:
        train_sys, val_sys, test_sys = geographic_split(system_reset)

    # Re-index by pwsid
    splits: dict[str, Any] = {}
    impute_stats = None

    for split_name, split_df in [("train", train_sys), ("val", val_sys), ("test", test_sys)]:
        if split_df.empty:
            splits[split_name] = (pd.DataFrame(), pd.Series(dtype=float))
            continue

        split_indexed = split_df.set_index("pwsid")
        # Drop region column (not a feature)
        if "epa_region" in split_indexed.columns:
            split_indexed = split_indexed.drop(columns=["epa_region"])
        # Drop columns derived from monitoring results (target leakage)
        split_indexed = drop_leakage_columns(split_indexed)

        X, y, stats = assemble_feature_matrix(
            split_indexed,
            *feature_dfs,
            categorical_columns=categorical_columns,
            drop_na_threshold=drop_na_threshold,
            impute_stats=impute_stats,
        )

        # Drop excluded features (monitoring/provenance columns; supports globs
        # like "*_nan" via resolve_excluded_columns).
        if exclude_features:
            drop_cols = resolve_excluded_columns(X.columns, exclude_features)
            if drop_cols:
                logger.info(
                    "Excluding %d feature(s) from %s split: %s",
                    len(drop_cols),
                    split_name,
                    drop_cols,
                )
                X = X.drop(columns=drop_cols)

        # Use train stats for val/test imputation
        if split_name == "train":
            impute_stats = stats

        splits[split_name] = (X, y)

    # Optional externally-supplied holdout population (e.g. an independent set of
    # wells for zero-shot transfer). Aggregated and assembled with the *train*
    # imputation/columns so it lands in the training feature space, then returned
    # under the ``"holdout"`` key. Used by T6 (public-supply → domestic arsenic).
    if holdout_df is not None and not holdout_df.empty and impute_stats is not None:
        holdout_agg = aggregate_to_system_level(
            holdout_df, analyte, target=target, deduplicate_strategy=deduplicate_strategy
        )
        if not holdout_agg.empty:
            holdout_indexed = drop_leakage_columns(holdout_agg)
            Xh, yh, _ = assemble_feature_matrix(
                holdout_indexed,
                *feature_dfs,
                categorical_columns=categorical_columns,
                drop_na_threshold=drop_na_threshold,
                impute_stats=impute_stats,
            )
            if exclude_features:
                Xh = Xh.drop(columns=resolve_excluded_columns(Xh.columns, exclude_features))
            splits["holdout"] = (Xh, yh)

    return splits
