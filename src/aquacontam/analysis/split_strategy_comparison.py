"""Split strategy comparison: random vs geographic evaluation ablation.

Compares AquaContam's full feature set against a baseline proximity/demographic
feature set comparable to those used in prior PFAS prediction studies, and
quantifies the metric inflation from random vs geographic splitting.

Includes both the original single-model ablation (``run_split_ablation``)
and a multi-model 2x2 comparison matrix (``run_split_comparison_matrix``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def build_baseline_feature_set(feature_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Build a baseline proximity/demographic feature set from available features.

    Selects only proximity and population-related features, which approximate
    the feature set used in prior PFAS prediction studies.

    Parameters
    ----------
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.

    Returns
    -------
    pd.DataFrame
        Subset of features approximating the baseline feature set.
    """
    baseline_prefixes = (
        "nearest_",
        "count_",
        "pct_",  # population demographics
    )
    selected_dfs = []
    for df in feature_dfs:
        cols = [c for c in df.columns if any(c.startswith(p) for p in baseline_prefixes)]
        if cols:
            selected_dfs.append(df[cols])

    if not selected_dfs:
        return cast(pd.DataFrame, pd.DataFrame())

    return cast(pd.DataFrame, pd.DataFrame(pd.concat(selected_dfs, axis=1)))


def build_fernandez_feature_set(feature_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Build a feature set replicating Fernandez et al. (2023) categories.

    Selects only the feature types used in Fernandez et al.'s RF/GBT models:
    facility proximity, buffer counts, land use fractions, and basic demographics.
    Explicitly excludes monitoring intensity, system characteristics, EJScreen
    indices, and aquifer type features.

    Parameters
    ----------
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.

    Returns
    -------
    pd.DataFrame
        Subset of features matching Fernandez et al.'s categories.
    """
    # Specific proximity features (distance to contamination sources)
    proximity_features = {
        "nearest_industrial_km",
        "nearest_military_km",
        "nearest_wwtp_km",
        "nearest_airport_km",
        "nearest_landfill_km",
    }
    # Buffer count features
    buffer_features = {
        "count_industrial_5km",
        "count_military_5km",
        "count_wwtp_5km",
        "count_airport_5km",
        "count_landfill_5km",
    }
    # Land use prefixes (NLCD fractions)
    land_use_prefixes = (
        "pct_developed_",
        "pct_agriculture_",
        "pct_forest_",
        "pct_wetland_",
    )
    # Demographics (basic, not EJScreen indices)
    demographic_features = {
        "pct_people_of_color",
        "pct_low_income",
    }

    exact_matches = proximity_features | buffer_features | demographic_features

    selected_dfs = []
    for df in feature_dfs:
        cols = [
            c
            for c in df.columns
            if c in exact_matches or any(c.startswith(p) for p in land_use_prefixes)
        ]
        if cols:
            selected_dfs.append(df[cols])

    if not selected_dfs:
        return cast(pd.DataFrame, pd.DataFrame())

    return cast(pd.DataFrame, pd.DataFrame(pd.concat(selected_dfs, axis=1)))


def run_split_ablation(
    wq_df: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    analyte: str = "PFOS",
    seed: int = 42,
) -> dict[str, Any]:
    """Run split strategy ablation: random split vs geographic split.

    Trains XGBoost on baseline-subset features with random 5-fold CV and
    geographic splits, then computes the metric inflation ratio.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Water quality data.
    feature_dfs : list[pd.DataFrame]
        Full feature DataFrames.
    analyte : str
        Analyte to predict (default "PFOS").
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Results with keys: ``random_split_metrics``, ``geographic_split_metrics``,
        ``inflation_ratio``, ``baseline_features_used``.
    """
    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        derive_system_epa_regions,
        drop_leakage_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    # Build baseline feature subset
    baseline_features = build_baseline_feature_set(feature_dfs)
    if baseline_features.empty:
        logger.warning("No baseline-compatible features found")
        return {"error": "no_baseline_features"}

    # Prepare data
    sys_targets = aggregate_to_system_level(wq_df, analyte, target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _ = assemble_feature_matrix(sys_targets, baseline_features)

    if X.empty or len(y.unique()) < 2:
        return {"error": "insufficient_data"}

    baseline_features_used = list(X.columns)

    # --- Geographic split ---
    # Regions must come from the sample-level frame so the coordinate fallback
    # can rescue systems with non-state PWSID prefixes (WQP synthetic IDs).
    regions = derive_system_epa_regions(wq_df, X.index)
    train_idx = X.index[regions.isin((1, 3, 4, 5, 6))]
    val_idx = X.index[regions.isin((2, 7))]
    test_idx = X.index[regions.isin((8, 9, 10))]

    geo_metrics: dict[str, float] = {}
    if len(train_idx) >= 10 and len(test_idx) >= 10:
        model = XGBoostClassifier(config={"n_estimators": 100, "random_state": seed})
        model.fit(X.loc[train_idx], y.loc[train_idx], X_val=X.loc[val_idx], y_val=y.loc[val_idx])
        preds = model.predict(X.loc[test_idx])
        probs = model.predict_proba(X.loc[test_idx])
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        geo_metrics = compute_classification_metrics(
            y.loc[test_idx], preds, probs, metrics=["auroc", "auprc"]
        )

    # --- Random split ---
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    random_fold_metrics: list[dict[str, float]] = []
    for train_i, test_i in skf.split(X, y):
        model = XGBoostClassifier(config={"n_estimators": 100, "random_state": seed})
        model.fit(X.iloc[train_i], y.iloc[train_i])
        preds = model.predict(X.iloc[test_i])
        probs = model.predict_proba(X.iloc[test_i])
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        fold_m = compute_classification_metrics(
            y.iloc[test_i], preds, probs, metrics=["auroc", "auprc"]
        )
        random_fold_metrics.append(fold_m)

    random_metrics = {
        k: float(np.mean([m.get(k, 0.0) for m in random_fold_metrics])) for k in ["auroc", "auprc"]
    }

    # Inflation ratio
    inflation_ratio: dict[str, float] = {}
    for k in ["auroc", "auprc"]:
        geo_val = geo_metrics.get(k, 0.0)
        rand_val = random_metrics.get(k, 0.0)
        if geo_val > 0:
            inflation_ratio[k] = rand_val / geo_val
        else:
            inflation_ratio[k] = float("nan")

    return {
        "random_split_metrics": random_metrics,
        "geographic_split_metrics": geo_metrics,
        "inflation_ratio": inflation_ratio,
        "baseline_features_used": baseline_features_used,
        "n_features": len(baseline_features_used),
    }


@dataclass
class SplitComparisonResult:
    """Result from one cell of the 2x2 split comparison matrix."""

    model_name: str
    feature_set: str  # "baseline" or "full"
    split_strategy: str  # "random", "geographic", or "random_sizematched"
    metrics: dict[str, float] = field(default_factory=dict)
    bootstrap_ci: dict[str, dict[str, float]] | None = None
    fold_metrics: list[dict[str, float]] | None = None  # random splits only
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        d: dict[str, Any] = {
            "model_name": self.model_name,
            "feature_set": self.feature_set,
            "split_strategy": self.split_strategy,
            "metrics": self.metrics,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }
        if self.bootstrap_ci is not None:
            d["bootstrap_ci"] = self.bootstrap_ci
        if self.fold_metrics is not None:
            d["fold_metrics"] = self.fold_metrics
        return d


def _get_default_model_specs() -> list[tuple[str, type[BaseModel], dict[str, Any]]]:
    """Return default model specs for multi-model comparison.

    Returns
    -------
    list[tuple[str, type[BaseModel], dict]]
        Each element is (name, model_class, config_dict).
    """
    from aquacontam.models.xgboost import XGBoostClassifier

    specs: list[tuple[str, type[BaseModel], dict[str, Any]]] = [
        ("XGBoost", XGBoostClassifier, {"n_estimators": 100}),
    ]

    try:
        from aquacontam.models.random_forest import RandomForestClassifier as RFCls

        specs.append(("RandomForest", RFCls, {"n_estimators": 100}))
    except ImportError:
        pass

    try:
        from aquacontam.models.logistic import LogisticRegressionClassifier

        specs.append(("LogisticRegression", LogisticRegressionClassifier, {}))
    except ImportError:
        pass

    return specs


def _train_and_evaluate(
    model_cls: type[BaseModel],
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    seed: int = 42,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    n_bootstrap: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]] | None, np.ndarray | None]:
    """Train model and compute metrics with optional bootstrap CIs.

    Returns (metrics, bootstrap_ci_or_None, probs_or_None).
    """
    from aquacontam.benchmark.metrics import (
        bootstrap_classification_metrics,
        compute_classification_metrics,
    )

    cfg = {**config, "random_state": seed}
    model = model_cls(config=cfg)
    fit_kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["X_val"] = X_val
        fit_kwargs["y_val"] = y_val
    model.fit(X_train, y_train, **fit_kwargs)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    if probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs, metrics=["auroc", "auprc"])

    boot_ci = None
    if n_bootstrap > 0:
        boot_ci = bootstrap_classification_metrics(
            y_test,
            preds,
            probs,
            metrics=["auroc", "auprc"],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    return metrics, boot_ci, probs


def run_split_comparison_matrix(
    wq_df: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    model_specs: list[tuple[str, type[BaseModel], dict[str, Any]]] | None = None,
    analyte: str = "PFOS",
    seed: int = 42,
    n_bootstrap: int = 1000,
    compute_delong: bool = True,
    size_matched: bool = False,
) -> dict[str, Any]:
    """Comparison matrix: {Fernandez, Baseline, Full} x {Random, Geographic}.

    Trains multiple models on up to three feature sets with both split strategies.
    Computes bootstrap CIs for geographic splits and per-fold metrics for
    random 5-fold CV. The Fernandez feature set replicates the feature categories
    used by Fernandez et al. (2023) for direct comparison.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Water quality data.
    feature_dfs : list[pd.DataFrame]
        Full feature DataFrames indexed by pwsid.
    model_specs : list[tuple[str, type, dict]] or None
        Model specifications as ``(name, class, config)``. Defaults to
        XGBoost, Random Forest, and Logistic Regression.
    analyte : str
        Analyte to predict (default "PFOS").
    seed : int
        Random seed.
    n_bootstrap : int
        Number of bootstrap iterations for CIs (default 1000).
    compute_delong : bool
        Whether to compute DeLong tests comparing Baseline vs Full features.
    size_matched : bool
        If True, add a third ``"random_sizematched"`` arm per (model, feature
        set): random 5-fold CV with each fold's training set stratified-
        subsampled to the geographic training-set size, isolating the
        evaluation-protocol effect from the training-set-size advantage of
        random splits.

    Returns
    -------
    dict[str, Any]
        Keys: ``"results"`` (list of serialized SplitComparisonResult),
        ``"delong_tests"`` (dict of DeLong p-values per model),
        ``"n_models"``, ``"feature_set_sizes"``.
    """
    from sklearn.model_selection import StratifiedKFold

    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        derive_system_epa_regions,
        drop_leakage_columns,
    )

    if model_specs is None:
        model_specs = _get_default_model_specs()

    # Prepare data
    sys_targets = aggregate_to_system_level(wq_df, analyte, target="detected")
    sys_targets = drop_leakage_columns(sys_targets)

    # Build feature sets
    fernandez_features = build_fernandez_feature_set(feature_dfs)
    baseline_features = build_baseline_feature_set(feature_dfs)
    if baseline_features.empty:
        logger.warning("No baseline-compatible features found")
        return {"error": "no_baseline_features"}

    # Full features = concatenation of all feature_dfs
    full_features = pd.concat(feature_dfs, axis=1) if feature_dfs else pd.DataFrame()

    feature_sets: dict[str, pd.DataFrame] = {}
    if not fernandez_features.empty:
        feature_sets["fernandez"] = fernandez_features
    feature_sets["baseline"] = baseline_features
    feature_sets["full"] = full_features

    # Geographic split regions from the sample-level frame (coordinate rescue
    # for systems with non-state PWSID prefixes; see derive_system_epa_regions).
    sys_region = derive_system_epa_regions(wq_df)

    results_list: list[SplitComparisonResult] = []
    # Store geo probs for DeLong tests: {model_name: {feature_set: probs}}
    geo_probs: dict[str, dict[str, np.ndarray]] = {}
    geo_y_test: pd.Series | None = None

    for fs_name, fs_df in feature_sets.items():
        X, y, _ = assemble_feature_matrix(sys_targets, fs_df)
        if X.empty or len(y.unique()) < 2:
            logger.warning("Insufficient data for feature set %s", fs_name)
            continue

        # Map regions for geo split (sys_region is already numeric)
        regions = sys_region.reindex(X.index)
        train_idx = X.index[regions.isin((1, 3, 4, 5, 6))]
        val_idx = X.index[regions.isin((2, 7))]
        test_idx = X.index[regions.isin((8, 9, 10))]

        for model_name, model_cls, config in model_specs:
            # --- Geographic split ---
            if len(train_idx) >= 10 and len(test_idx) >= 10:
                metrics, boot_ci, probs = _train_and_evaluate(
                    model_cls,
                    config,
                    X.loc[train_idx],
                    y.loc[train_idx],
                    X.loc[test_idx],
                    y.loc[test_idx],
                    seed=seed,
                    X_val=X.loc[val_idx] if len(val_idx) > 0 else None,
                    y_val=y.loc[val_idx] if len(val_idx) > 0 else None,
                    n_bootstrap=n_bootstrap,
                )
                results_list.append(
                    SplitComparisonResult(
                        model_name=model_name,
                        feature_set=fs_name,
                        split_strategy="geographic",
                        metrics=metrics,
                        bootstrap_ci={k: v for k, v in boot_ci.items()} if boot_ci else None,
                        n_train=len(train_idx),
                        n_test=len(test_idx),
                    )
                )
                # Store for DeLong
                if probs is not None:
                    geo_probs.setdefault(model_name, {})[fs_name] = probs
                    if geo_y_test is None:
                        geo_y_test = y.loc[test_idx]

            # --- Random 5-fold CV ---
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_metrics_list: list[dict[str, float]] = []
            for train_i, test_i in skf.split(X, y):
                fold_m, _, _ = _train_and_evaluate(
                    model_cls,
                    config,
                    X.iloc[train_i],
                    y.iloc[train_i],
                    X.iloc[test_i],
                    y.iloc[test_i],
                    seed=seed,
                    n_bootstrap=0,
                )
                fold_metrics_list.append(fold_m)

            mean_metrics = {
                k: float(np.mean([m.get(k, 0.0) for m in fold_metrics_list]))
                for k in ["auroc", "auprc"]
            }
            results_list.append(
                SplitComparisonResult(
                    model_name=model_name,
                    feature_set=fs_name,
                    split_strategy="random",
                    metrics=mean_metrics,
                    fold_metrics=fold_metrics_list,
                    n_train=int(len(X) * 0.8),
                    n_test=int(len(X) * 0.2),
                )
            )

            # --- Random 5-fold CV, training set size-matched to geographic ---
            if size_matched and len(train_idx) >= 10 and len(test_idx) >= 10:
                from sklearn.model_selection import train_test_split

                skf_sm = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                sm_fold_metrics: list[dict[str, float]] = []
                for train_i, test_i in skf_sm.split(X, y):
                    if len(train_i) > len(train_idx):
                        train_i, _ = train_test_split(
                            train_i,
                            train_size=len(train_idx),
                            stratify=y.iloc[train_i],
                            random_state=seed,
                        )
                    fold_m, _, _ = _train_and_evaluate(
                        model_cls,
                        config,
                        X.iloc[train_i],
                        y.iloc[train_i],
                        X.iloc[test_i],
                        y.iloc[test_i],
                        seed=seed,
                        n_bootstrap=0,
                    )
                    sm_fold_metrics.append(fold_m)

                sm_mean = {
                    k: float(np.mean([m.get(k, 0.0) for m in sm_fold_metrics]))
                    for k in ["auroc", "auprc"]
                }
                results_list.append(
                    SplitComparisonResult(
                        model_name=model_name,
                        feature_set=fs_name,
                        split_strategy="random_sizematched",
                        metrics=sm_mean,
                        fold_metrics=sm_fold_metrics,
                        n_train=len(train_idx),
                        n_test=int(len(X) * 0.2),
                    )
                )

    # DeLong tests: compare Baseline vs Full for each model (geographic split)
    delong_tests: dict[str, dict[str, float]] = {}
    if compute_delong and geo_y_test is not None:
        from aquacontam.benchmark.metrics import delong_test

        for model_name, probs_dict in geo_probs.items():
            if "baseline" in probs_dict and "full" in probs_dict:
                dl = delong_test(geo_y_test, probs_dict["baseline"], probs_dict["full"])
                delong_tests[model_name] = dl

    return {
        "results": [r.to_dict() for r in results_list],
        "delong_tests": delong_tests,
        "n_models": len(model_specs),
        "feature_set_sizes": {k: len(v.columns) for k, v in feature_sets.items()},
    }


def split_comparison_summary_table(results: list[SplitComparisonResult]) -> pd.DataFrame:
    """Format comparison results as a publication-ready summary table.

    Parameters
    ----------
    results : list[SplitComparisonResult]
        Results from ``run_split_comparison_matrix()``.

    Returns
    -------
    pd.DataFrame
        Table with columns: Model, Feature Set, Split, AUROC, AUPRC.
    """
    rows = []
    for r in results:
        auroc = r.metrics.get("auroc", float("nan"))
        auprc = r.metrics.get("auprc", float("nan"))
        ci_str = ""
        if r.bootstrap_ci and "auroc" in r.bootstrap_ci:
            ci = r.bootstrap_ci["auroc"]
            ci_str = f" [{ci.get('ci_lower', 0):.3f}-{ci.get('ci_upper', 0):.3f}]"
        elif r.fold_metrics:
            std = float(np.std([m.get("auroc", 0) for m in r.fold_metrics]))
            ci_str = f" \u00b1 {std:.3f}"
        rows.append(
            {
                "Model": r.model_name,
                "Feature Set": r.feature_set.title(),
                "Split": r.split_strategy.title(),
                "AUROC": f"{auroc:.3f}{ci_str}",
                "AUPRC": f"{auprc:.3f}",
                "n_train": r.n_train,
                "n_test": r.n_test,
            }
        )
    return cast(pd.DataFrame, pd.DataFrame(rows))


def compute_inflation_across_seeds(
    random_aurocs: list[float],
    geographic_auroc: float,
) -> dict[str, float]:
    """Summarize AUROC inflation from random CV across multiple seeds.

    Parameters
    ----------
    random_aurocs : list[float]
        AUROC under random split for each seed.
    geographic_auroc : float
        AUROC under the fixed geographic split (single value).

    Returns
    -------
    dict[str, float]
        Summary with mean/std/min/max of random AUROC and inflation.
    """
    arr = np.array(random_aurocs)
    inflation = arr - geographic_auroc

    return {
        "geographic_auroc": geographic_auroc,
        "random_auroc_mean": float(arr.mean()),
        "random_auroc_std": float(arr.std()),
        "random_auroc_min": float(arr.min()),
        "random_auroc_max": float(arr.max()),
        "inflation_mean": float(inflation.mean()),
        "inflation_std": float(inflation.std()),
        "inflation_min": float(inflation.min()),
        "inflation_max": float(inflation.max()),
        "n_seeds": len(random_aurocs),
        "all_positive_inflation": bool(np.all(inflation > 0)),
    }
