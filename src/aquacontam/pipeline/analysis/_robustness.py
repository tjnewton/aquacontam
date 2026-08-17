"""Data-based CV and sensitivity analysis functions.

Functions in this module take ``wq_df`` and ``feature_dfs`` and train
their own models from scratch to evaluate robustness, stability, and
sensitivity of predictions.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, cast

from aquacontam.pipeline._strict import is_strict
from aquacontam.pipeline.assembly import assemble_with_split_imputation
from aquacontam.pipeline.models import load_model_configs as _load_model_configs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leave-One-Region-Out cross-validation
# ---------------------------------------------------------------------------


def _run_loro_cv(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    exclude_features: list[str] | None = None,
) -> None:
    """Run Leave-One-Region-Out cross-validation on T1 and T4.

    Provides per-region test performance to characterize fold-size
    imbalance and geographic variation in model performance.

    ``exclude_features`` (exact names and/or glob patterns) drops columns after
    assembly -- used by the provenance-free run so the honest environmental-signal
    LORO is computed on the same restricted feature set as the headline model.
    """
    import numpy as np
    import pandas as pd

    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
        resolve_excluded_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier
    from aquacontam.preprocessing.splits import leave_one_region_out

    # Map pwsid → data source for detection-only ablation
    if "source" in wq_df.columns:
        pwsid_source: pd.Series | None = wq_df.groupby("pwsid")["source"].first()
    else:
        pwsid_source = None

    # Task configs: (task_label, analyte, target)
    loro_tasks = [
        ("T1", "PFOS", "detected"),
        ("T4", "lead", "action_level"),
    ]

    # Models to evaluate
    yaml_configs = _load_model_configs()
    model_specs: list[tuple[str, type, dict[str, Any]]] = []

    xgb_cfg = dict(yaml_configs.get("xgboost_classifier", {}))
    xgb_cfg["random_state"] = seed
    model_specs.append(("xgboost", XGBoostClassifier, xgb_cfg))

    # Random Forest is the headline T1 model -- include it in LORO so the
    # per-region table reports the headline model, not only XGBoost/CatBoost.
    from aquacontam.models.random_forest import RandomForestClassifier

    rf_cfg = dict(yaml_configs.get("random_forest_classifier", {}))
    rf_cfg["random_state"] = seed
    model_specs.append(("random_forest", RandomForestClassifier, rf_cfg))

    try:
        from aquacontam.models.catboost import CatBoostClassifier

        cb_cfg = dict(yaml_configs.get("catboost_classifier", {}))
        cb_cfg["random_state"] = seed
        model_specs.append(("catboost", CatBoostClassifier, cb_cfg))
    except (ImportError, OSError, RuntimeError):
        logger.warning("CatBoost not available for LORO CV")

    all_task_results: dict[str, dict[str, dict[str, Any]]] = {}

    for task_label, analyte, target in loro_tasks:
        logger.info("Running LORO cross-validation on %s (%s/%s)...", task_label, analyte, target)

        try:
            sys_targets = aggregate_to_system_level(wq_df, analyte, target=target)
            sys_targets = drop_leakage_columns(sys_targets)
            X, y, _, regions = assemble_with_split_imputation(
                sys_targets, feature_dfs, wq_df=wq_df
            )
            if exclude_features:
                drop_cols = resolve_excluded_columns(X.columns, exclude_features)
                if drop_cols:
                    logger.info("LORO %s: excluding %d feature(s)", task_label, len(drop_cols))
                    X = X.drop(columns=drop_cols)
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed to build features for LORO %s", task_label, exc_info=True)
            if is_strict():
                raise
            continue

        # Build DataFrame with region info for LORO
        loro_df = pd.DataFrame({"epa_region": regions}, index=X.index)

        all_model_results: dict[str, dict[str, Any]] = {}

        for model_name, ModelClass, base_cfg in model_specs:
            logger.info("LORO CV %s model: %s", task_label, model_name)
            fold_results: list[dict[str, Any]] = []

            folds = leave_one_region_out(loro_df, seed=seed)

            for train_df, val_df, test_df, test_region in folds:
                train_idx = train_df.index.intersection(X.index)
                val_idx = val_df.index.intersection(X.index)
                test_idx = test_df.index.intersection(X.index)

                if len(train_idx) < 10 or len(test_idx) < 10:
                    logger.warning("Skipping LORO fold %d (insufficient data)", test_region)
                    continue

                try:
                    model = ModelClass(config=base_cfg.copy())
                    model.fit(
                        X.loc[train_idx],
                        y.loc[train_idx],
                        X_val=X.loc[val_idx],
                        y_val=y.loc[val_idx],
                    )
                    preds = model.predict(X.loc[test_idx])
                    probs = model.predict_proba(X.loc[test_idx])
                    if probs.ndim == 2 and probs.shape[1] == 2:
                        probs = probs[:, 1]

                    metrics = compute_classification_metrics(
                        y.loc[test_idx],
                        preds,
                        probs,
                        metrics=["auroc", "auprc", "f1"],
                    )
                    fold_results.append(
                        {
                            "test_region": int(test_region),
                            "n_train": len(train_idx),
                            "n_val": len(val_idx),
                            "n_test": len(test_idx),
                            **metrics,
                        }
                    )
                    # Persist per-system OOF predictions for the region-5 fold so a
                    # Minnesota-specific held-out AUROC can be read from the frozen
                    # loro_cv.json. Region 5 is the leave-out fold whose model never
                    # trains on any MN system, so the MN subset is leak-free. Captured
                    # for both the full and provenance-free runs via this same path
                    # (~40 KB/model; well under the 5 MB freeze size guard).
                    if int(test_region) == 5:
                        fold_results[-1]["oof_predictions"] = {
                            "pwsid": [str(p) for p in test_idx],
                            "y_true": [int(v) for v in y.loc[test_idx]],
                            "y_prob": [float(v) for v in probs],
                        }
                    logger.info(
                        "  LORO %s fold %d: n_test=%d, AUROC=%.4f, AUPRC=%.4f",
                        task_label,
                        test_region,
                        len(test_idx),
                        metrics.get("auroc", float("nan")),
                        metrics.get("auprc", float("nan")),
                    )

                    # Detection-only source ablation for this fold
                    if pwsid_source is not None:
                        from aquacontam.analysis.ablation import (
                            evaluate_loro_fold_excluding_detection_only,
                        )

                        fold_source_labels = (
                            pd.Series(test_idx, name="pwsid").map(pwsid_source).fillna("unknown")
                        )
                        try:
                            do_result = evaluate_loro_fold_excluding_detection_only(
                                y.loc[test_idx],
                                probs,
                                fold_source_labels,
                                test_region=int(test_region),
                            )
                            fold_results[-1]["detection_only_ablation"] = do_result
                            if do_result.get("n_excluded", 0) > 0:
                                filt_auroc = do_result.get("excluding_detection_only", {}).get(
                                    "auroc", float("nan")
                                )
                                logger.info(
                                    "    Detection-only ablation fold %d: "
                                    "excluded=%d, filtered AUROC=%.4f",
                                    test_region,
                                    do_result["n_excluded"],
                                    filt_auroc,
                                )
                        except (ValueError, RuntimeError, ArithmeticError):
                            logger.warning(
                                "Detection-only ablation failed for LORO fold %d (%s/%s)",
                                test_region,
                                model_name,
                                task_label,
                                exc_info=True,
                            )
                            if is_strict():
                                raise

                except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
                    logger.warning(
                        "Failed LORO fold %d (%s/%s)",
                        test_region,
                        model_name,
                        task_label,
                        exc_info=True,
                    )
                    if is_strict():
                        raise

            if fold_results:
                aurocs = [
                    f["auroc"] for f in fold_results if not np.isnan(f.get("auroc", float("nan")))
                ]
                model_entry: dict[str, Any] = {
                    "n_folds": len(fold_results),
                    "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
                    "std_auroc": float(np.std(aurocs)) if aurocs else float("nan"),
                    "folds": fold_results,
                }

                # Summarize detection-only ablation across folds
                do_folds = [
                    f["detection_only_ablation"]
                    for f in fold_results
                    if "detection_only_ablation" in f
                    and f["detection_only_ablation"].get("excluding_detection_only")
                ]
                if do_folds:
                    affected_regions = {f["test_region"] for f in do_folds}
                    # Recompute mean AUROC substituting filtered values for
                    # affected folds, keeping originals for unaffected folds
                    adj_aurocs = []
                    for fr in fold_results:
                        rgn = fr["test_region"]
                        if rgn in affected_regions:
                            do_ab = fr.get("detection_only_ablation", {})
                            filt = do_ab.get("excluding_detection_only", {})
                            a = filt.get("auroc", fr.get("auroc", float("nan")))
                        else:
                            a = fr.get("auroc", float("nan"))
                        if not np.isnan(a):
                            adj_aurocs.append(a)

                    model_entry["detection_only_summary"] = {
                        "n_folds_affected": len(do_folds),
                        "affected_regions": sorted(affected_regions),
                        "mean_auroc_adjusted": (
                            float(np.mean(adj_aurocs)) if adj_aurocs else float("nan")
                        ),
                        "std_auroc_adjusted": (
                            float(np.std(adj_aurocs)) if adj_aurocs else float("nan")
                        ),
                    }

                all_model_results[model_name] = model_entry
                logger.info(
                    "LORO CV %s/%s: mean AUROC=%.4f +/- %.4f (%d folds)",
                    task_label,
                    model_name,
                    model_entry["mean_auroc"],
                    model_entry["std_auroc"],
                    model_entry["n_folds"],
                )

        if all_model_results:
            all_task_results[task_label] = all_model_results

    if all_task_results:
        out_path = output_dir / "loro_cv.json"
        out_path.write_text(json.dumps(all_task_results, indent=2, default=str))
        logger.info("LORO CV results saved to %s", out_path)

    # Write focused LORO detection-only ablation file
    loro_do_results: dict[str, dict[str, Any]] = {}
    for tl, model_results in all_task_results.items():
        task_do: dict[str, Any] = {}
        for mn, mresults in model_results.items():
            folds_with_do = [
                f
                for f in mresults.get("folds", [])
                if "detection_only_ablation" in f
                and f["detection_only_ablation"].get("n_excluded", 0) > 0
            ]
            if folds_with_do:
                task_do[mn] = {
                    "folds": [f["detection_only_ablation"] for f in folds_with_do],
                    "summary": mresults.get("detection_only_summary", {}),
                }
        if task_do:
            loro_do_results[tl] = task_do

    if loro_do_results:
        do_path = output_dir / "loro_detection_only_ablation.json"
        do_path.write_text(json.dumps(loro_do_results, indent=2, default=str))
        logger.info("LORO detection-only ablation saved to %s", do_path)


# ---------------------------------------------------------------------------
# Multi-seed stability
# ---------------------------------------------------------------------------


def _run_multi_seed_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    exclude_features: list[str] | None = None,
) -> None:
    """Run multi-seed stability analysis on T1 and T4.

    Trains XGBoost, CatBoost, Logistic Regression, and Random Forest across
    multiple random seeds to quantify variance attributable to random
    initialization vs. data/model structure. Reports mean +/- std for AUROC and
    AUPRC. ``exclude_features`` (exact names and/or glob patterns) drops columns
    after assembly for the provenance-free variant.
    """
    import numpy as np

    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
        resolve_excluded_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    # Load stability seeds from config (fallback to defaults)
    default_seeds = [42, 123, 456, 789, 2024]
    try:
        from aquacontam._config import load_experiment_config

        exp_cfg = load_experiment_config()
        stability_seeds: list[int] = exp_cfg.get("stability_seeds", default_seeds)
    except (FileNotFoundError, KeyError):
        stability_seeds = default_seeds

    # Task configs: (task_label, analyte, target)
    seed_tasks = [
        ("T1", "PFOS", "detected"),
        ("T4", "lead", "action_level"),
    ]

    # Models to evaluate
    yaml_configs = _load_model_configs()
    model_specs: list[tuple[str, type, dict[str, Any]]] = []

    xgb_cfg = dict(yaml_configs.get("xgboost_classifier", {}))
    model_specs.append(("xgboost", XGBoostClassifier, xgb_cfg))

    try:
        from aquacontam.models.catboost import CatBoostClassifier

        cb_cfg = dict(yaml_configs.get("catboost_classifier", {}))
        model_specs.append(("catboost", CatBoostClassifier, cb_cfg))
    except (ImportError, OSError, RuntimeError):
        logger.warning("CatBoost not available for multi-seed analysis")

    try:
        from aquacontam.models.logistic import LogisticRegressionClassifier

        lr_cfg = dict(yaml_configs.get("logistic_regression", {}))
        model_specs.append(("logistic_regression", LogisticRegressionClassifier, lr_cfg))
    except (ImportError, OSError, RuntimeError):
        logger.warning("LogisticRegression wrapper not available for multi-seed analysis")

    # Random Forest is the headline T1 model and previously had no seed
    # distribution -- include it so the headline number's stability is reported.
    from aquacontam.models.random_forest import RandomForestClassifier

    rf_cfg = dict(yaml_configs.get("random_forest_classifier", {}))
    model_specs.append(("random_forest", RandomForestClassifier, rf_cfg))

    # Geographic split regions
    train_regions = {1, 3, 4, 5, 6}
    val_regions = {2, 7}
    test_regions = {8, 9, 10}

    all_task_results: dict[str, dict[str, dict[str, Any]]] = {}

    for task_label, analyte, target in seed_tasks:
        logger.info("Multi-seed stability analysis on %s (%s/%s)...", task_label, analyte, target)

        try:
            sys_targets = aggregate_to_system_level(wq_df, analyte, target=target)
            sys_targets = drop_leakage_columns(sys_targets)
            X, y, _, regions = assemble_with_split_imputation(
                sys_targets, feature_dfs, wq_df=wq_df
            )
            if exclude_features:
                drop_cols = resolve_excluded_columns(X.columns, exclude_features)
                if drop_cols:
                    logger.info(
                        "Multi-seed %s: excluding %d feature(s)", task_label, len(drop_cols)
                    )
                    X = X.drop(columns=drop_cols)
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed to build features for multi-seed %s", task_label, exc_info=True)
            if is_strict():
                raise
            continue

        # Build train/val/test masks
        train_mask = regions.isin(train_regions)
        val_mask = regions.isin(val_regions)
        test_mask = regions.isin(test_regions)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X.loc[val_mask], y.loc[val_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        if len(X_train) < 10 or len(X_test) < 10:
            logger.warning("Insufficient data for multi-seed %s", task_label)
            continue

        all_model_results: dict[str, dict[str, Any]] = {}

        for model_name, ModelClass, base_cfg in model_specs:
            logger.info("Multi-seed %s model: %s", task_label, model_name)
            auroc_per_seed: list[float] = []
            auprc_per_seed: list[float] = []

            for s in stability_seeds:
                try:
                    cfg = base_cfg.copy()
                    cfg["random_state"] = s

                    model = ModelClass(config=cfg)
                    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                    preds = model.predict(X_test)
                    probs = model.predict_proba(X_test)
                    if probs.ndim == 2 and probs.shape[1] == 2:
                        probs = probs[:, 1]

                    metrics = compute_classification_metrics(
                        y_test, preds, probs, metrics=["auroc", "auprc"]
                    )
                    auroc_per_seed.append(float(metrics.get("auroc", float("nan"))))
                    auprc_per_seed.append(float(metrics.get("auprc", float("nan"))))

                    logger.info(
                        "  seed=%d: AUROC=%.4f, AUPRC=%.4f",
                        s,
                        metrics.get("auroc", float("nan")),
                        metrics.get("auprc", float("nan")),
                    )
                except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
                    logger.warning(
                        "Failed multi-seed %s/%s seed=%d",
                        task_label,
                        model_name,
                        s,
                        exc_info=True,
                    )
                    if is_strict():
                        raise

            if auroc_per_seed:
                all_model_results[model_name] = {
                    "seeds": stability_seeds[: len(auroc_per_seed)],
                    "auroc_per_seed": auroc_per_seed,
                    "auprc_per_seed": auprc_per_seed,
                    "mean_auroc": float(np.mean(auroc_per_seed)),
                    "std_auroc": float(np.std(auroc_per_seed)),
                    "mean_auprc": float(np.mean(auprc_per_seed)),
                    "std_auprc": float(np.std(auprc_per_seed)),
                }
                logger.info(
                    "Multi-seed %s/%s: AUROC=%.4f +/- %.4f, AUPRC=%.4f +/- %.4f (%d seeds)",
                    task_label,
                    model_name,
                    all_model_results[model_name]["mean_auroc"],
                    all_model_results[model_name]["std_auroc"],
                    all_model_results[model_name]["mean_auprc"],
                    all_model_results[model_name]["std_auprc"],
                    len(auroc_per_seed),
                )

        if all_model_results:
            all_task_results[task_label] = all_model_results

    if all_task_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "multi_seed_stability.json"
        out_path.write_text(json.dumps(all_task_results, indent=2, default=str))
        logger.info("Multi-seed stability results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Region heterogeneity
# ---------------------------------------------------------------------------


def _run_region_heterogeneity(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Analyze per-region heterogeneity in T1 performance.

    Builds the T1 (PFOS/detected) feature matrix, assigns EPA regions,
    and runs heterogeneity analysis to explain why LORO CV performance
    varies across regions (e.g. AUROC 0.618 in Region 4 vs 0.761 in Region 3).
    """

    from aquacontam.analysis.regional import analyze_regional_heterogeneity
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )

    logger.info("Running per-region heterogeneity analysis (T1: PFOS/detected)...")

    # Build T1 feature matrix with train-only imputation
    try:
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to build features for region heterogeneity", exc_info=True)
        if is_strict():
            raise
        return

    # Try to load top SHAP features from prior SHAP analysis
    top_features: list[str] | None = None
    shap_path = output_dir / "shap_T1.json"
    if shap_path.exists():
        try:
            shap_data = json.loads(shap_path.read_text())
            mean_abs = shap_data.get("mean_abs_shap", {})
            if mean_abs:
                sorted_feats = sorted(mean_abs, key=lambda k: mean_abs[k], reverse=True)
                top_features = sorted_feats[:5]
                logger.info("Using top 5 SHAP features: %s", top_features)
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Could not load SHAP features, using variance-based", exc_info=True)

    try:
        result = analyze_regional_heterogeneity(
            X, y, regions, top_features=top_features, best_region=3, worst_region=4
        )
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Region heterogeneity analysis failed", exc_info=True)
        if is_strict():
            raise
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "region_heterogeneity.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Region heterogeneity results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------


def _run_feature_ablation(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run feature category ablation study on T1 and T4.

    For T1: uses ``run_feature_ablation()`` with a fixed geographic split and
    paired bootstrap significance tests on the test set.

    For T4: uses LORO CV (low positive rate requires all-region evaluation),
    aggregates out-of-fold predictions across folds, then runs
    ``paired_bootstrap_test()`` on the concatenated OOF set for significance.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames.
    output_dir : Path
        Results output directory.
    seed : int
        Random seed.
    """
    import yaml

    from aquacontam.analysis.ablation import (
        _columns_matching_prefixes,
        ablation_summary,
        run_feature_ablation,
    )
    from aquacontam.benchmark._utils import get_proba
    from aquacontam.benchmark.metrics import paired_bootstrap_test
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    # Load ablation categories from config
    config_path = Path("configs/experiment.yaml")
    feature_categories: dict[str, list[str]] = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        feature_categories = config.get("ablation", {}).get("feature_categories", {})

    if not feature_categories:
        logger.info("No ablation categories configured — skipping")
        return

    # Only T1 and T4 use fixed geographic splits suitable for drop-one-category
    # ablation.  T5 (transfer) and T7 (temporal) have dedicated ablation via
    # _run_transfer_ablation() and the --no-monitoring-features flag,
    # respectively -- running them through this path would duplicate T1's data
    # (same analyte/target) and produce identical, misleading results.
    ablation_tasks = [
        ("T1", "PFOS", "detected"),
        ("T4", "lead", "action_level"),
    ]

    all_ablation_results: list[dict[str, Any]] = []
    t1_ablation_results: list[Any] = []  # AblationResult objects for summary CSV

    for task_name, analyte, target in ablation_tasks:
        logger.info("Running feature ablation for %s (%s)", task_name, analyte)

        try:
            sys_targets = aggregate_to_system_level(wq_df, analyte, target=target)
            sys_targets = drop_leakage_columns(sys_targets)
            X, y, _, regions = assemble_with_split_imputation(
                sys_targets, feature_dfs, wq_df=wq_df
            )

            # Diagnostic: log which ablation categories matched how many columns
            for cat, prefixes in feature_categories.items():
                n = sum(1 for c in X.columns if any(c.startswith(p) for p in prefixes))
                level = logging.INFO if n > 0 else logging.WARNING
                logger.log(level, "Ablation '%s': %d matching columns", cat, n)

            # T4 (low positive rate) uses LORO CV to avoid single-class folds;
            # T1 has enough positive rate for a single geographic split.
            # For T4, we aggregate out-of-fold predictions across all LORO
            # folds and run paired_bootstrap_test on the concatenated set
            # (each system appears in exactly one fold's holdout).
            if task_name == "T4":
                import numpy as np

                model_config = {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "random_state": seed,
                }
                unique_regions = sorted(regions.dropna().unique())

                # Per-category OOF prediction accumulators
                oof_preds: dict[str, dict[str, list[np.ndarray]]] = {}
                # Also accumulate full-model baseline scores per fold
                fold_baselines: list[float] = []
                fold_scores: dict[str, list[float]] = {}
                n_features_removed: dict[str, int] = {}
                valid_folds = 0

                for holdout_region in unique_regions:
                    train_idx = X.index[regions != holdout_region]
                    val_idx = X.index[regions == holdout_region]

                    if len(train_idx) < 10 or len(val_idx) < 10:
                        logger.info(
                            "  LORO fold region=%s: skipping (<10 samples)",
                            holdout_region,
                        )
                        continue
                    if len(np.unique(y.loc[train_idx])) < 2:
                        logger.info(
                            "  LORO fold region=%s: skipping (single-class y_train)",
                            holdout_region,
                        )
                        continue
                    if len(np.unique(y.loc[val_idx])) < 2:
                        logger.info(
                            "  LORO fold region=%s: skipping (single-class y_val)",
                            holdout_region,
                        )
                        continue

                    valid_folds += 1

                    # Train full model on this fold
                    full_model = XGBoostClassifier(config=model_config.copy())
                    full_model.fit(X.loc[train_idx], y.loc[train_idx])
                    probs_full_fold = get_proba(full_model, X.loc[val_idx])
                    if probs_full_fold is None:
                        continue

                    from sklearn.metrics import average_precision_score

                    y_val_arr = np.asarray(y.loc[val_idx])
                    baseline_score = float(average_precision_score(y_val_arr, probs_full_fold))
                    fold_baselines.append(baseline_score)

                    # Train ablated models per category
                    for cat, prefixes in feature_categories.items():
                        drop_cols = _columns_matching_prefixes(X, prefixes)
                        if not drop_cols:
                            continue

                        X_tr_abl = X.loc[train_idx].drop(columns=drop_cols, errors="ignore")
                        X_va_abl = X.loc[val_idx].drop(columns=drop_cols, errors="ignore")
                        if X_tr_abl.empty:
                            continue

                        n_features_removed[cat] = len(drop_cols)

                        abl_model = XGBoostClassifier(config=model_config.copy())
                        abl_model.fit(X_tr_abl, y.loc[train_idx])
                        probs_abl_fold = get_proba(abl_model, X_va_abl)
                        if probs_abl_fold is None:
                            continue

                        abl_score = float(average_precision_score(y_val_arr, probs_abl_fold))
                        fold_scores.setdefault(cat, []).append(abl_score)

                        # Accumulate OOF predictions for bootstrap test
                        if cat not in oof_preds:
                            oof_preds[cat] = {
                                "y_true": [],
                                "probs_full": [],
                                "probs_ablated": [],
                            }
                        oof_preds[cat]["y_true"].append(y_val_arr)
                        oof_preds[cat]["probs_full"].append(probs_full_fold)
                        oof_preds[cat]["probs_ablated"].append(probs_abl_fold)

                if valid_folds > 0 and oof_preds:
                    # Baseline row (all features)
                    baseline_avg = float(np.nanmean(fold_baselines))
                    all_ablation_results.append(
                        {
                            "category": "all_features",
                            "n_features_removed": 0,
                            "baseline_score": baseline_avg,
                            "ablated_score": baseline_avg,
                            "delta": 0.0,
                            "task": task_name,
                            "n_folds": valid_folds,
                        }
                    )

                    for cat, arrays in oof_preds.items():
                        y_all = np.concatenate(arrays["y_true"])
                        pf_all = np.concatenate(arrays["probs_full"])
                        pa_all = np.concatenate(arrays["probs_ablated"])

                        sig = paired_bootstrap_test(
                            y_all,
                            pf_all,
                            pa_all,
                            metric_fn="auprc",
                            n_iterations=1000,
                            seed=seed,
                        )

                        avg_entry = {
                            "category": cat,
                            "n_features_removed": n_features_removed.get(cat, 0),
                            "baseline_score": float(np.nanmean(fold_baselines)),
                            "ablated_score": float(
                                np.nanmean(fold_scores.get(cat, [float("nan")]))
                            ),
                            "delta": float(np.nanmean(fold_scores.get(cat, [float("nan")])))
                            - float(np.nanmean(fold_baselines)),
                            "task": task_name,
                            "n_folds": valid_folds,
                            "significance": {"auprc": sig},
                        }
                        if np.isnan(avg_entry["ablated_score"]):  # type: ignore[call-overload]
                            logger.warning(
                                "  %s/%s: all folds produced NaN ablation score",
                                task_name,
                                cat,
                            )
                            avg_entry["note"] = "all folds NaN"
                        all_ablation_results.append(avg_entry)

                    logger.info(
                        "T4 LORO ablation: %d valid folds, %d categories with "
                        "bootstrap significance",
                        valid_folds,
                        len(oof_preds),
                    )
                else:
                    logger.warning("No valid LORO folds for %s ablation", task_name)
            else:
                train_idx = X.index[regions.isin((1, 3, 4, 5, 6))]
                val_idx = X.index[regions.isin((2, 7))]
                test_idx = X.index[regions.isin((8, 9, 10))]

                if len(train_idx) < 10 or len(val_idx) < 10 or len(test_idx) < 10:
                    logger.warning("Insufficient data for %s ablation", task_name)
                    continue

                model_config = {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "random_state": seed,
                }
                sig_results = run_feature_ablation(
                    XGBoostClassifier,
                    model_config,
                    X.loc[train_idx],
                    y.loc[train_idx],
                    X.loc[val_idx],
                    y.loc[val_idx],
                    X.loc[test_idx],
                    y.loc[test_idx],
                    feature_categories,
                    n_bootstrap=1000,
                    seed=seed,
                )
                t1_ablation_results = sig_results

                baseline_auprc = sig_results[0].metrics.get("auprc", 0.0)
                for ar in sig_results:
                    ablated_auprc = ar.metrics.get("auprc", 0.0)
                    entry: dict[str, Any] = {
                        "category": "all_features" if ar.category == "full" else ar.category,
                        "task": task_name,
                        "n_features_removed": ar.n_features_dropped,
                        "baseline_score": baseline_auprc,
                        "ablated_score": ablated_auprc,
                        "delta": ablated_auprc - baseline_auprc,
                        "significance": ar.significance,
                    }
                    all_ablation_results.append(entry)

        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed ablation for %s", task_name, exc_info=True)
            if is_strict():
                raise

    # Apply FDR correction across ablation categories per task before saving JSON
    if all_ablation_results:
        from aquacontam.analysis.equity import apply_multiple_testing_correction

        for tn in ("T1", "T4"):
            task_entries = [
                e
                for e in all_ablation_results
                if e.get("task") == tn and e.get("category") != "all_features"
            ]
            for metric in ("auroc", "auprc"):
                raw_p = [
                    e.get("significance", {}).get(metric, {}).get("p_value") for e in task_entries
                ]
                valid_p = [p for p in raw_p if p is not None and not math.isnan(p)]
                if not valid_p:
                    continue
                correction = apply_multiple_testing_correction(valid_p)
                j = 0
                for e, p in zip(task_entries, raw_p, strict=True):
                    sig = e.setdefault("significance", {}).setdefault(metric, {})
                    if p is not None and not math.isnan(p):
                        sig["p_value_fdr"] = correction["corrected_p_values"][j]
                        sig["significant_fdr"] = correction["reject"][j]
                        j += 1

    if all_ablation_results:
        ablation_path = output_dir / "feature_ablation.json"
        ablation_path.write_text(json.dumps(all_ablation_results, indent=2, default=str))
        logger.info("Feature ablation results saved to %s", ablation_path)

    if t1_ablation_results:
        summary_df = ablation_summary(t1_ablation_results)
        summary_path = output_dir / "feature_ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Ablation summary (with significance) saved to %s", summary_path)


# ---------------------------------------------------------------------------
# CNN1D feature ordering sensitivity
# ---------------------------------------------------------------------------


def _run_cnn1d_feature_ordering(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    n_orderings: int = 5,
) -> None:
    """Run CNN1D feature ordering sensitivity analysis.

    Trains CNN1D with multiple random feature orderings to quantify variance
    attributable to arbitrary feature adjacency vs genuine model capacity.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames.
    output_dir : Path
        Results output directory.
    seed : int
        Base random seed.
    n_orderings : int
        Number of random feature orderings to test (default 5).
    """
    import numpy as np

    try:
        from aquacontam.models.cnn1d import CNN1DClassifier
    except (ImportError, OSError, RuntimeError):
        logger.info("PyTorch not available — skipping CNN1D feature ordering analysis")
        return

    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import prepare_train_val_test

    logger.info("Running CNN1D feature ordering sensitivity (%d orderings)...", n_orderings)

    # Use prepare_train_val_test for proper train-only imputation (no leakage)
    splits = prepare_train_val_test(wq_df, "PFOS", *feature_dfs, target="detected")
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    if len(X_train) < 10 or len(X_test) < 10:
        logger.warning("Insufficient data for CNN1D ordering analysis")
        return

    yaml_configs = _load_model_configs()
    base_cfg = dict(yaml_configs.get("cnn1d_classifier", {}))

    results_list: list[dict[str, Any]] = []

    def _train_and_eval(cfg: dict[str, Any], label: str) -> dict[str, Any] | None:
        """Train CNN1D with given config and return metrics dict, or None on failure."""
        m = CNN1DClassifier(config=cfg)
        try:
            m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            preds = m.predict(X_test)
            probs = m.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]
            mets = compute_classification_metrics(y_test, preds, probs, metrics=["auroc", "auprc"])
            logger.info("  %s: AUROC=%.4f AUPRC=%.4f", label, mets["auroc"], mets["auprc"])
            return mets
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed CNN1D %s ordering", label, exc_info=True)
            if is_strict():
                raise
            return None

    # Alphabetical ordering (default)
    base_cfg["feature_order"] = "alphabetical"
    base_cfg["random_state"] = seed
    mets = _train_and_eval(base_cfg.copy(), "alphabetical")
    if mets is not None:
        results_list.append({"ordering": "alphabetical", "seed": seed, **mets})

    # Domain ordering (groups features by domain: proximity -> land use -> hydro -> demographics)
    cfg_domain = base_cfg.copy()
    cfg_domain["feature_order"] = "domain"
    cfg_domain["random_state"] = seed
    mets = _train_and_eval(cfg_domain, "domain")
    if mets is not None:
        results_list.append({"ordering": "domain", "seed": seed, **mets})

    # Random orderings
    for i in range(n_orderings):
        ordering_seed = seed + i + 1
        cfg = base_cfg.copy()
        cfg["feature_order"] = "random"
        cfg["random_state"] = ordering_seed
        mets = _train_and_eval(cfg, f"random_{ordering_seed}")
        if mets is not None:
            results_list.append(
                {"ordering": f"random_{ordering_seed}", "seed": ordering_seed, **mets}
            )

    if results_list:
        aurocs = [r["auroc"] for r in results_list if not np.isnan(r.get("auroc", float("nan")))]
        auprcs = [r["auprc"] for r in results_list if not np.isnan(r.get("auprc", float("nan")))]
        summary = {
            "n_orderings": len(results_list),
            "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
            "auprc_mean": float(np.mean(auprcs)) if auprcs else float("nan"),
            "auprc_std": float(np.std(auprcs)) if auprcs else float("nan"),
            "results": results_list,
        }

        out_path = output_dir / "cnn1d_feature_ordering.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(
            "CNN1D feature ordering: AUROC=%.4f+/-%.4f, AUPRC=%.4f+/-%.4f",
            summary["auroc_mean"],
            summary["auroc_std"],
            summary["auprc_mean"],
            summary["auprc_std"],
        )


# ---------------------------------------------------------------------------
# Sensitivity analysis (drop-NA threshold + buffer radius)
# ---------------------------------------------------------------------------


def _run_sensitivity_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run sensitivity analysis: drop-NA threshold and buffer radius variants.

    Tests robustness of PFOS detection (T1) to:
    1. ``drop_na_threshold`` in feature assembly (0.3-0.7)
    2. Proximity buffer radius subsets (all, 1km, 5km, 10km, dist-only)

    Results saved to ``results/sensitivity_analysis.json``.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames.
    output_dir : Path
        Results output directory.
    seed : int
        Random seed.
    """
    import numpy as np
    import pandas as pd

    try:
        from aquacontam.benchmark.metrics import compute_classification_metrics
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            assemble_feature_matrix,
            drop_leakage_columns,
        )
        from aquacontam.models.xgboost import XGBoostClassifier
        from aquacontam.preprocessing.splits import assign_epa_region
    except ImportError:
        logger.info("Missing dependencies — skipping sensitivity analysis")
        return

    logger.info("Running sensitivity analysis...")

    # Prepare base data
    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)

    if "epa_region" not in sys_targets.columns:
        # Coordinate-rescued region from the sample-level wq_df (has lat/lon).
        from aquacontam.features.assembly import _safe_mode

        _dfr = assign_epa_region(wq_df)
        _sys_region = _dfr.groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
        sys_targets["epa_region"] = sys_targets.index.map(_sys_region.to_dict())

    regions = pd.to_numeric(sys_targets["epa_region"], errors="coerce")
    train_mask = regions.isin((1, 3, 4, 5, 6))
    val_mask = regions.isin((2, 7))

    if train_mask.sum() < 10 or val_mask.sum() < 10:
        logger.warning("Insufficient data for sensitivity analysis")
        return

    yaml_configs = _load_model_configs()
    xgb_cfg = dict(yaml_configs.get("xgboost_classifier", {}))
    xgb_cfg.setdefault("n_estimators", 100)
    xgb_cfg.setdefault("max_depth", 6)
    xgb_cfg["random_state"] = seed

    # --- Sub-analysis 1: drop_na_threshold ---
    threshold_results: list[dict[str, Any]] = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        try:
            train_targets = sys_targets.loc[train_mask].drop(columns=["epa_region"])
            val_targets = sys_targets.loc[val_mask].drop(columns=["epa_region"])

            X_train, y_train, impute_stats = assemble_feature_matrix(
                train_targets, *feature_dfs, drop_na_threshold=thresh
            )
            X_val, y_val, _ = assemble_feature_matrix(
                val_targets, *feature_dfs, impute_stats=impute_stats
            )

            model = XGBoostClassifier(config=xgb_cfg.copy())
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]
            metrics = compute_classification_metrics(
                y_val, preds, probs, metrics=["auroc", "auprc"]
            )
            threshold_results.append(
                {
                    "threshold": thresh,
                    "n_features": int(X_train.shape[1]),
                    "auroc": float(metrics["auroc"]),
                    "auprc": float(metrics["auprc"]),
                }
            )
            logger.info(
                "  drop_na_threshold=%.1f: %d features, AUROC=%.4f, AUPRC=%.4f",
                thresh,
                X_train.shape[1],
                metrics["auroc"],
                metrics["auprc"],
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed sensitivity for threshold=%.1f", thresh, exc_info=True)
            if is_strict():
                raise

    # --- Sub-analysis 2: proximity buffer radius subsets ---
    buffer_results: list[dict[str, Any]] = []

    # Build full feature matrix (default threshold=0.5)
    train_targets = sys_targets.loc[train_mask].drop(columns=["epa_region"])
    val_targets = sys_targets.loc[val_mask].drop(columns=["epa_region"])
    X_train_full, y_train, impute_stats = assemble_feature_matrix(train_targets, *feature_dfs)
    X_val_full, y_val, _ = assemble_feature_matrix(
        val_targets, *feature_dfs, impute_stats=impute_stats
    )

    def _col_subset(X: pd.DataFrame, variant: str) -> pd.DataFrame:
        """Select column subsets based on buffer radius variant."""
        if variant == "all_radii":
            return X
        elif variant == "1km_only":
            drop = [c for c in X.columns if any(f"_{r}m" in c for r in ("5000", "10000"))]
            return cast(pd.DataFrame, X.drop(columns=drop, errors="ignore"))
        elif variant == "5km_only":
            drop = [c for c in X.columns if any(f"_{r}m" in c for r in ("1000", "10000"))]
            return cast(pd.DataFrame, X.drop(columns=drop, errors="ignore"))
        elif variant == "10km_only":
            drop = [c for c in X.columns if any(f"_{r}m" in c for r in ("1000", "5000"))]
            return cast(pd.DataFrame, X.drop(columns=drop, errors="ignore"))
        elif variant == "dist_only":
            drop = [c for c in X.columns if "_n_" in c or c.startswith("count_")]
            return cast(pd.DataFrame, X.drop(columns=drop, errors="ignore"))
        return X

    for variant in ["all_radii", "1km_only", "5km_only", "10km_only", "dist_only"]:
        try:
            X_tr = _col_subset(X_train_full, variant)
            X_va = _col_subset(X_val_full, variant)
            # Count how many proximity columns remain
            prox_cols = [c for c in X_tr.columns if "nearest_" in c or "_n_" in c or "count_" in c]

            if X_tr.shape[1] == 0:
                logger.warning("No features left for variant=%s", variant)
                continue

            model = XGBoostClassifier(config=xgb_cfg.copy())
            model.fit(X_tr, y_train)
            preds = model.predict(X_va)
            probs = model.predict_proba(X_va)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]
            metrics = compute_classification_metrics(
                y_val, preds, probs, metrics=["auroc", "auprc"]
            )
            buffer_results.append(
                {
                    "variant": variant,
                    "n_proximity_cols": len(prox_cols),
                    "n_total_cols": int(X_tr.shape[1]),
                    "auroc": float(metrics["auroc"]),
                    "auprc": float(metrics["auprc"]),
                }
            )
            logger.info(
                "  buffer_%s: %d prox cols, AUROC=%.4f, AUPRC=%.4f",
                variant,
                len(prox_cols),
                metrics["auroc"],
                metrics["auprc"],
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed sensitivity for variant=%s", variant, exc_info=True)
            if is_strict():
                raise

    if threshold_results or buffer_results:
        summary = {
            "drop_na_threshold": threshold_results,
            "buffer_radii": buffer_results,
        }
        out_path = output_dir / "sensitivity_analysis.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Sensitivity analysis saved to %s", out_path)

        if threshold_results:
            aurocs = [r["auroc"] for r in threshold_results if not np.isnan(r["auroc"])]
            if aurocs:
                logger.info(
                    "  Threshold sensitivity: AUROC range [%.4f, %.4f]",
                    min(aurocs),
                    max(aurocs),
                )
            else:
                logger.warning("  Threshold sensitivity: all AUROC values are NaN")

    # --- Sub-analysis 3: detection limit fill value ---
    try:
        from aquacontam.analysis.sensitivity import detection_limit_sensitivity

        dl_result = detection_limit_sensitivity(
            XGBoostClassifier,
            X_train_full,
            y_train,
            X_val_full,
            y_val,
            config=xgb_cfg,
        )
        dl_out = output_dir / "detection_limit_sensitivity.json"
        dl_out.write_text(json.dumps(dl_result, indent=2, default=str))
        logger.info("Detection limit sensitivity saved to %s", dl_out)
        if dl_result["summary"].get("n_successful", 0) > 0:
            logger.info(
                "  DL sensitivity: %s range %.4f",
                dl_result["summary"]["metric"],
                dl_result["summary"].get("range", 0.0),
            )
    except (ImportError, ValueError, RuntimeError):
        logger.warning("Detection limit sensitivity failed", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# Dedup comparison
# ---------------------------------------------------------------------------


def _run_dedup_comparison(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Compare T1 performance with and without sample deduplication."""
    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        drop_leakage_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier
    from aquacontam.preprocessing.splits import assign_epa_region, geographic_split

    try:
        import pandas as pd

        variants: dict[str, dict[str, Any]] = {}
        for label, strategy in [("no_dedup", None), ("dedup_max", "max")]:
            sys_agg = aggregate_to_system_level(
                wq_df, "PFOS", target="detected", deduplicate_strategy=strategy
            )
            sys_agg = drop_leakage_columns(sys_agg)

            # Geographic split (coordinate-rescued region from sample-level wq_df)
            from aquacontam.features.assembly import _safe_mode

            _dfr = assign_epa_region(wq_df)
            _sys_region = _dfr.groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
            df_with_region = sys_agg.reset_index().copy()
            df_with_region["epa_region"] = df_with_region["pwsid"].map(_sys_region)
            train_sys, val_sys, test_sys = geographic_split(df_with_region)

            train_idx = train_sys.set_index("pwsid")
            val_idx = val_sys.set_index("pwsid")
            test_idx = test_sys.set_index("pwsid")

            # Drop region + leakage for feature assembly
            _drop = [
                "epa_region",
                "any_detected",
                "detection_rate",
                "max_concentration",
            ]
            train_idx = train_idx.drop(columns=[c for c in _drop if c in train_idx.columns])
            val_idx = val_idx.drop(columns=[c for c in _drop if c in val_idx.columns])
            test_idx = test_idx.drop(columns=[c for c in _drop if c in test_idx.columns])

            X_tr, y_tr, stats = assemble_feature_matrix(train_idx, *feature_dfs)
            X_val, y_val, _ = assemble_feature_matrix(val_idx, *feature_dfs, impute_stats=stats)
            X_te, y_te, _ = assemble_feature_matrix(test_idx, *feature_dfs, impute_stats=stats)

            if X_tr.empty or len(y_tr.unique()) < 2:
                logger.warning("Insufficient data for dedup variant %s", label)
                continue

            model = XGBoostClassifier(config={"n_estimators": 100, "random_state": seed})
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]

            metrics = compute_classification_metrics(
                y_te, preds, probs, metrics=["auroc", "auprc"]
            )
            n_samples_stats = (
                sys_agg["n_samples"] if "n_samples" in sys_agg.columns else pd.Series()
            )
            variants[label] = {
                "metrics": metrics,
                "n_systems": len(sys_agg),
                "mean_n_samples": float(n_samples_stats.mean()) if len(n_samples_stats) else 0,
                "median_n_samples": float(n_samples_stats.median()) if len(n_samples_stats) else 0,
            }

        # Compute deltas
        if "no_dedup" in variants and "dedup_max" in variants:
            for metric in ("auroc", "auprc"):
                base = variants["no_dedup"]["metrics"].get(metric, 0)
                dedup = variants["dedup_max"]["metrics"].get(metric, 0)
                variants["delta"] = variants.get("delta", {})
                variants["delta"][metric] = dedup - base

        out_path = output_dir / "dedup_comparison.json"
        out_path.write_text(json.dumps(variants, indent=2, default=str))
        logger.info("Dedup comparison saved to %s", out_path)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Dedup comparison failed", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# Split comparison
# ---------------------------------------------------------------------------


def _run_split_comparison(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    size_matched: bool = False,
) -> None:
    """Run split strategy comparison (simple + matrix)."""
    try:
        from aquacontam.analysis.split_strategy_comparison import (
            run_split_ablation,
            run_split_comparison_matrix,
        )

        # Original simple comparison
        simple_results = run_split_ablation(wq_df, feature_dfs, seed=seed)

        # Multi-model 2x2 matrix comparison
        matrix_results: dict[str, Any] = {}
        try:
            matrix_results = run_split_comparison_matrix(
                wq_df, feature_dfs, seed=seed, n_bootstrap=1000, size_matched=size_matched
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Split comparison matrix failed", exc_info=True)
            if is_strict():
                raise

        combined = {"simple": simple_results, "matrix": matrix_results}
        out_path = output_dir / "split_comparison.json"
        out_path.write_text(json.dumps(combined, indent=2, default=str))
        logger.info("Split comparison saved to %s", out_path)
    except ImportError:
        logger.info("Split comparison module not available — skipping")
    except (OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Split comparison failed", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# Strengthening analyses (peer-review requested)
# ---------------------------------------------------------------------------


def _run_spatial_block_bootstrap(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Region-block bootstrap of the LORO headline CI (reads loro_cv.json)."""
    from aquacontam.analysis.strengthening import region_block_bootstrap

    loro_path = output_dir / "loro_cv.json"
    if not loro_path.exists():
        logger.warning("loro_cv.json missing; run --loro-cv before --spatial-block-bootstrap")
        return
    loro = json.loads(loro_path.read_text())
    out: dict[str, Any] = {}
    for task in ("T1", "T4"):
        for model in ("xgboost", "random_forest"):
            folds = loro.get(task, {}).get(model, {}).get("folds")
            if folds:
                res = region_block_bootstrap(folds, seed=seed)
                out[f"{task}_{model}"] = res
    (output_dir / "spatial_block_bootstrap.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    logger.info("Spatial block bootstrap saved to %s", output_dir / "spatial_block_bootstrap.json")


def _run_buffered_loro(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    buffer_km: float = 50.0,
) -> None:
    """Buffered-boundary LORO for T1 (with-provenance + provenance-free)."""
    from aquacontam.analysis.strengthening import buffered_boundary_loro
    from aquacontam.features.assembly import PROVENANCE_FREE_EXCLUDE

    try:
        res = buffered_boundary_loro(
            wq_df,
            feature_dfs,
            buffer_km=buffer_km,
            seed=seed,
            exclude_features=list(PROVENANCE_FREE_EXCLUDE),
        )
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Buffered LORO failed", exc_info=True)
        if is_strict():
            raise
        return
    (output_dir / "loro_cv_buffered.json").write_text(json.dumps(res, indent=2, default=str))
    logger.info("Buffered LORO saved to %s", output_dir / "loro_cv_buffered.json")


def _run_icp_recoverability(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """ICP monitoring-source recoverability probe (writes icp_recoverability.json)."""
    from aquacontam.analysis.strengthening import icp_recoverability_probe

    try:
        res = icp_recoverability_probe(wq_df, feature_dfs, seed=seed)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("ICP recoverability probe failed", exc_info=True)
        if is_strict():
            raise
        return
    (output_dir / "icp_recoverability.json").write_text(json.dumps(res, indent=2, default=str))
    logger.info("ICP recoverability saved to %s", output_dir / "icp_recoverability.json")


# ---------------------------------------------------------------------------
# Coordinate sensitivity
# ---------------------------------------------------------------------------


def _run_coordinate_sensitivity(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run coordinate perturbation sensitivity analysis."""
    try:
        from aquacontam.analysis.coordinate_sensitivity import (
            coordinate_sensitivity_summary,
            run_coordinate_sensitivity,
        )
        from aquacontam.models.xgboost import XGBoostClassifier

        results = run_coordinate_sensitivity(
            XGBoostClassifier,
            {"n_estimators": 100},
            wq_df,
            feature_dfs=feature_dfs,
            magnitudes_km=(0.0, 1.0, 5.0, 10.0, 20.0),
            n_seeds=5,
            base_seed=seed,
        )

        summary_df = coordinate_sensitivity_summary(results)
        out_data: dict[str, Any] = {
            "summary": summary_df.to_dict(orient="records"),
            "raw_results": [
                {
                    "magnitude_km": r.magnitude_km,
                    "seed": r.seed,
                    "metrics": r.metrics,
                    "n_systems": r.n_systems,
                    "mean_displacement_km": r.mean_displacement_km,
                }
                for r in results
            ],
        }
        out_path = output_dir / "coordinate_sensitivity.json"
        out_path.write_text(json.dumps(out_data, indent=2, default=str))
        logger.info("Coordinate sensitivity saved to %s", out_path)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Coordinate sensitivity failed", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# MCL exceedance analysis
# ---------------------------------------------------------------------------


def _run_mcl_exceedance_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Compare detection-based (T1) vs MCL-exceedance prediction targets.

    Trains XGBoost on the MCL exceedance target (systems exceeding
    EPA PFAS Maximum Contaminant Levels) and compares AUROC/AUPRC
    to the detection target used in T1.
    """
    import json as json_mod

    import numpy as np

    from aquacontam.analysis.mcl_threshold import (
        compute_mcl_exceedance,
        mcl_summary_statistics,
    )
    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    # Step 1: Compute MCL exceedance targets
    try:
        mcl_df = compute_mcl_exceedance(wq_df)
    except (ValueError, KeyError):
        logger.warning("MCL exceedance computation failed", exc_info=True)
        if is_strict():
            raise
        return

    mcl_stats = mcl_summary_statistics(mcl_df)
    logger.info(
        "MCL exceedance: %d systems, %.1f%% exceeding",
        mcl_stats.get("n_systems", 0),
        mcl_stats.get("exceedance_rate", 0.0) * 100,
    )

    if mcl_stats.get("n_systems", 0) < 50:
        logger.warning("Too few systems for MCL analysis (%d)", mcl_stats.get("n_systems", 0))
        return

    # Step 2: Build features for T1 detection target (PFOS) for comparison
    try:
        sys_targets_det = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets_det = drop_leakage_columns(sys_targets_det)
        X_det, y_det, _, regions_det = assemble_with_split_imputation(
            sys_targets_det, feature_dfs, wq_df=wq_df
        )
    except (ImportError, OSError, ValueError, RuntimeError):
        logger.warning("Failed to build detection features for MCL comparison", exc_info=True)
        if is_strict():
            raise
        return

    # Step 3: Build MCL exceedance target aligned with same features
    # Merge MCL target onto the feature matrix index (pwsid)
    mcl_indexed = mcl_df.set_index("pwsid")["mcl_exceedance"]
    common_idx = X_det.index.intersection(mcl_indexed.index)

    if len(common_idx) < 50:
        logger.warning(
            "Insufficient overlap between MCL and feature data (%d systems)", len(common_idx)
        )
        return

    X_mcl = X_det.loc[common_idx]
    y_mcl = mcl_indexed.loc[common_idx]
    regions_mcl = regions_det.loc[common_idx]

    # Geographic split
    train_regions = {1, 3, 4, 5, 6}
    val_regions = {2, 7}
    test_regions = {8, 9, 10}

    yaml_configs = _load_model_configs()
    xgb_cfg = dict(yaml_configs.get("xgboost_classifier", {}))
    xgb_cfg["random_state"] = seed

    results_out: dict[str, Any] = {"mcl_summary": mcl_stats}

    for label, X, y, regions in [
        ("detection", X_det, y_det, regions_det),
        ("mcl_exceedance", X_mcl, y_mcl, regions_mcl),
    ]:
        train_mask = regions.isin(train_regions)
        val_mask = regions.isin(val_regions)
        test_mask = regions.isin(test_regions)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X.loc[val_mask], y.loc[val_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        if len(X_train) < 10 or len(X_test) < 10:
            logger.warning("Insufficient data for MCL %s target", label)
            continue

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning("Single class for MCL %s target", label)
            continue

        try:
            model = XGBoostClassifier(config=xgb_cfg.copy())
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]

            metrics = compute_classification_metrics(
                y_test,
                preds,
                probs,
                metrics=["auroc", "auprc", "f1", "sensitivity", "specificity"],
            )
            results_out[label] = {
                "metrics": metrics,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_test": float(y_test.mean()),
            }
            logger.info(
                "  MCL %s: AUROC=%.3f, AUPRC=%.3f",
                label,
                metrics.get("auroc", 0.0),
                metrics.get("auprc", 0.0),
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("MCL %s model training failed", label, exc_info=True)
            if is_strict():
                raise

    out_path = output_dir / "mcl_exceedance_analysis.json"
    out_path.write_text(json_mod.dumps(results_out, indent=2, default=str))
    logger.info("MCL exceedance analysis saved to %s", out_path)


# ---------------------------------------------------------------------------
# Validation set reuse bias quantification
# ---------------------------------------------------------------------------


def _run_val_reuse_bias(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Quantify validation set reuse bias in holdout-LORO AUROC gap.

    Runs three matched experiments varying only validation handling
    (same test set, same hyperparameters) to decompose the AUROC gap
    between the geographic holdout and LORO CV. The gap endpoints are
    read from ``results.json`` / ``loro_cv.json`` in ``output_dir``.

    Results written to ``output_dir / "val_reuse_bias.json"``.
    """
    import json as json_mod_local

    import numpy as np

    from aquacontam.analysis.val_reuse_bias import run_val_reuse_experiment
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )

    logger.info("Running validation set reuse bias quantification...")

    # Load stability seeds from experiment config
    default_seeds = [42, 123, 456, 789, 2024]
    try:
        from aquacontam._config import load_experiment_config

        exp_cfg = load_experiment_config()
        stability_seeds: list[int] = exp_cfg.get("stability_seeds", default_seeds)
    except (FileNotFoundError, KeyError):
        stability_seeds = default_seeds

    # Load XGBoost config
    yaml_configs = _load_model_configs()
    xgb_cfg = dict(yaml_configs.get("xgboost_classifier", {}))

    # Build features for T1 (PFOS detection)
    try:
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to build features for val-reuse bias analysis", exc_info=True)
        if is_strict():
            raise
        return

    # Load existing LORO per-region results for comparison
    loro_per_region: dict[int, float] = {}
    loro_mean_auroc: float | None = None
    loro_path = output_dir / "loro_cv.json"
    if loro_path.exists():
        try:
            import json as json_reader

            loro_data = json_reader.loads(loro_path.read_text())
            t1_xgb = loro_data.get("T1", {}).get("xgboost", {})
            fold_aurocs: list[float] = []
            for fold in t1_xgb.get("folds", []):
                r = fold.get("test_region")
                auroc = fold.get("auroc")
                if r is not None and auroc is not None:
                    loro_per_region[int(r)] = float(auroc)
                if auroc is not None:
                    fold_aurocs.append(float(auroc))
            loro_mean_auroc = t1_xgb.get("mean_auroc")
            if loro_mean_auroc is None and fold_aurocs:
                loro_mean_auroc = float(np.mean(fold_aurocs))
        except (KeyError, ValueError, OSError):
            logger.warning("Could not load LORO per-region results", exc_info=True)

    # Canonical single-split holdout AUROC (Table 2, XGBoost T1) for the
    # holdout-to-LORO gap — derived from results.json rather than hardcoded.
    holdout_auroc: float | None = None
    results_path = output_dir / "results.json"
    if results_path.exists():
        try:
            import json as json_reader

            for entry in json_reader.loads(results_path.read_text()):
                if (
                    entry.get("task") == "T1"
                    and entry.get("model") == "xgboost_classifier"
                    and entry.get("metrics", {}).get("auroc") is not None
                ):
                    holdout_auroc = float(entry["metrics"]["auroc"])
                    break
        except (KeyError, ValueError, OSError):
            logger.warning("Could not load holdout AUROC from results.json", exc_info=True)

    # Run the experiment
    try:
        result = run_val_reuse_experiment(
            X,
            y,
            regions,
            model_config=xgb_cfg,
            seeds=stability_seeds,
            loro_per_region=loro_per_region or None,
            holdout_auroc=holdout_auroc,
            loro_mean_auroc=loro_mean_auroc,
        )
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Val-reuse bias experiment failed", exc_info=True)
        if is_strict():
            raise
        return

    out_path = output_dir / "val_reuse_bias.json"
    out_path.write_text(json_mod_local.dumps(result, indent=2, default=str))
    logger.info("Val-reuse bias results saved to %s", out_path)

    # Log key findings
    decomp = result.get("decomposition", {})
    logger.info(
        "Val-reuse effect: %.4f AUROC (%.1f%% of holdout-LORO gap)",
        decomp.get("val_reuse_effect", 0.0),
        decomp.get("pct_explained_by_val_reuse", 0.0),
    )
