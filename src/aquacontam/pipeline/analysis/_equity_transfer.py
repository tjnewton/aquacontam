"""Hybrid analysis functions requiring both fitted models and raw data.

Functions in this module need both ``wq_df``/``feature_dfs`` *and*
``fitted_models`` or ``downloaded`` paths -- they evaluate existing
models on new subsets or train specialised models for equity, causal,
and transfer analyses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aquacontam.pipeline._strict import is_strict
from aquacontam.pipeline.assembly import assemble_with_split_imputation
from aquacontam.pipeline.models import load_model_configs as _load_model_configs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equity analysis
# ---------------------------------------------------------------------------


def _run_equity_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
) -> None:
    """Run equity analysis and export results.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames.
    fitted_models : dict[str, tuple[Any, str, float]]
        Fitted models from training (model, task_type, optimal_threshold).
    output_dir : Path
        Results output directory.
    """
    import pandas as pd

    # Check if demographics features are available
    demo_available = any(
        "pct_people_of_color" in df.columns for df in feature_dfs if isinstance(df, pd.DataFrame)
    )
    if not demo_available:
        logger.info("Demographics features unavailable — skipping equity analysis")
        return

    # Use T1 model if available
    if "T1" not in fitted_models:
        logger.info("No T1 fitted model — skipping equity analysis")
        return

    model, _, threshold = fitted_models["T1"]

    try:
        import numpy as np

        from aquacontam.analysis.equity import (
            analyze_equity,
            apply_multiple_testing_correction,
            compute_monitoring_adjusted_burden_ratio,
            permutation_test_burden_ratio,
        )
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )

        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)

        # Fallback: if split produced empty X (e.g. no valid EPA regions),
        # assemble on full dataset (test-only contexts like unit tests).
        if X.empty:
            from aquacontam.features.assembly import assemble_feature_matrix

            logger.warning("No valid EPA regions — assembling features on full dataset")
            X, y, _ = assemble_feature_matrix(sys_targets, *feature_dfs)
            regions = pd.Series(np.nan, index=X.index)

        # Restrict to test-set systems only (EPA regions 8, 9, 10)
        # so equity metrics reflect out-of-sample performance.
        test_mask = regions.isin((8, 9, 10))
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]
        if X_test.empty:
            logger.warning("No test-set systems — using full dataset for equity analysis")
            X_test, y_test = X, y

        # Use predict_proba + stored threshold instead of predict (which
        # uses 0.5 and produces all-negative predictions under class imbalance)

        # Align X_test columns to match what the model was trained on.
        # Equity analysis uses the *full* T1 system set (which may have more
        # aquifer type dummies than the geographic train split), so drop
        # extra columns and add missing ones as zeros.
        try:
            train_cols = model.feature_names if hasattr(model, "feature_names") else None
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            if is_strict():
                raise
            train_cols = None
        # Check wrapper class directly (MLP, CNN1D, GNN store feature_names_in_)
        if train_cols is None:
            fn = getattr(model, "feature_names_in_", None)
            if fn is not None:
                train_cols = list(fn)
        # Then check sklearn inner model
        if train_cols is None:
            inner = getattr(model, "_model", None)
            if inner is not None:
                fn = getattr(inner, "feature_names_in_", None)
                if fn is None:
                    fn = getattr(inner, "feature_name_", None)
                if fn is not None:
                    train_cols = list(fn)
        if train_cols is not None:
            missing = [c for c in train_cols if c not in X_test.columns]
            for c in missing:
                X_test[c] = 0.0
            X_test = X_test[train_cols]

        try:
            probas = model.predict_proba(X_test)
            if probas.ndim == 2 and probas.shape[1] == 2:
                probas = probas[:, 1]
        except NotImplementedError:
            probas = None

        if probas is not None:
            preds = (np.asarray(probas) >= threshold).astype(int)
        else:
            preds = model.predict(X_test)

        # Find demographics DataFrame among feature_dfs
        demo_df = None
        for df in feature_dfs:
            if isinstance(df, pd.DataFrame) and "pct_people_of_color" in df.columns:
                demo_df = df
                break

        if demo_df is None:
            return

        # Align demographics with test-set feature matrix index
        demo_aligned = demo_df.reindex(X_test.index)

        results = []
        group_cols = [
            "pct_people_of_color",
            "pct_low_income",
            "pct_limited_english",
            "pct_less_hs_education",
        ]
        for group_col in group_cols:
            if group_col not in demo_aligned.columns:
                continue
            report = analyze_equity(y_test, preds, probas, demo_aligned, group_col)

            # Permutation test for statistical significance of burden ratio
            group_values = np.asarray(demo_aligned[group_col], dtype=float)
            perm_result = permutation_test_burden_ratio(
                y_test,
                group_values,
                n_permutations=10_000,
                seed=42,
            )

            # Monitoring-adjusted (IPW) burden ratio: removes the part of the
            # raw disparity attributable to differential monitoring intensity,
            # using the SAME high/low thresholds so the two are comparable.
            # Reported regardless of whether the disparity attenuates.
            adjusted: dict[str, float] | None = None
            if "n_samples" in X_test.columns:
                adjusted = compute_monitoring_adjusted_burden_ratio(
                    y_test,
                    group_values,
                    np.asarray(X_test["n_samples"], dtype=float),
                    X_test,
                    high_threshold=float(report.metadata.get("high_threshold", 0.0)),
                    low_threshold=float(report.metadata.get("low_threshold", 0.0)),
                    n_permutations=10_000,
                    seed=42,
                )

            results.append(
                {
                    "group": group_col,
                    "burden_ratio": report.burden_ratio,
                    "burden_ratio_ipw_adjusted": (
                        adjusted.get("burden_ratio_ipw") if adjusted else None
                    ),
                    "p_value_ipw_adjusted": adjusted.get("p_value") if adjusted else None,
                    "prediction_ratio": report.prediction_ratio,
                    "n_high": report.n_high,
                    "n_low": report.n_low,
                    "group_metrics": report.group_metrics,
                    "p_value": perm_result.get("p_value"),
                    "threshold_used": threshold,
                }
            )

        # Apply BH-FDR correction for multiple testing across demographic groups
        if results:
            raw_pvalues: list[float] = [float(r.get("p_value") or 1.0) for r in results]  # type: ignore[arg-type]
            fdr = apply_multiple_testing_correction(raw_pvalues, method="fdr_bh")
            for i, r in enumerate(results):
                r["p_value_fdr"] = fdr["corrected_p_values"][i]
                r["reject_fdr"] = fdr["reject"][i]

            out_path = output_dir / "equity_analysis.json"
            out_path.write_text(json.dumps(results, indent=2, default=str))
            logger.info("Exported equity analysis to %s", out_path)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to run equity analysis", exc_info=True)
        if is_strict():
            raise


def _run_monitoring_inequity(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Quantify national demographic inequity in MONITORING intensity.

    Distinct from the contamination burden ratio: this asks whether sampling
    EFFORT itself is unequally distributed across demographic groups -- a novel
    environmental-justice finding and direct evidence for the monitoring-process
    thesis. Computed over all monitored systems (national), controlling for
    system size (population served). Writes ``monitoring_inequity.json``.
    """
    try:
        from aquacontam.analysis.monitoring_equity import (
            DEFAULT_GROUP_COLS,
            analyze_monitoring_inequity,
        )
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )

        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, _y, _stats, _regions = assemble_with_split_imputation(
            sys_targets, feature_dfs, wq_df=wq_df
        )
        if X.empty:
            from aquacontam.features.assembly import assemble_feature_matrix

            X, _y, _stats = assemble_feature_matrix(sys_targets, *feature_dfs)

        if "n_samples" not in X.columns:
            logger.warning("n_samples not in features -- skipping monitoring inequity analysis")
            return
        demo_cols = tuple(c for c in DEFAULT_GROUP_COLS if c in X.columns)
        if not demo_cols:
            logger.warning("No demographic columns -- skipping monitoring inequity analysis")
            return

        population = X["population_served"] if "population_served" in X.columns else None
        result = analyze_monitoring_inequity(
            X["n_samples"],
            X[list(demo_cols)],
            population=population,
            group_cols=demo_cols,
            n_permutations=10_000,
            seed=seed,
        )
        out_path = output_dir / "monitoring_inequity.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info(
            "Exported monitoring inequity analysis to %s (%d dimensions)", out_path, len(result)
        )
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to run monitoring inequity analysis", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# LORO equity analysis (per-region burden ratios)
# ---------------------------------------------------------------------------


def _run_loro_equity_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    n_permutations: int = 1_000,
    min_systems: int = 30,
) -> None:
    """Run LORO cross-validated equity analysis across all EPA regions.

    For each held-out region, trains a fresh model on the remaining 9
    regions and computes burden ratios on the held-out region's test set.
    This extends the primary equity analysis (restricted to Regions 8, 9, 10)
    to provide per-region EJ estimates as supplementary material.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames (must include demographics).
    output_dir : Path
        Results output directory.
    seed : int
        Random seed for reproducibility.
    n_permutations : int
        Permutations per burden-ratio test (default 1,000).
    min_systems : int
        Minimum systems with demographics in a region to run analysis.
    """
    import pandas as pd

    # Check if demographics features are available
    demo_available = any(
        "pct_people_of_color" in df.columns for df in feature_dfs if isinstance(df, pd.DataFrame)
    )
    if not demo_available:
        logger.info("Demographics features unavailable — skipping LORO equity analysis")
        return

    try:
        import numpy as np

        from aquacontam.analysis.equity import (
            analyze_equity,
            apply_multiple_testing_correction,
            permutation_test_burden_ratio,
        )
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )
        from aquacontam.models.xgboost import XGBoostClassifier
        from aquacontam.preprocessing.splits import leave_one_region_out

        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)

        if X.empty:
            logger.warning("Empty feature matrix — skipping LORO equity analysis")
            return

        # Find demographics DataFrame
        demo_df = None
        for df in feature_dfs:
            if isinstance(df, pd.DataFrame) and "pct_people_of_color" in df.columns:
                demo_df = df
                break
        if demo_df is None:
            return

        # Load XGBoost config from YAML
        yaml_configs = _load_model_configs()
        xgb_cfg = dict(yaml_configs.get("xgboost_classifier", {}))
        xgb_cfg["random_state"] = seed

        # Build LORO folds
        loro_df = pd.DataFrame({"epa_region": regions}, index=X.index)
        folds = leave_one_region_out(loro_df, seed=seed)

        group_cols = [
            "pct_people_of_color",
            "pct_low_income",
            "pct_limited_english",
            "pct_less_hs_education",
        ]

        per_region: list[dict[str, Any]] = []

        for train_df, val_df, test_df, test_region in folds:
            train_idx = train_df.index.intersection(X.index)
            val_idx = val_df.index.intersection(X.index)
            test_idx = test_df.index.intersection(X.index)

            if len(train_idx) < 10 or len(test_idx) < 10:
                logger.warning("Skipping LORO equity fold %d (insufficient data)", test_region)
                continue

            # Align demographics with held-out region
            demo_aligned = demo_df.reindex(test_idx)
            n_with_demo = demo_aligned["pct_people_of_color"].notna().sum()
            if n_with_demo < min_systems:
                logger.info(
                    "Region %d: only %d systems with demographics (<%d) — skipping",
                    test_region,
                    n_with_demo,
                    min_systems,
                )
                continue

            # Train fresh model on remaining regions
            try:
                model = XGBoostClassifier(config=xgb_cfg.copy())
                model.fit(
                    X.loc[train_idx],
                    y.loc[train_idx],
                    X_val=X.loc[val_idx],
                    y_val=y.loc[val_idx],
                )
            except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
                logger.warning("Failed to train model for LORO equity fold %d", test_region)
                if is_strict():
                    raise
                continue

            # Align test columns with trained model
            X_test = X.loc[test_idx].copy()
            try:
                train_cols = model.feature_names if hasattr(model, "feature_names") else None
            except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
                if is_strict():
                    raise
                train_cols = None
            if train_cols is None:
                fn = getattr(model, "feature_names_in_", None)
                if fn is not None:
                    train_cols = list(fn)
            if train_cols is None:
                inner = getattr(model, "_model", None)
                if inner is not None:
                    fn = getattr(inner, "feature_names_in_", None)
                    if fn is None:
                        fn = getattr(inner, "feature_name_", None)
                    if fn is not None:
                        train_cols = list(fn)
            if train_cols is not None:
                for c in [c for c in train_cols if c not in X_test.columns]:
                    X_test[c] = 0.0
                X_test = X_test[train_cols]

            # Get predictions
            try:
                probas = model.predict_proba(X_test)
                if probas.ndim == 2 and probas.shape[1] == 2:
                    probas = probas[:, 1]
            except NotImplementedError:
                probas = None

            if probas is not None:
                preds = (np.asarray(probas) >= 0.5).astype(int)
            else:
                preds = model.predict(X_test)

            y_test = y.loc[test_idx]

            # Run equity analysis for each demographic group
            region_results: list[dict[str, Any]] = []
            for group_col in group_cols:
                if group_col not in demo_aligned.columns:
                    continue
                report = analyze_equity(y_test, preds, probas, demo_aligned, group_col)
                group_values = np.asarray(demo_aligned[group_col], dtype=float)
                perm_result = permutation_test_burden_ratio(
                    y_test,
                    group_values,
                    n_permutations=n_permutations,
                    seed=seed,
                )
                region_results.append(
                    {
                        "region": int(test_region),
                        "group": group_col,
                        "burden_ratio": report.burden_ratio,
                        "prediction_ratio": report.prediction_ratio,
                        "n_high": report.n_high,
                        "n_low": report.n_low,
                        "n_systems": len(test_idx),
                        "n_with_demographics": int(n_with_demo),
                        "p_value": perm_result.get("p_value"),
                    }
                )

            per_region.extend(region_results)
            logger.info(
                "LORO equity region %d: %d systems, %d with demographics, %d group results",
                test_region,
                len(test_idx),
                n_with_demo,
                len(region_results),
            )

        if not per_region:
            logger.warning("No regions produced equity results")
            return

        # Global FDR correction across all region x group tests
        raw_pvalues = [r.get("p_value", 1.0) for r in per_region]
        fdr = apply_multiple_testing_correction(raw_pvalues, method="fdr_bh")
        for i, r in enumerate(per_region):
            r["p_value_fdr"] = fdr["corrected_p_values"][i]
            r["reject_fdr"] = fdr["reject"][i]

        # Compute per-group summary across regions
        summary: dict[str, dict[str, Any]] = {}
        for group_col in group_cols:
            group_entries = [r for r in per_region if r["group"] == group_col]
            ratios = [r["burden_ratio"] for r in group_entries if np.isfinite(r["burden_ratio"])]
            n_sig = sum(1 for r in group_entries if r.get("reject_fdr", False))
            weights = [r["n_systems"] for r in group_entries if np.isfinite(r["burden_ratio"])]
            summary[group_col] = {
                "n_regions_analyzed": len(group_entries),
                "n_regions_significant_fdr": n_sig,
                "median_burden_ratio": float(np.median(ratios)) if ratios else float("nan"),
                "mean_burden_ratio": float(np.mean(ratios)) if ratios else float("nan"),
                "weighted_mean_burden_ratio": (
                    float(np.average(ratios, weights=weights))
                    if ratios and weights
                    else float("nan")
                ),
                "min_burden_ratio": float(np.min(ratios)) if ratios else float("nan"),
                "max_burden_ratio": float(np.max(ratios)) if ratios else float("nan"),
            }

        output = {
            "per_region": per_region,
            "summary": summary,
            "metadata": {
                "n_permutations": n_permutations,
                "min_systems": min_systems,
                "fdr_method": "fdr_bh",
                "seed": seed,
                "model": "XGBoostClassifier",
                "task": "T1_PFOS_detected",
            },
        }

        out_path = output_dir / "loro_equity_analysis.json"
        out_path.write_text(json.dumps(output, indent=2, default=str))
        logger.info("Exported LORO equity analysis to %s", out_path)

    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to run LORO equity analysis", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# Group conformal prediction
# ---------------------------------------------------------------------------


def _run_group_conformal_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run group-conditional conformal prediction by EPA region.

    Calibrates separate conformal thresholds per EPA region so each region
    receives independent coverage guarantees (>= 1-alpha).
    """

    try:
        from aquacontam.calibration.conformal import GroupConformalClassifier
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )
        from aquacontam.models.xgboost import XGBoostClassifier
    except ImportError:
        logger.info("Group conformal dependencies not available — skipping")
        return

    try:
        # Build T1 feature matrix with train-only imputation
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
        regions = regions.fillna(0).astype(int)

        from aquacontam._config import load_experiment_config

        exp_cfg = load_experiment_config()
        split_cfg = exp_cfg.get("split", {})
        train_regs = set(split_cfg.get("train_regions", [1, 3, 4, 5, 6]))
        val_regs = set(split_cfg.get("val_regions", [2, 7]))
        test_regs = set(split_cfg.get("test_regions", [8, 9, 10]))

        train_mask = regions.isin(train_regs)
        val_mask = regions.isin(val_regs)
        test_mask = regions.isin(test_regs)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_val, y_val = X.loc[val_mask], y.loc[val_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        test_regions = regions.loc[test_mask].to_numpy()

        # Train XGBoost (best overall model)
        model_cfg = exp_cfg.get("models", {}).get("xgboost_classifier", {})
        model = XGBoostClassifier(config=model_cfg)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val.to_numpy(), y_val.to_numpy())],
            verbose=False,
        )

        group_conformal_results: list[dict[str, Any]] = []
        for alpha in (0.05, 0.10, 0.20):
            gcc = GroupConformalClassifier(model, alpha=alpha)
            # Calibrate on val set with val regions
            val_regions = regions.loc[val_mask].to_numpy()
            gcc.calibrate(X_val, y_val, val_regions)

            # Evaluate on test set
            stats = gcc.coverage_and_set_size(X_test, y_test, test_regions)
            entry: dict[str, Any] = {
                "task": "T1",
                "model": "xgboost_classifier",
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "coverage": stats["coverage"],
                "avg_set_size": stats["avg_set_size"],
                "per_region": {str(k): v for k, v in stats.get("per_group", {}).items()},
            }
            group_conformal_results.append(entry)
            logger.info(
                "Group conformal alpha=%.2f: coverage=%.3f, avg_set_size=%.2f",
                alpha,
                stats["coverage"],
                stats["avg_set_size"],
            )

        if group_conformal_results:
            out_path = output_dir / "group_conformal_results.json"
            out_path.write_text(json.dumps(group_conformal_results, indent=2, default=str))
            logger.info("Group conformal results saved to %s", out_path)

    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to run group conformal analysis", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# Transfer ablation (T5)
# ---------------------------------------------------------------------------


def _run_transfer_ablation(
    wq_df: Any,
    feature_dfs: list[Any],
    downloaded: dict[str, Path],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run T5 transfer with/without system characteristics ablation.

    Validates whether cross-contaminant transfer is driven by genuine
    environmental signal or shared system characteristics.
    """
    import pandas as pd

    try:
        from aquacontam.benchmark.transfer import run_t5_ablated_transfer
        from aquacontam.models.xgboost import XGBoostClassifier
    except ImportError:
        logger.info("Required modules not available — skipping transfer ablation")
        return

    yaml_configs = _load_model_configs()
    cfg = dict(yaml_configs.get("xgboost_classifier", {}))
    cfg["random_state"] = seed
    if cfg.get("scale_pos_weight") == "auto":
        del cfg["scale_pos_weight"]

    sdwis_path = downloaded.get("sdwis")
    ucmr5_path = downloaded.get("ucmr5")
    source_data = pd.read_parquet(sdwis_path) if sdwis_path and sdwis_path.exists() else wq_df
    target_data = pd.read_parquet(ucmr5_path) if ucmr5_path and ucmr5_path.exists() else wq_df

    try:
        model = XGBoostClassifier(config=cfg)
        results = run_t5_ablated_transfer(
            model=model,
            source_data=source_data,
            target_data=target_data,
            feature_dfs=feature_dfs,
            source_analyte="lead",
            target_analyte="PFOS",
        )

        transfer_ablation: dict[str, Any] = {}
        for mode, result in results.items():
            transfer_ablation[mode] = {
                "metrics": result.metrics,
                "metadata": {
                    k: v
                    for k, v in result.metadata.items()
                    if k
                    not in (
                        "y_true",
                        "y_prob",
                        "latitudes",
                        "longitudes",
                        "split_labels",
                    )
                },
            }

        # Compute PWSID overlap between source and target
        source_pwsids = set(source_data["pwsid"].unique())
        target_pwsids = set(target_data["pwsid"].unique())
        overlap = source_pwsids & target_pwsids
        transfer_ablation["pwsid_overlap"] = {
            "n_source": len(source_pwsids),
            "n_target": len(target_pwsids),
            "n_overlap": len(overlap),
            "overlap_fraction": len(overlap) / max(len(target_pwsids), 1),
        }

        out_path = output_dir / "transfer_ablation.json"
        out_path.write_text(json.dumps(transfer_ablation, indent=2, default=str))
        logger.info("Transfer ablation results saved to %s", out_path)

        # Log key comparison
        full_auroc = results["full"].metrics.get("auroc", float("nan"))
        abl_auroc = results["no_system_chars"].metrics.get("auroc", float("nan"))
        logger.info(
            "T5 transfer ablation: full AUROC=%.3f, no-system-chars AUROC=%.3f (delta=%.3f)",
            full_auroc,
            abl_auroc,
            abl_auroc - full_auroc,
        )
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Transfer ablation failed", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# Lift analysis
# ---------------------------------------------------------------------------


def _run_lift_analysis(
    wq_df: Any,
    feature_dfs: list[Any],
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run decile/lift analysis for full and monitoring-free models."""
    try:
        from aquacontam.benchmark.metrics import compute_lift_analysis
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )

        if "T1" not in fitted_models:
            logger.info("No T1 fitted model — skipping lift analysis")
            return

        model, _, _ = fitted_models["T1"]

        # Prepare data with train-only imputation
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, impute_stats, regions = assemble_with_split_imputation(
            sys_targets, feature_dfs, wq_df=wq_df
        )

        if X.empty:
            return

        # Get test set
        test_idx = X.index[regions.isin((8, 9, 10))]

        if len(test_idx) < 20:
            logger.warning("Too few test systems for lift analysis")
            return

        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        # Align columns to match trained model's features
        train_features = getattr(getattr(model, "_model", None), "feature_names_in_", None)
        if train_features is not None:
            train_cols = list(train_features)
            # Add missing columns as zeros, drop extra columns
            for c in train_cols:
                if c not in X_test.columns:
                    X_test[c] = 0
            X_test = X_test[train_cols]
            X = X.reindex(columns=train_cols, fill_value=0)

        # Representative national base rate to anchor lift against, so the
        # reported lift is not conflated with the ENRICHED in-sample prevalence
        # (the test set pools detection-enriched sources). Set to the national
        # UCMR5 PFOS detection rate (~6.8%, Extended Data Fig. 3); the in-sample
        # prevalence is also reported (overall_detection_rate) for transparency.
        ucmr5_reference_prevalence = 0.068

        # Full model lift
        probs = model.predict_proba(X_test)
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]

        full_lift = compute_lift_analysis(
            y_test.values, probs, reference_prevalence=ucmr5_reference_prevalence
        )

        # Monitoring dependence (uses train-set medians to avoid leakage)
        from aquacontam.analysis.monitoring_bias import compute_monitoring_dependence

        train_medians = {k: v for k, v in impute_stats.items() if isinstance(v, (int, float))}
        monitoring_dep = compute_monitoring_dependence(
            model,
            X_test,
            monitoring_columns=["n_samples", "mean_detection_limit"],
            impute_medians=train_medians,
        )
        logger.info(
            "Monitoring dependence: shift=%.4f, max=%.4f, corr=%.4f",
            monitoring_dep["prediction_shift"],
            monitoring_dep["max_shift"],
            monitoring_dep["correlation"],
        )

        # Monitoring-free lift
        monitoring_cols = [c for c in X_test.columns if c in ("n_samples", "mean_detection_limit")]
        nomon_lift: dict[str, Any] = {}
        if monitoring_cols:
            from aquacontam.models.xgboost import XGBoostClassifier

            X_nomon = X.drop(columns=monitoring_cols, errors="ignore")
            train_idx = X.index[regions.isin((1, 3, 4, 5, 6))]
            val_idx = X.index[regions.isin((2, 7))]

            nomon_model = XGBoostClassifier(config={"n_estimators": 500, "random_state": seed})
            nomon_model.fit(
                X_nomon.loc[train_idx],
                y.loc[train_idx],
                X_val=X_nomon.loc[val_idx] if len(val_idx) > 0 else None,
                y_val=y.loc[val_idx] if len(val_idx) > 0 else None,
            )
            nomon_probs = nomon_model.predict_proba(X_nomon.loc[test_idx])
            if nomon_probs.ndim == 2 and nomon_probs.shape[1] == 2:
                nomon_probs = nomon_probs[:, 1]
            nomon_lift = compute_lift_analysis(
                y_test.values, nomon_probs, reference_prevalence=ucmr5_reference_prevalence
            )

        combined: dict[str, Any] = {
            "full_model": full_lift,
            "monitoring_free": nomon_lift,
            "monitoring_dependence": monitoring_dep,
        }
        out_path = output_dir / "lift_analysis.json"
        out_path.write_text(json.dumps(combined, indent=2, default=str))
        logger.info("Lift analysis saved to %s", out_path)
    except ImportError:
        logger.info("Lift analysis modules not available — skipping")
    except (OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Lift analysis failed", exc_info=True)
        if is_strict():
            raise


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------


def _run_external_validation(
    wq_df: Any,
    feature_dfs: list[Any],
    fitted_models: dict[str, tuple[Any, str, float]],
    downloaded: dict[str, Path],
    output_dir: Path,
) -> None:
    """Run external validation on state databases.

    Loads state DB Parquet files from the download cache and evaluates
    the fitted T1 model on each independent state dataset.
    """
    import pandas as pd

    if "T1" not in fitted_models:
        logger.info("No T1 fitted model — skipping external validation")
        return

    try:
        from aquacontam.benchmark.external_validation import run_external_validation
    except ImportError:
        logger.info("External validation module not available — skipping")
        return

    model, _, _ = fitted_models["T1"]

    # Load UCMR5 data for overlap detection
    ucmr5_path = downloaded.get("ucmr5")
    ucmr5_data = (
        pd.read_parquet(ucmr5_path) if ucmr5_path and ucmr5_path.exists() else pd.DataFrame()
    )

    # Build state DataFrames from downloaded sources
    state_sources = ["mi_mpart", "ca_geotracker", "nj_dep", "nc_deq", "mo_dnr", "wa_doh"]
    state_dfs: dict[str, pd.DataFrame] = {}
    for name in state_sources:
        path = downloaded.get(name)
        if path and path.exists():
            state_dfs[name] = pd.read_parquet(path)
            logger.info("Loaded state DB %s: %d rows", name, len(state_dfs[name]))

    if not state_dfs:
        logger.info("No state DB data available — skipping external validation")
        return

    try:
        results = run_external_validation(model, ucmr5_data, state_dfs, feature_dfs)
        out_path = output_dir / "external_validation.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("External validation saved to %s", out_path)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("External validation failed", exc_info=True)
        if is_strict():
            raise

    # WQP training-region validation
    wqp_path = downloaded.get("wqp")
    if wqp_path and wqp_path.exists():
        try:
            from aquacontam.benchmark.external_validation import (
                run_wqp_regional_validation,
            )

            wqp_data = pd.read_parquet(wqp_path)
            wqp_results = run_wqp_regional_validation(model, wqp_data, ucmr5_data, feature_dfs)
            wqp_out = output_dir / "wqp_regional_validation.json"
            wqp_out.write_text(json.dumps(wqp_results, indent=2, default=str))
            logger.info("WQP regional validation saved to %s", wqp_out)
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("WQP regional validation failed", exc_info=True)
            if is_strict():
                raise


# ---------------------------------------------------------------------------
# Causal deconfounding (DML)
# ---------------------------------------------------------------------------


def _run_causal_deconfounding(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run causal deconfounding analysis via Double Machine Learning.

    Estimates causal effects of environmental features after orthogonalizing
    against monitoring intensity confounders.
    """
    try:
        import pandas as pd

        from aquacontam.analysis.causal_deconfounding import (
            _build_feature_to_category,
            category_displacement_test,
            causal_feature_analysis,
            compare_dml_shap_rankings,
            train_deconfounded_model,
        )
        from aquacontam.features.assembly import (
            aggregate_to_system_level,
            drop_leakage_columns,
        )

        # Build T1 feature matrix with train-only imputation
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, _ = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)

        # Load confounder list from experiment config
        from aquacontam._config import load_experiment_config

        exp_cfg = load_experiment_config()
        abl_cfg = exp_cfg.get("ablation", {})
        causal_cfg = abl_cfg.get("causal_analysis", {})
        confounders = causal_cfg.get(
            "confounders",
            [
                "n_samples",
                "population_served",
                "log_population_served",
                "mean_detection_limit",
            ],
        )
        n_folds = causal_cfg.get("n_folds", 5)

        # Load SHAP importances if available (try multiple sources)
        shap_importances = None
        for shap_filename in ("feature_importance.json", "shap_T1.json"):
            shap_path = output_dir / shap_filename
            if not shap_path.exists():
                continue
            import json as _json

            shap_data = _json.loads(shap_path.read_text())
            if isinstance(shap_data, dict):
                # shap_T1.json has nested "mean_abs_shap" key
                if "mean_abs_shap" in shap_data and isinstance(shap_data["mean_abs_shap"], dict):
                    shap_importances = pd.Series(shap_data["mean_abs_shap"])
                else:
                    shap_importances = pd.Series(shap_data)
                break

        logger.info("Running causal deconfounding with confounders: %s", confounders)

        # Causal feature analysis
        causal_df = causal_feature_analysis(
            X,
            y,
            confounders,
            shap_importances=shap_importances,
            n_folds=n_folds,
            seed=seed,
        )
        causal_path = output_dir / "causal_deconfounding.json"
        causal_df.to_json(causal_path, orient="records", indent=2)
        logger.info(
            "Causal deconfounding results saved to %s (%d features)",
            causal_path,
            len(causal_df),
        )

        # Scale-free DML-vs-SHAP rank comparison
        demo_prefixes = abl_cfg.get("feature_categories", {}).get("demographics", [])
        rank_comparison = compare_dml_shap_rankings(
            causal_df,
            demographic_prefixes=demo_prefixes if demo_prefixes else None,
        )
        rank_path = output_dir / "dml_shap_rank_comparison.json"
        rank_path.write_text(json.dumps(rank_comparison, indent=2))
        logger.info(
            "DML-vs-SHAP rank comparison: Spearman rho=%.3f, n=%d features",
            rank_comparison.get("spearman_rho", float("nan")),
            rank_comparison.get("n_features", 0),
        )

        # Category-level displacement permutation test
        feature_categories = abl_cfg.get("feature_categories", {})
        if feature_categories and rank_comparison.get("feature_ranks"):
            feat_names = [r["feature"] for r in rank_comparison["feature_ranks"]]
            feat_to_cat = _build_feature_to_category(feat_names, feature_categories)
            cat_test_df = category_displacement_test(
                rank_comparison, feat_to_cat, n_permutations=10_000, seed=seed
            )
            cat_test_path = output_dir / "category_displacement_test.csv"
            cat_test_df.to_csv(cat_test_path, index=False)
            sig_cats = cat_test_df.loc[cat_test_df["significant_fdr"], "category"].tolist()
            logger.info(
                "Category displacement test: %d/%d categories significant (FDR<0.05): %s",
                len(sig_cats),
                len(cat_test_df),
                ", ".join(sig_cats) if sig_cats else "none",
            )

        # Deconfounded AUROC
        deconf_result = train_deconfounded_model(X, y, confounders, n_folds=n_folds, seed=seed)
        deconf_path = output_dir / "deconfounded_auroc.json"
        deconf_path.write_text(json.dumps(deconf_result, indent=2))
        logger.info(
            "Deconfounded AUROC: %.3f (original: %.3f)",
            deconf_result["deconfounded_auroc"],
            deconf_result["original_auroc"],
        )

    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Failed to run causal deconfounding", exc_info=True)
        if is_strict():
            raise
