"""Results-based analysis functions.

Functions in this module operate on pre-computed results (predictions,
probabilities, metadata) and fitted models.  They do **not** accept
``wq_df`` or ``feature_dfs`` and therefore never re-train models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from aquacontam.pipeline._strict import is_strict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial autocorrelation
# ---------------------------------------------------------------------------


def _run_spatial_autocorrelation(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run Moran's I spatial autocorrelation on model residuals.

    Uses stored test-set predictions and coordinates from results metadata.
    """
    import numpy as np

    from aquacontam.analysis.spatial_autocorrelation import analyze_spatial_autocorrelation

    sa_results: list[dict[str, Any]] = []

    for r in results:
        task = r.get("task", "")
        meta = r.get("metadata", {})
        y_true = meta.get("y_true")
        y_prob = meta.get("y_prob")
        latitudes = meta.get("latitudes")
        longitudes = meta.get("longitudes")
        split_labels = meta.get("split_labels")

        if y_true is None or y_prob is None or latitudes is None or longitudes is None:
            continue

        try:
            report = analyze_spatial_autocorrelation(
                y_true=np.asarray(y_true),
                y_prob_or_pred=np.asarray(y_prob),
                latitude=np.asarray(latitudes),
                longitude=np.asarray(longitudes),
                split_labels=np.asarray(split_labels) if split_labels else None,
                threshold_km=50.0,
            )

            overall = report.get("overall")
            entry: dict[str, Any] = {
                "task": task,
                "model": r.get("model", ""),
            }
            if overall is not None:
                entry["morans_i"] = overall.statistic
                entry["morans_i_expected"] = overall.expected
                entry["morans_i_z"] = overall.z_score
                entry["morans_i_p"] = overall.p_value
                entry["n"] = overall.n

            splits = report.get("splits", {})
            for split_name, sr in splits.items():
                if sr is not None:
                    entry[f"morans_i_{split_name}"] = sr.statistic
                    entry[f"morans_i_{split_name}_p"] = sr.p_value

            # Multi-threshold decay profile
            multi = report.get("multi_threshold", [])
            if multi:
                entry["multi_threshold"] = multi

            sa_results.append(entry)
            logger.info(
                "  Moran's I for %s/%s: I=%.4f (p=%.4f)",
                task,
                r.get("model", ""),
                entry.get("morans_i", float("nan")),
                entry.get("morans_i_p", float("nan")),
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning(
                "Failed spatial autocorrelation for %s/%s",
                task,
                r.get("model", ""),
                exc_info=True,
            )
            if is_strict():
                raise

    if sa_results:
        out_path = output_dir / "spatial_autocorrelation.json"
        out_path.write_text(json.dumps(sa_results, indent=2, default=str))
        logger.info("Spatial autocorrelation results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


def _run_calibration_analysis(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Compute calibration metrics (ECE, Brier, reliability diagram data).

    Uses stored test-set predictions from results metadata.
    """
    import numpy as np

    from aquacontam.benchmark.metrics import (
        calibration_curve_data,
        expected_calibration_error,
    )

    cal_results: list[dict[str, Any]] = []

    for r in results:
        task = r.get("task", "")
        meta = r.get("metadata", {})

        if task == "T2":  # Skip regression
            continue

        y_true = meta.get("y_true")
        y_prob = meta.get("y_prob")

        if y_true is None or y_prob is None:
            continue

        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_prob_arr = np.asarray(y_prob, dtype=float)

            ece_result = expected_calibration_error(y_true_arr, y_prob_arr)
            curve_data = calibration_curve_data(y_true_arr, y_prob_arr)

            # Brier score
            brier = float(np.mean((y_true_arr - y_prob_arr) ** 2))

            entry = {
                "task": task,
                "model": r.get("model", ""),
                "ece": ece_result["ece"],
                "mce": ece_result["mce"],
                "brier_score": brier,
                "calibration_curve": curve_data,
            }
            cal_results.append(entry)
            logger.info(
                "  Calibration for %s/%s: ECE=%.4f, Brier=%.4f",
                task,
                r.get("model", ""),
                ece_result["ece"],
                brier,
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning(
                "Failed calibration for %s/%s",
                task,
                r.get("model", ""),
                exc_info=True,
            )
            if is_strict():
                raise

    if cal_results:
        out_path = output_dir / "calibration_analysis.json"
        out_path.write_text(json.dumps(cal_results, indent=2, default=str))
        logger.info("Calibration analysis saved to %s", out_path)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _run_bootstrap_cis(
    results: list[dict[str, Any]],
    output_dir: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Add bootstrap confidence intervals to all classification results.

    Reads stored test-set predictions from results metadata and computes
    bootstrap CIs. Updates results in-place and returns them.

    Parameters
    ----------
    results : list[dict]
        Results from ``_train_and_evaluate``.
    output_dir : Path
        Results output directory.
    n_bootstrap : int
        Number of bootstrap iterations (default 1000).
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        Updated results with bootstrap CIs in metadata.
    """
    from aquacontam.benchmark.metrics import bootstrap_classification_metrics

    for r in results:
        task = r.get("task", "")
        meta = r.get("metadata", {})

        # Skip regression tasks and results that already have CIs
        if task == "T2" or "bootstrap_ci" in meta:
            continue

        # Bootstrap CIs require stored test predictions. y_pred may be absent
        # (some task result builders store only y_true/y_prob); derive it from
        # the probabilities so AUROC/AUPRC CIs (which use y_prob) can still be
        # computed. See PR #41: the T4 retrain produces rows with y_prob only.
        import numpy as np

        y_true = meta.get("y_true")
        y_pred = meta.get("y_pred")
        y_prob = meta.get("y_prob")

        if y_pred is None and y_prob is not None:
            y_pred = (np.asarray(y_prob) >= 0.5).astype(int)

        if y_true is None or y_pred is None:
            continue

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        y_prob_arr = np.asarray(y_prob) if y_prob is not None else None

        try:
            ci_results = bootstrap_classification_metrics(
                y_true_arr,
                y_pred_arr,
                y_prob_arr,
                metrics=["auroc", "auprc"],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            meta["bootstrap_ci"] = ci_results
            r["metadata"] = meta
            logger.info(
                "  Bootstrap CIs for %s/%s: AUROC [%.3f-%.3f], AUPRC [%.3f-%.3f]",
                task,
                r.get("model", ""),
                ci_results["auroc"]["ci_lower"],
                ci_results["auroc"]["ci_upper"],
                ci_results["auprc"]["ci_lower"],
                ci_results["auprc"]["ci_upper"],
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning(
                "Failed to compute bootstrap CIs for %s/%s",
                task,
                r.get("model", ""),
                exc_info=True,
            )
            if is_strict():
                raise

    # Write separate bootstrap_ci.json summary
    ci_rows: list[dict[str, Any]] = []
    for r in results:
        meta = r.get("metadata", {})
        ci = meta.get("bootstrap_ci")
        if ci is None:
            continue
        for metric_name, metric_ci in ci.items():
            ci_rows.append(
                {
                    "task": r.get("task", ""),
                    "model": r.get("model", ""),
                    "metric": metric_name,
                    "point_estimate": metric_ci.get("point"),
                    "ci_lower": metric_ci.get("ci_lower"),
                    "ci_upper": metric_ci.get("ci_upper"),
                    "std": metric_ci.get("std"),
                }
            )
    if ci_rows:
        ci_path = output_dir / "bootstrap_ci.json"
        ci_path.write_text(json.dumps(ci_rows, indent=2))
        logger.info("Bootstrap CI summary written to %s (%d entries)", ci_path, len(ci_rows))

    return results


# ---------------------------------------------------------------------------
# Model comparison (DeLong tests)
# ---------------------------------------------------------------------------


def _run_model_comparison(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run pairwise DeLong tests between all models on each task.

    Groups results by task, extracts stored test-set predictions, and
    runs pairwise DeLong tests with FDR correction.

    Parameters
    ----------
    results : list[dict]
        Results from ``_train_and_evaluate``.
    output_dir : Path
        Results output directory.
    """
    import numpy as np

    from aquacontam.benchmark.metrics import compare_models_pairwise

    # Group results by task
    task_groups: dict[str, dict[str, Any]] = {}
    for r in results:
        task = r.get("task", "")
        model = r.get("model", "")
        meta = r.get("metadata", {})

        # Skip regression tasks
        if task == "T2":
            continue

        y_true = meta.get("y_true")
        y_prob = meta.get("y_prob")

        if y_true is None or y_prob is None:
            continue

        if task not in task_groups:
            task_groups[task] = {}

        # Extract test-set predictions only
        split_labels = meta.get("split_labels", [])
        if split_labels:
            y_arr = np.asarray(y_true)
            p_arr = np.asarray(y_prob)
            s_arr = np.asarray(split_labels)
            test_mask = s_arr == "test"
            if test_mask.any():
                task_groups[task][model] = {
                    "y_true": y_arr[test_mask],
                    "y_prob": p_arr[test_mask],
                }
            else:
                task_groups[task][model] = {"y_true": y_arr, "y_prob": p_arr}
        else:
            task_groups[task][model] = {
                "y_true": np.asarray(y_true),
                "y_prob": np.asarray(y_prob),
            }

    all_comparisons: dict[str, list[dict[str, Any]]] = {}

    for task, model_data in task_groups.items():
        if len(model_data) < 2:
            logger.info("Skipping DeLong comparison for %s (< 2 models)", task)
            continue

        # All models must share the same y_true -- use first model's
        first_model = next(iter(model_data))
        y_true = model_data[first_model]["y_true"]
        model_probs = {name: data["y_prob"] for name, data in model_data.items()}

        try:
            comparison_df = compare_models_pairwise(y_true, model_probs, method="delong")
            comparison_records = comparison_df.to_dict(orient="records")
            all_comparisons[task] = cast(list[dict[str, Any]], comparison_records)

            # Log significant differences
            n_sig = int(comparison_df["significant"].sum())
            logger.info(
                "DeLong comparison for %s: %d pairs, %d significant (FDR<0.05)",
                task,
                len(comparison_df),
                n_sig,
            )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Failed DeLong comparison for %s", task, exc_info=True)
            if is_strict():
                raise

    if all_comparisons:
        out_path = output_dir / "model_comparison.json"
        out_path.write_text(json.dumps(all_comparisons, indent=2, default=str))
        logger.info("Model comparisons saved to %s", out_path)


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------


def _run_shap_analysis(
    results: list[dict[str, Any]],
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
) -> None:
    """Compute SHAP values for best classification models on T1 and T4."""
    try:
        from aquacontam.analysis.interpretability import compute_shap_values
    except ImportError:
        logger.info("SHAP not available (install shap) — skipping SHAP analysis")
        return

    _trivial_models = {"dummy_classifier", "logistic_regression"}

    for task_name in ("T1", "T4"):
        if task_name not in fitted_models:
            continue
        model, task_type, _ = fitted_models[task_name]
        if task_type != "classification":
            continue
        if model.name in _trivial_models:
            logger.warning(
                "Skipping SHAP for %s/%s — trivial model produces uninformative values",
                task_name,
                model.name,
            )
            continue

        # Get test data from results metadata
        for r in results:
            if r.get("task") != task_name or r.get("model") != model.name:
                continue
            meta = r.get("metadata", {})
            X_test = meta.get("X_test")
            if X_test is None:
                continue

            try:
                import pandas as pd

                X_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
                shap_df = compute_shap_values(model, X_df)
                shap_dict = {
                    "task": task_name,
                    "model": model.name,
                    "mean_abs_shap": shap_df.abs().mean().sort_values(ascending=False).to_dict(),
                }
                out_path = output_dir / f"shap_{task_name}.json"
                out_path.write_text(json.dumps(shap_dict, indent=2, default=str))
                logger.info("SHAP values exported for %s/%s", task_name, model.name)
            except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
                logger.warning("SHAP failed for %s/%s", task_name, model.name, exc_info=True)
                if is_strict():
                    raise
            break


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------


def _run_conformal_analysis(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run conformal prediction analysis at multiple alpha levels.

    Uses validation set for calibration and test set for evaluation,
    following proper split conformal inference. Reuses the
    ``ConformalClassifier`` from ``aquacontam.calibration.conformal``.
    """
    import numpy as np

    try:
        from aquacontam.calibration.conformal import ConformalClassifier
    except ImportError:
        logger.info("Calibration module not available — skipping conformal analysis")
        return

    conformal_results: list[dict[str, Any]] = []

    for r in results:
        task = r.get("task", "")
        meta = r.get("metadata", {})
        if task == "T2":
            continue

        y_true = meta.get("y_true")
        y_prob = meta.get("y_prob")
        split_labels = meta.get("split_labels")
        if y_true is None or y_prob is None or split_labels is None:
            continue

        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_prob_arr = np.asarray(y_prob, dtype=float)
            split_arr = np.asarray(split_labels)

            # Separate val (calibration) and test (evaluation) by split labels
            val_mask = split_arr == "val"
            test_mask = split_arr == "test"

            if val_mask.sum() < 10 or test_mask.sum() < 10:
                logger.info(
                    "Insufficient val/test samples for conformal: %s/%s (val=%d, test=%d)",
                    task,
                    r.get("model", ""),
                    val_mask.sum(),
                    test_mask.sum(),
                )
                continue

            cal_y = y_true_arr[val_mask]
            cal_probs = y_prob_arr[val_mask]
            test_y = y_true_arr[test_mask]
            test_probs = y_prob_arr[test_mask]

            # Build a lightweight wrapper so ConformalClassifier can call
            # predict_proba -- it expects (n, 2) output.  The wrapper
            # treats its input X as raw P(class=1) probabilities so the
            # same instance works for both calibration and test data.
            class _ProbaWrapper:
                """Wrap precomputed probabilities as a model-like object."""

                def predict_proba(self, X: Any) -> np.ndarray:
                    p = np.asarray(X, dtype=float)
                    return np.column_stack([1.0 - p, p])

            for alpha in (0.05, 0.10, 0.20):
                # Calibrate on val, evaluate on test
                cc = ConformalClassifier(_ProbaWrapper(), alpha=alpha)  # type: ignore[arg-type]
                cc.calibrate(cal_probs, cal_y)

                stats = cc.coverage_and_set_size(test_probs, test_y)

                conformal_results.append(
                    {
                        "task": task,
                        "model": r.get("model", ""),
                        "alpha": alpha,
                        "target_coverage": 1.0 - alpha,
                        "coverage": stats["coverage"],
                        "avg_set_size": stats["avg_set_size"],
                        "singleton_frac": stats["singleton_frac"],
                        "both_classes_frac": stats["both_classes_frac"],
                    }
                )
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Conformal failed for %s/%s", task, r.get("model", ""), exc_info=True)
            if is_strict():
                raise

    if conformal_results:
        out_path = output_dir / "conformal_results.json"
        out_path.write_text(json.dumps(conformal_results, indent=2, default=str))
        logger.info("Conformal results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------


def _run_power_analysis(
    output_dir: Path,
) -> None:
    """Compute post-hoc statistical power for key paper tests.

    Reads DeLong, equity, DML, and ablation results from JSON files
    and reports achieved power for each statistical test.
    """
    from aquacontam.analysis.power_analysis import (
        compute_auroc_comparison_power,
        compute_regression_coefficient_power,
        summarize_power,
    )

    try:
        power_summary = summarize_power(output_dir)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to compute base power summary", exc_info=True)
        if is_strict():
            raise
        power_summary = {}

    # Extend with ablation bootstrap power (from feature_ablation.json)
    ablation_power: list[dict[str, Any]] = []
    ablation_path = output_dir / "feature_ablation.json"
    if ablation_path.exists():
        try:
            ablation_data = json.loads(ablation_path.read_text())
            for entry in ablation_data:
                sig = entry.get("significance", {})
                for metric_name, sig_dict in sig.items():
                    delta = sig_dict.get("delta", 0.0)
                    ci_lower = sig_dict.get("ci_lower", 0.0)
                    ci_upper = sig_dict.get("ci_upper", 0.0)
                    # Approximate SE from bootstrap CI width
                    ci_width = ci_upper - ci_lower
                    se_approx = ci_width / (2 * 1.96) if ci_width > 0 else 1e-10
                    n_test = entry.get("n_test", 2000)
                    pwr = compute_regression_coefficient_power(abs(delta), se_approx, n_test)
                    ablation_power.append(
                        {
                            "category": entry.get("category", ""),
                            "task": entry.get("task", ""),
                            "metric": metric_name,
                            "delta": delta,
                            "power": pwr["power"],
                        }
                    )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Could not parse feature_ablation.json for power", exc_info=True)

    # Extend with LORO fold power
    loro_power: list[dict[str, Any]] = []
    loro_path = output_dir / "loro_cv.json"
    if loro_path.exists():
        try:
            loro_data = json.loads(loro_path.read_text())
            for task_name, task_data in loro_data.items():
                for model_name, model_data in task_data.items():
                    # loro_cv.json is flat (mean_auroc lives directly on the model dict);
                    # fall back to model_data, not {}, so real values are read rather than
                    # sentinel defaults (matches the multi-seed handler below).
                    summary = model_data.get("summary", model_data)
                    mean_auroc = summary.get("mean_auroc", 0.0)
                    std_auroc = summary.get("std_auroc", 0.01)
                    n_folds = summary.get("n_folds", 10)
                    # Power to detect that AUROC > 0.5 (chance)
                    if std_auroc > 0 and n_folds > 1:
                        pwr = compute_auroc_comparison_power(
                            mean_auroc, 0.5, n_folds * 200, n_folds * 200
                        )
                        loro_power.append(
                            {
                                "task": task_name,
                                "model": model_name,
                                "mean_auroc": mean_auroc,
                                "std_auroc": std_auroc,
                                "n_folds": n_folds,
                                "power_vs_chance": pwr["power"],
                            }
                        )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Could not parse loro_cv.json for power", exc_info=True)

    result = {
        "core_tests": power_summary,
        "ablation_tests": ablation_power,
        "loro_tests": loro_power,
    }

    out_path = output_dir / "power_analysis.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Power analysis saved to %s", out_path)


# ---------------------------------------------------------------------------
# Sensitivity bounds (Rosenbaum + E-values)
# ---------------------------------------------------------------------------


def _compute_e_value(beta: float) -> float:
    """Approximate E-value (VanderWeele & Ding 2017) for a standardized DML effect.

    ``beta`` is the SEMI-STANDARDIZED DML coefficient (effect on the residualized
    outcome per 1 standard deviation of the residualized feature; see
    ``causal_feature_analysis``), so this E-value is scale-invariant -- it no longer
    depends on the feature's measurement units (the previous ``exp(|beta|)`` made a
    metre-scale distance and the same effect in km give different E-values).

    Following VanderWeele & Ding (2017), a standardized effect ``d`` is mapped to an
    approximate risk ratio ``RR = exp(0.91 * |d|)``, then
    ``E = RR + sqrt(RR * (RR - 1))``. This is an APPROXIMATE E-value for a
    standardized continuous exposure and should be interpreted as such.
    """
    import numpy as np

    rr = np.exp(0.91 * abs(beta))
    if rr <= 1.0:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def _run_sensitivity_bounds(
    output_dir: Path,
) -> None:
    """Compute Rosenbaum sensitivity bounds and E-values for DML estimates.

    Loads causal deconfounding results and computes:
    - Rosenbaum Gamma* tipping point for each significant DML coefficient
    - E-values (VanderWeele & Ding 2017) for each coefficient
    """
    dml_path = output_dir / "causal_deconfounding.json"
    if not dml_path.exists():
        logger.info("No causal_deconfounding.json found — skipping sensitivity bounds")
        return

    try:
        dml_data = json.loads(dml_path.read_text())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse causal_deconfounding.json", exc_info=True)
        if is_strict():
            raise
        return

    from aquacontam.analysis.causal_deconfounding import sensitivity_analysis

    # Extract per-feature DML results — may be a flat list or dict with "per_feature" key
    if isinstance(dml_data, list):
        dml_results: list[dict[str, Any]] = dml_data
    else:
        dml_results = dml_data.get("per_feature", [])
    if not dml_results:
        logger.info("No per-feature DML results — skipping sensitivity bounds")
        return

    try:
        bounds_by_feature = sensitivity_analysis(dml_results)
    except (ValueError, ArithmeticError):
        logger.warning("Rosenbaum sensitivity analysis failed", exc_info=True)
        if is_strict():
            raise
        return

    output: dict[str, Any] = {}
    for feat, bounds_df in bounds_by_feature.items():
        # Find Gamma* (tipping point): smallest gamma where significant becomes False
        gamma_star = None
        for _, row in bounds_df.iterrows():
            if not row["significant"]:
                gamma_star = float(row["gamma"])
                break

        # Get the causal effect for E-value computation
        feat_row = next((r for r in dml_results if r["feature"] == feat), None)
        beta = feat_row["causal_effect"] if feat_row else 0.0
        e_val = _compute_e_value(beta)

        output[feat] = {
            "gamma_star": gamma_star,
            "e_value": e_val,
            "causal_effect": beta,
            "std_error": feat_row["std_error"] if feat_row else None,
            "bounds_table": bounds_df.to_dict(orient="records"),
        }

    output["_metadata"] = {
        "feature_scaling": "semi_standardized_per_1sd",
        "e_value_method": "vanderweele_approx_exp_0.91",
        "note": (
            "causal_effect is the semi-standardized DML coefficient (per 1-SD of the "
            "residualized feature); E-values are approximate and scale-invariant."
        ),
    }
    out_path = output_dir / "sensitivity_bounds.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Sensitivity bounds saved to %s (%d features)", out_path, len(output) - 1)


# ---------------------------------------------------------------------------
# Detection-only source ablation
# ---------------------------------------------------------------------------


def _run_detection_only_ablation(
    wq_df: Any,
    results: list[dict[str, Any]],
    output_dir: Path,
    fitted_models: dict[str, tuple[Any, str, float]] | None = None,
) -> None:
    """Evaluate T1 model performance excluding detection-only data sources.

    Detection-only sources (MI MPART, SDWIS, WA DOH) report only
    detected/non-detected outcomes without concentration data. This
    analysis tests whether their inclusion inflates classification metrics.

    Uses ``X_test`` from result metadata and re-predicts via ``fitted_models``
    when available, so that the **full** test set (including systems without
    geocoded coordinates) is evaluated. Falls back to the coordinate-filtered
    ``y_true``/``y_prob`` when ``fitted_models`` is not provided.
    """
    import numpy as np
    import pandas as pd

    from aquacontam.analysis.ablation import evaluate_excluding_detection_only
    from aquacontam.benchmark._utils import get_proba

    # Map pwsid → source from the merged WQ data
    if "source" not in wq_df.columns:
        logger.warning("No 'source' column in wq_df — skipping detection-only ablation")
        return

    pwsid_source = wq_df.groupby("pwsid")["source"].first()

    ablation_results: list[dict[str, Any]] = []

    for r in results:
        task = r.get("task", "")
        if task != "T1":
            continue

        meta = r.get("metadata", {})
        X_test = meta.get("X_test")
        model_name = r.get("model", "")

        # Preferred path: re-predict on full X_test via fitted_models.
        # Use an explicit DataFrame check: ``hasattr(x, "index")`` is True for
        # plain strings (str.index), which would slip a non-frame past the guard.
        if isinstance(X_test, pd.DataFrame) and not X_test.empty:
            pwsids = list(X_test.index)
            X_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

            # Find matching model in fitted_models or result model
            model_obj = None
            if fitted_models and task in fitted_models:
                fm = fitted_models[task][0]
                # Use fitted_models only if it matches this result's model,
                # or if we have no other option
                if fm.name == model_name:
                    model_obj = fm

            if model_obj is not None:
                try:
                    y_prob_arr = np.asarray(get_proba(model_obj, X_df))
                    # Reconstruct y_true from splits
                    from aquacontam.features.assembly import (
                        aggregate_to_system_level,
                    )

                    splits = meta.get("_splits")
                    if splits and "test" in splits:
                        y_true_arr = pd.Series(splits["test"][1].values, index=X_df.index)
                    else:
                        # Reconstruct y_true from system aggregation
                        analyte = meta.get("analyte", "PFOS")
                        sys_agg = aggregate_to_system_level(wq_df, analyte, target="detected")
                        y_true_arr = sys_agg.reindex(X_df.index)["target"].fillna(0)
                except (ValueError, KeyError, TypeError, IndexError):
                    logger.debug(
                        "Re-prediction failed for %s/%s — falling back",
                        task,
                        model_name,
                        exc_info=True,
                    )
                    model_obj = None

            if model_obj is None:
                # Fall back to coordinate-filtered y_true/y_prob
                y_true = meta.get("y_true")
                y_prob = meta.get("y_prob")
                split_labels = meta.get("split_labels")
                if y_true is None or y_prob is None:
                    continue
                # Use only test-split entries from spatial_acc
                if split_labels:
                    mask = [s == "test" for s in split_labels]
                    y_true_arr = pd.Series(np.asarray(y_true)[mask])
                    y_prob_arr = np.asarray(y_prob)[mask]
                    # pwsids from spatial_acc are coordinate-filtered
                    # — detection-only sources may not appear
                    from aquacontam.benchmark._utils import get_system_coordinates

                    sys_coords = get_system_coordinates(wq_df)
                    valid = pd.Index(pwsids).isin(sys_coords.index)
                    pwsids = [p for p, v in zip(pwsids, valid, strict=True) if v]
                else:
                    y_true_arr = pd.Series(np.asarray(y_true))
                    y_prob_arr = np.asarray(y_prob)
        else:
            # No X_test available — use coordinate-filtered metadata
            y_true = meta.get("y_true")
            y_prob = meta.get("y_prob")
            pwsids_meta = meta.get("pwsids")
            if y_true is None or y_prob is None:
                continue
            y_true_arr = pd.Series(np.asarray(y_true))
            y_prob_arr = np.asarray(y_prob)
            if pwsids_meta is not None:
                pwsids = list(pwsids_meta)
            else:
                logger.warning("No pwsids or X_test for %s/%s — skipping", task, model_name)
                continue

        if len(y_true_arr) != len(pwsids):
            logger.warning(
                "Size mismatch for %s/%s: %d predictions vs %d pwsids — skipping",
                task,
                model_name,
                len(y_true_arr),
                len(pwsids),
            )
            continue

        source_labels = pd.Series(pwsids).map(pwsid_source).fillna("unknown")

        try:
            result = evaluate_excluding_detection_only(y_true_arr, y_prob_arr, source_labels)
            result["task"] = task
            result["model"] = model_name
            ablation_results.append(result)

            logger.info(
                "  Detection-only ablation %s/%s: all=%.3f, filtered=%.3f (delta=%.3f)",
                task,
                model_name,
                result["all_sources"].get("auroc", 0.0),
                result.get("excluding_detection_only", {}).get("auroc", 0.0),
                result.get("delta", {}).get("auroc", 0.0),
            )
        except (ValueError, RuntimeError, ArithmeticError):
            logger.warning(
                "Detection-only ablation failed for %s/%s",
                task,
                model_name,
                exc_info=True,
            )
            if is_strict():
                raise

    if ablation_results:
        out_path = output_dir / "detection_only_ablation.json"
        out_path.write_text(json.dumps(ablation_results, indent=2, default=str))
        logger.info("Detection-only ablation saved to %s", out_path)


# ---------------------------------------------------------------------------
# SHAP interaction analysis
# ---------------------------------------------------------------------------

_SPATIAL_PREFIXES = (
    "dist_",
    "count_",
    "density_",
    "knn_",
    "kernel_",
    "ring_",
    "land_use_",
    "nlcd_",
    "aquifer_",
)
_SYSTEM_PREFIXES = ("population_", "system_type_", "source_water_", "owner_type_")
_DEMO_PREFIXES = ("pct_", "median_", "ej_")


def _classify_feature(name: str) -> str:
    """Classify a feature name into spatial, system, demographic, or other."""
    if any(name.startswith(p) for p in _SPATIAL_PREFIXES):
        return "spatial"
    if any(name.startswith(p) for p in _SYSTEM_PREFIXES):
        return "system"
    if any(name.startswith(p) for p in _DEMO_PREFIXES):
        return "demographic"
    return "other"


def _run_shap_interaction_analysis(
    results: list[dict[str, Any]],
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
) -> None:
    """Compute SHAP interaction values for the best tree-based T1 model.

    Identifies top feature interaction pairs and classifies them by
    category (spatial x system, spatial x spatial, etc.).
    """
    try:
        from aquacontam.analysis.interpretability import compute_shap_interactions
    except ImportError:
        logger.info("SHAP not available — skipping interaction analysis")
        return

    _trivial_models = {"dummy_classifier", "logistic_regression"}

    if "T1" not in fitted_models:
        logger.info("No fitted T1 model — skipping SHAP interaction analysis")
        return

    model, task_type, _ = fitted_models["T1"]
    if task_type != "classification" or model.name in _trivial_models:
        logger.info("T1 model %s not suitable for SHAP interactions", model.name)
        return

    # Get test data from results metadata
    for r in results:
        if r.get("task") != "T1" or r.get("model") != model.name:
            continue
        meta = r.get("metadata", {})
        X_test = meta.get("X_test")
        if X_test is None:
            continue

        try:
            import pandas as pd

            X_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            interaction_df = compute_shap_interactions(model, X_df, top_n=15)

            if interaction_df.empty:
                logger.info("No SHAP interactions computed")
                break

            # Classify interaction pairs by feature category
            pairs: list[dict[str, Any]] = []
            category_counts: dict[str, int] = {}
            for _, row in interaction_df.iterrows():
                cat_a = _classify_feature(str(row["feature_a"]))
                cat_b = _classify_feature(str(row["feature_b"]))
                pair_cat = f"{min(cat_a, cat_b)}_x_{max(cat_a, cat_b)}"
                category_counts[pair_cat] = category_counts.get(pair_cat, 0) + 1
                pairs.append(
                    {
                        "feature_a": str(row["feature_a"]),
                        "feature_b": str(row["feature_b"]),
                        "mean_abs_interaction": float(row["mean_abs_interaction"]),
                        "category": pair_cat,
                    }
                )

            result = {
                "task": "T1",
                "model": model.name,
                "top_interactions": pairs,
                "n_samples_used": min(len(X_df), 1000),
                "interaction_summary": category_counts,
            }
            out_path = output_dir / "shap_interactions_T1.json"
            out_path.write_text(json.dumps(result, indent=2, default=str))
            logger.info("SHAP interactions saved to %s (%d pairs)", out_path, len(pairs))
        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("SHAP interactions failed for T1/%s", model.name, exc_info=True)
            if is_strict():
                raise
        break


# ---------------------------------------------------------------------------
# Multi-seed inflation check
# ---------------------------------------------------------------------------


def _run_seed_inflation_check(
    output_dir: Path,
) -> None:
    """Check whether the reported seed=42 results are inflated vs multi-seed mean.

    Loads multi-seed stability results and compares the reported
    single-seed value to the multi-seed distribution, computing
    z-scores and maximum optimism.
    """
    import numpy as np

    ms_path = output_dir / "multi_seed_stability.json"
    results_path = output_dir / "results.json"

    if not ms_path.exists():
        logger.info("No multi_seed_stability.json — skipping inflation check")
        return
    if not results_path.exists():
        logger.info("No results.json — skipping inflation check")
        return

    try:
        ms_data = json.loads(ms_path.read_text())
        all_results = json.loads(results_path.read_text())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse multi-seed or results JSON", exc_info=True)
        if is_strict():
            raise
        return

    # Build lookup of reported AUROC from results.json
    reported: dict[str, dict[str, float]] = {}
    for r in all_results:
        task = r.get("task", "")
        model = r.get("model", "")
        metrics = r.get("metrics", {})
        auroc = metrics.get("auroc")
        if auroc is not None:
            reported.setdefault(task, {})[model] = float(auroc)

    inflation_results: dict[str, dict[str, dict[str, Any]]] = {}
    max_z = 0.0
    max_optimism_any = 0.0
    any_inflated = False

    for task_name, task_data in ms_data.items():
        inflation_results[task_name] = {}
        for model_name, model_data in task_data.items():
            # Handle both flat and nested structures:
            # Flat: {mean_auroc, std_auroc, auroc_per_seed, ...}
            # Nested: {summary: {mean_auroc, ...}, per_seed: {...}}
            summary = model_data.get("summary", model_data)
            mean_auroc = summary.get("mean_auroc", 0.0)
            std_auroc = summary.get("std_auroc", 0.01)

            # Match model names across JSON files (e.g. "xgboost" → "xgboost_classifier")
            task_reported = reported.get(task_name, {})
            reported_auroc = task_reported.get(model_name)
            if reported_auroc is None:
                # Try common suffixes
                for suffix in ("_classifier", "_regressor", "_default"):
                    reported_auroc = task_reported.get(model_name + suffix)
                    if reported_auroc is not None:
                        break
            if reported_auroc is None:
                continue

            # Skip if mean_auroc is 0 (missing data)
            if mean_auroc < 1e-6:
                continue

            # Compute z-score
            z_score = (reported_auroc - mean_auroc) / std_auroc if std_auroc > 1e-10 else 0.0

            # Extract per-seed AUROCs — may be list or dict
            per_seed = model_data.get("per_seed", {})
            auroc_per_seed = model_data.get("auroc_per_seed", [])
            if auroc_per_seed:
                seed_aurocs = [float(v) for v in auroc_per_seed]
            elif isinstance(per_seed, dict) and per_seed:
                seed_aurocs = [v.get("auroc", 0.0) for v in per_seed.values()]
            else:
                seed_aurocs = [mean_auroc]
            max_opt = float(np.max(seed_aurocs) - np.mean(seed_aurocs))

            is_inflated = abs(z_score) > 2.0

            inflation_results[task_name][model_name] = {
                "reported_seed": 42,
                "reported_auroc": reported_auroc,
                "mean_auroc_multi_seed": mean_auroc,
                "std_auroc_multi_seed": std_auroc,
                "inflation_z_score": float(z_score),
                "max_optimism": max_opt,
                "is_inflated": is_inflated,
            }

            max_z = max(max_z, abs(z_score))
            max_optimism_any = max(max_optimism_any, max_opt)
            if is_inflated:
                any_inflated = True

    result = {
        **inflation_results,
        "summary": {
            "any_inflation_detected": any_inflated,
            "max_z_score": float(max_z),
            "max_optimism_any_model": float(max_optimism_any),
        },
    }

    out_path = output_dir / "seed_inflation_check.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info(
        "Seed inflation check saved to %s (any_inflated=%s, max_z=%.2f)",
        out_path,
        any_inflated,
        max_z,
    )
