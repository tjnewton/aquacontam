"""Result export, webapp predictions, feature importances, and paper assets."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from aquacontam.pipeline._strict import is_strict

logger = logging.getLogger(__name__)


def export_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Export results as JSON and leaderboard submissions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge with existing results (update matching task/model, append new)
    results_path = output_dir / "results.json"
    existing: list[dict[str, Any]] = []
    if results_path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            existing = json.loads(results_path.read_text())

    # Index existing by (model, task) for dedup
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for r in existing:
        key = (r.get("model", ""), r.get("task", ""))
        merged[key] = r
    for r in results:
        key = (r.get("model", ""), r.get("task", ""))
        merged[key] = r

    all_results = list(merged.values())

    # Strip non-serializable keys (e.g. DataFrames) from metadata for JSON export.
    # Temporarily remove and restore so in-memory results retain X_test for SHAP.
    saved: list[dict[str, object]] = []
    for r in all_results:
        meta = r.get("metadata", {})
        saved.append({k: meta.pop(k) for k in ("X_test", "X_val") if k in meta})

    results_path.write_text(json.dumps(all_results, indent=2, default=str))

    # Restore stripped keys
    for r, s in zip(all_results, saved, strict=True):
        if s:
            r.get("metadata", {}).update(s)
    logger.info("Results written to %s (%d entries)", results_path, len(all_results))

    # Generate per-model leaderboard submissions
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    model_results: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        model_name = r["model"]
        if model_name not in model_results:
            model_results[model_name] = []
        entry: dict[str, Any] = {"task": r["task"], "metrics": r["metrics"]}
        for meta_key in ("analyte", "split", "fold"):
            if meta_key in r.get("metadata", {}):
                entry[meta_key] = r["metadata"][meta_key]
        model_results[model_name].append(entry)

    for model_name, task_results in model_results.items():
        submission = {
            "schema_version": "1.0",
            "model_name": model_name,
            "results": task_results,
        }
        sub_path = submissions_dir / f"{model_name}.json"
        sub_path.write_text(json.dumps(submission, indent=2, default=str))

    logger.info("Leaderboard submissions written to %s", submissions_dir)


def generate_webapp_predictions(
    wq_df: Any,
    feature_dfs: list[Any],
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
) -> None:
    """Generate Hive-partitioned Parquet predictions for the webapp.

    Uses fitted models from ``train_and_evaluate`` to produce prediction
    files that the Dash webapp can load directly.  Only T1, T2, and T4
    are supported (they have analyte mappings in ``inference/predict.py``).

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.
    fitted_models : dict[str, tuple[BaseModel, str]]
        ``{task_name: (model, task_type)}`` from training.
    output_dir : Path
        Root output directory; predictions go into ``output_dir / "predictions"``.
    """
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        build_system_geodataframe,
        drop_leakage_columns,
    )
    from aquacontam.inference.predict import generate_grid_predictions

    # Representative analytes per task — first default analyte
    task_analyte: dict[str, str] = {"T1": "PFOS", "T2": "PFOS", "T4": "lead"}
    task_targets: dict[str, str] = {
        "T1": "detected",
        "T2": "max_concentration",
        "T4": "action_level",
    }

    # Filter to tasks that have analyte mappings
    supported = {k: v for k, v in fitted_models.items() if k in task_analyte}
    if not supported:
        logger.info("No supported tasks (T1/T2/T4) in fitted models — skipping predictions")
        return

    predictions_dir = output_dir / "predictions"

    # Build system GeoDataFrame once for the grid
    systems_gdf = build_system_geodataframe(wq_df)
    base_grid_df = systems_gdf.reset_index()[["pwsid", "latitude", "longitude"]].rename(
        columns={"pwsid": "grid_id"}
    )

    # Generate predictions per task — each task needs its own analyte-specific
    # feature matrix because aggregate_to_system_level produces analyte-dependent
    # columns (n_samples, mean_detection_limit, etc.).
    for task_name, (model, _task_type, _threshold) in supported.items():
        analyte = task_analyte[task_name]
        target_type = task_targets[task_name]
        sys_targets = aggregate_to_system_level(wq_df, analyte, target=target_type)
        sys_targets = drop_leakage_columns(sys_targets)

        X_all, _, _ = assemble_feature_matrix(sys_targets, *feature_dfs)

        # Capture training-time columns from the model if available
        train_cols: list[str] | None = None
        if hasattr(model, "feature_names_in_"):
            train_cols = list(model.feature_names_in_)
        elif hasattr(model, "_model"):
            inner = model._model
            if hasattr(inner, "feature_names_in_"):
                train_cols = list(inner.feature_names_in_)

        # Align grid to this task's feature matrix index
        grid_df = base_grid_df[base_grid_df["grid_id"].isin(X_all.index)]

        # Reindex feature matrix to match grid ordering and align columns
        X_pred = X_all.loc[grid_df["grid_id"].to_numpy()]
        if train_cols is not None:
            X_pred = X_pred.reindex(columns=train_cols, fill_value=0.0)

        logger.info(
            "Generating predictions for %s: %d systems, %d features",
            task_name,
            len(X_pred),
            X_pred.shape[1],
        )

        generate_grid_predictions(
            models={task_name: model},
            feature_df=X_pred,
            grid=grid_df,
            output_dir=predictions_dir,
        )


def export_feature_importances(
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
) -> Any:
    """Export feature importances from tree-based fitted models.

    Parameters
    ----------
    fitted_models : dict[str, tuple[Any, str, float]]
        ``{task_name: (model, task_type, threshold)}`` from training.
    output_dir : Path
        Results output directory.

    Returns
    -------
    pd.Series | None
        Feature importance series, or None if no model supports it.
    """
    # Prefer T1 (PFAS detection), then T4 (heavy metals)
    for task_name in ("T1", "T4", "T2", "T3"):
        if task_name not in fitted_models:
            continue
        model, _, _ = fitted_models[task_name]
        try:
            importance = model.feature_importances()
            imp_dict = {str(k): float(v) for k, v in importance.items()}
            out_path = output_dir / "feature_importance.json"
            out_path.write_text(json.dumps(imp_dict, indent=2))
            logger.info(
                "Exported feature importances from %s/%s: %d features",
                task_name,
                model.name,
                len(imp_dict),
            )
            return importance
        except NotImplementedError:
            logger.info("Model %s does not support feature_importances()", model.name)
        except (ValueError, RuntimeError, TypeError, OSError):
            logger.warning(
                "Failed to export feature importances from %s", task_name, exc_info=False
            )
            if is_strict():
                raise

    logger.info("No fitted model supports feature_importances() — skipping export")
    return None


def export_icp_diagnostics(
    results: list[dict[str, Any]],
    fitted_models: dict[str, tuple[Any, str, float]],
    output_dir: Path,
    icp_models: dict[str, Any] | None = None,
) -> None:
    """Export ICP model diagnostics (adversary R-squared, training history, permutation importance).

    Parameters
    ----------
    results : list[dict]
        Benchmark results list.
    fitted_models : dict[str, tuple[Any, str, float]]
        ``{task_name: (model, task_type, threshold)}`` from training. Holds the
        tree SHAP model per task, which carries no ICP diagnostics.
    output_dir : Path
        Results output directory.
    icp_models : dict[str, Any] | None
        ``{task_name: fitted_icp_model}`` from :func:`train_and_evaluate`. This
        is the channel that actually carries the ICP's per-epoch history; when
        provided it is used in preference to scanning ``fitted_models`` (which
        never contains an ICP model because a tree model is chosen for SHAP).
    """
    diagnostics: dict[str, Any] = {}

    # Collect ICP results from benchmark
    icp_results = [r for r in results if r.get("model", "").startswith("icp")]
    if not icp_results:
        logger.info("No ICP results found — skipping ICP diagnostics export")
        return

    diagnostics["results"] = []
    for r in icp_results:
        diagnostics["results"].append(
            {
                "task": r.get("task"),
                "model": r.get("model"),
                "metrics": r.get("metrics", {}),
            }
        )

    # Collect diagnostics from the fitted ICP models. Prefer the dedicated
    # icp_models channel (train_and_evaluate retains the ICP per task); fall
    # back to scanning fitted_models for legacy callers (that scan never
    # matches, since the SHAP model is a tree model).
    if icp_models is not None:
        icp_by_task = dict(icp_models)
    else:
        icp_by_task = {t: m for t, (m, _tt, _th) in fitted_models.items()}

    for task_name, model in icp_by_task.items():
        if not getattr(model, "name", "").startswith("icp"):
            continue

        # Adversary R-squared
        if hasattr(model, "get_adversary_r2"):
            diagnostics.setdefault("adversary_r2", {})[task_name] = model.get_adversary_r2()

        # Training history
        if hasattr(model, "get_training_history"):
            history = model.get_training_history()
            if history:
                diagnostics.setdefault("history", {})[task_name] = history

    out_path = output_dir / "icp_diagnostics.json"
    out_path.write_text(json.dumps(diagnostics, indent=2, default=str))
    logger.info("Exported ICP diagnostics to %s", out_path)


def generate_paper_assets(output_dir: Path, data_dir: Path) -> None:
    """Generate paper figures and tables.

    Parameters
    ----------
    output_dir : Path
        Results directory (used for loading pipeline outputs).
    data_dir : Path
        Root data directory.
    """
    # Ensure project root is importable (paper/ lives at project root)
    import sys

    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from paper.generate_figures import main as fig_main
    except ImportError:
        logger.warning("matplotlib not available — skipping paper figure generation")
        return

    try:
        from paper.generate_tables import main as tbl_main
    except ImportError:
        logger.warning("tabulate not available — skipping paper table generation")
        return

    figures_dir = Path("paper") / "figures"
    tables_dir = Path("paper") / "tables"

    try:
        from click.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(
            fig_main,
            [
                "--results",
                str(output_dir),
                "--output",
                str(figures_dir),
                "--data-dir",
                str(data_dir),
            ],
        )
        if result.exit_code == 0:
            logger.info("Generated paper figures to %s", figures_dir)
        else:
            logger.warning("Figure generation failed: %s", result.output)

        # ED Fig. 4a (ICP convergence) and the reliability diagram are now drawn
        # by generate_figures directly: export_icp_diagnostics persists the ICP's
        # per-epoch history into icp_diagnostics.json (via the icp_models channel
        # from train_and_evaluate), so the base run no longer needs the out-of-band
        # out-of-band re-fit. That shell-out is retired (the mechanism is
        # deprecated but kept for provenance).

        result = runner.invoke(
            tbl_main,
            ["--results", str(output_dir), "--output", str(tables_dir)],
        )
        if result.exit_code == 0:
            logger.info("Generated paper tables to %s", tables_dir)
        else:
            logger.warning("Table generation failed: %s", result.output)
    except (RuntimeError, OSError, ImportError):
        logger.warning("Failed to generate paper assets", exc_info=True)
        if is_strict():
            raise


#: Sources excluded from the public dataset archive until their agencies confirm
#: redistribution (docs/DATA_TERMS.md is the authoritative per-source record).
ZENODO_EXCLUDE_SOURCES: tuple[str, ...] = ("mn_mdh",)

_ZENODO_README = """\
# AquaContam curated dataset archive

This archive is the reproducible copy of record for the harmonized AquaContam
dataset. The compiled data are licensed CC BY 4.0 where source terms permit;
per-source attributions, terms, and redistribution status are listed in
`docs/DATA_TERMS.md` of the code repository
(https://github.com/tjnewton/aquacontam), whose code is licensed under the
Apache License 2.0.

Minnesota exclusion: the Minnesota MDH records are excluded from this archive
pending the agency's redistribution confirmation, so the `data/train.parquet`
split differs from the paper's analyzed training data by those records. The
same export can be requested from MDH (health.drinkingwater@state.mn.us) and
verified against the SHA-256 pinned in the repository's
`data/checksums.sha256`; the documented loader (`aquacontam.data.mn_mdh`)
parses the workbook locally.

Contents: `data/` (train/val/test splits), `features/` (derived feature
tables), `metadata/` (split assignments), `predictions/` (per-system model
prediction export), `dataset_card.json`, `analyte_info.json`,
`feature_columns.json`, and `manifest.json` (SHA-256 checksums, POSIX paths).

AquaContam is a research benchmark, not a regulatory or operational
risk-assessment product.
"""


def export_zenodo_dataset(
    wq_df: Any,
    feature_dfs: list[Any],
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Export Zenodo-ready dataset archive.

    Applies the redistribution exclusions (``ZENODO_EXCLUDE_SOURCES``), copies
    the per-system prediction export into the archive, and writes the archive
    README documenting the Minnesota delta.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames.
    data_dir : Path
        Root data directory.
    output_dir : Path
        Results output directory.
    """
    import pandas as pd

    try:
        from aquacontam.export.dataset import export_dataset
        from aquacontam.export.metadata import write_metadata_files

        # Build named feature dict from cached feature files
        features_dir = data_dir / "interim" / "features"
        named_features: dict[str, pd.DataFrame] = {}
        if features_dir.exists():
            for p in sorted(features_dir.glob("*.parquet")):
                named_features[p.stem] = pd.read_parquet(p)
                logger.info("Loaded feature file %s: %d rows", p.stem, len(named_features[p.stem]))

        # Fall back to positional feature_dfs if no cached files
        if not named_features and feature_dfs:
            feature_names = ["proximity", "land_use", "hydrogeology", "demographics"]
            for i, df in enumerate(feature_dfs):
                name = feature_names[i] if i < len(feature_names) else f"features_{i}"
                named_features[name] = df

        # Apply the redistribution exclusions once, up front, so the metadata
        # counts and the exported splits describe the same (filtered) population.
        wq_export = wq_df
        if "source" in wq_df.columns:
            wq_export = wq_df[~wq_df["source"].isin(list(ZENODO_EXCLUDE_SOURCES))]

        zenodo_dir = output_dir / "zenodo-dataset"
        zenodo_dir.mkdir(parents=True, exist_ok=True)

        # Metadata files FIRST: export_dataset writes the manifest last, so the
        # card/analyte/feature files must already be final when it hashes them
        # (writing them afterwards leaves stale checksums in the manifest).
        write_metadata_files(
            zenodo_dir,
            feature_dfs=named_features if named_features else None,
            n_systems=len(wq_export["pwsid"].unique()) if "pwsid" in wq_export.columns else None,
            n_samples=len(wq_export),
        )

        pred_dir = output_dir / "predictions"
        export_dataset(
            data=wq_export,
            feature_dfs=named_features if named_features else None,
            output_dir=zenodo_dir,
            exclude_sources=ZENODO_EXCLUDE_SOURCES,
            predictions_dir=pred_dir if pred_dir.is_dir() else None,
            readme_text=_ZENODO_README,
        )
        logger.info("Exported Zenodo dataset to %s", zenodo_dir)
    except (ImportError, OSError, ValueError, TypeError):
        logger.warning("Failed to export Zenodo dataset", exc_info=True)
        if is_strict():
            raise
