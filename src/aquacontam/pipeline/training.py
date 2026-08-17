"""Model training and evaluation orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aquacontam._constants import AMBIENT_WQ_SOURCES
from aquacontam.pipeline._strict import is_strict
from aquacontam.pipeline.models import get_model_instances

logger = logging.getLogger(__name__)


def _save_task_checkpoint(
    task_name: str,
    task_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save results for a completed task to a checkpoint file."""
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{task_name}_results.json"
    ckpt_path.write_text(json.dumps(task_results, indent=2, default=str))
    logger.info("Saved checkpoint for %s (%d results)", task_name, len(task_results))


def _load_completed_tasks(
    output_dir: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Load results from previously checkpointed tasks.

    Returns
    -------
    tuple[set[str], list[dict[str, Any]]]
        ``(completed_task_names, loaded_results)``.
    """
    ckpt_dir = output_dir / "checkpoints"
    completed: set[str] = set()
    results: list[dict[str, Any]] = []
    if not ckpt_dir.exists():
        return completed, results
    for path in sorted(ckpt_dir.glob("*_results.json")):
        task_name = path.stem.replace("_results", "")
        try:
            task_results = json.loads(path.read_text())
            completed.add(task_name)
            results.extend(task_results)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt checkpoint %s — will re-run task", path)
    return completed, results


def build_task_kwargs(
    task_name: str,
    model: Any,
    wq_df: Any,
    feature_dfs: list[Any],
    downloaded: dict[str, Path],
    *,
    use_random_split: bool = False,
    exclude_features: list[str] | None = None,
) -> dict[str, Any]:
    """Build task-specific keyword arguments for benchmark task functions.

    T1/T2/T3/T4 use ``(model, data, feature_dfs)``.
    T5 uses ``(model, source_data, target_data, feature_dfs, ...)``.
    T6 uses ``(model, data, well_data, feature_dfs, ...)``.
    T7 uses ``(model, ucmr3_data, ucmr5_data, feature_dfs, ...)``.
    """
    import pandas as pd

    base: dict[str, Any] = {"model": model, "feature_dfs": feature_dfs}
    if use_random_split:
        base["split_strategy"] = "random"
    # Only pass exclude_features to tasks that support it (T1, T2, T4, T5, T7)
    _EXCLUDE_FEATURES_TASKS = {"T1", "T2", "T4", "T5", "T7"}
    if exclude_features and task_name in _EXCLUDE_FEATURES_TASKS:
        base["exclude_features"] = exclude_features

    if task_name == "T5":
        # T5: cross-contaminant transfer (heavy metals -> PFAS)
        sdwis_path = downloaded.get("sdwis")
        ucmr5_path = downloaded.get("ucmr5")
        source_data = pd.read_parquet(sdwis_path) if sdwis_path and sdwis_path.exists() else wq_df
        target_data = pd.read_parquet(ucmr5_path) if ucmr5_path and ucmr5_path.exists() else wq_df
        base["source_data"] = source_data
        base["target_data"] = target_data
    elif task_name == "T6":
        # T6: private well risk extrapolation
        njpw_path = downloaded.get("nj_private_wells")
        base["data"] = wq_df
        if njpw_path and njpw_path.exists():
            base["well_data"] = pd.read_parquet(njpw_path)
        else:
            logger.warning("NJ private wells data not available for T6")
            base["well_data"] = pd.DataFrame()
    elif task_name == "T7":
        # T7: temporal prediction (UCMR3 -> UCMR5)
        ucmr3_path = downloaded.get("ucmr3")
        ucmr5_path = downloaded.get("ucmr5")
        base["ucmr3_data"] = (
            pd.read_parquet(ucmr3_path) if ucmr3_path and ucmr3_path.exists() else pd.DataFrame()
        )
        base["ucmr5_data"] = (
            pd.read_parquet(ucmr5_path) if ucmr5_path and ucmr5_path.exists() else pd.DataFrame()
        )
    else:
        # T1, T2, T3, T4 — standard signature
        base["data"] = wq_df

    return base


def _exclude_ambient_sources(wq_df: Any) -> Any:
    """Drop ambient-source rows from a water-quality frame for the main benchmark.

    Ambient environmental-monitoring sources (:data:`AMBIENT_WQ_SOURCES` —
    groundwater/surface water) sample a different population than PWS finished-water
    compliance monitoring, so they are excluded from the main train/val/test splits
    and reserved for external validation. Defensive: returns ``wq_df`` unchanged if it
    has no ``source`` column or contains none of the ambient sources.
    """
    if not (hasattr(wq_df, "columns") and "source" in wq_df.columns):
        return wq_df
    ambient_mask = wq_df["source"].isin(AMBIENT_WQ_SOURCES)
    n_ambient = int(ambient_mask.sum())
    if not n_ambient:
        return wq_df
    dropped = sorted(wq_df.loc[ambient_mask, "source"].unique())
    logger.info(
        "Excluded %d ambient-source rows (%s) from main benchmark splits "
        "(retained for external validation)",
        n_ambient,
        ", ".join(dropped),
    )
    return wq_df[~ambient_mask].copy()


def train_and_evaluate(
    wq_df: Any,
    feature_dfs: list[Any],
    downloaded: dict[str, Path],
    output_dir: Path,
    *,
    seed: int = 42,
    task_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
    use_random_split: bool = False,
    exclude_features: list[str] | None = None,
    resume: bool = True,
    use_tuned_params: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, tuple[Any, str, float]], dict[str, Any]]:
    """Train models and evaluate on benchmark tasks.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Merged water quality data.
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.
    downloaded : dict[str, Path]
        Mapping of source name -> downloaded file path.
    output_dir : Path
        Directory for results output.
    seed : int
        Random seed.
    task_filter : list[str] | None
        If provided, only run these task names.
    model_filter : list[str] | None
        If provided, only use these model names.
    use_random_split : bool
        If True, use random splits instead of geographic stratification
        (for ablation comparison). Default False.
    exclude_features : list[str] | None
        Column name prefixes to exclude from feature matrices.  Used by
        ``--no-monitoring-features`` to drop ``n_samples`` and
        ``mean_detection_limit``.
    resume : bool
        If True (default), load results from previous checkpoints and
        skip already-completed tasks.  Set to False for a clean run.

    Returns
    -------
    tuple[list[dict], dict[str, tuple[Any, str, float]], dict[str, Any]]
        ``(results, fitted_models, icp_models)`` — results for all task/model
        combinations; a mapping of task name to
        ``(fitted_model, task_type, optimal_threshold)`` for the first
        non-trivial (SHAP-capable) model per task; and a mapping of task name
        to the fitted ICP model, retained so its per-epoch training history
        can be exported (the SHAP model in ``fitted_models`` is a tree model,
        which does not carry ICP diagnostics).
    """
    from aquacontam.benchmark.registry import get_task, list_tasks

    output_dir.mkdir(parents=True, exist_ok=True)

    # Exclude ambient environmental-monitoring sources from the MAIN benchmark splits
    # (they remain in the full dataset for the external-validation framework).
    wq_df = _exclude_ambient_sources(wq_df)

    if use_random_split:
        logger.info("Using RANDOM splits (ablation mode) — geographic stratification disabled")

    all_tasks = list_tasks()
    if task_filter:
        all_tasks = [t for t in all_tasks if t.name in task_filter]

    # Load checkpoints from any previous (interrupted) run
    completed_tasks: set[str] = set()
    all_results: list[dict[str, Any]] = []
    if resume:
        completed_tasks, prior_results = _load_completed_tasks(output_dir)
        if completed_tasks:
            all_results.extend(prior_results)
            logger.info(
                "Resuming: loaded %d results from %d completed tasks",
                len(prior_results),
                len(completed_tasks),
            )

    fitted_models: dict[str, tuple[Any, str, float]] = {}
    # ICP models retained per task so export_icp_diagnostics can persist their
    # per-epoch training history (ED Fig. 4a). fitted_models holds the tree
    # SHAP model, which carries no ICP diagnostics, so ICP needs its own channel.
    icp_models: dict[str, Any] = {}

    for task_info in all_tasks:
        task_name = task_info.name
        task_type = task_info.task_type

        # Skip already-checkpointed tasks
        if task_name in completed_tasks:
            logger.info("Skipping %s (checkpoint exists)", task_name)
            continue

        # T6 (private-well arsenic transfer) is not a per-model task in this
        # loop: it uses a separate CC0 source (USGS NGA) and a two-population
        # zero-shot design. It runs via the dedicated runner
        # (pipeline/t6_arsenic.py; `reproduce.py --t6-arsenic`).
        if task_name == "T6":
            logger.info("Skipping T6 in the task loop — run `reproduce.py --t6-arsenic`")
            continue

        # Map multilabel_classification -> classification for model selection
        model_task_type = "classification" if "classification" in task_type else "regression"
        models = get_model_instances(
            model_filter,
            model_task_type,
            seed,
            task_name=task_name,
            use_tuned_params=use_tuned_params,
        )

        _, task_fn = get_task(task_name)

        task_results: list[dict[str, Any]] = []
        for model in models:
            logger.info("Running %s with %s", task_name, model.name)
            try:
                kwargs = build_task_kwargs(
                    task_name,
                    model,
                    wq_df,
                    feature_dfs,
                    downloaded,
                    use_random_split=use_random_split,
                    exclude_features=exclude_features,
                )
                result = task_fn(**kwargs)
                result_dict = {
                    "task": task_name,
                    "model": model.name,
                    "metrics": result.metrics,
                    "split_metrics": result.split_metrics,
                    "metadata": result.metadata,
                }
                task_results.append(result_dict)
                all_results.append(result_dict)
                if result.metrics:
                    # Store the first non-trivial model for SHAP/feature importance.
                    # We prefer tree-based models (XGBoost, RF, etc.) because
                    # TreeExplainer gives exact SHAP values. The first non-trivial
                    # model is used rather than the highest-AUROC model, because
                    # tree models iterate first in the model registry and
                    # TreeExplainer compatibility is more important than marginal
                    # AUROC differences for interpretability.
                    _trivial = {"dummy_classifier", "logistic_regression"}
                    prev = fitted_models.get(task_name)
                    # Extract optimal threshold from validation set (Youden's J)
                    thresh_info = result.metadata.get("threshold_optimization", {})
                    val_analysis = thresh_info.get("val_analysis", {})
                    opt_threshold = (
                        val_analysis.get("optimal_threshold", 0.5) if val_analysis else 0.5
                    )
                    if prev is None or (prev[0].name in _trivial and model.name not in _trivial):
                        fitted_models[task_name] = (model, task_type, opt_threshold)
                    # Retain the fitted ICP so its per-epoch training history
                    # survives to export (fresh instance per task, so this
                    # captures the just-fitted model before the loop moves on).
                    if model.name.startswith("icp"):
                        icp_models[task_name] = model
                logger.info("  %s: %s", model.name, result.metrics)
            except (ValueError, RuntimeError, TypeError, ImportError) as exc:
                logger.warning(
                    "Skipping %s/%s",
                    task_name,
                    model.name,
                    exc_info=True,
                )
                # GNN models require coordinates that not all tasks provide —
                # this is an expected incompatibility, not a pipeline error.
                _is_gnn_coord_error = isinstance(exc, ValueError) and "coords" in str(exc).lower()
                if is_strict() and not _is_gnn_coord_error:
                    raise

        # Checkpoint after all models for this task complete
        if task_results:
            _save_task_checkpoint(task_name, task_results, output_dir)

    return all_results, fitted_models, icp_models
