"""Hyperparameter optimization analysis functions.

Functions in this module run grid search, Bayesian HPO (Optuna), and
hyperparameter sensitivity sweeps on T1 data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aquacontam.pipeline._strict import is_strict
from aquacontam.pipeline.assembly import assemble_with_split_imputation
from aquacontam.pipeline.models import load_model_configs as _load_model_configs
from aquacontam.pipeline.utils import _grid_size, _is_nan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fair (symmetric) tuning
# ---------------------------------------------------------------------------


def _run_fair_tuning(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run symmetric hyperparameter tuning for all model families.

    Addresses methodological fairness: all model families receive equal HPO
    treatment (grid search on T1 validation set) rather than only XGBoost.
    Results are exported to ``tuning_comparison.json`` with per-model best
    configs and tuned-vs-default performance deltas.
    """
    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.benchmark.tuning import grid_search
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    try:
        from aquacontam.models.random_forest import (
            RandomForestClassifier as RFClassifier,
        )
    except ImportError:
        RFClassifier = None  # type: ignore[assignment,misc]

    # Prepare T1 data with geographic splits (train-only imputation)
    try:
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)

        train_mask = regions.isin((1, 3, 4, 5, 6))
        val_mask = regions.isin((2, 7))
        test_mask = regions.isin((8, 9, 10))
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Fair tuning: failed to prepare T1 data", exc_info=True)
        if is_strict():
            raise
        return

    # Load tuning grids and default configs from experiment.yaml
    from aquacontam._config import load_experiment_config

    try:
        full_config = load_experiment_config()
    except (FileNotFoundError, KeyError):
        logger.warning("Fair tuning: experiment.yaml not found")
        return

    tuning_grids = full_config.get("tuning", {})
    model_configs = full_config.get("models", {})

    if not tuning_grids:
        logger.info("No tuning grids defined — skipping fair tuning")
        return

    # Map grid key -> (model_cls, default_config_key)
    model_registry: dict[str, tuple[type, str]] = {
        "xgboost_classifier": (XGBoostClassifier, "xgboost_classifier"),
    }
    if RFClassifier is not None:
        model_registry["random_forest_classifier"] = (
            RFClassifier,
            "random_forest_classifier",
        )

    # Optional model families
    for name, mod_path, cls_name, cfg_key in [
        (
            "lightgbm_classifier",
            "aquacontam.models.lightgbm",
            "LightGBMClassifier",
            "lightgbm_classifier",
        ),
        (
            "catboost_classifier",
            "aquacontam.models.catboost",
            "CatBoostClassifier",
            "catboost_classifier",
        ),
        (
            "mlp_classifier",
            "aquacontam.models.mlp",
            "MLPClassifier",
            "mlp_classifier",
        ),
        (
            "cnn1d_classifier",
            "aquacontam.models.cnn1d",
            "CNN1DClassifier",
            "cnn1d_classifier",
        ),
    ]:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            model_registry[name] = (cls, cfg_key)
        except (ImportError, OSError, RuntimeError):
            logger.info("Fair tuning: %s not available", name)

    tuning_results: list[dict[str, Any]] = []

    for grid_key, param_grid in tuning_grids.items():
        if grid_key not in model_registry:
            logger.info("Fair tuning: no model class for %s — skipping", grid_key)
            continue

        model_cls, default_cfg_key = model_registry[grid_key]

        # Build base config from defaults
        base_cfg = dict(model_configs.get(default_cfg_key, {}))
        base_cfg["random_state"] = seed
        if base_cfg.get("scale_pos_weight") == "auto":
            del base_cfg["scale_pos_weight"]

        logger.info("Fair tuning: %s (%d combos)", grid_key, _grid_size(param_grid))

        try:
            best_config, all_grid_results = grid_search(
                model_cls,
                param_grid,
                X_train,
                y_train,
                X_val,
                y_val,
                metric="auprc",
                base_config=base_cfg,
            )

            # Evaluate best tuned config on TEST set
            tuned_model = model_cls(config=best_config)
            tuned_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            tuned_preds = tuned_model.predict(X_test)
            tuned_probs = None
            try:
                p = tuned_model.predict_proba(X_test)
                tuned_probs = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p
            except NotImplementedError:
                pass
            tuned_metrics = compute_classification_metrics(y_test, tuned_preds, tuned_probs)

            # Evaluate default config on TEST set
            default_model = model_cls(config=base_cfg)
            default_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            default_preds = default_model.predict(X_test)
            default_probs = None
            try:
                p = default_model.predict_proba(X_test)
                default_probs = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p
            except NotImplementedError:
                pass
            default_metrics = compute_classification_metrics(y_test, default_preds, default_probs)

            # Compute deltas
            delta_auroc = tuned_metrics.get("auroc", 0) - default_metrics.get("auroc", 0)
            delta_auprc = tuned_metrics.get("auprc", 0) - default_metrics.get("auprc", 0)

            entry = {
                "model": grid_key,
                "n_configs_tried": len(all_grid_results),
                "best_val_auprc": max(
                    (r["score"] for r in all_grid_results if not _is_nan(r["score"])),
                    default=float("nan"),
                ),
                "best_config": {
                    k: v for k, v in best_config.items() if k in param_grid or k == "random_state"
                },
                "tuned_test_metrics": {
                    "auroc": tuned_metrics.get("auroc"),
                    "auprc": tuned_metrics.get("auprc"),
                },
                "default_test_metrics": {
                    "auroc": default_metrics.get("auroc"),
                    "auprc": default_metrics.get("auprc"),
                },
                "delta_auroc": delta_auroc,
                "delta_auprc": delta_auprc,
            }
            tuning_results.append(entry)

            logger.info(
                "  %s: tuned AUPRC=%.4f default=%.4f (delta=%+.4f)",
                grid_key,
                tuned_metrics.get("auprc", 0),
                default_metrics.get("auprc", 0),
                delta_auprc,
            )

        except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
            logger.warning("Fair tuning: %s failed", grid_key, exc_info=True)
            if is_strict():
                raise
            tuning_results.append(
                {
                    "model": grid_key,
                    "error": "tuning failed",
                }
            )

    # Export results
    out_path = output_dir / "tuning_comparison.json"
    out_path.write_text(json.dumps(tuning_results, indent=2, default=str))
    logger.info("Fair tuning results saved to %s (%d models)", out_path, len(tuning_results))


# ---------------------------------------------------------------------------
# Optuna Bayesian HPO
# ---------------------------------------------------------------------------


# Per-model-key base config injections (e.g. GNNClassifier needs ``arch``
# to differentiate gcn vs sage). The HPO driver merges these into base_config
# before each trial.
_MODEL_BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "gnn_gcn_classifier": {"arch": "gcn"},
    "gnn_sage_classifier": {"arch": "sage"},
}

# Models whose ``fit()`` consumes ``use_gpu``/``gpu_id`` config keys. Other
# models (sklearn, torch) either reject these keys outright (RF raised
# ``unexpected keyword argument 'use_gpu'``) or pick up the GPU via
# ``AQUACONTAM_GPU_ID`` env var inside ``get_device()``.
_GPU_AWARE_TREE_PREFIXES: tuple[str, ...] = ("xgboost_", "lightgbm_", "catboost_")

# Models whose GPU kernel is known to crash (SIGSEGV) on the data shapes /
# objectives we use. Force these onto CPU until subprocess isolation lands.
# Why: catboost_regressor on T2 segfaulted on its very first trial
# (depth=10, iterations=937, RMSE objective). The classifier on T1 with the
# same search-space ran 8h on GPU cleanly, so this is specific to the RMSE
# kernel + T2 data shape. SIGSEGV cannot be caught from Python, so a subprocess
# wrapper would be needed to retry-on-CPU dynamically.
_GPU_UNSTABLE_MODELS: frozenset[str] = frozenset({"catboost_regressor"})


def _effective_gpu_id(model_key: str, gpu_id: int | None) -> int | None:
    """Compute the post-remap GPU id to forward to ``optuna_search`` for ``model_key``.

    Returns ``None`` (CPU) for non-GPU-aware libraries (e.g. RandomForest,
    sklearn baselines) and for models in ``_GPU_UNSTABLE_MODELS``. Otherwise
    returns the caller's ``gpu_id``.
    """
    if model_key in _GPU_UNSTABLE_MODELS:
        return None
    if any(model_key.startswith(p) for p in _GPU_AWARE_TREE_PREFIXES):
        return gpu_id
    return None


def _build_hpo_model_registry() -> dict[str, type]:
    """Lazy-import every model class HPO can tune. Skips unavailable optional deps."""
    registry: dict[str, type] = {}

    # Tree-based and sklearn baselines (always available).
    from aquacontam.models.xgboost import XGBoostClassifier, XGBoostRegressor

    registry["xgboost_classifier"] = XGBoostClassifier
    registry["xgboost_regressor"] = XGBoostRegressor

    from aquacontam.models.random_forest import (
        RandomForestClassifier,
        RandomForestRegressor,
    )

    registry["random_forest_classifier"] = RandomForestClassifier
    registry["random_forest_regressor"] = RandomForestRegressor

    from aquacontam.models.logistic import LogisticRegressionClassifier

    registry["logistic_regression"] = LogisticRegressionClassifier

    # Optional deps: lazy-import + skip on ImportError.
    _OPTIONAL: list[tuple[str, str, str]] = [
        ("lightgbm_classifier", "aquacontam.models.lightgbm", "LightGBMClassifier"),
        ("lightgbm_regressor", "aquacontam.models.lightgbm", "LightGBMRegressor"),
        ("catboost_classifier", "aquacontam.models.catboost", "CatBoostClassifier"),
        ("catboost_regressor", "aquacontam.models.catboost", "CatBoostRegressor"),
        ("mlp_classifier", "aquacontam.models.mlp", "MLPClassifier"),
        ("mlp_regressor", "aquacontam.models.mlp", "MLPRegressor"),
        ("cnn1d_classifier", "aquacontam.models.cnn1d", "CNN1DClassifier"),
        ("cnn1d_regressor", "aquacontam.models.cnn1d", "CNN1DRegressor"),
        ("deep_tobit_classifier", "aquacontam.models.deep_tobit", "DeepTobitClassifier"),
        ("deep_tobit_regressor", "aquacontam.models.deep_tobit", "DeepTobitRegressor"),
        (
            "zi_tobit_classifier",
            "aquacontam.models.zero_inflated_tobit",
            "ZeroInflatedTobitClassifier",
        ),
        (
            "zi_tobit_regressor",
            "aquacontam.models.zero_inflated_tobit",
            "ZeroInflatedTobitRegressor",
        ),
        ("icp_classifier", "aquacontam.models.icp", "ICPClassifier"),
        ("icp_regressor", "aquacontam.models.icp", "ICPRegressor"),
        ("gnn_gcn_classifier", "aquacontam.models.gnn", "GNNClassifier"),
        ("gnn_sage_classifier", "aquacontam.models.gnn", "GNNClassifier"),
        ("tabpfn_classifier", "aquacontam.models.tabpfn", "TabPFNClassifier"),
        ("tobit_regressor", "aquacontam.models.tobit", "TobitRegressor"),
        ("aft_regressor", "aquacontam.models.aft", "AFTRegressor"),
        ("xgboost_aft_regressor", "aquacontam.models.xgboost_aft", "XGBoostAFTRegressor"),
        ("hurdle_regressor", "aquacontam.models.hurdle", "HurdleRegressor"),
        ("voting_ensemble_classifier", "aquacontam.models.ensemble", "VotingEnsembleClassifier"),
        (
            "stacking_ensemble_classifier",
            "aquacontam.models.ensemble",
            "StackingEnsembleClassifier",
        ),
    ]
    for name, mod_path, cls_name in _OPTIONAL:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            registry[name] = getattr(mod, cls_name)
        except (ImportError, OSError, RuntimeError):
            logger.info("HPO registry: %s not available — skipping", name)
    return registry


def _save_optuna_results(out_path: Path, entries: list[dict[str, Any]]) -> None:
    """Merge ``entries`` into the JSON at ``out_path`` (upsert by ``(model, task)``)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except (OSError, json.JSONDecodeError):
            existing = []
    # Migrate legacy entries that lack the ``task`` field.
    for e in existing:
        if isinstance(e, dict) and "task" not in e:
            e["task"] = "T1"
    # Build (model, task) -> entry index map.
    by_key = {(e.get("model"), e.get("task")): i for i, e in enumerate(existing)}
    for entry in entries:
        key = (entry.get("model"), entry.get("task"))
        if key in by_key:
            existing[by_key[key]] = entry
        else:
            existing.append(entry)
            by_key[key] = len(existing) - 1
    out_path.write_text(json.dumps(existing, indent=2, default=str))


def _run_optuna_tuning(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
    n_trials: int = 300,
    tasks: list[str] | None = None,
    models: list[str] | None = None,
    gpu_id: int | None = None,
    sampler: str = "tpe_multi",
    pruner: str = "hyperband",
    storage_dir: Path | None = None,
    timeout_per_study: float | None = None,
    inproc: bool = False,
) -> None:
    """Run Optuna Bayesian HPO across (task, model) combinations.

    Studies are persisted to ``storage_dir/{model}_{task}.db`` (sqlite),
    enabling resume-on-restart. Best configs are merged into
    ``output_dir/optuna_tuning.json`` with one entry per (model, task).

    Parameters
    ----------
    wq_df, feature_dfs : pipeline data
        Standard water-quality DataFrame + auxiliary feature DataFrames.
    output_dir : Path
        Where to write ``optuna_tuning.json``.
    seed : int
        Sampler seed; also passed as ``random_state`` into model configs.
    n_trials : int, default 300
        Trials per (model, task).
    tasks : list[str] or None
        Subset of ``["T1","T2","T3","T4","T5","T6","T7"]``. ``None`` = all
        currently-implemented prep funcs (T1, T2, T4).
    models : list[str] or None
        Subset of registry keys to tune. ``None`` = every registered model.
    gpu_id : int or None
        GPU index after CUDA_VISIBLE_DEVICES remap (use 0 when --gpu-id was set).
    sampler, pruner : str
        Forwarded to ``optuna_search``.
    storage_dir : Path or None
        Default ``output_dir/"optuna_studies"``.
    timeout_per_study : float or None
        Max wall-clock seconds per (model, task).
    """
    try:
        from aquacontam.benchmark._prep_data import PREP_FUNCTIONS, TaskData
        from aquacontam.benchmark.optuna_hpo import (
            DEFAULT_DIRECTIONS,
            DEFAULT_METRICS,
            SEARCH_SPACES_BY_TASK,
            optuna_search,
        )
    except ImportError:
        logger.info("Optuna not available — skipping Bayesian HPO")
        return

    # Default to the headline tasks; T6 is implemented but opt-in via --hpo-tasks
    # (it needs the NGA arsenic source). T3/T5/T7 prep is still stubbed.
    if tasks is None:
        tasks = ["T1", "T2", "T4"]
    # Filter to implemented tasks; warn on others.
    _IMPLEMENTED_TASKS = {"T1", "T2", "T4", "T6"}
    skipped_tasks = [t for t in tasks if t not in _IMPLEMENTED_TASKS]
    if skipped_tasks:
        logger.warning(
            "HPO: tasks %s have no prep helper yet; skipping. Implemented: %s",
            skipped_tasks,
            sorted(_IMPLEMENTED_TASKS),
        )
    tasks = [t for t in tasks if t in _IMPLEMENTED_TASKS]
    if not tasks:
        logger.warning("HPO: no implemented tasks selected — nothing to tune")
        return

    if storage_dir is None:
        storage_dir = output_dir / "optuna_studies"
    storage_dir.mkdir(parents=True, exist_ok=True)

    registry = _build_hpo_model_registry()

    out_path = output_dir / "optuna_tuning.json"

    for task in tasks:
        task_spaces = SEARCH_SPACES_BY_TASK.get(task, {})
        if not task_spaces:
            logger.warning("HPO: no search spaces for task %s — skipping", task)
            continue

        # Filter models: must be in (registry ∩ task_spaces ∩ user filter).
        candidate_models = [m for m in task_spaces if m in registry]
        if models is not None:
            candidate_models = [m for m in candidate_models if m in models]
        if not candidate_models:
            logger.warning("HPO: no models to tune for task %s (after filters)", task)
            continue

        prep_fn = PREP_FUNCTIONS[task]
        metric = DEFAULT_METRICS.get(task, "auprc")
        direction = DEFAULT_DIRECTIONS.get(metric, "maximize")

        for model_key in candidate_models:
            model_cls = registry[model_key]
            logger.info(
                "HPO: tuning %s on %s (%d trials, sampler=%s, pruner=%s, metric=%s)...",
                model_key,
                task,
                n_trials,
                sampler,
                pruner,
                metric,
            )

            try:
                task_data: TaskData | None = prep_fn(model_cls, wq_df, feature_dfs)
            except (ValueError, RuntimeError, ArithmeticError, OSError):
                logger.warning(
                    "HPO: data prep failed for %s/%s",
                    task,
                    model_key,
                    exc_info=True,
                )
                if is_strict():
                    raise
                continue
            if task_data is None:
                logger.info("HPO: no data for %s/%s — skipping", task, model_key)
                continue

            study_db = storage_dir / f"{model_key}_{task}.db"
            study_url = f"sqlite:///{study_db.as_posix()}"
            study_name = f"{model_key}_{task}_seed{seed}"

            # Skip studies that already reached the target trial count, so a
            # naive relaunch with the same `--n-trials` doesn't double-tune
            # completed work. Resume below-target studies with only the
            # remaining trial budget: optuna's ``study.optimize(n_trials=N)``
            # runs N NEW trials, so passing the full target on resume would
            # overshoot (e.g. 226 completed + 300 more = 526).
            trials_to_run = n_trials
            if study_db.exists():
                try:
                    import optuna

                    existing_study = optuna.load_study(study_name=study_name, storage=study_url)
                    n_complete = sum(
                        1
                        for t in existing_study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    )
                    if n_complete >= n_trials:
                        logger.info(
                            "HPO: %s/%s already has %d completed trials (>= %d) — skipping",
                            model_key,
                            task,
                            n_complete,
                            n_trials,
                        )
                        continue
                    if n_complete:
                        trials_to_run = n_trials - n_complete
                        logger.info(
                            "HPO: %s/%s has %d completed trials — topping up with %d to reach %d",
                            model_key,
                            task,
                            n_complete,
                            trials_to_run,
                            n_trials,
                        )
                    del existing_study  # release before optuna_search re-opens the DB
                except (OSError, RuntimeError, ImportError):
                    # Corrupt or unreadable DB — let optuna_search recreate or resume.
                    pass

            # Tree-based / non-NN models report no intermediate values, so
            # Hyperband degenerates; switch to median pruner. NN models
            # (mlp, cnn1d, deep_tobit, zi_tobit, icp, gnn) keep hyperband.
            # Why: substring matching previously caught e.g. deep_tobit_regressor
            # via "tobit_regressor"; use exact membership instead.
            _NON_HYPERBAND_MODELS = {
                "xgboost_classifier",
                "xgboost_regressor",
                "xgboost_aft_regressor",
                "lightgbm_classifier",
                "lightgbm_regressor",
                "catboost_classifier",
                "catboost_regressor",
                "random_forest_classifier",
                "random_forest_regressor",
                "tobit_regressor",
                "aft_regressor",
                "logistic_regression",
                "hurdle_regressor",
                "tabpfn_classifier",
                "voting_ensemble_classifier",
                "stacking_ensemble_classifier",
            }
            this_pruner = (
                "median"
                if model_key in _NON_HYPERBAND_MODELS and pruner == "hyperband"
                else pruner
            )

            # GPU routing: tree libs receive use_gpu/gpu_id config keys, others
            # don't (sklearn raises on unexpected kwargs; torch models read
            # AQUACONTAM_GPU_ID env var instead). Models in _GPU_UNSTABLE_MODELS
            # stay on CPU even though they're tree libs (e.g. catboost_regressor
            # segfaults on T2 RMSE objective).
            effective_gpu_id = _effective_gpu_id(model_key, gpu_id)

            base_config = dict(_MODEL_BASE_CONFIGS.get(model_key, {}))

            try:
                best_config, study = optuna_search(
                    model_cls,
                    task_data.X_train,
                    task_data.y_train,
                    task_data.X_val,
                    task_data.y_val,
                    search_space=task_spaces[model_key],
                    n_trials=trials_to_run,
                    metric=metric,
                    seed=seed,
                    timeout=timeout_per_study,
                    storage=study_url,
                    study_name=study_name,
                    sampler=sampler,
                    pruner=this_pruner,
                    direction=direction,
                    gpu_id=effective_gpu_id,
                    base_config=base_config,
                    fit_kwargs=task_data.fit_extra,
                    model_name=model_key,
                    inproc=inproc,
                )
                entry = {
                    "model": model_key,
                    "task": task,
                    "n_trials": len(study.trials),
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "best_config": {k: v for k, v in best_config.items()},
                    "study_path": str(study_db),
                    "metric": metric,
                    "direction": direction,
                    "sampler": sampler,
                    "pruner": this_pruner,
                    "seed": seed,
                }
                _save_optuna_results(out_path, [entry])
                logger.info(
                    "  %s/%s: best %s=%.4f (%d trials)",
                    model_key,
                    task,
                    metric,
                    study.best_value,
                    len(study.trials),
                )
            except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
                logger.warning(
                    "HPO: tuning %s/%s failed",
                    task,
                    model_key,
                    exc_info=True,
                )
                if is_strict():
                    raise
                _save_optuna_results(
                    out_path,
                    [{"model": model_key, "task": task, "error": "tuning failed"}],
                )

    logger.info("Optuna HPO complete; results at %s", out_path)


# ---------------------------------------------------------------------------
# Hyperparameter sensitivity
# ---------------------------------------------------------------------------


def _run_hyperparam_sensitivity(
    wq_df: Any,
    feature_dfs: list[Any],
    output_dir: Path,
    *,
    seed: int = 42,
) -> None:
    """Run hyperparameter sensitivity analysis for key model families."""
    try:
        from aquacontam.analysis.sensitivity import hyperparameter_sensitivity
    except ImportError:
        logger.info("Sensitivity module not available — skipping")
        return

    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    try:
        sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
        sys_targets = drop_leakage_columns(sys_targets)
        X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)

        train_idx = X.index[regions.isin((1, 3, 4, 5, 6))]
        val_idx = X.index[regions.isin((2, 7))]

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

        yaml_configs = _load_model_configs()

        # Load tuning grid from experiment config (top-level "tuning" key)
        try:
            from aquacontam._config import load_experiment_config

            full_cfg = load_experiment_config()
            tuning_grid = full_cfg.get("tuning", {}).get("xgboost_classifier", {})
        except (FileNotFoundError, KeyError):
            tuning_grid = {}

        if tuning_grid:
            from aquacontam.analysis.sensitivity import generate_config_variants

            base_cfg = dict(yaml_configs.get("xgboost_classifier", {}))
            base_cfg["random_state"] = seed
            if base_cfg.get("scale_pos_weight") == "auto":
                del base_cfg["scale_pos_weight"]

            configs = generate_config_variants(base_cfg, tuning_grid, max_configs=18)
            sensitivity_results = hyperparameter_sensitivity(
                XGBoostClassifier, configs, X_train, y_train, X_val, y_val
            )
            out_path = output_dir / "hyperparam_sensitivity.json"
            out_path.write_text(json.dumps(sensitivity_results, indent=2, default=str))
            logger.info("Hyperparameter sensitivity saved to %s", out_path)
    except (ImportError, OSError, ValueError, RuntimeError, ArithmeticError):
        logger.warning("Hyperparameter sensitivity failed", exc_info=True)
        if is_strict():
            raise
