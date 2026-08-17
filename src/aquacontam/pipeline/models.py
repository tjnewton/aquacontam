"""Model configuration loading and instance creation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_model_configs() -> dict[str, dict[str, Any]]:
    """Load model hyperparameters from experiment.yaml.

    Falls back to sensible defaults if the config file is missing.

    Returns
    -------
    dict[str, dict[str, Any]]
        Model name -> hyperparameter dict.
    """
    try:
        from aquacontam._config import load_experiment_config

        return load_experiment_config(section="models")
    except (FileNotFoundError, KeyError):
        logger.warning("experiment.yaml not found — using built-in defaults")
        return {}


def with_name_override(model: Any, name: str) -> Any:
    """Override a model's name property for result reporting."""
    model._name_override = name
    # Monkey-patch the name property to return the override
    original_class = type(model)

    class _NamedModel(original_class):  # type: ignore[valid-type, misc]
        @property
        def name(self) -> str:
            return str(self._name_override)

    model.__class__ = _NamedModel
    return model


def get_model_instances(
    model_filter: list[str] | None,
    task_type: str,
    seed: int,
    *,
    task_name: str | None = None,
    use_tuned_params: bool = False,
    tuned_params_path: str | None = None,
) -> list[Any]:
    """Create model instances based on filters.

    Loads hyperparameters from ``configs/experiment.yaml`` when available,
    falling back to sensible defaults. When ``use_tuned_params=True`` and
    ``task_name`` is set, prefers the best config from
    ``results/optuna_tuning.json`` for that ``(model, task)`` pair, falling
    back to YAML when no tuned config is available.

    Returns
    -------
    list[BaseModel]
    """
    from aquacontam.models.logistic import DummyClassifierBaseline, LogisticRegressionClassifier
    from aquacontam.models.random_forest import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from aquacontam.models.xgboost import (
        XGBoostClassifier,
        XGBoostRegressor,
    )

    yaml_configs = load_model_configs()

    _load_tuned: Any = None
    if use_tuned_params and task_name:
        try:
            from aquacontam.benchmark.optuna_hpo import load_best_config

            _load_tuned = load_best_config
        except ImportError:
            _load_tuned = None

    def _cfg(model_name: str) -> dict[str, Any]:
        """Get config for model, merging YAML with seed override.

        When ``use_tuned_params=True``, prefers tuned config from
        ``optuna_tuning.json`` for ``(model_name, task_name)``.
        """
        cfg: dict[str, Any] | None = None
        if _load_tuned is not None and task_name is not None:
            path = tuned_params_path or "results/optuna_tuning.json"
            cfg = _load_tuned(model_name, task_name, path)
        cfg = dict(yaml_configs.get(model_name, {})) if cfg is None else dict(cfg)
        cfg["random_state"] = seed
        # Remove string-valued keys that need runtime computation
        if cfg.get("scale_pos_weight") == "auto":
            del cfg["scale_pos_weight"]
        return cfg

    all_models: dict[str, dict[str, Any]] = {
        "dummy": {
            "classification": lambda: DummyClassifierBaseline(
                config=_cfg("dummy_classifier"),
            ),
        },
        "logistic_regression": {
            "classification": lambda: LogisticRegressionClassifier(
                config=_cfg("logistic_regression"),
            ),
        },
        "xgboost": {
            "classification": lambda: XGBoostClassifier(
                config=_cfg("xgboost_classifier"),
            ),
            "regression": lambda: XGBoostRegressor(
                config=_cfg("xgboost_regressor"),
            ),
        },
        "xgboost_default": {
            "classification": lambda: with_name_override(
                XGBoostClassifier(config=_cfg("xgboost_classifier_default")),
                "xgboost_default",
            ),
        },
        "random_forest": {
            "classification": lambda: RandomForestClassifier(
                config=_cfg("random_forest_classifier"),
            ),
            "regression": lambda: RandomForestRegressor(
                config=_cfg("random_forest_regressor"),
            ),
        },
    }

    # Try importing deep learning models only when needed
    if not model_filter or "mlp" in model_filter:
        try:
            from aquacontam.models.mlp import MLPClassifier, MLPRegressor

            all_models["mlp"] = {
                "classification": lambda: MLPClassifier(
                    config=_cfg("mlp_classifier"),
                ),
                "regression": lambda: MLPRegressor(
                    config=_cfg("mlp_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "cnn1d" in model_filter:
        try:
            from aquacontam.models.cnn1d import CNN1DClassifier, CNN1DRegressor

            all_models["cnn1d"] = {
                "classification": lambda: CNN1DClassifier(
                    config=_cfg("cnn1d_classifier"),
                ),
                "regression": lambda: CNN1DRegressor(
                    config=_cfg("cnn1d_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "lightgbm" in model_filter:
        try:
            from aquacontam.models.lightgbm import LightGBMClassifier, LightGBMRegressor

            all_models["lightgbm"] = {
                "classification": lambda: LightGBMClassifier(
                    config=_cfg("lightgbm_classifier"),
                ),
                "regression": lambda: LightGBMRegressor(
                    config=_cfg("lightgbm_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "catboost" in model_filter:
        try:
            from aquacontam.models.catboost import CatBoostClassifier, CatBoostRegressor

            all_models["catboost"] = {
                "classification": lambda: CatBoostClassifier(
                    config=_cfg("catboost_classifier"),
                ),
                "regression": lambda: CatBoostRegressor(
                    config=_cfg("catboost_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "gnn_gcn" in model_filter:
        try:
            from aquacontam.models.gnn import GNNClassifier

            all_models["gnn_gcn"] = {
                "classification": lambda: GNNClassifier(
                    config=_cfg("gnn_gcn_classifier"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "gnn_sage" in model_filter:
        try:
            from aquacontam.models.gnn import GNNClassifier as _GNNSage

            all_models["gnn_sage"] = {
                "classification": lambda: _GNNSage(
                    config=_cfg("gnn_sage_classifier"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "deep_tobit" in model_filter:
        try:
            from aquacontam.models.deep_tobit import DeepTobitClassifier, DeepTobitRegressor

            all_models["deep_tobit"] = {
                "classification": lambda: DeepTobitClassifier(
                    config=_cfg("deep_tobit_classifier"),
                ),
                "regression": lambda: DeepTobitRegressor(
                    config=_cfg("deep_tobit_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "hurdle" in model_filter:
        try:
            from aquacontam.models.hurdle import HurdleRegressor

            all_models["hurdle"] = {
                "regression": lambda: HurdleRegressor(
                    config=_cfg("hurdle_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "xgboost_aft" in model_filter:
        try:
            from aquacontam.models.xgboost_aft import XGBoostAFTRegressor

            all_models["xgboost_aft"] = {
                "regression": lambda: XGBoostAFTRegressor(
                    config=_cfg("xgboost_aft_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "zi_tobit" in model_filter:
        try:
            from aquacontam.models.zero_inflated_tobit import ZeroInflatedTobitRegressor

            all_models["zi_tobit"] = {
                "regression": lambda: ZeroInflatedTobitRegressor(
                    config=_cfg("zi_tobit_regressor"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "tabpfn" in model_filter:
        try:
            from aquacontam.models.tabpfn import TabPFNClassifier

            all_models["tabpfn"] = {
                "classification": lambda: TabPFNClassifier(
                    config=_cfg("tabpfn_classifier"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "voting_ensemble" in model_filter:
        try:
            from aquacontam.models.ensemble import VotingEnsembleClassifier

            all_models["voting_ensemble"] = {
                "classification": lambda: VotingEnsembleClassifier(
                    config=_cfg("voting_ensemble_classifier"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "stacking_ensemble" in model_filter:
        try:
            from aquacontam.models.ensemble import StackingEnsembleClassifier

            all_models["stacking_ensemble"] = {
                "classification": lambda: StackingEnsembleClassifier(
                    config=_cfg("stacking_ensemble_classifier"),
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "icp" in model_filter:
        try:
            from aquacontam.models.icp import ICPClassifier, ICPRegressor

            all_models["icp"] = {
                "classification": lambda: ICPClassifier(config=_cfg("icp_classifier")),
                "regression": lambda: ICPRegressor(config=_cfg("icp_regressor")),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    if not model_filter or "hurdle_aft_ensemble" in model_filter:
        try:
            from aquacontam.models.ensemble import AveragingEnsembleRegressor
            from aquacontam.models.hurdle import HurdleRegressor as _HurdleReg
            from aquacontam.models.xgboost_aft import XGBoostAFTRegressor as _AFTReg

            all_models["hurdle_aft_ensemble"] = {
                "regression": lambda: AveragingEnsembleRegressor(
                    models=[
                        _HurdleReg(config=_cfg("hurdle_regressor")),
                        _AFTReg(config=_cfg("xgboost_aft_regressor")),
                    ],
                ),
            }
        except (ImportError, OSError, RuntimeError):
            pass

    filtered_names = model_filter or list(all_models.keys())
    models = []
    for name in filtered_names:
        if name in all_models and task_type in all_models[name]:
            models.append(all_models[name][task_type]())

    return models
