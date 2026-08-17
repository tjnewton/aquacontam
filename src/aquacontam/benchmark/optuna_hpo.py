"""Bayesian hyperparameter optimization with Optuna.

Per-task search-space registry, persistent sqlite studies (resumable),
configurable samplers/pruners, and GPU passthrough for tree libraries.

Public API:
    SEARCH_SPACES_BY_TASK   — nested {task: {model_name: space}}
    SEARCH_SPACES           — alias for SEARCH_SPACES_BY_TASK["T1"] (back-compat)
    PARAM_TRANSFORMS        — categorical-string-to-list mappings
    optuna_search(...)      — driver function (now accepts storage, sampler, pruner, gpu_id)
    load_best_config(name, task, results_path) -> dict | None
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# List-valued param mappings (categorical strings → concrete lists)
# ---------------------------------------------------------------------------

_HIDDEN_SIZE_MAP: dict[str, list[int]] = {
    "small_2": [64, 32],
    "medium_2": [128, 64],
    "medium_3": [128, 64, 32],
    "large_3": [256, 128, 64],
    "large_4": [256, 128, 64, 32],
    "xl_3": [512, 256, 128],
    "xl_4": [512, 256, 128, 64],
    "deep_5": [256, 256, 128, 64, 32],
}

_N_FILTERS_MAP: dict[str, list[int]] = {
    "small_2": [32, 64],
    "medium_3": [32, 64, 32],
    "large_3": [64, 128, 64],
    "deep_4": [32, 64, 128, 64],
    "wide_3": [128, 256, 128],
}

# Encoder sizes for ICP (mirrors hidden-size keys for consistency).
_ENCODER_SIZE_MAP: dict[str, list[int]] = {
    "medium_2": [128, 64],
    "medium_3": [128, 64, 32],
    "large_3": [256, 128, 64],
}

# Ensemble base-model presets (categorical → concrete list of model names).
_ENSEMBLE_BASE_PRESETS: dict[str, list[str]] = {
    "xgb_rf": ["xgboost", "random_forest"],
    "xgb_lgbm": ["xgboost", "lightgbm"],
    "xgb_cat": ["xgboost", "catboost"],
    "lgbm_cat": ["lightgbm", "catboost"],
    "all_four": ["xgboost", "random_forest", "lightgbm", "catboost"],
}

PARAM_TRANSFORMS: dict[str, dict[str, dict[str, list[Any]]]] = {
    "mlp_classifier": {"hidden_sizes": _HIDDEN_SIZE_MAP},
    "mlp_regressor": {"hidden_sizes": _HIDDEN_SIZE_MAP},
    "cnn1d_classifier": {"n_filters": _N_FILTERS_MAP},
    "cnn1d_regressor": {"n_filters": _N_FILTERS_MAP},
    "deep_tobit_classifier": {"hidden_sizes": _HIDDEN_SIZE_MAP},
    "deep_tobit_regressor": {"hidden_sizes": _HIDDEN_SIZE_MAP},
    "zi_tobit_classifier": {"hidden_sizes": _HIDDEN_SIZE_MAP},
    "zi_tobit_regressor": {"hidden_sizes": _HIDDEN_SIZE_MAP},
    "icp_classifier": {"encoder_sizes": _ENCODER_SIZE_MAP},
    "icp_regressor": {"encoder_sizes": _ENCODER_SIZE_MAP},
    "voting_ensemble_classifier": {"base_models": _ENSEMBLE_BASE_PRESETS},
    "stacking_ensemble_classifier": {"base_models": _ENSEMBLE_BASE_PRESETS},
}


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
# Format: (type_spec, *args) — "int"/"float"/"float_log"/"categorical".
# All numeric args are strings (parsed at suggest time) for ConfigSpace
# friendliness with optuna's suggest API.

# T1 (binary detection classification) — 13 classifiers + ensembles.
_T1_SPACES: dict[str, dict[str, tuple[str, ...]]] = {
    "xgboost_classifier": {
        "n_estimators": ("int", "200", "2000"),
        "max_depth": ("int", "3", "12"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "subsample": ("float", "0.5", "1.0"),
        "colsample_bytree": ("float", "0.4", "1.0"),
        "min_child_weight": ("int", "1", "30"),
        "reg_alpha": ("float_log", "0.001", "10.0"),
        "reg_lambda": ("float_log", "0.001", "10.0"),
        "gamma": ("float_log", "0.001", "10.0"),
    },
    "random_forest_classifier": {
        "n_estimators": ("int", "200", "1500"),
        "max_depth": ("categorical", "10", "20", "30", "50", "None"),
        "max_features": ("categorical", "sqrt", "log2"),
        "min_samples_leaf": ("int", "1", "30"),
        "min_samples_split": ("int", "2", "20"),
    },
    "lightgbm_classifier": {
        "n_estimators": ("int", "200", "2000"),
        "num_leaves": ("int", "15", "511"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "subsample": ("float", "0.5", "1.0"),
        "colsample_bytree": ("float", "0.4", "1.0"),
        "min_child_samples": ("int", "5", "100"),
        "reg_alpha": ("float_log", "0.001", "10.0"),
        "reg_lambda": ("float_log", "0.001", "10.0"),
        "min_split_gain": ("float_log", "0.0001", "1.0"),
    },
    "catboost_classifier": {
        "iterations": ("int", "300", "2000"),
        "depth": ("int", "4", "10"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "l2_leaf_reg": ("float_log", "1.0", "30.0"),
        "bagging_temperature": ("float", "0.0", "1.0"),
        "border_count": ("int", "32", "255"),
        "random_strength": ("float_log", "0.1", "10.0"),
    },
    "mlp_classifier": {
        "hidden_sizes": (
            "categorical",
            "small_2",
            "medium_2",
            "medium_3",
            "large_3",
            "large_4",
            "xl_3",
            "xl_4",
            "deep_5",
        ),
        "learning_rate": ("float_log", "0.00001", "0.01"),
        "dropout": ("float", "0.0", "0.6"),
        "batch_size": ("categorical", "64", "128", "256", "512", "1024"),
        "weight_decay": ("float_log", "1e-7", "0.01"),
        "epochs": ("int", "50", "300"),
        "patience": ("int", "5", "30"),
    },
    "cnn1d_classifier": {
        "n_filters": (
            "categorical",
            "small_2",
            "medium_3",
            "large_3",
            "deep_4",
            "wide_3",
        ),
        "kernel_size": ("categorical", "3", "5", "7"),
        "learning_rate": ("float_log", "0.00001", "0.01"),
        "dropout": ("float", "0.0", "0.6"),
        "batch_size": ("categorical", "64", "128", "256", "512"),
        "weight_decay": ("float_log", "1e-7", "0.01"),
        "epochs": ("int", "50", "300"),
        "patience": ("int", "5", "30"),
    },
    "deep_tobit_classifier": {
        "hidden_sizes": (
            "categorical",
            "medium_2",
            "medium_3",
            "large_3",
            "large_4",
            "xl_3",
        ),
        "dropout": ("float", "0.1", "0.5"),
        "learning_rate": ("float_log", "0.00001", "0.005"),
        "batch_size": ("categorical", "64", "128", "256", "512"),
        "epochs": ("int", "80", "400"),
        "patience": ("int", "10", "40"),
        "weight_decay": ("float_log", "1e-7", "0.01"),
        "default_detection_limit": ("float_log", "0.0001", "0.1"),
    },
    "zi_tobit_classifier": {
        "hidden_sizes": (
            "categorical",
            "small_2",
            "medium_2",
            "medium_3",
            "large_3",
        ),
        "dropout": ("float", "0.2", "0.6"),
        "learning_rate": ("float_log", "0.00001", "0.005"),
        "batch_size": ("categorical", "64", "128", "256", "512"),
        "epochs": ("int", "80", "400"),
        "patience": ("int", "10", "40"),
        "weight_decay": ("float_log", "1e-7", "0.01"),
        "default_detection_limit": ("float_log", "0.0001", "0.1"),
    },
    "icp_classifier": {
        "encoder_sizes": ("categorical", "medium_2", "medium_3", "large_3"),
        "head_size": ("categorical", "16", "32", "64", "128"),
        "dropout": ("float", "0.2", "0.5"),
        "learning_rate": ("float_log", "0.0001", "0.005"),
        "adv_learning_rate": ("float_log", "0.001", "0.1"),
        "epochs": ("int", "100", "400"),
        "warmup_epochs": ("int", "10", "60"),
        "batch_size": ("categorical", "128", "256", "512"),
        "patience": ("int", "15", "40"),
        "lambda_adv": ("float_log", "0.01", "5.0"),
        "lambda_dro": ("float_log", "0.001", "1.0"),
        "adv_steps": ("int", "1", "10"),
    },
    "gnn_gcn_classifier": {
        "n_hidden": ("categorical", "32", "64", "128", "256"),
        "dropout": ("float", "0.1", "0.6"),
        "learning_rate": ("float_log", "0.0001", "0.05"),
        "epochs": ("int", "100", "400"),
        "patience": ("int", "10", "40"),
        "k": ("int", "3", "30"),
        "weight_decay": ("float_log", "1e-7", "0.01"),
    },
    "gnn_sage_classifier": {
        "n_hidden": ("categorical", "32", "64", "128", "256"),
        "dropout": ("float", "0.1", "0.6"),
        "learning_rate": ("float_log", "0.0001", "0.05"),
        "epochs": ("int", "100", "400"),
        "patience": ("int", "10", "40"),
        "k": ("int", "3", "30"),
        "weight_decay": ("float_log", "1e-7", "0.01"),
    },
    "tabpfn_classifier": {
        "n_estimators": ("int", "1", "4"),
        "fit_mode": ("categorical", "low_memory", "fit_preprocessors"),
    },
    "logistic_regression": {
        "C": ("float_log", "0.0001", "100.0"),
        "penalty": ("categorical", "l1", "l2"),
        "solver": ("categorical", "lbfgs", "liblinear", "saga"),
        "class_weight": ("categorical", "balanced", "None"),
        "max_iter": ("int", "500", "5000"),
    },
    "voting_ensemble_classifier": {
        # Only soft voting supported: hard voting omits predict_proba which
        # AUPRC needs. sklearn's VotingClassifier(voting="hard").predict_proba()
        # raises AttributeError, failing the trial.
        "voting": ("categorical", "soft"),
        "base_models": (
            "categorical",
            "xgb_rf",
            "xgb_lgbm",
            "xgb_cat",
            "lgbm_cat",
            "all_four",
        ),
    },
    "stacking_ensemble_classifier": {
        "base_models": (
            "categorical",
            "xgb_rf",
            "xgb_lgbm",
            "xgb_cat",
            "lgbm_cat",
            "all_four",
        ),
        "cv": ("int", "3", "10"),
        "passthrough": ("categorical", "True", "False"),
    },
}

# T2 (concentration regression) — regressors only.
_T2_SPACES: dict[str, dict[str, tuple[str, ...]]] = {
    "xgboost_regressor": {
        "n_estimators": ("int", "200", "2000"),
        "max_depth": ("int", "3", "12"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "subsample": ("float", "0.5", "1.0"),
        "colsample_bytree": ("float", "0.4", "1.0"),
        "min_child_weight": ("int", "1", "30"),
        "reg_alpha": ("float_log", "0.001", "10.0"),
        "reg_lambda": ("float_log", "0.001", "10.0"),
        "gamma": ("float_log", "0.001", "10.0"),
    },
    "random_forest_regressor": {
        "n_estimators": ("int", "200", "1500"),
        "max_depth": ("categorical", "10", "20", "30", "50", "None"),
        "max_features": ("categorical", "sqrt", "log2"),
        "min_samples_leaf": ("int", "1", "30"),
        "min_samples_split": ("int", "2", "20"),
    },
    "lightgbm_regressor": {
        "n_estimators": ("int", "200", "2000"),
        "num_leaves": ("int", "15", "511"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "subsample": ("float", "0.5", "1.0"),
        "colsample_bytree": ("float", "0.4", "1.0"),
        "min_child_samples": ("int", "5", "100"),
        "reg_alpha": ("float_log", "0.001", "10.0"),
        "reg_lambda": ("float_log", "0.001", "10.0"),
        "min_split_gain": ("float_log", "0.0001", "1.0"),
    },
    "catboost_regressor": {
        "iterations": ("int", "300", "2000"),
        "depth": ("int", "4", "10"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "l2_leaf_reg": ("float_log", "1.0", "30.0"),
        "bagging_temperature": ("float", "0.0", "1.0"),
        "border_count": ("int", "32", "255"),
        "random_strength": ("float_log", "0.1", "10.0"),
    },
    "mlp_regressor": _T1_SPACES["mlp_classifier"],
    "cnn1d_regressor": _T1_SPACES["cnn1d_classifier"],
    "deep_tobit_regressor": _T1_SPACES["deep_tobit_classifier"],
    "zi_tobit_regressor": _T1_SPACES["zi_tobit_classifier"],
    "icp_regressor": _T1_SPACES["icp_classifier"],
    "tobit_regressor": {
        "max_iter": ("int", "200", "5000"),
        "method": ("categorical", "L-BFGS-B", "Powell", "trust-constr"),
    },
    "aft_regressor": {
        "penalizer": ("float_log", "0.0001", "10.0"),
        "y_max_multiplier": ("float", "1.05", "1.5"),
    },
    "xgboost_aft_regressor": {
        "n_estimators": ("int", "200", "2000"),
        "max_depth": ("int", "3", "12"),
        "learning_rate": ("float_log", "0.005", "0.3"),
        "subsample": ("float", "0.5", "1.0"),
        "colsample_bytree": ("float", "0.4", "1.0"),
        "min_child_weight": ("int", "1", "30"),
        "aft_loss_distribution": ("categorical", "normal", "logistic", "extreme"),
        "aft_loss_distribution_scale": ("float_log", "0.1", "5.0"),
        "reg_alpha": ("float_log", "0.001", "10.0"),
        "reg_lambda": ("float_log", "0.001", "10.0"),
    },
    "hurdle_regressor": {
        "gate_n_estimators": ("int", "200", "1500"),
        "gate_max_depth": ("int", "3", "10"),
        "gate_learning_rate": ("float_log", "0.005", "0.3"),
        "gate_min_child_weight": ("int", "1", "30"),
        "intensity_n_estimators": ("int", "100", "1000"),
        "intensity_max_depth": ("int", "2", "6"),
        "intensity_learning_rate": ("float_log", "0.005", "0.2"),
        "intensity_reg_alpha": ("float_log", "0.001", "10.0"),
        "intensity_reg_lambda": ("float_log", "0.001", "30.0"),
        "intensity_min_child_weight": ("int", "5", "50"),
        "log_transform": ("categorical", "True", "False"),
    },
}

# T3 / T4 (multilabel classification) — same spaces as T1 classifiers; metric
# will be mean-AUPRC at runtime. T5 / T6 / T7 also reuse T1 spaces (they vary
# the data split, not the model architecture). ``logistic_regression`` is the
# one classification model whose registry key lacks the ``classifier`` suffix,
# so it is matched explicitly — without it the derived tasks silently tuned
# logistic at the default C=1.0. _T1_SPACES holds only classifiers +
# logistic_regression (no regressors), so this keeps every classification model.
_T3_SPACES = {
    k: v for k, v in _T1_SPACES.items() if "classifier" in k or k == "logistic_regression"
}

# T4 is system-month detection — the spatial graph is 5.68x larger than T1's
# system-level graph (64,409 vs 11,344 nodes). KNN graphs in PyG are directed,
# so edge count is ~2*N*k. T1 at k=30 gives ~680k edges; capping T4 at k=10
# yields ~1.3M edges (~2x T1 — still tractable). The wider T1 bounds caused
# repeated CUDA wedges in T4 gnn_gcn HPO; this narrows the trigger surface.
# See PR #41 round 3 for the full root-cause analysis.
_T4_SPACES = dict(_T3_SPACES)
_T4_SPACES["gnn_gcn_classifier"] = {
    "n_hidden": ("categorical", "32", "64", "128"),
    "dropout": ("float", "0.1", "0.6"),
    "learning_rate": ("float_log", "0.0001", "0.05"),
    "epochs": ("int", "100", "200"),
    "patience": ("int", "10", "30"),
    "k": ("int", "3", "10"),
    "weight_decay": ("float_log", "1e-7", "0.01"),
}
_T4_SPACES["gnn_sage_classifier"] = dict(_T4_SPACES["gnn_gcn_classifier"])

# TabPFN hard-rejects training sets larger than its max_train_size (default
# 10000). T4 system-month detection has ~47k training rows after the
# geographic split, so every TabPFN trial fails at the size check in
# src/aquacontam/models/tabpfn.py:118 with ValueError. Dropping the entry
# from the T4 registry avoids wasting trials on a known inapplicability.
# T1 (~8k training rows) is still under the cap, so tabpfn_classifier stays
# in _T1_SPACES / _T3_SPACES. See PR #41 round 3 follow-up.
_T4_SPACES.pop("tabpfn_classifier", None)

# logistic_regression's T1 space allows saga + l1 + max_iter up to 5000. On
# T4's ~47k rows with high-cardinality one-hot features this is pathologically
# slow: saga never converges in the per-trial budget, and L1-regularised fits
# (liblinear — also where _fixup_logistic_solver routes lbfgs+l1) ran 50+ min
# for larger C because liblinear's coordinate descent costs ~13s/iteration on
# the wide feature matrix. L2 fits with lbfgs/liblinear converge in ~2-3s. T4
# therefore fixes the penalty to the fast, standard-baseline L2 and caps
# max_iter; the sweep then runs in-process (logistic is CPU-only) in minutes.
# Mirrors the GNN T4 narrowing above. See PR #41 item 3 follow-up.
_T4_SPACES["logistic_regression"] = {
    "C": ("float_log", "0.0001", "100.0"),
    "penalty": ("categorical", "l2"),
    "solver": ("categorical", "lbfgs", "liblinear"),
    "class_weight": ("categorical", "balanced", "None"),
    "max_iter": ("int", "200", "1000"),
}

_T5_SPACES = _T3_SPACES
# T6 (arsenic public-supply -> domestic transfer) tunes only the tabular tree/
# linear/MLP models that suit it (matching the standalone runner's model set in
# pipeline/t6_arsenic.py); GNN/TabPFN/deep models don't fit T6's ~26k-well
# geogenic-arsenic task (TabPFN exceeds its 10k cap; GNN builds a large graph
# per trial). logistic_regression is retained (part of the T6 set and asserted
# by tests).
_T6_MODEL_KEYS = (
    "logistic_regression",
    "random_forest_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
    "catboost_classifier",
    "mlp_classifier",
)
_T6_SPACES = {k: v for k, v in _T3_SPACES.items() if k in _T6_MODEL_KEYS}
# T6's public-supply matrix has the same wide one-hot (aquifer) as T4, so reuse
# T4's narrowed logistic space (L2, no saga, capped max_iter) — the wide saga/L1
# space is pathologically slow on a wide one-hot matrix (see the T4 narrowing).
_T6_SPACES["logistic_regression"] = _T4_SPACES["logistic_regression"]
_T7_SPACES = _T3_SPACES

SEARCH_SPACES_BY_TASK: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "T1": _T1_SPACES,
    "T2": _T2_SPACES,
    "T3": _T3_SPACES,
    "T4": _T4_SPACES,
    "T5": _T5_SPACES,
    "T6": _T6_SPACES,
    "T7": _T7_SPACES,
}

# Backward-compat alias: legacy code imports SEARCH_SPACES expecting T1 shape.
SEARCH_SPACES: dict[str, dict[str, tuple[str, ...]]] = _T1_SPACES


# ---------------------------------------------------------------------------
# Per-task metric defaults (objective direction inferred elsewhere)
# ---------------------------------------------------------------------------

DEFAULT_METRICS: dict[str, str] = {
    "T1": "auprc",
    "T2": "rmse",
    # T3 / T4 are multilabel in the full benchmark, but the HPO prep helpers
    # in benchmark/_prep_data.py train on a single analyte (binary task) for
    # tractability — so we tune on plain auprc, not mean_auprc. The
    # ``compute_classification_metrics`` helper doesn't recognize mean_auprc
    # and would return nothing → every trial fail_value=-inf.
    "T3": "auprc",
    "T4": "auprc",
    "T5": "auprc",
    "T6": "auprc",
    "T7": "auprc",
}

DEFAULT_DIRECTIONS: dict[str, str] = {
    "auprc": "maximize",
    "auroc": "maximize",
    "f1": "maximize",
    "mean_auprc": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
    "r2": "maximize",
}

# Metric -> task_type for ``BaseModel.evaluate()`` dispatch.
_METRIC_TASK_TYPE: dict[str, str] = {
    "rmse": "regression",
    "mae": "regression",
    "r2": "regression",
    "auprc": "classification",
    "auroc": "classification",
    "f1": "classification",
    "accuracy": "classification",
    "balanced_accuracy": "classification",
    "precision": "classification",
    "recall": "classification",
    "mean_auprc": "classification",
}


# ---------------------------------------------------------------------------
# Param suggest + transforms
# ---------------------------------------------------------------------------


def _apply_param_transforms(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Replace categorical strings with their list-valued counterparts."""
    transforms = PARAM_TRANSFORMS.get(model_name, {})
    for param_name, mapping in transforms.items():
        if param_name in config and config[param_name] in mapping:
            config[param_name] = mapping[config[param_name]]
    return config


def _suggest_param(trial: Any, name: str, spec: tuple[str, ...]) -> Any:
    """Suggest a hyperparameter value from a type spec.

    Type specs:
      - ``("int", low, high)``
      - ``("float", low, high)``
      - ``("float_log", low, high)``
      - ``("categorical", *choices)`` — ``"None"`` decodes to ``None``,
        digits to ``int``, ``"True"``/``"False"`` to ``bool``.
    """
    type_spec = spec[0]
    if type_spec == "int":
        return trial.suggest_int(name, int(spec[1]), int(spec[2]))
    if type_spec == "float":
        return trial.suggest_float(name, float(spec[1]), float(spec[2]))
    if type_spec == "float_log":
        return trial.suggest_float(name, float(spec[1]), float(spec[2]), log=True)
    if type_spec == "categorical":
        choices: list[Any] = []
        for v in spec[1:]:
            if v == "None":
                choices.append(None)
            elif v == "True":
                choices.append(True)
            elif v == "False":
                choices.append(False)
            elif isinstance(v, str) and v.lstrip("-").isdigit():
                choices.append(int(v))
            else:
                choices.append(v)
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unknown type spec: {type_spec}")


# ---------------------------------------------------------------------------
# Sampler / pruner factories
# ---------------------------------------------------------------------------


def _make_sampler(name: str, seed: int) -> Any:
    import optuna

    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "tpe_multi":
        return optuna.samplers.TPESampler(
            seed=seed, multivariate=True, group=True, constant_liar=True
        )
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"Unknown sampler: {name}")


def _make_pruner(name: str) -> Any:
    import optuna

    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=10, reduction_factor=3)
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown pruner: {name}")


# ---------------------------------------------------------------------------
# Conditional-sampling fixups (post-suggest)
# ---------------------------------------------------------------------------


def _fixup_logistic_solver(config: dict[str, Any]) -> None:
    """Ensure logistic regression solver supports the chosen penalty."""
    if config.get("penalty") == "l1" and config.get("solver") == "lbfgs":
        config["solver"] = "liblinear"


def _expand_hurdle_prefixes(config: dict[str, Any]) -> dict[str, Any]:
    """Reshape gate_*/intensity_* flat keys into nested gate_config/intensity_config."""
    gate_cfg: dict[str, Any] = {}
    intensity_cfg: dict[str, Any] = {}
    out: dict[str, Any] = {}
    for k, v in config.items():
        if k.startswith("gate_"):
            gate_cfg[k[len("gate_") :]] = v
        elif k.startswith("intensity_"):
            intensity_cfg[k[len("intensity_") :]] = v
        else:
            out[k] = v
    if gate_cfg:
        out["gate_config"] = gate_cfg
    if intensity_cfg:
        out["intensity_config"] = intensity_cfg
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def optuna_search(
    model_cls: type,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    *,
    search_space: dict[str, tuple[str, ...]] | None = None,
    n_trials: int = 50,
    metric: str = "auprc",
    base_config: dict[str, Any] | None = None,
    seed: int = 42,
    timeout: float | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    sampler: str = "tpe_multi",
    pruner: str = "hyperband",
    direction: str | None = None,
    gpu_id: int | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    model_name: str | None = None,
    inproc: bool = False,
) -> tuple[dict[str, Any], Any]:
    """Run Bayesian hyperparameter optimization with Optuna.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Model class to optimize.
    X_train, y_train, X_val, y_val : array-like
        Training and validation data.
    search_space : dict, optional
        Parameter space; defaults to ``SEARCH_SPACES`` lookup by model name.
    n_trials : int, default 50
        Number of trials.
    metric : str, default "auprc"
        Metric to optimize. Direction inferred from ``DEFAULT_DIRECTIONS``
        unless ``direction`` is explicit.
    base_config : dict, optional
        Base config merged into each trial's suggested params.
    seed : int
        Sampler seed.
    timeout : float, optional
        Max wall-clock seconds.
    storage : str, optional
        SQLite URL (e.g. ``sqlite:///path/to/study.db``). Enables resume
        across runs via ``load_if_exists=True``.
    study_name : str, optional
        Study name; required when ``storage`` is set.
    sampler : {"tpe", "tpe_multi", "cmaes"}, default "tpe_multi"
    pruner : {"hyperband", "median", "none"}, default "hyperband"
    direction : {"maximize", "minimize"}, optional
        Overrides default inferred from metric.
    gpu_id : int, optional
        Forwarded into ``base_config`` as ``use_gpu=True, gpu_id=N`` for
        tree-library models. After ``CUDA_VISIBLE_DEVICES`` remap, pass
        ``gpu_id=0``.
    fit_kwargs : dict, optional
        Extra kwargs to pass to ``model.fit()`` (e.g. ``censored``,
        ``detection_limits`` for T2; ``X_val``, ``y_val`` are always passed).
    model_name : str, optional
        Registry key used for ``SEARCH_SPACES`` and ``PARAM_TRANSFORMS``
        lookup. Defaults to ``model_cls(config={}).name``. Pass explicitly
        when the class's ``name`` property doesn't match the registry key
        (e.g. ``VotingEnsembleClassifier.name == "voting_ensemble"`` but
        the registry key is ``"voting_ensemble_classifier"``).
    inproc : bool, default False
        If True, run trials in-process (legacy path). Provides no crash
        isolation — a wedged CUDA kernel hangs the whole sweep. Use only
        for debugging the subprocess machinery itself.

    Returns
    -------
    (best_config, study) : tuple
    """
    import optuna

    from aquacontam.benchmark._trial_runner import (
        get_timeout,
        run_trial_inproc,
        run_trial_with_timeout,
    )

    # Honor the env-var escape hatch in addition to the kwarg.
    if not inproc and os.environ.get("AQUACONTAM_HPO_INPROC", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    ):
        inproc = True

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Determine model name for search-space lookup and param transforms.
    # Caller-supplied ``model_name`` wins; otherwise fall back to the class's
    # ``.name`` property. A defensive warning flags the case where the two
    # disagree, which silently disables param transforms.
    inferred_name: str | None = None
    try:
        tmp = model_cls(config={})
        inferred_name = tmp.name
    except (ValueError, TypeError, RuntimeError):
        pass
    if model_name is None:
        model_name = inferred_name
    elif inferred_name is not None and inferred_name != model_name:
        logger.warning(
            "Inferred class name %r differs from caller-supplied model_name %r; "
            "using %r for PARAM_TRANSFORMS lookup",
            inferred_name,
            model_name,
            model_name,
        )

    if search_space is None:
        if model_name and model_name in SEARCH_SPACES:
            search_space = SEARCH_SPACES[model_name]
        else:
            raise ValueError(
                f"No predefined search space for {model_cls.__name__}. "
                f"Provide search_space explicitly. "
                f"Available: {sorted(SEARCH_SPACES)}"
            )

    base = dict(base_config or {})
    if gpu_id is not None:
        base.setdefault("use_gpu", True)
        base.setdefault("gpu_id", gpu_id)

    fit_kwargs = dict(fit_kwargs or {})

    if direction is None:
        direction = DEFAULT_DIRECTIONS.get(metric, "maximize")

    fail_value = float("-inf") if direction == "maximize" else float("inf")
    task_type = _METRIC_TASK_TYPE.get(metric, "classification")

    # Cache numpy conversions of X/y once per study so per-trial pickling is
    # fast (DataFrame pickle is 5-10x slower than numpy). The arrays will be
    # the same across all N trials of this study, so do the conversion once.
    def _to_ndarray(arr: Any) -> Any:
        if isinstance(arr, pd.DataFrame):
            return arr.to_numpy()
        if isinstance(arr, pd.Series):
            return arr.to_numpy()
        return arr

    X_train_arr = _to_ndarray(X_train)
    y_train_arr = _to_ndarray(y_train)
    X_val_arr = _to_ndarray(X_val)
    y_val_arr = _to_ndarray(y_val)

    # Build the env dict the subprocess will export before importing torch.
    # post-CVD-remap gpu_id is always 0 (the only visible device in the child).
    trial_env: dict[str, str] = {}
    if gpu_id is not None:
        trial_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        trial_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu_id))
        trial_env["AQUACONTAM_GPU_ID"] = "0"
    trial_env.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    )
    if "PYTHONPATH" in os.environ:
        trial_env["PYTHONPATH"] = os.environ["PYTHONPATH"]

    timeout_per_trial = get_timeout(model_name or model_cls.__name__)

    def objective(trial: optuna.Trial) -> float:
        config = dict(base)
        for param_name, spec in search_space.items():  # type: ignore[union-attr]
            config[param_name] = _suggest_param(trial, param_name, spec)
        config["random_state"] = seed
        if model_name:
            _apply_param_transforms(config, model_name)
            if model_name == "logistic_regression":
                _fixup_logistic_solver(config)
            if model_name == "hurdle_regressor":
                config = _expand_hurdle_prefixes(config)
        # Strip "auto" sentinels that constructors don't accept.
        if config.get("scale_pos_weight") == "auto":
            config.pop("scale_pos_weight")

        if inproc:
            result = run_trial_inproc(
                model_cls=model_cls,
                config=config,
                X_train=X_train_arr,
                y_train=y_train_arr,
                X_val=X_val_arr,
                y_val=y_val_arr,
                fit_kwargs=fit_kwargs,
                metric=metric,
                task_type=task_type,
            )
        else:
            result = run_trial_with_timeout(
                model_module=model_cls.__module__,
                model_class_name=model_cls.__name__,
                config=config,
                X_train=X_train_arr,
                y_train=y_train_arr,
                X_val=X_val_arr,
                y_val=y_val_arr,
                fit_kwargs=fit_kwargs,
                metric=metric,
                task_type=task_type,
                timeout_s=timeout_per_trial,
                env=trial_env,
                gpu_id=gpu_id,
                model_key=model_name or model_cls.__name__,
            )

        if result.status == "ok" and result.value is not None:
            return float(result.value)
        # timeout / error / crash — log and return fail sentinel so Optuna
        # marks the trial COMPLETE with fail_value and proceeds.
        logger.warning(
            "Trial %d failed (%s): %s",
            trial.number,
            result.status,
            result.message,
        )
        return fail_value

    sampler_obj = _make_sampler(sampler, seed)
    pruner_obj = _make_pruner(pruner)

    # Apply WAL mode to sqlite storage URLs. Prevents torn writes when a
    # subprocess child is killed mid-DB-update (timeout path).
    storage_for_optuna = storage
    if (
        storage_for_optuna
        and storage_for_optuna.startswith("sqlite:")
        and "journal_mode" not in storage_for_optuna
    ):
        sep = "&" if "?" in storage_for_optuna else "?"
        storage_for_optuna = f"{storage_for_optuna}{sep}journal_mode=WAL"

    if storage_for_optuna is not None:
        if study_name is None:
            study_name = f"{model_name or model_cls.__name__}_seed{seed}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_for_optuna,
            sampler=sampler_obj,
            pruner=pruner_obj,
            direction=direction,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(sampler=sampler_obj, pruner=pruner_obj, direction=direction)

    # Stale-RUNNING cleanup: mark trials still in RUNNING state from a prior
    # process kill as FAIL. Without this, the study DB accumulates phantom
    # in-flight trials. Wrapped in broad except because the storage API is
    # semi-private and could change between Optuna versions.
    try:
        from datetime import datetime, timedelta

        stale_threshold = datetime.now() - timedelta(hours=1)
        running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        stale_count = 0
        for trial in running:
            if trial.datetime_start and trial.datetime_start < stale_threshold:
                study._storage.set_trial_state_values(
                    trial._trial_id, state=optuna.trial.TrialState.FAIL
                )
                stale_count += 1
        if stale_count:
            logger.info(
                "optuna_search: marked %d stale RUNNING trial(s) as FAIL in %s",
                stale_count,
                study_name,
            )
    except (AttributeError, RuntimeError, ImportError) as exc:
        logger.warning("optuna_search: stale-trial cleanup failed: %s (continuing)", exc)

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_config = dict(base)
    best_config.update(study.best_params)
    best_config["random_state"] = seed
    if model_name:
        _apply_param_transforms(best_config, model_name)
        if model_name == "logistic_regression":
            _fixup_logistic_solver(best_config)
        if model_name == "hurdle_regressor":
            best_config = _expand_hurdle_prefixes(best_config)

    logger.info(
        "Optuna HPO complete: best %s=%.4f after %d trials (sampler=%s, pruner=%s)",
        metric,
        study.best_value,
        len(study.trials),
        sampler,
        pruner,
    )

    return best_config, study


# ---------------------------------------------------------------------------
# Best-config persistence helpers
# ---------------------------------------------------------------------------


def load_best_config(
    model_name: str,
    task: str = "T1",
    results_path: Path | str = "results/optuna_tuning.json",
) -> dict[str, Any] | None:
    """Return tuned ``best_config`` for ``(model_name, task)`` or ``None``.

    Tolerates legacy entries that lack a ``task`` field by treating them
    as T1 (the only task the legacy driver tuned).
    """
    p = Path(results_path)
    if not p.exists():
        return None
    try:
        entries = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_task = entry.get("task", "T1")  # legacy entries default to T1
        if entry.get("model") == model_name and entry_task == task:
            cfg = entry.get("best_config")
            if isinstance(cfg, dict):
                return dict(cfg)
    return None
