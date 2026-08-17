"""TypedDict definitions for model configuration validation.

Each model family has a TypedDict that documents the accepted hyperparameters.
These are used for **warning-only** validation — unknown keys trigger a warning
but do not raise errors, since underlying frameworks may accept extra params.
"""

from __future__ import annotations

from typing import TypedDict


class XGBoostClassifierConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.xgboost.XGBoostClassifier`."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float
    gamma: float
    scale_pos_weight: float | str
    early_stopping_rounds: int | None
    eval_metric: str
    random_state: int
    use_gpu: bool
    gpu_id: int
    device: str
    tree_method: str


class XGBoostRegressorConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.xgboost.XGBoostRegressor`."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float
    gamma: float
    early_stopping_rounds: int | None
    eval_metric: str
    random_state: int
    use_gpu: bool
    gpu_id: int
    device: str
    tree_method: str


class CatBoostClassifierConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.catboost.CatBoostClassifier`."""

    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    bagging_temperature: float
    border_count: int
    random_strength: float
    auto_class_weights: str
    class_weights: list[float]
    early_stopping_rounds: int | None
    verbose: int
    random_state: int
    random_seed: int
    use_gpu: bool
    gpu_id: int
    task_type: str
    devices: str


class CatBoostRegressorConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.catboost.CatBoostRegressor`."""

    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    bagging_temperature: float
    border_count: int
    random_strength: float
    early_stopping_rounds: int | None
    verbose: int
    random_state: int
    random_seed: int
    use_gpu: bool
    gpu_id: int
    task_type: str
    devices: str


class LightGBMClassifierConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.lightgbm.LightGBMClassifier`."""

    n_estimators: int
    num_leaves: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_samples: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    is_unbalance: bool
    early_stopping_rounds: int | None
    verbose: int
    random_state: int
    use_gpu: bool
    gpu_id: int
    device: str
    gpu_platform_id: int
    gpu_device_id: int


class LightGBMRegressorConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.lightgbm.LightGBMRegressor`."""

    n_estimators: int
    num_leaves: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_samples: int
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float
    early_stopping_rounds: int | None
    verbose: int
    random_state: int
    use_gpu: bool
    gpu_id: int
    device: str
    gpu_platform_id: int
    gpu_device_id: int


class RandomForestClassifierConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.random_forest.RandomForestClassifier`."""

    n_estimators: int
    max_features: str | float
    max_depth: int | None
    min_samples_leaf: int
    min_samples_split: int
    class_weight: str | None
    n_jobs: int
    random_state: int


class RandomForestRegressorConfig(TypedDict, total=False):
    """Hyperparameters for :class:`~aquacontam.models.random_forest.RandomForestRegressor`."""

    n_estimators: int
    max_features: str | float
    max_depth: int | None
    min_samples_leaf: int
    min_samples_split: int
    n_jobs: int
    random_state: int
