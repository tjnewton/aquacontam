"""LightGBM baseline models for classification and regression."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aquacontam.models._configs import LightGBMClassifierConfig, LightGBMRegressorConfig
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _apply_gpu_params(params: dict[str, Any]) -> None:
    """Translate ``use_gpu``/``gpu_id`` into LightGBM GPU API in-place.

    LightGBM 4.x PyPI wheels build with ``device='gpu'`` (OpenCL).
    Falls back silently to CPU if the wheel was built without GPU.
    """
    use_gpu = params.pop("use_gpu", False)
    gpu_id = params.pop("gpu_id", 0)
    if use_gpu:
        params.setdefault("device", "gpu")
        params.setdefault("gpu_platform_id", 0)
        params.setdefault("gpu_device_id", gpu_id)


class LightGBMClassifier(BaseModel):
    """LightGBM binary classifier with auto class weighting.

    Parameters
    ----------
    config : dict, optional
        LightGBM hyperparameters. Set ``is_unbalance=True`` (default)
        to let LightGBM handle class imbalance automatically.
    """

    _config_type = LightGBMClassifierConfig

    @property
    def name(self) -> str:
        return "lightgbm_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the LightGBM classifier.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training labels (0/1).
        **kwargs
            Optional: ``X_val``, ``y_val`` for early stopping.
        """
        params = dict(self.config)
        _apply_gpu_params(params)

        # Enable automatic class balancing by default
        params.setdefault("is_unbalance", True)

        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        callbacks: list[Any] = []
        if X_val is not None and y_val is not None and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        self._model = lgb.LGBMClassifier(**params)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self._model.fit(X_train, y_train, **fit_kwargs)


class LightGBMRegressor(BaseModel):
    """LightGBM regressor for concentration prediction.

    Parameters
    ----------
    config : dict, optional
        LightGBM hyperparameters.
    """

    _config_type = LightGBMRegressorConfig

    @property
    def name(self) -> str:
        return "lightgbm_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the LightGBM regressor."""
        params = dict(self.config)
        _apply_gpu_params(params)
        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        callbacks: list[Any] = []
        if X_val is not None and y_val is not None and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        self._model = lgb.LGBMRegressor(**params)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self._model.fit(X_train, y_train, **fit_kwargs)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
