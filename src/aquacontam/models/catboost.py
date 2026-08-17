"""CatBoost baseline models for classification and regression."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier as _CatBoostCls
from catboost import CatBoostRegressor as _CatBoostReg

from aquacontam.models._configs import CatBoostClassifierConfig, CatBoostRegressorConfig
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _apply_gpu_params(params: dict[str, Any]) -> None:
    """Translate ``use_gpu``/``gpu_id`` into CatBoost GPU API in-place.

    Sets ``task_type='GPU'`` and ``devices=str(gpu_id)``. CatBoost on GPU
    does not accept ``auto_class_weights='Balanced'`` reliably, so it is
    stripped when GPU is enabled (callers can pass explicit ``class_weights``).
    """
    use_gpu = params.pop("use_gpu", False)
    gpu_id = params.pop("gpu_id", 0)
    if use_gpu:
        params.setdefault("task_type", "GPU")
        params.setdefault("devices", str(gpu_id))
        if params.pop("auto_class_weights", None) is not None:
            logger.info(
                "CatBoost GPU: stripping auto_class_weights "
                "(not GPU-compatible); pass explicit class_weights if needed"
            )


class CatBoostClassifier(BaseModel):
    """CatBoost binary classifier with balanced class weights.

    Parameters
    ----------
    config : dict, optional
        CatBoost hyperparameters. ``auto_class_weights="Balanced"`` is set
        by default. ``verbose=0`` suppresses training output.
    """

    _config_type = CatBoostClassifierConfig

    @property
    def name(self) -> str:
        return "catboost_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the CatBoost classifier.

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
        params.setdefault("auto_class_weights", "Balanced")
        params.setdefault("verbose", 0)
        _apply_gpu_params(params)

        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        if early_stopping_rounds is not None:
            params["early_stopping_rounds"] = early_stopping_rounds

        self._model = _CatBoostCls(**params)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        fit_kwargs: dict[str, Any] = {"verbose": 0}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        logger.info(
            "Trained CatBoost classifier with %d iterations",
            params.get("iterations", self._model.tree_count_),
        )

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels.

        Returns
        -------
        np.ndarray
            Integer class labels (0 or 1).
        """
        return np.asarray(self._model.predict(X)).astype(int)


class CatBoostRegressor(BaseModel):
    """CatBoost regressor for concentration prediction.

    Parameters
    ----------
    config : dict, optional
        CatBoost hyperparameters. ``verbose=0`` suppresses training output.
    """

    _config_type = CatBoostRegressorConfig

    @property
    def name(self) -> str:
        return "catboost_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the CatBoost regressor.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training targets.
        **kwargs
            Optional: ``X_val``, ``y_val`` for early stopping.
        """
        params = dict(self.config)
        params.setdefault("verbose", 0)
        _apply_gpu_params(params)

        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        if early_stopping_rounds is not None:
            params["early_stopping_rounds"] = early_stopping_rounds

        self._model = _CatBoostReg(**params)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        fit_kwargs: dict[str, Any] = {"verbose": 0}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        logger.info(
            "Trained CatBoost regressor with %d iterations",
            params.get("iterations", self._model.tree_count_),
        )

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
