"""XGBoost baseline models for classification and regression."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from aquacontam.models._configs import XGBoostClassifierConfig, XGBoostRegressorConfig
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _apply_gpu_params(params: dict[str, Any]) -> None:
    """Translate ``use_gpu``/``gpu_id`` into XGBoost 3.x GPU API in-place.

    Sets ``device='cuda:{gpu_id}'`` and ``tree_method='hist'``. The legacy
    ``tree_method='gpu_hist'`` was removed in xgboost 3.x.
    """
    use_gpu = params.pop("use_gpu", False)
    gpu_id = params.pop("gpu_id", 0)
    if use_gpu:
        params.setdefault("device", f"cuda:{gpu_id}")
        params.setdefault("tree_method", "hist")


class XGBoostClassifier(BaseModel):
    """XGBoost binary classifier with auto class weighting.

    Parameters
    ----------
    config : dict, optional
        XGBoost hyperparameters. Supports ``scale_pos_weight="auto"``
        to compute weight from class imbalance ratio.
    """

    _config_type = XGBoostClassifierConfig

    @property
    def name(self) -> str:
        return "xgboost_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the XGBoost classifier.

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
        y_arr = np.asarray(y_train)

        # Auto-compute scale_pos_weight from class ratio
        if params.get("scale_pos_weight") == "auto":
            n_neg = int((y_arr == 0).sum())
            n_pos = int((y_arr == 1).sum())
            if n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos
                logger.info(
                    "Auto scale_pos_weight: %.2f (neg=%d, pos=%d)",
                    params["scale_pos_weight"],
                    n_neg,
                    n_pos,
                )
            else:
                params["scale_pos_weight"] = 1.0

        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        # Pass early_stopping_rounds to constructor (xgboost >= 1.6 API)
        if X_val is not None and y_val is not None and early_stopping_rounds:
            params["early_stopping_rounds"] = early_stopping_rounds

        self._model = xgb.XGBClassifier(**params)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)


class XGBoostRegressor(BaseModel):
    """XGBoost regressor for concentration prediction.

    Parameters
    ----------
    config : dict, optional
        XGBoost hyperparameters.
    """

    _config_type = XGBoostRegressorConfig

    @property
    def name(self) -> str:
        return "xgboost_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the XGBoost regressor."""
        params = dict(self.config)
        _apply_gpu_params(params)
        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")

        if X_val is not None and y_val is not None and early_stopping_rounds:
            params["early_stopping_rounds"] = early_stopping_rounds

        self._model = xgb.XGBRegressor(**params)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
