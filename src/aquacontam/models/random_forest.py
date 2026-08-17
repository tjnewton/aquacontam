"""Random Forest baseline models for classification and regression."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier as _SklearnRFC,
)
from sklearn.ensemble import (
    RandomForestRegressor as _SklearnRFR,
)

from aquacontam.models._configs import RandomForestClassifierConfig, RandomForestRegressorConfig
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestClassifier(BaseModel):
    """Random Forest classifier with balanced class weights.

    Default configuration uses ``class_weight="balanced"``,
    ``n_estimators=500``, ``max_features="sqrt"``, ``n_jobs=-1``.
    """

    _config_type = RandomForestClassifierConfig

    @property
    def name(self) -> str:
        return "random_forest_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Random Forest classifier."""
        params: dict[str, Any] = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        params.update(self.config)

        self._model = _SklearnRFC(**params)
        self._model.fit(X_train, y_train)
        logger.info("Trained RF classifier with %d estimators", params["n_estimators"])


class RandomForestRegressor(BaseModel):
    """Random Forest regressor for concentration prediction.

    Default configuration uses ``n_estimators=500``,
    ``max_features="sqrt"``, ``n_jobs=-1``.
    """

    _config_type = RandomForestRegressorConfig

    @property
    def name(self) -> str:
        return "random_forest_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Random Forest regressor."""
        params: dict[str, Any] = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "n_jobs": -1,
        }
        params.update(self.config)

        self._model = _SklearnRFR(**params)
        self._model.fit(X_train, y_train)
        logger.info("Trained RF regressor with %d estimators", params["n_estimators"])

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")
