"""Logistic Regression and Dummy (class prior) baseline models.

These serve as simple baselines to contextualize performance of more
complex models. If XGBoost doesn't meaningfully outperform logistic
regression, the feature engineering may be more important than model
complexity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier as _SklearnDummy
from sklearn.linear_model import LogisticRegression as _SklearnLR
from sklearn.preprocessing import StandardScaler

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier(BaseModel):
    """Logistic Regression classifier with standardized features.

    Default configuration uses ``class_weight="balanced"``,
    ``max_iter=1000``, ``solver="lbfgs"``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._scaler: StandardScaler | None = None

    @property
    def name(self) -> str:
        return "logistic_regression"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Logistic Regression classifier."""
        params: dict[str, Any] = {
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
        }
        params.update(self.config)

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        self._model = _SklearnLR(**params)
        self._model.fit(X_scaled, y_train)
        # Store feature names for alignment
        if hasattr(X_train, "columns"):
            self.feature_names_in_ = list(X_train.columns)
        logger.info("Trained LogisticRegression (C=%.4f)", self._model.C)

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]
        return np.asarray(self._model.predict(X_scaled))

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]
        return np.asarray(self._model.predict_proba(X_scaled))


class DummyClassifierBaseline(BaseModel):
    """Dummy classifier that predicts based on class prior distribution.

    Always predicts the most frequent class (or samples from the class
    distribution when ``strategy="stratified"``). Used to establish the
    minimum performance floor.
    """

    @property
    def name(self) -> str:
        return "dummy_classifier"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Fit the dummy classifier (learns class distribution)."""
        strategy = self.config.get("strategy", "prior")
        self._model = _SklearnDummy(
            strategy=strategy, random_state=self.config.get("random_state")
        )
        self._model.fit(X_train, y_train)
        # Store feature names for alignment
        if hasattr(X_train, "columns"):
            self.feature_names_in_ = list(X_train.columns)
        logger.info("Fitted DummyClassifier (strategy=%s)", strategy)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        return np.asarray(self._model.predict_proba(X))
