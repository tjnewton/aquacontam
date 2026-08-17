"""Abstract base class for all baseline models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class that every model implementation must follow.

    Parameters
    ----------
    config : dict
        Model hyperparameters and configuration.
    """

    requires_censoring_metadata: bool = False

    #: Set True in subclasses that need per-sample coordinates passed to ``fit``
    #: (e.g. graph neural networks). Only such models trigger the coordinate
    #: alignment/subsetting in the benchmark task runners; all other models train
    #: on the full split regardless of coordinate availability.
    requires_coords: bool = False

    #: Set in subclasses to a TypedDict class from ``_configs.py``
    #: to enable warning-only validation of config keys.
    _config_type: type | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._model: Any = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Warn on unrecognized config keys (if ``_config_type`` is set)."""
        if self._config_type is None or not self.config:
            return
        known = set(self._config_type.__annotations__)
        unknown = set(self.config) - known
        if unknown:
            logger.warning(
                "Unknown config keys for %s: %s (known: %s)",
                type(self).__name__,
                sorted(unknown),
                sorted(known),
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this model (e.g., 'xgboost', 'gnn')."""

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training labels.
        """

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate predictions.

        Default implementation delegates to ``self._model.predict(X)``.
        Override in subclasses that need custom prediction logic
        (e.g., torch-based models, censoring-aware models).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values (regression) or class labels (classification).
        """
        self._check_fitted()
        return np.asarray(self._model.predict(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate probability estimates (classification tasks only).

        Default implementation delegates to ``self._model.predict_proba(X)``.
        Regressor subclasses should override to raise ``NotImplementedError``.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Probability estimates, shape (n_samples, n_classes).
        """
        self._check_fitted()
        return np.asarray(self._model.predict_proba(X))

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted — call .fit() first")

    def feature_importances(self) -> pd.Series:
        """Get feature importance scores from the underlying model.

        Returns
        -------
        pd.Series
            Feature names → importance values, sorted descending.
        """
        self._check_fitted()
        if not hasattr(self._model, "feature_importances_"):
            raise NotImplementedError(f"{self.name} does not support feature_importances()")
        importances = self._model.feature_importances_
        names: list[str]
        if hasattr(self._model, "feature_names_in_"):
            names = list(self._model.feature_names_in_)
        else:
            names = [f"f{i}" for i in range(len(importances))]
        return cast(
            pd.Series,
            pd.Series(importances, index=names, name="importance").sort_values(ascending=False),
        )

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        metrics: list[str] | None = None,
        *,
        task_type: str = "classification",
    ) -> dict[str, float]:
        """Evaluate the model on a test set.

        Parameters
        ----------
        X_test : pd.DataFrame | np.ndarray
            Test features.
        y_test : pd.Series | np.ndarray
            True labels.
        metrics : list[str] | None
            Metric names to compute. Defaults to ``["accuracy", "auroc"]``
            for classification or ``["rmse", "mae", "r2"]`` for regression.
        task_type : str
            ``"classification"`` or ``"regression"``.

        Returns
        -------
        dict[str, float]
            Mapping of metric name to computed value.
        """
        self._check_fitted()
        preds = self.predict(X_test)

        if task_type == "regression":
            from aquacontam.benchmark.metrics import compute_regression_metrics

            if metrics is None:
                metrics = ["rmse", "mae", "r2"]
            return compute_regression_metrics(y_test, preds, metrics=metrics)

        # Classification (default for backward compat)
        from aquacontam.benchmark.metrics import compute_classification_metrics

        if metrics is None:
            metrics = ["accuracy", "auroc"]

        try:
            probas = self.predict_proba(X_test)
            if probas.ndim == 2 and probas.shape[1] == 2:
                probas = probas[:, 1]
        except NotImplementedError:
            probas = None

        return compute_classification_metrics(y_test, preds, probas, metrics=metrics)
