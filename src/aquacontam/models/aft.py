"""Accelerated Failure Time regression via lifelines.

Uses Weibull AFT model with left-censoring handled via time-reversal trick:
the response is reflected about its maximum so that left-censoring becomes
right-censoring, which lifelines handles natively.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class AFTRegressor(BaseModel):
    """Weibull Accelerated Failure Time regression for censored data.

    Handles left-censored water quality observations by reversing the time
    axis (``y_max - y``) so that left-censoring becomes right-censoring,
    fitting a Weibull AFT model via ``lifelines.WeibullAFTFitter``, then
    reversing predictions back to the original scale.

    Parameters
    ----------
    config : dict, optional
        Model configuration. Supported keys:

        - ``penalizer`` : float, default 0.01
            L2 penalizer strength for the AFT fitter.
        - ``y_max_multiplier`` : float, default 1.1
            Multiplier applied to ``y_train.max()`` to compute the
            reflection ceiling.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._y_max: float | None = None
        self._feature_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "aft_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Weibull AFT model.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features, shape ``(n_samples, n_features)``.
        y_train : pd.Series | np.ndarray
            Training target values (must be positive).
        **kwargs
            Optional keyword arguments:

            - ``censored`` : array-like of bool
                Boolean mask where ``True`` indicates the observation is
                left-censored. If ``None``, all observations are treated
                as fully observed.
        """
        from lifelines import WeibullAFTFitter

        # Convert to numpy / extract feature names
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
            feature_cols = list(X_train.columns)
            X_df = X_train.copy()
        else:
            X_arr = np.asarray(X_train, dtype=np.float64)
            feature_cols = [f"f{i}" for i in range(X_arr.shape[1])]
            self._feature_names = feature_cols
            X_df = pd.DataFrame(X_arr, columns=feature_cols)

        y = np.asarray(y_train, dtype=np.float64)

        censored = kwargs.get("censored")
        if censored is not None:
            censored = np.asarray(censored, dtype=bool)
            event_observed = ~censored
        else:
            event_observed = np.ones(len(y), dtype=bool)

        # Time-reversal trick: reflect y so left-censoring becomes right-censoring
        multiplier = self.config.get("y_max_multiplier", 1.1)
        self._y_max = float(y.max() * multiplier)
        reversed_y = self._y_max - y

        # Ensure strictly positive durations (required by Weibull)
        reversed_y = np.maximum(reversed_y, 1e-8)

        # Build lifelines DataFrame
        df = X_df.reset_index(drop=True).copy()
        df["T"] = reversed_y
        df["E"] = event_observed.astype(int)

        penalizer = self.config.get("penalizer", 0.01)
        fitter = WeibullAFTFitter(penalizer=penalizer)
        fitter.fit(df, duration_col="T", event_col="E")

        self._model = fitter

        n_censored = int((~event_observed).sum())
        logger.info(
            "Weibull AFT fit: %d observed, %d censored, y_max=%.4f",
            int(event_observed.sum()),
            n_censored,
            self._y_max,
        )

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict expected concentration values.

        Uses the fitted Weibull AFT model to predict median survival times
        in the reversed scale, then reflects back to the original scale.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values, shape ``(n_samples,)``.
        """
        self._check_fitted()
        assert self._y_max is not None
        assert self._feature_names is not None

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            X_df = pd.DataFrame(X_arr, columns=self._feature_names)

        X_df = X_df.reset_index(drop=True)

        # Predict median survival in reversed scale
        predicted_reversed = self._model.predict_median(X_df)
        predicted_reversed = np.asarray(predicted_reversed, dtype=np.float64).ravel()

        # Reverse the transformation
        predictions = self._y_max - predicted_reversed
        return predictions

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression.

        Raises
        ------
        NotImplementedError
            Always raised; AFT is a regression model.
        """
        raise NotImplementedError("Regression models do not support predict_proba")
