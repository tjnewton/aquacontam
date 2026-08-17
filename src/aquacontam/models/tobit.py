"""Tobit Type I censored regression via maximum likelihood estimation.

Handles left-censored water quality data where non-detect observations
are known only to be below the detection limit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class TobitRegressor(BaseModel):
    """Tobit Type I left-censored regression model.

    Maximizes the censored normal log-likelihood where detected observations
    contribute the normal PDF and left-censored observations contribute the
    normal CDF evaluated at the detection limit.

    Parameters
    ----------
    config : dict, optional
        Model configuration. Supported keys:

        - ``max_iter`` : int, default 1000
            Maximum iterations for the optimizer.
        - ``method`` : str, default "L-BFGS-B"
            Optimization method passed to ``scipy.optimize.minimize``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._beta: np.ndarray | None = None
        self._sigma: float | None = None
        self._feature_names: list[str] | None = None
        self._converged: bool | None = None

    @property
    def converged(self) -> bool | None:
        """Whether the MLE optimization converged. None if not yet fitted."""
        return self._converged

    @property
    def name(self) -> str:
        return "tobit_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the Tobit regression model via MLE.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features, shape ``(n_samples, n_features)``.
        y_train : pd.Series | np.ndarray
            Training target values.
        **kwargs
            Optional keyword arguments:

            - ``censored`` : array-like of bool
                Boolean mask where ``True`` indicates the observation is
                left-censored (non-detect).
            - ``detection_limits`` : array-like of float
                Detection limit for each observation. Required when
                ``censored`` contains any ``True`` values. For censored
                observations, ``y_train`` values are ignored in favor of
                the detection limit.
        """
        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)

        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)

        censored = kwargs.get("censored")
        detection_limits = kwargs.get("detection_limits")

        if censored is not None:
            censored = np.asarray(censored, dtype=bool)
        if detection_limits is not None:
            detection_limits = np.asarray(detection_limits, dtype=np.float64)

        n_samples, n_features = X.shape

        # Prepend intercept column
        X_aug = np.column_stack([np.ones(n_samples), X])

        # Initialize beta from OLS on detected-only subset
        if censored is not None and censored.any():
            detected_mask = ~censored
            X_det = X_aug[detected_mask]
            y_det = y[detected_mask]
        else:
            X_det = X_aug
            y_det = y

        # OLS: beta = (X'X)^{-1} X'y
        beta_init, residuals, _, _ = np.linalg.lstsq(X_det, y_det, rcond=None)
        if len(residuals) > 0 and len(y_det) > n_features + 1:
            sigma_init = np.sqrt(residuals[0] / len(y_det))
        else:
            sigma_init = np.std(y_det - X_det @ beta_init) + 1e-6

        # If no censoring, just use OLS solution
        if censored is None or not censored.any():
            self._beta = beta_init
            self._sigma = float(max(sigma_init, 1e-6))
            self._converged = True
            self._model = True  # mark as fitted
            logger.info("Tobit fit (no censoring): OLS solution, sigma=%.4f", self._sigma)
            return

        # Pack parameters: [beta..., log_sigma]
        theta0 = np.concatenate([beta_init, [np.log(max(sigma_init, 1e-6))]])

        def neg_log_likelihood(theta: np.ndarray) -> float:
            beta = theta[:-1]
            sigma = np.exp(theta[-1])
            mu = X_aug @ beta

            ll = 0.0

            # Detected (uncensored) observations: normal log-PDF
            det_mask = ~censored
            if det_mask.any():
                ll += np.sum(stats.norm.logpdf(y[det_mask], loc=mu[det_mask], scale=sigma))

            # Censored observations: normal log-CDF at detection limit
            cens_mask = censored
            if cens_mask.any():
                assert detection_limits is not None
                z = (detection_limits[cens_mask] - mu[cens_mask]) / sigma
                # Clip z to avoid log(0)
                log_cdf = stats.norm.logcdf(z)
                ll += np.sum(log_cdf)

            return -ll

        max_iter = self.config.get("max_iter", 1000)
        method = self.config.get("method", "L-BFGS-B")

        result = optimize.minimize(
            neg_log_likelihood,
            theta0,
            method=method,
            options={"maxiter": max_iter},
        )

        self._beta = result.x[:-1]
        self._sigma = float(np.exp(result.x[-1]))
        self._converged = bool(result.success)
        self._model = True  # mark as fitted

        n_censored = int(censored.sum())
        if not result.success:
            logger.warning(
                "Tobit MLE optimization did not converge: %s. "
                "Coefficients may be unreliable. Consider increasing max_iter "
                "or checking for data issues (collinearity, extreme values).",
                result.message,
            )
        logger.info(
            "Tobit MLE fit: %d detected, %d censored, sigma=%.4f, converged=%s",
            n_samples - n_censored,
            n_censored,
            self._sigma,
            result.success,
        )

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict expected values (latent variable mean).

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
        X_arr = np.asarray(X, dtype=np.float64)
        assert self._beta is not None
        # beta[0] is intercept, beta[1:] are feature coefficients
        return np.asarray(X_arr @ self._beta[1:] + self._beta[0])

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression.

        Raises
        ------
        NotImplementedError
            Always raised; Tobit is a regression model.
        """
        raise NotImplementedError("Regression models do not support predict_proba")
