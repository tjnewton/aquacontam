"""Post-hoc probability calibration methods.

Provides isotonic, Platt (sigmoid), and temperature scaling calibration
for fitted classification models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from aquacontam.models.base import BaseModel

# ---------------------------------------------------------------------------
# Temperature-scaling helpers
# ---------------------------------------------------------------------------


def _apply_temperature(raw_proba: np.ndarray, temperature: float) -> np.ndarray:
    """Scale class probabilities by *temperature*.

    Parameters
    ----------
    raw_proba : np.ndarray
        Shape ``(n, 2)`` raw probability estimates.
    temperature : float
        Temperature parameter (>0). Values >1 soften; <1 sharpen.

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` calibrated probabilities.
    """
    # Clip to avoid log(0)
    eps = 1e-15
    p1 = np.clip(raw_proba[:, 1], eps, 1 - eps)
    logits = np.log(p1 / (1 - p1))
    scaled_logits = logits / temperature
    cal_p1 = 1.0 / (1.0 + np.exp(-scaled_logits))
    return np.column_stack([1.0 - cal_p1, cal_p1])


def _temperature_nll(
    raw_proba: np.ndarray,
    y_true: np.ndarray,
    temperature: float,
) -> float:
    """Negative log-likelihood (binary cross-entropy) at a given temperature.

    Parameters
    ----------
    raw_proba : np.ndarray
        1-D array of P(class=1) estimates.
    y_true : np.ndarray
        1-D binary labels.
    temperature : float
        Temperature parameter.

    Returns
    -------
    float
        Mean negative log-likelihood.
    """
    eps = 1e-15
    p = np.clip(raw_proba, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    scaled_logits = logits / temperature
    cal_p = 1.0 / (1.0 + np.exp(-scaled_logits))
    cal_p = np.clip(cal_p, eps, 1 - eps)
    nll = -(y_true * np.log(cal_p) + (1 - y_true) * np.log(1 - cal_p))
    return float(np.mean(nll))


def _fit_temperature(raw_proba: np.ndarray, y_true: np.ndarray) -> float:
    """Find the optimal temperature by minimising NLL.

    Parameters
    ----------
    raw_proba : np.ndarray
        1-D array of P(class=1) estimates from the uncalibrated model.
    y_true : np.ndarray
        1-D binary labels.

    Returns
    -------
    float
        Optimal temperature.
    """
    result = minimize_scalar(
        lambda t: _temperature_nll(raw_proba, y_true, t),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Calibrator wrappers (isotonic / Platt)
# ---------------------------------------------------------------------------


class _IsotonicCalibrator:
    """Isotonic-regression calibrator that maps raw P(class=1) to calibrated probs."""

    def __init__(self) -> None:
        self._ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, proba_1: np.ndarray, y: np.ndarray) -> _IsotonicCalibrator:
        self._ir.fit(proba_1, y)
        return self

    def predict_proba(self, proba_1: np.ndarray) -> np.ndarray:
        """Return (n, 2) calibrated probabilities."""
        cal_p1 = np.clip(self._ir.predict(proba_1), 0.0, 1.0)
        return np.column_stack([1.0 - cal_p1, cal_p1])


class _PlattCalibrator:
    """Platt (sigmoid / logistic) calibrator on raw P(class=1)."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10_000)

    def fit(self, proba_1: np.ndarray, y: np.ndarray) -> _PlattCalibrator:
        self._lr.fit(proba_1.reshape(-1, 1), y)
        return self

    def predict_proba(self, proba_1: np.ndarray) -> np.ndarray:
        """Return (n, 2) calibrated probabilities."""
        return np.asarray(self._lr.predict_proba(proba_1.reshape(-1, 1)))


# ---------------------------------------------------------------------------
# CalibratedModel wrapper
# ---------------------------------------------------------------------------


class CalibratedModel(BaseModel):
    """Wraps a fitted model with post-hoc calibrated probabilities.

    Parameters
    ----------
    wrapped_model : BaseModel
        An already-fitted classification model.
    calibrator : Any
        A fitted calibrator object (``_IsotonicCalibrator``,
        ``_PlattCalibrator``) or a ``float`` temperature value.
    method : str
        One of ``"isotonic"``, ``"platt"``, ``"temperature"``.
    """

    def __init__(
        self,
        wrapped_model: BaseModel,
        calibrator: Any,
        method: str,
    ) -> None:
        super().__init__(config=wrapped_model.config)
        self._wrapped = wrapped_model
        self._calibrator = calibrator
        self._method = method
        self._temperature: float | None = None
        # Expose the underlying sklearn model for feature_importances, etc.
        self._model = wrapped_model._model

        if method == "temperature":
            self._temperature = float(calibrator)

    @property
    def name(self) -> str:
        return f"{self._wrapped.name}_calibrated_{self._method}"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Not supported -- use :func:`calibrate` to create a ``CalibratedModel``."""
        raise RuntimeError("CalibratedModel is already fitted -- use calibrate() to create one")

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict class labels (delegates to the wrapped model).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        return self._wrapped.predict(X, **kwargs)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return calibrated class probabilities.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)`` calibrated probabilities.
        """
        raw_proba = self._wrapped.predict_proba(X, **kwargs)

        if self._method == "temperature":
            assert self._temperature is not None
            return _apply_temperature(raw_proba, self._temperature)

        # isotonic / platt -- calibrator operates on P(class=1)
        return np.asarray(self._calibrator.predict_proba(raw_proba[:, 1]))


# ---------------------------------------------------------------------------
# Public calibration entry-point
# ---------------------------------------------------------------------------


def calibrate(
    model: BaseModel,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    *,
    method: str = "isotonic",
) -> CalibratedModel:
    """Calibrate a fitted classification model on held-out data.

    Parameters
    ----------
    model : BaseModel
        A fitted classification model.
    X_val : pd.DataFrame | np.ndarray
        Calibration features (held-out from training).
    y_val : pd.Series | np.ndarray
        Calibration labels.
    method : str
        Calibration method: ``"isotonic"``, ``"platt"``, or ``"temperature"``.

    Returns
    -------
    CalibratedModel
        A calibrated wrapper around the original model.

    Raises
    ------
    ValueError
        If *method* is not one of the supported values.
    """
    valid_methods = {"isotonic", "platt", "temperature"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    raw_proba = model.predict_proba(X_val)
    p1 = raw_proba[:, 1]
    y_arr = np.asarray(y_val)

    if method == "temperature":
        temperature = _fit_temperature(p1, y_arr)
        return CalibratedModel(model, temperature, method)

    calibrator: _IsotonicCalibrator | _PlattCalibrator
    if method == "isotonic":
        calibrator = _IsotonicCalibrator().fit(p1, y_arr)
    else:  # platt
        calibrator = _PlattCalibrator().fit(p1, y_arr)

    return CalibratedModel(model, calibrator, method)


# ---------------------------------------------------------------------------
# Reliability diagram helper
# ---------------------------------------------------------------------------


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute data for a reliability (calibration) diagram.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_prob : np.ndarray
        Predicted P(class=1).
    n_bins : int
        Number of bins for the calibration curve.

    Returns
    -------
    dict
        Keys: ``"fraction_of_positives"``, ``"mean_predicted_value"``,
        ``"ece"`` (expected calibration error), ``"n_bins"``.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # ECE: weighted average of |fraction_of_positives - mean_predicted_value|
    # Weight each bin by its fraction of total samples
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    # calibration_curve may drop empty bins, so align bin_counts
    non_empty = bin_counts > 0
    non_empty_counts = bin_counts[non_empty]
    total = non_empty_counts.sum()

    if total > 0 and len(non_empty_counts) == len(fraction_of_positives):
        weights = non_empty_counts / total
        ece = float(np.sum(weights * np.abs(fraction_of_positives - mean_predicted_value)))
    else:
        # Fallback: unweighted mean
        ece = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))

    return {
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "ece": ece,
        "n_bins": n_bins,
    }
