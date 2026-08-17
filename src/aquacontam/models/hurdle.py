"""Hurdle (two-part) regressor for zero-inflated concentration data.

Decomposes concentration prediction into two stages:
1. Gate: Binary classifier predicting P(detected) — trained on all systems
2. Intensity: Regressor predicting E[Y | detected] — trained only on detected systems

Combined prediction: E[Y] = P(detected) * E[Y | detected]

The gate reuses T1-successful classifiers (XGBoost by default) while the intensity
model trains on the ~3% of systems with actual detections, using log-transform
and strong regularization to handle the small sample size.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# Lazy model resolution to avoid top-level imports of optional dependencies.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "xgboost_classifier": ("aquacontam.models.xgboost", "XGBoostClassifier"),
    "xgboost_regressor": ("aquacontam.models.xgboost", "XGBoostRegressor"),
    "random_forest_classifier": (
        "aquacontam.models.random_forest",
        "RandomForestClassifier",
    ),
    "random_forest_regressor": (
        "aquacontam.models.random_forest",
        "RandomForestRegressor",
    ),
    "lightgbm_classifier": ("aquacontam.models.lightgbm", "LightGBMClassifier"),
    "lightgbm_regressor": ("aquacontam.models.lightgbm", "LightGBMRegressor"),
}


def _resolve_model(name: str, config: dict[str, Any] | None = None) -> BaseModel:
    """Instantiate a model by registry name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_MODEL_REGISTRY)}")
    import importlib

    module_path, cls_name = _MODEL_REGISTRY[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(config=config)  # type: ignore[no-any-return]


class HurdleRegressor(BaseModel):
    """Two-part hurdle model for zero-inflated concentration regression.

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``gate_model``: classifier name (default ``"xgboost_classifier"``)
        - ``gate_config``: config dict for gate classifier
        - ``intensity_model``: regressor name (default ``"xgboost_regressor"``)
        - ``intensity_config``: config dict for intensity regressor
        - ``log_transform``: apply log1p to intensity targets (default ``True``)
        - ``random_state``: seed (default ``42``)
    """

    @property
    def name(self) -> str:
        return "hurdle_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the hurdle model.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training target (concentrations; 0.0 for non-detects).
        **kwargs
            Optional: ``X_val``, ``y_val`` for early stopping.
        """
        cfg = dict(self.config)
        self._log_transform = cfg.get("log_transform", True)
        seed = cfg.get("random_state", 42)

        y_arr = np.asarray(y_train, dtype=np.float64).ravel()
        detected = (y_arr > 0).astype(int)

        self._feature_names: list[str] | None
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
            X_np = X_train.to_numpy().astype(np.float64)
        else:
            self._feature_names = None
            X_np = np.asarray(X_train, dtype=np.float64)

        # --- Part 1: Gate classifier (all data) ---
        gate_name = cfg.get("gate_model", "xgboost_classifier")
        gate_cfg = cfg.get("gate_config") or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "scale_pos_weight": "auto",
            "early_stopping_rounds": 50,
            "eval_metric": "aucpr",
            "random_state": seed,
        }
        self._gate = _resolve_model(gate_name, gate_cfg)

        gate_kwargs: dict[str, Any] = {}
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        if X_val is not None and y_val is not None:
            y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()
            gate_kwargs["X_val"] = X_val
            gate_kwargs["y_val"] = (y_val_arr > 0).astype(int)

        self._gate.fit(X_np if self._feature_names is None else X_train, detected, **gate_kwargs)

        n_detected = int(detected.sum())
        logger.info(
            "Hurdle gate trained on %d samples (%d detected, %.1f%%)",
            len(y_arr),
            n_detected,
            100.0 * n_detected / max(len(y_arr), 1),
        )

        # --- Part 2: Intensity regressor (detected only) ---
        det_mask = detected.astype(bool)
        if n_detected < 2:
            logger.warning(
                "Too few detected samples (%d) for intensity model; "
                "predictions will be gate probability * fallback mean",
                n_detected,
            )
            self._intensity = None
            self._fallback_mean = float(y_arr[det_mask].mean()) if n_detected > 0 else 0.0
            self._smearing_factor = 1.0
            self._model = self._gate._model  # satisfy _check_fitted
            return

        X_det = X_np[det_mask]
        y_det = y_arr[det_mask]

        y_det_transformed = np.log1p(y_det) if self._log_transform else y_det.copy()

        intensity_name = cfg.get("intensity_model", "xgboost_regressor")
        intensity_cfg = cfg.get("intensity_config") or {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "min_child_weight": 10,
            "random_state": seed,
        }
        self._intensity = _resolve_model(intensity_name, intensity_cfg)

        # No validation split for intensity (too few samples)
        if isinstance(X_train, pd.DataFrame):
            X_det_df = pd.DataFrame(X_det, columns=self._feature_names)
            self._intensity.fit(X_det_df, y_det_transformed)
        else:
            self._intensity.fit(X_det, y_det_transformed)

        # Duan (1983) smearing estimator for unbiased back-transformation
        if self._log_transform:
            resid = y_det_transformed - self._intensity.predict(
                pd.DataFrame(X_det, columns=self._feature_names) if self._feature_names else X_det
            )
            self._smearing_factor = float(np.mean(np.exp(resid)))
        else:
            self._smearing_factor = 1.0

        self._fallback_mean = 0.0
        self._model = self._gate._model  # satisfy _check_fitted

        logger.info(
            "Hurdle intensity trained on %d detected samples (smearing=%.3f)",
            n_detected,
            self._smearing_factor,
        )

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict concentrations: P(detected) * E[Y | detected]."""
        self._check_fitted()

        # Gate probabilities
        gate_proba = self._gate.predict_proba(X)
        if gate_proba.ndim == 2 and gate_proba.shape[1] >= 2:
            p_detected = gate_proba[:, 1]
        elif gate_proba.ndim == 2:
            # Single-class edge case (all zeros in training)
            p_detected = np.zeros(len(gate_proba))
        else:
            p_detected = gate_proba

        if self._intensity is None:
            return np.asarray(p_detected * self._fallback_mean)

        # Intensity predictions
        raw_intensity = self._intensity.predict(X)

        if self._log_transform:
            intensity = np.expm1(np.clip(raw_intensity, 0.0, 20.0)) * self._smearing_factor
        else:
            intensity = raw_intensity * self._smearing_factor

        intensity = np.clip(intensity, 0.0, None)
        result = p_detected * intensity
        return np.asarray(np.clip(result, 0.0, None))

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")

    def feature_importances(self) -> pd.Series:
        """Delegate to gate model's feature importances."""
        self._check_fitted()
        return self._gate.feature_importances()
