"""XGBoost AFT (Accelerated Failure Time) survival regressor for censored data.

Uses XGBoost's native ``survival:aft`` objective to properly model interval-censored
observations. Non-detects are encoded as intervals [0, DL] rather than point values 0.0,
providing statistically correct treatment of left-censored environmental monitoring data.

Requires the native ``xgb.train()`` API (not the sklearn wrapper) because DMatrix
``label_lower_bound`` / ``label_upper_bound`` columns are needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# Small epsilon to avoid log1p(0) = 0 lower bound (AFT requires lower < upper for intervals)
_EPS: float = 1e-6


class XGBoostAFTRegressor(BaseModel):
    """XGBoost AFT survival regressor for left-censored concentration data.

    Parameters
    ----------
    config : dict, optional
        Hyperparameters:
        - ``n_estimators``: number of boosting rounds (default ``500``)
        - ``max_depth``: max tree depth (default ``6``)
        - ``learning_rate``: step size shrinkage (default ``0.05``)
        - ``subsample``: row subsampling ratio (default ``0.8``)
        - ``colsample_bytree``: column subsampling ratio (default ``0.8``)
        - ``min_child_weight``: min sum of instance weight in child (default ``5``)
        - ``aft_loss_distribution``: ``"normal"``, ``"logistic"``, or ``"extreme"``
          (default ``"normal"`` — log-normal in original scale)
        - ``aft_loss_distribution_scale``: scale parameter (default ``1.0``)
        - ``early_stopping_rounds``: patience for early stopping (default ``50``)
        - ``random_state``: seed (default ``42``)
    """

    requires_censoring_metadata: bool = True

    @property
    def name(self) -> str:
        return "xgboost_aft_regressor"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the XGBoost AFT regressor.

        Parameters
        ----------
        X_train : pd.DataFrame | np.ndarray
            Training features.
        y_train : pd.Series | np.ndarray
            Training target values (concentrations).
        **kwargs
            Optional: ``X_val``, ``y_val``, ``censored``, ``detection_limits``,
            ``censored_val``, ``detection_limits_val``.
        """
        cfg = dict(self.config)
        n_rounds = cfg.pop("n_estimators", 500)
        early_stopping_rounds = cfg.pop("early_stopping_rounds", 50)
        seed = cfg.pop("random_state", 42)

        y_arr = np.asarray(y_train, dtype=np.float64).ravel()
        n = len(y_arr)

        # Feature names for importance
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
            X_np = X_train.to_numpy().astype(np.float64)
        else:
            self._feature_names = [f"f{i}" for i in range(X_train.shape[1])]
            X_np = np.asarray(X_train, dtype=np.float64)

        # Store training max for output clamping
        self._y_train_max = float(np.nanmax(y_arr))

        # Prepare censoring arrays
        censored = kwargs.get("censored")
        detection_limits = kwargs.get("detection_limits")
        if censored is not None:
            cens = np.asarray(censored, dtype=np.float64).ravel()
        else:
            cens = np.zeros(n, dtype=np.float64)
        if detection_limits is not None:
            dl = np.asarray(detection_limits, dtype=np.float64).ravel()
        else:
            dl = np.where(cens > 0.5, y_arr, 0.0).astype(np.float64)

        # Encode as interval-censored in log1p space
        lower, upper = self._encode_intervals(y_arr, cens, dl)

        dtrain = xgb.DMatrix(X_np, feature_names=self._feature_names)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)

        # XGBoost params
        params: dict[str, Any] = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": cfg.pop("aft_loss_distribution", "normal"),
            "aft_loss_distribution_scale": cfg.pop("aft_loss_distribution_scale", 1.0),
            "max_depth": cfg.pop("max_depth", 6),
            "learning_rate": cfg.pop("learning_rate", 0.05),
            "subsample": cfg.pop("subsample", 0.8),
            "colsample_bytree": cfg.pop("colsample_bytree", 0.8),
            "min_child_weight": cfg.pop("min_child_weight", 5),
            "reg_alpha": cfg.pop("reg_alpha", 0.0),
            "reg_lambda": cfg.pop("reg_lambda", 1.0),
            "seed": seed,
            "verbosity": 0,
        }

        # GPU passthrough — survival:aft GPU support is partial in xgb 3.x;
        # try GPU first, fall back to CPU if xgb.train raises.
        use_gpu = cfg.pop("use_gpu", False)
        gpu_id = cfg.pop("gpu_id", 0)
        if use_gpu:
            params["device"] = f"cuda:{gpu_id}"
            params["tree_method"] = "hist"

        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]

        # Validation set
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_val_arr = np.asarray(y_val, dtype=np.float64).ravel()
            cens_val = kwargs.get("censored_val")
            dl_val = kwargs.get("detection_limits_val")
            if cens_val is not None:
                cens_val = np.asarray(cens_val, dtype=np.float64).ravel()
            else:
                cens_val = np.zeros(len(y_val_arr), dtype=np.float64)
            if dl_val is not None:
                dl_val = np.asarray(dl_val, dtype=np.float64).ravel()
            else:
                dl_val = np.where(cens_val > 0.5, y_val_arr, 0.0).astype(np.float64)

            lower_val, upper_val = self._encode_intervals(y_val_arr, cens_val, dl_val)

            if isinstance(X_val, pd.DataFrame):
                X_val_np = X_val.to_numpy().astype(np.float64)
            else:
                X_val_np = np.asarray(X_val, dtype=np.float64)

            dval = xgb.DMatrix(X_val_np, feature_names=self._feature_names)
            dval.set_float_info("label_lower_bound", lower_val)
            dval.set_float_info("label_upper_bound", upper_val)
            evals.append((dval, "val"))

        try:
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_rounds,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError as exc:
            if "device" in params and "cuda" in str(params.get("device", "")):
                logger.warning("XGBoost AFT GPU training failed (%s); retrying on CPU", exc)
                params.pop("device", None)
                params.pop("tree_method", None)
                self._booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=n_rounds,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
                    verbose_eval=False,
                )
            else:
                raise
        self._model = self._booster  # satisfy _check_fitted

    @staticmethod
    def _encode_intervals(
        y: np.ndarray, censored: np.ndarray, detection_limits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode observations as interval-censored bounds in log1p space.

        Parameters
        ----------
        y : np.ndarray
            Observed concentrations.
        censored : np.ndarray
            Binary censoring indicators (1 = censored).
        detection_limits : np.ndarray
            Detection limits for each observation.

        Returns
        -------
        lower, upper : tuple[np.ndarray, np.ndarray]
            Lower and upper bounds in log1p space. For detected observations,
            lower == upper (point observation). For censored observations,
            lower = epsilon, upper = log1p(DL).
        """
        is_censored = censored > 0.5

        # Detected: point observation [log1p(y), log1p(y)]
        lower = np.log1p(np.clip(y, 0.0, None))
        upper = lower.copy()

        # Censored: interval [eps, log1p(DL)]
        lower[is_censored] = _EPS
        upper[is_censored] = np.log1p(np.clip(detection_limits[is_censored], _EPS, None))

        # Ensure lower < upper for censored (AFT requirement)
        too_small = is_censored & (upper <= lower + _EPS)
        upper[too_small] = lower[too_small] + 2 * _EPS

        return lower, upper

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict concentration values.

        Returns back-transformed predictions: expm1(booster.predict(X)),
        clamped to [0, 2 * y_train_max].
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy().astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        dtest = xgb.DMatrix(X_np, feature_names=self._feature_names)
        log_preds = self._booster.predict(dtest)

        # Back-transform from log1p space
        max_log = np.log1p(self._y_train_max * 2.0) if self._y_train_max > 0 else 10.0
        log_preds = np.clip(log_preds, 0.0, max_log)
        preds = np.expm1(log_preds)

        # Clamp to valid range
        y_cap = self._y_train_max * 2.0 if self._y_train_max > 0 else np.inf
        preds = np.clip(preds, 0.0, y_cap)

        return np.asarray(preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not supported for regression."""
        raise NotImplementedError("Regression models do not support predict_proba")

    def feature_importances(self) -> pd.Series:
        """Get feature importance scores from the booster."""
        self._check_fitted()
        scores = self._booster.get_score(importance_type="gain")
        # Fill missing features with 0
        importances = {f: scores.get(f, 0.0) for f in self._feature_names}
        return pd.Series(importances, name="importance").sort_values(ascending=False)
