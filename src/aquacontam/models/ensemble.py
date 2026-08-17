"""Ensemble models (Voting + Stacking classifiers, Averaging regressor)."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _SKRFClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _build_base_estimators(
    base_models: list[str],
    seed: int = 42,
) -> list[tuple[str, Any]]:
    """Build sklearn-native base estimators for ensemble models.

    Parameters
    ----------
    base_models : list[str]
        Model names to include (e.g. ``["xgboost", "random_forest"]``).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[str, estimator]]
        Named estimators for sklearn ensemble constructors.
    """
    estimators: list[tuple[str, Any]] = []

    for name in base_models:
        if name == "random_forest":
            estimators.append(
                (
                    "random_forest",
                    _SKRFClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                )
            )
        elif name == "xgboost":
            try:
                from xgboost import XGBClassifier

                estimators.append(
                    (
                        "xgboost",
                        XGBClassifier(
                            n_estimators=500,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            eval_metric="aucpr",
                            random_state=seed,
                            verbosity=0,
                        ),
                    )
                )
            except ImportError:
                logger.info("XGBoost not available — skipping in ensemble")
        elif name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                estimators.append(
                    (
                        "lightgbm",
                        LGBMClassifier(
                            n_estimators=500,
                            num_leaves=63,
                            learning_rate=0.05,
                            is_unbalance=True,
                            verbose=-1,
                            random_state=seed,
                        ),
                    )
                )
            except ImportError:
                logger.info("LightGBM not available — skipping in ensemble")
        elif name == "catboost":
            try:
                from catboost import CatBoostClassifier as _CatBoostCls

                estimators.append(
                    (
                        "catboost",
                        _CatBoostCls(
                            iterations=500,
                            depth=6,
                            learning_rate=0.05,
                            auto_class_weights="Balanced",
                            verbose=0,
                            random_seed=seed,
                        ),
                    )
                )
            except ImportError:
                logger.info("CatBoost not available — skipping in ensemble")
        else:
            logger.warning("Unknown base model for ensemble: %s", name)

    return estimators


class VotingEnsembleClassifier(BaseModel):
    """Soft-voting ensemble over tree-based classifiers.

    Parameters
    ----------
    config : dict, optional
        Configuration with keys:

        - ``base_models`` : list[str] — names of base learners
          (default: ``["xgboost", "random_forest"]``)
        - ``voting`` : str — ``"soft"`` or ``"hard"`` (default: ``"soft"``)
        - ``random_state`` : int — random seed (default: 42)
    """

    @property
    def name(self) -> str:
        return "voting_ensemble"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the voting ensemble."""
        base_models = self.config.get("base_models", ["xgboost", "random_forest"])
        voting = self.config.get("voting", "soft")
        seed = self.config.get("random_state", 42)

        estimators = _build_base_estimators(base_models, seed=seed)
        if len(estimators) < 2:
            raise RuntimeError(
                f"VotingEnsemble requires >= 2 base models, got {len(estimators)}. "
                f"Requested: {base_models}"
            )

        self._model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        logger.info(
            "Trained voting ensemble with %d base models: %s",
            len(estimators),
            [e[0] for e in estimators],
        )

    def feature_importances(self) -> pd.Series:
        """Average feature importances across base estimators.

        Returns
        -------
        pd.Series
            Feature names -> importance values, sorted descending.
        """
        self._check_fitted()
        importances_list: list[np.ndarray] = []
        for est in self._model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances_list.append(est.feature_importances_)

        if not importances_list:
            raise NotImplementedError("No base estimators expose feature_importances_")

        avg_importances = np.mean(importances_list, axis=0)

        names: list[str]
        if hasattr(self._model.estimators_[0], "feature_names_in_"):
            names = list(self._model.estimators_[0].feature_names_in_)
        else:
            names = [f"f{i}" for i in range(len(avg_importances))]

        return cast(
            pd.Series,
            pd.Series(avg_importances, index=names, name="importance").sort_values(
                ascending=False
            ),
        )


class StackingEnsembleClassifier(BaseModel):
    """Stacking ensemble with logistic regression meta-learner.

    Parameters
    ----------
    config : dict, optional
        Configuration with keys:

        - ``base_models`` : list[str] — names of base learners
          (default: ``["xgboost", "random_forest"]``)
        - ``final_estimator`` : str — ``"logistic"`` (default)
        - ``cv`` : int — cross-validation folds for stacking (default: 5)
        - ``passthrough`` : bool — pass original features to meta-learner
          (default: False)
        - ``random_state`` : int — random seed (default: 42)
    """

    @property
    def name(self) -> str:
        return "stacking_ensemble"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train the stacking ensemble."""
        base_models = self.config.get("base_models", ["xgboost", "random_forest"])
        cv = self.config.get("cv", 5)
        passthrough = self.config.get("passthrough", False)
        seed = self.config.get("random_state", 42)

        estimators = _build_base_estimators(base_models, seed=seed)
        if len(estimators) < 2:
            raise RuntimeError(
                f"StackingEnsemble requires >= 2 base models, got {len(estimators)}. "
                f"Requested: {base_models}"
            )

        final_estimator = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )

        self._model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict_proba",
            passthrough=passthrough,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        logger.info(
            "Trained stacking ensemble with %d base models (cv=%d): %s",
            len(estimators),
            cv,
            [e[0] for e in estimators],
        )

    def feature_importances(self) -> pd.Series:
        """Average feature importances across base estimators.

        Returns
        -------
        pd.Series
            Feature names -> importance values, sorted descending.
        """
        self._check_fitted()
        importances_list: list[np.ndarray] = []
        for _name, est in self._model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                importances_list.append(est.feature_importances_)

        if not importances_list:
            raise NotImplementedError("No base estimators expose feature_importances_")

        avg_importances = np.mean(importances_list, axis=0)

        names: list[str]
        first_est = next(iter(self._model.named_estimators_.values()))
        if hasattr(first_est, "feature_names_in_"):
            names = list(first_est.feature_names_in_)
        else:
            names = [f"f{i}" for i in range(len(avg_importances))]

        return cast(
            pd.Series,
            pd.Series(avg_importances, index=names, name="importance").sort_values(
                ascending=False
            ),
        )


class AveragingEnsembleRegressor(BaseModel):
    """Weighted-average ensemble of regression models.

    Parameters
    ----------
    config : dict, optional
        Configuration with keys:

        - ``models`` : list[BaseModel] — pre-instantiated sub-models
        - ``weights`` : list[float] | None — averaging weights (default: equal)
    """

    requires_censoring_metadata: bool = True

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        models: list[BaseModel] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__(config)
        self._sub_models = models or []
        self._weights = weights

    @property
    def name(self) -> str:
        return "hurdle_aft_ensemble"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Train all sub-models sequentially."""
        if not self._sub_models:
            raise RuntimeError("AveragingEnsembleRegressor requires at least 1 sub-model")

        for model in self._sub_models:
            model.fit(X_train, y_train, **kwargs)

        self._model = True  # mark as fitted
        logger.info(
            "Trained averaging ensemble with %d sub-models: %s",
            len(self._sub_models),
            [m.name for m in self._sub_models],
        )

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict as weighted average of sub-model predictions."""
        self._check_fitted()
        preds = [m.predict(X, **kwargs) for m in self._sub_models]
        weights = self._weights or [1.0 / len(preds)] * len(preds)
        result = np.zeros_like(preds[0], dtype=np.float64)
        for w, p in zip(weights, preds, strict=True):
            result += w * np.asarray(p, dtype=np.float64)
        return result

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        """Not applicable for regression — raises NotImplementedError."""
        raise NotImplementedError("predict_proba is not supported for regression ensembles")

    def feature_importances(self) -> pd.Series:
        """Average feature importances across sub-models."""
        self._check_fitted()
        imp_list: list[pd.Series] = []
        for model in self._sub_models:
            with contextlib.suppress(NotImplementedError, AttributeError):
                imp_list.append(model.feature_importances())

        if not imp_list:
            raise NotImplementedError("No sub-models expose feature_importances")

        combined = pd.concat(imp_list, axis=1).fillna(0.0)
        avg = combined.mean(axis=1)
        result: pd.Series = avg.sort_values(ascending=False).rename("importance")
        return result
