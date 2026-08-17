"""Model interpretability via SHAP values and permutation importance.

Provides unified interfaces for computing feature importance explanations
across all model families (tree, linear, deep learning).

Typical usage::

    from aquacontam.analysis.interpretability import (
        compute_shap_values,
        permutation_importance,
        summary_plot_data,
    )

    sv = compute_shap_values(model, X_test)
    plot_data = summary_plot_data(sv, X_test, top_n=15)
    perm_imp = permutation_importance(model, X_test, y_test)
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance as _sklearn_perm_importance
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


def _infer_explainer_method(inner_model: Any) -> str:
    """Infer the best SHAP explainer method for a given sklearn/xgboost model.

    Parameters
    ----------
    inner_model : Any
        The underlying fitted model object (e.g., ``XGBClassifier``,
        ``RandomForestClassifier``).

    Returns
    -------
    str
        One of ``"tree"``, ``"linear"``, or ``"kernel"``.
    """
    # XGBoost models
    if hasattr(inner_model, "get_booster"):
        return "tree"

    # Check class hierarchy by module and name for robustness
    model_type = type(inner_model).__name__
    model_module = type(inner_model).__module__ or ""

    # Tree-based sklearn models
    tree_names = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
    }
    if model_type in tree_names:
        return "tree"

    # LightGBM
    if "lightgbm" in model_module:
        return "tree"

    # Linear models
    linear_names = {
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
    }
    if model_type in linear_names:
        return "linear"

    return "kernel"


def _safe_tree_explainer(inner_model: Any) -> Any:
    """Create a SHAP TreeExplainer, working around xgboost >=3.0 / shap <0.50 bug.

    shap <0.50 fails to parse xgboost's bracketed ``base_score`` format
    (e.g. ``"[2.7E-1]"``). We use ``unittest.mock.patch`` to scope the
    workaround to the shap module only, avoiding global state mutation.
    """
    try:
        return shap.TreeExplainer(inner_model)
    except ValueError:
        from unittest.mock import patch

        _tree_mod = shap.explainers._tree
        _builtin_float = _tree_mod.float if hasattr(_tree_mod, "float") else float

        def _tolerant_float(x: Any) -> float:  # type: ignore[misc]
            if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                return _builtin_float(x.strip("[]"))
            return _builtin_float(x)

        with patch.object(_tree_mod, "float", _tolerant_float, create=True):
            return shap.TreeExplainer(inner_model)


def compute_shap_values(
    model: BaseModel,
    X: pd.DataFrame,
    method: str = "auto",
) -> pd.DataFrame:
    """Compute SHAP values for the given model and feature matrix.

    Parameters
    ----------
    model : BaseModel
        A fitted AquaContam model (must have ``_model`` attribute).
    X : pd.DataFrame
        Feature matrix to explain.
    method : str, optional
        Explainer strategy: ``"auto"`` (default), ``"tree"``, ``"linear"``,
        or ``"kernel"``. When ``"auto"``, the method is inferred from the
        underlying model type.

    Returns
    -------
    pd.DataFrame
        SHAP values with the same index and columns as *X*.

    Raises
    ------
    RuntimeError
        If the model has not been fitted.
    ValueError
        If *method* is not one of the accepted values.
    """
    model._check_fitted()

    # Warn if model is trivial (Dummy/constant) — SHAP values will be all zeros.
    _trivial_names = {"dummy_classifier", "dummy_regressor", "DummyClassifier", "DummyRegressor"}
    model_name = getattr(model, "name", "") or type(model._model).__name__
    if model_name in _trivial_names:
        logger.warning(
            "Computing SHAP values for trivial model '%s' — values will be uninformative",
            model_name,
        )

    if method not in ("auto", "tree", "linear", "kernel"):
        raise ValueError(
            f"Unknown SHAP method {method!r}. Expected 'auto', 'tree', 'linear', or 'kernel'."
        )

    inner = model._model

    if method == "auto":
        method = _infer_explainer_method(inner)
        logger.info("Auto-detected SHAP method: %s", method)

    if method == "tree":
        explainer = _safe_tree_explainer(inner)
    elif method == "linear":
        explainer = shap.LinearExplainer(inner, X)
    else:  # kernel
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(inner.predict_proba, background)

    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats:
    # - TreeExplainer for binary classifiers may return a list [class_0, class_1]
    # - Some explainers return a 3-D array (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # For binary classification, take the positive class (index 1)
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) — take positive class
        shap_values = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]

    return cast(pd.DataFrame, pd.DataFrame(shap_values, index=X.index, columns=X.columns))


def summary_plot_data(
    shap_values: pd.DataFrame,
    X: pd.DataFrame,
    top_n: int = 20,
) -> dict[str, Any]:
    """Prepare data for a SHAP summary / beeswarm plot.

    Parameters
    ----------
    shap_values : pd.DataFrame
        SHAP values from :func:`compute_shap_values`.
    X : pd.DataFrame
        Original feature values (same shape as *shap_values*).
    top_n : int, optional
        Number of top features to include (default 20).

    Returns
    -------
    dict
        Keys:

        - ``"feature_names"`` : list[str] — top feature names by mean |SHAP|.
        - ``"mean_abs_shap"`` : list[float] — mean absolute SHAP per feature.
        - ``"shap_values"`` : list[list[float]] — SHAP values per feature
          (for beeswarm plots).
        - ``"feature_values"`` : list[list[float]] — corresponding raw feature
          values per feature.
    """
    mean_abs = shap_values.abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(top_n).index.tolist()

    return {
        "feature_names": top_features,
        "mean_abs_shap": mean_abs.loc[top_features].tolist(),
        "shap_values": [shap_values[f].tolist() for f in top_features],
        "feature_values": [X[f].tolist() for f in top_features],
    }


def force_plot_data(
    shap_values: pd.DataFrame,
    X: pd.DataFrame,
    idx: int,
) -> dict[str, Any]:
    """Prepare data for a single-instance SHAP force plot.

    Parameters
    ----------
    shap_values : pd.DataFrame
        SHAP values from :func:`compute_shap_values`.
    X : pd.DataFrame
        Original feature values.
    idx : int
        Row index (positional) of the instance to explain.

    Returns
    -------
    dict
        Keys:

        - ``"base_value"`` : float — expected model output (mean of SHAP
          contributions plus the base value is approximated as the mean
          prediction offset).
        - ``"shap_values"`` : dict[str, float] — feature name to SHAP value.
        - ``"feature_values"`` : dict[str, float] — feature name to raw value.
    """
    sv_row = shap_values.iloc[idx]
    x_row = X.iloc[idx]

    # Approximate the base value as the negative sum of mean SHAP values
    # (true base_value comes from the explainer; here we reconstruct from the
    # DataFrame-only API as expected_value + shap_sum = prediction)
    base_value = float(-shap_values.mean().sum())

    return {
        "base_value": base_value,
        "shap_values": sv_row.to_dict(),
        "feature_values": x_row.to_dict(),
    }


class _SklearnAdapter(ClassifierMixin, BaseEstimator):
    """Adapter wrapping a :class:`BaseModel` to satisfy the sklearn estimator API.

    This is used internally by :func:`permutation_importance` so that
    ``sklearn.inspection.permutation_importance`` can call ``predict`` and
    ``predict_proba`` on AquaContam models.

    Inherits from :class:`~sklearn.base.BaseEstimator` and
    :class:`~sklearn.base.ClassifierMixin` to provide the ``__sklearn_tags__``
    interface required by modern sklearn (>=1.6).

    Parameters
    ----------
    model : BaseModel
        A fitted AquaContam model.
    """

    def __init__(self, model: BaseModel | None = None) -> None:
        self.model = model

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> _SklearnAdapter:
        """No-op fit (model is already fitted).

        Required by sklearn's ``permutation_importance`` parameter validation.
        """
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels or regression targets."""
        assert self.model is not None
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        assert self.model is not None
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        """Class labels known to the underlying model."""
        assert self.model is not None
        inner = self.model._model
        if hasattr(inner, "classes_"):
            return np.asarray(inner.classes_)
        return np.array([0, 1])

    def __sklearn_is_fitted__(self) -> bool:
        """Tell sklearn that this adapter wraps an already-fitted model."""
        return True


def _metric_to_scorer(metric: str) -> Any:
    """Convert a metric name string to a sklearn scorer.

    Parameters
    ----------
    metric : str
        One of ``"auprc"``, ``"auroc"``, or ``"accuracy"``.

    Returns
    -------
    str | sklearn scorer
        A scorer compatible with ``sklearn.inspection.permutation_importance``.

    Raises
    ------
    ValueError
        If *metric* is not recognized.
    """
    if metric == "auprc":
        return make_scorer(
            average_precision_score,
            needs_proba=True,
            response_method="predict_proba",
        )
    if metric == "auroc":
        return make_scorer(
            roc_auc_score,
            needs_proba=True,
            response_method="predict_proba",
        )
    if metric == "accuracy":
        return "accuracy"
    raise ValueError(f"Unknown metric {metric!r}. Expected 'auprc', 'auroc', or 'accuracy'.")


def permutation_importance(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    metric: str = "auprc",
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance for an AquaContam model.

    Parameters
    ----------
    model : BaseModel
        A fitted AquaContam model.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        True labels.
    metric : str, optional
        Scoring metric: ``"auprc"`` (default), ``"auroc"``, or
        ``"accuracy"``.
    n_repeats : int, optional
        Number of permutation repeats (default 10).
    seed : int, optional
        Random state for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        Columns ``"importance_mean"`` and ``"importance_std"``, indexed by
        feature name.
    """
    model._check_fitted()
    adapter = _SklearnAdapter(model)
    scorer = _metric_to_scorer(metric)

    result = _sklearn_perm_importance(
        adapter,
        X,
        y,
        n_repeats=n_repeats,
        scoring=scorer,
        random_state=seed,
    )

    return cast(
        pd.DataFrame,
        pd.DataFrame(
            {
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            },
            index=X.columns,
        ),
    )


def compute_shap_interactions(
    model: BaseModel,
    X: pd.DataFrame,
    *,
    top_n: int = 10,
    max_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute SHAP interaction values for top feature pairs.

    Uses SHAP TreeExplainer's ``shap_interaction_values`` for tree-based
    models. For non-tree models, falls back to pairwise main-effect products
    as a lightweight approximation.

    Parameters
    ----------
    model : BaseModel
        Fitted AquaContam model.
    X : pd.DataFrame
        Feature matrix (test set).
    top_n : int
        Number of top interaction pairs to return.
    max_samples : int
        Maximum samples to use (SHAP interactions are O(n * p^2)).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature_a``, ``feature_b``, ``mean_abs_interaction``.
        Sorted by interaction strength, top ``top_n`` pairs.
    """
    inner = model.model_ if hasattr(model, "model_") else model
    method = _infer_explainer_method(inner)

    # Subsample for performance
    if len(X) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sub = X.iloc[idx]
    else:
        X_sub = X

    if method == "tree":
        try:
            explainer = _safe_tree_explainer(inner)
            interaction_values = explainer.shap_interaction_values(X_sub)
            # interaction_values shape: (n_samples, n_features, n_features)
            # For binary classification, may return list of 2 arrays
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1]  # positive class

            # Mean absolute interaction across samples
            mean_abs = np.abs(interaction_values).mean(axis=0)
            features = list(X_sub.columns)
            n_feat = len(features)

            # Extract upper triangle (pairs, not diagonal)
            pairs: list[dict[str, object]] = []
            for i in range(n_feat):
                for j in range(i + 1, n_feat):
                    pairs.append(
                        {
                            "feature_a": features[i],
                            "feature_b": features[j],
                            "mean_abs_interaction": float(mean_abs[i, j]),
                        }
                    )

            df = pd.DataFrame(pairs)
            return cast(
                pd.DataFrame, df.nlargest(top_n, "mean_abs_interaction").reset_index(drop=True)
            )

        except (ValueError, RuntimeError, TypeError):
            logger.warning(
                "SHAP interaction values failed, falling back to main-effect products",
                exc_info=True,
            )

    # Fallback: approximate interactions via SHAP main-effect products
    sv = compute_shap_values(model, X_sub)
    if sv is None or sv.empty:
        return cast(
            pd.DataFrame, pd.DataFrame(columns=["feature_a", "feature_b", "mean_abs_interaction"])
        )

    # Mean |SHAP_i * SHAP_j| as interaction proxy
    sv_arr = sv.to_numpy()
    features = list(sv.columns)
    n_feat = len(features)
    pairs = []
    for i in range(min(n_feat, 50)):  # limit to top-50 features
        for j in range(i + 1, min(n_feat, 50)):
            interaction = float(np.abs(sv_arr[:, i] * sv_arr[:, j]).mean())
            pairs.append(
                {
                    "feature_a": features[i],
                    "feature_b": features[j],
                    "mean_abs_interaction": interaction,
                }
            )

    df = pd.DataFrame(pairs)
    if df.empty:
        return cast(pd.DataFrame, df)
    return cast(pd.DataFrame, df.nlargest(top_n, "mean_abs_interaction").reset_index(drop=True))
