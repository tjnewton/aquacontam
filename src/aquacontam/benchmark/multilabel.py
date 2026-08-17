"""T3: Multi-PFAS profile prediction (multilabel classification).

Predicts the detection profile across multiple PFAS analytes simultaneously.
Unlike T1 (single-analyte binary), T3 outputs a binary vector indicating
which of *k* PFAS are detected at each water system.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from aquacontam._constants import (
    MULTILABEL_METRICS,
    T3_DEFAULT_ANALYTES,
)
from aquacontam.benchmark._utils import _safe_mode
from aquacontam.benchmark.metrics import compute_multilabel_metrics
from aquacontam.benchmark.registry import TaskResult, register_task
from aquacontam.features.assembly import assemble_feature_matrix
from aquacontam.models.base import BaseModel
from aquacontam.preprocessing.splits import assign_epa_region, geographic_split

logger = logging.getLogger(__name__)


def _supports_eval_set(estimator: Any) -> bool:
    """Check whether *estimator*.fit() accepts an ``eval_set`` parameter."""
    try:
        sig = inspect.signature(estimator.fit)
        return "eval_set" in sig.parameters
    except (ValueError, TypeError):
        return False


class EvalSetMultiOutputClassifier:
    """Multi-output classifier that forwards ``eval_set`` to sub-estimators.

    Unlike sklearn's ``MultiOutputClassifier``, this wrapper splits Y_val
    column-wise and passes per-label ``eval_set`` to estimators that support
    it (XGBoost, LightGBM, CatBoost), enabling early stopping for T3.

    Parameters
    ----------
    estimator : sklearn estimator
        The base estimator to clone for each output.
    n_jobs : int
        Not used (kept for API compatibility). Fitting is sequential.
    early_stopping_rounds : int or None
        Patience for early stopping. Used to build LightGBM callbacks.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        n_jobs: int = 1,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.estimators_: list[Any] = []

    @staticmethod
    def _eval_fit_kwargs(
        estimator: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int | None,
    ) -> dict[str, Any]:
        """Build fit kwargs for estimators that accept eval_set."""
        kwargs: dict[str, Any] = {"eval_set": [(X_val, y_val)]}
        # LightGBM uses callbacks for early stopping (not a constructor param)
        est_type = type(estimator).__name__
        if "LGBM" in est_type and early_stopping_rounds is not None:
            try:
                import lightgbm as lgb

                kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds)]
            except ImportError:
                pass
        return kwargs

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> EvalSetMultiOutputClassifier:
        """Fit one cloned estimator per output column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_outputs)
        eval_set : tuple (X_val, Y_val) or None
            Validation data. Y_val is split column-wise for each sub-estimator.
        """
        n_outputs = Y.shape[1]
        self.estimators_ = []
        for i in range(n_outputs):
            est = clone(self.estimator)
            y_i = Y[:, i]
            if eval_set is not None and _supports_eval_set(est):
                X_val, Y_val = eval_set
                y_val_i = Y_val[:, i]
                kwargs = self._eval_fit_kwargs(est, X_val, y_val_i, self.early_stopping_rounds)
                est.fit(X, y_i, **kwargs)
            else:
                est.fit(X, y_i)
            self.estimators_.append(est)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each output."""
        return np.column_stack([est.predict(X) for est in self.estimators_])

    def predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        """Predict class probabilities for each output.

        Returns a list of arrays (one per output), each of shape
        ``(n_samples, n_classes)``.
        """
        return [est.predict_proba(X) for est in self.estimators_]


def aggregate_to_multilabel_target(
    df: pd.DataFrame,
    analytes: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Aggregate long-form samples to a multilabel target matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame with ``pwsid``, ``analyte``, ``censored``.
    analytes : tuple or list of str
        Analytes to include as label columns.

    Returns
    -------
    pd.DataFrame
        Indexed by ``pwsid`` with columns ``detected_{analyte}`` (int 0/1)
        for each analyte, plus ``n_samples`` and ``detection_rate_{analyte}``.
    """
    required = {"pwsid", "analyte", "censored"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(analytes) == 0:
        raise ValueError("analytes must be non-empty")

    result_rows: dict[str, dict[str, Any]] = {}

    for analyte in analytes:
        filtered = df[df["analyte"] == analyte]
        grouped = filtered.groupby("pwsid", observed=True)

        for pwsid, group in grouped:
            if pwsid not in result_rows:
                result_rows[pwsid] = {"n_samples": 0}
            result_rows[pwsid]["n_samples"] += len(group)
            detected = bool((~group["censored"]).any())
            result_rows[pwsid][f"detected_{analyte}"] = int(detected)
            rate = float((~group["censored"]).mean())
            result_rows[pwsid][f"detection_rate_{analyte}"] = rate

    if not result_rows:
        cols = ["n_samples"]
        for a in analytes:
            cols.extend([f"detected_{a}", f"detection_rate_{a}"])
        empty: pd.DataFrame = pd.DataFrame(columns=cols)
        empty.index.name = "pwsid"
        return empty

    result_df: pd.DataFrame = pd.DataFrame.from_dict(result_rows, orient="index")
    result_df.index.name = "pwsid"

    # Fill missing analytes with 0 (system has no samples for that analyte)
    for analyte in analytes:
        det_col = f"detected_{analyte}"
        rate_col = f"detection_rate_{analyte}"
        if det_col not in result_df.columns:
            result_df[det_col] = 0
        if rate_col not in result_df.columns:
            result_df[rate_col] = 0.0
        result_df[det_col] = result_df[det_col].fillna(0).astype(int)
        result_df[rate_col] = result_df[rate_col].fillna(0.0)

    return result_df


def prepare_multilabel_train_val_test(
    df: pd.DataFrame,
    analytes: tuple[str, ...] | list[str],
    *feature_dfs: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    drop_na_threshold: float = 0.5,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare train/val/test splits for multilabel classification.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame with standard schema.
    analytes : tuple or list of str
        Analytes for label columns.
    *feature_dfs : pd.DataFrame
        Feature DataFrames indexed by pwsid.
    categorical_columns : list[str] or None
        Columns to one-hot encode.
    drop_na_threshold : float
        Threshold for dropping high-NaN columns.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Keys ``"train"``, ``"val"``, ``"test"`` mapping to
        ``(X, Y)`` tuples where Y has shape (n_systems, n_analytes).
    """
    # Assign EPA region for splitting
    df_with_region = assign_epa_region(df)

    # Aggregate to multilabel target
    multi_agg = aggregate_to_multilabel_target(df_with_region, analytes)

    # Build target columns
    label_cols = [f"detected_{a}" for a in analytes]

    # Add EPA region info
    system_regions = df_with_region.groupby("pwsid", observed=True)["epa_region"].agg(_safe_mode)
    multi_agg = multi_agg.join(system_regions)

    # Need a dummy "target" column for geographic_split compatibility
    multi_agg["target"] = 0

    # Geographic split
    system_reset = multi_agg.reset_index()
    train_sys, val_sys, test_sys = geographic_split(system_reset)

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    impute_stats = None

    for split_name, split_df in [("train", train_sys), ("val", val_sys), ("test", test_sys)]:
        if split_df.empty:
            splits[split_name] = (pd.DataFrame(), pd.DataFrame(columns=label_cols))
            continue

        split_indexed = split_df.set_index("pwsid")

        # Extract Y (multilabel matrix)
        Y = split_indexed[label_cols].copy()
        # Ensure integer type
        Y = Y.fillna(0).astype(int)

        # Drop columns not needed for features (keep n_samples as a feature)
        drop_cols = label_cols + [
            c
            for c in split_indexed.columns
            if c.startswith("detection_rate_") or c == "epa_region"
        ]
        feature_targets = split_indexed.drop(
            columns=[c for c in drop_cols if c in split_indexed.columns]
        )

        X, _y_dummy, stats = assemble_feature_matrix(
            feature_targets,
            *feature_dfs,
            categorical_columns=categorical_columns,
            drop_na_threshold=drop_na_threshold,
            impute_stats=impute_stats,
        )

        if split_name == "train":
            impute_stats = stats

        splits[split_name] = (X, Y)

    return splits


def _build_inner_estimator(model: BaseModel) -> Any:
    """Extract an unfitted sklearn estimator from a BaseModel config.

    Parameters
    ----------
    model : BaseModel
        Model instance whose config is used to create the estimator.

    Returns
    -------
    sklearn estimator
        Unfitted estimator suitable for ``MultiOutputClassifier``.
    """
    name = model.name

    # Dummy baseline — must check before other patterns
    if "dummy" in name:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="stratified", random_state=42)

    if "xgboost" in name or "xgb" in name:
        from xgboost import XGBClassifier

        config = dict(model.config)
        config.pop("scale_pos_weight", None)
        config.pop("eval_metric", None)
        config.setdefault("random_state", 42)
        return XGBClassifier(**config)

    if "lightgbm" in name or "lgbm" in name:
        from lightgbm import LGBMClassifier

        config = dict(model.config)
        config.pop("scale_pos_weight", None)
        # early_stopping_rounds is not a valid LGBMClassifier param;
        # EvalSetMultiOutputClassifier uses it via callbacks instead.
        config.pop("early_stopping_rounds", None)
        config.pop("callbacks", None)
        config.setdefault("random_state", 42)
        config.setdefault("verbose", -1)
        return LGBMClassifier(**config)

    if "catboost" in name:
        from catboost import CatBoostClassifier as CatCls

        config = dict(model.config)
        config.pop("scale_pos_weight", None)
        config.pop("auto_class_weights", None)
        config.pop("callbacks", None)
        seed = config.pop("random_state", 42)  # Remove sklearn alias, keep value
        config.setdefault("random_seed", seed)  # Use CatBoost's native param
        config.setdefault("verbose", 0)
        return CatCls(**config)

    if "random_forest" in name or "rf" in name:
        from sklearn.ensemble import RandomForestClassifier as SkRF

        config = dict(model.config)
        config.setdefault("n_estimators", 100)
        config.setdefault("random_state", 42)
        return SkRF(**config)

    if "logistic" in name:
        from sklearn.linear_model import LogisticRegression

        config = dict(model.config)
        config.setdefault("max_iter", 1000)
        config.setdefault("random_state", 42)
        # Filter to valid LogisticRegression params
        valid_params = {"C", "penalty", "solver", "max_iter", "random_state", "class_weight"}
        lr_config = {k: v for k, v in config.items() if k in valid_params}
        return LogisticRegression(**lr_config)

    if "mlp" in name:
        from sklearn.neural_network import MLPClassifier as SkMLP

        logger.info(
            "Using sklearn MLPClassifier proxy for %s (PyTorch models "
            "cannot plug into sklearn MultiOutputClassifier)",
            name,
        )
        return SkMLP(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )

    if "cnn" in name or "conv" in name:
        from sklearn.neural_network import MLPClassifier as SkMLP

        logger.info(
            "Using sklearn MLPClassifier proxy (wider architecture) for %s "
            "(PyTorch models cannot plug into sklearn MultiOutputClassifier)",
            name,
        )
        return SkMLP(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )

    if "gnn" in name or "graph" in name:
        from sklearn.ensemble import GradientBoostingClassifier

        logger.warning(
            "GNN model %s cannot be used with sklearn MultiOutputClassifier. "
            "Using GradientBoostingClassifier as proxy for T3 multilabel task.",
            name,
        )
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )

    # Fallback: logistic regression (with warning)
    from sklearn.linear_model import LogisticRegression

    logger.warning(
        "No explicit T3 handler for model %r — falling back to "
        "LogisticRegression. Results may not reflect the model's true "
        "capabilities.",
        name,
    )
    return LogisticRegression(max_iter=1000, random_state=42)


@register_task(
    name="T3",
    description="Multi-PFAS profile prediction (multilabel)",
    task_type="multilabel_classification",
    primary_metric="macro_auprc",
    analytes=T3_DEFAULT_ANALYTES,
)
def run_t3(
    *,
    model: BaseModel,
    data: pd.DataFrame,
    feature_dfs: list[pd.DataFrame] | None = None,
    analytes: tuple[str, ...] | list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> TaskResult:
    """Run T3: multi-PFAS profile prediction (multilabel classification).

    Parameters
    ----------
    model : BaseModel
        Model whose config is used to build a per-label estimator.
    data : pd.DataFrame
        Water quality DataFrame with standard schema.
    feature_dfs : list[pd.DataFrame], optional
        Feature DataFrames indexed by pwsid.
    analytes : tuple or list of str, optional
        PFAS analytes for the multilabel target. Defaults to
        ``T3_DEFAULT_ANALYTES``.
    categorical_columns : list[str], optional
        Categorical columns to one-hot encode.

    Returns
    -------
    TaskResult
    """
    if feature_dfs is None:
        feature_dfs = []
    if analytes is None:
        analytes = T3_DEFAULT_ANALYTES

    label_cols = [f"detected_{a}" for a in analytes]

    splits = prepare_multilabel_train_val_test(
        data,
        analytes,
        *feature_dfs,
        categorical_columns=categorical_columns,
    )

    X_train, Y_train = splits["train"]

    if X_train.empty:
        logger.warning("Empty training set for T3")
        return TaskResult(
            task_name="T3",
            model_name=model.name,
            metrics={},
            metadata={"analytes": list(analytes), "error": "empty training set"},
        )

    # Build multi-output classifier with eval_set support for early stopping
    inner = _build_inner_estimator(model)
    early_stopping_rounds = model.config.get("early_stopping_rounds")
    multi_clf = EvalSetMultiOutputClassifier(inner, early_stopping_rounds=early_stopping_rounds)

    X_val, Y_val = splits.get("val", (pd.DataFrame(), pd.DataFrame()))
    eval_set_arg = None
    if not X_val.empty:
        eval_set_arg = (X_val.to_numpy(), Y_val.to_numpy())

    multi_clf.fit(X_train.to_numpy(), Y_train.to_numpy(), eval_set=eval_set_arg)

    # Evaluate per split
    split_metrics: dict[str, dict[str, float]] = {}
    for split_name, (X, Y) in splits.items():
        if X.empty:
            continue

        X_np = X.to_numpy()
        Y_np = Y.to_numpy()
        Y_pred = multi_clf.predict(X_np)

        # Get probabilities
        try:
            Y_prob_list = multi_clf.predict_proba(X_np)
            Y_prob = np.column_stack([p[:, 1] for p in Y_prob_list])
        except (AttributeError, IndexError):
            Y_prob = None

        split_metrics[split_name] = compute_multilabel_metrics(
            Y_np,
            Y_pred,
            Y_prob,
            label_names=list(analytes),
            metrics=list(MULTILABEL_METRICS),
        )

    primary = split_metrics.get("test", split_metrics.get("val", {}))

    # Note: T3 does not store spatial metadata (y_true, y_prob, latitudes, etc.)
    # because multi-output predictions are incompatible with the single-output
    # spatial autocorrelation and calibration analyses in reproduce.py.
    return TaskResult(
        task_name="T3",
        model_name=model.name,
        metrics=primary,
        split_metrics=split_metrics,
        metadata={
            "analytes": list(analytes),
            "n_train": len(X_train),
            "n_labels": len(analytes),
            "label_columns": label_cols,
        },
    )
