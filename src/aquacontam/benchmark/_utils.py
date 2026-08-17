"""Shared helpers for benchmark task implementations.

Consolidates common patterns used across T1-T7 task runners to avoid
code duplication while keeping imports lightweight.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel


def _safe_mode(x: pd.Series) -> Any:
    """Return the mode of a Series, or NaN if empty."""
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else np.nan


def get_proba(model: BaseModel, X: pd.DataFrame) -> np.ndarray | None:
    """Extract positive-class probabilities, or None if unsupported."""
    try:
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs
    except NotImplementedError:
        return None


def get_system_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Compute median latitude/longitude per water system.

    Parameters
    ----------
    data : pd.DataFrame
        Water quality DataFrame with ``pwsid``, ``latitude``, ``longitude``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``pwsid`` with ``latitude`` and ``longitude`` columns.
    """
    if data.empty or "latitude" not in data.columns or "longitude" not in data.columns:
        return cast(pd.DataFrame, pd.DataFrame(columns=["latitude", "longitude"]))
    coords: pd.DataFrame = data.groupby("pwsid", observed=True)[["latitude", "longitude"]].median()
    result: pd.DataFrame = coords.dropna()
    return result


def accumulate_spatial_metadata(
    y: pd.Series,
    preds_or_probs: np.ndarray | None,
    index: pd.Index,
    sys_coords: pd.DataFrame,
    split_name: str,
    accumulators: dict[str, list],
) -> None:
    """Align predictions to systems with valid coordinates and extend accumulators.

    Parameters
    ----------
    y : pd.Series
        True labels, indexed by pwsid.
    preds_or_probs : np.ndarray | None
        Predictions or probabilities (same length as ``y``).
    index : pd.Index
        Index of the feature matrix (pwsids).
    sys_coords : pd.DataFrame
        System coordinates from ``get_system_coordinates()``.
    split_name : str
        Name of the split (e.g. ``"train"``, ``"val"``, ``"test"``).
    accumulators : dict[str, list]
        Dict with keys ``y_true``, ``y_prob``, ``latitudes``, ``longitudes``,
        ``split_labels``. Lists are extended in-place.
    """
    if preds_or_probs is None:
        return
    # Find systems that have valid coordinates
    valid = index.isin(sys_coords.index)
    if not valid.any():
        return
    valid_pwsids = index[valid]
    coords = sys_coords.loc[valid_pwsids]
    accumulators["y_true"].extend(y.loc[valid_pwsids].tolist())
    accumulators["y_prob"].extend(preds_or_probs[valid].tolist())
    accumulators["latitudes"].extend(coords["latitude"].tolist())
    accumulators["longitudes"].extend(coords["longitude"].tolist())
    accumulators["split_labels"].extend([split_name] * int(valid.sum()))


def fit_with_val(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    **extra_kwargs: Any,
) -> None:
    """Fit model, passing validation data only if val set is non-empty.

    Extra keyword arguments (e.g. ``coords`` for GNN models) are forwarded
    to ``model.fit()`` unchanged.
    """
    fit_kwargs: dict[str, Any] = dict(extra_kwargs)
    if not X_val.empty:
        fit_kwargs["X_val"] = X_val
        fit_kwargs["y_val"] = y_val
    model.fit(X_train, y_train, **fit_kwargs)
