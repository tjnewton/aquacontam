"""Split conformal prediction for binary classification.

Provides distribution-free prediction sets with finite-sample coverage
guarantees under the exchangeability assumption. When calibration and test data
come from geographically disjoint regions (as in our EPA-region split),
exchangeability is broken by design, so the marginal guarantee no longer holds
exactly: empirically the marginal coverage stays near nominal, but per-region
coverage varies and can dip mildly *below* the target (≈0.917 at alpha = 0.05 in
the weakest region) rather than being uniformly conservative. Report observed
per-region coverage alongside the marginal value, and prefer Mondrian
(group-conditional) or weighted conformal when region-conditional validity is
required. See Barber et al. (2023) for conformal prediction under distribution
shift.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


class ConformalClassifier:
    """Conformal predictor that wraps a fitted binary classifier.

    Produces prediction *sets* with finite-sample coverage guarantees
    using split (inductive) conformal inference.

    Parameters
    ----------
    model : BaseModel
        A fitted binary classification model with ``predict_proba``.
    alpha : float
        Desired miscoverage rate. The prediction sets target
        ``1 - alpha`` marginal coverage.
    """

    def __init__(self, model: BaseModel, alpha: float = 0.1) -> None:
        self._model = model
        self.alpha = alpha
        self._threshold: float | None = None

    def calibrate(
        self,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.Series | np.ndarray,
    ) -> None:
        """Compute the conformal threshold on calibration data.

        Parameters
        ----------
        X_cal : pd.DataFrame | np.ndarray
            Calibration features (held out from training).
        y_cal : pd.Series | np.ndarray
            Calibration labels (binary 0/1).
        """
        probs = self._model.predict_proba(X_cal)[:, 1]
        y = np.asarray(y_cal)

        # Nonconformity score = 1 - p(true class)
        scores = np.where(y == 1, 1.0 - probs, probs)

        n = len(scores)
        n_min = math.ceil(1.0 / self.alpha)
        if n < n_min:
            logger.warning(
                "Calibration set size %d < minimum %d for alpha=%.3f. "
                "Quantile will be clamped to 1.0, producing vacuous prediction sets.",
                n,
                n_min,
                self.alpha,
            )
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._threshold = float(np.quantile(scores, min(q, 1.0)))

    def predict_sets(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Produce prediction sets for new samples.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_samples, 2)`` where column 0
            indicates class 0 is included and column 1 indicates class 1
            is included in the prediction set.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called.
        """
        if self._threshold is None:
            raise RuntimeError(
                "ConformalClassifier has not been calibrated -- call .calibrate() first"
            )

        probs = self._model.predict_proba(X)[:, 1]
        threshold = self._threshold

        # Include class 1 if its nonconformity score <= threshold
        #   score for class 1 = 1 - prob  =>  include if prob >= 1 - threshold
        include_1 = probs >= (1.0 - threshold)

        # Include class 0 if its nonconformity score <= threshold
        #   score for class 0 = prob  =>  include if prob <= threshold
        include_0 = probs <= threshold

        return np.column_stack([include_0, include_1])

    def coverage_and_set_size(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate coverage and set-size statistics.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Test features.
        y : pd.Series | np.ndarray
            True binary labels.

        Returns
        -------
        dict
            Keys: ``"coverage"``, ``"avg_set_size"``,
            ``"singleton_frac"``, ``"both_classes_frac"``.
        """
        pred_sets = self.predict_sets(X)
        y_arr = np.asarray(y)

        # Coverage: fraction where the true class is included
        true_class_included = np.where(y_arr == 1, pred_sets[:, 1], pred_sets[:, 0])
        coverage = float(np.mean(true_class_included))

        set_sizes = pred_sets.sum(axis=1)
        avg_set_size = float(np.mean(set_sizes))
        singleton_frac = float(np.mean(set_sizes == 1))
        both_classes_frac = float(np.mean(set_sizes == 2))

        return {
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "singleton_frac": singleton_frac,
            "both_classes_frac": both_classes_frac,
        }


class GroupConformalClassifier:
    """Group-conditional conformal predictor with per-group coverage guarantees.

    Calibrates separate conformal thresholds per group (e.g., EPA region),
    ensuring each group independently achieves ≥ 1-alpha coverage.

    Parameters
    ----------
    model : BaseModel
        A fitted binary classification model with ``predict_proba``.
    alpha : float
        Desired miscoverage rate per group.
    """

    def __init__(
        self, model: BaseModel, alpha: float = 0.1, min_group_size: int | None = None
    ) -> None:
        self._model = model
        self.alpha = alpha
        self.min_group_size = min_group_size or max(math.ceil(1.0 / alpha), 30)
        self._group_thresholds: dict[Any, float] = {}
        self._group_cal_sizes: dict[Any, int] = {}
        self._small_groups: set[Any] = set()
        self._fallback_threshold: float | None = None

    def calibrate(
        self,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray,
    ) -> None:
        """Compute per-group conformal thresholds on calibration data.

        Parameters
        ----------
        X_cal : pd.DataFrame | np.ndarray
            Calibration features.
        y_cal : pd.Series | np.ndarray
            Calibration labels (binary 0/1).
        groups : pd.Series | np.ndarray
            Group labels for each sample (e.g., EPA region).
        """
        probs = self._model.predict_proba(X_cal)[:, 1]
        y = np.asarray(y_cal)
        groups_arr = np.asarray(groups)

        # Nonconformity scores
        scores = np.where(y == 1, 1.0 - probs, probs)

        # Fallback: marginal threshold over all data (computed first for use below)
        n_all = len(scores)
        n_min = math.ceil(1.0 / self.alpha)
        if n_all < n_min:
            logger.warning(
                "Total calibration set size %d < minimum %d for alpha=%.3f. "
                "Conformal coverage guarantees may not hold.",
                n_all,
                n_min,
                self.alpha,
            )
        q_all = np.ceil((n_all + 1) * (1 - self.alpha)) / n_all
        self._fallback_threshold = float(np.quantile(scores, min(q_all, 1.0)))

        # Compute per-group thresholds
        unique_groups = np.unique(groups_arr)
        for g in unique_groups:
            mask = groups_arr == g
            g_scores = scores[mask]
            n = len(g_scores)
            if n == 0:
                continue
            self._group_cal_sizes[g] = n
            if n < self.min_group_size:
                logger.warning(
                    "Group '%s' calibration set size %d < minimum %d for alpha=%.3f. "
                    "Falling back to marginal threshold for this group.",
                    g,
                    n,
                    self.min_group_size,
                    self.alpha,
                )
                self._group_thresholds[g] = self._fallback_threshold
                self._small_groups.add(g)
                continue
            q = np.ceil((n + 1) * (1 - self.alpha)) / n
            self._group_thresholds[g] = float(np.quantile(g_scores, min(q, 1.0)))

    def predict_sets(
        self,
        X: pd.DataFrame | np.ndarray,
        groups: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Produce prediction sets using group-specific thresholds.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Input features.
        groups : pd.Series | np.ndarray
            Group labels for each sample.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_samples, 2)``.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called.
        """
        if not self._group_thresholds:
            raise RuntimeError(
                "GroupConformalClassifier has not been calibrated -- call .calibrate() first"
            )

        probs = self._model.predict_proba(X)[:, 1]
        groups_arr = np.asarray(groups)
        n = len(probs)

        include_0 = np.zeros(n, dtype=bool)
        include_1 = np.zeros(n, dtype=bool)

        for i in range(n):
            g = groups_arr[i]
            threshold = self._group_thresholds.get(g, self._fallback_threshold) or 0.5
            include_1[i] = probs[i] >= (1.0 - threshold)
            include_0[i] = probs[i] <= threshold

        return np.column_stack([include_0, include_1])

    def coverage_and_set_size(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate per-group and overall coverage and set-size statistics.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Test features.
        y : pd.Series | np.ndarray
            True binary labels.
        groups : pd.Series | np.ndarray
            Group labels for each sample.

        Returns
        -------
        dict
            Keys: ``"coverage"``, ``"avg_set_size"``, ``"per_group"``
            (dict of group → coverage and avg_set_size).
        """
        pred_sets = self.predict_sets(X, groups)
        y_arr = np.asarray(y)
        groups_arr = np.asarray(groups)

        true_class_included = np.where(y_arr == 1, pred_sets[:, 1], pred_sets[:, 0])
        coverage = float(np.mean(true_class_included))
        set_sizes = pred_sets.sum(axis=1)
        avg_set_size = float(np.mean(set_sizes))

        # Per-group stats
        per_group: dict[str, dict[str, Any]] = {}
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            g_coverage = float(np.mean(true_class_included[mask]))
            g_set_size = float(np.mean(set_sizes[mask]))
            # A per-group guarantee is only valid if this group had its OWN
            # calibration data (n_calibration > 0) AND was large enough
            # (not in _small_groups). Groups absent from the calibration set
            # (e.g. test-region groups calibrated on disjoint val regions) fall
            # back to the marginal threshold in predict_sets and therefore have
            # NO per-group guarantee -- they must not be reported as reliable.
            n_cal_g = self._group_cal_sizes.get(g, 0)
            per_group[str(g)] = {
                "coverage": g_coverage,
                "avg_set_size": g_set_size,
                "n": int(mask.sum()),
                "n_calibration": n_cal_g,
                "reliable_guarantee": (g not in self._small_groups) and n_cal_g > 0,
            }

        return {
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "per_group": per_group,
        }


def split_conformal(
    model: BaseModel,
    X_cal: pd.DataFrame | np.ndarray,
    y_cal: pd.Series | np.ndarray,
    *,
    alpha: float = 0.1,
) -> ConformalClassifier:
    """Convenience function: create and calibrate a conformal classifier.

    Parameters
    ----------
    model : BaseModel
        A fitted binary classification model.
    X_cal : pd.DataFrame | np.ndarray
        Calibration features.
    y_cal : pd.Series | np.ndarray
        Calibration labels.
    alpha : float
        Desired miscoverage rate.

    Returns
    -------
    ConformalClassifier
        A calibrated conformal classifier ready for ``predict_sets``.
    """
    cc = ConformalClassifier(model, alpha=alpha)
    cc.calibrate(X_cal, y_cal)
    return cc
