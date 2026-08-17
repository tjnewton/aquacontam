"""Propensity score estimation for IPW reweighting.

Trains a simple model to predict monitoring intensity from causal-only
features, then computes inverse propensity weights (IPW) that correct
for monitoring selection bias.

Assumptions
-----------
IPW validity requires: (1) conditional ignorability — monitoring selection
is independent of contamination given observed features, (2) positivity —
all systems have non-zero probability of high/low monitoring, and (3)
correct propensity model specification. UCMR5 monitoring is MNAR (mandatory
for systems >3,300 population), so (1) may be partially violated. Estimates
should be interpreted as *partially adjusted* rather than fully debiased.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Monitoring-related columns that should NOT be used as propensity predictors
_MONITORING_COLUMNS = frozenset({"n_samples", "mean_detection_limit"})


def compute_propensity_scores(
    X: pd.DataFrame,
    n_samples: np.ndarray,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Estimate propensity scores for monitoring intensity.

    Trains a logistic regression to predict ``high_monitoring``
    (n_samples > median) from causal features (excluding monitoring columns).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    n_samples : np.ndarray
        Number of samples per system (monitoring intensity).
    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Propensity scores P(high_monitoring | X), shape ``(n,)``.
    """
    # Drop monitoring-related columns
    causal_cols = [c for c in X.columns if c not in _MONITORING_COLUMNS]
    X_causal = X[causal_cols].to_numpy().astype(np.float64)

    # Binary treatment: high monitoring = above median
    median_n = float(np.median(n_samples))
    treatment = (n_samples > median_n).astype(int)

    # Logistic regression for propensity estimation
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_causal, treatment)
    propensity = model.predict_proba(X_causal)[:, 1]

    logger.info(
        "Propensity scores: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
        propensity.mean(),
        propensity.std(),
        propensity.min(),
        propensity.max(),
    )
    return np.asarray(propensity)


def compute_ipw_weights(
    propensity: np.ndarray,
    n_samples: np.ndarray,
    *,
    clip_range: tuple[float, float] = (0.1, 10.0),
    stabilize: bool = True,
) -> np.ndarray:
    """Compute stabilized inverse propensity weights.

    Parameters
    ----------
    propensity : np.ndarray
        Propensity scores P(T=1 | X), shape ``(n,)``.
    n_samples : np.ndarray
        Monitoring intensity values.
    clip_range : tuple[float, float]
        Min/max bounds for clipping raw weights.
    stabilize : bool
        If True, normalize weights to mean 1.

    Returns
    -------
    np.ndarray
        IPW weights, shape ``(n,)``.
    """
    median_n = float(np.median(n_samples))
    treatment = (n_samples > median_n).astype(float)

    # Clip propensity to avoid extreme weights
    p_clipped = np.clip(propensity, 0.01, 0.99)

    # IPW: treated get 1/p, control get 1/(1-p)
    weights = np.where(treatment > 0.5, 1.0 / p_clipped, 1.0 / (1.0 - p_clipped))

    # Clip weights
    weights = np.clip(weights, clip_range[0], clip_range[1])

    # Stabilize: normalize to mean 1
    if stabilize and weights.sum() > 0:
        weights = weights / weights.mean()

    logger.info(
        "IPW weights: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
        weights.mean(),
        weights.std(),
        weights.min(),
        weights.max(),
    )
    return weights
