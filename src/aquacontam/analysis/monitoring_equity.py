"""National monitoring-inequity analysis.

The paper's central thesis is that machine-learning models of drinking-water
contamination predominantly encode the *monitoring process* rather than the
environment. A direct corollary -- and a novel environmental-justice finding in
its own right -- is that monitoring EFFORT is itself unequally distributed across
demographic lines: communities differ systematically in how intensively they are
sampled, which biases both contamination estimates and the equity conclusions
drawn from them.

This module quantifies that disparity among monitored systems, using only data
the pipeline already assembles (samples per system, demographics, population
served). Two complementary statistics per demographic dimension:

1. ``monitoring_ratio`` -- mean monitoring intensity in the high-demographic group
   (>= ``high_percentile``) divided by the reference group (< ``low_percentile``),
   directly analogous to the contamination burden ratio but on sampling effort.
   Significance via a permutation test.
2. ``size_adjusted_coef`` -- the standardized partial association between the
   demographic indicator and log monitoring intensity, CONTROLLING for log
   population served. UCMR5 monitoring is mandatory for systems serving > 3,300
   people, so system size is a regulatory-design confounder that must be adjusted
   for; the size-adjusted coefficient isolates the demographic association beyond
   system size.

Scope/limitation: this analysis is over MONITORED systems (the only ones with a
sample count). A stronger "selection" analysis -- whether a system is monitored at
all -- requires an eligible-but-unmonitored denominator (the full SDWIS public
water system inventory). If such a universe is supplied, ``analyze_monitoring_
inequity`` can be extended to model selection; by default it reports intensity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Demographic indicators analysed by default (EJScreen columns).
DEFAULT_GROUP_COLS: tuple[str, ...] = (
    "pct_people_of_color",
    "pct_low_income",
    "pct_limited_english",
    "pct_less_hs_education",
)


def _monitoring_ratio(
    n_samples: np.ndarray, g: np.ndarray, high_t: float, low_t: float
) -> tuple[float, float, float]:
    """Return ``(ratio, mean_high, mean_low)`` of monitoring intensity."""
    high = g >= high_t
    low = g < low_t
    mean_high = float(n_samples[high].mean()) if high.any() else 0.0
    mean_low = float(n_samples[low].mean()) if low.any() else 0.0
    if mean_low <= 0.0:
        return (float("inf") if mean_high > 0.0 else 1.0), mean_high, mean_low
    return mean_high / mean_low, mean_high, mean_low


def _size_adjusted_association(
    n_samples: np.ndarray, g: np.ndarray, population: np.ndarray | None
) -> dict[str, float]:
    """Standardized partial association of ``g`` on log monitoring intensity.

    Regresses ``log1p(n_samples)`` on the standardized demographic indicator and
    (when available) standardized log population served. Returns the demographic
    coefficient, its HC-free OLS standard error, t-statistic and two-sided p-value.
    The coefficient is on standardized predictors, so it is comparable across
    dimensions and isolates the demographic association beyond system size.
    """
    from scipy import stats

    y = np.log1p(n_samples)

    def _z(v: np.ndarray) -> np.ndarray:
        sd = np.std(v)
        return (v - np.mean(v)) / sd if sd > 1e-12 else np.zeros_like(v)

    cols = [np.ones_like(y), _z(g)]
    if population is not None and np.std(population) > 1e-12:
        cols.append(_z(np.log1p(population)))
    design = np.column_stack(cols)

    # OLS via least squares; demographic coefficient is index 1.
    beta, _resid, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    n, k = design.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    coef = float(beta[1])
    t_stat = coef / max(se, 1e-12)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
    return {
        "size_adjusted_coef": coef,
        "size_adjusted_se": se,
        "size_adjusted_p": p_value,
        "adjusted_for_population": bool(population is not None and np.std(population) > 1e-12),
    }


def analyze_monitoring_inequity(
    n_samples: np.ndarray | pd.Series,
    demographics: pd.DataFrame,
    *,
    population: np.ndarray | pd.Series | None = None,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    high_percentile: float = 80.0,
    low_percentile: float = 50.0,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Quantify demographic inequity in monitoring intensity (national).

    Parameters
    ----------
    n_samples : array-like
        Samples per monitored system (monitoring intensity).
    demographics : pd.DataFrame
        EJScreen demographic indicators aligned to ``n_samples``.
    population : array-like | None
        Population served per system, used as the size control in the
        size-adjusted association. When None, only the unadjusted ratio is given.
    group_cols : tuple[str, ...]
        Demographic columns to analyse.
    high_percentile, low_percentile : float
        High-burden / reference percentile cut points (defaults match the EJ
        burden-ratio analysis: 80th / 50th).

    Returns
    -------
    dict[str, dict[str, float]]
        Per-dimension: ``monitoring_ratio``, ``mean_high``, ``mean_low``,
        ``p_value``, ``n_high``, ``n_low``, plus the size-adjusted association
        fields when ``population`` is provided.
    """
    ns = np.asarray(n_samples, dtype=float)
    pop = np.asarray(population, dtype=float) if population is not None else None
    out: dict[str, dict[str, float]] = {}

    for col in group_cols:
        if col not in demographics.columns:
            continue
        g = np.asarray(demographics[col], dtype=float)
        valid = ~(np.isnan(ns) | np.isnan(g))
        if pop is not None:
            valid &= ~np.isnan(pop)
        if valid.sum() < 10:
            continue
        ns_v, g_v = ns[valid], g[valid]
        pop_v = pop[valid] if pop is not None else None

        high_t = float(np.percentile(g_v, high_percentile))
        low_t = float(np.percentile(g_v, low_percentile))
        ratio, mean_high, mean_low = _monitoring_ratio(ns_v, g_v, high_t, low_t)

        # Permutation test: shuffle the demographic assignment, recompute the ratio.
        rng = np.random.RandomState(seed)
        count_extreme = 0
        for _ in range(n_permutations):
            perm_ratio, _, _ = _monitoring_ratio(ns_v, rng.permutation(g_v), high_t, low_t)
            if (
                not np.isnan(perm_ratio)
                and not np.isnan(ratio)
                and abs(perm_ratio - 1.0) >= abs(ratio - 1.0)
            ):
                count_extreme += 1
        p_value = (count_extreme + 1) / (n_permutations + 1)

        entry: dict[str, float] = {
            "monitoring_ratio": float(ratio),
            "mean_high": mean_high,
            "mean_low": mean_low,
            "p_value": float(p_value),
            "n_high": int((g_v >= high_t).sum()),
            "n_low": int((g_v < low_t).sum()),
            "n_systems": int(valid.sum()),
        }
        entry.update(_size_adjusted_association(ns_v, g_v, pop_v))
        out[col] = entry

    return out
