"""Monitoring-design decomposition of environmental-justice disparity estimates.

Pure, frame-level estimators behind ``paper/compute_monitoring_design.py``
(pre-registered as decision EST-D2). They ask how much of the published
people-of-color PFOS detection-burden ratios depends on *how* systems were
monitored:

1. **Design-restricted ratios.** Recompute a burden ratio from the records of one
   monitoring design only (the federal census vs state programs), holding the
   population and the demographic cut-offs fixed.
2. **Composition share.** How much of a pooled ratio disappears when the
   high-share group is given the low-share group's mix of monitoring designs.
3. **Outcome-selective sampling.** Whether a targeted program sampled systems
   where a census later found contamination, measured as a risk ratio.
4. **Census-vs-targeted specification grid.** The 36-cell grid for California.

Conventions mirror :func:`aquacontam.analysis.equity.analyze_equity`:
- Groups are the >= 80th and < 50th percentiles of the demographic share among
  systems with a non-missing value; the middle and unknown groups are excluded.
- The burden ratio is the detection rate in the high group over the rate in the
  low group. It is ``inf`` when only the low rate is zero and ``1.0`` when both
  are zero.
- A system is *detected* if any of its informative records is uncensored.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = [
    "AMBIENT_SOURCES",
    "DETECTION_ONLY_SOURCES",
    "FEDERAL_SOURCES",
    "REGULATED_PFAS",
    "STATE_NDR_SOURCES",
    "BurdenEstimate",
    "bootstrap_burden_ratio",
    "burden_estimate",
    "burden_ratio",
    "common_reporting_limits",
    "composition_share",
    "detection_labels",
    "first_sample_dates",
    "group_masks",
    "is_stable",
    "mantel_haenszel_risk_ratio",
    "percentile_cutoffs",
    "risk_ratio",
    "spec_grid",
    "weighted_mean",
]

#: Federal census programs (UCMR3: systems serving >10,000; UCMR5: >3,300 plus a
#: representative sample of smaller systems).
FEDERAL_SOURCES: tuple[str, ...] = ("ucmr3", "ucmr5")
#: State programs that report non-detects as well as detections.
STATE_NDR_SOURCES: tuple[str, ...] = ("ca_geotracker", "mn_mdh", "nj_dep", "mo_dnr")
#: State sources that report detections only (every record is a detection).
DETECTION_ONLY_SOURCES: tuple[str, ...] = ("mi_mpart", "wa_doh")
#: Ambient (non-public-water-system) monitoring.
AMBIENT_SOURCES: tuple[str, ...] = ("wqp",)
#: The six PFAS regulated by the April 2024 National Primary Drinking Water Regulation.
REGULATED_PFAS: tuple[str, ...] = ("PFOA", "PFOS", "PFNA", "PFHxS", "PFBS", "HFPO-DA")

_RL_TOL = 1e-9  # absolute tolerance (ug/L) when comparing values with reporting limits


# --------------------------------------------------------------------------- groups
def percentile_cutoffs(
    values: Iterable[float], high_pct: float = 80.0, low_pct: float = 50.0
) -> tuple[float, float]:
    """High and low cut-offs of a demographic share, ignoring missing values.

    Parameters
    ----------
    values : iterable of float
        Demographic shares; NaN marks a system without demographics.
    high_pct, low_pct : float
        Percentiles defining the high (>=) and low (<) groups.

    Returns
    -------
    tuple of float
        ``(high_cutoff, low_cutoff)``; ``(0.0, 0.0)`` when no value is present.
    """
    arr = np.asarray(list(values), dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.percentile(valid, high_pct)), float(np.percentile(valid, low_pct))


def group_masks(
    values: Iterable[float], cutoffs: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean high/low masks for fixed cut-offs (missing values fall in neither)."""
    arr = np.asarray(list(values), dtype=float)
    hi_cut, lo_cut = cutoffs
    valid = ~np.isnan(arr)
    return valid & (arr >= hi_cut), valid & (arr < lo_cut)


def burden_ratio(y: Iterable[float], high: np.ndarray, low: np.ndarray) -> float:
    """Detection rate in the high group over the rate in the low group.

    Matches :func:`aquacontam.analysis.equity.analyze_equity`: ``inf`` when only
    the low rate is zero, ``1.0`` when both rates are zero.
    """
    yv = np.asarray(list(y), dtype=float)
    rate_high = float(yv[high].mean()) if high.any() else 0.0
    rate_low = float(yv[low].mean()) if low.any() else 0.0
    if rate_low == 0.0:
        return math.inf if rate_high > 0.0 else 1.0
    return rate_high / rate_low


@dataclass(frozen=True)
class BurdenEstimate:
    """A burden ratio with the counts a reader needs to judge it."""

    burden_ratio: float
    n_high: int
    n_low: int
    detections_high: int
    detections_low: int
    n_systems: int
    n_with_demographics: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary (non-finite ratios become ``None``)."""
        br = self.burden_ratio
        return {
            "burden_ratio": br if math.isfinite(br) else None,
            "n_high": self.n_high,
            "n_low": self.n_low,
            "detections_high": self.detections_high,
            "detections_low": self.detections_low,
            "n_systems": self.n_systems,
            "n_with_demographics": self.n_with_demographics,
        }


def burden_estimate(
    y: Iterable[float], demographic: Iterable[float], cutoffs: tuple[float, float]
) -> BurdenEstimate:
    """Burden ratio plus group counts at fixed cut-offs."""
    yv = np.asarray(list(y), dtype=float)
    dv = np.asarray(list(demographic), dtype=float)
    if yv.shape != dv.shape:
        raise ValueError(f"y and demographic lengths differ: {yv.shape} vs {dv.shape}")
    high, low = group_masks(dv, cutoffs)
    return BurdenEstimate(
        burden_ratio=burden_ratio(yv, high, low),
        n_high=int(high.sum()),
        n_low=int(low.sum()),
        detections_high=int(yv[high].sum()),
        detections_low=int(yv[low].sum()),
        n_systems=int(yv.size),
        n_with_demographics=int((~np.isnan(dv)).sum()),
    )


def is_stable(estimate: BurdenEstimate, *, min_detections: int = 10, min_high: int = 30) -> bool:
    """True when both groups have >= ``min_detections`` detections and n_high >= ``min_high``."""
    return (
        estimate.detections_high >= min_detections
        and estimate.detections_low >= min_detections
        and estimate.n_high >= min_high
    )


# --------------------------------------------------------------------------- labels
def common_reporting_limits(
    records: pd.DataFrame,
    analytes: Sequence[str],
    *,
    census_source: str = "ucmr5",
    targeted_source: str = "ca_geotracker",
) -> dict[str, float]:
    """Per-analyte common reporting limit: the higher of the two designs' native limits.

    The census limit is the modal ``detection_limit`` of the census records (UCMR5
    minimum reporting levels are fixed per analyte). The targeted limit is the
    median of the targeted records (laboratory limits vary). Values in ug/L.
    """
    out: dict[str, float] = {}
    for analyte in analytes:
        rl = records.loc[records["analyte"] == analyte]
        census = rl.loc[rl["source"] == census_source, "detection_limit"].dropna()
        targeted = rl.loc[rl["source"] == targeted_source, "detection_limit"].dropna()
        if census.empty and targeted.empty:
            continue
        census_rl = float(census.mode().iloc[0]) if not census.empty else 0.0
        targeted_rl = float(targeted.median()) if not targeted.empty else 0.0
        out[analyte] = max(census_rl, targeted_rl)
    return out


def detection_labels(
    records: pd.DataFrame,
    *,
    analytes: Sequence[str],
    sources: Sequence[str] | None = None,
    common_rl: Mapping[str, float] | None = None,
) -> pd.Series:
    """System-level any-detection labels from a subset of monitoring records.

    Parameters
    ----------
    records : pd.DataFrame
        Long-form samples with ``pwsid``, ``analyte``, ``source``, ``censored``,
        ``concentration`` and ``detection_limit``.
    analytes : sequence of str
        A system is detected if any listed analyte is detected.
    sources : sequence of str, optional
        Restrict to records from these sources (``None`` keeps every source).
    common_rl : mapping, optional
        Per-analyte common reporting limit (ug/L). When given, a detection below
        the limit becomes a non-detect. A non-detect whose own limit exceeds the
        common limit is uninformative and dropped, as is a detection without a
        concentration. Systems left with no informative record are excluded.

    Returns
    -------
    pd.Series
        0/1 labels indexed by ``pwsid``, one row per system with an informative record.
    """
    sel = records.loc[records["analyte"].isin(list(analytes))]
    if sources is not None:
        sel = sel.loc[sel["source"].isin(list(sources))]
    if sel.empty:
        return pd.Series(dtype=int, name="detected")
    detected = ~sel["censored"].astype(bool).to_numpy()
    informative = np.ones(len(sel), dtype=bool)
    if common_rl is not None:
        limits = sel["analyte"].map(dict(common_rl)).to_numpy(dtype=float)
        conc = sel["concentration"].to_numpy(dtype=float)
        dl = sel["detection_limit"].to_numpy(dtype=float)
        has_limit = ~np.isnan(limits)
        # detections below the common limit are re-censored; missing concentration is unknowable
        det_above = detected & (conc >= limits - _RL_TOL)
        det_unknown = detected & np.isnan(conc)
        # a non-detect is informative only if its own limit is at or below the common limit
        nd_informative = ~detected & ~np.isnan(dl) & (dl <= limits + _RL_TOL)
        informative = has_limit & (nd_informative | (detected & ~det_unknown))
        detected = det_above & informative
    frame = pd.DataFrame(
        {"pwsid": sel["pwsid"].to_numpy(), "detected": detected, "informative": informative}
    )
    frame = frame.loc[frame["informative"]]
    labels = frame.groupby("pwsid", observed=True)["detected"].any().astype(int)
    labels.name = "detected"
    return cast(pd.Series, labels)


def first_sample_dates(
    records: pd.DataFrame, *, sources: Sequence[str], analytes: Sequence[str]
) -> pd.Series:
    """Earliest ``sample_date`` per system among the given sources and analytes."""
    sel = records.loc[
        records["source"].isin(list(sources)) & records["analyte"].isin(list(analytes))
    ]
    dates = pd.to_datetime(sel["sample_date"], errors="coerce")
    return cast(pd.Series, dates.groupby(sel["pwsid"].to_numpy()).min().dropna())


# ------------------------------------------------------------------------ inference
def bootstrap_burden_ratio(
    y: Iterable[float],
    demographic: Iterable[float],
    cutoffs: tuple[float, float],
    *,
    n_boot: int,
    seed: int,
    strata: Iterable[Any] | None = None,
) -> tuple[float, float]:
    """Percentile 95% CI of a burden ratio by resampling systems (cut-offs held fixed).

    With ``strata``, systems are resampled within each stratum (e.g., EPA region).
    Non-finite draws are ignored; returns ``(nan, nan)`` if none are finite.
    """
    yv = np.asarray(list(y), dtype=float)
    high, low = group_masks(demographic, cutoffs)
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(yv.size)]
    else:
        sv = np.asarray(list(strata), dtype=object)
        groups = [np.flatnonzero(sv == s) for s in pd.unique(sv)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups if g.size])
        draws[b] = burden_ratio(yv[idx], high[idx], low[idx])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted mean over finite values (the published LORO national convention)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return math.nan
    return float(np.average(v[keep], weights=w[keep]))


def composition_share(
    y: Iterable[float], high: np.ndarray, low: np.ndarray, strata: Iterable[Any]
) -> dict[str, float]:
    """Share of log(burden ratio) explained by the groups' different design mixes.

    The counterfactual ratio gives the high group the low group's distribution
    over design strata (e.g., federal-only, state-only, both) while keeping the
    high group's within-stratum detection rates. The share is
    ``(log BR - log BR_cf) / log BR``. It is NaN when the counterfactual is
    undefined (a stratum present in the low group is absent from the high group)
    or when BR <= 0 or BR == 1.
    """
    yv = np.asarray(list(y), dtype=float)
    sv = np.asarray(list(strata), dtype=object)
    br = burden_ratio(yv, high, low)
    rate_low = float(yv[low].mean()) if low.any() else 0.0
    nan = {"burden_ratio": br, "burden_ratio_counterfactual": math.nan, "share": math.nan}
    if rate_low == 0.0 or not high.any():
        return nan
    num = 0.0
    for s in pd.unique(sv[low]):
        w_low = float(np.mean(sv[low] == s))
        in_high = high & (sv == s)
        if not in_high.any():
            return nan
        num += w_low * float(yv[in_high].mean())
    br_cf = num / rate_low
    if not (math.isfinite(br) and br > 0 and br != 1.0 and br_cf > 0):
        return {"burden_ratio": br, "burden_ratio_counterfactual": br_cf, "share": math.nan}
    share = (math.log(br) - math.log(br_cf)) / math.log(br)
    return {"burden_ratio": br, "burden_ratio_counterfactual": br_cf, "share": share}


def risk_ratio(exposed: Iterable[bool], outcome: Iterable[float]) -> float:
    """P(outcome | exposed) / P(outcome | unexposed); NaN if undefined."""
    e = np.asarray(list(exposed), dtype=bool)
    o = np.asarray(list(outcome), dtype=float)
    if not e.any() or e.all():
        return math.nan
    p1, p0 = float(o[e].mean()), float(o[~e].mean())
    return p1 / p0 if p0 > 0 else math.nan


def mantel_haenszel_risk_ratio(
    exposed: Iterable[bool], outcome: Iterable[float], strata: Iterable[Any]
) -> float:
    """Mantel-Haenszel pooled risk ratio across strata; NaN if undefined."""
    e = np.asarray(list(exposed), dtype=bool)
    o = np.asarray(list(outcome), dtype=float)
    s = np.asarray(list(strata), dtype=object)
    num = den = 0.0
    for level in pd.unique(s):
        m = s == level
        n = int(m.sum())
        n1 = int((e & m).sum())
        n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        a = float(o[e & m].sum())
        c = float(o[~e & m].sum())
        num += a * n0 / n
        den += c * n1 / n
    return num / den if den > 0 else math.nan


# ------------------------------------------------------------------------ spec grid
OUTCOMES: dict[str, tuple[str, ...]] = {
    "PFOS": ("PFOS",),
    "PFOA_or_PFOS": ("PFOA", "PFOS"),
    "six_regulated": REGULATED_PFAS,
}
RL_MODES: tuple[str, ...] = ("native", "common")
TARGETED_SETS: tuple[str, ...] = ("first_sampled_2019_2022", "all_geotracker")
POPULATIONS: tuple[str, ...] = ("all", "gt3300", "overlap")
PRIMARY_SPEC: dict[str, str] = {
    "outcome": "PFOS",
    "rl": "common",
    "targeted": "first_sampled_2019_2022",
    "population": "all",
}


def spec_grid() -> list[dict[str, Any]]:
    """The 36-cell California census-vs-targeted grid, exactly one cell marked primary."""
    cells: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        for rl in RL_MODES:
            for targeted in TARGETED_SETS:
                for population in POPULATIONS:
                    spec: dict[str, Any] = {
                        "outcome": outcome,
                        "rl": rl,
                        "targeted": targeted,
                        "population": population,
                    }
                    spec["primary"] = spec == {**PRIMARY_SPEC}
                    cells.append(spec)
    return cells
