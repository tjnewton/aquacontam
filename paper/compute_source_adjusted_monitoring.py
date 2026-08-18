#!/usr/bin/env python
"""Source-adjusted monitoring-intensity disparity (the pivotal equity claim).

The headline "communities of color are sampled 1.85x more intensively" (frozen
``monitoring_inequity.json``; pinned ``clm_mon_ratio`` SIGNIFICANT) is estimated from
``n_samples``, a per-system sample count pooled across data sources whose per-system
sampling density differs by ~an order of magnitude (dense state programs vs a single
federal UCMR cycle). The size-adjusted association controls for population only -- there is
no data-source term -- so "1.85x" conflates cross-source corpus assembly with agency
allocation. This pre-specified compute (standing decision, 2026-07-12) re-estimates the
disparity three ways and lets the numbers decide the reframe:

  (i)  size-adjusted association WITH data-source fixed effects (dominant source per system);
  (ii) a single homogeneous source (UCMR5-only) estimate -- ratio + size-adjusted coef;
  (iii) the pooled ratio under both, next to the frozen unadjusted 1.85x.

Reproduction gate (HALT on failure -- do NOT chase by re-running the pipeline): the frozen
unadjusted POC monitoring ratio (1.847...) AND size-adjusted coefficient (0.060...) must
reproduce from local data via the exact published path (``analyze_monitoring_inequity`` on
the assembled X) before anything is written. This certifies the assembly IS the published
population. Output goes directly into the frozen archive (then ``freeze_results.py
--rehash-only`` and ``stamp_derivations.py``); additive-only, no existing artifact edited.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "source_adjusted_monitoring.json"
RATIO_TOL = 0.02
COEF_TOL = 1e-3
SEED = 42
POC = "pct_people_of_color"


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ols(design, y):
    """OLS via lstsq; returns (beta, se, dof) with homoscedastic SEs (matches frozen)."""
    import numpy as np

    beta, _resid, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    n, k = design.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
    return beta, se, dof


def _z(v):
    import numpy as np

    sd = np.std(v)
    return (v - np.mean(v)) / sd if sd > 1e-12 else np.zeros_like(v)


def _size_adjusted(n_samples, demo, population, source=None):
    """Std. partial association of demo on log monitoring intensity.

    Mirrors ``monitoring_equity._size_adjusted_association`` (intercept + z(demo) +
    z(log1p(pop))), optionally adding data-source fixed-effect dummies. Returns the demo
    coefficient (index 1), its OLS SE, t, two-sided p, and n.
    """
    import numpy as np
    from scipy import stats

    y = np.log1p(n_samples)
    cols = [np.ones_like(y), _z(demo)]
    if population is not None and np.std(population) > 1e-12:
        cols.append(_z(np.log1p(population)))
    if source is not None:
        # one-hot the source label, drop the most common level as reference
        srcs = np.asarray(source)
        levels, counts = np.unique(srcs, return_counts=True)
        ref = levels[int(np.argmax(counts))]
        for lv in levels:
            if lv == ref:
                continue
            cols.append((srcs == lv).astype(float))
    design = np.column_stack(cols)
    beta, se, dof = _ols(design, y)
    coef = float(beta[1])
    se1 = float(se[1])
    t_stat = coef / max(se1, 1e-12)
    p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
    return {
        "coef": coef,
        "se": se1,
        "t": t_stat,
        "p": p,
        "n": int(len(y)),
        "ref_source": (str(ref) if source is not None else None),
    }


def _ratio(n_samples, demo, hi_pct=80.0, lo_pct=50.0):
    """mean(intensity | demo>=P80) / mean(intensity | demo<P50) -- frozen split semantics."""
    import numpy as np

    hi_thr = float(np.percentile(demo, hi_pct))
    lo_thr = float(np.percentile(demo, lo_pct))
    hi = n_samples[demo >= hi_thr]
    lo = n_samples[demo < lo_thr]
    if lo.size == 0 or float(lo.mean()) == 0.0:
        return float("nan"), int(hi.size), int(lo.size)
    return float(hi.mean() / lo.mean()), int(hi.size), int(lo.size)


def main() -> int:
    import numpy as np
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.analysis.monitoring_equity import analyze_monitoring_inequity
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, _y, _stats, _regions = assemble_with_split_imputation(
        sys_targets, feature_dfs, wq_df=wq_df
    )
    if "n_samples" not in X.columns or POC not in X.columns:
        logger.error("assembled X lacks n_samples/%s; cannot reproduce the frozen statistic", POC)
        return 1
    population = X["population_served"] if "population_served" in X.columns else None

    # ---- reproduction gate: exact published path -----------------------------
    frozen = json.loads((FROZEN / "monitoring_inequity.json").read_text(encoding="utf-8"))
    repro = analyze_monitoring_inequity(
        X["n_samples"],
        X[[POC]],
        population=population,
        group_cols=(POC,),
        n_permutations=1,  # ratio + coef are permutation-independent; keep the gate fast
        seed=SEED,
    )
    obs_ratio = float(repro[POC]["monitoring_ratio"])
    obs_coef = float(repro[POC]["size_adjusted_coef"])
    fz_ratio = float(frozen[POC]["monitoring_ratio"])
    fz_coef = float(frozen[POC]["size_adjusted_coef"])
    if abs(obs_ratio - fz_ratio) > RATIO_TOL or abs(obs_coef - fz_coef) > COEF_TOL:
        logger.error(
            "GATE FAILED (HALT): ratio %.5f vs frozen %.5f (tol %.3f); coef %.6f vs frozen "
            "%.6f (tol %.4f). Assembly is NOT the published population; do not re-run the "
            "pipeline to chase it.",
            obs_ratio, fz_ratio, RATIO_TOL, obs_coef, fz_coef, COEF_TOL,
        )
        return 2
    logger.info(
        "Gate OK: ratio %.5f vs frozen %.5f; coef %.6f vs frozen %.6f",
        obs_ratio, fz_ratio, obs_coef, fz_coef,
    )

    # ---- per-system data-source label (dominant + UCMR5-only) ----------------
    src = wq_df.dropna(subset=["source"]).groupby("pwsid")["source"]
    dominant = src.agg(lambda s: s.value_counts().idxmax()).reindex(X.index)
    ucmr5_only = src.agg(lambda s: bool((s == "ucmr5").all())).reindex(X.index).fillna(False)
    n_by_src = src.agg(lambda s: s.value_counts().idxmax()).reindex(X.index).value_counts()
    logger.info("dominant-source system counts: %s", n_by_src.to_dict())

    n_samples = X["n_samples"].to_numpy(dtype=float)
    demo = X[POC].to_numpy(dtype=float)
    pop = population.to_numpy(dtype=float) if population is not None else None
    dom = dominant.to_numpy()
    has_src = ~pd.isna(dominant.to_numpy())

    # (i) size-adjusted WITH source fixed effects (over systems with a source label)
    fe = _size_adjusted(
        n_samples[has_src], demo[has_src],
        (pop[has_src] if pop is not None else None),
        source=dom[has_src],
    )
    logger.info("(i) source-FE demo coef %.6f (p=%.4g, n=%d)", fe["coef"], fe["p"], fe["n"])

    # (ii) UCMR5-only single-source
    u = ucmr5_only.to_numpy(dtype=bool)
    u_ratio, u_nhi, u_nlo = _ratio(n_samples[u], demo[u])
    u_fit = _size_adjusted(
        n_samples[u], demo[u], (pop[u] if pop is not None else None), source=None
    )
    logger.info(
        "(ii) UCMR5-only: ratio %.4f (n_hi=%d,n_lo=%d), coef %.6f (p=%.4g, n=%d)",
        u_ratio, u_nhi, u_nlo, u_fit["coef"], u_fit["p"], u_fit["n"],
    )

    # (iii) dominant-UCMR5 single-source estimate. Strict UCMR5-only (every record ucmr5) is
    # degenerate in the assembled population (most UCMR5 systems also carry a few SDWIS
    # lead/copper rows), so the faithful single-homogeneous-source subset is systems whose
    # MAJORITY of records are UCMR5. Report its ratio AND size-adjusted coefficient.
    dom_ucmr5 = dominant.to_numpy() == "ucmr5"
    du_ratio, du_nhi, du_nlo = _ratio(n_samples[dom_ucmr5], demo[dom_ucmr5])
    du_fit = _size_adjusted(
        n_samples[dom_ucmr5], demo[dom_ucmr5],
        (pop[dom_ucmr5] if pop is not None else None), source=None,
    )
    logger.info(
        "(iii) dominant-UCMR5: ratio %.4f (n_hi=%d,n_lo=%d), coef %.6f (p=%.4g, n=%d)",
        du_ratio, du_nhi, du_nlo, du_fit["coef"], du_fit["p"], du_fit["n"],
    )

    payload = {
        "_meta": {
            "source": (
                "R7 M1 source-adjusted monitoring-intensity disparity. Re-estimates the "
                "frozen unadjusted POC monitoring ratio (monitoring_inequity.json) adding "
                "(i) data-source fixed effects and (ii) a UCMR5-only single-source subset. "
                "Reproduction-gated on the frozen POC ratio AND size-adjusted coef via the "
                "exact analyze_monitoring_inequity path before writing."
            ),
            "reproduction_gate": {
                "poc_ratio_obs": obs_ratio,
                "poc_ratio_frozen": fz_ratio,
                "poc_coef_obs": obs_coef,
                "poc_coef_frozen": fz_coef,
                "ratio_tol": RATIO_TOL,
                "coef_tol": COEF_TOL,
            },
            "dominant_source_system_counts": {str(k): int(v) for k, v in n_by_src.items()},
            "method": (
                "OLS log1p(n_samples) ~ z(pct_people_of_color) + z(log1p(population)) "
                "[+ data-source dummies, most-common level as reference]; homoscedastic OLS "
                "SEs matching monitoring_equity._size_adjusted_association. UCMR5-only = "
                "systems whose every record has source==ucmr5."
            ),
        },
        "unadjusted_frozen": {
            "monitoring_ratio": fz_ratio,
            "size_adjusted_coef": fz_coef,
            "size_adjusted_p": float(frozen[POC]["size_adjusted_p"]),
            "n_systems": int(frozen[POC]["n_systems"]),
        },
        "source_fixed_effects": {
            "size_adjusted_coef": fe["coef"],
            "size_adjusted_se": fe["se"],
            "size_adjusted_t": fe["t"],
            "size_adjusted_p": fe["p"],
            "n_systems": fe["n"],
            "ref_source": fe["ref_source"],
            "attenuation_vs_unadjusted": (
                float(1.0 - fe["coef"] / fz_coef) if fz_coef else float("nan")
            ),
        },
        "ucmr5_only": {
            "monitoring_ratio": u_ratio,
            "n_high": u_nhi,
            "n_low": u_nlo,
            "size_adjusted_coef": u_fit["coef"],
            "size_adjusted_se": u_fit["se"],
            "size_adjusted_t": u_fit["t"],
            "size_adjusted_p": u_fit["p"],
            "n_systems": u_fit["n"],
        },
        "dominant_ucmr5": {
            "monitoring_ratio": du_ratio,
            "n_high": du_nhi,
            "n_low": du_nlo,
            "size_adjusted_coef": du_fit["coef"],
            "size_adjusted_se": du_fit["se"],
            "size_adjusted_t": du_fit["t"],
            "size_adjusted_p": du_fit["p"],
            "n_systems": du_fit["n"],
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)
    logger.info(
        "SUMMARY  unadj_ratio=%.3f coef=%.4f | FE_coef=%.4f(p=%.3g,att=%.0f%%) | "
        "UCMR5_ratio=%.3f coef=%.4f(p=%.3g)",
        fz_ratio, fz_coef, fe["coef"], fe["p"],
        100 * (1.0 - fe["coef"] / fz_coef) if fz_coef else float("nan"),
        u_ratio, u_fit["coef"], u_fit["p"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
