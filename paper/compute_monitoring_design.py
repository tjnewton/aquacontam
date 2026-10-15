#!/usr/bin/env python
"""Monitoring design and the published disparity estimates (pre-registered as EST-D2).

The paper's people-of-color PFOS detection-burden ratios (test regions 1.878; region 9
1.337; national leave-one-region-out weighted mean 1.482) pool records from different
monitoring designs: the federal census (UCMR3/UCMR5), state programs, detection-only
state sources and ambient sites. This pre-registered compute asks how much those
estimates depend on design. It is additive only; no existing frozen artifact is edited.

Estimands (specified in ``paper/DECISIONS.md`` EST-D2 before this script was first run):

* **E1**: design-restricted burden ratios. The published ratio is recomputed from
  non-detect-reporting sources only, from the federal census only, and from state
  programs only. Population and demographic cut-offs are held fixed.
* **E2**: composition share of log(BR) explained by the high- and low-share groups'
  different design mixes (federal-only / state-only / both).
* **E3**: outcome-selective sampling in California. Among UCMR5 PFOS systems, the risk
  ratio of a UCMR5 detection for systems GeoTracker sampled before UCMR5 vs the rest.
* **E4**: California census (UCMR5) vs targeted (GeoTracker DDW, first sampled
  2019-2022) burden ratios over a 36-cell specification grid.

Reproduction gate (HALT on failure; never re-run the pipeline to chase a value):
* test-region POC burden ratio 1.878035 (±5e-3), n_high/n_low exactly 618/1,544;
* region 9 1.337262 (±5e-3), with counts 437/1,080/2,410/2,179;
* all ten LORO regional ratios (±5e-3) and the n_systems-weighted mean 1.482330 (±5e-3);
* 1,421 GeoTracker systems in the test regions;
* the union of the per-design labels reproduces every published system label.

Output: ``results/paper_frozen/monitoring_design.json``. After writing, run
``freeze_results.py --rehash-only``, add the G8 rows to DERIVATIONS.tsv, and stamp them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "monitoring_design.json"
DATA = REPO / "data"
MERGED_WQ = DATA / "interim" / "merged_wq.parquet"
DEMOGRAPHICS = DATA / "interim" / "features" / "demographics.parquet"
SYSTEM_CHARS = DATA / "interim" / "features" / "system_characteristics.parquet"

POC = "pct_people_of_color"
TEST_REGIONS = (8, 9, 10)
N_BOOT = 5000
SEED = 42
GATE_TOL = 5e-3
MIN_REGION_SYSTEMS = 30
MIN_DETECTIONS = 10
MIN_HIGH = 30
TARGETED_WINDOW = ("2019-01-01", "2022-12-31")  # order-driven GeoTracker phase (entry date)
GT_ALIGNED_WINDOW = (
    "2023-01-01",
    "2025-12-31",
)  # sensitivity: GeoTracker records aligned to UCMR5
SIZE_CUTOFF = 3300
CA_PREFIX = "CA"

# Frozen reproduction targets (EST-D2).
EXPECT_TEST = {"burden_ratio": 1.878035, "n_high": 618, "n_low": 1544}
EXPECT_R9 = {
    "burden_ratio": 1.337262,
    "n_high": 437,
    "n_low": 1080,
    "n_systems": 2410,
    "n_with_demographics": 2179,
}
EXPECT_NATIONAL = 1.482330
EXPECT_TEST_GT_SYSTEMS = 1421

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_monitoring_design")


# ----------------------------------------------------------------------------- utils
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(obj: Any) -> Any:
    """JSON-safe copy: non-finite floats become None; numpy scalars become Python."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _ci(draws: Any) -> tuple[float, float]:
    import numpy as np

    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _excludes(ci: tuple[float, float], null: float) -> bool:
    lo, hi = ci
    return bool(math.isfinite(lo) and math.isfinite(hi) and (lo > null or hi < null))


# ---------------------------------------------------------------------- population
def _prep_population() -> tuple[Any, Any, Any, Any, Any]:
    """The published ambient-inclusive population: (wq_df, index, y, regions, demo)."""
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    wq_df = pd.read_parquet(MERGED_WQ)
    feature_dfs = extract_features(DATA, wq_df, _discover_downloaded(DATA), skip_large=False)
    sys_targets = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS", target="detected"))
    X, y, _stats, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    demo_df = next(df for df in feature_dfs if isinstance(df, pd.DataFrame) and POC in df.columns)
    return wq_df, X.index, y, regions, demo_df


# ---------------------------------------------------------------------------- gate
def _reproduction_gate(wq_df, index, y, regions, demo_df, md) -> tuple[bool, dict[str, Any]]:
    """Check every EST-D2 reproduction target; return (passed, report)."""
    import numpy as np
    import pandas as pd

    report: dict[str, Any] = {}
    failures: list[str] = []
    poc = demo_df[POC]

    # test regions
    t_idx = index[regions.reindex(index).isin(TEST_REGIONS).to_numpy()]
    t_poc = poc.reindex(t_idx).to_numpy(dtype=float)
    t_est = md.burden_estimate(
        y.loc[t_idx].to_numpy(dtype=float), t_poc, md.percentile_cutoffs(t_poc)
    )
    report["test"] = t_est.as_dict()
    if abs(t_est.burden_ratio - EXPECT_TEST["burden_ratio"]) > GATE_TOL:
        failures.append(f"test BR {t_est.burden_ratio:.6f} vs {EXPECT_TEST['burden_ratio']}")
    if (t_est.n_high, t_est.n_low) != (EXPECT_TEST["n_high"], EXPECT_TEST["n_low"]):
        failures.append(f"test n_high/n_low {t_est.n_high}/{t_est.n_low} vs 618/1544")

    # every LORO region + the national weighted mean
    frozen_loro = json.loads((FROZEN / "loro_equity_analysis.json").read_text(encoding="utf-8"))
    fz = {e["region"]: e for e in frozen_loro["per_region"] if e["group"] == POC}
    per_region: dict[int, dict[str, Any]] = {}
    for r in sorted(fz):
        r_idx = index[(regions.reindex(index) == r).to_numpy()]
        r_poc = poc.reindex(r_idx).to_numpy(dtype=float)
        est = md.burden_estimate(
            y.loc[r_idx].to_numpy(dtype=float), r_poc, md.percentile_cutoffs(r_poc)
        )
        per_region[r] = est.as_dict()
        if abs(est.burden_ratio - float(fz[r]["burden_ratio"])) > GATE_TOL:
            failures.append(f"region {r} BR {est.burden_ratio:.6f} vs {fz[r]['burden_ratio']:.6f}")
        if est.n_systems != int(fz[r]["n_systems"]):
            failures.append(f"region {r} n_systems {est.n_systems} vs {fz[r]['n_systems']}")
    r9 = per_region.get(9, {})
    for key, want in EXPECT_R9.items():
        got = r9.get(key)
        ok = abs(float(got) - want) <= GATE_TOL if key == "burden_ratio" else got == want
        if not ok:
            failures.append(f"region 9 {key} {got} vs {want}")
    national = md.weighted_mean(
        [
            math.nan if per_region[r]["burden_ratio"] is None else per_region[r]["burden_ratio"]
            for r in per_region
        ],
        [per_region[r]["n_systems"] for r in per_region],
    )
    report["per_region"] = per_region
    report["national_weighted_mean"] = national
    if abs(national - EXPECT_NATIONAL) > GATE_TOL:
        failures.append(f"national {national:.6f} vs {EXPECT_NATIONAL}")

    # GeoTracker systems in the test regions
    pfos = wq_df.loc[wq_df["analyte"] == "PFOS"]
    gt_sys = set(pfos.loc[pfos["source"] == "ca_geotracker", "pwsid"])
    n_gt_test = int(pd.Index(t_idx).isin(list(gt_sys)).sum())
    report["test_geotracker_systems"] = n_gt_test
    if n_gt_test != EXPECT_TEST_GT_SYSTEMS:
        failures.append(f"test GeoTracker systems {n_gt_test} vs {EXPECT_TEST_GT_SYSTEMS}")

    # union of per-design labels == published labels
    groups = {
        "federal": md.FEDERAL_SOURCES,
        "state_ndr": md.STATE_NDR_SOURCES,
        "detection_only": md.DETECTION_ONLY_SOURCES,
        "ambient": md.AMBIENT_SOURCES,
    }
    labels = pd.concat(
        {g: md.detection_labels(wq_df, analytes=["PFOS"], sources=s) for g, s in groups.items()},
        axis=1,
    ).reindex(index)
    union = labels.max(axis=1)
    mismatch = int((union.to_numpy(dtype=float) != y.reindex(index).to_numpy(dtype=float)).sum())
    report["label_union_mismatches"] = mismatch
    if mismatch:
        failures.append(f"label union disagrees with published labels for {mismatch} systems")
    unlabeled = int(np.isnan(union.to_numpy(dtype=float)).sum())
    report["systems_without_design_label"] = unlabeled

    report["failures"] = failures
    report["passed"] = not failures
    return not failures, report


# ------------------------------------------------------------------------------ E1
def _row(md, y_sub, poc_sub, cutoffs, *, strata=None, ci_override=None) -> dict[str, Any]:
    est = md.burden_estimate(y_sub, poc_sub, cutoffs)
    out = est.as_dict()
    if ci_override is not None:
        out["ci_lower"], out["ci_upper"] = ci_override
        out["ci_source"] = "published"
    else:
        lo, hi = md.bootstrap_burden_ratio(
            y_sub, poc_sub, cutoffs, n_boot=N_BOOT, seed=SEED, strata=strata
        )
        out["ci_lower"], out["ci_upper"] = lo, hi
        out["ci_source"] = f"system bootstrap (n_boot={N_BOOT}, seed={SEED})"
    out["stable"] = md.is_stable(est, min_detections=MIN_DETECTIONS, min_high=MIN_HIGH)
    return out


def _design_label_frames(md, wq_df, index):
    import pandas as pd

    def lab(sources):
        return md.detection_labels(wq_df, analytes=["PFOS"], sources=sources).reindex(index)

    return pd.DataFrame(
        {
            "ndr": lab(md.FEDERAL_SOURCES + md.STATE_NDR_SOURCES),
            "federal": lab(md.FEDERAL_SOURCES),
            "state": lab(md.STATE_NDR_SOURCES),
        }
    )


def _e1_scope(md, scope_idx, y, poc, design, *, pooled_ci=None, state_label="state"):
    """Pooled / NDR / federal / state rows for one scope at the scope's fixed cut-offs."""
    import numpy as np

    poc_all = poc.reindex(scope_idx).to_numpy(dtype=float)
    cutoffs = md.percentile_cutoffs(poc_all)
    rows = {
        "pooled": _row(
            md, y.loc[scope_idx].to_numpy(dtype=float), poc_all, cutoffs, ci_override=pooled_ci
        )
    }
    for col, name in (("ndr", "ndr_pooled"), ("federal", "federal"), ("state", state_label)):
        lab = design[col].reindex(scope_idx)
        keep = lab.notna().to_numpy()
        if keep.sum() == 0:
            rows[name] = {"n_systems": 0}
            continue
        rows[name] = _row(md, lab.to_numpy(dtype=float)[keep], np.asarray(poc_all)[keep], cutoffs)
    return {"cutoffs": {"high": cutoffs[0], "low": cutoffs[1]}, "rows": rows}


def _e1_national(md, index, y, regions, poc, design, published_ci):
    """Region-weighted national rows; each region uses its own fixed published cut-offs."""
    import numpy as np

    reg = regions.reindex(index)
    region_ids = sorted(int(r) for r in reg.dropna().unique())
    per_region: dict[int, dict[str, Any]] = {}
    series = {"pooled": y.reindex(index)}
    for col in ("ndr", "federal", "state"):
        series[col] = design[col]
    for r in region_ids:
        r_idx = index[(reg == r).to_numpy()]
        r_poc = poc.reindex(r_idx).to_numpy(dtype=float)
        cut = md.percentile_cutoffs(r_poc)
        entry: dict[str, Any] = {"cutoffs": {"high": cut[0], "low": cut[1]}}
        for name, lab_all in series.items():
            lab = lab_all.reindex(r_idx)
            keep = lab.notna().to_numpy()
            n = int(keep.sum())
            if n < MIN_REGION_SYSTEMS:
                entry[name] = {"n_systems": n, "included": False}
                continue
            est = md.burden_estimate(lab.to_numpy(dtype=float)[keep], r_poc[keep], cut)
            entry[name] = {**est.as_dict(), "included": True}
        per_region[r] = entry

    def national(name, regions_subset=None):
        rs = [r for r in region_ids if per_region[r][name].get("included")]
        if regions_subset is not None:
            rs = [r for r in rs if r in regions_subset]
        vals = [per_region[r][name]["burden_ratio"] for r in rs]
        wts = [per_region[r][name]["n_systems"] for r in rs]
        vals = [v if v is not None else math.nan for v in vals]
        return md.weighted_mean(vals, wts), rs

    rng = np.random.default_rng(SEED)

    def boot(name, rs):
        draws = np.empty(N_BOOT)
        prepared = []
        for r in rs:
            r_idx = index[(reg == r).to_numpy()]
            lab = series[name].reindex(r_idx)
            keep = lab.notna().to_numpy()
            yv = lab.to_numpy(dtype=float)[keep]
            pv = poc.reindex(r_idx).to_numpy(dtype=float)[keep]
            cut = (per_region[r]["cutoffs"]["high"], per_region[r]["cutoffs"]["low"])
            hi, lo = md.group_masks(pv, cut)
            prepared.append((yv, hi, lo, int(keep.sum())))
        for b in range(N_BOOT):
            vals, wts = [], []
            for yv, hi, lo, n in prepared:
                s = rng.integers(0, n, n)
                vals.append(md.burden_ratio(yv[s], hi[s], lo[s]))
                wts.append(n)
            draws[b] = md.weighted_mean(vals, wts)
        return _ci(draws)

    rows: dict[str, Any] = {}
    pooled_val, pooled_rs = national("pooled")
    rows["pooled"] = {
        "burden_ratio": pooled_val,
        "regions": pooled_rs,
        "ci_lower": published_ci[0],
        "ci_upper": published_ci[1],
        "ci_source": "published (inference_strengthening region-block bootstrap)",
    }
    for name in ("ndr", "federal", "state"):
        val, rs = national(name)
        lo, hi = boot(name, rs) if rs else (math.nan, math.nan)
        rows["ndr_pooled" if name == "ndr" else name] = {
            "burden_ratio": val,
            "regions": rs,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_source": f"region-stratified system bootstrap (n_boot={N_BOOT}, seed={SEED})",
        }
    state_rs = rows["state"]["regions"]
    val, rs = national("pooled", regions_subset=set(state_rs))
    rows["pooled_state_regions"] = {
        "burden_ratio": val,
        "regions": rs,
        "note": "published pooled per-region ratios restricted to the state-only row's regions",
    }
    return {"rows": rows, "per_region": per_region}


# ------------------------------------------------------------------------------ E2
def _e2_scope(md, scope_idx, poc, design, pooled_ci):
    import numpy as np

    lab = design.reindex(scope_idx)
    keep = lab["ndr"].notna().to_numpy()
    fed = lab["federal"].notna().to_numpy()
    st = lab["state"].notna().to_numpy()
    strata = np.where(fed & st, "both", np.where(fed, "federal_only", "state_only"))[keep]
    yv = lab["ndr"].to_numpy(dtype=float)[keep]
    poc_all = poc.reindex(scope_idx).to_numpy(dtype=float)
    cutoffs = md.percentile_cutoffs(poc_all)  # the scope's published cut-offs
    pv = poc_all[keep]
    hi, lo = md.group_masks(pv, cutoffs)
    point = md.composition_share(yv, hi, lo, strata)
    rng = np.random.default_rng(SEED)
    n = int(keep.sum())
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        draws[b] = md.composition_share(yv[s], hi[s], lo[s], strata[s])["share"]
    ci = _ci(draws)
    reportable = _excludes(pooled_ci, 1.0)
    counts = {
        str(k): {"high": int(((strata == k) & hi).sum()), "low": int(((strata == k) & lo).sum())}
        for k in ("federal_only", "state_only", "both")
    }
    # Descriptive only (EST-D2 addendum): detections and rates within each stratum.
    rates: dict[str, dict[str, Any]] = {}
    for k in ("federal_only", "state_only", "both"):
        m = strata == k
        d_hi, n_hi = int(yv[m & hi].sum()), int((m & hi).sum())
        d_lo, n_lo = int(yv[m & lo].sum()), int((m & lo).sum())
        rates[k] = {
            "n_systems": int(m.sum()),
            "detections": int(yv[m].sum()),
            "rate_all": float(yv[m].mean()) if m.any() else math.nan,
            "detections_high": d_hi,
            "detections_low": d_lo,
            "rate_high": d_hi / n_hi if n_hi else math.nan,
            "rate_low": d_lo / n_lo if n_lo else math.nan,
        }
    return {
        **point,
        "share_ci_lower": ci[0],
        "share_ci_upper": ci[1],
        "reportable": reportable,
        "pooled_ci_used": list(pooled_ci),
        "n_systems": n,
        "strata_counts": counts,
        "strata_rates": rates,
    }


# ------------------------------------------------------------------------------ E3
def _e3(md, wq_df, demo_df, sys_chars):
    import numpy as np
    import pandas as pd

    ca = wq_df.loc[wq_df["pwsid"].astype(str).str.startswith(CA_PREFIX)]
    ucmr = ca.loc[(ca["source"] == "ucmr5") & (ca["analyte"] == "PFOS")]
    gt = ca.loc[(ca["source"] == "ca_geotracker") & (ca["analyte"] == "PFOS")]
    systems = pd.Index(sorted(ucmr["pwsid"].unique()))
    outcome = md.detection_labels(ucmr, analytes=["PFOS"]).reindex(systems)

    ucmr_dates = ucmr.groupby("pwsid")["sample_date"].agg(lambda s: set(pd.to_datetime(s)))
    first_ucmr = ucmr.groupby("pwsid")["sample_date"].min()
    gt = gt.loc[gt["pwsid"].isin(systems)].copy()
    gt["sample_date"] = pd.to_datetime(gt["sample_date"])
    same_day = [
        d in ucmr_dates.get(p, set()) for p, d in zip(gt["pwsid"], gt["sample_date"], strict=True)
    ]
    n_same_day = int(np.sum(same_day))
    gt_kept = gt.loc[~np.asarray(same_day, dtype=bool)]
    first_gt = gt_kept.groupby("pwsid")["sample_date"].min()
    exposed_pre = pd.Series(False, index=systems)
    has_gt = first_gt.reindex(systems)
    exposed_pre[(has_gt < first_ucmr.reindex(systems)).fillna(False).to_numpy()] = True
    exposed_any = has_gt.notna()

    yv = outcome.to_numpy(dtype=float)
    size = sys_chars["population_served"].reindex(systems)
    water = sys_chars["source_water_type"].reindex(systems).fillna("unknown")
    size_lbl = np.where(
        size.isna(), "size_unknown", np.where(size > SIZE_CUTOFF, "gt3300", "le3300")
    )
    water_lbl = np.where(water.isin(["GW", "SW"]), water, "other")
    strata = np.char.add(np.char.add(size_lbl.astype(str), "|"), water_lbl.astype(str))

    def rr_block(exposed):
        e = exposed.to_numpy(dtype=bool)
        point = md.risk_ratio(e, yv)
        mh = md.mantel_haenszel_risk_ratio(e, yv, strata)
        rng = np.random.default_rng(SEED)
        n = len(yv)
        d_rr, d_mh = np.empty(N_BOOT), np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rng.integers(0, n, n)
            d_rr[b] = md.risk_ratio(e[s], yv[s])
            d_mh[b] = md.mantel_haenszel_risk_ratio(e[s], yv[s], strata[s])
        return {
            "risk_ratio": point,
            "ci_lower": _ci(d_rr)[0],
            "ci_upper": _ci(d_rr)[1],
            "mantel_haenszel_risk_ratio": mh,
            "mh_ci_lower": _ci(d_mh)[0],
            "mh_ci_upper": _ci(d_mh)[1],
            "n_exposed": int(e.sum()),
            "n_unexposed": int((~e).sum()),
            "p_detect_exposed": float(yv[e].mean()) if e.any() else math.nan,
            "p_detect_unexposed": float(yv[~e].mean()) if (~e).any() else math.nan,
        }

    primary = rr_block(exposed_pre)
    any_sampling = rr_block(exposed_any)

    # secondary: targeting by POC (exposure rate high vs low POC among these systems)
    pv = demo_df[POC].reindex(systems).to_numpy(dtype=float)
    cut = md.percentile_cutoffs(pv)
    hi, lo = md.group_masks(pv, cut)
    e = exposed_pre.to_numpy(dtype=float)
    ratio = md.burden_ratio(e, hi, lo)
    rng = np.random.default_rng(SEED)
    n = len(e)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        draws[b] = md.burden_ratio(e[s], hi[s], lo[s])
    lo_ci, hi_ci = _ci(draws)
    return {
        "n_systems": len(systems),
        "n_same_day_geotracker_records_dropped": n_same_day,
        "primary_pre_ucmr5_sampling": primary,
        "any_sampling": any_sampling,
        "targeting_by_poc": {
            "exposure_rate_ratio_high_vs_low": ratio,
            "ci_lower": lo_ci,
            "ci_upper": hi_ci,
            "n_high": int(hi.sum()),
            "n_low": int(lo.sum()),
            # Descriptive only (EST-D2 addendum).
            "n_exposed_high": int(e[hi].sum()),
            "n_exposed_low": int(e[lo].sum()),
        },
        "strata": sorted(set(strata.tolist())),
    }


# ------------------------------------------------------------------------------ E4
def _e4(md, wq_df, demo_df, sys_chars):
    import numpy as np
    import pandas as pd

    ca = wq_df.loc[wq_df["pwsid"].astype(str).str.startswith(CA_PREFIX)]
    ca = ca.loc[ca["analyte"].isin(md.REGULATED_PFAS)].copy()
    ca["sample_date"] = pd.to_datetime(ca["sample_date"])
    ucmr = ca.loc[ca["source"] == "ucmr5"]
    gt = ca.loc[ca["source"] == "ca_geotracker"]
    common_rl = md.common_reporting_limits(ca, md.REGULATED_PFAS)
    gt_first = gt.groupby("pwsid")["sample_date"].min()
    t0, t1 = (pd.Timestamp(x) for x in TARGETED_WINDOW)
    targeted_primary = set(gt_first.index[(gt_first >= t0) & (gt_first <= t1)])
    targeted_all = set(gt_first.index)
    pop = sys_chars["population_served"]
    poc = demo_df[POC]

    def labels(frame, analytes, rl_mode):
        return md.detection_labels(
            frame, analytes=analytes, common_rl=common_rl if rl_mode == "common" else None
        )

    def cell(spec, *, cut_mode="combined", gt_frame=None):
        analytes = md.OUTCOMES[spec["outcome"]]
        tset = targeted_primary if spec["targeted"] == "first_sampled_2019_2022" else targeted_all
        c_lab = labels(ucmr, analytes, spec["rl"])
        gframe = gt if gt_frame is None else gt_frame
        t_lab = labels(gframe.loc[gframe["pwsid"].isin(tset)], analytes, spec["rl"])
        c_sys, t_sys = set(c_lab.index), set(t_lab.index)
        if spec["population"] == "gt3300":
            big = set(pop.index[pop > SIZE_CUTOFF])
            c_sys, t_sys = c_sys & big, t_sys & big
        elif spec["population"] == "overlap":
            both = c_sys & t_sys
            c_sys, t_sys = both, both
        union = sorted(c_sys | t_sys)
        u_poc = poc.reindex(union).to_numpy(dtype=float)
        in_c = np.isin(union, sorted(c_sys))
        in_t = np.isin(union, sorted(t_sys))
        c_y = c_lab.reindex(union).to_numpy(dtype=float)
        t_y = t_lab.reindex(union).to_numpy(dtype=float)
        if cut_mode == "combined":
            cut_c = cut_t = md.percentile_cutoffs(u_poc)
        else:
            cut_c = md.percentile_cutoffs(u_poc[in_c])
            cut_t = md.percentile_cutoffs(u_poc[in_t])
        est_c = md.burden_estimate(c_y[in_c], u_poc[in_c], cut_c)
        est_t = md.burden_estimate(t_y[in_t], u_poc[in_t], cut_t)

        def delta(bc, bt):
            return (
                math.log(bt) - math.log(bc)
                if (0 < bc < math.inf and 0 < bt < math.inf)
                else math.nan
            )

        d = delta(est_c.burden_ratio, est_t.burden_ratio)
        hi_c, lo_c = md.group_masks(u_poc, cut_c)
        hi_t, lo_t = md.group_masks(u_poc, cut_t)
        rng = np.random.default_rng(SEED)
        n = len(union)
        draws = np.empty(N_BOOT)
        draws_c, draws_t = np.empty(N_BOOT), np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rng.integers(0, n, n)
            sc, st_ = s[in_c[s]], s[in_t[s]]
            bc = md.burden_ratio(c_y[sc], hi_c[sc], lo_c[sc])
            bt = md.burden_ratio(t_y[st_], hi_t[st_], lo_t[st_])
            draws[b] = delta(bc, bt)
            draws_c[b], draws_t[b] = bc, bt
        ci = _ci(draws)
        # Descriptive only (EST-D2 addendum): per-design CIs from the same joint draws.
        ci_c, ci_t = _ci(draws_c), _ci(draws_t)
        stable = md.is_stable(
            est_c, min_detections=MIN_DETECTIONS, min_high=MIN_HIGH
        ) and md.is_stable(est_t, min_detections=MIN_DETECTIONS, min_high=MIN_HIGH)
        return {
            **{k: spec[k] for k in ("outcome", "rl", "targeted", "population")},
            "primary": bool(spec.get("primary", False)),
            "census": {**est_c.as_dict(), "ci_lower": ci_c[0], "ci_upper": ci_c[1]},
            "targeted_design": {**est_t.as_dict(), "ci_lower": ci_t[0], "ci_upper": ci_t[1]},
            "delta_log_br": d,
            "ratio_targeted_over_census": math.exp(d) if math.isfinite(d) else math.nan,
            "delta_ci_lower": ci[0],
            "delta_ci_upper": ci[1],
            "ratio_ci_lower": math.exp(ci[0]) if math.isfinite(ci[0]) else math.nan,
            "ratio_ci_upper": math.exp(ci[1]) if math.isfinite(ci[1]) else math.nan,
            "stable": stable,
            "n_union": n,
        }

    grid = [cell(spec) for spec in md.spec_grid()]
    primary_spec = next(c for c in md.spec_grid() if c["primary"])
    primary = next(c for c in grid if c["primary"])
    design_cuts = cell(primary_spec, cut_mode="design_specific")
    a0, a1 = (pd.Timestamp(x) for x in GT_ALIGNED_WINDOW)
    aligned = cell(
        primary_spec, gt_frame=gt.loc[(gt["sample_date"] >= a0) & (gt["sample_date"] <= a1)]
    )

    stable_cells = [c for c in grid if c["stable"] and math.isfinite(c["delta_log_br"])]
    signs = [1 if c["delta_log_br"] > 0 else -1 for c in stable_cells if c["delta_log_br"] != 0]
    majority = (1 if sum(signs) > 0 else -1) if signs else 0
    agree = (sum(1 for s in signs if s == majority) / len(signs)) if signs else math.nan
    ratios = [c["ratio_targeted_over_census"] for c in stable_cells]
    summary = {
        "n_cells": len(grid),
        "n_stable": len(stable_cells),
        "majority_sign": majority,
        "fraction_stable_agreeing_with_majority": agree,
        "ratio_range_stable": [min(ratios), max(ratios)] if ratios else [math.nan, math.nan],
        "census_br_range_stable": (
            [
                min(c["census"]["burden_ratio"] for c in stable_cells),
                max(c["census"]["burden_ratio"] for c in stable_cells),
            ]
            if stable_cells
            else [math.nan, math.nan]
        ),
        "targeted_br_range_stable": (
            [
                min(c["targeted_design"]["burden_ratio"] for c in stable_cells),
                max(c["targeted_design"]["burden_ratio"] for c in stable_cells),
            ]
            if stable_cells
            else [math.nan, math.nan]
        ),
    }
    return {
        "common_reporting_limits_ug_per_l": common_rl,
        "n_targeted_primary_systems": len(targeted_primary),
        "n_targeted_all_systems": len(targeted_all),
        "primary": primary,
        "primary_design_specific_cutoffs": design_cuts,
        "primary_geotracker_2023_2025_window": aligned,
        "spec_curve": grid,
        "summary": summary,
    }


# ---------------------------------------------------------------------------- main
def main() -> int:
    import pandas as pd

    from aquacontam.analysis import monitoring_design as md

    wq_df, index, y, regions, demo_df = _prep_population()
    logger.info("Population: %d systems", len(index))

    ok, gate = _reproduction_gate(wq_df, index, y, regions, demo_df, md)
    if not ok:
        logger.error("REPRODUCTION GATE FAILED (HALT; nothing written): %s", gate["failures"])
        return 2
    logger.info("Reproduction gate OK: %s", {k: gate[k] for k in ("national_weighted_mean",)})

    poc = demo_df[POC]
    design = _design_label_frames(md, wq_df, index)
    reg = regions.reindex(index)
    t_idx = index[reg.isin(TEST_REGIONS).to_numpy()]
    r9_idx = index[(reg == 9).to_numpy()]

    burden_ci = json.loads((FROZEN / "group_burden_ci.json").read_text(encoding="utf-8"))
    test_ci = tuple(burden_ci["burden_cis"][POC]["burden_ratio_ci"])
    infer = json.loads((FROZEN / "inference_strengthening.json").read_text(encoding="utf-8"))
    bb = infer["m4c_burden_ratio_ci"]["block_bootstrap"]
    national_ci = (float(bb["ci_lower"]), float(bb["ci_upper"]))

    e1 = {
        "test": _e1_scope(md, t_idx, y, poc, design, pooled_ci=test_ci, state_label="state_ca"),
        "region9": _e1_scope(md, r9_idx, y, poc, design, state_label="state_ca"),
        "national": _e1_national(md, index, y, regions, poc, design, national_ci),
    }
    r9_pooled = e1["region9"]["rows"]["pooled"]
    r9_ci = (r9_pooled["ci_lower"], r9_pooled["ci_upper"])
    e2 = {
        "test": _e2_scope(md, t_idx, poc, design, test_ci),
        "region9": _e2_scope(md, r9_idx, poc, design, r9_ci),
    }
    sys_chars = pd.read_parquet(SYSTEM_CHARS)
    e3 = _e3(md, wq_df, demo_df, sys_chars)
    e4 = _e4(md, wq_df, demo_df, sys_chars)

    prim = e4["primary"]
    s = e4["summary"]
    prose_rule: dict[str, Any] = {
        "design_mix_explains_test": bool(
            e2["test"]["reportable"]
            and _excludes((e2["test"]["share_ci_lower"], e2["test"]["share_ci_upper"]), 0.0)
        ),
        "design_mix_explains_region9": bool(
            e2["region9"]["reportable"]
            and _excludes((e2["region9"]["share_ci_lower"], e2["region9"]["share_ci_upper"]), 0.0)
        ),
        "outcome_selective_targeting": bool(
            _excludes(
                (
                    e3["primary_pre_ucmr5_sampling"]["ci_lower"],
                    e3["primary_pre_ucmr5_sampling"]["ci_upper"],
                ),
                1.0,
            )
            and _excludes((e3["any_sampling"]["ci_lower"], e3["any_sampling"]["ci_upper"]), 1.0)
        ),
        "direction_statable": bool(
            prim["stable"]
            and _excludes((prim["delta_ci_lower"], prim["delta_ci_upper"]), 0.0)
            and math.isfinite(s["fraction_stable_agreeing_with_majority"])
            and s["fraction_stable_agreeing_with_majority"] >= 0.8
        ),
    }
    prose_rule["direction"] = (
        ("targeted_lower" if prim["delta_log_br"] < 0 else "targeted_higher")
        if prose_rule["direction_statable"]
        else None
    )

    payload = {
        "_meta": {
            "decision": "EST-D2 (paper/DECISIONS.md), pre-registered 2026-09-24 before first run",
            "disclosure": (
                "Exploratory pilots (2026-09-16 session scratch; 2026-09-24 read-only "
                "feasibility checks) preceded this specification. All 36 grid cells and "
                "all sensitivities are reported."
            ),
            "addendum": (
                "Descriptive outputs (E2 per-stratum rates, E3 exposure counts, E4 per-design "
                "CIs from the existing joint draws) were added after the first run and before "
                "freezing. No estimand, CI, cut-off or prose-rule input changed (EST-D2 addendum)."
            ),
            "n_boot": N_BOOT,
            "seed": SEED,
            "cutoff_percentiles": {"high": 80, "low": 50},
            "min_region_systems": MIN_REGION_SYSTEMS,
            "stability": {"min_detections": MIN_DETECTIONS, "min_high": MIN_HIGH},
            "targeted_window": list(TARGETED_WINDOW),
            "geotracker_aligned_window": list(GT_ALIGNED_WINDOW),
            "size_cutoff_population_served": SIZE_CUTOFF,
            "n_specs": len(md.spec_grid()),
            "design_groups": {
                "federal": list(md.FEDERAL_SOURCES),
                "state_ndr": list(md.STATE_NDR_SOURCES),
                "detection_only": list(md.DETECTION_ONLY_SOURCES),
                "ambient": list(md.AMBIENT_SOURCES),
            },
            "inputs_sha256": {
                "data/interim/merged_wq.parquet": _sha256(MERGED_WQ),
                "data/interim/features/demographics.parquet": _sha256(DEMOGRAPHICS),
                "data/interim/features/system_characteristics.parquet": _sha256(SYSTEM_CHARS),
            },
        },
        "reproduction_gate": gate,
        "e1_design_restricted": e1,
        "e2_composition": e2,
        "e3_outcome_selectivity": e3,
        "e4_ca_census_vs_targeted": e4,
        "prose_rule": prose_rule,
    }
    OUT_PATH.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", OUT_PATH)
    logger.info("prose rule: %s", prose_rule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
