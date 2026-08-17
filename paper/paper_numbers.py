#!/usr/bin/env python
"""Single source of truth for headline numbers in the manuscript prose.

Every headline metric in the paper markdown is wrapped in an idempotent marker::

    AUROC <!--pn:t1_full_auroc-->0.852<!--/pn--> ...

The value between the markers is rendered FROM the frozen result snapshot
(``results/paper_frozen/``) by :func:`sync`, so the prose is correct by
construction and re-syncable after any future re-freeze (unlike the one-shot
``[X.XX]`` placeholders). HTML comments are invisible in GitHub render and
stripped by pandoc, so the DOCX shows only the value; ``--strip`` additionally
removes the markers for a guaranteed-clean export.

Usage::

    python paper/paper_numbers.py --write    # fill every marker from frozen
    python paper/paper_numbers.py --check     # assert prose == frozen (gate)
    python paper/paper_numbers.py --strip --out-dir build/  # markers removed

The gate (`--check`) is a value comparison, not a brittle prose regex: a marker
either equals its frozen-derived value or fails. ``check_paper_numbers`` is
called by ``paper/verify_paper.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
PAPER = REPO / "paper"
#: Committed snapshots that live OUTSIDE the frozen archive (so results/paper_frozen/
#: stays byte-unchanged) yet make prose gating independent of the git-ignored parquet.
SNAPSHOTS = PAPER / "frozen_snapshots"

#: Paper markdown files (under paper/) scanned for markers.
PAPER_FILES: tuple[str, ...] = (
    "skeleton.md",
    "supplementary_information.md",
    "extended_data.md",
    "cover_letter.md",
    "reporting_summary.md",
    "plain_language_primer.md",
)

#: Reader-facing files scanned for markers that live at the repo ROOT (not under paper/).
#: LEADERBOARD.md is NOT here: it is a GENERATED artifact, regenerated from the frozen
#: archive by paper/regenerate_leaderboard.py and verified by byte-diff-match (like
#: paper/tables/), so it carries no pn markers. The mechanism is kept for future root files.
ROOT_FILES: tuple[str, ...] = ()


def _default_files() -> list[Path]:
    """All scanned files: paper/ markdown + repo-root reader-facing files."""
    return [PAPER / f for f in PAPER_FILES if (PAPER / f).exists()] + [
        REPO / f for f in ROOT_FILES if (REPO / f).exists()
    ]


#: Marker syntax: <!--pn:ID-->VALUE<!--/pn--> (VALUE re-synced in place).
_MARKER = re.compile(r"<!--pn:([a-z0-9_]+)-->(.*?)<!--/pn-->", re.DOTALL)
_STRIP = re.compile(r"<!--pn:[a-z0-9_]+-->(.*?)<!--/pn-->", re.DOTALL)


# --------------------------------------------------------------------------
# Frozen-source readers (read LIVE from the frozen snapshot at call time)
# --------------------------------------------------------------------------
def _results_index(name: str = "results.json") -> dict[tuple[str, str], dict[str, float]]:
    path = FROZEN / name
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("results", [])
    return {(r.get("task", ""), r.get("model", "")): r.get("metrics", {}) for r in rows}


def _metric(task: str, model: str, key: str, name: str = "results.json") -> float:
    v = _results_index(name).get((task, model), {}).get(key)
    if v is None:
        raise KeyError(f"{name}: ({task},{model}).{key} not found")
    return float(v)


def _best(task: str, key: str, name: str = "results.json", *, lowest: bool = False) -> float:
    vals = [
        m[key]
        for (t, _mdl), m in _results_index(name).items()
        if t == task and isinstance(m.get(key), (int, float))
    ]
    if not vals:
        raise KeyError(f"{name}: no {key} for task {task}")
    return float(min(vals) if lowest else max(vals))


def _json(name: str) -> Any:
    return json.loads((FROZEN / name).read_text(encoding="utf-8"))


def _loro_mean(name: str) -> float:
    d = _json(name)
    t1 = d.get("T1", {}) if isinstance(d, dict) else {}
    v = t1.get("mean_auroc")
    if v is None and isinstance(t1.get("xgboost"), dict):
        v = t1["xgboost"].get("mean_auroc")
    if v is None:
        raise KeyError(f"{name}: T1 mean_auroc not found")
    return float(v)


def _ablation_baseline(task: str) -> float:
    rows = [r for r in _json("feature_ablation.json") if r.get("task") == task]
    if not rows:
        raise KeyError(f"feature_ablation: no rows for {task}")
    return float(rows[0]["baseline_score"])


def _ablation_min_delta(task: str) -> float:
    rows = [r for r in _json("feature_ablation.json") if r.get("task") == task]
    return float(min(r.get("delta", 0.0) for r in rows))


def _ablation_min_ablated(task: str) -> float:
    """Ablated score of the most-impactful (largest-drop) category for a task."""
    rows = [r for r in _json("feature_ablation.json") if r.get("task") == task]
    return float(min(rows, key=lambda r: r.get("delta", 0.0))["ablated_score"])


def _detonly(side: str, key: str) -> float:
    """Headline detection-only ablation (T1 entry with the largest all-sources AUPRC)."""
    rows = [
        e
        for e in _json("detection_only_ablation.json")
        if isinstance(e, dict) and e.get("task") == "T1" and "all_sources" in e
    ]
    e = max(rows, key=lambda r: r["all_sources"].get("auprc", 0.0))
    return float(e[side][key])


def _ext(source: str, key: str) -> float:
    e = _json("external_validation.json")[source]
    return float(e["metrics"][key] if key in ("auroc", "auprc") else e[key])


def _ext_ca(key: str) -> float:
    return _ext("ca_geotracker", key)


def _group_err(group: str, side: str, key: str) -> float:
    er = _json("group_error_calibration.json")["error_rates"][group][side]
    return float(er[key])


def _mon_ineq(group: str, key: str) -> float:
    return float(_json("monitoring_inequity.json")[group][key])


def _src_adj(section: str, key: str) -> float:
    return float(_json("source_adjusted_monitoring.json")[section][key])


def _recov(which: str, probe: str = "logistic") -> float:
    """ICP recoverability probe AUROC from representation vs raw features (logistic or mlp)."""
    p = _json("icp_recoverability.json")["probes"][probe]
    return float(
        p["auroc_from_icp_representation"] if which == "rep" else p["auroc_from_raw_features"]
    )


def _dml_adjusted() -> float:
    return float(_json("deconfounded_auroc.json")["deconfounded_auroc"])


def _t4_loro(name: str) -> float:
    t4 = _json(name).get("T4", {})
    for m in ("xgboost", "random_forest", "catboost"):
        if isinstance(t4.get(m), dict) and "mean_auroc" in t4[m]:
            return float(t4[m]["mean_auroc"])
    raise KeyError(f"{name}: T4 mean_auroc not found")


def _t4_loro_std(name: str) -> float:
    """T4 LORO with-provenance across-fold s.d. (loro_cv.json)."""
    return float(_json(name)["T4"]["xgboost"]["std_auroc"])


def _loro_pf_region(extremum: str) -> float:
    folds = _json("loro_cv_provenance_free.json")["T1"]["xgboost"]["folds"]
    aurocs = [f["auroc"] for f in folds]
    return float(min(aurocs) if extremum == "min" else max(aurocs))


def _mn_loro(key: str = "auroc", name: str = "loro_cv.json") -> float:
    """Minnesota-specific held-out generalization metric.

    The leave-region-5-out LORO fold trains on every other region (so its model
    never sees any Minnesota system) and is tested on region 5. This reads that
    fold's per-system OOF predictions (xgboost, the LORO-reported model), subsets
    to MN PWSIDs, and returns the leak-free AUROC or AUPRC.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    folds = _json(name)["T1"]["xgboost"]["folds"]
    fold = next(f for f in folds if f.get("test_region") == 5 and "oof_predictions" in f)
    oof = fold["oof_predictions"]
    keep = [str(p).startswith("MN") for p in oof["pwsid"]]
    y_true = [v for v, k in zip(oof["y_true"], keep, strict=True) if k]
    y_prob = [v for v, k in zip(oof["y_prob"], keep, strict=True) if k]
    scorer = roc_auc_score if key == "auroc" else average_precision_score
    return float(scorer(y_true, y_prob))


def _sens_robust_count() -> float:
    """Features robust through Rosenbaum Gamma=3.0 (matches verify_paper's count)."""
    sb = _json("sensitivity_bounds.json")
    feats = {k: v for k, v in sb.items() if isinstance(v, dict) and "bounds_table" in v}
    n = 0
    for v in feats.values():
        hi = [r for r in v.get("bounds_table", []) if r.get("gamma", 0) >= 2.999]
        if hi and all(r.get("significant") for r in hi):
            n += 1
    return float(n)


def _split_comp(split: str, key: str) -> float:
    sc = _json("split_comparison.json")["simple"]
    return float(sc[f"{split}_split_metrics"][key])


def _split_inflation(key: str) -> float:
    return float(_json("split_comparison.json")["simple"]["inflation_ratio"][key])


def _demo_rank(feature: str, key: str) -> int:
    """A demographic feature's rank entry from dml_shap_rank_comparison.json."""
    for e in _json("dml_shap_rank_comparison.json").get("demographic_ranks", []):
        if e.get("feature") == feature:
            return int(e[key])
    raise KeyError(f"{feature} not in demographic_ranks")


def _western_burden(key: str = "burden_ratio") -> float:
    """People-of-color burden ratio for the western test region (EPA Region 9)."""
    for e in _json("loro_equity_analysis.json").get("per_region", []):
        if e.get("group") == "pct_people_of_color" and str(e.get("region")) == "9":
            return float(e[key])
    raise KeyError("region-9 POC burden not found")


def _equity_poc(key: str) -> float:
    """People-of-color burden from the geographic test set (equity_analysis.json)."""
    for e in _json("equity_analysis.json"):
        if e.get("group") == "pct_people_of_color":
            return float(e[key])
    raise KeyError("pct_people_of_color not found in equity_analysis")


def _areal_poc(scale: str) -> float:
    a = _json("areal_apportionment.json")
    if scale == "single":
        return float(a["baseline_single_nearest"]["pct_people_of_color"])
    return float(a["apportioned"][scale]["pct_people_of_color"])


def _dataset_snapshot() -> dict[str, Any]:
    return json.loads((SNAPSHOTS / "dataset_snapshot.json").read_text(encoding="utf-8"))


def _merged_count(kind: str) -> int:
    """Reader-facing dataset totals from the COMMITTED snapshot (not the parquet).

    ``paper/frozen_snapshots/dataset_snapshot.json`` was verified byte-equal to
    ``data/interim/merged_wq.parquet`` (see the snapshot README) but is committed,
    so the gate is CI-safe and no longer skips ``ds_*`` when the git-ignored parquet
    is absent. Return semantics match the previous parquet implementation.
    """
    snap = _dataset_snapshot()
    if kind in ("samples", "systems", "geocoded"):
        return int(snap["totals"][kind])
    if kind.startswith("src:"):
        return int(snap["sources"][kind.split(":", 1)[1]]["records"])
    raise ValueError(kind)


def _ds_source(source: str, field: str) -> float:
    """A per-source Table-1 value (records / pfas_analytes / censoring_pct) from the snapshot."""
    v = _dataset_snapshot()["sources"][source][field]
    if v is None:
        raise KeyError(f"dataset_snapshot: sources.{source}.{field} is null")
    return float(v)


# ---- referee-report Phase-2 loaders (all read verified frozen paths) --------------------
def _shap(feature: str) -> float:
    """T1 XGBoost mean |SHAP| for a feature (shap_T1.json)."""
    return float(_json("shap_T1.json")["mean_abs_shap"][feature])


def _conf(which: str) -> float:
    """Group-conformal T1 coverage / set-size / region n (group_conformal_results.json[0])."""
    e = _json("group_conformal_results.json")
    e = e[0] if isinstance(e, list) else e
    pr = e["per_region"]
    if which == "marginal":
        return float(e["coverage"])
    if which == "marginal_set":
        return float(e["avg_set_size"])
    if which == "marginal_n":
        return float(sum(pr[r]["n"] for r in pr))
    if which == "rmax":
        return float(max(pr[r]["coverage"] for r in pr))
    kind, region = which[0], which[1:]  # r=coverage, n=count, s=avg set size
    field = {"r": "coverage", "n": "n", "s": "avg_set_size"}[kind]
    return float(pr[region][field])


def _dml_feat(feature: str, key: str) -> float:
    """A causal_deconfounding.json per-feature value (causal_effect / p_value_fdr)."""
    row = next(r for r in _json("causal_deconfounding.json") if r.get("feature") == feature)
    return float(row[key])


def _dml_sig_count() -> float:
    return float(sum(1 for r in _json("causal_deconfounding.json") if r.get("significant_fdr")))


def _dml_total() -> float:
    return float(len(_json("causal_deconfounding.json")))


def _sens_gamma(feature: str) -> float:
    """Rosenbaum gamma* tipping point (sensitivity_bounds.json); raises if robust (None)."""
    v = _json("sensitivity_bounds.json")[feature].get("gamma_star")
    if v is None:
        raise KeyError(f"sensitivity_bounds: {feature} gamma_star is None (robust)")
    return float(v)


def _split_matrix(model: str, feature_set: str, split: str) -> float:
    """A split_comparison.json matrix cell AUPRC (model_name/feature_set/split_strategy)."""
    for r in _json("split_comparison.json")["matrix"]["results"]:
        if (
            r["model_name"] == model
            and r["feature_set"] == feature_set
            and r["split_strategy"] == split
        ):
            return float(r["metrics"]["auprc"])
    raise KeyError(f"split_comparison matrix: {model}/{feature_set}/{split}")


def _inf(path: str) -> float:
    """A dotted-path value from inference_strengthening.json."""
    d: Any = _json("inference_strengthening.json")
    for p in path.split("."):
        d = d[p]
    return float(d)


def _leak_delta(model: str, feature_set: str) -> float:
    """Random-vs-geographic AUPRC inflation (%) for a split_comparison matrix cell."""
    r = _split_matrix(model, feature_set, "random")
    g = _split_matrix(model, feature_set, "geographic")
    return float(round((r - g) / g * 100))


def _power(test: str, key: str) -> float:
    """A post-hoc power core-test value (power_analysis.json)."""
    return float(_json("power_analysis.json")["core_tests"][test][key])


def _valreuse(path: str) -> float:
    """A dotted-path value from val_reuse_bias.json (S27 decomposition)."""
    d: Any = _json("val_reuse_bias.json")
    for p in path.split("."):
        d = d[p]
    return float(d)


def _coord_5km(key: str) -> float:
    """5-km coordinate-perturbation delta (abs) from coordinate_sensitivity.json."""
    for s in _json("coordinate_sensitivity.json")["summary"]:
        if s["magnitude_km"] == 5.0:
            return float(abs(s[key]))
    raise KeyError("coordinate_sensitivity: 5km summary")


def _conf_marginal_pct() -> float:
    """Group-conformal marginal coverage as a percentage (group_conformal_results.json)."""
    e = _json("group_conformal_results.json")
    e = e[0] if isinstance(e, list) else e
    return float(e["coverage"] * 100)


def _arsenic(path: str) -> float:
    """A dotted-path model metric from t6_arsenic.json['models']."""
    d: Any = _json("t6_arsenic.json")["models"]
    for p in path.split("."):
        d = d[p]
    return float(d)


def _dmlrank(feature: str, key: str) -> float:
    """A DML-vs-SHAP rank/displacement for a feature (dml_shap_rank_comparison.json)."""
    for r in _json("dml_shap_rank_comparison.json")["feature_ranks"]:
        if r.get("feature") == feature:
            return float(abs(r[key]) if key == "displacement" else r[key])
    raise KeyError(f"dml_shap_rank_comparison: {feature}")


def _dml_sig_p01() -> float:
    """Count of DML coefficients significant at p_FDR < 0.01 (causal_deconfounding.json)."""
    return float(
        sum(1 for r in _json("causal_deconfounding.json") if r.get("p_value_fdr", 9) < 0.01)
    )


def _mss(task: str, model: str, key: str) -> float:
    """A multi-seed stability value (multi_seed_stability.json)."""
    return float(_json("multi_seed_stability.json")[task][model][key])


def _seedinf(key: str) -> float:
    """A seed-inflation summary value (seed_inflation_check.json)."""
    return float(_json("seed_inflation_check.json")["summary"][key])


def _reghet_ks(feature: str) -> float:
    """Region-heterogeneity KS statistic for a named feature (region_heterogeneity.json)."""
    for f in _json("region_heterogeneity.json")["feature_distributions"]:
        if f.get("feature") == feature:
            return float(f["ks_statistic"])
    raise KeyError(f"region_heterogeneity: {feature}")


def _bootci(task: str, model: str, metric: str, bound: str) -> float:
    """A bootstrap CI bound (bootstrap_ci.json)."""
    for x in _json("bootstrap_ci.json"):
        if x.get("task") == task and x.get("model") == model and x.get("metric") == metric:
            return float(x[f"ci_{bound}"])
    raise KeyError(f"bootstrap_ci: {task}/{model}/{metric}")


def _lift(key: str) -> float:
    """A monitoring-free lift value (lift_analysis.json)."""
    return float(_json("lift_analysis.json")["monitoring_free"][key])


def _sbb(path: str) -> float:
    """A dotted-path value from spatial_block_bootstrap.json."""
    d: Any = _json("spatial_block_bootstrap.json")
    for p in path.split("."):
        d = d[p]
    return float(d)


def _spatial_xgb(km: float) -> float:
    """T1 XGBoost residual Moran's I at a distance threshold (spatial_autocorrelation.json)."""
    x = next(
        r for r in _json("spatial_autocorrelation.json") if r["model"] == "xgboost_classifier"
    )
    if km == 50:
        return float(x["morans_i"])
    return float(next(t["statistic"] for t in x["multi_threshold"] if t["threshold_km"] == km))


def _sens(key: str) -> float:
    """Preprocessing-sensitivity invariant value (sensitivity_analysis.json first threshold)."""
    return float(_json("sensitivity_analysis.json")["drop_na_threshold"][0][key])


def _calib_ece(model: str) -> float:
    """Pre-calibration ECE for a T1 model (calibration_analysis.json)."""
    for r in _json("calibration_analysis.json"):
        if r.get("task") == "T1" and r.get("model") == model:
            return float(r["ece"])
    raise KeyError(f"calibration_analysis: T1/{model}")


def _cnn1d(key: str) -> float:
    """A top-level cnn1d_feature_ordering.json value (auroc_mean/std, auprc_mean/std)."""
    return float(_json("cnn1d_feature_ordering.json")[key])


def _cnn1d_ordering(ordering: str, key: str = "auroc") -> float:
    """Per-ordering CNN1D metric (alphabetical/domain) from cnn1d_feature_ordering.json."""
    for r in _json("cnn1d_feature_ordering.json")["results"]:
        if r.get("ordering") == ordering:
            return float(r[key])
    raise KeyError(f"cnn1d ordering {ordering}")


def _conf_model(model: str, key: str = "coverage") -> float:
    """Per-model conformal metric at alpha=0.05 (conformal_results.json)."""
    for r in _json("conformal_results.json"):
        if (
            r.get("task") == "T1"
            and r.get("model") == model
            and abs(r.get("alpha", 0) - 0.05) < 1e-6
        ):
            return float(r[key])
    raise KeyError(f"conformal_results: T1/{model}@0.05")


def _mn_asc(path: str) -> float:
    """A value (dotted path) from mn_ascertainment.json."""
    d: Any = _json("mn_ascertainment.json")
    for p in path.split("."):
        d = d[p]
    return float(d)


def _loro_std(name: str) -> float:
    """T1 LORO with-provenance across-fold s.d. (loro_cv.json)."""
    return float(_json(name)["T1"]["xgboost"]["std_auroc"])


def _ext_ci(source: str, bound: str) -> float:
    """External-validation AUROC bootstrap-CI bound (external_validation.json)."""
    return float(_json("external_validation.json")[source]["bootstrap_ci"]["auroc"][f"ci_{bound}"])


def _dml_rank(key: str) -> float:
    """DML-vs-SHAP rank-comparison stat (dml_shap_rank_comparison.json)."""
    return float(_json("dml_shap_rank_comparison.json")[key])


def _dml_original() -> float:
    return float(_json("deconfounded_auroc.json")["original_auroc"])


def _ej_burden_max() -> float:
    return float(
        _json("loro_equity_analysis.json")["summary"]["pct_people_of_color"]["max_burden_ratio"]
    )


def _fmt3(v: float) -> str:
    return f"{v:.3f}"


def _fmt2(v: float) -> str:
    return f"{v:.2f}"


def _fmt1(v: float) -> str:
    return f"{v:.1f}"


def _fmt3m(v: float) -> str:
    """3-decimal with a unicode minus sign (U+2212) to match paper style."""
    return f"{v:.3f}".replace("-", "−")  # noqa: RUF001


def _fmt0(v: float) -> str:
    return f"{round(v)}"


def _commas(v: float) -> str:
    return f"{round(v):,}"


#: id -> (description, source callable -> value, formatter).
REGISTRY: dict[str, tuple[str, Callable[[], float], Callable[[float], str]]] = {
    # --- T1 (PFAS detection) ---
    "t1_full_auroc": (
        "T1 XGBoost full AUROC",
        lambda: _metric("T1", "xgboost_classifier", "auroc"),
        _fmt3,
    ),
    "t1_full_auprc": (
        "T1 XGBoost full AUPRC",
        lambda: _metric("T1", "xgboost_classifier", "auprc"),
        _fmt3,
    ),
    "t1_pf_auroc": (
        "T1 XGBoost provenance-free AUROC",
        lambda: _metric("T1", "xgboost_classifier", "auroc", "results_provenance_free.json"),
        _fmt3,
    ),
    "t1_pf_auprc": (
        "T1 XGBoost provenance-free AUPRC",
        lambda: _metric("T1", "xgboost_classifier", "auprc", "results_provenance_free.json"),
        _fmt3,
    ),
    "t1_icp_auroc": ("T1 ICP AUROC", lambda: _metric("T1", "icp_classifier", "auroc"), _fmt3),
    "t1_best_auroc": ("T1 best AUROC", lambda: _best("T1", "auroc"), _fmt3),
    "t1_best_auprc": ("T1 best AUPRC", lambda: _best("T1", "auprc"), _fmt3),
    # --- T4 (lead action-level) ---
    "t4_full_auroc": (
        "T4 XGBoost full AUROC",
        lambda: _metric("T4", "xgboost_classifier", "auroc"),
        _fmt3,
    ),
    "t4_full_auprc": (
        "T4 XGBoost full AUPRC",
        lambda: _metric("T4", "xgboost_classifier", "auprc"),
        _fmt3,
    ),
    "t4_pf_auroc": (
        "T4 XGBoost provenance-free AUROC",
        lambda: _metric("T4", "xgboost_classifier", "auroc", "results_provenance_free.json"),
        _fmt3,
    ),
    "t4_icp_auroc": ("T4 ICP AUROC", lambda: _metric("T4", "icp_classifier", "auroc"), _fmt3),
    "t4_best_auroc": ("T4 best AUROC", lambda: _best("T4", "auroc"), _fmt3),
    "t4_best_auprc": ("T4 best AUPRC", lambda: _best("T4", "auprc"), _fmt3),
    # --- T2 ---
    "t2_best_rmse": ("T2 best RMSE", lambda: _best("T2", "rmse", lowest=True), _fmt3),
    # --- T3/T5/T7 secondary-task headline AUROCs (results.json, xgboost) ---
    "t3_micro_auroc": (
        "T3 XGBoost micro-AUROC",
        lambda: _metric("T3", "xgboost_classifier", "micro_auroc"),
        _fmt3,
    ),
    "dt_t1_auroc": (
        "T1 Deep Tobit AUROC",
        lambda: _metric("T1", "deep_tobit_classifier", "auroc"),
        _fmt3,
    ),
    "dt_t5_auroc": (
        "T5 Deep Tobit transfer AUROC",
        lambda: _metric("T5", "deep_tobit_classifier", "auroc"),
        _fmt3,
    ),
    "t5_best_auroc": (
        "highest T5 transfer AUROC (any model)",
        lambda: _best("T5", "auroc"),
        _fmt3,
    ),
    "vr_r8_holdout": (
        "per-region holdout AUROC R8",
        lambda: _valreuse("per_region_comparison.8.holdout_auroc"),
        _fmt3,
    ),
    "vr_r9_holdout": (
        "per-region holdout AUROC R9",
        lambda: _valreuse("per_region_comparison.9.holdout_auroc"),
        _fmt3,
    ),
    "vr_r10_holdout": (
        "per-region holdout AUROC R10",
        lambda: _valreuse("per_region_comparison.10.holdout_auroc"),
        _fmt3,
    ),
    "gnn_gcn_t1": ("T1 GCN AUROC", lambda: _metric("T1", "gnn_gcn_classifier", "auroc"), _fmt3),
    "gnn_sage_t1": (
        "T1 GraphSAGE AUROC",
        lambda: _metric("T1", "gnn_sage_classifier", "auroc"),
        _fmt3,
    ),
    "t5_auroc": ("T5 XGBoost AUROC", lambda: _metric("T5", "xgboost_classifier", "auroc"), _fmt3),
    "t7_auroc": ("T7 XGBoost AUROC", lambda: _metric("T7", "xgboost_classifier", "auroc"), _fmt3),
    "t4_ci_lo": (
        "T4 XGBoost AUROC CI lower",
        lambda: _bootci("T4", "xgboost_classifier", "auroc", "lower"),
        _fmt3,
    ),
    "t4_ci_hi": (
        "T4 XGBoost AUROC CI upper",
        lambda: _bootci("T4", "xgboost_classifier", "auroc", "upper"),
        _fmt3,
    ),
    "t4_loro_sd": (
        "T4 LORO with-provenance across-fold s.d.",
        lambda: _t4_loro_std("loro_cv.json"),
        _fmt3,
    ),
    "lift_top_decile": (
        "monitoring-free top-decile lift",
        lambda: _lift("top_decile_lift"),
        _fmt1,
    ),
    # --- LORO ---
    "loro_full_auroc": (
        "T1 LORO with-provenance mean AUROC",
        lambda: _loro_mean("loro_cv.json"),
        _fmt3,
    ),
    "loro_pf_auroc": (
        # m1: report the block-bootstrap mean so the point estimate and its CI are one
        # internally-consistent pair (inference_strengthening.json); the fold-mean 0.694 and
        # the bootstrap CI 0.610-0.781 came from different computations.
        "T1 LORO provenance-free block-bootstrap mean AUROC",
        lambda: _inf("m4a_provenance_free_loro_ci.block_bootstrap.mean"),
        _fmt3,
    ),
    "loro_pf_ci_lo": (
        "T1 prov-free LORO block-bootstrap CI lower",
        lambda: _inf("m4a_provenance_free_loro_ci.block_bootstrap.ci_lower"),
        _fmt3,
    ),
    "loro_pf_ci_hi": (
        "T1 prov-free LORO block-bootstrap CI upper",
        lambda: _inf("m4a_provenance_free_loro_ci.block_bootstrap.ci_upper"),
        _fmt3,
    ),
    # --- Ablation ---
    "t1_abl_baseline_auprc": (
        "T1 ablation baseline AUPRC",
        lambda: _ablation_baseline("T1"),
        _fmt3,
    ),
    "t4_abl_baseline_auprc": (
        "T4 ablation baseline AUPRC",
        lambda: _ablation_baseline("T4"),
        _fmt3,
    ),
    "t4_abl_monitoring_delta": (
        "T4 monitoring-ablation min ΔAUPRC",
        lambda: _ablation_min_delta("T4"),
        _fmt3m,
    ),
    "t4_abl_monitoring_ablated": (
        "T4 AUPRC after removing monitoring features",
        lambda: _ablation_min_ablated("T4"),
        _fmt3,
    ),
    # --- Detection-only ablation (T1 headline) ---
    "t1_detonly_all_auprc": (
        "T1 AUPRC with detection-only sources",
        lambda: _detonly("all_sources", "auprc"),
        _fmt3,
    ),
    "t1_detonly_excl_auprc": (
        "T1 AUPRC without detection-only sources",
        lambda: _detonly("excluding_detection_only", "auprc"),
        _fmt3,
    ),
    "t1_detonly_all_auroc": (
        "T1 AUROC with detection-only sources",
        lambda: _detonly("all_sources", "auroc"),
        _fmt3,
    ),
    "t1_detonly_excl_auroc": (
        "T1 AUROC without detection-only sources",
        lambda: _detonly("excluding_detection_only", "auroc"),
        _fmt3,
    ),
    # --- External validation (CA) ---
    "ca_ext_auroc": ("CA GeoTracker external AUROC", lambda: _ext_ca("auroc"), _fmt3),
    "ca_ext_auprc": ("CA GeoTracker external AUPRC", lambda: _ext_ca("auprc"), _fmt3),
    "ca_ext_n": ("CA GeoTracker external n systems", lambda: float(_ext_ca("n_systems")), _commas),
    "ej_burden_test": (
        "POC burden, geographic test set",
        lambda: _equity_poc("burden_ratio"),
        _fmt2,
    ),
    "ej_burden_test_ipw": (
        "POC burden, test set, IPW-adjusted",
        lambda: _equity_poc("burden_ratio_ipw_adjusted"),
        _fmt2,
    ),
    "nj_ext_auroc": ("NJ DEP external AUROC", lambda: _ext("nj_dep", "auroc"), _fmt3),
    "nj_ext_n": ("NJ DEP external n systems", lambda: float(_ext("nj_dep", "n_systems")), _commas),
    "mo_ext_auroc": ("MO DNR external AUROC", lambda: _ext("mo_dnr", "auroc"), _fmt3),
    "mo_ext_n": ("MO DNR external n systems", lambda: float(_ext("mo_dnr", "n_systems")), _commas),
    # external-validation AUPRC / detection-rate / AUROC-CI (Extended Data Table 5 panels)
    "nj_ext_auprc": ("NJ DEP external AUPRC", lambda: _ext("nj_dep", "auprc"), _fmt3),
    "nj_ext_det": ("NJ DEP detection rate", lambda: _ext("nj_dep", "detection_rate"), _fmt3),
    "nj_ext_ci_lo": ("NJ DEP AUROC CI lower", lambda: _ext_ci("nj_dep", "lower"), _fmt3),
    "nj_ext_ci_hi": ("NJ DEP AUROC CI upper", lambda: _ext_ci("nj_dep", "upper"), _fmt3),
    "mo_ext_auprc": ("MO DNR external AUPRC", lambda: _ext("mo_dnr", "auprc"), _fmt3),
    "mo_ext_det": ("MO DNR detection rate", lambda: _ext("mo_dnr", "detection_rate"), _fmt3),
    "mo_ext_ci_lo": ("MO DNR AUROC CI lower", lambda: _ext_ci("mo_dnr", "lower"), _fmt3),
    "mo_ext_ci_hi": ("MO DNR AUROC CI upper", lambda: _ext_ci("mo_dnr", "upper"), _fmt3),
    "ca_ext_det": (
        "CA GeoTracker detection rate",
        lambda: _ext("ca_geotracker", "detection_rate"),
        _fmt3,
    ),
    # DML-vs-SHAP rank comparison (dml_shap_rank_comparison.json)
    "dml_spearman_rho": ("DML-SHAP Spearman rho", lambda: _dml_rank("spearman_rho"), _fmt2),
    "dml_spearman_p": ("DML-SHAP Spearman p", lambda: _dml_rank("spearman_p"), _fmt3),
    "dml_spearman_n": ("DML-SHAP ranked feature count", lambda: _dml_rank("n_features"), _commas),
    # --- M4 group error / calibration (people of color) ---
    "ej_fnr_high": (
        "POC high-share FNR",
        lambda: _group_err("pct_people_of_color", "high", "fnr"),
        _fmt2,
    ),
    "ej_fnr_low": (
        "POC low-share FNR",
        lambda: _group_err("pct_people_of_color", "low", "fnr"),
        _fmt2,
    ),
    # --- R5 M1 group-gap significance test (group_gap_tests.json, C10) ---
    "ej_fnr_pval": (
        "POC FNR-gap two-proportion p-value",
        lambda: _json("group_gap_tests.json")["pct_people_of_color"]["fnr_gap_test"]["p_value"],
        _fmt2,
    ),
    "ej_auroc_pval": (
        "POC AUROC-gap two-proportion p-value (high vs low share)",
        lambda: _json("group_gap_tests.json")["pct_people_of_color"]["auroc_gap_test"]["p_value"],
        _fmt3,
    ),
    "ej_cov_high": (
        "POC high-share conformal coverage (group_gap_tests)",
        lambda: _json("group_gap_tests.json")["pct_people_of_color"]["coverage_gap_test"][
            "p_high"
        ],
        _fmt3,
    ),
    "ej_cov_low": (
        "POC low-share conformal coverage (group_gap_tests)",
        lambda: _json("group_gap_tests.json")["pct_people_of_color"]["coverage_gap_test"]["p_low"],
        _fmt3,
    ),
    "ej_cov_pval": (
        "POC coverage-gap two-proportion p-value (favouring high share)",
        lambda: _json("group_gap_tests.json")["pct_people_of_color"]["coverage_gap_test"][
            "p_value"
        ],
        _fmt3,
    ),
    # --- R5 M4 headline seed stability (headline_harness_stability.json, C11) ---
    "hh_mean": (
        "T1 headline 5-seed mean AUROC on its own harness",
        lambda: _json("headline_harness_stability.json")["mean_auroc"],
        _fmt3,
    ),
    "hh_sd": (
        "T1 headline 5-seed AUROC s.d. on its own harness",
        lambda: _json("headline_harness_stability.json")["std_auroc"],
        _fmt3,
    ),
    "hh_streamlined": (
        "T1 xgboost AUROC on the streamlined robustness harness (reconciliation)",
        lambda: next(
            r["value"]
            for r in _json("headline_harness_stability.json")["reconciliation"]
            if "streamlined" in r["harness"]
        ),
        _fmt3,
    ),
    "hh_splitcomp": (
        "T1 xgboost full-feature geographic AUROC on the split-comparison harness",
        lambda: next(
            r["value"]
            for r in _json("headline_harness_stability.json")["reconciliation"]
            if r["harness"] == "split-comparison"
        ),
        _fmt3,
    ),
    # --- R5 M6 common-reporting-limit AUPRC + re-censored count (common_rl_sensitivity.json) ---
    "crl_base_auprc": (
        "T1 default-xgboost AUPRC, un-recensored baseline (C2/M6a)",
        lambda: _json("common_rl_sensitivity.json")["baseline"]["auprc"],
        _fmt3,
    ),
    "crl_common_auprc": (
        "T1 AUPRC after re-censoring to the common reporting limit (C2/M6a)",
        lambda: _json("common_rl_sensitivity.json")["common_rl"]["auprc"],
        _fmt3,
    ),
    "crl_recensored": (
        "PFOS detections reclassified as non-detect under the common reporting limit",
        lambda: _json("common_rl_sensitivity.json")["_meta"]["n_pfos_detections_recensored"],
        _commas,
    ),
    # --- R5 M7 MCL-exceedance vs detection deployment metrics (mcl_lift.json, C14) ---
    "mcl_exc_auroc": (
        "MCL-exceedance target test AUROC (C14)",
        lambda: _json("mcl_lift.json")["mcl_exceedance"]["auroc"],
        _fmt3,
    ),
    "mcl_exc_lift": (
        "MCL-exceedance target top-decile lift (C14)",
        lambda: _json("mcl_lift.json")["mcl_exceedance"]["top_decile_lift"],
        _fmt2,
    ),
    "mcl_det_auroc": (
        "Detection target test AUROC on the MCL-lift harness (C14)",
        lambda: _json("mcl_lift.json")["detection"]["auroc"],
        _fmt3,
    ),
    "mcl_det_lift": (
        "Detection target top-decile lift on the MCL-lift harness (C14)",
        lambda: _json("mcl_lift.json")["detection"]["top_decile_lift"],
        _fmt2,
    ),
    "mcl_exc_auprc": (
        "MCL-exceedance target test AUPRC (mcl_exceedance_analysis.json)",
        lambda: _json("mcl_exceedance_analysis.json")["mcl_exceedance"]["metrics"]["auprc"],
        _fmt3,
    ),
    # --- R5 m9 PFNA-only MCL exceeders (pfna_exceedance_count.json, C17) ---
    "pfna_only": (
        "PFNA-only MCL exceeders (exceed no other regulated analyte)",
        lambda: _json("pfna_exceedance_count.json")["n_pfna_only_exceeders"],
        _fmt0,
    ),
    "pfna_total": (
        "Systems exceeding the PFNA individual MCL",
        lambda: _json("pfna_exceedance_count.json")["n_pfna_exceeders"],
        _fmt0,
    ),
    "mcl_union_exceeders": (
        "Systems in the default MCL-exceedance union",
        lambda: _json("pfna_exceedance_count.json")["union_exceeders"],
        _commas,
    ),
    # --- R5 m2 equal-mass subgroup ECE with bootstrap CIs (group_ece_debiased.json, C15) ---
    "ece_mass_high": (
        "PoC high-share equal-mass ECE",
        lambda: _json("group_ece_debiased.json")["pct_people_of_color"]["high"]["ece_equal_mass"],
        _fmt3,
    ),
    "ece_mass_high_lo": (
        "PoC high-share equal-mass ECE CI lower",
        lambda: _json("group_ece_debiased.json")["pct_people_of_color"]["high"][
            "ece_equal_mass_ci"
        ][0],
        _fmt3,
    ),
    "ece_mass_high_hi": (
        "PoC high-share equal-mass ECE CI upper",
        lambda: _json("group_ece_debiased.json")["pct_people_of_color"]["high"][
            "ece_equal_mass_ci"
        ][1],
        _fmt3,
    ),
    "ece_mass_low": (
        "PoC low-share equal-mass ECE",
        lambda: _json("group_ece_debiased.json")["pct_people_of_color"]["low"]["ece_equal_mass"],
        _fmt3,
    ),
    "ece_mass_low_lo": (
        "PoC low-share equal-mass ECE CI lower",
        lambda: _json("group_ece_debiased.json")["pct_people_of_color"]["low"][
            "ece_equal_mass_ci"
        ][0],
        _fmt3,
    ),
    "ece_mass_low_hi": (
        "PoC low-share equal-mass ECE CI upper",
        lambda: _json("group_ece_debiased.json")["pct_people_of_color"]["low"][
            "ece_equal_mass_ci"
        ][1],
        _fmt3,
    ),
    # --- R5 m10 excluded unknown-demographics group (group_burden_ci.json, C19) ---
    "unknown_auroc": (
        "Unknown-demographics group T1 AUROC (excluded from equity analysis)",
        lambda: _json("group_burden_ci.json")["unknown_group"]["auroc"],
        _fmt3,
    ),
    "unknown_n": (
        "Unknown-demographics group test-set count",
        lambda: _json("group_burden_ci.json")["unknown_group"]["n_samples"],
        _commas,
    ),
    "unknown_pct": (
        "Unknown-demographics group share of the test set (%)",
        lambda: _json("group_burden_ci.json")["unknown_group"]["fraction_of_test"] * 100,
        _fmt1,
    ),
    # --- R5 M3 T1 LORO leaderboard means (loro_cv_full.json, C18) ---
    "loro_t1_catboost": (
        "T1 LORO mean AUROC, CatBoost (LORO leader)",
        lambda: _json("loro_cv_full.json")["T1"]["catboost"]["mean_auroc"],
        _fmt3,
    ),
    "loro_t1_voting": (
        "T1 LORO mean AUROC, Voting ensemble",
        lambda: _json("loro_cv_full.json")["T1"]["voting_ensemble"]["mean_auroc"],
        _fmt3,
    ),
    "loro_t1_xgboost": (
        "T1 LORO mean AUROC, XGBoost (fixed-split leader, LORO third)",
        lambda: _json("loro_cv_full.json")["T1"]["xgboost"]["mean_auroc"],
        _fmt3,
    ),
    "loro_t1_gnn_gcn": (
        "T1 LORO mean AUROC, GCN (unconditional linear-fallback inference)",
        lambda: _json("loro_cv_full.json")["T1"]["gnn_gcn"]["mean_auroc"],
        _fmt3,
    ),
    "loro_t1_gnn_sage": (
        "T1 LORO mean AUROC, GraphSAGE (unconditional linear-fallback inference)",
        lambda: _json("loro_cv_full.json")["T1"]["gnn_sage"]["mean_auroc"],
        _fmt3,
    ),
    "loro_t1_icp": (
        "T1 LORO mean AUROC, ICP (monitoring-invariant model)",
        lambda: _json("loro_cv_full.json")["T1"]["icp"]["mean_auroc"],
        _fmt3,
    ),
    # --- R6 D1 region-10 context (why the fixed West-only 0.864 is the upper end) ---
    "r10_loro_auroc": (
        "Region-10 held-out AUROC (T1 XGBoost LORO fold; the most separable region)",
        lambda: next(
            f["auroc"]
            for f in _json("loro_cv_full.json")["T1"]["xgboost"]["folds"]
            if f["test_region"] == 10
        ),
        _fmt3,
    ),
    "r10_det_rate": (
        "Region-10 PFAS detection rate (highest of the ten EPA regions)",
        lambda: _json("region_heterogeneity.json")["detection_rates"]["10"]["detection_rate"],
        _fmt3,
    ),
    # --- R6 R1-1 ambient-source population asymmetry (headline excludes ambient;
    #     LORO / split-comparison include it) ---
    "n_headline_systems": (
        "Fixed-split benchmark population (ambient-excluded), sum of T1 train/val/test",
        lambda: (lambda md: float(md["n_train"] + md["n_val"] + md["n_test"]))(
            next(
                r["metadata"]
                for r in _json("results.json")
                if r.get("task") == "T1" and r.get("model") == "xgboost_classifier"
            )
        ),
        _commas,
    ),
    "n_ambient_incl_systems": (
        "LORO / split-comparison population (ambient-included), sum of T1 LORO fold n_test",
        lambda: float(
            sum(f["n_test"] for f in _json("loro_cv_full.json")["T1"]["xgboost"]["folds"])
        ),
        _commas,
    ),
    # --- R6 R5-3 WQP cross-source diagnostic (below-chance pooled AUROC, Simpson artifact) ---
    "wqp_pooled_auroc": (
        "WQP cross-source regional validation, pooled AUROC (below-chance Simpson artifact)",
        lambda: _json("wqp_regional_validation.json")["overall_metrics"]["auroc"],
        _fmt2,
    ),
    # --- R6 R2-1 DML first-stage strength (near-orthogonality of features to confounders) ---
    "dml_neg_r2_count": (
        "DML features with negative cross-fitted residualization R2 (near-orthogonal first stage)",
        lambda: float(
            sum(
                1 for r in _json("causal_deconfounding.json") if r.get("residualization_r2", 0) < 0
            )
        ),
        _fmt0,
    ),
    # --- R5 M5 cluster-robust t(9) recalibration (cluster_ci_calibration.json, C12) ---
    "loro_pf_t9_lo": (
        "PF LORO AUROC t(9) CI lower",
        lambda: _json("cluster_ci_calibration.json")["loro_auroc"]["t_gminus1_ci"][0],
        _fmt3,
    ),
    "loro_pf_t9_hi": (
        "PF LORO AUROC t(9) CI upper",
        lambda: _json("cluster_ci_calibration.json")["loro_auroc"]["t_gminus1_ci"][1],
        _fmt3,
    ),
    "ej_burden_t9_lo": (
        "PoC burden t(9) CI lower",
        lambda: _json("cluster_ci_calibration.json")["poc_burden"]["t_gminus1_ci"][0],
        _fmt2,
    ),
    "ej_burden_t9_hi": (
        "PoC burden t(9) CI upper",
        lambda: _json("cluster_ci_calibration.json")["poc_burden"]["t_gminus1_ci"][1],
        _fmt2,
    ),
    "ej_ece_high": (
        "POC high-share ECE",
        lambda: _group_err("pct_people_of_color", "high", "ece"),
        _fmt2,
    ),
    "ej_ece_low": (
        "POC low-share ECE",
        lambda: _group_err("pct_people_of_color", "low", "ece"),
        _fmt2,
    ),
    "ej_n_high": (
        "POC high-share n",
        lambda: _group_err("pct_people_of_color", "high", "n"),
        _commas,
    ),
    "ej_test_n": (
        "Equity/disparity geographically held-out test-set size (Regions 8/9/10)",
        lambda: _json("split_prevalence.json")["_meta"]["n_test_geographic"],
        _commas,
    ),
    "ej_n_low": (
        "POC low-share n",
        lambda: _group_err("pct_people_of_color", "low", "n"),
        _commas,
    ),
    "ej_auroc_high": (
        "POC high-share per-group AUROC",
        lambda: _group_err("pct_people_of_color", "high", "auroc"),
        _fmt3,
    ),
    "ej_auroc_low": (
        "POC low-share per-group AUROC",
        lambda: _group_err("pct_people_of_color", "low", "auroc"),
        _fmt3,
    ),
    # --- ICP recoverability probe / DML / T4 LORO / regional range ---
    "recov_rep": (
        "provenance recoverability from ICP representation",
        lambda: _recov("rep"),
        _fmt2,
    ),
    "recov_raw": ("provenance recoverability from raw features", lambda: _recov("raw"), _fmt2),
    "recov_rep_mlp": (
        "provenance recoverability from ICP rep (MLP probe)",
        lambda: _recov("rep", "mlp"),
        _fmt2,
    ),
    "recov_raw_mlp": (
        "provenance recoverability from raw (MLP probe)",
        lambda: _recov("raw", "mlp"),
        _fmt2,
    ),
    # --- S4 CNN1D feature-ordering sensitivity (cnn1d_feature_ordering.json) ---
    # S9 spatial: with-provenance LORO block-bootstrap CI + residual Moran's I
    "loro_full_ci_lo": (
        "T1 with-prov LORO block-bootstrap CI lower",
        lambda: _sbb("T1_xgboost.block_bootstrap.ci_lower"),
        _fmt3,
    ),
    "loro_full_ci_hi": (
        "T1 with-prov LORO block-bootstrap CI upper",
        lambda: _sbb("T1_xgboost.block_bootstrap.ci_upper"),
        _fmt3,
    ),
    "spatial_xgb_50": ("T1 XGBoost residual Moran's I @50km", lambda: _spatial_xgb(50), _fmt3),
    "spatial_xgb_200": ("T1 XGBoost residual Moran's I @200km", lambda: _spatial_xgb(200), _fmt3),
    # S11 multi-seed stability + seed inflation
    # S19 post-hoc statistical power (power_analysis.json core_tests)
    "pow_delong_es": (
        "DeLong power effect size",
        lambda: _power("delong_auroc", "effect_size"),
        _fmt3,
    ),
    "pow_delong_se": (
        "DeLong power pooled SE",
        lambda: _power("delong_auroc", "pooled_se"),
        _fmt3,
    ),
    "pow_delong": ("DeLong comparison power", lambda: _power("delong_auroc", "power"), _fmt2),
    "pow_ej_es": (
        "EJ burden power effect size",
        lambda: _power("ej_burden_ratio", "effect_size"),
        _fmt2,
    ),
    "pow_dml_es": (
        "DML coefficient power effect size",
        lambda: _power("dml_coefficient", "effect_size"),
        _fmt3,
    ),
    "pow_dml_se": ("DML coefficient power SE", lambda: _power("dml_coefficient", "se"), _fmt3),
    "pow_dml": ("DML coefficient power", lambda: _power("dml_coefficient", "power"), _fmt2),
    # S27 validation-reuse decomposition (val_reuse_bias.json)
    "vr_a_auroc": (
        "val-reuse A (triple-use) mean AUROC",
        lambda: _valreuse("conditions.triple_use.mean_auroc"),
        _fmt3,
    ),
    "vr_a_sd": ("val-reuse A s.d.", lambda: _valreuse("conditions.triple_use.std_auroc"), _fmt3),
    "vr_b_auroc": (
        "val-reuse B (internal-val) mean AUROC",
        lambda: _valreuse("conditions.internal_val.mean_auroc"),
        _fmt3,
    ),
    "vr_b_sd": ("val-reuse B s.d.", lambda: _valreuse("conditions.internal_val.std_auroc"), _fmt3),
    "vr_c_auroc": (
        "val-reuse C (no-ES) mean AUROC",
        lambda: _valreuse("conditions.no_es.mean_auroc"),
        _fmt3,
    ),
    "vr_c_sd": ("val-reuse C s.d.", lambda: _valreuse("conditions.no_es.std_auroc"), _fmt3),
    "vr_ac": (
        "val-reuse effect (A-C)",
        lambda: _valreuse("decomposition.val_reuse_effect"),
        _fmt3,
    ),
    "vr_ab": (
        "val-identity effect (A-B)",
        lambda: _valreuse("decomposition.val_identity_effect"),
        _fmt3,
    ),
    "vr_pct": (
        "pct of holdout-LORO gap from val reuse",
        lambda: _valreuse("decomposition.pct_explained_by_val_reuse"),
        _fmt1,
    ),
    "coord_5km_dauroc": (
        "5-km coordinate-perturbation AUROC change (abs)",
        lambda: _coord_5km("delta_auroc"),
        _fmt3,
    ),
    "conf_marginal_pct": ("group-conformal marginal coverage %", _conf_marginal_pct, _fmt1),
    "conf_marginal_cov": (
        "group-conformal marginal test-set coverage at alpha=0.05 (decimal)",
        lambda: (lambda e: (e[0] if isinstance(e, list) else e)["coverage"])(
            _json("group_conformal_results.json")
        ),
        _fmt3,
    ),
    # T6 arsenic geogenic-contaminant generalization (t6_arsenic.json)
    "arsenic_delta": (
        "arsenic RF random-vs-geo AUROC gap",
        lambda: _arsenic("random_forest_classifier.leakage_inflation_auroc"),
        _fmt2,
    ),
    "arsenic_rf_random": (
        "arsenic RF random-split AUROC",
        lambda: _arsenic("random_forest_classifier.random.public_test.auroc"),
        _fmt2,
    ),
    "arsenic_rf_geo": (
        "arsenic RF geographic AUROC",
        lambda: _arsenic("random_forest_classifier.geographic.public_test.auroc"),
        _fmt2,
    ),
    "arsenic_transfer": (
        "arsenic public->domestic transfer AUROC (max)",
        lambda: _arsenic("mlp_classifier.geographic.domestic_holdout.auroc"),
        _fmt2,
    ),
    # S14 DML-vs-SHAP rank displacements (dml_shap_rank_comparison.json)
    "dml_edu_shap": (
        "pct_less_hs_education SHAP rank",
        lambda: _dmlrank("pct_less_hs_education", "shap_rank"),
        _fmt0,
    ),
    "dml_edu_dml": (
        "pct_less_hs_education DML rank",
        lambda: _dmlrank("pct_less_hs_education", "dml_rank"),
        _fmt0,
    ),
    "dml_wetland_disp": (
        "pct_wetland_5km rank displacement (abs)",
        lambda: _dmlrank("pct_wetland_5km", "displacement"),
        _fmt0,
    ),
    "dml_sig_p01": ("DML coefficients significant at p_FDR<0.01", _dml_sig_p01, _fmt0),
    "mss_t1_xgb_auroc_sd": (
        "T1 XGBoost across-seed AUROC s.d.",
        lambda: _mss("T1", "xgboost", "std_auroc"),
        _fmt3,
    ),
    "mss_t1_cat_auroc_sd": (
        "T1 CatBoost across-seed AUROC s.d.",
        lambda: _mss("T1", "catboost", "std_auroc"),
        _fmt3,
    ),
    "mss_t1_cat_auprc_sd": (
        "T1 CatBoost across-seed AUPRC s.d.",
        lambda: _mss("T1", "catboost", "std_auprc"),
        _fmt3,
    ),
    "seed_max_optimism": (
        "max across-seed optimism (any model)",
        lambda: _seedinf("max_optimism_any_model"),
        _fmt3,
    ),
    # S12 region heterogeneity KS statistics
    "reghet_wetland_ks": (
        "best-vs-worst region wetland KS",
        lambda: _reghet_ks("pct_wetland_5km"),
        _fmt3,
    ),
    "reghet_nsamples_ks": (
        "best-vs-worst region n_samples KS",
        lambda: _reghet_ks("n_samples"),
        _fmt3,
    ),
    "sens_nfeat": ("preprocessing-invariant feature count", lambda: _sens("n_features"), _commas),
    "sens_auroc": ("preprocessing-invariant T1 AUROC", lambda: _sens("auroc"), _fmt3),
    "sens_auprc": ("preprocessing-invariant T1 AUPRC", lambda: _sens("auprc"), _fmt3),
    "cnn1d_auroc_mean": ("CNN1D AUROC mean across orderings", lambda: _cnn1d("auroc_mean"), _fmt3),
    "cnn1d_auroc_std": ("CNN1D AUROC s.d.", lambda: _cnn1d("auroc_std"), _fmt3),
    "cnn1d_auprc_mean": ("CNN1D AUPRC mean", lambda: _cnn1d("auprc_mean"), _fmt3),
    "cnn1d_auprc_std": ("CNN1D AUPRC s.d.", lambda: _cnn1d("auprc_std"), _fmt3),
    "cnn1d_alpha_auroc": (
        "CNN1D alphabetical-ordering AUROC",
        lambda: _cnn1d_ordering("alphabetical"),
        _fmt3,
    ),
    "cnn1d_domain_auroc": (
        "CNN1D domain-ordering AUROC",
        lambda: _cnn1d_ordering("domain"),
        _fmt3,
    ),
    # --- S5 pre-calibration ECE per T1 model (calibration_analysis.json) ---
    "calib_ece_cnn1d": ("T1 CNN1D ECE", lambda: _calib_ece("cnn1d_classifier"), _fmt3),
    "calib_ece_logreg": ("T1 LogReg ECE", lambda: _calib_ece("logistic_regression"), _fmt3),
    "calib_ece_catboost": ("T1 CatBoost ECE", lambda: _calib_ece("catboost_classifier"), _fmt3),
    "calib_ece_rf": ("T1 RandomForest ECE", lambda: _calib_ece("random_forest_classifier"), _fmt3),
    "calib_ece_lgbm": ("T1 LightGBM ECE", lambda: _calib_ece("lightgbm_classifier"), _fmt3),
    "calib_ece_xgb": ("T1 XGBoost ECE", lambda: _calib_ece("xgboost_classifier"), _fmt3),
    "calib_ece_tabpfn": ("T1 TabPFN ECE", lambda: _calib_ece("tabpfn_classifier"), _fmt3),
    # --- S6 per-model conformal coverage (conformal_results.json) ---
    "conf_xgb_cov": (
        "XGBoost marginal coverage @a=0.05",
        lambda: _conf_model("xgboost_classifier"),
        _fmt3,
    ),
    "conf_xgb_set": (
        "XGBoost avg set size @a=0.05",
        lambda: _conf_model("xgboost_classifier", "avg_set_size"),
        _fmt3,
    ),
    "conf_catboost": (
        "CatBoost marginal coverage @a=0.05",
        lambda: _conf_model("catboost_classifier"),
        _fmt3,
    ),
    "conf_lgbm": (
        "LightGBM marginal coverage @a=0.05",
        lambda: _conf_model("lightgbm_classifier"),
        _fmt3,
    ),
    "dml_adjusted_auroc": ("DML-adjusted/deconfounded T1 AUROC", _dml_adjusted, _fmt3),
    "t4_loro_full": (
        "T4 LORO with-provenance mean AUROC",
        lambda: _t4_loro("loro_cv.json"),
        _fmt3,
    ),
    "t4_loro_pf": (
        "T4 LORO provenance-free mean AUROC",
        lambda: _t4_loro("loro_cv_provenance_free.json"),
        _fmt3,
    ),
    "loro_pf_region_min": (
        "T1 prov-free weakest-region LORO AUROC",
        lambda: _loro_pf_region("min"),
        _fmt2,
    ),
    "loro_pf_region_max": (
        "T1 prov-free strongest-region LORO AUROC",
        lambda: _loro_pf_region("max"),
        _fmt2,
    ),
    # --- Minnesota MDH held-out generalization (leave-region-5-out LORO, MN subset) ---
    "mn_loro_auroc": ("Minnesota held-out region-5 LORO AUROC", lambda: _mn_loro("auroc"), _fmt3),
    "mn_loro_auprc": ("Minnesota held-out region-5 LORO AUPRC", lambda: _mn_loro("auprc"), _fmt3),
    # --- Monitoring inequity (sampling allocation) ---
    "mon_ratio_poc": (
        "POC sampling-intensity ratio",
        lambda: _mon_ineq("pct_people_of_color", "monitoring_ratio"),
        _fmt2,
    ),
    "mon_ratio_income": (
        "low-income sampling ratio",
        lambda: _mon_ineq("pct_low_income", "monitoring_ratio"),
        _fmt2,
    ),
    "mon_ratio_edu": (
        "low-education sampling ratio",
        lambda: _mon_ineq("pct_less_hs_education", "monitoring_ratio"),
        _fmt2,
    ),
    "mon_mean_high_poc": (
        "POC high-share mean samples",
        lambda: _mon_ineq("pct_people_of_color", "mean_high"),
        _fmt1,
    ),
    "mon_mean_low_poc": (
        "POC low-share mean samples",
        lambda: _mon_ineq("pct_people_of_color", "mean_low"),
        _fmt1,
    ),
    "mon_coef_poc": (
        "POC size-adjusted log-samples coef",
        lambda: _mon_ineq("pct_people_of_color", "size_adjusted_coef"),
        _fmt3,
    ),
    "mon_coef_income": (
        "low-income size-adjusted coef",
        lambda: _mon_ineq("pct_low_income", "size_adjusted_coef"),
        _fmt3m,
    ),
    "mon_coef_edu": (
        "low-education size-adjusted coef",
        lambda: _mon_ineq("pct_less_hs_education", "size_adjusted_coef"),
        _fmt3m,
    ),
    # --- Source-adjusted monitoring intensity (R7 M1) ---
    "mon_coef_fe": (
        "POC size-adjusted coef, data-source fixed effects",
        lambda: _src_adj("source_fixed_effects", "size_adjusted_coef"),
        _fmt3,
    ),
    "mon_ratio_ucmr5": (
        "POC monitoring ratio, dominant-UCMR5 single source",
        lambda: _src_adj("dominant_ucmr5", "monitoring_ratio"),
        _fmt2,
    ),
    "mon_coef_ucmr5": (
        "POC size-adjusted coef, dominant-UCMR5 single source",
        lambda: _src_adj("dominant_ucmr5", "size_adjusted_coef"),
        _fmt3,
    ),
    # --- Areal apportionment (POC burden) ---
    "ej_areal_single": ("POC burden single-nearest", lambda: _areal_poc("single"), _fmt2),
    "ej_areal_5km": ("POC burden 5km areal", lambda: _areal_poc("5km"), _fmt2),
    "ej_areal_10km": ("POC burden 10km areal", lambda: _areal_poc("10km"), _fmt2),
    # --- Spatial leakage (random vs geographic split, T1 baseline features) ---
    "leak_random_auprc": ("T1 random-split AUPRC", lambda: _split_comp("random", "auprc"), _fmt3),
    "leak_geo_auprc": (
        "T1 geographic-split AUPRC",
        lambda: _split_comp("geographic", "auprc"),
        _fmt3,
    ),
    "leak_random_auroc": ("T1 random-split AUROC", lambda: _split_comp("random", "auroc"), _fmt3),
    "leak_geo_auroc": (
        "T1 geographic-split AUROC",
        lambda: _split_comp("geographic", "auroc"),
        _fmt3,
    ),
    "leak_auprc_ratio": (
        "T1 AUPRC inflation ratio (random / geographic split)",
        lambda: _split_comp("random", "auprc") / _split_comp("geographic", "auprc"),
        _fmt2,
    ),
    "dml_poc_shap_rank": (
        "pct_people_of_color SHAP importance rank",
        lambda: _demo_rank("pct_people_of_color", "shap_rank"),
        _fmt0,
    ),
    "dml_poc_dml_rank": (
        "pct_people_of_color DML adjusted-coefficient rank",
        lambda: _demo_rank("pct_people_of_color", "dml_rank"),
        _fmt0,
    ),
    "loro_full_wmean": (
        "With-provenance LORO system-count-weighted mean AUROC (same estimator as PF; M4)",
        lambda: _json("spatial_block_bootstrap.json")["T1_xgboost"]["weighted_mean_auroc"],
        _fmt3,
    ),
    "cutpoint_poc_min": (
        "POC burden ratio, minimum across cutpoint schemes (C4/M8a)",
        lambda: _json("cutpoint_sensitivity.json")["poc_summary"]["min_ratio"],
        _fmt2,
    ),
    "cutpoint_poc_max": (
        "POC burden ratio, maximum across cutpoint schemes (C4/M8a)",
        lambda: _json("cutpoint_sensitivity.json")["poc_summary"]["max_ratio"],
        _fmt2,
    ),
    "ej_base_high": (
        "High-POC group detection base rate (M8c)",
        lambda: _json("group_error_calibration.json")["error_rates"]["pct_people_of_color"][
            "high"
        ]["positive_rate"],
        _fmt2,
    ),
    "ej_base_low": (
        "Low-POC group detection base rate (M8c)",
        lambda: _json("group_error_calibration.json")["error_rates"]["pct_people_of_color"]["low"][
            "positive_rate"
        ],
        _fmt2,
    ),
    "loro_pf_fold_mean_ref": (
        "Provenance-free LORO unweighted fold mean (reference for the buffered check)",
        lambda: _json("loro_cv_provenance_free.json")["T1"]["xgboost"]["mean_auroc"],
        _fmt3,
    ),
    "coordtail_0km": (
        "T1 env-feature AUROC at 0 km coordinate perturbation (C3/M6b)",
        lambda: next(
            r.get("auroc_mean", r.get("mean_auroc"))
            for r in _json("coordinate_tail_sensitivity.json")["summary"]
            if abs(float(r["magnitude_km"])) < 1e-9
        ),
        _fmt3,
    ),
    "coordtail_270km": (
        "T1 env-feature AUROC at 270 km (p95 error tail) perturbation (C3/M6b)",
        lambda: next(
            r.get("auroc_mean", r.get("mean_auroc"))
            for r in _json("coordinate_tail_sensitivity.json")["summary"]
            if abs(float(r["magnitude_km"]) - 270.0) < 1e-9
        ),
        _fmt3,
    ),
    "misgeo_n_slim": (
        "Systems in the slim frozen national T1-PFOS risk surface (Supplementary Fig. 9)",
        lambda: _json("predictions_slim.json")["_meta"]["n"],
        _commas,
    ),
    "misgeo_flagged": (
        "Systems flagged PWSID-prefix vs coordinate EPA-region mismatch "
        "(state-polygon rule, v2; C-F, #55 item 6)",
        lambda: _json("misgeocode_sensitivity_v2.json")["n_flagged_systems"],
        _commas,
    ),
    "misgeo_t1_excl_auroc": (
        "T1 XGBoost AUROC with region-mismatched systems excluded (v2; C-F)",
        lambda: _json("misgeocode_sensitivity_v2.json")["per_task"]["T1"]["auroc_excluded"],
        _fmt3,
    ),
    "misgeo_t4_excl_auroc": (
        "T4 XGBoost AUROC with region-mismatched systems excluded (v2; C-F)",
        lambda: _json("misgeocode_sensitivity_v2.json")["per_task"]["T4"]["auroc_excluded"],
        _fmt3,
    ),
    "crl_base_auroc": (
        "T1 default-xgboost AUROC, un-recensored baseline (C2/M6a)",
        lambda: _json("common_rl_sensitivity.json")["baseline"]["auroc"],
        _fmt3,
    ),
    "crl_common_auroc": (
        "T1 AUROC after re-censoring to the common reporting limit (C2/M6a)",
        lambda: _json("common_rl_sensitivity.json")["common_rl"]["auroc"],
        _fmt3,
    ),
    "conf_poc_high": (
        "Conformal coverage, high-POC group at alpha=0.05 (C9/m10)",
        lambda: _json("group_conformal_demographic.json")["results"]["pct_people_of_color"][
            "by_alpha"
        ]["0.05"]["high_group_coverage"],
        _fmt3,
    ),
    "conf_poc_low": (
        "Conformal coverage, low-POC group at alpha=0.05 (C9/m10)",
        lambda: _json("group_conformal_demographic.json")["results"]["pct_people_of_color"][
            "by_alpha"
        ]["0.05"]["low_group_coverage"],
        _fmt3,
    ),
    "recov_pf": (
        "Source recoverability AUROC from the provenance-free feature set, logistic (C6/m1)",
        lambda: _json("pf_recoverability.json")["probes"]["logistic"]["auroc_from_pf_features"],
        _fmt2,
    ),
    "perm_reg_naive": (
        "Regions FDR-significant under the naive label-shuffle null (C5/M8b, shared assembly)",
        lambda: _json("spatial_permutation.json")["regions_significant"]["naive_shuffle_fdr"],
        _fmt0,
    ),
    "perm_reg_spatial": (
        "Regions FDR-significant under the spatial rotation null (C5/M8b, shared assembly)",
        lambda: _json("spatial_permutation.json")["regions_significant"]["rotation_spatial_fdr"],
        _fmt0,
    ),
    "perm_poc_p_spatial": (
        "POC national monitoring-ratio p under stratified-by-region permutation (C5/M8b)",
        lambda: _json("spatial_permutation.json")["national_monitoring_intensity"][
            "pct_people_of_color"
        ]["p_stratified_by_region"],
        lambda v: "0.001" if v < 0.001 else f"{v:.3f}",
    ),
    "lift_oof_pooled": (
        "Out-of-region pooled OOF top-decile lift, PF T1 LORO (C7/m11)",
        lambda: _json("loro_lift.json")["pooled_oof_lift_top_decile"],
        _fmt2,
    ),
    "lift_oof_weighted": (
        "Out-of-region system-count-weighted per-region top-decile lift (C7/m11)",
        lambda: _json("loro_lift.json")["weighted_mean_per_region_lift"],
        _fmt2,
    ),
    "buff_full_mean": (
        "Buffered-boundary LORO mean AUROC, with provenance (C8/m9 fixed variant)",
        lambda: _json("loro_cv_buffered.json")["mean_auroc_with_provenance"],
        _fmt3,
    ),
    "buff_pf_mean": (
        "Buffered-boundary LORO mean AUROC, provenance-free (C8/m9 fixed variant)",
        lambda: _json("loro_cv_buffered.json")["mean_auroc_provenance_free"],
        _fmt3,
    ),
    "prev_geo_test": (
        "Geographic test-set prevalence, split-comparison assembly (C1/M5)",
        lambda: _json("split_prevalence.json")["prevalence_geographic_test"],
        _fmt3,
    ),
    "prev_random_folds": (
        "Random-fold (pooled) prevalence, split-comparison assembly (C1/M5)",
        lambda: _json("split_prevalence.json")["prevalence_random_folds_pooled"],
        _fmt3,
    ),
    "dml_overlap_count": (
        "DML features flagged non-identified by the overlap/positivity warning (M8d)",
        lambda: sum(1 for e in _json("causal_deconfounding.json") if e.get("overlap_warning")),
        _fmt0,
    ),
    # --- EJ extras ---
    "ej_burden_west": ("POC burden, western test region", _western_burden, _fmt2),
    "ej_west_n_high": (
        "Region-9 high-POC group size (its own counts, not pooled — m6)",
        lambda: _western_burden("n_high"),
        _fmt0,
    ),
    "ej_west_n_low": (
        "Region-9 reference group size (its own counts, not pooled — m6)",
        lambda: _western_burden("n_low"),
        _commas,
    ),
    "sens_robust_count": (
        "features robust through Rosenbaum Gamma=3.0",
        _sens_robust_count,
        _commas,
    ),
    "t1_dummy_auprc": (
        "T1 dummy-baseline AUPRC",
        lambda: _metric("T1", "dummy_classifier", "auprc"),
        _fmt3,
    ),
    # --- M2: Fig. 3 SHAP feature importances (shap_T1.json) ---
    "shap_nsamples": ("T1 SHAP |n_samples| (rank 1)", lambda: _shap("n_samples"), _fmt3),
    "shap_ownertype": (
        "T1 SHAP |owner_type_nan| (rank 2)",
        lambda: _shap("owner_type_nan"),
        _fmt3,
    ),
    "shap_detlimit": (
        "T1 SHAP |mean_detection_limit|",
        lambda: _shap("mean_detection_limit"),
        _fmt3,
    ),
    # --- M3: group-conformal coverage (group_conformal_results.json) ---
    "conf_marginal": ("conformal marginal coverage", lambda: _conf("marginal"), _fmt3),
    "conf_r9": ("conformal Region 9 coverage", lambda: _conf("r9"), _fmt3),
    "conf_rmax": ("conformal best-region coverage", lambda: _conf("rmax"), _fmt3),
    "conf_n_r9": ("conformal Region 9 calibration n", lambda: _conf("n9"), _commas),
    "conf_n_r8": ("conformal Region 8 calibration n", lambda: _conf("n8"), _commas),
    "conf_r10": ("conformal Region 10 coverage", lambda: _conf("r10"), _fmt3),
    "conf_n_r10": ("conformal Region 10 calibration n", lambda: _conf("n10"), _commas),
    "conf_s8": ("conformal Region 8 avg set size", lambda: _conf("s8"), _fmt3),
    "conf_s9": ("conformal Region 9 avg set size", lambda: _conf("s9"), _fmt3),
    "conf_s10": ("conformal Region 10 avg set size", lambda: _conf("s10"), _fmt3),
    "conf_marginal_set": ("conformal marginal avg set size", lambda: _conf("marginal_set"), _fmt3),
    "conf_marginal_n": ("conformal marginal pooled n", lambda: _conf("marginal_n"), _commas),
    # --- M4: DML adjusted association + Rosenbaum sensitivity (POC vs low-income swap) ---
    "dml_poc_effect": (
        "DML POC adjusted effect",
        lambda: _dml_feat("pct_people_of_color", "causal_effect"),
        _fmt3,
    ),
    "dml_income_pfdr": (
        "DML low-income adjusted p_FDR",
        lambda: _dml_feat("pct_low_income", "p_value_fdr"),
        _fmt2,
    ),
    "sens_income_gamma": (
        "low-income Rosenbaum gamma*",
        lambda: _sens_gamma("pct_low_income"),
        _fmt1,
    ),
    # --- m6: DML significant-coefficient count ---
    "dml_sig_count": ("DML FDR-significant feature count", _dml_sig_count, _commas),
    "dml_total": ("DML total features tested", _dml_total, _commas),
    # --- M5: spatial-leakage AUPRC, logistic-regression full feature set ---
    "leak_lr_random_auprc": (
        "LogReg-full random-split AUPRC",
        lambda: _split_matrix("LogisticRegression", "full", "random"),
        _fmt3,
    ),
    "leak_lr_geo_auprc": (
        "LogReg-full geographic-split AUPRC",
        lambda: _split_matrix("LogisticRegression", "full", "geographic"),
        _fmt3,
    ),
    "leak_gbm_delta_pct": (
        "benchmark GBM random-vs-geo AUPRC inflation %",
        lambda: _leak_delta("XGBoost", "baseline"),
        _fmt0,
    ),
    "leak_lr_delta_pct": (
        "LogReg-full random-vs-geo AUPRC inflation %",
        lambda: _leak_delta("LogisticRegression", "full"),
        _fmt0,
    ),
    # --- m3: DML pre-adjustment AUROC (deconfounded_auroc.json) ---
    "dml_original_auroc": ("DML pre-adjustment AUROC", _dml_original, _fmt3),
    # --- m5: EJ burden canonical values ---
    "ej_burden_max": ("POC max LORO regional burden ratio", _ej_burden_max, _fmt2),
    "ej_burden_natl": (
        "POC national weighted-mean burden ratio",
        lambda: _inf("m4c_burden_ratio_ci.weighted_mean_burden_ratio"),
        _fmt2,
    ),
    "ej_burden_ci_lo": (
        "POC national burden block-bootstrap CI lower",
        lambda: _inf("m4c_burden_ratio_ci.block_bootstrap.ci_lower"),
        _fmt2,
    ),
    "ej_burden_ci_hi": (
        "POC national burden block-bootstrap CI upper",
        lambda: _inf("m4c_burden_ratio_ci.block_bootstrap.ci_upper"),
        _fmt2,
    ),
    "loro_full_sd": (
        "T1 LORO with-provenance across-fold s.d.",
        lambda: _loro_std("loro_cv.json"),
        _fmt3,
    ),
    # --- m4/audit-x2: Table-1 per-source record counts leaked into prose (dataset_snapshot) ---
    "ds_ucmr5": ("UCMR5 record count", lambda: _ds_source("ucmr5", "records"), _commas),
    "ds_ucmr3": ("UCMR3 record count", lambda: _ds_source("ucmr3", "records"), _commas),
    "ds_ohepa": ("OH EPA record count", lambda: _ds_source("oh_epa", "records"), _commas),
    "ds_modnr": ("MO DNR record count", lambda: _ds_source("mo_dnr", "records"), _commas),
    "ds_njdep": ("NJ DEP record count", lambda: _ds_source("nj_dep", "records"), _commas),
    "ds_wqp": ("WQP record count", lambda: _ds_source("wqp", "records"), _commas),
    "cens_ucmr5": (
        "UCMR5 left-censoring rate %",
        lambda: _ds_source("ucmr5", "censoring_pct"),
        _fmt1,
    ),
    "cens_ucmr3": (
        "UCMR3 left-censoring rate %",
        lambda: _ds_source("ucmr3", "censoring_pct"),
        _fmt1,
    ),
    "cens_wqp": ("WQP left-censoring rate %", lambda: _ds_source("wqp", "censoring_pct"), _fmt1),
    # Minnesota MDH ascertainment corroboration (mn_ascertainment.json)
    "mn_systems": ("MN MDH public water systems", lambda: _mn_asc("n_mn_systems_mdh"), _commas),
    "mn_overlap": (
        "MN systems sampled by both UCMR and MDH",
        lambda: _mn_asc("pfos.n_overlap_systems"),
        _commas,
    ),
    "mn_ucmr_nd": ("MN UCMR-nondetect systems", lambda: _mn_asc("pfos.n_ucmr_nondetect"), _commas),
    "mn_flip": (
        "MN UCMR-nondetect flipped to detected by MDH",
        lambda: _mn_asc("pfos.n_flip_nd_to_detect"),
        _commas,
    ),
    "mn_flip_rate": (
        "MN flip rate %",
        lambda: _mn_asc("pfos.flip_rate_of_ucmr_nondetect") * 100,
        _fmt1,
    ),
    # --- Dataset totals ---
    "ds_sdwis": ("SDWIS record count", lambda: float(_merged_count("src:sdwis")), _commas),
    "ds_ca": (
        "CA GeoTracker record count",
        lambda: float(_merged_count("src:ca_geotracker")),
        _commas,
    ),
    "ds_mn_mdh": (
        "MN MDH record count",
        lambda: float(_merged_count("src:mn_mdh")),
        _commas,
    ),
    "ds_samples": ("total samples", lambda: float(_merged_count("samples")), _commas),
    "ds_systems": ("total systems", lambda: float(_merged_count("systems")), _commas),
    "ds_geocoded": ("geocoded systems", lambda: float(_merged_count("geocoded")), _commas),
}

#: Registry ids whose value appears ONLY in a generated table or figure (Table 1,
#: Table 2, Extended Data Table 1, Fig. 5) — not in narrative prose. Kept in the
#: registry for reuse and documentation, but exempt from the prose-completeness
#: check (those artifacts are regenerated from frozen by generate_tables.py).
TABLE_ONLY: frozenset[str] = frozenset(
    {
        "ds_ca",
        "ds_mn_mdh",
        "ds_samples",
        "t1_dummy_auprc",
        "t1_abl_baseline_auprc",
        "t2_best_rmse",
        "t4_best_auprc",
        "t4_best_auroc",
    }
)


def expected(marker_id: str) -> str:
    if marker_id not in REGISTRY:
        raise KeyError(f"unknown marker id: {marker_id}")
    _desc, source, fmt = REGISTRY[marker_id]
    return fmt(source())


def sync(*, write: bool, files: list[Path] | None = None) -> list[str]:
    """Sync (write) or check markers against frozen. Returns list of problems.

    Problems: unknown marker id; stale value (check mode); a REGISTRY id with no
    marker present in any file (completeness).
    """
    files = files or _default_files()
    problems: list[str] = []
    seen_ids: set[str] = set()
    skipped_ids: set[str] = set()

    for fp in files:
        text = fp.read_text(encoding="utf-8")

        def _repl(m: re.Match[str], _fname: str = fp.name) -> str:
            mid, cur = m.group(1), m.group(2)
            seen_ids.add(mid)
            try:
                exp = expected(mid)
            except KeyError as e:
                problems.append(f"{_fname}: {e}")
                return m.group(0)
            except (FileNotFoundError, ImportError):
                # Source data/dep unavailable in this environment (e.g. the
                # gitignored data/interim parquet behind the ds_* loaders is
                # absent in CI, or pandas/pyarrow isn't installed; the committed
                # frozen JSONs are still checked). Can't verify here -> skip
                # without failing the gate.
                skipped_ids.add(mid)
                return m.group(0)
            if cur != exp:
                if write:
                    return f"<!--pn:{mid}-->{exp}<!--/pn-->"
                problems.append(f"{_fname}: pn:{mid} = {cur!r} but frozen = {exp!r}")
            return m.group(0)

        new = _MARKER.sub(_repl, text)
        if write and new != text:
            fp.write_text(new, encoding="utf-8")

    missing = sorted(set(REGISTRY) - seen_ids - TABLE_ONLY)
    if missing:
        problems.append(f"registry ids with no marker in any paper file: {missing}")
    if skipped_ids:
        print(
            f"  (paper_numbers: skipped {len(skipped_ids)} marker(s) with unavailable "
            f"source data in this environment: {sorted(skipped_ids)})"
        )
    return problems


def strip_markers(text: str) -> str:
    """Remove markers, keeping the inner value (for clean DOCX export)."""
    return _STRIP.sub(r"\1", text)


def check_paper_numbers() -> list[str]:
    """Gate hook for verify_paper: 0 problems == pass."""
    return sync(write=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="Fill markers from frozen.")
    g.add_argument("--check", action="store_true", help="Assert markers match frozen.")
    g.add_argument(
        "--strip", action="store_true", help="Print marker-stripped files to --out-dir."
    )
    ap.add_argument("--out-dir", default=None, help="Output dir for --strip.")
    args = ap.parse_args()

    if args.strip:
        out = Path(args.out_dir or (REPO / "build" / "paper_stripped"))
        out.mkdir(parents=True, exist_ok=True)
        for f in PAPER_FILES:
            src = PAPER / f
            if src.exists():
                (out / f).write_text(
                    strip_markers(src.read_text(encoding="utf-8")), encoding="utf-8"
                )
        print(f"Stripped {len(PAPER_FILES)} files -> {out}")
        return 0

    problems = sync(write=args.write)
    if args.write:
        print(f"Synced markers from frozen ({len(REGISTRY)} registry ids)")
        if problems:
            print("Problems:")
            for p in problems:
                print(f"  - {p}")
        return 1 if problems else 0
    # check
    if problems:
        print(f"✗ paper-number check: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"✓ paper-number check: all markers match frozen ({len(REGISTRY)} registry ids)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
