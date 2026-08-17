#!/usr/bin/env python
"""Generate publication tables for AquaContam (Nature Water format).

Usage::

    python paper/generate_tables.py --results results/ --output paper/tables/
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Any

import click
import pandas as pd

from aquacontam._constants import MODEL_DISPLAY_NAMES as _MODEL_DISPLAY_NAMES

# gate_lib is a sibling script under paper/, imported either as a package member (when
# generate_tables is imported, e.g. via aquacontam.pipeline.export) or by bare name (when run
# directly as `python paper/generate_tables.py`, which puts paper/ on sys.path). Resolve both.
try:
    from paper.gate_lib import degenerate_metric_models
except ImportError:  # pragma: no cover - script-dir-on-path fallback
    from gate_lib import degenerate_metric_models  # type: ignore[import-not-found,no-redef]

logger = logging.getLogger(__name__)


def _pub_name(code_name: str) -> str:
    """Convert internal model code name to publication display name."""
    return _MODEL_DISPLAY_NAMES.get(code_name, code_name)


def _load_provenance_free_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load the environment-only run's results (the honest "without" column).

    Prefers the frozen-snapshot name written by ``paper/freeze_results.py``
    (the broad ``--provenance-free`` run); the legacy no-monitoring names are
    retained as fallbacks for pre-reframe runs.
    """
    import json

    for path in (
        results_dir / "results_provenance_free.json",
        results_dir / "results_no_monitoring.json",  # legacy file name
        results_dir.parent / "results_no_monitoring" / "results.json",  # legacy layout
    ):
        if path.exists():
            return json.loads(path.read_text())  # type: ignore[no-any-return]
    return []


def table_dataset_summary(output_dir: Path) -> Path:
    """Table 1: Dataset summary — sources, records, analytes, coverage, censoring."""
    data = {
        "Source": [
            "UCMR5",
            "UCMR3",
            "SDWIS",
            "MI MPART",
            "CA GeoTracker‖",
            "OH EPA",
            "WA DOH",
            "NJ DEP",
            "NC DEQ",
            "WQP‖",
            "MO DNR",
            "MN MDH",
            "NJ Private Wells*",
            "EJScreen*",
        ],
        "Records": [
            "1,928,117",
            "1,069,174",
            "916,899",
            "5,640",
            "324,254",
            "26,554",
            "9,251",
            "248,107",
            "1,953",
            "35,528",
            "75,971",
            "247,230",
            "~10K",
            "~220K",
        ],
        "Analytes": [
            "29 PFAS + Li",
            "6 PFAS + 32",
            "Pb/Cu",
            "5 PFAS",
            "29 PFAS",
            "6 PFAS",
            "14 PFAS",
            "25 PFAS",
            "5 PFAS",
            "13 PFAS",
            "29 PFAS",
            "27 PFAS",
            "PFAS + metals",
            "Demographics",
        ],
        "Period": [
            "2023",
            "2013-2015",
            "2016-2022",
            "2019-2023",
            "2019-2023",
            "2020-2024",
            "2023-2024",
            "2019-2023",
            "2020-2023",
            "2020-2024",
            "2020-2023",
            "2005-2026",
            "2020-2023",
            "2023",
        ],
        "Censoring Rate": [
            "97.1%",
            "76.4%",
            "0.3%§",
            "0%†",
            "83.1%",
            "0%†",
            "0%†",
            "~80%",
            "~80%",
            "~39%",
            "~99%",
            "~92%",
            "~75%",
            "N/A",
        ],
        "Status": [
            "Available",
            "Available",
            "Available",
            "Available",
            "Available",
            "Excluded‡",
            "Available",
            "Available",
            "Unavailable",
            "Available",
            "Available",
            "Available",
            "Unavailable",
            "Available (GDB)",
        ],
    }
    df = pd.DataFrame(data)

    path = output_dir / "table1_dataset_summary"
    df.to_csv(f"{path}.csv", index=False)
    caption = (
        "Dataset summary. "
        "\\dag Detection-only reporting: source data includes only "
        "detected samples; non-detects are not available. "
        "\\S SDWIS reports Lead and Copper Rule 90th-percentile compliance "
        "values, genuine quantitative measurements only rarely left-censored; "
        "the near-zero (0.3%) rate reflects this measurement mechanism, not "
        "detection-only reporting. "
        "\\ddag Excluded from analysis: zero unique PWSIDs after deduplication "
        "with national sources. "
        "\\textbardbl Ambient environmental monitoring (groundwater/surface "
        "water), a different population from public-water-system finished water; "
        "reserved for external validation, not included in the main benchmark "
        "train/validation/test splits."
    )
    df.to_latex(f"{path}.tex", index=False, caption=caption, label="tab:datasets")
    df.to_markdown(f"{path}.md", index=False)
    # Append footnotes to markdown
    with open(f"{path}.md", "a") as f:
        f.write(
            "\n*Auxiliary source: provides private-well measurements (NJ Private "
            "Wells) or demographic covariates (EJScreen) rather than public-water-"
            "system compliance monitoring; not part of the primary benchmark "
            "population count.\n"
            "†Detection-only reporting: source data includes only detected "
            "samples; non-detects are not available.\n"
            "§SDWIS reports Lead and Copper Rule 90th-percentile compliance "
            "values, which are genuine quantitative measurements that are only "
            "rarely left-censored; the 0% censoring reflects this measurement "
            "mechanism, not detection-only reporting.\n"
            "‡Excluded from analysis: zero unique PWSIDs after deduplication "
            "with national sources.\n"
            "‖Ambient environmental monitoring (groundwater/surface water), a "
            "different population from public-water-system finished water; "
            "reserved for external validation, not included in the main "
            "benchmark train/validation/test splits.\n"
        )
    return Path(f"{path}.csv")


def _format_metric_with_ci(
    value: float,
    ci: dict[str, float] | None,
) -> str:
    """Format a metric value with optional bootstrap CI.

    Parameters
    ----------
    value : float
        Point estimate.
    ci : dict or None
        Bootstrap CI dict with keys ``ci_lower``, ``ci_upper``.

    Returns
    -------
    str
        Formatted string like ``"0.950 [0.941-0.958]"`` or ``"0.950"``.
    """
    import math

    if math.isnan(value):
        return "—"
    if ci and not math.isnan(ci.get("ci_lower", float("nan"))):
        return f"{value:.3f} [{ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}]"
    return f"{value:.3f}"


def table_benchmark_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Table 2: Benchmark results matrix — models x tasks x metrics.

    Formats primary metrics with 95% bootstrap CIs when available.
    """
    if not results:
        df = pd.DataFrame({"Note": ["No results available"]})
    else:
        # M2/R6-R1-5: T3 emits two groups of byte-identical micro metrics (a silent
        # non-convergence fallback). Flag those cells "non-converged" here too, so this
        # table is consistent with table_benchmark_full and never shows the fallback as
        # a plausible number.
        t3_metrics = {
            r.get("model", ""): {
                "micro_auroc": r.get("metrics", {}).get("micro_auroc", float("nan")),
                "micro_auprc": r.get("metrics", {}).get("micro_auprc", float("nan")),
            }
            for r in results
            if r.get("task") == "T3"
        }
        t3_degenerate = degenerate_metric_models(t3_metrics, ("micro_auroc", "micro_auprc"))
        rows = []
        for r in results:
            task = r.get("task", "")
            model = r.get("model", "")
            metrics = r.get("metrics", {})
            metadata = r.get("metadata", {})
            bootstrap_ci = metadata.get("bootstrap_ci", {})

            row: dict[str, Any] = {"Model": _pub_name(model), "Task": task}

            # Format AUROC and AUPRC with CIs for classification tasks
            t3_flagged = task == "T3" and model in t3_degenerate
            for key in ("auroc", "auprc"):
                if t3_flagged:
                    row[key.upper()] = "non-converged"
                    continue
                val = metrics.get(key)
                if val is not None:
                    ci = bootstrap_ci.get(key)
                    row[key.upper()] = _format_metric_with_ci(float(val), ci)
                # For T3 multilabel tasks, fall back to micro-averaged metrics
                elif task == "T3":
                    micro_key = f"micro_{key}"
                    micro_val = metrics.get(micro_key)
                    if micro_val is not None:
                        ci = bootstrap_ci.get(micro_key)
                        row[key.upper()] = _format_metric_with_ci(float(micro_val), ci)

            # Include other key metrics without CIs — all go through formatter
            for key in (
                "f1",
                "precision",
                "recall",
                "rmse",
                "mae",
                "r2",
                "detected_rmse",
                "censored_rmse",
                "concordance_index",
            ):
                val = metrics.get(key)
                if val is not None:
                    row[key.upper()] = _format_metric_with_ci(float(val), None)

            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.fillna("—")

    path = output_dir / "table2_benchmark_results"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(f"{path}.tex", index=False, caption="Benchmark results.", label="tab:results")
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


# Map the loro_cv_full.json short family names to the results.json (fixed-split) model keys.
_LORO_SHORT_TO_RESULTS = {
    "xgboost": "xgboost_classifier",
    "random_forest": "random_forest_classifier",
    "catboost": "catboost_classifier",
    "logistic_regression": "logistic_regression",
    "lightgbm": "lightgbm_classifier",
    "mlp": "mlp_classifier",
    "cnn1d": "cnn1d_classifier",
    "gnn_gcn": "gnn_gcn_classifier",
    "gnn_sage": "gnn_sage_classifier",
    "deep_tobit": "deep_tobit_classifier",
    "tabpfn": "tabpfn_classifier",
    "voting_ensemble": "voting_ensemble",
    "stacking_ensemble": "stacking_ensemble",
    "icp": "icp_classifier",
}
#: t_{0.975, 9} — the same critical value paper/compute_cluster_ci.py uses for G = 10
#: EPA-region clusters (df = G - 1 = 9). Kept in sync deliberately (R5 M3/M5).
_T_CRIT_9 = 2.262157162740992


def _t9_cluster_ci(fold_aurocs: list[float]) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a t(G-1) cluster-robust 95% CI over region folds.

    Mirrors paper/compute_cluster_ci.py: simple mean +/- t_{0.975,G-1} * sd/sqrt(G),
    sd with ddof = 1. Used for the leave-one-region-out leaderboard column so the
    primary ranking carries a leakage-aware interval, not an i.i.d. bootstrap one.
    """
    import numpy as np

    v = np.asarray(fold_aurocs, dtype=float)
    g = len(v)
    mean = float(v.mean())
    if g < 2:
        return mean, mean, mean
    se = float(v.std(ddof=1)) / math.sqrt(g)
    return mean, mean - _T_CRIT_9 * se, mean + _T_CRIT_9 * se


def table_benchmark_condensed(
    results: list[dict[str, Any]],
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Table 2: LORO-ranked T1 PFAS-detection leaderboard (referee R5 M3).

    The benchmark's primary ranking is leave-one-region-out (LORO) mean AUROC with a
    t(G-1) cluster-robust 95% CI (the leakage-resistant protocol), sorted descending.
    The single fixed geographic split (West-only, EPA Regions 8/9/10) AUROC with its
    i.i.d. bootstrap CI is shown alongside as a transparent secondary column; that
    bootstrap CI is anti-conservative under the residual spatial autocorrelation the
    paper documents (Moran's I up to ~0.67; see caption). Non-converged families are
    flagged in the "LORO folds" column and never shown as a rankable number: gnn_gcn /
    gnn_sage failed under in-process CUDA-context corruption, TabPFN ran only 2 of 10
    folds (10k-sample cap). This discharges both the M3 re-ranking and the M2
    degenerate-cell audit (gate_lib.flag_degenerate_metric_cells is wired below).
    """
    import json

    loro_path = (results_dir / "loro_cv_full.json") if results_dir is not None else None
    fixed = {
        r.get("model"): r
        for r in results
        if r.get("task") == "T1" and r.get("model") in _LORO_SHORT_TO_RESULTS.values()
    }

    if not loro_path or not loro_path.exists():
        df = pd.DataFrame({"Note": ["loro_cv_full.json unavailable"]})
        path = output_dir / "table2_benchmark_condensed"
        df.to_csv(f"{path}.csv", index=False)
        df.to_latex(f"{path}.tex", index=False)
        df.to_markdown(f"{path}.md", index=False)
        return Path(f"{path}.csv")

    loro = json.loads(loro_path.read_text(encoding="utf-8"))
    t1 = loro["T1"]
    non_conv = loro["_meta"]["non_converged"]
    failed = set(non_conv.get("failed", []))
    partial = non_conv.get("partial_folds", {})

    def _fixed_cell(short: str) -> str:
        r = fixed.get(_LORO_SHORT_TO_RESULTS[short], {})
        val = r.get("metrics", {}).get("auroc")
        if val is None:
            return "—"
        ci = r.get("metadata", {}).get("bootstrap_ci", {}).get("auroc")
        return _format_metric_with_ci(float(val), ci)

    converged: list[tuple[str, float, str, str]] = []
    flagged: list[tuple[str, str, str, str]] = []
    for short, entry in t1.items():
        display = _pub_name(_LORO_SHORT_TO_RESULTS[short])
        if short in failed:
            flagged.append((display, "—", _fixed_cell(short), "did not converge"))
            continue
        if short in partial:
            n = partial[short]
            flagged.append((display, "—", _fixed_cell(short), f"{n}/10 (10k cap)"))
            continue
        folds = [f["auroc"] for f in entry["folds"]]
        mean, lo, hi = _t9_cluster_ci(folds)
        loro_cell = f"{mean:.3f} [{lo:.3f}-{hi:.3f}]"
        converged.append((display, mean, loro_cell, _fixed_cell(short)))

    converged.sort(key=lambda t: t[1], reverse=True)

    # M2 audit: no two converged families may share a byte-identical LORO mean (a silent
    # non-convergence fallback masquerading as a result). T1 cells are genuinely distinct;
    # if that ever changes the offending cells are relabelled FLAGGED rather than shown.
    degen_models = degenerate_metric_models(
        {d: {"loro_auroc": m} for d, m, _c, _f in converged}, ("loro_auroc",)
    )

    rows = []
    for display, _mean, loro_cell, fixed_cell in converged:
        rows.append(
            {
                "Model": display,
                "LORO AUROC (mean, 95% CI)": ("FLAGGED" if display in degen_models else loro_cell),
                "Fixed-split AUROC (95% CI)": fixed_cell,
                "LORO folds": "10/10",
            }
        )
    for display, loro_cell, fixed_cell, flag in flagged:
        rows.append(
            {
                "Model": display,
                "LORO AUROC (mean, 95% CI)": loro_cell,
                "Fixed-split AUROC (95% CI)": fixed_cell,
                "LORO folds": flag,
            }
        )
    df = pd.DataFrame(rows).fillna("—")

    path = output_dir / "table2_benchmark_condensed"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption=(
            "T1 PFAS-detection benchmark, ranked by leave-one-region-out (LORO) mean AUROC. "
            "LORO mean AUROC carries a t(9) cluster-robust 95\\% CI over the 10 EPA-region "
            "folds (primary, leakage-resistant). The fixed geographic-split (West-only, EPA "
            "Regions 8/9/10) AUROC and its i.i.d. bootstrap CI are shown alongside; that "
            "bootstrap CI is anti-conservative under the residual spatial autocorrelation "
            "documented in the paper (Moran's I up to ~0.67). Non-converged families are "
            "flagged in LORO folds, never shown as a metric. Full multi-task results in "
            "Supplementary Table 9."
        ),
        label="tab:results-condensed",
    )
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_benchmark_full(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Supplementary Table 9: Complete benchmark results matrix.

    Full version of Table 2 with all models x tasks x metrics, relocated
    to Supplementary Information for Nature Water compliance.
    """
    # M2 (R5): T3 has NaN macro-AUROC for every model and two groups of models emitting
    # byte-identical micro metrics (a silent non-convergence fallback). Detect those cells
    # up front so they are rendered "non-converged", never as plausible numbers.
    t3_metrics = {
        r.get("model", ""): {
            "micro_auroc": r.get("metrics", {}).get("micro_auroc", float("nan")),
            "micro_auprc": r.get("metrics", {}).get("micro_auprc", float("nan")),
        }
        for r in results
        if r.get("task") == "T3"
    }
    t3_degenerate = degenerate_metric_models(t3_metrics, ("micro_auroc", "micro_auprc"))

    # Reuse the existing full table generation
    if not results:
        df = pd.DataFrame({"Note": ["No results available"]})
    else:
        rows = []
        for r in results:
            task = r.get("task", "")
            model = r.get("model", "")
            metrics = r.get("metrics", {})
            metadata = r.get("metadata", {})
            bootstrap_ci = metadata.get("bootstrap_ci", {})

            row: dict[str, Any] = {"Model": _pub_name(model), "Task": task}
            t3_flagged = task == "T3" and model in t3_degenerate
            for key in ("auroc", "auprc"):
                if t3_flagged:
                    row[key.upper()] = "non-converged"
                    continue
                val = metrics.get(key)
                if val is not None:
                    ci = bootstrap_ci.get(key)
                    row[key.upper()] = _format_metric_with_ci(float(val), ci)
                elif task == "T3":
                    micro_key = f"micro_{key}"
                    micro_val = metrics.get(micro_key)
                    if micro_val is not None:
                        ci = bootstrap_ci.get(micro_key)
                        row[key.upper()] = _format_metric_with_ci(float(micro_val), ci)

            for key in (
                "f1",
                "precision",
                "recall",
                "rmse",
                "mae",
                "r2",
                "detected_rmse",
                "censored_rmse",
                "concordance_index",
            ):
                val = metrics.get(key)
                if val is not None:
                    row[key.upper()] = _format_metric_with_ci(float(val), None)

            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.fillna("—")

    path = output_dir / "table_supp9_benchmark_full"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption="Complete benchmark results for all models and tasks.",
        label="tab:supp-benchmark-full",
    )
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_feature_importance(
    importance_df: pd.Series | None,
    output_dir: Path,
    *,
    shap_df: pd.Series | None = None,
) -> Path:
    """Supplementary Table: Feature importance ranking — top 30.

    Prefers SHAP values (``shap_df``) over tree importance (``importance_df``)
    when both are provided, since SHAP values are the metric cited in the paper.
    """
    if shap_df is not None and not shap_df.empty:
        df = shap_df.head(30).reset_index()
        df.columns = ["Feature", "Mean |SHAP|"]
    elif importance_df is not None and not importance_df.empty:
        df = importance_df.head(30).reset_index()
        df.columns = ["Feature", "Importance"]
    else:
        df = pd.DataFrame({"Feature": ["N/A"], "Mean |SHAP|": [0.0]})

    path = output_dir / "table_supp_feature_importance"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(f"{path}.tex", index=False, caption="Feature importance.", label="tab:features")
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_equity_summary(
    equity_data: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Extended Data Table 4: Equity framework outputs.

    Columns include per-group AUROC and permutation test p-values when
    the equity pipeline provides ``group_metrics`` and ``p_value`` fields.
    """
    if equity_data:
        # Wire actual equity results from pipeline
        if isinstance(equity_data, list):
            rows = []
            for entry in equity_data:
                group = entry.get("group", "")
                label = group.replace("pct_", "").replace("_", " ").title()

                # Extract per-group AUROC when available
                gm = entry.get("group_metrics", {})
                auroc_high = gm.get("high", {}).get("auroc")
                auroc_low = gm.get("low", {}).get("auroc")

                p_fdr = entry.get("p_value_fdr")
                row: dict[str, Any] = {
                    "Demographic Group": label,
                    "Burden Ratio": f"{entry.get('burden_ratio', 0.0):.2f}",
                    "Prediction Ratio": f"{entry.get('prediction_ratio', 0.0):.2f}",
                    "AUROC (High)": (f"{auroc_high:.3f}" if auroc_high is not None else "\u2014"),
                    "AUROC (Low)": (f"{auroc_low:.3f}" if auroc_low is not None else "\u2014"),
                    "N (High)": entry.get("n_high", 0),
                    "N (Low)": entry.get("n_low", 0),
                    "p-value": (
                        f"{entry['p_value']:.4f}" if entry.get("p_value") is not None else "\u2014"
                    ),
                    "p-value (FDR)": (f"{p_fdr:.4f}" if p_fdr is not None else "\u2014"),
                }
                rows.append(row)
            # Filter groups with empty reference group (e.g. Limited English N_low=0)
            rows = [r for r in rows if r["N (Low)"] > 0]
            df = pd.DataFrame(rows) if rows else pd.DataFrame(equity_data)
        else:
            df = pd.DataFrame(equity_data)
    else:
        # EJScreen data unavailable — show framework structure
        df = pd.DataFrame(
            {
                "Demographic Group": [
                    "People of Color (>80th pct)",
                    "Low Income (>80th pct)",
                    "Limited English (>80th pct)",
                    "Less Than HS Education (>80th pct)",
                ],
                "Burden Ratio": ["*", "*", "*", "*"],
                "Prediction Ratio": ["*", "*", "*", "*"],
                "AUROC (High)": ["*", "*", "*", "*"],
                "AUROC (Low)": ["*", "*", "*", "*"],
                "N (High)": ["*", "*", "*", "*"],
                "N (Low)": ["*", "*", "*", "*"],
                "p-value": ["*", "*", "*", "*"],
                "p-value (FDR)": ["*", "*", "*", "*"],
            }
        )

    path = output_dir / "table_ext4_equity_summary"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(f"{path}.tex", index=False, caption="Equity summary.", label="tab:equity")
    df.to_markdown(f"{path}.md", index=False)

    # Also write as Extended Data Table 4 (detailed EJ framework outputs)
    ext_path = output_dir / "table_ext4_ej_framework"
    df.to_csv(f"{ext_path}.csv", index=False)
    df.to_latex(
        f"{ext_path}.tex",
        index=False,
        caption="Environmental justice framework outputs.",
        label="tab:ext-ej-framework",
    )
    df.to_markdown(f"{ext_path}.md", index=False)

    return Path(f"{path}.csv")


def table_regional_performance(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Supplementary Table: Per-EPA-region AUROC/AUPRC for classification tasks.

    Extracts per-region test metrics from ``metadata["region_metrics"]`` when
    available. If per-region breakdowns have not been computed yet, generates
    a placeholder table showing the expected structure.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Benchmark results (from ``results.json``).
    output_dir : Path
        Output directory for CSV / LaTeX / Markdown files.

    Returns
    -------
    Path
        Path to the generated CSV file.
    """
    classification_tasks = ("T1", "T3", "T4")
    rows: list[dict[str, Any]] = []

    if results:
        for r in results:
            task = r.get("task", "")
            if task not in classification_tasks:
                continue

            model = r.get("model", "")
            metadata = r.get("metadata", {})
            region_metrics = metadata.get("region_metrics")

            if region_metrics:
                # region_metrics expected as dict[region_id, metrics_dict]
                for region, rm in region_metrics.items():
                    rows.append(
                        {
                            "Task": task,
                            "Model": _pub_name(model),
                            "Region": str(region),
                            "AUROC": f"{rm.get('auroc', float('nan')):.3f}",
                            "AUPRC": f"{rm.get('auprc', float('nan')):.3f}",
                            "N": int(rm.get("n_samples", 0)),
                        }
                    )

    if rows:
        df = pd.DataFrame(rows)
    else:
        # No per-region data yet — generate placeholder structure
        placeholder_regions = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
        ]
        placeholder_rows = []
        for region in placeholder_regions:
            placeholder_rows.append(
                {
                    "Task": "T1",
                    "Model": "XGBoost",
                    "Region": region,
                    "AUROC": "*",
                    "AUPRC": "*",
                    "N": "*",
                }
            )
        df = pd.DataFrame(placeholder_rows)

    path = output_dir / "table_regional_performance"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption="Per-EPA-region classification performance.",
        label="tab:regional",
    )
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_split_comparison(
    geo_results: list[dict[str, Any]],
    random_results: list[dict[str, Any]] | None,
    output_dir: Path,
) -> Path:
    """Table 3: Geographic vs. random split comparison.

    Compares test-set AUROC and AUPRC between geographic stratification
    and standard random splitting on T1 and T4.
    """
    if not geo_results and not random_results:
        df = pd.DataFrame(
            {"Note": ["Run pipeline with --random-split to generate comparison data"]}
        )
    else:
        rows = []
        # Index results by (task, model) for lookup
        geo_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for r in geo_results:
            key = (r.get("task", ""), r.get("model", ""))
            geo_by_key[key] = r

        rand_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        if random_results:
            for r in random_results:
                key = (r.get("task", ""), r.get("model", ""))
                rand_by_key[key] = r

        import math

        def _fmt(v: float) -> str:
            return "\u2014" if math.isnan(v) else f"{v:.3f}"

        compare_tasks = ("T1", "T4")
        all_models = sorted(
            {r.get("model", "") for r in geo_results if r.get("task", "") in compare_tasks}
        )

        for task in compare_tasks:
            for model in all_models:
                key = (task, model)
                geo_r = geo_by_key.get(key)
                rand_r = rand_by_key.get(key)

                geo_auroc = geo_r["metrics"].get("auroc", float("nan")) if geo_r else float("nan")
                geo_auprc = geo_r["metrics"].get("auprc", float("nan")) if geo_r else float("nan")
                rand_auroc = (
                    rand_r["metrics"].get("auroc", float("nan")) if rand_r else float("nan")
                )
                rand_auprc = (
                    rand_r["metrics"].get("auprc", float("nan")) if rand_r else float("nan")
                )

                delta_auroc = (
                    rand_auroc - geo_auroc
                    if not (math.isnan(rand_auroc) or math.isnan(geo_auroc))
                    else float("nan")
                )

                rows.append(
                    {
                        "Task": task,
                        "Model": _pub_name(model),
                        "Geo AUROC": _fmt(geo_auroc),
                        "Random AUROC": _fmt(rand_auroc),
                        "Delta AUROC": _fmt(delta_auroc),
                        "Geo AUPRC": _fmt(geo_auprc),
                        "Random AUPRC": _fmt(rand_auprc),
                    }
                )

        df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame({"Note": ["No comparable results for T1/T4"]})
        )

    path = output_dir / "table3_split_comparison"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption="Geographic vs.\\ random split comparison.",
        label="tab:splits",
    )
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_t6_arsenic(output_dir: Path, results_dir: Path) -> Path:
    """T6: arsenic transfer (public-supply -> domestic) and geographic-leakage inflation.

    Reads ``t6_arsenic.json`` (from ``pipeline.t6_arsenic.run_t6_experiment``).
    Each row is a model: in-distribution public-supply AUROC under geographic vs.
    random split (and the inflation between them), plus the zero-shot domestic
    holdout AUROC/AUPRC.
    """
    import json
    import math

    path = output_dir / "table_t6_arsenic"
    t6_path = results_dir / "t6_arsenic.json"
    if not t6_path.exists():
        df = pd.DataFrame({"Note": ["Run reproduce.py --t6-arsenic to generate t6_arsenic.json"]})
        df.to_csv(f"{path}.csv", index=False)
        df.to_markdown(f"{path}.md", index=False)
        return Path(f"{path}.csv")

    data = json.loads(t6_path.read_text(encoding="utf-8"))

    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{v:.3f}"

    rows = []
    for name, e in data.get("models", {}).items():
        geo = e.get("geographic", {})
        rnd = e.get("random", {})
        rows.append(
            {
                "Model": _pub_name(name),
                "Geo public AUROC": _fmt(geo.get("public_test", {}).get("auroc")),
                "Random public AUROC": _fmt(rnd.get("public_test", {}).get("auroc")),
                "Inflation AUROC": _fmt(e.get("leakage_inflation_auroc")),
                "Domestic AUROC": _fmt(geo.get("domestic_holdout", {}).get("auroc")),
                "Domestic AUPRC": _fmt(geo.get("domestic_holdout", {}).get("auprc")),
            }
        )
    df = pd.DataFrame(rows)
    caption = (
        "T6 arsenic transfer (USGS National Groundwater Aggregation, CC0). Train on "
        f"public-supply wells (n={data.get('n_public_wells')}), evaluate zero-shot on "
        f"domestic wells (n={data.get('n_domestic_wells')}). 'Inflation' is random minus "
        "geographic split AUROC on the public-supply test set (geographic-leakage effect); "
        "'Domestic' is the zero-shot held-out domestic-well metric."
    )
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(f"{path}.tex", index=False, caption=caption, label="tab:t6arsenic")
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_ext_feature_catalog(output_dir: Path) -> Path:
    """Extended Data Table 1: Complete feature catalog."""
    features = [
        # Proximity features (EPA FRS)
        ("nearest_industrial_km", "Proximity", "Distance to nearest industrial facility (km)"),
        ("nearest_military_km", "Proximity", "Distance to nearest military installation (km)"),
        ("nearest_wwtp_km", "Proximity", "Distance to nearest wastewater treatment plant (km)"),
        ("nearest_airport_km", "Proximity", "Distance to nearest airport (km)"),
        ("nearest_landfill_km", "Proximity", "Distance to nearest landfill (km)"),
        ("count_industrial_1km", "Proximity", "Industrial facilities within 1 km"),
        ("count_industrial_5km", "Proximity", "Industrial facilities within 5 km"),
        ("count_industrial_10km", "Proximity", "Industrial facilities within 10 km"),
        ("count_military_5km", "Proximity", "Military installations within 5 km"),
        ("count_wwtp_5km", "Proximity", "WWTPs within 5 km"),
        # Land use features (NLCD 2019)
        ("frac_developed_1km", "Land Use", "Developed land fraction (1 km buffer)"),
        ("frac_developed_5km", "Land Use", "Developed land fraction (5 km buffer)"),
        ("frac_agriculture_1km", "Land Use", "Agricultural land fraction (1 km buffer)"),
        ("frac_agriculture_5km", "Land Use", "Agricultural land fraction (5 km buffer)"),
        ("frac_forest_1km", "Land Use", "Forest land fraction (1 km buffer)"),
        ("frac_wetland_1km", "Land Use", "Wetland fraction (1 km buffer)"),
        ("frac_water_1km", "Land Use", "Open water fraction (1 km buffer)"),
        # Hydrogeology features (USGS Principal Aquifers)
        ("aquifer_type_*", "Hydrogeology", "Principal aquifer type (one-hot, ~15 categories)"),
        ("aquifer_lithology_*", "Hydrogeology", "Aquifer lithology (one-hot)"),
        ("aquifer_confinement_*", "Hydrogeology", "Aquifer confinement status (one-hot)"),
        # Demographics (EJScreen, when available)
        ("pct_people_of_color", "Demographics", "Percent people of color (block group)"),
        ("pct_low_income", "Demographics", "Percent low income (block group)"),
        ("pct_less_hs_education", "Demographics", "Percent less than HS education"),
        ("pct_limited_english", "Demographics", "Percent linguistically isolated"),
        ("ej_index_water", "Demographics", "EJScreen water quality EJ index"),
        ("ej_supplemental_index", "Demographics", "EJScreen supplemental EJ index"),
        # System-level features (from aggregation)
        ("n_samples", "System", "Number of monitoring samples for the system"),
        ("mean_detection_limit", "System", "Mean analytical detection limit (µg/L)"),
    ]
    df = pd.DataFrame(features, columns=["Feature", "Category", "Description"])

    path = output_dir / "table_ext1_feature_catalog"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(f"{path}.tex", index=False, caption="Feature catalog.", label="tab:ext_features")
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_ext_per_analyte_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Supplementary Table 4: benchmark results by task (representative analyte).

    Rows are (model, task) pairs with each task's representative target
    analyte, not a per-analyte disaggregation — the genuine per-PFAS
    breakdown lives in Supplementary Fig. 4 (T3 per-analyte AUROC) and in
    the per-label ``auroc_*`` columns of the accompanying CSV. The
    historical ``table_ext2_per_analyte`` filename is kept to avoid
    churning committed references.
    """
    if not results:
        df = pd.DataFrame({"Note": ["No results available"]})
    else:
        rows = []
        for r in results:
            analyte = r.get("metadata", {}).get("analyte", "all")
            row = {
                "Model": r.get("model", ""),
                "Task": r.get("task", ""),
                "Analyte": analyte,
            }
            row.update(r.get("metrics", {}))
            rows.append(row)
        df = pd.DataFrame(rows)

    path = output_dir / "table_ext2_per_analyte"
    # Full matrix (~44 cols) kept in CSV/TeX; the caption directs readers here for
    # per-label, micro-averaged, and regression metrics.
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption=(
            "Benchmark results by task (representative analyte per task; "
            "per-label metrics in the accompanying CSV)."
        ),
        label="tab:ext_analyte",
    )

    # The embedded DOCX table is a legible, classification-focused slice: the full
    # 44-column frame is otherwise shrunk to an unreadable 6pt in Word. Keep
    # classification tasks only (drop regression T2) and the caption's promised
    # columns plus the T3 multilabel macro averages.
    display_cols = [
        "Model",
        "Task",
        "Analyte",
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
        "macro_auroc",
        "macro_auprc",
    ]
    if "Task" in df.columns:
        slim = df[df["Task"] != "T2"].copy()
        slim = slim[[c for c in display_cols if c in slim.columns]].dropna(axis=1, how="all")
        num_cols = slim.select_dtypes(include="number").columns
        slim[num_cols] = slim[num_cols].round(3)
        slim = slim.fillna("")
    else:
        slim = df
    slim.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_external_validation(
    output_dir: Path,
    results_dir: Path | None = None,
    *,
    update_skeleton: bool = True,
) -> Path:
    """External validation: per-state database metrics (7-column format).

    Generates the table and optionally updates ``[TBD]`` placeholders in
    ``paper/skeleton.md`` (Extended Data Table 6).
    """
    tabulate = __import__("tabulate")

    # Display-name mapping: internal key -> paper name
    _DISPLAY_NAMES: dict[str, str] = {
        "mi_mpart": "MI MPART",
        "ca_geotracker": "CA GeoTracker",
        "oh_epa": "OH EPA",
        "wa_doh": "WA DOH",
        "nj_dep": "NJ DEP",
        "nc_deq": "NC DEQ",
        "mo_dnr": "MO DNR",
    }

    # EPA region -> split role (from geographic_split defaults)
    _TRAIN_REGIONS = {1, 3, 4, 5, 6}
    _VAL_REGIONS = {2, 7}
    _TEST_REGIONS = {8, 9, 10}

    def _split_role(region: int | None) -> str:
        if region is None:
            return "—"
        if region in _TRAIN_REGIONS:
            return "train"
        if region in _VAL_REGIONS:
            return "val"
        if region in _TEST_REGIONS:
            return "test"
        return "—"

    # Ordered: external validation first (NJ, MO), then domain shift (CA),
    # then single-class / unavailable databases
    _STATE_ORDER = [
        "nj_dep",
        "mo_dnr",
        "ca_geotracker",
        "mi_mpart",
        "oh_epa",
        "wa_doh",
        "nc_deq",
    ]
    _STATE_REGIONS: dict[str, int] = {
        "mi_mpart": 5,
        "ca_geotracker": 9,
        "oh_epa": 5,
        "wa_doh": 10,
        "nj_dep": 2,
        "nc_deq": 4,
        "mo_dnr": 7,
    }

    # Load results JSON (dict keyed by state name)
    raw: dict[str, Any] = {}
    if results_dir:
        import json

        ext_path = results_dir / "external_validation.json"
        if ext_path.exists():
            raw = json.loads(ext_path.read_text())

    # Build 7-column rows
    rows: list[dict[str, Any]] = []
    footnotes: list[str] = []
    for key in _STATE_ORDER:
        display = _DISPLAY_NAMES[key]
        region = _STATE_REGIONS[key]
        role = _split_role(region)
        entry = raw.get(key, {})
        metrics = entry.get("metrics", {})
        boot_ci = entry.get("bootstrap_ci", {})

        n_sys = entry.get("n_systems", "—")
        auroc_val = metrics.get("auroc", "—")
        auprc = metrics.get("auprc", "—")
        det_rate = entry.get("detection_rate", "—")

        # Format AUROC with bootstrap CI when available
        if isinstance(auroc_val, float):
            auroc_ci = boot_ci.get("auroc")
            auroc_str = _format_metric_with_ci(auroc_val, auroc_ci)
        else:
            auroc_str = auroc_val

        rows.append(
            {
                "State DB": display,
                "EPA Region": f"R{region}",
                "Split Role": role,
                "n Systems": n_sys if n_sys == "—" else int(n_sys),
                "AUROC [95% CI]": auroc_str,
                "AUPRC": f"{auprc:.3f}" if isinstance(auprc, float) else auprc,
                "Detection Rate": (f"{det_rate:.3f}" if isinstance(det_rate, float) else det_rate),
            }
        )

        # Collect DeLong chance-test footnote
        import math

        chance_test = entry.get("chance_test")
        p_val = chance_test.get("p_value") if chance_test else None
        if p_val is not None and not math.isnan(p_val):
            footnotes.append(
                f"*{display}: DeLong test vs. chance (AUROC = 0.5): "
                f"z = {chance_test['z_statistic']:.2f}, "
                f"p = {chance_test['p_value']:.3f}.*"
            )

    df = pd.DataFrame(rows)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    if footnotes:
        table_str += "\n\n" + " ".join(footnotes)
    out_path = output_dir / "table_external_validation.md"
    out_path.write_text(f"## External Validation and Domain Shift Analysis\n\n{table_str}\n")
    logger.info("Table written: %s", out_path)

    # Update [TBD] placeholders in skeleton.md (only when we have real data)
    if update_skeleton and raw:
        _update_skeleton_table6(rows)

    return out_path


def _update_skeleton_table6(rows: list[dict[str, Any]]) -> None:
    """Replace ``[TBD]`` values in Extended Data Table 6 of extended_data.md."""
    extended_path = Path(__file__).resolve().parent / "extended_data.md"
    if not extended_path.exists():
        logger.warning(
            "extended_data.md not found at %s — skipping TBD replacement", extended_path
        )
        return

    text = extended_path.read_text()
    original = text
    for row in rows:
        state_db = row["State DB"]
        # Find the markdown table row starting with "| <State DB> |"
        # and replace [TBD] with actual values
        prefix = f"| {state_db} |"
        for line in text.splitlines():
            if line.startswith(prefix) and "[TBD]" in line:
                new_line = (
                    f"| {state_db} "
                    f"| {row['EPA Region']} "
                    f"| {row['Split Role']} "
                    f"| {row['n Systems']} "
                    f"| {row['AUROC [95% CI]']} "
                    f"| {row['AUPRC']} "
                    f"| {row['Detection Rate']} |"
                )
                text = text.replace(line, new_line)
                break

    # Update DeLong footnote if present
    import json
    import math
    import re

    results_dir = Path(__file__).resolve().parent.parent / "results"
    ext_path = results_dir / "external_validation.json"
    if ext_path.exists():
        ext_data = json.loads(ext_path.read_text())
        ca_entry = ext_data.get("ca_geotracker", {})
        chance_test = ca_entry.get("chance_test")
        if chance_test:
            z_stat = chance_test.get("z_statistic", float("nan"))
            p_val = chance_test.get("p_value", float("nan"))
            if not math.isnan(z_stat) and not math.isnan(p_val):
                boot_ci = ca_entry.get("bootstrap_ci", {}).get("auroc", {})
                ci_lo = boot_ci.get("ci_lower", 0)
                ci_hi = boot_ci.get("ci_upper", 1)
                # Replace the entire DeLong footnote block (may span multiple lines)
                old_pattern = (
                    r"†DeLong test vs\. chance.*?consistent with\n?"
                    r"(?:.*?consistent with\n)?"  # optional duplicate line
                    r"domain shift from groundwater monitoring points to\n?"
                    r"public water systems\."
                )
                new_footnote = (
                    f"†DeLong test vs. chance (AUROC = 0.5): "
                    f"z = {z_stat:.2f}, p = {p_val:.3f}. The 95% CI\n"
                    f"[{ci_lo:.3f}, {ci_hi:.3f}] spans 0.5, "
                    f"indicating performance is not distinguishable from\n"
                    f"chance — consistent with domain shift from "
                    f"groundwater monitoring points to\n"
                    f"public water systems."
                )
                text = re.sub(old_pattern, new_footnote, text, flags=re.DOTALL)

    # Only rewrite when a placeholder/footnote actually changed. An unconditional write
    # re-emits the whole file with the platform line ending (LF on Linux), which mutates a
    # CRLF-committed manuscript and trips the G4 "PROSE MUTATED" / G12 DOCX-stale checks in
    # CI even though no content changed.
    if text != original:
        extended_path.write_text(text)
        logger.info("Updated Extended Data Table 6 in %s", extended_path)


def table_temporal_diagnosis(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Temporal diagnosis: per-analyte T7 metrics and detection limit changes."""
    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        import json

        diag_path = results_dir / "temporal_diagnosis.json"
        if diag_path.exists():
            raw = json.loads(diag_path.read_text())
            data = raw if isinstance(raw, list) else raw.get("per_analyte", [])

    if not data:
        data = [
            {"analyte": a, "ucmr3_det_rate": "—", "ucmr5_det_rate": "—", "auroc": "—"}
            for a in ("PFOS", "PFOA", "PFBS", "PFHxS")
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_temporal_diagnosis.md"
    out_path.write_text(f"## Temporal Diagnosis: Per-Analyte T7 Metrics\n\n{table_str}\n")
    logger.info("Table written: %s", out_path)
    return out_path


def table_split_strategy_comparison(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Split strategy comparison: feature set x split strategy metrics.

    Handles both legacy format (flat dict with random_split_metrics/
    geographic_split_metrics) and new format (``{"simple": ..., "matrix": ...}``).
    """
    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        import json

        comparison_path = results_dir / "split_comparison.json"
        if comparison_path.exists():
            raw = json.loads(comparison_path.read_text())
            if isinstance(raw, dict):
                # New format: {"simple": {...}, "matrix": {...}}
                matrix = raw.get("matrix", {})
                matrix_results = matrix.get("results", [])
                delong_tests = matrix.get("delong_tests", {})

                if matrix_results:
                    for entry in matrix_results:
                        model = entry.get("model_name", "?")
                        fs = entry.get("feature_set", "?")
                        split = entry.get("split_strategy", "?")
                        # The size-matched arm goes to its own supplementary
                        # table; keep this one the canonical 2x2.
                        if split == "random_sizematched":
                            continue
                        metrics = entry.get("metrics", {})
                        auroc_str = f"{metrics.get('auroc', 0):.3f}"
                        auprc_str = f"{metrics.get('auprc', 0):.3f}"

                        # Add CI/std info
                        boot = entry.get("bootstrap_ci", {})
                        if boot and "auroc" in boot:
                            ci = boot["auroc"]
                            auroc_str += (
                                f" [{ci.get('ci_lower', 0):.3f}-{ci.get('ci_upper', 0):.3f}]"
                            )
                        fold_m = entry.get("fold_metrics")
                        if fold_m:
                            import numpy as _np

                            std = float(_np.std([m.get("auroc", 0) for m in fold_m]))
                            auroc_str += f" \u00b1 {std:.3f}"

                        # DeLong test: geographic-split Full vs Baseline features,
                        # paired on the shared test set. Report the magnitude and
                        # DIRECTION (delta = Full - Baseline AUROC), never a bare
                        # "0.000" (LR-Full is significantly WORSE, not better).
                        delong_p = "\u2014"
                        if split == "geographic" and fs == "full":
                            dl = delong_tests.get(model, {})
                            if "p_value" in dl:
                                p = float(dl["p_value"])
                                p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
                                full_minus_baseline = float(dl.get("auroc_b", 0.0)) - float(
                                    dl.get("auroc_a", 0.0)
                                )
                                delong_p = f"{p_str} (\u0394AUROC {full_minus_baseline:+.3f})"

                        data.append(
                            {
                                "Model": _pub_name(model),
                                "Features": fs.title(),
                                "Split": split.title(),
                                "AUROC": auroc_str,
                                "AUPRC": auprc_str,
                                "DeLong p": delong_p,
                            }
                        )
                else:
                    # Legacy format
                    simple = raw.get("simple", raw)
                    for split_type in ("random_split", "geographic_split"):
                        metrics = simple.get(f"{split_type}_metrics", {})
                        data.append(
                            {
                                "Model": "XGBoost",
                                "Features": "Baseline",
                                "Split": split_type.replace("_split", "").title(),
                                "AUROC": f"{metrics.get('auroc', 0):.3f}" if metrics else "\u2014",
                                "AUPRC": f"{metrics.get('auprc', 0):.3f}" if metrics else "\u2014",
                                "DeLong p": "\u2014",
                            }
                        )

    if not data:
        data = [
            {
                "Model": "XGBoost",
                "Features": "Baseline",
                "Split": "Random",
                "AUROC": "\u2014",
                "AUPRC": "\u2014",
                "DeLong p": "\u2014",
            },
            {
                "Model": "XGBoost",
                "Features": "Baseline",
                "Split": "Geographic",
                "AUROC": "\u2014",
                "AUPRC": "\u2014",
                "DeLong p": "\u2014",
            },
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_brennan_comparison.md"
    footnote = (
        "\N{DAGGER}DeLong p: paired test of Full-feature vs Baseline-feature AUROC on the "
        "shared geographic test set (reported on the geographic/Full rows); "
        "\N{GREEK CAPITAL LETTER DELTA}AUROC = Full \N{MINUS SIGN} Baseline, so a negative "
        "value means the Full feature set performs worse."
    )
    out_path.write_text(
        f"## Split Comparison: Feature Set \N{MULTIPLICATION SIGN} Split Strategy\n\n"
        f"{table_str}\n\n{footnote}\n",
        encoding="utf-8",
    )
    logger.info("Table written: %s", out_path)
    return out_path


def table_supp_split_sizematched(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path | None:
    """Size-matched random-split control: random CV with the training set
    subsampled to the geographic training-set size, isolating the evaluation-
    protocol effect from the training-set-size advantage of random splits.

    Emitted only when the ``random_sizematched`` arm is present in
    ``split_comparison.json`` (``--split-comparison --split-size-matched``).
    """
    if not results_dir:
        return None
    import json

    comparison_path = results_dir / "split_comparison.json"
    if not comparison_path.exists():
        return None
    raw = json.loads(comparison_path.read_text())
    matrix_results = raw.get("matrix", {}).get("results", []) if isinstance(raw, dict) else []
    if not any(e.get("split_strategy") == "random_sizematched" for e in matrix_results):
        return None

    tabulate = __import__("tabulate")
    _split_label = {
        "random": "Random",
        "random_sizematched": "Random (size-matched)",
        "geographic": "Geographic",
    }
    data: list[dict[str, Any]] = []
    for entry in matrix_results:
        split = entry.get("split_strategy", "?")
        metrics = entry.get("metrics", {})
        auroc_str = f"{metrics.get('auroc', 0):.3f}"
        fold_m = entry.get("fold_metrics")
        if fold_m:
            import numpy as _np

            std = float(_np.std([m.get("auroc", 0) for m in fold_m]))
            auroc_str += f" ± {std:.3f}"
        data.append(
            {
                "Model": _pub_name(entry.get("model_name", "?")),
                "Features": entry.get("feature_set", "?").title(),
                "Split": _split_label.get(split, split),
                "n train": f"{entry.get('n_train', 0):,}",
                "AUROC": auroc_str,
                "AUPRC": f"{metrics.get('auprc', 0):.3f}",
            }
        )

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_supp_split_sizematched.md"
    out_path.write_text(
        "## Split Comparison with Size-Matched Random Control\n\n"
        f"{table_str}\n\n"
        "Random (size-matched): random 5-fold CV with each fold's training set "
        "stratified-subsampled to the geographic training-set size.\n"
    )
    logger.info("Table written: %s", out_path)
    return out_path


def table_hyperparam_sensitivity(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Hyperparameter sensitivity: model x metric mean±std."""
    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        import json

        sens_path = results_dir / "hyperparam_sensitivity.json"
        if sens_path.exists():
            raw = json.loads(sens_path.read_text())
            summary = raw.get("summary", {})
            # summary is flat: {metric, n_configs, mean, std, min, max}
            metric_name = summary.get("metric", "unknown")
            if "mean" in summary:
                data.append(
                    {
                        "metric": metric_name,
                        "mean": f"{summary.get('mean', 0):.4f}",
                        "std": f"{summary.get('std', 0):.4f}",
                        "min": f"{summary.get('min', 0):.4f}",
                        "max": f"{summary.get('max', 0):.4f}",
                    }
                )

    if not data:
        data = [
            {"metric": "AUPRC", "mean": "—", "std": "—", "min": "—", "max": "—"},
            {"metric": "AUROC", "mean": "—", "std": "—", "min": "—", "max": "—"},
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_hyperparam_sensitivity.md"
    out_path.write_text(f"## Hyperparameter Sensitivity: Metric Variation\n\n{table_str}\n")
    logger.info("Table written: %s", out_path)
    return out_path


def table_calibration_comparison(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Calibration comparison: pre/post calibration ECE per model."""
    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        import json

        cal_path = results_dir / "calibration_comparison.json"
        if cal_path.exists():
            raw = json.loads(cal_path.read_text())
            if isinstance(raw, list):
                for entry in raw:
                    data.append(
                        {
                            "task": entry.get("task", ""),
                            "model": _pub_name(entry.get("model", "")),
                            "method": entry.get("method", ""),
                            "ece_before": f"{entry.get('ece_before', 0):.4f}",
                            "ece_after": f"{entry.get('ece_after', 0):.4f}",
                        }
                    )

    if not data:
        data = [
            {
                "task": "T1",
                "model": "XGBoost",
                "method": "isotonic",
                "ece_before": "—",
                "ece_after": "—",
            },
            {
                "task": "T1",
                "model": "XGBoost",
                "method": "Platt",
                "ece_before": "—",
                "ece_after": "—",
            },
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_calibration_comparison.md"
    out_path.write_text(f"## Calibration Comparison: Pre/Post ECE\n\n{table_str}\n")
    logger.info("Table written: %s", out_path)
    return out_path


def table_tuning_comparison(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Tuning comparison: default vs tuned per model on T1 test set.

    Shows that all model families received symmetric HPO treatment,
    addressing the methodological fairness concern.
    """
    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        import json

        tuning_path = results_dir / "tuning_comparison.json"
        if tuning_path.exists():
            raw = json.loads(tuning_path.read_text())
            for entry in raw:
                if "error" in entry:
                    continue
                default_m = entry.get("default_test_metrics", {})
                tuned_m = entry.get("tuned_test_metrics", {})
                delta_auprc = entry.get("delta_auprc", 0)
                data.append(
                    {
                        "Model": _pub_name(entry.get("model", "").replace("_classifier", "")),
                        "Configs": entry.get("n_configs_tried", "—"),
                        "Default AUROC": f"{default_m.get('auroc', 0):.3f}",
                        "Tuned AUROC": f"{tuned_m.get('auroc', 0):.3f}",
                        "Default AUPRC": f"{default_m.get('auprc', 0):.3f}",
                        "Tuned AUPRC": f"{tuned_m.get('auprc', 0):.3f}",
                        "ΔAUPRC": f"{delta_auprc:+.3f}",
                    }
                )

    if not data:
        data = [
            {
                "Model": "—",
                "Configs": "—",
                "Default AUROC": "—",
                "Tuned AUROC": "—",
                "Default AUPRC": "—",
                "Tuned AUPRC": "—",
                "ΔAUPRC": "—",
            }
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_tuning_comparison.md"
    out_path.write_text(
        "## Hyperparameter Tuning Comparison: Default vs Tuned (T1 PFAS Detection)\n\n"
        f"{table_str}\n",
        encoding="utf-8",
    )
    logger.info("Table written: %s", out_path)

    # Also write as Extended Data Table 2 for verify_paper.py pattern matching
    ext_path = output_dir / "table_ext2_tuning_comparison"
    Path(f"{ext_path}.md").write_text(
        f"## Extended Data Table 2: Hyperparameter Tuning Comparison\n\n{table_str}\n",
        encoding="utf-8",
    )
    df.to_csv(f"{ext_path}.csv", index=False)
    df.to_latex(
        f"{ext_path}.tex",
        index=False,
        caption="Hyperparameter tuning comparison.",
        label="tab:ext-tuning",
    )

    return out_path


def table_ext5_ablation(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Extended Data Table 3: Feature category ablation results.

    Drop-one-category ablation study for T1 (PFAS detection) and T4 (heavy
    metal action level) using XGBoost.
    """
    import json
    import math

    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        ablation_path = results_dir / "feature_ablation.json"
        if ablation_path.exists():
            raw = json.loads(ablation_path.read_text())
            if isinstance(raw, list):
                for entry in raw:
                    baseline = entry.get("baseline_score", entry.get("baseline_auprc", 0))
                    ablated = entry.get("ablated_score", entry.get("ablated_auprc", 0))
                    delta = entry.get("delta", entry.get("delta_auprc", ablated - baseline))
                    sig = entry.get("significance", {})
                    # Prefer AUPRC significance; fall back to AUROC
                    sig_dict = sig.get("auprc", sig.get("auroc", {}))
                    p_val = sig_dict.get("p_value")
                    if p_val is not None:
                        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
                    else:
                        p_str = "—"
                    p_fdr = sig_dict.get("p_value_fdr")
                    if p_fdr is not None:
                        p_fdr_str = "<0.001" if p_fdr < 0.001 else f"{p_fdr:.3f}"
                    else:
                        p_fdr_str = "—"
                    data.append(
                        {
                            "Task": entry.get("task", ""),
                            "Category": entry.get("category", ""),
                            "N Features": entry.get("n_features_removed", 0),
                            "Baseline AUPRC": f"{baseline:.3f}"
                            if not math.isnan(baseline)
                            else "—",
                            "Ablated AUPRC": f"{ablated:.3f}" if not math.isnan(ablated) else "—",
                            "ΔAUPRC": f"{delta:+.4f}" if not math.isnan(delta) else "—",
                            "p-value": p_str,
                            "p (FDR)": p_fdr_str,
                        }
                    )

    if not data:
        data = [
            {
                "Task": "T1",
                "Category": "—",
                "N Features": "—",
                "Baseline AUPRC": "—",
                "Ablated AUPRC": "—",
                "ΔAUPRC": "—",
                "p-value": "—",
                "p (FDR)": "—",
            }
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)

    path = output_dir / "table_ext3_ablation"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption="Feature category ablation results.",
        label="tab:ext-ablation",
    )
    out_path = Path(f"{path}.md")
    out_path.write_text(
        f"## Extended Data Table 3: Feature Category Ablation Results\n\n{table_str}\n"
    )
    logger.info("Table written: %s", out_path)
    return out_path


def table_ext5_literature_comparison(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Extended Data Table 5: Literature comparison.

    Prior-study rows are literature values; the AquaContam rows are read from
    the controlled split-comparison (``split_comparison.json`` "simple" block:
    XGBoost on the baseline proximity/demographic feature set) so they are
    traceable to the frozen snapshot rather than hardcoded.
    """
    import json

    # Traceable AquaContam XGBoost (baseline feature set) random vs geographic.
    aqua = {
        "random": {"auroc": float("nan"), "auprc": float("nan")},
        "geographic": {"auroc": float("nan"), "auprc": float("nan")},
    }
    if results_dir is not None:
        sc_path = results_dir / "split_comparison.json"
        if sc_path.exists():
            simple = json.loads(sc_path.read_text()).get("simple", {})
            rm = simple.get("random_split_metrics", {})
            gm = simple.get("geographic_split_metrics", {})
            if rm and gm:
                aqua["random"] = {"auroc": rm.get("auroc"), "auprc": rm.get("auprc")}
                aqua["geographic"] = {"auroc": gm.get("auroc"), "auprc": gm.get("auprc")}

    def _f(v: Any) -> str:
        import math as _m

        return "—" if v is None or (isinstance(v, float) and _m.isnan(v)) else f"{v:.3f}"

    data = [
        {
            "Study": "Hu et al.",
            "Year": 2016,
            "Dataset Size": "~4800 sites",
            "Analytes": "6 PFAS",
            "Geographic Scope": "National (US)",
            "Split Method": "N/A (spatial regression)",
            "Best AUROC": "R\u00b2=0.38-0.62",
            "Best AUPRC": "NR",
            "Feature Categories": "Industrial proximity, land use, hydrology",
            "Open Code": "No",
        },
        {
            "Study": "Tokranov et al.",
            "Year": 2024,
            "Dataset Size": "~5000 wells",
            "Analytes": "PFAS",
            "Geographic Scope": "National (US)",
            "Split Method": "Random CV",
            "Best AUROC": "~0.85",
            "Best AUPRC": "NR",
            "Feature Categories": "Land use, hydrogeology, demographics",
            "Open Code": "Partial",
        },
        {
            "Study": "Fernandez et al.",
            "Year": 2023,
            "Dataset Size": "~4200 wells",
            "Analytes": "6 PFAS",
            "Geographic Scope": "Michigan",
            "Split Method": "10-fold stratified CV",
            "Best AUROC": ">0.90",
            "Best AUPRC": "NR",
            "Feature Categories": "Proximity, land use, demographics, hydrogeology",
            "Open Code": "No",
        },
        {
            "Study": "AquaContam (random split)*",
            "Year": 2026,
            "Dataset Size": "12136 systems",
            "Analytes": "29 PFAS + heavy metals",
            "Geographic Scope": "National (US)",
            "Split Method": "Random split",
            "Best AUROC": _f(aqua["random"]["auroc"]),
            "Best AUPRC": _f(aqua["random"]["auprc"]),
            "Feature Categories": "Proximity, land use, hydrogeology, demographics, system chars",
            "Open Code": "Yes",
        },
        {
            "Study": "AquaContam (geographic)*",
            "Year": 2026,
            "Dataset Size": "12136 systems",
            "Analytes": "29 PFAS + heavy metals",
            "Geographic Scope": "National (US)",
            "Split Method": "Geographic (EPA region)",
            "Best AUROC": _f(aqua["geographic"]["auroc"]),
            "Best AUPRC": _f(aqua["geographic"]["auprc"]),
            "Feature Categories": "Proximity, land use, hydrogeology, demographics, system chars",
            "Open Code": "Yes",
        },
    ]
    df = pd.DataFrame(data)

    path = output_dir / "table_ext5_literature_comparison"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption="Literature comparison. *XGBoost on the baseline feature set "
        "(controlled split comparison; split_comparison.json).",
        label="tab:ext-literature",
    )
    df.to_markdown(f"{path}.md", index=False)
    return Path(f"{path}.csv")


def table_monitoring_comparison(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table: Full vs provenance-free performance comparison.

    Side-by-side comparison of T1 and T4 results with the full feature set and
    with the environment-only (provenance-free) set -- monitoring intensity,
    system size, and the ``*_nan`` source-proxy indicators dropped -- to
    quantify surveillance/ascertainment bias.
    """
    import json
    import math

    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []

    def _load_results(path: Path) -> list[dict[str, Any]]:
        if path.exists():
            return json.loads(path.read_text())  # type: ignore[no-any-return]
        return []

    if results_dir:
        with_mon = _load_results(results_dir / "results.json")
        without_mon = _load_provenance_free_results(results_dir)

        def _fmt(v: float) -> str:
            return "\u2014" if math.isnan(v) else f"{v:.3f}"

        # Index by (task, model)
        with_by_key = {(r.get("task", ""), r.get("model", "")): r for r in with_mon}
        without_by_key = {(r.get("task", ""), r.get("model", "")): r for r in without_mon}

        compare_tasks = ("T1", "T4")
        all_models = sorted(
            {r.get("model", "") for r in with_mon if r.get("task", "") in compare_tasks}
        )

        for task in compare_tasks:
            for model in all_models:
                key = (task, model)
                w = with_by_key.get(key)
                wo = without_by_key.get(key)

                w_auroc = w["metrics"].get("auroc", float("nan")) if w else float("nan")
                w_auprc = w["metrics"].get("auprc", float("nan")) if w else float("nan")
                wo_auroc = wo["metrics"].get("auroc", float("nan")) if wo else float("nan")
                wo_auprc = wo["metrics"].get("auprc", float("nan")) if wo else float("nan")

                d_auroc = (
                    wo_auroc - w_auroc
                    if not (math.isnan(wo_auroc) or math.isnan(w_auroc))
                    else float("nan")
                )
                d_auprc = (
                    wo_auprc - w_auprc
                    if not (math.isnan(wo_auprc) or math.isnan(w_auprc))
                    else float("nan")
                )

                data.append(
                    {
                        "Task": task,
                        "Model": _pub_name(model),
                        "AUROC (with)": _fmt(w_auroc),
                        "AUROC (without)": _fmt(wo_auroc),
                        "\u0394AUROC": _fmt(d_auroc),
                        "AUPRC (with)": _fmt(w_auprc),
                        "AUPRC (without)": _fmt(wo_auprc),
                        "\u0394AUPRC": _fmt(d_auprc),
                    }
                )

    if not data:
        data = [
            {
                "Task": "T1",
                "Model": "\u2014",
                "AUROC (with)": "\u2014",
                "AUROC (without)": "\u2014",
                "\u0394AUROC": "\u2014",
                "AUPRC (with)": "\u2014",
                "AUPRC (without)": "\u2014",
                "\u0394AUPRC": "\u2014",
            }
        ]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_supp_monitoring_comparison.md"
    out_path.write_text(
        f"## Supplementary: Full vs Provenance-Free (Environment-Only) Performance\n\n{table_str}\n",
        encoding="utf-8",
    )
    df.to_csv(str(output_dir / "table_supp_monitoring_comparison.csv"), index=False)
    logger.info("Table written: %s", out_path)
    return out_path


def table_lift_analysis(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table: Decile-based lift analysis.

    Shows detection rates per predicted-risk decile for full and
    monitoring-free models.
    """
    tabulate = __import__("tabulate")

    data: list[dict[str, Any]] = []
    if results_dir:
        import json

        lift_path = results_dir / "lift_analysis.json"
        if lift_path.exists():
            raw = json.loads(lift_path.read_text())
            for model_label, key in [
                ("Full model", "full_model"),
                ("Monitoring-free", "monitoring_free"),
            ]:
                lift_data = raw.get(key, {})
                if not lift_data:
                    continue
                rates = lift_data.get("decile_detection_rates", [])
                overall = lift_data.get("overall_detection_rate", 0)
                top_lift = lift_data.get("top_decile_lift", float("nan"))
                quintile_cap = lift_data.get("top_quintile_capture", float("nan"))

                for i, rate in enumerate(rates):
                    data.append(
                        {
                            "Model": model_label,
                            "Decile": f"D{i + 1}",
                            "Detection Rate": f"{rate:.1%}",
                            "Cumulative Lift": f"{lift_data.get('cumulative_lift', [0] * 10)[i]:.2f}x",
                        }
                    )
                data.append(
                    {
                        "Model": model_label,
                        "Decile": "Summary",
                        "Detection Rate": f"Overall: {overall:.1%}",
                        "Cumulative Lift": f"Top-decile lift: {top_lift:.1f}x, Top-quintile capture: {quintile_cap:.1%}",
                    }
                )

    if not data:
        data = [{"Model": "—", "Decile": "—", "Detection Rate": "—", "Cumulative Lift": "—"}]

    df = pd.DataFrame(data)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_supp_lift_analysis.md"
    out_path.write_text(f"## Supplementary: Decile-Based Lift Analysis\n\n{table_str}\n")
    df.to_csv(str(output_dir / "table_supp_lift_analysis.csv"), index=False)
    logger.info("Table written: %s", out_path)
    return out_path


def table_sensitivity_analysis(
    output_dir: Path,
    *,
    results_dir: Path = Path("results"),
) -> Path:
    """Generate supplementary sensitivity analysis table.

    Two sub-tables: drop-NA threshold sweep and buffer radius variants.

    Parameters
    ----------
    output_dir : Path
        Directory for output files.
    results_dir : Path
        Directory containing ``sensitivity_analysis.json``.

    Returns
    -------
    Path
        Path to the generated markdown file.
    """
    import json

    tabulate = importlib.import_module("tabulate")

    sens_path = results_dir / "sensitivity_analysis.json"
    if not sens_path.exists():
        out_path = output_dir / "table_supp_sensitivity.md"
        out_path.write_text("## Table S: Sensitivity Analysis\n\nNo data available.\n")
        logger.warning("sensitivity_analysis.json not found — writing placeholder")
        return out_path

    data = json.loads(sens_path.read_text())
    sections = []

    # Sub-table 1: drop_na_threshold
    thresh_data = data.get("drop_na_threshold", [])
    if thresh_data:
        rows = [
            {
                "Threshold": r["threshold"],
                "Features": r["n_features"],
                "AUROC": f"{r['auroc']:.4f}",
                "AUPRC": f"{r['auprc']:.4f}",
            }
            for r in thresh_data
        ]
        df_thresh = pd.DataFrame(rows)
        sections.append(
            "### Drop-NA Threshold\n\n"
            + tabulate.tabulate(df_thresh, headers="keys", tablefmt="pipe", showindex=False)
        )
        df_thresh.to_csv(str(output_dir / "table_supp_sensitivity_threshold.csv"), index=False)

    # Sub-table 2: buffer radii
    buffer_data = data.get("buffer_radii", [])
    if buffer_data:
        rows = [
            {
                "Variant": r["variant"],
                "Proximity Cols": r["n_proximity_cols"],
                "AUROC": f"{r['auroc']:.4f}",
                "AUPRC": f"{r['auprc']:.4f}",
            }
            for r in buffer_data
        ]
        df_buffer = pd.DataFrame(rows)
        sections.append(
            "### Buffer Radius Variants\n\n"
            + tabulate.tabulate(df_buffer, headers="keys", tablefmt="pipe", showindex=False)
        )
        df_buffer.to_csv(str(output_dir / "table_supp_sensitivity_buffer.csv"), index=False)

    out_path = output_dir / "table_supp_sensitivity.md"
    out_path.write_text("## Table S: Sensitivity Analysis\n\n" + "\n\n".join(sections) + "\n")

    # LaTeX version
    tex_lines = ["\\begin{table}[ht]", "\\caption{Sensitivity Analysis}", "\\centering"]
    if thresh_data:
        tex_lines.append("\\begin{subtable}{\\textwidth}")
        tex_lines.append("\\caption{Drop-NA Threshold}")
        tex_lines.append("\\begin{tabular}{cccc}")
        tex_lines.append("\\hline")
        tex_lines.append("Threshold & Features & AUROC & AUPRC \\\\")
        tex_lines.append("\\hline")
        for r in thresh_data:
            tex_lines.append(
                f"{r['threshold']} & {r['n_features']} & {r['auroc']:.4f} & {r['auprc']:.4f} \\\\"
            )
        tex_lines.append("\\hline")
        tex_lines.append("\\end{tabular}")
        tex_lines.append("\\end{subtable}")
    if buffer_data:
        tex_lines.append("\\begin{subtable}{\\textwidth}")
        tex_lines.append("\\caption{Buffer Radius Variants}")
        tex_lines.append("\\begin{tabular}{cccc}")
        tex_lines.append("\\hline")
        tex_lines.append("Variant & Proximity Cols & AUROC & AUPRC \\\\")
        tex_lines.append("\\hline")
        for r in buffer_data:
            tex_lines.append(
                f"{r['variant']} & {r['n_proximity_cols']} & "
                f"{r['auroc']:.4f} & {r['auprc']:.4f} \\\\"
            )
        tex_lines.append("\\hline")
        tex_lines.append("\\end{tabular}")
        tex_lines.append("\\end{subtable}")
    tex_lines.append("\\end{table}")
    tex_path = output_dir / "table_supp_sensitivity.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")

    logger.info("Sensitivity table written: %s", out_path)
    return out_path


def table_causal_deconfounding(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Extended Data Table 10: Causal adjusted association results.

    Reads ``causal_deconfounding.json`` and ``deconfounded_auroc.json``.

    Parameters
    ----------
    output_dir : Path
        Output directory for table files.
    results_dir : Path, optional
        Results directory.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    import json

    tabulate = importlib.import_module("tabulate")

    if results_dir is None:
        results_dir = Path("results")

    causal_path = results_dir / "causal_deconfounding.json"
    auroc_path = results_dir / "deconfounded_auroc.json"

    rows: list[dict[str, str]] = []

    # Load rank comparison data if available
    rank_path = results_dir / "dml_shap_rank_comparison.json"
    rank_lookup: dict[str, dict[str, int]] = {}
    if rank_path.exists():
        rank_data = json.loads(rank_path.read_text())
        for fr in rank_data.get("feature_ranks", []):
            rank_lookup[fr["feature"]] = fr

    if causal_path.exists():
        causal_data = json.loads(causal_path.read_text())
        sorted_data = sorted(
            causal_data, key=lambda x: abs(x.get("causal_effect") or 0), reverse=True
        )
        for d in sorted_data[:15]:
            entry = {
                "Feature": d["feature"],
                "SHAP Importance": f"{d.get('shap_importance') or 0:.4f}",
                "Causal Effect": f"{d.get('causal_effect') or 0:.4f}",
                "Std Error": f"{d.get('std_error') or 0:.4f}",
                "p-value": f"{d.get('p_value') or 0:.2e}",
            }
            if "p_value_fdr" in d:
                entry["p-value (FDR)"] = f"{d['p_value_fdr']:.2e}"
            fr = rank_lookup.get(d["feature"])
            entry["DML Rank"] = str(fr["dml_rank"]) if fr else "—"
            entry["SHAP Rank"] = str(fr["shap_rank"]) if fr else "—"
            rows.append(entry)
    else:
        for i in range(5):
            rows.append(
                {
                    "Feature": f"feature_{i}",
                    "SHAP Importance": "—",
                    "Causal Effect": "—",
                    "Std Error": "—",
                    "p-value": "—",
                    "p-value (FDR)": "—",
                    "DML Rank": "—",
                    "SHAP Rank": "—",
                }
            )

    df = pd.DataFrame(rows)

    # Footer note
    footer = ""
    if auroc_path.exists():
        auroc_data = json.loads(auroc_path.read_text())
        orig = auroc_data.get("original_auroc", 0)
        deconf = auroc_data.get("deconfounded_auroc", 0)
        footer = f"\nOriginal AUROC: {orig:.3f}, Adjusted AUROC: {deconf:.3f}"
    if rank_path.exists():
        rho = rank_data.get("spearman_rho", 0)
        n_feat = rank_data.get("n_features", 0)
        footer += f"\nSpearman rho = {rho:.2f} across {n_feat} features"

    csv_path = output_dir / "table_ext10_causal_deconfounding.csv"
    df.to_csv(csv_path, index=False)

    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    md_path = output_dir / "table_ext10_causal_deconfounding.md"
    md_path.write_text(
        "## Extended Data Table 10: Causal Adjusted Associations\n\n" + md_text + footer + "\n"
    )

    tex_lines = [
        "\\begin{table}[ht]",
        "\\caption{Causal Adjusted Associations: SHAP vs DML Adjusted Effects (Top 15 Features)}",
        "\\centering",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Feature & SHAP & Causal Effect & Std Error & p-value & DML Rank & SHAP Rank \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        tex_lines.append(
            f"{row['Feature']} & {row['SHAP Importance']} & {row['Causal Effect']} & "
            f"{row['Std Error']} & {row['p-value']} & {row['DML Rank']} & "
            f"{row['SHAP Rank']} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    tex_path = output_dir / "table_ext10_causal_deconfounding.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")

    logger.info("Causal adjusted association table written: %s", csv_path)
    return csv_path


def table_group_conformal(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Extended Data Table 11: Group conformal prediction results.

    Reads ``group_conformal_results.json``.

    Parameters
    ----------
    output_dir : Path
        Output directory for table files.
    results_dir : Path, optional
        Results directory.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    import json

    tabulate = importlib.import_module("tabulate")

    if results_dir is None:
        results_dir = Path("results")

    conformal_path = results_dir / "group_conformal_results.json"
    rows: list[dict[str, str]] = []

    if conformal_path.exists():
        conformal_data = json.loads(conformal_path.read_text())
        for entry in conformal_data:
            per_region = entry.get("per_region", {})
            region_coverages = {
                f"Region {r}": f"{per_region[r]['coverage']:.3f}"
                for r in sorted(per_region.keys(), key=int)
            }
            rows.append(
                {
                    "Alpha": f"{entry.get('alpha', 0):.2f}",
                    "Target": f"{entry.get('target_coverage', 0):.2f}",
                    "Overall Coverage": f"{entry.get('coverage', 0):.3f}",
                    "Avg Set Size": f"{entry.get('avg_set_size', 0):.3f}",
                    **region_coverages,
                }
            )
    else:
        rows.append(
            {
                "Alpha": "0.05",
                "Target": "0.95",
                "Overall Coverage": "—",
                "Avg Set Size": "—",
                "Region 8": "—",
                "Region 9": "—",
                "Region 10": "—",
            }
        )

    df = pd.DataFrame(rows)

    csv_path = output_dir / "table_ext11_group_conformal.csv"
    df.to_csv(csv_path, index=False)

    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    md_path = output_dir / "table_ext11_group_conformal.md"
    md_path.write_text(
        "## Extended Data Table 11: Group Conformal Prediction\n\n" + md_text + "\n"
    )

    tex_lines = [
        "\\begin{table}[ht]",
        "\\caption{Group Conformal Prediction: Per-Region Coverage}",
        "\\centering",
        f"\\begin{{tabular}}{{{'l' * len(df.columns)}}}",
        "\\hline",
        " & ".join(df.columns) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        tex_lines.append(" & ".join(str(v) for v in row.to_numpy()) + " \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    tex_path = output_dir / "table_ext11_group_conformal.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")

    logger.info("Group conformal table written: %s", csv_path)
    return csv_path


def table_dedup_comparison(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Extended Data Table: Deduplication impact on T1 performance.

    Reads ``dedup_comparison.json``.

    Parameters
    ----------
    output_dir : Path
        Output directory for table files.
    results_dir : Path, optional
        Results directory.

    Returns
    -------
    Path
        Path to the written markdown file.
    """
    import json

    tabulate = importlib.import_module("tabulate")

    if results_dir is None:
        results_dir = Path("results")

    dedup_path = results_dir / "dedup_comparison.json"
    rows: list[dict[str, str]] = []

    if dedup_path.exists():
        raw = json.loads(dedup_path.read_text())
        for label, display in [("no_dedup", "No dedup"), ("dedup_max", "Dedup (max)")]:
            variant = raw.get(label, {})
            metrics = variant.get("metrics", {})
            rows.append(
                {
                    "Variant": display,
                    "n_systems": str(variant.get("n_systems", "—")),
                    "mean_n_samples": f"{variant.get('mean_n_samples', 0):.1f}",
                    "AUROC": f"{metrics.get('auroc', 0):.3f}" if metrics else "—",
                    "AUPRC": f"{metrics.get('auprc', 0):.3f}" if metrics else "—",
                }
            )
        delta = raw.get("delta", {})
        if delta:
            rows.append(
                {
                    "Variant": "Delta",
                    "n_systems": "—",
                    "mean_n_samples": "—",
                    "AUROC": f"{delta.get('auroc', 0):+.4f}",
                    "AUPRC": f"{delta.get('auprc', 0):+.4f}",
                }
            )
    else:
        rows = [
            {
                "Variant": "No dedup",
                "n_systems": "—",
                "mean_n_samples": "—",
                "AUROC": "—",
                "AUPRC": "—",
            },
            {
                "Variant": "Dedup (max)",
                "n_systems": "—",
                "mean_n_samples": "—",
                "AUROC": "—",
                "AUPRC": "—",
            },
        ]

    df = pd.DataFrame(rows)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_dedup_comparison.md"
    out_path.write_text(f"## Deduplication Impact on T1 Performance\n\n{table_str}\n")
    logger.info("Table written: %s", out_path)
    return out_path


def table_coordinate_sensitivity(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Extended Data Table: Coordinate perturbation sensitivity analysis.

    Reads ``coordinate_sensitivity.json``.

    Parameters
    ----------
    output_dir : Path
        Output directory for table files.
    results_dir : Path, optional
        Results directory.

    Returns
    -------
    Path
        Path to the written markdown file.
    """
    import json

    tabulate = importlib.import_module("tabulate")

    if results_dir is None:
        results_dir = Path("results")

    sens_path = results_dir / "coordinate_sensitivity.json"
    rows: list[dict[str, str]] = []

    if sens_path.exists():
        raw = json.loads(sens_path.read_text())
        summary = raw.get("summary", [])
        for entry in summary:
            mag = entry.get("magnitude_km", 0)
            rows.append(
                {
                    "Perturbation (km)": f"{mag:.0f}",
                    "AUROC (mean \u00b1 std)": _fmt_mean_std(
                        entry.get("auroc_mean"), entry.get("auroc_std")
                    ),
                    "AUPRC (mean \u00b1 std)": _fmt_mean_std(
                        entry.get("auprc_mean"), entry.get("auprc_std")
                    ),
                    "\u0394 AUROC": f"{entry.get('delta_auroc', 0):+.4f}"
                    if entry.get("delta_auroc") is not None
                    else "\u2014",
                    "\u0394 AUPRC": f"{entry.get('delta_auprc', 0):+.4f}"
                    if entry.get("delta_auprc") is not None
                    else "\u2014",
                }
            )
    if not rows:
        for mag in (0, 1, 5, 10, 20):
            rows.append(
                {
                    "Perturbation (km)": str(mag),
                    "AUROC (mean \u00b1 std)": "\u2014",
                    "AUPRC (mean \u00b1 std)": "\u2014",
                    "\u0394 AUROC": "\u2014",
                    "\u0394 AUPRC": "\u2014",
                }
            )

    df = pd.DataFrame(rows)
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    out_path = output_dir / "table_coordinate_sensitivity.md"
    out_path.write_text(f"## Coordinate Perturbation Sensitivity Analysis\n\n{table_str}\n")
    logger.info("Table written: %s", out_path)
    return out_path


def _fmt_mean_std(mean: float | None, std: float | None) -> str:
    """Format mean +/- std for table display."""
    if mean is None:
        return "\u2014"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} \u00b1 {std:.3f}"


def table_monitoring_invariance(
    output_dir: Path,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 13: Monitoring-invariance decomposition.

    Compares approaches to handling monitoring dependence: full model,
    provenance-free (environment-only features), ICP (adversarially
    invariant), and DML adjusted.
    """
    import json
    import math

    rows: list[dict[str, str]] = []

    def _fmt(v: float) -> str:
        return "\u2014" if math.isnan(v) else f"{v:.3f}"

    def _get_metric(results: list[dict[str, Any]], task: str, model: str, metric: str) -> float:
        for r in results:
            if r.get("task") == task and r.get("model") == model:
                return float(r.get("metrics", {}).get(metric, float("nan")))
        return float("nan")

    with_mon: list[dict[str, Any]] = []
    without_mon: list[dict[str, Any]] = []
    icp_diag: dict[str, Any] = {}
    causal_data: dict[str, Any] = {}

    if results_dir:
        res_path = results_dir / "results.json"
        if res_path.exists():
            with_mon = json.loads(res_path.read_text())

        without_mon = _load_provenance_free_results(results_dir)

        icp_path = results_dir / "icp_diagnostics.json"
        if icp_path.exists():
            icp_diag = json.loads(icp_path.read_text())

        causal_path = results_dir / "causal_deconfounding.json"
        if causal_path.exists():
            causal_data = json.loads(causal_path.read_text())
        # deconfounded_auroc lives in its own export when absent from the
        # causal_deconfounding.json top level.
        if not (isinstance(causal_data, dict) and "deconfounded_auroc" in causal_data):
            deconf_path = results_dir / "deconfounded_auroc.json"
            if deconf_path.exists():
                deconf_data = json.loads(deconf_path.read_text())
                deconf_val = deconf_data.get("deconfounded_auroc")
                if deconf_val is not None:
                    if isinstance(causal_data, dict):
                        causal_data["deconfounded_auroc"] = deconf_val
                    else:
                        causal_data = {"deconfounded_auroc": deconf_val}

    # XGBoost full model
    xgb_model = "xgboost_classifier"
    for task in ("T1", "T4"):
        auroc = _get_metric(with_mon, task, xgb_model, "auroc")
        auprc = _get_metric(with_mon, task, xgb_model, "auprc")
        rows.append(
            {
                "Approach": "Full model (XGBoost)",
                "Task": task,
                "AUROC": _fmt(auroc),
                "AUPRC": _fmt(auprc),
                "Notes": "All features",
            }
        )

    # Provenance-free (environment-only) model
    for task in ("T1", "T4"):
        auroc = _get_metric(without_mon, task, xgb_model, "auroc")
        auprc = _get_metric(without_mon, task, xgb_model, "auprc")
        rows.append(
            {
                "Approach": "Provenance-free (XGBoost)",
                "Task": task,
                "AUROC": _fmt(auroc),
                "AUPRC": _fmt(auprc),
                "Notes": "Provenance features dropped (environment-only)",
            }
        )

    # ICP — source AUROC/AUPRC from results.json (canonical). After the M1
    # target-leakage fix, T4 was re-run and spliced into results.json (ICP T4
    # AUROC 0.607); icp_diagnostics.json was NOT in that re-splice, so its T4
    # entry is stale (0.623). Use icp_diagnostics.json only for the adversary-R2
    # annotation, which is a qualitative invariance diagnostic.
    for task in ("T1", "T4"):
        icp_auroc = _get_metric(with_mon, task, "icp_classifier", "auroc")
        icp_auprc = _get_metric(with_mon, task, "icp_classifier", "auprc")
        adv_r2 = icp_diag.get("adversary_r2", {}).get(task, float("nan"))
        notes = "Adversarially invariant"
        if not math.isnan(adv_r2):
            notes += f" (adv R\u00b2={adv_r2:.3f})"
        rows.append(
            {
                "Approach": "ICP",
                "Task": task,
                "AUROC": _fmt(icp_auroc),
                "AUPRC": _fmt(icp_auprc),
                "Notes": notes,
            }
        )

    # DML adjusted (T1 only typically)
    deconf_auroc = float("nan")
    if causal_data:
        # causal_data may have a top-level 'deconfounded_auroc' key
        if isinstance(causal_data, dict):
            deconf_auroc = float(causal_data.get("deconfounded_auroc", float("nan")))
        elif isinstance(causal_data, list) and causal_data:
            deconf_auroc = float(causal_data[0].get("deconfounded_auroc", float("nan")))
    rows.append(
        {
            "Approach": "DML adjusted",
            "Task": "T1",
            "AUROC": _fmt(deconf_auroc),
            "AUPRC": "\u2014",
            "Notes": "Post-hoc adjusted association",
        }
    )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp11_monitoring_invariance"
    df.to_csv(f"{path}.csv", index=False)
    df.to_latex(
        f"{path}.tex",
        index=False,
        caption="Monitoring-invariance decomposition.",
        label="tab:monitoring_invariance",
    )

    tabulate = importlib.import_module("tabulate")
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 13: Monitoring-Invariance Decomposition\n\n{table_str}\n",
        encoding="utf-8",
    )

    # Promoted to a main display item (Table 3) in the reframe; emit both names
    # so the skeleton's "Table 3" reference and legacy SI references resolve.
    main_path = output_dir / "table3_monitoring_invariance"
    df.to_csv(f"{main_path}.csv", index=False)
    df.to_latex(
        f"{main_path}.tex",
        index=False,
        caption="Monitoring-invariance decomposition.",
        label="tab:monitoring_invariance_main",
    )
    Path(f"{main_path}.md").write_text(
        f"## Table 3: Monitoring-Invariance Decomposition\n\n{table_str}\n",
        encoding="utf-8",
    )

    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_mcl_exceedance(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 14: MCL exceedance comparison."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    mcl_path = results_dir / "mcl_exceedance_analysis.json"
    rows: list[dict[str, str]] = []

    if mcl_path.exists():
        data = json.loads(mcl_path.read_text())
        for target in ("detection", "mcl_exceedance"):
            info = data.get(target, {})
            metrics = info.get("metrics", {})
            rows.append(
                {
                    "Target": target.replace("_", " ").title(),
                    "AUROC": f"{metrics.get('auroc', 0):.3f}",
                    "AUPRC": f"{metrics.get('auprc', 0):.3f}",
                    "F1": f"{metrics.get('f1', 0):.3f}",
                    "N Train": str(info.get("n_train", "—")),
                    "N Test": str(info.get("n_test", "—")),
                    "Pos Rate (test)": f"{info.get('positive_rate_test', 0):.1%}",
                }
            )
    else:
        rows.append({"Target": "—", "AUROC": "—", "AUPRC": "—", "F1": "—"})

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_mcl_exceedance"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 14: MCL Exceedance Comparison\n\n{md_text}\n"
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_power_analysis(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 15: Statistical power summary."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    pwr_path = results_dir / "power_analysis.json"
    # Fail loud on a missing/partial input rather than emitting a placeholder or
    # zero-filled row (R5 m3 — a silent fallback here is exactly how a wrong power
    # table shipped; the generator must break, not degrade).
    if not pwr_path.exists():
        raise FileNotFoundError(
            f"table_power_analysis: {pwr_path} missing — refusing to emit a placeholder table"
        )
    data = json.loads(pwr_path.read_text())
    core = data.get("core_tests", {})
    if not core:
        raise ValueError(f"table_power_analysis: no core_tests in {pwr_path}")
    rows: list[dict[str, str]] = []
    for test_name, test_data in core.items():
        missing = [k for k in ("effect_size", "power", "alpha") if k not in test_data]
        if missing:
            raise ValueError(f"table_power_analysis: {test_name} missing {missing} in {pwr_path}")
        rows.append(
            {
                "Test": test_name.replace("_", " ").title(),
                "Effect Size": f"{test_data['effect_size']:.4f}",
                "Power": f"{test_data['power']:.3f}",
                "Alpha": f"{test_data['alpha']:.2f}",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_power_analysis"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 15: Statistical Power Summary\n\n{md_text}\n"
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_sensitivity_bounds(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 16: Sensitivity bounds for top DML features."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    sb_path = results_dir / "sensitivity_bounds.json"
    rows: list[dict[str, str]] = []

    gamma_checkpoints = [1.0, 1.5, 2.0, 2.5, 3.0]

    if sb_path.exists():
        data = json.loads(sb_path.read_text())
        # Sort by E-value descending, take top 20 (skip namespaced metadata keys).
        feat_items = [(f, v) for f, v in data.items() if not f.startswith("_")]
        sorted_feats = sorted(feat_items, key=lambda x: x[1].get("e_value", 0), reverse=True)
        for feat, info in sorted_feats[:20]:
            gamma_star = info.get("gamma_star")
            row: dict[str, str] = {
                "Feature": feat,
                "Adjusted Effect": f"{info.get('causal_effect', 0):.4f}",
                "E-value": f"{info.get('e_value', 1):.2f}",
                "Γ*": f"{gamma_star:.2f}" if gamma_star is not None else "> 3.0",
            }
            # Add per-gamma significance columns
            bounds = {round(b["gamma"], 1): b for b in info.get("bounds_table", [])}
            for g in gamma_checkpoints:
                b = bounds.get(g)
                if b is not None:
                    row[f"Sig@Γ={g}"] = "✓" if b.get("significant", False) else "✗"
                else:
                    row[f"Sig@Γ={g}"] = "—"
            rows.append(row)
    else:
        rows.append({"Feature": "—", "Adjusted Effect": "—", "E-value": "—", "Γ*": "—"})

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_sensitivity_bounds"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(f"## Supplementary Table 16: Sensitivity Bounds\n\n{md_text}\n")
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_detection_only(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 17: Detection-only source ablation."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    do_path = results_dir / "detection_only_ablation.json"
    rows: list[dict[str, str]] = []

    if do_path.exists():
        data = json.loads(do_path.read_text())
        for entry in data:
            all_m = entry.get("all_sources", {})
            filt_m = entry.get("excluding_detection_only", {})
            delta_m = entry.get("delta", {})
            rows.append(
                {
                    "Model": entry.get("model", "—"),
                    "All AUROC": f"{all_m.get('auroc', 0):.3f}",
                    "Filtered AUROC": f"{filt_m.get('auroc', 0):.3f}" if filt_m else "—",
                    "Delta AUROC": f"{delta_m.get('auroc', 0):+.3f}" if delta_m else "—",
                    "N All": str(entry.get("n_all", "—")),
                    "N Filtered": str(entry.get("n_filtered", "—")),
                    "N Excluded": str(entry.get("n_excluded", "—")),
                }
            )
    else:
        rows.append(
            {
                "Model": "—",
                "All AUROC": "—",
                "Filtered AUROC": "—",
                "Delta AUROC": "—",
                "N All": "—",
                "N Filtered": "—",
                "N Excluded": "—",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_detection_only"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 17: Detection-Only Source Ablation\n\n{md_text}\n"
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_loro_detection_only(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 18: LORO per-region detection-only source ablation."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    do_path = results_dir / "loro_detection_only_ablation.json"
    rows: list[dict[str, str]] = []

    if do_path.exists():
        data = json.loads(do_path.read_text())
        for task_label, models in data.items():
            for model_name, mdata in models.items():
                for fold in mdata.get("folds", []):
                    all_m = fold.get("all_sources", {})
                    filt_m = fold.get("excluding_detection_only", {})
                    delta_m = fold.get("delta", {})
                    rows.append(
                        {
                            "Task": task_label,
                            "Model": model_name,
                            "Region": str(fold.get("test_region", "—")),
                            "All AUROC": f"{all_m.get('auroc', 0):.3f}",
                            "Filtered AUROC": (f"{filt_m.get('auroc', 0):.3f}" if filt_m else "—"),
                            "Delta AUROC": (f"{delta_m.get('auroc', 0):+.3f}" if delta_m else "—"),
                            "N All": str(fold.get("n_all", "—")),
                            "N Filtered": str(fold.get("n_filtered", "—")),
                            "N Excluded": str(fold.get("n_excluded", "—")),
                        }
                    )
    if not rows:
        rows.append(
            {
                "Task": "—",
                "Model": "—",
                "Region": "—",
                "All AUROC": "—",
                "Filtered AUROC": "—",
                "Delta AUROC": "—",
                "N All": "—",
                "N Filtered": "—",
                "N Excluded": "—",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_loro_detection_only"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 18: LORO Per-Region Detection-Only Ablation\n\n{md_text}\n"
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_shap_interactions(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 19: Top feature interaction pairs."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    si_path = results_dir / "shap_interactions_T1.json"
    rows: list[dict[str, str]] = []

    if si_path.exists():
        data = json.loads(si_path.read_text())
        for pair in data.get("top_interactions", [])[:15]:
            rows.append(
                {
                    "Feature A": pair.get("feature_a", "—"),
                    "Feature B": pair.get("feature_b", "—"),
                    "Interaction Strength": f"{pair.get('mean_abs_interaction', 0):.4f}",
                    "Category": pair.get("category", "—"),
                }
            )
    else:
        rows.append(
            {
                "Feature A": "—",
                "Feature B": "—",
                "Interaction Strength": "—",
                "Category": "—",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_interactions"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 19: Top Feature Interaction Pairs\n\n{md_text}\n"
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_loro_equity(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 20: LORO per-region EJ burden ratios."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    le_path = results_dir / "loro_equity_analysis.json"
    rows: list[dict[str, str]] = []

    # EPA region name mapping
    region_names = {
        1: "New England",
        2: "NY/NJ",
        3: "Mid-Atlantic",
        4: "Southeast",
        5: "Great Lakes",
        6: "South Central",
        7: "Central",
        8: "Mountain",
        9: "Pacific SW",
        10: "Pacific NW",
    }
    group_labels = {
        "pct_people_of_color": "People of Color",
        "pct_low_income": "Low Income",
        "pct_limited_english": "Limited English",
        "pct_less_hs_education": "Less HS Education",
    }

    if le_path.exists():
        data = json.loads(le_path.read_text())
        for entry in data.get("per_region", []):
            region = entry.get("region", "?")
            sig = "Yes" if entry.get("reject_fdr", False) else "No"
            rows.append(
                {
                    "Region": f"{region} ({region_names.get(region, '')})",
                    "Group": group_labels.get(entry.get("group", ""), entry.get("group", "—")),
                    "N Systems": str(entry.get("n_systems", "—")),
                    "N High": str(entry.get("n_high", "—")),
                    "N Low": str(entry.get("n_low", "—")),
                    "Burden Ratio": f"{entry.get('burden_ratio', 0):.2f}",
                    "p-value": f"{entry.get('p_value', 1.0):.4f}",
                    "p (FDR)": f"{entry.get('p_value_fdr', 1.0):.4f}",
                    "Sig.": sig,
                }
            )
    else:
        rows.append(
            {
                "Region": "—",
                "Group": "—",
                "N Systems": "—",
                "N High": "—",
                "N Low": "—",
                "Burden Ratio": "—",
                "p-value": "—",
                "p (FDR)": "—",
                "Sig.": "—",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_loro_equity"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 20: LORO Per-Region EJ Burden Ratios\n\n{md_text}\n"
    )
    logger.info("Table written: %s", f"{path}.csv")

    # Update skeleton.md placeholders with actual values
    if le_path.exists():
        _update_skeleton_loro_equity(le_path)

    return Path(f"{path}.csv")


def _update_skeleton_loro_equity(le_path: Path) -> None:
    """Replace ``[X.XX]`` LORO equity placeholders in skeleton.md."""
    import json
    import re

    skeleton_path = Path(__file__).resolve().parent / "skeleton.md"
    if not skeleton_path.exists():
        return

    data = json.loads(le_path.read_text())

    # Extract people-of-color burden ratios per region
    poc_entries = [
        e for e in data.get("per_region", []) if e.get("group") == "pct_people_of_color"
    ]
    if not poc_entries:
        return

    # Find min/max burden ratio entries
    min_entry = min(poc_entries, key=lambda e: e.get("burden_ratio", float("inf")))
    max_entry = max(poc_entries, key=lambda e: e.get("burden_ratio", 0))
    min_br = min_entry.get("burden_ratio", 0)
    max_br = max_entry.get("burden_ratio", 0)
    min_region = min_entry.get("region", "?")
    max_region = max_entry.get("region", "?")

    # Weighted national mean from summary
    summary = data.get("summary", {}).get("pct_people_of_color", {})
    weighted_mean = summary.get("weighted_mean_burden_ratio", summary.get("mean_burden_ratio", 0))

    text = skeleton_path.read_text()
    original = text

    # Replace the placeholder line:
    #   "yielding people-of-color burden ratios ranging from [X.XX] (Region [Y]) to"
    #   "[X.XX] (Region [Z]) with a weighted national mean of [X.XX] (Supplementary"
    old_pattern = (
        r"yielding people-of-color burden ratios ranging from \[X\.XX\] \(Region \[Y\]\) to\n"
        r"\[X\.XX\] \(Region \[Z\]\) with a weighted national mean of \[X\.XX\]"
    )
    new_text = (
        f"yielding people-of-color burden ratios ranging from {min_br:.2f} "
        f"(Region {min_region}) to\n"
        f"{max_br:.2f} (Region {max_region}) with a weighted national mean of "
        f"{weighted_mean:.2f}"
    )
    text = re.sub(old_pattern, new_text, text)

    # Only rewrite when a placeholder actually changed (see _update_skeleton_table6): an
    # unconditional write re-emits the file with the platform line ending, tripping G4's
    # PROSE-MUTATED check on Linux CI even though no content changed.
    if text != original:
        skeleton_path.write_text(text)
        logger.info("Updated LORO equity placeholders in %s", skeleton_path)


def table_multi_seed_stability(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 2: Multi-seed stability analysis."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    ms_path = results_dir / "multi_seed_stability.json"
    rows: list[dict[str, str]] = []

    if ms_path.exists():
        data = json.loads(ms_path.read_text())
        for task_name, models in sorted(data.items()):
            for model_name, info in sorted(models.items()):
                mean_auroc = info.get("mean_auroc", 0)
                std_auroc = info.get("std_auroc", 0)
                mean_auprc = info.get("mean_auprc", 0)
                std_auprc = info.get("std_auprc", 0)
                n_seeds = len(info.get("seeds", []))
                rows.append(
                    {
                        "Task": task_name,
                        "Model": model_name,
                        "AUROC (mean ± SD)": f"{mean_auroc:.3f} ± {std_auroc:.3f}",
                        "AUPRC (mean ± SD)": f"{mean_auprc:.3f} ± {std_auprc:.3f}",
                        "N Seeds": str(n_seeds),
                    }
                )
    else:
        rows.append(
            {
                "Task": "—",
                "Model": "—",
                "AUROC (mean ± SD)": "—",
                "AUPRC (mean ± SD)": "—",
                "N Seeds": "—",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_multi_seed"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(f"## Supplementary Table 2: Multi-Seed Stability\n\n{md_text}\n")
    logger.info("Multi-seed stability table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_t2_diagnostic(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 22: T2 concentration regression diagnostics."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    results_path = results_dir / "results.json"
    rows: list[dict[str, str]] = []

    if results_path.exists():
        data = json.loads(results_path.read_text())
        t2_results = [r for r in data if r.get("task") == "T2"]
        for r in t2_results:
            metrics = r.get("metrics", {})
            det_metrics = r.get("detected_only_metrics", {})
            rows.append(
                {
                    "Model": r.get("model", "—"),
                    "Overall R²": f"{metrics.get('r2', float('nan')):.3f}",
                    "Overall RMSE": f"{metrics.get('rmse', float('nan')):.4f}",
                    "Detected-only R²": (
                        f"{det_metrics.get('r2', float('nan')):.3f}" if det_metrics else "—"
                    ),
                    "Concordance Index": (
                        f"{metrics.get('concordance_index', float('nan')):.3f}"
                        if "concordance_index" in metrics
                        else "—"
                    ),
                }
            )
    if not rows:
        rows.append(
            {
                "Model": "—",
                "Overall R²": "—",
                "Overall RMSE": "—",
                "Detected-only R²": "—",
                "Concordance Index": "—",
            }
        )

    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_t2_diagnostic"
    df.to_csv(f"{path}.csv", index=False)
    md_text = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 22: T2 Regression Diagnostics\n\n{md_text}\n"
    )
    logger.info("T2 diagnostic table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_val_reuse_bias(
    output_dir: Path,
    *,
    results_dir: Path | None = None,
) -> Path:
    """Supplementary Table 23: Validation set reuse bias decomposition."""
    import json

    tabulate = importlib.import_module("tabulate")
    if results_dir is None:
        results_dir = Path("results")

    path = output_dir / "table_supp_val_reuse_bias"
    json_path = results_dir / "val_reuse_bias.json"

    rows: list[dict[str, str]] = []
    decomp_rows: list[dict[str, str]] = []

    if json_path.exists():
        data = json.loads(json_path.read_text())
        conditions = data.get("conditions", {})
        for label in ("triple_use", "internal_val", "no_es"):
            c = conditions.get(label, {})
            rows.append(
                {
                    "Condition": label.replace("_", " ").title(),
                    "Description": c.get("description", "—"),
                    "N train": str(c.get("n_train", "—")),
                    "N val": str(c.get("n_val", "—")),
                    "Mean AUROC": f"{c.get('mean_auroc', float('nan')):.3f}",
                    "SD AUROC": f"{c.get('std_auroc', float('nan')):.3f}",
                }
            )

        decomp = data.get("decomposition", {})
        for key in ("val_reuse_effect", "val_identity_effect", "es_benefit"):
            val = decomp.get(key, float("nan"))
            decomp_rows.append(
                {
                    "Component": key.replace("_", " ").title(),
                    "ΔAUROC": f"{val:+.4f}",
                    "Description": decomp.get(f"{key}_description", "—"),
                }
            )
        pct = decomp.get("pct_explained_by_val_reuse", float("nan"))
        decomp_rows.append(
            {
                "Component": "% of holdout-LORO gap explained",
                "ΔAUROC": f"{pct:.1f}%",
                "Description": f"Gap = {decomp.get('holdout_loro_gap', 0.145)}",
            }
        )
    else:
        rows.append(
            {
                "Condition": "—",
                "Description": "val_reuse_bias.json not found",
                "N train": "—",
                "N val": "—",
                "Mean AUROC": "—",
                "SD AUROC": "—",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{path}.csv", index=False)

    md_parts = ["## Supplementary Table 23: Validation Set Reuse Bias Decomposition\n"]
    md_parts.append("### Experimental Conditions\n")
    md_parts.append(tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False))
    if decomp_rows:
        df_d = pd.DataFrame(decomp_rows)
        md_parts.append("\n\n### Decomposition\n")
        md_parts.append(tabulate.tabulate(df_d, headers="keys", tablefmt="pipe", showindex=False))
    md_parts.append("\n")
    Path(f"{path}.md").write_text("\n".join(md_parts))
    logger.info("Val-reuse bias table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


ALL_TABLE_FUNCTIONS = [
    table_dataset_summary,
    table_benchmark_results,
    table_feature_importance,
    table_equity_summary,
    table_regional_performance,
    table_split_comparison,
    table_ext_feature_catalog,
    table_ext_per_analyte_results,
    table_external_validation,
    table_temporal_diagnosis,
    table_split_strategy_comparison,
    table_supp_split_sizematched,
    table_hyperparam_sensitivity,
    table_tuning_comparison,
    table_calibration_comparison,
    table_ext5_ablation,
    table_ext5_literature_comparison,
    table_monitoring_comparison,
    table_lift_analysis,
    table_sensitivity_analysis,
    table_causal_deconfounding,
    table_group_conformal,
    table_dedup_comparison,
    table_coordinate_sensitivity,
    table_benchmark_condensed,
    table_benchmark_full,
    table_monitoring_invariance,
    table_mcl_exceedance,
    table_power_analysis,
    table_sensitivity_bounds,
    table_detection_only,
    table_loro_detection_only,
    table_shap_interactions,
    table_loro_equity,
    table_multi_seed_stability,
    table_t2_diagnostic,
    table_val_reuse_bias,
]


def _load_table_data(
    results_dir: Path,
) -> dict[str, Any]:
    """Load pipeline outputs for table generation.

    Parameters
    ----------
    results_dir : Path
        Results directory from reproduce.py.

    Returns
    -------
    dict[str, Any]
        Keys: ``importance`` (pd.Series | None), ``equity`` (dict | None),
        ``random_results`` (list | None).
    """
    import json

    out: dict[str, Any] = {}

    # Feature importance (tree-based)
    importance_path = results_dir / "feature_importance.json"
    if importance_path.exists():
        imp_data = json.loads(importance_path.read_text())
        out["importance"] = pd.Series(imp_data, name="importance").sort_values(ascending=False)
    else:
        out["importance"] = None

    # SHAP values (preferred for paper tables)
    shap_path = results_dir / "shap_T1.json"
    if shap_path.exists():
        shap_data = json.loads(shap_path.read_text())
        shap_values = shap_data.get("mean_abs_shap", shap_data)
        out["shap"] = pd.Series(shap_values, name="shap").sort_values(ascending=False)
    else:
        out["shap"] = None

    # Equity analysis
    equity_path = results_dir / "equity_analysis.json"
    if equity_path.exists():
        out["equity"] = json.loads(equity_path.read_text())
    else:
        out["equity"] = None

    # Random-split results (for Table 5 comparison)
    random_results_path = results_dir / "results_random_split.json"
    if random_results_path.exists():
        out["random_results"] = json.loads(random_results_path.read_text())
    else:
        # Also check results_random/ directory
        alt_path = results_dir.parent / "results_random" / "results.json"
        if alt_path.exists():
            out["random_results"] = json.loads(alt_path.read_text())
        else:
            out["random_results"] = None

    # External validation
    ext_path = results_dir / "external_validation.json"
    if ext_path.exists():
        out["external_validation"] = json.loads(ext_path.read_text())
    else:
        out["external_validation"] = None

    # Split strategy comparison
    split_comp_path = results_dir / "split_comparison.json"
    if split_comp_path.exists():
        out["split_comparison"] = json.loads(split_comp_path.read_text())
    else:
        out["split_comparison"] = None

    # Calibration comparison
    cal_comp_path = results_dir / "calibration_comparison.json"
    if cal_comp_path.exists():
        out["calibration_comparison"] = json.loads(cal_comp_path.read_text())
    else:
        out["calibration_comparison"] = None

    # Hyperparam sensitivity
    sens_path = results_dir / "hyperparam_sensitivity.json"
    if sens_path.exists():
        out["hyperparam_sensitivity"] = json.loads(sens_path.read_text())
    else:
        out["hyperparam_sensitivity"] = None

    return out


_EJ_DIM_LABEL = {
    "pct_people_of_color": "People of color",
    "pct_low_income": "Low income",
    "pct_limited_english": "Limited English",
    "pct_less_hs_education": "Less than HS education",
}


def table_supp_group_error_calibration(output_dir: Path, results_dir: Path) -> Path:
    """Supplementary Table 25: per-group error rates and calibration (T1).

    Reads ``group_error_calibration.json`` (per-group FPR/FNR derived from frozen
    confusion-matrix summaries; ECE from a gated T1 reproduction).
    """
    import json

    def _f(v: Any) -> str:
        return "—" if v is None else f"{float(v):.3f}"

    # R5 m2: equal-mass (equal-count) ECE with bootstrap CIs, less bin-occupancy sensitive
    # than the equal-width ECE at small subgroup sizes.
    em_src = results_dir / "group_ece_debiased.json"
    ece_mass = json.loads(em_src.read_text()) if em_src.exists() else {}

    def _ece_mass_cell(col: str, level: str) -> str:
        g = (ece_mass.get(col) or {}).get(level) if isinstance(ece_mass.get(col), dict) else None
        if not g or g.get("ece_equal_mass") is None:
            return "—"
        ci = g.get("ece_equal_mass_ci") or []
        base = f"{float(g['ece_equal_mass']):.3f}"
        if len(ci) == 2:
            return f"{base} [{float(ci[0]):.3f}-{float(ci[1]):.3f}]"
        return base

    rows: list[dict[str, Any]] = []
    src = results_dir / "group_error_calibration.json"
    if src.exists():
        data = json.loads(src.read_text())
        for col, dim in data.get("error_rates", {}).items():
            for level in ("high", "low"):
                g = dim.get(level)
                if not g:
                    continue
                rows.append(
                    {
                        "Dimension": _EJ_DIM_LABEL.get(col, col),
                        "Group": level,
                        "N": int(g.get("n", 0)),
                        "AUROC": _f(g.get("auroc")),
                        "FPR": _f(g.get("fpr")),
                        "FNR": _f(g.get("fnr")),
                        "ECE (equal-width)": _f(g.get("ece")),
                        "ECE (equal-mass) [95% CI]": _ece_mass_cell(col, level),
                    }
                )
    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_group_error_calibration"
    df.to_csv(f"{path}.csv", index=False)
    tabulate = importlib.import_module("tabulate")
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 25: Per-Group Error Rates and Calibration (T1)\n\n{table_str}\n",
        encoding="utf-8",
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


def table_supp_areal_apportionment(output_dir: Path, results_dir: Path) -> Path:
    """Supplementary Table 26: areal-apportionment robustness of EJ burden ratios.

    Reads ``areal_apportionment.json`` (single-nearest baseline vs population-
    weighted areal apportionment at 5-km and 10-km buffers).
    """
    import json

    def _f(v: Any) -> str:
        return "—" if v is None else f"{float(v):.3f}"

    rows: list[dict[str, Any]] = []
    src = results_dir / "areal_apportionment.json"
    if src.exists():
        data = json.loads(src.read_text())
        base = data.get("baseline_single_nearest", {})
        app = data.get("apportioned", {})
        for col in (
            "pct_people_of_color",
            "pct_low_income",
            "pct_less_hs_education",
            "pct_limited_english",
        ):
            rows.append(
                {
                    "Dimension": _EJ_DIM_LABEL.get(col, col),
                    "Single-nearest": _f(base.get(col)),
                    "Apportioned 5 km": _f(app.get("5km", {}).get(col)),
                    "Apportioned 10 km": _f(app.get("10km", {}).get(col)),
                }
            )
    df = pd.DataFrame(rows)
    path = output_dir / "table_supp_areal_apportionment"
    df.to_csv(f"{path}.csv", index=False)
    tabulate = importlib.import_module("tabulate")
    table_str = tabulate.tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
    Path(f"{path}.md").write_text(
        f"## Supplementary Table 26: Areal-Apportionment Robustness of Burden Ratios"
        f"\n\n{table_str}\n",
        encoding="utf-8",
    )
    logger.info("Table written: %s", f"{path}.csv")
    return Path(f"{path}.csv")


@click.command()
@click.option("--results", default="results", help="Results directory from reproduce.py.")
@click.option("--output", "-o", default="paper/tables", help="Output directory.")
def main(results: str, output: str) -> None:
    """Generate all paper tables."""
    import json

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = Path(results)
    results_data: list[dict[str, Any]] = []
    results_json = results_path / "results.json"
    if results_json.exists():
        results_data = json.loads(results_json.read_text())

    table_data = _load_table_data(results_path)

    click.echo(f"Generating {len(ALL_TABLE_FUNCTIONS)} tables to {output_dir}")

    paths = []
    paths.append(table_dataset_summary(output_dir))
    paths.append(table_benchmark_results(results_data, output_dir))
    paths.append(
        table_feature_importance(
            table_data["importance"], output_dir, shap_df=table_data.get("shap")
        )
    )
    paths.append(table_equity_summary(table_data["equity"], output_dir))
    paths.append(table_regional_performance(results_data, output_dir))
    paths.append(table_split_comparison(results_data, table_data["random_results"], output_dir))
    paths.append(table_t6_arsenic(output_dir, results_dir=results_path))
    paths.append(table_ext_feature_catalog(output_dir))
    paths.append(table_ext_per_analyte_results(results_data, output_dir))
    paths.append(table_external_validation(output_dir, results_dir=results_path))
    paths.append(table_temporal_diagnosis(output_dir, results_dir=results_path))
    paths.append(table_split_strategy_comparison(output_dir, results_dir=results_path))
    sizematched_path = table_supp_split_sizematched(output_dir, results_dir=results_path)
    if sizematched_path is not None:
        paths.append(sizematched_path)
    paths.append(table_hyperparam_sensitivity(output_dir, results_dir=results_path))
    paths.append(table_tuning_comparison(output_dir, results_dir=results_path))
    paths.append(table_calibration_comparison(output_dir, results_dir=results_path))
    paths.append(table_ext5_ablation(output_dir, results_dir=results_path))
    paths.append(table_ext5_literature_comparison(output_dir, results_dir=results_path))
    paths.append(table_monitoring_comparison(output_dir, results_dir=results_path))
    paths.append(table_lift_analysis(output_dir, results_dir=results_path))
    paths.append(table_sensitivity_analysis(output_dir, results_dir=results_path))
    paths.append(table_causal_deconfounding(output_dir, results_dir=results_path))
    paths.append(table_group_conformal(output_dir, results_dir=results_path))
    paths.append(table_dedup_comparison(output_dir, results_dir=results_path))
    paths.append(table_coordinate_sensitivity(output_dir, results_dir=results_path))
    paths.append(table_benchmark_condensed(results_data, output_dir, results_dir=results_path))
    paths.append(table_benchmark_full(results_data, output_dir))
    paths.append(table_monitoring_invariance(output_dir, results_dir=results_path))
    paths.append(table_mcl_exceedance(output_dir, results_dir=results_path))
    paths.append(table_power_analysis(output_dir, results_dir=results_path))
    paths.append(table_sensitivity_bounds(output_dir, results_dir=results_path))
    paths.append(table_detection_only(output_dir, results_dir=results_path))
    paths.append(table_loro_detection_only(output_dir, results_dir=results_path))
    paths.append(table_shap_interactions(output_dir, results_dir=results_path))
    paths.append(table_loro_equity(output_dir, results_dir=results_path))
    paths.append(table_multi_seed_stability(output_dir, results_dir=results_path))
    paths.append(table_t2_diagnostic(output_dir, results_dir=results_path))
    paths.append(table_val_reuse_bias(output_dir, results_dir=results_path))
    paths.append(table_supp_group_error_calibration(output_dir, results_dir=results_path))
    paths.append(table_supp_areal_apportionment(output_dir, results_dir=results_path))

    for p in paths:
        click.echo(f"  {p}")

    click.echo("Done!")


if __name__ == "__main__":
    main()
