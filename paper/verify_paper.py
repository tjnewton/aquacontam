#!/usr/bin/env python
"""Verify paper skeleton consistency against pipeline outputs.

Checks figure/table file existence, sequential numbering, SHAP data
availability, and metric spot-checks against results.json.

Usage::

    python paper/verify_paper.py
    python paper/verify_paper.py --results-dir results/
    python paper/verify_paper.py --skeleton paper/skeleton.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

click = None
try:
    import click as _click

    click = _click
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------

#: Pattern for main figures: "Fig. 1", "Figure 1", "Fig. 2a"
MAIN_FIG_RE = re.compile(r"(?:Fig\.|Figure)\s+(\d+)", re.IGNORECASE)

#: Pattern for extended data figures: "Extended Data Fig. 1"
EXT_FIG_RE = re.compile(r"Extended\s+Data\s+(?:Fig\.|Figure)\s+(\d+)", re.IGNORECASE)

#: Pattern for main tables: "Table 1"
MAIN_TABLE_RE = re.compile(r"(?<!Extended Data )Table\s+(\d+)", re.IGNORECASE)

#: Pattern for extended data tables: "Extended Data Table 1"
EXT_TABLE_RE = re.compile(r"Extended\s+Data\s+Table\s+(\d+)", re.IGNORECASE)

#: Pattern for supplementary tables: "Supplementary Table 1"
SUPP_TABLE_RE = re.compile(r"Supplementary\s+Table\s+(\d+)", re.IGNORECASE)


def parse_figure_refs(text: str) -> tuple[set[int], set[int]]:
    """Extract main and extended figure numbers from skeleton text.

    Returns
    -------
    tuple[set[int], set[int]]
        ``(main_figs, ext_figs)`` — sets of referenced figure numbers.
    """
    # Parse extended first so we can exclude them from main matches
    ext_figs = {int(m) for m in EXT_FIG_RE.findall(text)}
    # Remove extended data refs to avoid double-counting
    cleaned = EXT_FIG_RE.sub("", text)
    main_figs = {int(m) for m in MAIN_FIG_RE.findall(cleaned)}
    return main_figs, ext_figs


def parse_table_refs(text: str) -> tuple[set[int], set[int]]:
    """Extract main and extended table numbers from skeleton text.

    Returns
    -------
    tuple[set[int], set[int]]
        ``(main_tables, ext_tables)`` — sets of referenced table numbers.
    """
    ext_tables = {int(m) for m in EXT_TABLE_RE.findall(text)}
    cleaned = EXT_TABLE_RE.sub("", text)
    # Also strip Supplementary Table refs to avoid false positives
    cleaned = SUPP_TABLE_RE.sub("", cleaned)
    main_tables = {int(m) for m in MAIN_TABLE_RE.findall(cleaned)}
    return main_tables, ext_tables


def check_sequential(numbers: set[int], label: str) -> list[str]:
    """Check that numbers form a contiguous 1..N sequence.

    Returns list of warning strings for any gaps.
    """
    if not numbers:
        return []
    warnings: list[str] = []
    max_n = max(numbers)
    for i in range(1, max_n + 1):
        if i not in numbers:
            warnings.append(f"Gap in {label} numbering: {label} {i} missing")
    return warnings


#: Definition headings, e.g. "### Supplementary Table 1: Title" or
#: "### Supplementary Table 13 / Table 3: Title" (separator may be ':' or ' /').
SUPP_TABLE_DEF_RE = re.compile(r"^#{2,4}\s+Supplementary\s+Table\s+(\d+)(b?)\b", re.MULTILINE)


def check_supplementary_table_sequence(supplementary_path: Path) -> list[str]:
    """Check Supplementary Table definitions are a contiguous 1..N, no dupes.

    Also flags any in-text ``Supplementary Table N`` reference that points past
    the last defined table. Returns warning strings (empty if all good).
    """
    if not supplementary_path.exists():
        return []
    text = supplementary_path.read_text(encoding="utf-8")
    defs = [int(n) for n, _suffix in SUPP_TABLE_DEF_RE.findall(text)]
    warnings: list[str] = []
    if not defs:
        return warnings
    # Duplicates
    seen: set[int] = set()
    for n in defs:
        if n in seen:
            warnings.append(f"Duplicate Supplementary Table definition: {n}")
        seen.add(n)
    # Contiguous 1..N
    warnings.extend(check_sequential(seen, "Supplementary Table"))
    # References must not exceed the largest defined table
    max_def = max(seen)
    refs = {int(m) for m in SUPP_TABLE_RE.findall(text)}
    for r in sorted(refs):
        if r > max_def:
            warnings.append(f"Supplementary Table {r} referenced but only {max_def} defined")
    return warnings


# ---------------------------------------------------------------------------
# File existence checks
# ---------------------------------------------------------------------------

_FIG_FILE_PATTERNS: dict[str, str] = {
    "main": "fig{n}_*",
    "ext": "fig_ext{n}_*",
}

_TABLE_FILE_PATTERNS: dict[str, str] = {
    "main": "table{n}_*",
    "ext": "table_ext{n}_*",
}


def check_figure_files(figs: set[int], figures_dir: Path, pattern_key: str = "main") -> list[str]:
    """Check that figure files exist for each referenced number."""
    warnings: list[str] = []
    pattern_tmpl = _FIG_FILE_PATTERNS[pattern_key]
    label = "Fig." if pattern_key == "main" else "Extended Data Fig."
    for n in sorted(figs):
        pattern = pattern_tmpl.format(n=n)
        matches = list(figures_dir.glob(pattern))
        if not matches:
            warnings.append(f"{label} {n}: no matching files in {figures_dir}")
    return warnings


def check_table_files(tables: set[int], tables_dir: Path, pattern_key: str = "main") -> list[str]:
    """Check that table files exist for each referenced number."""
    warnings: list[str] = []
    pattern_tmpl = _TABLE_FILE_PATTERNS[pattern_key]
    label = "Table" if pattern_key == "main" else "Extended Data Table"
    for n in sorted(tables):
        pattern = pattern_tmpl.format(n=n)
        matches = list(tables_dir.glob(pattern))
        if not matches:
            warnings.append(f"{label} {n}: no matching files in {tables_dir}")
    return warnings


# ---------------------------------------------------------------------------
# Metric spot-checks
# ---------------------------------------------------------------------------

#: Expected approximate metric values from pipeline (task → metric → value)
#: AUROC and AUPRC are tracked independently (may come from different models).
_EXPECTED_METRICS: dict[str, dict[str, float]] = {
    "T1": {"auroc": 0.864, "auprc": 0.781},  # XGBoost best (Minnesota MDH added in-split)
    "T2": {
        "rmse": 0.154
    },  # MLP regressor; PWS-scale (ambient sources are external-validation only)
    "T4": {
        "auroc": 0.704,  # voting ensemble
        "auprc": 0.403,  # CatBoost
    },  # SDWIS non-detects kept; coordinate-subsetting bug fixed (GNN-only)
}

#: Tolerance for metric comparison
_METRIC_TOLERANCE = 0.05


def extract_metrics_from_results(
    results_path: Path,
) -> dict[str, dict[str, float]]:
    """Extract per-task best metrics from results.json.

    Returns dict of task_name → {metric: value}.
    """
    if not results_path.exists():
        return {}

    results = json.loads(results_path.read_text(encoding="utf-8"))
    task_best: dict[str, dict[str, float]] = {}

    for r in results:
        task = r.get("task", "")
        metrics = r.get("metrics", {})
        if task not in task_best:
            task_best[task] = {}
        # Track best AUROC and best AUPRC independently (may come from different models)
        auroc = metrics.get("auroc")
        if auroc is not None and auroc > task_best[task].get("auroc", -1):
            task_best[task]["auroc"] = auroc
        auprc = metrics.get("auprc")
        if auprc is not None and auprc > task_best[task].get("auprc", -1):
            task_best[task]["auprc"] = auprc
        # Track best RMSE (lower is better) for regression tasks
        rmse = metrics.get("rmse")
        if rmse is not None and rmse < task_best[task].get("rmse", float("inf")):
            task_best[task]["rmse"] = rmse

    return task_best


def check_supplementary_metrics(
    supplementary_path: Path,
    results_path: Path,
) -> list[str]:
    """Cross-check supplementary information metrics against results.json.

    Extracts the Deep Tobit T1 AUROC from S13 and compares against
    the ``deep_tobit_classifier`` entry in results.json.

    Parameters
    ----------
    supplementary_path : Path
        Path to ``supplementary_information.md``.
    results_path : Path
        Path to ``results.json``.

    Returns
    -------
    list[str]
        Warning strings for any detected inconsistencies.
    """
    warnings: list[str] = []

    if not supplementary_path.exists():
        warnings.append(f"Supplementary file not found: {supplementary_path}")
        return warnings

    if not results_path.exists():
        return warnings  # already warned by check_metrics

    supp_text = supplementary_path.read_text(encoding="utf-8")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Extract Deep Tobit T1 AUROC from S13 text (e.g. "T1 AUROC (0.249")
    s13_match = re.search(r"T1\s+AUROC\s*\((\d+\.\d+)", supp_text)
    if s13_match is None:
        return warnings  # pattern not found, nothing to check

    s13_auroc = float(s13_match.group(1))

    # Find deep_tobit_classifier T1 result in results.json
    for r in results:
        model = r.get("model", "")
        task = r.get("task", "")
        if "deep_tobit" in model.lower() and "classifier" in model.lower() and task == "T1":
            actual_auroc = r.get("metrics", {}).get("auroc")
            if actual_auroc is not None and abs(actual_auroc - s13_auroc) > _METRIC_TOLERANCE:
                warnings.append(
                    f"S13 Deep Tobit T1 AUROC: supplementary says {s13_auroc:.3f}, "
                    f"results.json says {actual_auroc:.3f} "
                    f"(delta={actual_auroc - s13_auroc:+.3f}, "
                    f"tolerance={_METRIC_TOLERANCE})"
                )
            break

    return warnings


def check_metrics(
    results_path: Path,
) -> list[str]:
    """Spot-check key metrics from results.json against skeleton values."""
    warnings: list[str] = []
    actual = extract_metrics_from_results(results_path)

    if not actual:
        warnings.append(f"No results found at {results_path} — cannot verify metrics")
        return warnings

    for task, expected in _EXPECTED_METRICS.items():
        if task not in actual:
            warnings.append(f"Task {task} not found in results.json")
            continue
        for metric, exp_val in expected.items():
            act_val = actual[task].get(metric)
            if act_val is None:
                warnings.append(f"{task} {metric}: not found in results.json")
            elif abs(act_val - exp_val) > _METRIC_TOLERANCE:
                warnings.append(
                    f"{task} {metric}: skeleton says {exp_val:.3f}, "
                    f"results.json says {act_val:.3f} "
                    f"(delta={act_val - exp_val:+.3f}, tolerance={_METRIC_TOLERANCE})"
                )

    return warnings


def check_table2_rows(tables_dir: Path, results_path: Path) -> list[str]:
    """Validate EVERY per-model row of condensed Table 2 against results.json.

    Unlike :func:`check_metrics` (which only spot-checks best-across-models
    values), this parses the committed ``table2_benchmark_condensed.md`` and
    asserts each ``(task, model)`` AUROC/AUPRC matches ``results.json`` within a
    tight tolerance. It catches stale tables regenerated from a different run --
    the failure that put e.g. Stacking 0.872 and CNN1D 0.790 in the paper when no
    result file contained those values. Returns hard ERRORS on any mismatch.
    """
    errors: list[str] = []
    table_path = tables_dir / "table2_benchmark_condensed.md"
    if not table_path.exists() or not results_path.exists():
        return errors  # file existence is handled elsewhere (as warnings)

    try:
        from aquacontam._constants import MODEL_DISPLAY_NAMES
    except ImportError:
        return errors  # cannot map model codes -> display names without the package

    results = json.loads(results_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for r in results:
        pub = MODEL_DISPLAY_NAMES.get(r.get("model", ""), r.get("model", ""))
        lookup[(r.get("task", ""), pub)] = r.get("metrics", {})

    rows = [
        ln
        for ln in table_path.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith("|")
    ]
    if len(rows) < 3:
        return errors
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    try:
        i_task, i_model = header.index("Task"), header.index("Model")
        i_auroc, i_auprc = header.index("AUROC"), header.index("AUPRC")
    except ValueError:
        return errors  # unexpected header; structural checks live elsewhere

    tol = 1e-3
    float_re = re.compile(r"(\d+\.\d+)")
    for ln in rows[2:]:  # skip header + separator
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) <= max(i_task, i_model, i_auroc, i_auprc):
            continue
        task, model = cells[i_task], cells[i_model]
        actual = lookup.get((task, model))
        if actual is None:
            errors.append(f"Table 2 row ({task}, {model}) has no matching results.json entry")
            continue
        for metric, idx in (("auroc", i_auroc), ("auprc", i_auprc)):
            match = float_re.search(cells[idx])
            if not match:
                continue  # placeholder cell (e.g. "—")
            table_val = float(match.group(1))
            actual_val = actual.get(metric)
            if actual_val is not None and abs(table_val - float(actual_val)) > tol:
                errors.append(
                    f"Table 2 ({task}, {model}) {metric.upper()}: table {table_val:.4f} != "
                    f"results.json {float(actual_val):.4f} "
                    f"(|delta|={abs(table_val - float(actual_val)):.4f} > {tol})"
                )
    return errors


def check_monitoring_invariance_rows(tables_dir: Path, results_dir: Path) -> list[str]:
    """Validate the decomposition table's provenance-free rows (HARD ERRORS).

    Parses ``table_supp11_monitoring_invariance.csv`` and asserts each
    "Provenance-free" row's AUROC/AUPRC matches the frozen environment-only run
    (``results_provenance_free.json``) within a tight tolerance. An em-dash
    cell while the env-only results exist is also an error -- exactly the
    regression that dashed out Table 11 when the frozen snapshot predated the
    provenance-free run.
    """
    import csv

    errors: list[str] = []
    table_path = tables_dir / "table_supp11_monitoring_invariance.csv"
    pf_path = results_dir / "results_provenance_free.json"
    if not table_path.exists() or not pf_path.exists():
        return errors  # file existence is handled elsewhere (as warnings)

    pf_results = json.loads(pf_path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, float]] = {
        r.get("task", ""): r.get("metrics", {})
        for r in pf_results
        if r.get("model") == "xgboost_classifier"
    }

    tol = 1e-3
    with table_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("Approach") or "").startswith("Provenance-free"):
                continue
            task = (row.get("Task") or "").strip()
            actual = lookup.get(task)
            if actual is None:
                errors.append(
                    f"Decomposition table: no results_provenance_free.json entry "
                    f"for ({task}, xgboost_classifier)"
                )
                continue
            for metric in ("AUROC", "AUPRC"):
                cell = (row.get(metric) or "").strip()
                actual_val = actual.get(metric.lower())
                try:
                    table_val = float(cell)
                except ValueError:
                    errors.append(
                        f"Decomposition table ({task}) {metric} is {cell!r} while "
                        f"results_provenance_free.json has {actual_val} -- "
                        f"regenerate tables from the frozen snapshot"
                    )
                    continue
                if actual_val is not None and abs(table_val - float(actual_val)) > tol:
                    errors.append(
                        f"Decomposition table ({task}) {metric}: table {table_val:.4f} != "
                        f"results_provenance_free.json {float(actual_val):.4f} "
                        f"(|delta|={abs(table_val - float(actual_val)):.4f} > {tol})"
                    )
    return errors


def check_dml_significant_count(prose: str, results_dir: Path) -> list[str]:
    """Validate the cited DML significant-coefficient count against the JSON.

    Guards the bug class where prose said "50 of 126" while
    ``causal_deconfounding.json`` had 69 significant at FDR<0.05 (the 50 was the
    FDR<0.01 count). Any "N of 126" cited near a significance/FDR keyword must
    equal the FDR<0.05 count or the FDR<0.01 count from the frozen file.
    """
    errors: list[str] = []
    dml_path = results_dir / "causal_deconfounding.json"
    if not dml_path.exists():
        return errors
    dml = json.loads(dml_path.read_text(encoding="utf-8"))
    if not isinstance(dml, list) or not dml:
        return errors
    n_fdr05 = sum(1 for e in dml if e.get("significant_fdr") is True)
    n_fdr01 = sum(
        1
        for e in dml
        if isinstance(e.get("p_value_fdr"), (int, float)) and e["p_value_fdr"] < 0.01
    )
    valid = {n_fdr05, n_fdr01}
    total = len(dml)
    pattern = re.compile(rf"(\d+)\s+of\s+(?:the\s+)?{total}\b[^.]{{0,60}}", re.IGNORECASE)
    for m in pattern.finditer(prose):
        ctx = m.group(0).lower()
        # Only DML-significance claims; exclude Rosenbaum/Gamma sensitivity
        # sentences (which legitimately cite a different "N of 126" count).
        if "gamma" in ctx or "Γ" in m.group(0) or "rosenbaum" in ctx:
            continue
        if "fdr-corrected" in ctx or "coefficients are significant" in ctx:
            cited = int(m.group(1))
            if cited not in valid:
                errors.append(
                    f"DML significant-count '{cited} of {total}' does not match "
                    f"causal_deconfounding.json (FDR<0.05: {n_fdr05}, FDR<0.01: {n_fdr01})"
                )
    return errors


def check_sensitivity_gamma_count(prose: str, results_dir: Path) -> list[str]:
    """Validate the cited Rosenbaum Gamma=3.0 robustness count against the JSON.

    Guards the inversion bug where prose claimed "no feature remains significant
    at Gamma = 3.0" while ``sensitivity_bounds.json`` shows ~49 features robust
    through Gamma=3.0 (their bounds never cross the null in the tested range).
    """
    errors: list[str] = []
    sb_path = results_dir / "sensitivity_bounds.json"
    if not sb_path.exists():
        return errors
    sb = json.loads(sb_path.read_text(encoding="utf-8"))
    feats = {k: v for k, v in sb.items() if isinstance(v, dict) and "bounds_table" in v}
    if not feats:
        return errors
    robust = 0
    for v in feats.values():
        hi = [r for r in v.get("bounds_table", []) if r.get("gamma", 0) >= 2.999]
        if hi and all(r.get("significant") for r in hi):
            robust += 1
    # The false-inversion phrasing must not survive.
    if re.search(
        r"no feature[^.]{0,40}(?:remains? significant|survives?)[^.]{0,30}Γ\s*=\s*3", prose
    ):
        errors.append(
            f"Skeleton/SI claims 'no feature survives Gamma=3.0' but "
            f"sensitivity_bounds.json shows {robust} robust features"
        )
    # Any cited "N of 12x ... Gamma = 3.0" count must match.
    for m in re.finditer(r"(\d+)\s+of\s+12[0-9][^.]{0,80}", prose):
        ctx = m.group(0)
        if re.search(r"Γ\s*=\s*3|Gamma\s*=\s*3|Rosenbaum", ctx):
            cited = int(m.group(1))
            if abs(cited - robust) > 1:
                errors.append(
                    f"Sensitivity-bounds count '{cited}' near Gamma=3.0 does not match "
                    f"sensitivity_bounds.json robust count ({robust})"
                )
    return errors


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


def check_t6_arsenic(results_dir: Path, tables_dir: Path) -> list[str]:
    """Verify T6 arsenic-transfer artifacts and headline numbers.

    Confirms ``t6_arsenic.json`` exists, the geographic-leakage inflation is
    substantial (paper claims up to ~0.25 AUROC), the public→domestic transfer
    matches or exceeds the in-distribution geographic holdout for some model,
    and the T6 table is present.
    """
    import json

    warnings: list[str] = []
    t6_path = results_dir / "t6_arsenic.json"
    if not t6_path.exists():
        warnings.append(f"T6 results missing: {t6_path} — run reproduce.py --t6-arsenic")
        return warnings

    data = json.loads(t6_path.read_text(encoding="utf-8"))
    models = data.get("models", {})
    inflations = [
        e["leakage_inflation_auroc"]
        for e in models.values()
        if e.get("leakage_inflation_auroc") is not None
    ]
    if not inflations:
        warnings.append("T6: no leakage_inflation_auroc values in t6_arsenic.json")
    elif max(inflations) < 0.15:
        warnings.append(
            f"T6 max geographic-leakage inflation {max(inflations):.3f} < 0.15 "
            "— paper claims up to ~0.25"
        )

    transfer_ok = any(
        (e.get("transfer_minus_indist_auroc") or -1.0) >= 0.0 for e in models.values()
    )
    if models and not transfer_ok:
        warnings.append(
            "T6: zero-shot domestic transfer does not match/exceed the in-distribution "
            "geographic holdout for any model (paper claims it does)"
        )

    if not (tables_dir / "table_t6_arsenic.md").exists():
        warnings.append(
            "T6 table missing: paper/tables/table_t6_arsenic.md — run generate_tables.py"
        )
    return warnings


# Pre-fix / before-and-after literals the manuscript cites deliberately (e.g.
# "appearing to reach AUROC 0.962 ... corrected skill 0.677"). Text-vs-source
# reconciliation must never flag these as current-headline mismatches. Kept
# deliberately small (0.951/0.968/0.623 are legitimate CURRENT values elsewhere,
# so they are NOT allowlisted).
_INTENTIONAL_HISTORICAL_LITERALS: set[str] = {"0.962", "0.902"}


def _load_results_index(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Index a results-list JSON by ``(task, model)`` -> metrics dict."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("results", [])
    return {(r.get("task", ""), r.get("model", "")): r.get("metrics", {}) for r in rows}


def _precision_tol(num_str: str) -> float:
    """Half-ULP tolerance for a value written to the precision of ``num_str``.

    A cell printed as ``0.82`` (2 dp) is compared with tol 0.005; ``0.607``
    (3 dp) with tol 0.0005. Prevents false failures on lower-precision cells
    while still catching genuine disagreements.
    """
    decimals = len(num_str.split(".")[1]) if "." in num_str else 0
    return 0.5 * (10.0**-decimals) + 1e-9


def check_table3_invariance_rows(tables_dir: Path, results_dir: Path) -> list[str]:
    """Validate every row of Table 3 (monitoring invariance) against its source.

    Catches the ICP-T4 ``0.607``-in-text / stale-``0.623`` class: ICP rows are
    sourced from ``results.json`` (canonical, post-leakage-fix), NOT the stale
    ``icp_diagnostics.json``. Returns hard ERRORS on any mismatch.
    """
    errors: list[str] = []
    table_path = tables_dir / "table3_monitoring_invariance.md"
    if not table_path.exists():
        return errors

    main_idx = _load_results_index(results_dir / "results.json")
    pf_idx = _load_results_index(results_dir / "results_provenance_free.json")
    # (Approach, Task) -> (metrics-dict, source-label)
    row_source: dict[tuple[str, str], tuple[dict[str, float], str]] = {}
    for task in ("T1", "T4"):
        row_source[("Full model (XGBoost)", task)] = (
            main_idx.get((task, "xgboost_classifier"), {}),
            "results.json",
        )
        row_source[("Provenance-free (XGBoost)", task)] = (
            pf_idx.get((task, "xgboost_classifier"), {}),
            "results_provenance_free.json",
        )
        row_source[("ICP", task)] = (
            main_idx.get((task, "icp_classifier"), {}),
            "results.json",
        )

    float_re = re.compile(r"(\d+\.\d+)")
    rows = [ln for ln in table_path.read_text(encoding="utf-8").splitlines() if "|" in ln]
    for ln in rows[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 4:
            continue
        approach, task = cells[0], cells[1]
        src = row_source.get((approach, task))
        if src is None:
            continue  # DML-adjusted row checked elsewhere; others not sourced here
        metrics, label = src
        for metric, idx in (("auroc", 2), ("auprc", 3)):
            m = float_re.search(cells[idx])
            if not m:
                continue  # placeholder em-dash
            table_val = float(m.group(1))
            actual = metrics.get(metric)
            if actual is not None and abs(table_val - float(actual)) > _precision_tol(m.group(1)):
                errors.append(
                    f"Table 3 ({approach}, {task}) {metric.upper()}: table {table_val} != "
                    f"{label} {float(actual):.4f}"
                )
    return errors


#: Body superscript citation marker, e.g. ^12^ or ^9,10,11^ (comma-lists only;
#: the renumber emits no ranges). Author-affiliation markers in the byline are
#: excluded by starting the scan at the first Results-section citation.
_CITE_MARKER_RE = re.compile(r"\^(\d+(?:,\d+)*)\^")
#: Bibliography entry, e.g. "12. Author, ...".
_BIB_ENTRY_RE = re.compile(r"^(\d+)\.\s+\S", re.MULTILINE)


def check_bibliography_bijection(skeleton_text: str) -> list[str]:
    """Marker <-> bibliography-entry bijection + ascending first-appearance.

    Guards the reference renumber (and any future citation edit) against the
    silent-regression class it fixes: a dangling ``^N^`` with no entry, a
    bibliography entry never cited, or numbering that is not ascending by first
    appearance (Nature style). Structural only (no dash/prose sensitivity), so it
    is safe as a G2-gated HARD check. Returns hard ERRORS.
    """
    errors: list[str] = []

    m = re.search(r"^## References\s*$", skeleton_text, re.MULTILINE)
    if not m:
        # No bibliography section: only a defect if the text actually cites (a
        # bare fixture skeleton legitimately has neither). Dangling markers with
        # no reference list are reported; otherwise there is nothing to check.
        if _CITE_MARKER_RE.search(skeleton_text):
            return ["Bibliography: citation markers present but no '## References' section"]
        return []
    after = re.search(r"^## \w", skeleton_text[m.end() :], re.MULTILINE)
    body = skeleton_text[: m.start()]
    refs_block = (
        skeleton_text[m.end() : m.end() + after.start()] if after else skeleton_text[m.end() :]
    )

    # Skip the author byline / affiliation superscripts: start at the first
    # citation on or after the Introduction (the first '## '-headed section).
    intro = re.search(r"^## \w", body, re.MULTILINE)
    scan_from = intro.start() if intro else 0

    order: list[int] = []
    seen: set[int] = set()
    for mm in _CITE_MARKER_RE.finditer(body, scan_from):
        for tok in mm.group(1).split(","):
            n = int(tok)
            if n not in seen:
                seen.add(n)
                order.append(n)

    entry_nums = [int(x) for x in _BIB_ENTRY_RE.findall(refs_block)]
    entries = set(entry_nums)
    cited = set(order)

    if len(entry_nums) != len(entries):
        dupes = sorted({n for n in entry_nums if entry_nums.count(n) > 1})
        errors.append(f"Bibliography: duplicate entry numbers {dupes}")
    missing = cited - entries
    if missing:
        errors.append(f"Bibliography: cited but no entry (dangling markers): {sorted(missing)}")
    orphan = entries - cited
    if orphan:
        errors.append(f"Bibliography: entry never cited (orphan): {sorted(orphan)}")
    if entries and entries != set(range(1, max(entries) + 1)):
        errors.append(f"Bibliography: entry numbers are not contiguous 1..{max(entries)}")
    # Ascending by first appearance: the sequence of first-seen numbers must be 1,2,3,...
    if order != sorted(order) or order != list(range(1, len(order) + 1)):
        first_bad = next(
            (i + 1 for i, n in enumerate(order) if n != i + 1),
            None,
        )
        errors.append(
            "Bibliography: citations not in ascending order of first appearance "
            f"(first out-of-order at position {first_bad}: got {order[:12]}...)"
        )
    return errors


def check_stale_frozen_files(results_dir: Path) -> list[str]:
    """Flag frozen JSONs that still carry pre-leakage-fix T4 (~0.96) values.

    ``spatial_block_bootstrap.json`` and ``seed_inflation_check.json`` predate
    the target-leakage fix; a reader re-deriving a T4 CI from them would
    recover the artefact. Advisory WARNINGS (the files are intentionally retained
    as a provenance record, allowlisted by maintainer decision), plus a generic
    sweep that catches any other file whose T4 metric diverges far from the
    canonical ``results.json`` value.
    """
    warnings: list[str] = []
    main_idx = _load_results_index(results_dir / "results.json")
    canon = main_idx.get(("T4", "xgboost_classifier"), {}).get("auroc")
    if canon is None:
        return warnings

    def _leaves(obj: object, path: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _leaves(v, f"{path}.{k}")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            yield path, float(obj)

    for fname in sorted(p.name for p in results_dir.glob("*.json")):
        path = results_dir / fname
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        flagged = False
        for leaf_path, val in _leaves(data):
            lp = leaf_path.lower()
            # Only score METRIC leaves (auroc/auprc and their bootstrap summaries).
            # Metadata ratios in the same T4 blocks — ci = 0.95, width_ratio ≈ 0.93 —
            # sit in [0.90, 0.99] too and were this check's only triggers once the v2
            # re-freeze replaced the retained pre-fix files with fresh ones.
            last = lp.rsplit(".", 1)[-1]
            if not (
                "auroc" in last
                or "auprc" in last
                or last in ("mean", "ci_lower", "ci_upper", "point")
            ):
                continue
            if ("t4" in lp) and 0.90 <= val <= 0.99 and (val - float(canon)) > 0.15:
                warnings.append(
                    f"Stale T4 value in {fname} ({leaf_path.strip('.')}={val:.4f}): "
                    f"pre-leakage-fix artefact vs canonical results.json T4={float(canon):.3f}; "
                    f"not a current source (retained provenance record)"
                )
                flagged = True
                break
        if flagged:
            continue
    return warnings


def check_text_metric_reconciliation(prose: str, results_dir: Path) -> list[str]:
    """Reconcile current-headline metric values in prose with their frozen source.

    Curated anchors (not a blanket float scan, which is unreliable on a 6,000-word
    manuscript) tie a specific sentence to a frozen JSON value. Pre-fix literals
    in :data:`_INTENTIONAL_HISTORICAL_LITERALS` are skipped so the before/after
    narrative is never flagged. Catches the ICP-T4 ``0.607`` class plus the new
    M3/M4 numbers. Returns hard ERRORS.
    """
    errors: list[str] = []
    main_idx = _load_results_index(results_dir / "results.json")

    def _get(path: Path, *keys: str) -> float | None:
        if not path.exists():
            return None
        node: object = json.loads(path.read_text(encoding="utf-8"))
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return None
        return float(node) if isinstance(node, (int, float)) else None

    gec = results_dir / "group_error_calibration.json"
    aa = results_dir / "areal_apportionment.json"
    poc = ("error_rates", "pct_people_of_color")
    # (regex with N float groups, [N canonical values]); each group compared
    # positionally to its canonical value at the group's own printed precision.
    anchors: list[tuple[str, list[float | None]]] = [
        (
            r"monitoring-invariant ICP model\s+(\d\.\d{3})",
            [main_idx.get(("T4", "icp_classifier"), {}).get("auroc")],
        ),
        (
            r"false-negative\s+rate\s+(\d\.\d{2})\s+vs\s+(\d\.\d{2})",
            [_get(gec, *poc, "high", "fnr"), _get(gec, *poc, "low", "fnr")],
        ),
        (
            r"expected\s+calibration\s+error\s+(\d\.\d{2})\s+vs\s+(\d\.\d{2})",
            [_get(gec, *poc, "high", "ece"), _get(gec, *poc, "low", "ece")],
        ),
        (
            r"(\d\.\d{2})\s+single-nearest versus\s+(\d\.\d{2})\s+and\s+(\d\.\d{2})\s+at\s+5-km",
            [
                _get(aa, "baseline_single_nearest", "pct_people_of_color"),
                _get(aa, "apportioned", "5km", "pct_people_of_color"),
                _get(aa, "apportioned", "10km", "pct_people_of_color"),
            ],
        ),
    ]
    for pattern, canon_vals in anchors:
        for m in re.finditer(pattern, prose):
            for grp_i, canon in enumerate(canon_vals, start=1):
                if canon is None:
                    continue
                lit = m.group(grp_i)
                if lit in _INTENTIONAL_HISTORICAL_LITERALS:
                    continue
                cited = float(lit)
                if abs(cited - canon) > _precision_tol(lit):
                    errors.append(
                        f"Text reconciliation: '{m.group(0)[:60]}' cites {lit} but "
                        f"frozen source = {canon:.4f}"
                    )
    return errors


def verify_paper(
    skeleton_path: Path | None = None,
    results_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Run all paper consistency checks.

    Parameters
    ----------
    skeleton_path : Path, optional
        Path to skeleton.md. Defaults to ``paper/skeleton.md``.
    results_dir : Path, optional
        Path to pipeline output directory. Defaults to ``results/``.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(errors, warnings)`` — errors are hard failures, warnings are
        advisory.
    """
    paper_dir = Path("paper")
    if skeleton_path is None:
        skeleton_path = paper_dir / "skeleton.md"
    if results_dir is None:
        results_dir = Path("results/paper_frozen")

    errors: list[str] = []
    warnings: list[str] = []

    # Read skeleton
    if not skeleton_path.exists():
        errors.append(f"Skeleton not found: {skeleton_path}")
        return errors, warnings

    text = skeleton_path.read_text(encoding="utf-8")

    # Parse references
    main_figs, ext_figs = parse_figure_refs(text)
    main_tables, ext_tables = parse_table_refs(text)

    # Sequential numbering
    warnings.extend(check_sequential(main_figs, "Fig."))
    warnings.extend(check_sequential(ext_figs, "Extended Data Fig."))
    warnings.extend(check_sequential(main_tables, "Table"))
    warnings.extend(check_sequential(ext_tables, "Extended Data Table"))

    # File existence
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"

    if figures_dir.exists():
        errors.extend(check_figure_files(main_figs, figures_dir, "main"))
        errors.extend(check_figure_files(ext_figs, figures_dir, "ext"))
    else:
        warnings.append(f"Figures directory not found: {figures_dir}")

    if tables_dir.exists():
        errors.extend(check_table_files(main_tables, tables_dir, "main"))
        errors.extend(check_table_files(ext_tables, tables_dir, "ext"))
    else:
        warnings.append(f"Tables directory not found: {tables_dir}")

    # SHAP data check (Figure 3 dependency)
    shap_path = results_dir / "shap_T1.json"
    if not shap_path.exists():
        warnings.append(
            f"SHAP data missing: {shap_path} — Figure 3 (feature importance) "
            f"may not be reproducible without re-running the pipeline"
        )

    # Metric spot-checks
    results_path = results_dir / "results.json"
    if not results_path.exists():
        errors.append(
            f"results.json not found in {results_dir} — refusing to verify against a "
            f"missing/stale results dir. Pass --results-dir results/paper_frozen."
        )
        return errors, warnings
    warnings.extend(check_metrics(results_path))

    # Per-model Table 2 row validation against results.json (HARD ERRORS) --
    # catches a stale/regenerated-from-a-different-run benchmark table.
    errors.extend(check_table2_rows(tables_dir, results_path))

    # Decomposition-table provenance-free rows vs the frozen env-only run
    # (HARD ERRORS) -- catches a snapshot frozen before the provenance-free run.
    errors.extend(check_monitoring_invariance_rows(tables_dir, results_dir))

    # Full Table 3 row validation incl. ICP sourced from results.json (HARD
    # ERRORS) -- catches the ICP-T4 0.607-vs-stale-0.623 class.
    errors.extend(check_table3_invariance_rows(tables_dir, results_dir))

    # Stale pre-leakage-fix T4 (~0.96) values lingering in frozen JSONs (WARN).
    warnings.extend(check_stale_frozen_files(results_dir))

    # T6 arsenic transfer checks
    warnings.extend(check_t6_arsenic(results_dir, tables_dir))

    # Supplementary metrics cross-check
    supplementary_path = paper_dir / "supplementary_information.md"
    warnings.extend(check_supplementary_metrics(supplementary_path, results_path))

    # Supplementary Table numbering must be contiguous 1..N (post-renumbering gate)
    warnings.extend(check_supplementary_table_sequence(supplementary_path))

    # DML significant-count and Rosenbaum Gamma=3.0 robustness count vs the
    # frozen JSONs (HARD ERRORS) -- guards the 50-vs-69 and Gamma-inversion bugs
    # an earlier review pass caught. Check skeleton + supplementary prose.
    _supp = supplementary_path.read_text(encoding="utf-8") if supplementary_path.exists() else ""
    _prose = text + "\n" + _supp
    errors.extend(check_dml_significant_count(_prose, results_dir))
    errors.extend(check_sensitivity_gamma_count(_prose, results_dir))

    # Current-headline metric values in prose vs their frozen source (HARD
    # ERRORS): ICP-T4 0.607, per-group FNR/ECE (M4), areal apportionment (M3).
    errors.extend(check_text_metric_reconciliation(_prose, results_dir))

    # Reference marker <-> bibliography-entry bijection + ascending first-appearance
    # (HARD ERRORS) -- guards the renumber against dangling/orphan/out-of-order
    # regressions. Structural, no prose sensitivity.
    errors.extend(check_bibliography_bijection(text))

    # Single source of truth: every <!--pn:ID--> marker in the paper markdown must
    # equal its frozen-derived value, and every registry id must be marked (except
    # TABLE_ONLY). This is a decidable value comparison, not a brittle prose regex.
    try:
        from paper.paper_numbers import check_paper_numbers
    except ImportError:
        from paper_numbers import check_paper_numbers
    errors.extend(check_paper_numbers())

    # Phase 10 publication-readiness checks
    # 1. Abstract should no longer contain old T5 phrasing
    if "demonstrates shared geospatial risk factors" in text:
        warnings.append(
            "Abstract still contains old T5 phrasing: "
            "'demonstrates shared geospatial risk factors'"
        )

    # 2. Multi-seed stability results
    multi_seed_path = results_dir / "multi_seed_stability.json"
    if not multi_seed_path.exists():
        warnings.append(
            f"Multi-seed stability results missing: {multi_seed_path} — "
            f"run reproduce.py --multi-seed to generate"
        )

    # 3. Region heterogeneity results
    region_het_path = results_dir / "region_heterogeneity.json"
    if not region_het_path.exists():
        warnings.append(
            f"Region heterogeneity results missing: {region_het_path} — "
            f"run reproduce.py --region-heterogeneity to generate"
        )

    # 4. Extended Data Table 5 (literature comparison)
    lit_table_path = paper_dir / "tables" / "table_ext5_literature_comparison.csv"
    if not lit_table_path.exists():
        warnings.append(f"Literature comparison table missing: {lit_table_path}")

    # 5. New result files from Phase 9+ additions
    for fname in [
        "causal_deconfounding.json",
        "deconfounded_auroc.json",
        "group_conformal_results.json",
        "results_provenance_free.json",
        "loro_cv_provenance_free.json",
    ]:
        fpath = results_dir / fname
        if not fpath.exists():
            warnings.append(f"Result file missing: {fpath}")

    # 5b. S18-S22 supplementary analysis result files
    for fname, flag_name in [
        ("mcl_exceedance_analysis.json", "--mcl-exceedance"),
        ("power_analysis.json", "--power-analysis"),
        ("sensitivity_bounds.json", "--sensitivity-bounds"),
        ("seed_inflation_check.json", "--seed-inflation"),
    ]:
        fpath = results_dir / fname
        if not fpath.exists():
            warnings.append(
                f"S18-S22 result file missing: {fpath} — run reproduce.py {flag_name} to generate"
            )

    # 6. Stale text check — skeleton should reference sixteen model families
    _stale_counts = ("eight model families", "ten model families", "fifteen model families")
    if any(s in text.lower() for s in _stale_counts):
        warnings.append(
            "Skeleton contains a stale model family count — should be 'sixteen model families'"
        )

    # -----------------------------------------------------------------------
    # Phase: Nature Water submission-readiness checks
    # -----------------------------------------------------------------------

    # 7. EJ consistency check against equity_analysis.json
    equity_path = results_dir / "equity_analysis.json"
    if equity_path.exists():
        equity_data = json.loads(equity_path.read_text(encoding="utf-8"))
        if isinstance(equity_data, list):
            poc_entry = next(
                (e for e in equity_data if e.get("group") == "pct_people_of_color"),
                None,
            )
            if poc_entry:
                p_fdr = poc_entry.get("p_value_fdr", 1.0)
                # Skeleton should say p_FDR < 0.001 (actual is ~0.0004)
                if p_fdr >= 0.001:
                    warnings.append(f"EJ p_value_fdr is {p_fdr:.4f}, but skeleton claims < 0.001")

                # Burden ratio check
                burden_ratio = poc_entry.get("burden_ratio", 0)
                if burden_ratio > 0:
                    expected_br_str = f"{burden_ratio:.2f}"
                    if expected_br_str not in text and f"{burden_ratio:.1f}" not in text:
                        # Check what the skeleton actually claims
                        br_match = re.search(r"burden ratio[^.]*?(\d+\.\d+)", text, re.IGNORECASE)
                        if br_match:
                            skeleton_br = float(br_match.group(1))
                            if abs(skeleton_br - burden_ratio) > 0.1:
                                errors.append(
                                    f"EJ burden ratio: skeleton says {skeleton_br:.2f}, "
                                    f"actual is {burden_ratio:.2f}"
                                )

                # National (system-count-weighted) burden ratio check (robustness
                # §8-iii): verify the CANONICAL national ratio appears in a burden
                # context so a wrong Abstract value is caught. By design the two
                # estimators (loro_equity 1.48 vs inference_strengthening m4c 1.50) were
                # reconciled to the m4c weighted mean (the endorsed estimator with a CI);
                # it is also gated as pn:ej_burden_natl, so this cross-checks that gate.
                inf_path = results_dir / "inference_strengthening.json"
                if inf_path.exists():
                    inf = json.loads(inf_path.read_text(encoding="utf-8"))
                    nat = inf.get("m4c_burden_ratio_ci", {}).get("weighted_mean_burden_ratio")
                    if nat is not None and nat > 0:
                        num = re.escape(f"{nat:.2f}")
                        if not re.search(
                            rf"burden[^.]*?{num}|{num}[^.]*?burden", text, re.IGNORECASE
                        ):
                            errors.append(
                                f"National EJ burden ratio {nat:.2f} "
                                "(inference_strengthening.json m4c) not stated in a burden "
                                "context in the manuscript"
                            )

                # EJ sample sizes check
                n_high = poc_entry.get("n_high", 0)
                n_low = poc_entry.get("n_low", 0)
                rate_high = (
                    poc_entry.get("group_metrics", {}).get("high", {}).get("positive_rate", 0)
                )
                _ = poc_entry.get("group_metrics", {}).get("low", {}).get("positive_rate", 0)
                # Check for stale sample sizes
                for stale_n in ["1,482", "1,087", "437"]:
                    clean_n = stale_n.replace(",", "")
                    if (f"n_high = {stale_n}" in text or f"n_high ={stale_n}" in text) and int(
                        clean_n
                    ) != n_high:
                        errors.append(f"Skeleton says n_high={stale_n} but actual is {n_high}")
                    if (f"n_ref = {stale_n}" in text or f"n_ref ={stale_n}" in text) and int(
                        clean_n
                    ) != n_low:
                        errors.append(f"Skeleton says n_ref={stale_n} but actual is {n_low}")
                # Check for stale detection rates
                if rate_high > 0:
                    expected_pct = f"{rate_high * 100:.1f}%"
                    if "37.1%" in text and abs(rate_high - 0.371) > 0.02:
                        errors.append(
                            f"Skeleton says 37.1% high-burden detection rate "
                            f"but actual is {expected_pct}"
                        )
    else:
        warnings.append(f"Equity analysis missing: {equity_path}")

    # 8. LORO CV consistency check against loro_cv.json
    loro_path = results_dir / "loro_cv.json"
    if loro_path.exists():
        loro_data = json.loads(loro_path.read_text(encoding="utf-8"))
        # Use XGBoost as the canonical LORO model (reported in abstract/discussion)
        t1_loro = loro_data.get("T1", {}).get("xgboost", {})
        loro_mean = t1_loro.get("mean_auroc")
        loro_std = t1_loro.get("std_auroc")
        if loro_mean is not None:
            # Check for stale values (match LORO-context only, not CI bounds)
            if re.search(r"LORO[^.]*?0\.687|mean\s+AUROC\s+0\.687", text):
                errors.append(
                    f"Skeleton contains LORO mean AUROC 0.687 — actual is {loro_mean:.3f}"
                )
            if "0.048" in text and loro_std is not None:
                errors.append(f"Skeleton contains LORO SD 0.048 — actual is {loro_std:.3f}")
            # Internal consistency: every LORO mean cited in the paper must be
            # one of the computed values -- with-provenance (loro_cv.json) or
            # environment-only (loro_cv_provenance_free.json) XGBoost T1, in
            # UNWEIGHTED (fold-mean) or system-count-WEIGHTED form (the M4 fix
            # reports both models with the same weighted estimator; weighted
            # means live in spatial_block_bootstrap.json / inference_strengthening.json).
            expected_means = {f"{loro_mean:.3f}"}
            pf_loro_path = results_dir / "loro_cv_provenance_free.json"
            if pf_loro_path.exists():
                pf_loro = json.loads(pf_loro_path.read_text(encoding="utf-8"))
                pf_mean = pf_loro.get("T1", {}).get("xgboost", {}).get("mean_auroc")
                if pf_mean is not None:
                    expected_means.add(f"{pf_mean:.3f}")
            sbb_path = results_dir / "spatial_block_bootstrap.json"
            if sbb_path.exists():
                sbb = json.loads(sbb_path.read_text(encoding="utf-8"))
                wmean = sbb.get("T1_xgboost", {}).get("weighted_mean_auroc")
                if wmean is not None:
                    expected_means.add(f"{float(wmean):.3f}")
            inf_path = results_dir / "inference_strengthening.json"
            if inf_path.exists():
                inf = json.loads(inf_path.read_text(encoding="utf-8"))
                pf_wmean = inf.get("m4a_provenance_free_loro_ci", {}).get("weighted_mean_auroc")
                if pf_wmean is not None:
                    expected_means.add(f"{float(pf_wmean):.3f}")
            # Table 2 cites the full-family LORO leaderboard means (CatBoost 0.790,
            # voting 0.784, ...), so every family's leave-one-region-out mean is a computed value.
            full_loro_path = results_dir / "loro_cv_full.json"
            if full_loro_path.exists():
                full_loro = json.loads(full_loro_path.read_text(encoding="utf-8"))
                for fam in full_loro.get("T1", {}).values():
                    fm = fam.get("mean_auroc")
                    if isinstance(fm, (int, float)):
                        expected_means.add(f"{float(fm):.3f}")
            loro_patterns = re.findall(r"LORO[^.]*?(\d\.\d{3})", text)
            mean_pattern = re.findall(r"mean\s+AUROC\s+(\d\.\d{3})", text)
            unique_means = {p for p in loro_patterns + mean_pattern if 0.6 < float(p) < 0.85}
            unexpected = unique_means - expected_means
            if unexpected:
                warnings.append(
                    f"LORO mean value(s) {sorted(unexpected)} in skeleton are not in the "
                    f"computed set {sorted(expected_means)} "
                    f"(with-provenance / provenance-free XGBoost T1)"
                )
    else:
        warnings.append(f"LORO CV results missing: {loro_path}")

    # 9. Split comparison values check — every random/geographic AUROC the
    # prose cites in a split-comparison context must come from the snapshot
    # (guards the stale-literal bug class; same idiom as the LORO check above).
    split_comp_path = results_dir / "split_comparison.json"
    if split_comp_path.exists():
        split_data = json.loads(split_comp_path.read_text(encoding="utf-8"))
        simple = split_data.get("simple", {})
        rand_auroc = simple.get("random_split_metrics", {}).get("auroc")
        geo_auroc = simple.get("geographic_split_metrics", {}).get("auroc")
        if rand_auroc is not None and geo_auroc is not None:
            # Every AUROC the prose may legitimately cite from this comparison:
            # the simple-block headline pair plus every matrix arm (geographic,
            # random, and the size-matched control) across models/feature sets.
            expected_split = {f"{rand_auroc:.3f}", f"{geo_auroc:.3f}"}
            for entry in split_data.get("matrix", {}).get("results", []):
                a = entry.get("metrics", {}).get("auroc")
                if a is not None:
                    expected_split.add(f"{a:.3f}")
            # The canonical single-split holdout (Table 2) is cross-referenced
            # in the split-comparison prose; allow it too.
            res_path = results_dir / "results.json"
            if res_path.exists():
                for e in json.loads(res_path.read_text(encoding="utf-8")):
                    if e.get("task") == "T1" and e.get("model") == "xgboost_classifier":
                        ha = e.get("metrics", {}).get("auroc")
                        if ha is not None:
                            expected_split.add(f"{ha:.3f}")
                        break
            # Sentences citing the simple-block comparison ("random split ...
            # 0.xxx" / "geographic split ... 0.xxx" within one sentence).
            cited = re.findall(
                r"(?:random|geographic)[^.;]{0,120}?AUROC[^.;]{0,40}?(\d\.\d{3})", text
            )
            cited += re.findall(
                r"AUROC[^.;]{0,40}?(\d\.\d{3})[^.;]{0,120}?(?:random|geographic) split", text
            )
            stale = {
                c
                for c in cited
                if c not in expected_split
                # Only values plausibly from this comparison (not LORO/Table 2)
                and abs(float(c) - rand_auroc) < 0.12
            } - expected_split
            if stale:
                warnings.append(
                    f"Split-comparison AUROC value(s) {sorted(stale)} in skeleton are not "
                    f"in the computed set {sorted(expected_split)} (split_comparison.json)"
                )
    else:
        warnings.append(f"Split comparison missing: {split_comp_path}")

    # 10. Display item count. Nature Water Articles allow up to 8 display items
    # (figures, tables, or boxes); see https://www.nature.com/natwater/content.
    n_main_figs = len(main_figs)
    n_main_tables = len(main_tables)
    total_display = n_main_figs + n_main_tables
    if total_display > 8:
        errors.append(
            f"Too many display items for Article type: "
            f"{n_main_figs} figs + {n_main_tables} tables = {total_display} (max 8)"
        )

    # 11. Word count. Nature Water Articles allow up to 6,000 words of main text,
    # excluding abstract, references, and figure/table captions
    # (https://www.nature.com/natwater/content). We count the body between the
    # Abstract and Methods headings (Introduction + Results + Discussion).
    import re as _re

    # Extract main text: after Abstract, before Methods
    main_text = text
    if "## Methods" in main_text:
        main_text = main_text.split("## Methods")[0]
    if "## Abstract" in main_text:
        main_text = main_text.split("## Abstract")[1]

    # Remove markdown formatting artifacts
    clean = _re.sub(r"\^[\d,]+\^", "", main_text)  # remove superscript refs
    clean = _re.sub(r"[#*_`\[\]]", "", clean)  # remove markdown symbols
    clean = _re.sub(r"\(.*?\)", "", clean)  # remove parenthetical refs
    words = clean.split()
    word_count = len(words)
    if word_count > 6000:
        warnings.append(
            f"Main text word count (~{word_count}) exceeds the Nature Water "
            f"Article limit of 6,000 words"
        )

    # 11b. Abstract length (Nature Articles target ~150 words)
    if "## Abstract" in text:
        abstract_block = text.split("## Abstract", 1)[1]
        # Abstract runs to the first blank line followed by the intro body.
        abstract_block = _re.split(r"\n##\s|\nDrinking water contamination", abstract_block)[0]
        abs_clean = _re.sub(r"\^[\d,]+\^", "", abstract_block)
        abs_clean = _re.sub(r"[#*_`\[\]]", "", abs_clean)
        abstract_words = len(abs_clean.split())
        if abstract_words > 165:
            warnings.append(f"Abstract is ~{abstract_words} words; Nature Articles target ~150")

    # -----------------------------------------------------------------------
    # Phase: Audit-driven numerical consistency checks
    # -----------------------------------------------------------------------

    _TIGHT_TOLERANCE = 0.02

    # 12. External validation spot-check against external_validation.json
    ext_val_path = results_dir / "external_validation.json"
    if ext_val_path.exists():
        ext_val = json.loads(ext_val_path.read_text(encoding="utf-8"))
        # Values from the 2026-06-30 Minnesota-in-split rerun (B1).
        _expected_ext_val = {
            "ca_geotracker": 0.813,
            "mo_dnr": 0.829,
            "nj_dep": 0.789,
        }
        for source, expected_auroc in _expected_ext_val.items():
            entry = ext_val.get(source, {})
            actual_auroc = entry.get("metrics", {}).get("auroc")
            if actual_auroc is not None and abs(actual_auroc - expected_auroc) > _TIGHT_TOLERANCE:
                warnings.append(
                    f"External validation {source} AUROC: expected ~{expected_auroc:.3f}, "
                    f"got {actual_auroc:.3f}"
                )
        # NJ DEP should have metrics (not single-class)
        nj_entry = ext_val.get("nj_dep", {})
        if not nj_entry.get("metrics"):
            warnings.append("NJ DEP external validation has no metrics — expected AUROC ~0.760")

        # 12b. Bootstrap CIs should exist for databases with metrics
        for source in ("ca_geotracker", "nj_dep", "mo_dnr"):
            entry = ext_val.get(source, {})
            boot_ci = entry.get("bootstrap_ci", {})
            auroc_ci = boot_ci.get("auroc", {})
            if not auroc_ci.get("ci_lower"):
                warnings.append(f"{source} missing bootstrap CI for AUROC")
            elif auroc_ci["ci_lower"] >= auroc_ci["ci_upper"]:
                warnings.append(
                    f"{source} bootstrap CI inverted: "
                    f"[{auroc_ci['ci_lower']:.3f}, {auroc_ci['ci_upper']:.3f}]"
                )

        # 12c. CA GeoTracker DeLong chance test should exist
        ca_chance = ext_val.get("ca_geotracker", {}).get("chance_test")
        if ca_chance and ca_chance.get("p_value") is not None:
            import math

            if not math.isnan(ca_chance["p_value"]) and ca_chance["p_value"] < 0.05:
                warnings.append(
                    f"CA GeoTracker DeLong test vs chance: p={ca_chance['p_value']:.3f} "
                    "< 0.05 — unexpected; verify domain shift narrative"
                )
        elif ext_val.get("ca_geotracker", {}).get("metrics"):
            warnings.append("CA GeoTracker missing DeLong chance test")
    else:
        warnings.append(f"External validation results missing: {ext_val_path}")

    # 13. DML causal effect + scale-free rank comparison checks
    dml_path = results_dir / "causal_deconfounding.json"
    if dml_path.exists():
        dml_data = json.loads(dml_path.read_text(encoding="utf-8"))
        poc_entry = next(
            (e for e in dml_data if e.get("feature") == "pct_people_of_color"),
            None,
        )
        if poc_entry:
            # Coefficients are standardized (per 1-SD) since the DML
            # scale-invariance fix; check the skeleton's citation against the
            # JSON instead of a hardcoded pre-standardization value.
            causal_eff = poc_entry.get("causal_effect")
            cited = re.search(r"`?pct_people_of_color`?[^.]*?coefficient\s+(\d\.\d{3})", text)
            if (
                causal_eff is not None
                and cited is not None
                and abs(float(cited.group(1)) - causal_eff) > 0.0005
            ):
                warnings.append(
                    f"DML pct_people_of_color coefficient: skeleton cites "
                    f"{cited.group(1)}, causal_deconfounding.json has "
                    f"{causal_eff:.3f} (standardized, per 1-SD)"
                )
    else:
        warnings.append(f"Causal adjusted association results missing: {dml_path}")

    rank_path = results_dir / "dml_shap_rank_comparison.json"
    if rank_path.exists():
        rank_data = json.loads(rank_path.read_text(encoding="utf-8"))
        rho = rank_data.get("spearman_rho")
        if rho is not None and not (0.0 <= rho <= 0.40):
            warnings.append(f"DML Spearman rho: expected 0.0-0.40, got {rho:.2f}")
        # Verify the skeleton cites whichever demographic the frozen snapshot actually
        # upranks most (dynamic — the old hardcoded pct_less_hs_education expectation
        # was satisfied only by an incidental mention in the since-removed embedded
        # Extended Data, masking that the prose mis-stated the POC direction).
        demo = {fr["feature"]: fr["displacement"] for fr in rank_data.get("demographic_ranks", [])}
        top_demo = max(demo, key=demo.get) if demo else None
        if top_demo and top_demo not in text:
            warnings.append(
                f"Skeleton should cite the most-upranked demographic "
                f"({top_demo}, +{demo[top_demo]}) in the DML analysis"
            )
    else:
        warnings.append(f"DML rank comparison results missing: {rank_path}")

    # 14. Conformal coverage check against group_conformal_results.json
    conformal_path = results_dir / "group_conformal_results.json"
    if conformal_path.exists():
        conformal_data = json.loads(conformal_path.read_text(encoding="utf-8"))
        # Find alpha=0.05 entry
        alpha05 = next(
            (e for e in conformal_data if abs(e.get("alpha", 0) - 0.05) < 0.001),
            None,
        )
        if alpha05:
            coverage = alpha05.get("coverage")
            if coverage is not None and abs(coverage - 0.950) > _TIGHT_TOLERANCE:
                warnings.append(
                    f"Conformal coverage at alpha=0.05: expected ~0.950, got {coverage:.3f}"
                )
            # Check per-region coverage
            per_region = alpha05.get("per_region", {})
            for region, rdata in per_region.items():
                rcov = rdata.get("coverage", 0) if isinstance(rdata, dict) else 0
                if rcov < 0.90:
                    warnings.append(
                        f"Conformal coverage Region {region}: {rcov:.3f} < 0.90 target"
                    )
    else:
        warnings.append(f"Conformal results missing: {conformal_path}")

    # 15. Lift analysis check against lift_analysis.json
    lift_path = results_dir / "lift_analysis.json"
    if lift_path.exists():
        lift_data = json.loads(lift_path.read_text(encoding="utf-8"))
        full_capture = lift_data.get("full_model", {}).get("top_quintile_capture")
        if full_capture is not None and abs(full_capture - 0.519) > 0.03:
            warnings.append(
                f"Full model top quintile capture: expected ~0.519, got {full_capture:.3f}"
            )
        mf_capture = lift_data.get("monitoring_free", {}).get("top_quintile_capture")
        if mf_capture is not None and abs(mf_capture - 0.479) > 0.03:
            warnings.append(
                f"Monitoring-free top quintile capture: expected ~0.479, got {mf_capture:.3f}"
            )
    else:
        warnings.append(f"Lift analysis results missing: {lift_path}")

    # 16. LORO SD tighter check — uses XGBoost (canonical)
    if loro_path.exists():
        loro_data_2 = json.loads(loro_path.read_text(encoding="utf-8"))
        t1_loro_2 = loro_data_2.get("T1", {}).get("xgboost", {})
        loro_std_2 = t1_loro_2.get("std_auroc")
        if loro_std_2 is not None and abs(loro_std_2 - 0.113) > _TIGHT_TOLERANCE:
            warnings.append(f"LORO CV T1 XGBoost std_auroc: expected ~0.113, got {loro_std_2:.3f}")

    # 17. Table 1 record counts vs processed parquet files
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pq = None  # type: ignore[assignment]

        if pq is not None:
            _TABLE1_EXPECTED_COUNTS: dict[str, int] = {
                "ucmr5": 1_928_117,
                "ucmr3": 1_069_174,
                "sdwis": 916_899,
                "mi_mpart": 5_640,
                "ca_geotracker": 324_254,
                "wa_doh": 9_251,
                "nj_dep": 248_107,
                "wqp": 35_528,
                "mo_dnr": 75_971,
                "mn_mdh": 247_230,
            }
            for source, expected_count in _TABLE1_EXPECTED_COUNTS.items():
                parquet_path = processed_dir / f"{source}.parquet"
                if parquet_path.exists():
                    actual_count = pq.read_metadata(parquet_path).num_rows
                    if abs(actual_count - expected_count) / expected_count > 0.05:
                        warnings.append(
                            f"Table 1 vs parquet {source}: Table 1 says {expected_count:,}, "
                            f"parquet has {actual_count:,} rows"
                        )

            # Check OH EPA is excluded (parquet may still exist but not in pipeline)
            oh_parquet = processed_dir / "oh_epa.parquet"
            if oh_parquet.exists():
                oh_count = pq.read_metadata(oh_parquet).num_rows
                if oh_count > 0:
                    warnings.append(
                        f"OH EPA parquet exists ({oh_count:,} rows) but should be "
                        f"excluded from pipeline (unavailable: true in config)"
                    )

    # 18. Extended Data Table 5 consistency with external_validation.json
    ext_data_path = Path("paper/extended_data.md")
    if ext_data_path.exists() and ext_val_path.exists():
        ext_data_text = ext_data_path.read_text(encoding="utf-8")
        ext_val_data = json.loads(ext_val_path.read_text(encoding="utf-8"))
        _edt5_checks = {
            "CA GeoTracker": ("ca_geotracker", 0.813),
            "MO DNR": ("mo_dnr", 0.829),
            "NJ DEP": ("nj_dep", 0.789),
        }
        for label, (key, _expected) in _edt5_checks.items():
            actual = ext_val_data.get(key, {}).get("metrics", {}).get("auroc")
            if actual is not None:
                # Check if extended_data.md contains stale values
                edt5_match = re.search(
                    rf"{re.escape(label)}\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(\d\.\d+)",
                    ext_data_text,
                )
                if edt5_match:
                    edt5_val = float(edt5_match.group(1))
                    if abs(edt5_val - actual) > 0.02:
                        errors.append(
                            f"EDT5 {label} AUROC: extended_data.md says {edt5_val:.3f}, "
                            f"actual is {actual:.3f}"
                        )

    # 19. Check for stale "160,000" SDWIS system count in non-intro context
    sdwis_160k = re.findall(r"(?:SDWIS|compliance monitoring)[^.]*160.000", text)
    if sdwis_160k:
        warnings.append(
            "Skeleton references '160,000' near SDWIS — "
            "SDWIS has 916,899 records, not 160,000 systems"
        )

    return errors, warnings


def _main_cli() -> None:
    """CLI entry point (works with or without click)."""
    skeleton = Path("paper/skeleton.md")
    results_dir = Path("results/paper_frozen")

    # Simple argv parsing when click is not available
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--skeleton" and i + 1 < len(args):
            skeleton = Path(args[i + 1])
            i += 2
        elif args[i] == "--results-dir" and i + 1 < len(args):
            results_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    errors, warnings = verify_paper(skeleton, results_dir)

    if warnings:
        print(f"\n⚠  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n✗  {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        n_checks = len(warnings) + 1  # at least skeleton read
        print(f"\n✓  Paper verification passed ({n_checks} checks, {len(warnings)} warnings)")
        sys.exit(0)


if __name__ == "__main__":
    _main_cli()
