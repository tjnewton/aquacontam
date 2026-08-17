#!/usr/bin/env python
"""Regenerate LEADERBOARD.md deterministically from the FROZEN archive (referee M6).

The public leaderboard previously derived from a stale ``results/submissions/`` directory
that shipped the debunked pre-leakage-fix T4 numbers (xgboost lead AUPRC 0.8999, the very
target-leakage artefact the paper's T4 narrative debunks). This regenerates it from
``results/paper_frozen/results.json`` so every leaderboard number equals the paper's frozen
archive, with a PINNED timestamp so the output is byte-stable and diff-matchable (the
generated-artifact gate, like paper/tables/).

    python paper/regenerate_leaderboard.py           # rewrite LEADERBOARD.md from frozen
    python paper/regenerate_leaderboard.py --check    # regenerate in memory; exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from aquacontam.leaderboard.ranking import generate_leaderboard_document

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen" / "results.json"
OUT = REPO / "LEADERBOARD.md"
#: pinned so regeneration is deterministic (byte-stable diff-match); also records provenance.
PINNED_TS = "2026-08-05 — regenerated from the paper's frozen archive (results/paper_frozen)"
#: representative analyte per benchmark task (frozen results.json is per-task, not per-analyte).
TASK_ANALYTE = {
    "T1": "PFOS",
    "T2": "PFOA",
    "T3": "multi-PFAS",
    "T4": "lead",
    "T5": "transfer",
    "T6": "arsenic",
    "T7": "temporal",
}


def _submissions_from_frozen() -> dict[str, dict]:
    """Group frozen (task, model) rows into one submission dict per model."""
    rows = json.loads(FROZEN.read_text(encoding="utf-8"))
    by_model: dict[str, dict] = {}
    for r in rows:
        model, task = r.get("model"), r.get("task")
        if not model or not task:
            continue
        sub = by_model.setdefault(
            model, {"schema_version": "1.0", "model_name": model, "results": []}
        )
        sub["results"].append(
            {
                "task": task,
                "metrics": r.get("metrics", {}),
                "analyte": TASK_ANALYTE.get(task, task),
            }
        )
    return by_model


#: t_{0.975, 9} — the same critical value paper/compute_cluster_ci.py and generate_tables.py
#: use for the G = 10 EPA-region LORO folds (df = 9). Kept in sync deliberately (R5 M3).
_T_CRIT_9 = 2.262157162740992
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


def _t1_loro_section() -> str:
    """The canonical LORO-ranked T1 detection leaderboard, from the frozen archive (R5 M3).

    Primary ranking = leave-one-region-out mean AUROC with a t(9) cluster-robust CI; the
    fixed geographic-split (West-only) AUROC is shown alongside; non-converged families are
    flagged, never shown as a rankable number. Byte-identical to generate_tables.py's Table 2.
    """
    import math

    loro = json.loads((REPO / "results" / "paper_frozen" / "loro_cv_full.json").read_text())
    fixed = {
        r["model"]: r
        for r in json.loads(FROZEN.read_text(encoding="utf-8"))
        if r.get("task") == "T1"
    }
    t1 = loro["T1"]
    nc = loro["_meta"]["non_converged"]
    failed, partial = set(nc.get("failed", [])), nc.get("partial_folds", {})

    def fixed_auroc(short: str) -> str:
        v = fixed.get(_LORO_SHORT_TO_RESULTS[short], {}).get("metrics", {}).get("auroc")
        return f"{v:.3f}" if v is not None else "—"

    converged, flagged = [], []
    for short, entry in t1.items():
        name = _LORO_SHORT_TO_RESULTS[short]
        if short in failed:
            flagged.append((name, "—", fixed_auroc(short), "failed (CUDA)"))
        elif short in partial:
            flagged.append((name, "—", fixed_auroc(short), f"{partial[short]}/10 (10k cap)"))
        else:
            vals = [f["auroc"] for f in entry["folds"]]
            g = len(vals)
            mean = sum(vals) / g
            var = sum((x - mean) ** 2 for x in vals) / (g - 1)
            se = math.sqrt(var) / math.sqrt(g)
            lo, hi = mean - _T_CRIT_9 * se, mean + _T_CRIT_9 * se
            converged.append((name, mean, f"{mean:.3f} [{lo:.3f}-{hi:.3f}]", fixed_auroc(short)))
    converged.sort(key=lambda t: t[1], reverse=True)

    lines = [
        "## T1 PFAS-detection leaderboard (primary protocol: leave-one-region-out)",
        "",
        "Ranked by **leave-one-region-out (LORO) mean AUROC** across the 10 EPA regions — the",
        "leakage-resistant protocol this benchmark is built to reward. The fixed geographic-split",
        "AUROC (West-only test: EPA Regions 8/9/10) is shown alongside; its i.i.d. bootstrap CI is",
        "anti-conservative under residual spatial autocorrelation (Moran's I up to ~0.67), which is",
        "why LORO, not the single split, is primary. Tree ensembles are the strong baseline to beat;",
        "the top families are statistically comparable (the leaders differ by well under one AUROC",
        "point). Non-converged families are flagged, never shown as a metric.",
        "",
        "| Rank | Model | LORO AUROC (mean, 95% CI) | Fixed-split AUROC | LORO folds |",
        "|------|-------|---------------------------|-------------------|------------|",
    ]
    for i, (name, _m, cell, fx) in enumerate(converged, 1):
        lines.append(f"| {i} | {name} | {cell} | {fx} | 10/10 |")
    for name, cell, fx, flag in flagged:
        lines.append(f"| — | {name} | {cell} | {fx} | {flag} |")
    lines.append("")
    return "\n".join(lines)


def _submit_protocol() -> str:
    """The 'Submit a model' protocol — how a third party extends the benchmark (R5 M3)."""
    return "\n".join(
        [
            "## Submit a model",
            "",
            "The benchmark's primary evaluation is the leave-one-region-out (LORO) protocol above.",
            "To add your model to the T1 leaderboard against the same leakage-resistant evaluation:",
            "",
            "1. Implement your model against the `BaseModel` interface "
            "(`src/aquacontam/models/base.py`).",
            "2. Run it through the committed LORO harness — the same code path that produced this",
            "   table (`paper/compute_loro_full.py`, which calls `_run_loro_cv` over the 10 EPA-region",
            "   folds). Do not re-tune on the held-out region.",
            "3. Report the **LORO mean AUROC and its t(9) cluster-robust 95% CI**, plus the fixed",
            "   geographic-split AUROC for reference, and your per-fold AUROCs.",
            "4. Open a pull request adding your result. Include a reproducible config and a fixed seed",
            "   so the harness regenerates your numbers.",
            "",
            "Submissions that report only the single fixed split, or that re-tune on the test region,",
            "are not comparable and will be marked as such.",
            "",
        ]
    )


def _flag_t3_degenerate(doc: str) -> str:
    """Relabel the T3 rows of non-converged models (byte-identical fallback / CUDA) as
    'non-converged' in the multi-task context tables, so a silent fallback never shows as a
    plausible number (R5 M2). The degenerate set is detected from the frozen T3 metrics."""
    try:
        from paper.gate_lib import degenerate_metric_models
    except ImportError:  # pragma: no cover
        from gate_lib import degenerate_metric_models

    rows = [r for r in json.loads(FROZEN.read_text(encoding="utf-8")) if r.get("task") == "T3"]
    t3_metrics = {
        r["model"]: {
            "micro_auroc": r.get("metrics", {}).get("micro_auroc", float("nan")),
            "micro_auprc": r.get("metrics", {}).get("micro_auprc", float("nan")),
        }
        for r in rows
    }
    degenerate = degenerate_metric_models(t3_metrics, ("micro_auroc", "micro_auprc"))

    out, in_t3 = [], False
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.startswith("**T") and "(metric:" in stripped:
            in_t3 = stripped.startswith("**T3**")
        elif stripped.startswith("##"):
            in_t3 = False
        if in_t3 and line.startswith("|") and any(m in line for m in degenerate):
            # Replace the final value cell with a flag, keep the model/analyte cells.
            cells = line.split("|")
            if len(cells) >= 3:
                cells[-2] = " non-converged "
                line = "|".join(cells)
        out.append(line)
    return "\n".join(out)


def render() -> str:
    """Render the leaderboard markdown from the frozen archive (deterministic).

    Leads with the canonical LORO-ranked T1 leaderboard (the primary, leakage-resistant
    protocol) and a "Submit a model" protocol, then the multi-task per-task rankings as
    context with T3 non-convergence flagged (R5 M2/M3). Byte-stable via the pinned timestamp.
    """
    subs = _submissions_from_frozen()
    with tempfile.TemporaryDirectory() as td:
        for model, sub in subs.items():
            (Path(td) / f"{model}.json").write_text(json.dumps(sub), encoding="utf-8")
        base = generate_leaderboard_document(td, timestamp=PINNED_TS)

    base = _flag_t3_degenerate(base)
    # The frozen-archive regeneration has no submission files — say what it is.
    n_models = len(subs)
    base = base.replace(
        f"**{n_models} submissions** from {n_models} models",
        f"**{n_models} baseline models**, regenerated from the paper's frozen "
        "archive (`results/paper_frozen/`)",
        1,
    )
    # Insert the LORO T1 leaderboard + submit protocol right after the intro line and
    # before "## Overall Rankings" (which becomes multi-task context).
    marker = "## Overall Rankings"
    idx = base.index(marker)
    head, tail = base[:idx], base[idx:]
    tail = tail.replace(
        "## Overall Rankings",
        "## Multi-task rankings (context)\n\nThe per-task tables below are fixed-split "
        "reference results across all seven tasks; only T1 carries a converged LORO ranking "
        "(above) and T4 a three-model LORO check. T2/T3/T5/T7 are boundary or negative tasks.\n\n"
        "### Overall Rankings",
        1,
    )
    # Drop the generic "How to Submit" pointer; the concrete submit protocol lives up top.
    tail = tail.replace(
        "## How to Submit\n\nSee [`submissions/README.md`](submissions/README.md) for instructions.\n\n",
        "",
        1,
    )
    return head + _t1_loro_section() + "\n" + _submit_protocol() + "\n" + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Diff vs committed; exit 1 on drift.")
    args = ap.parse_args()
    doc = render()
    if args.check:
        committed = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if doc != committed:
            print("✗ LEADERBOARD.md differs from a fresh regeneration from the frozen archive")
            return 1
        print("✓ LEADERBOARD.md matches regeneration from the frozen archive")
        return 0
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT} from the frozen archive ({len(_submissions_from_frozen())} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
