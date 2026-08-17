#!/usr/bin/env python
"""Mechanical claim-band check (FINAL_FIX_CONTRACT_v2.md, bands B1-B7).

Run after the canonical re-freeze: each band is a closed-form predicate over the FRESH
frozen archive, anchored to POST-fix expected values. An EXPECTED-MOVE landing in range
is the fix working; a band breach means a central claim broke — the DAG driver HALTS and
escalates (never silently rewrites the thesis).

Exit 0 iff every band holds. ``--json PATH`` writes the machine-readable verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"


def _load(name: str) -> Any:
    with open(FROZEN / name, encoding="utf-8") as f:
        return json.load(f)


def _find_key(obj: Any, names: tuple[str, ...]) -> float | None:
    """Depth-first search for the first numeric value under any of `names`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            got = _find_key(v, names)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_key(v, names)
            if got is not None:
                return got
    return None


def _best_task_auroc(results: Any, task: str) -> float | None:
    """Max test AUROC across models for a task.

    results.json is a LIST of {task, model, metrics: {auroc, ...}} entries (the frozen
    slim format); a dict-of-tasks fallback is kept for robustness.
    """
    if isinstance(results, list):
        entries = [e for e in results if isinstance(e, dict) and e.get("task") == task]
    elif isinstance(results, dict) and isinstance(results.get(task), dict):
        entries = list(results[task].values())
    else:
        return None
    best: float | None = None
    for e in entries:
        auroc = _find_key(e.get("metrics", e), ("auroc", "test_auroc"))
        if auroc is not None and (best is None or auroc > best):
            best = auroc
    return best


def check_bands() -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []

    def band(band_id: str, desc: str, fn) -> None:
        try:
            ok, detail = fn()
        except (OSError, KeyError, TypeError, ValueError, StopIteration) as e:
            ok, detail = False, f"UNEVALUABLE ({type(e).__name__}: {e})"
        verdicts.append({"band": band_id, "desc": desc, "ok": bool(ok), "detail": detail})

    def b1():
        eq = _load("equity_analysis.json")
        poc = next(e for e in eq if e.get("group") == "pct_people_of_color")
        br, p = float(poc["burden_ratio"]), float(poc.get("p_value_fdr", poc.get("p_fdr", 1.0)))
        return br > 1.0 and p < 0.05, f"burden_ratio={br:.4f}, p_fdr={p:.2e}"

    band("B1", "POC monitoring burden > 1.0 and FDR-significant", b1)

    def b2():
        inf = _load("inference_strengthening.json")
        v = float(inf["m4c_burden_ratio_ci"]["weighted_mean_burden_ratio"])
        return v > 1.0, f"weighted_mean_burden_ratio={v:.4f}"

    band("B2", "National weighted burden ratio > 1.0", b2)

    def b3():
        inf = _load("inference_strengthening.json")
        m4a = inf["m4a_provenance_free_loro_ci"]
        lo = _find_key(m4a, ("ci_lower", "lower", "lo"))
        if lo is None:
            return False, "CI lower bound not found in m4a (fail-closed)"
        return lo > 0.5, f"m4a CI lower={lo:.4f}"

    band("B3", "PF out-of-region LORO 95% CI excludes 0.5", b3)

    def b4():
        gec = _load("group_error_calibration.json")
        poc = gec["error_rates"]["pct_people_of_color"]
        hi, lo = float(poc["high"]["fnr"]), float(poc["low"]["fnr"])
        return hi > lo, f"FNR high={hi:.4f} > low={lo:.4f} (gap {hi - lo:+.4f})"

    band("B4", "FNR-gap direction: high-POC > low-POC", b4)

    def b5():
        icp = _find_key(_load("icp_diagnostics.json"), ("t4_auroc",))
        if icp is None:
            # fall back to the paper_numbers loader source (same value the prose gates)
            sys.path.insert(0, str(REPO / "paper"))
            import paper_numbers as pn

            icp = float(str(pn.expected("t4_icp_auroc")))
        full = _best_task_auroc(_load("results.json"), "T4")
        if full is None:
            return False, "T4 best AUROC not found (fail-closed)"
        ok = icp <= 0.65 and (full - icp) >= 0.05
        return ok, f"ICP-T4={icp:.4f} (<=0.65), full-T4={full:.4f}, gap={full - icp:+.4f} (>=0.05)"

    band("B5", "T4 monitoring-invariant (env-only) AUROC <= 0.65 with >= 0.05 gap to full", b5)

    def b6():
        sc = _load("split_comparison.json")["simple"]
        r = float(sc["random_split_metrics"]["auroc"])
        g = float(sc["geographic_split_metrics"]["auroc"])
        return r > g, f"random={r:.4f} > geographic={g:.4f}"

    band("B6", "Spatial-leakage direction: random-split AUROC > geographic", b6)

    def b7():
        best = _best_task_auroc(_load("results_provenance_free.json"), "T1")
        if best is None:
            return False, "PF T1 best AUROC not found (fail-closed)"
        return best >= 0.70, f"PF T1 best AUROC={best:.4f} (>=0.70)"

    band("B7", "T1 provenance-free in-region AUROC >= 0.70", b7)

    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="Write the machine-readable verdict here.")
    args = ap.parse_args()

    verdicts = check_bands()
    print("Claim bands (FINAL_FIX_CONTRACT_v2 B1-B7):")
    for v in verdicts:
        print(f"  [{'HOLD' if v['ok'] else 'BREACH'}] {v['band']:3s} {v['desc']}: {v['detail']}")
    n_breach = sum(1 for v in verdicts if not v["ok"])
    if args.json:
        Path(args.json).write_text(
            json.dumps({"breaches": n_breach, "bands": verdicts}, indent=2) + "\n",
            encoding="utf-8",
        )
    if n_breach:
        print(f"✗ {n_breach} band breach(es) — HALT and escalate; do not rewrite the thesis.")
        return 1
    print("✓ all claim bands hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
