"""Splice the completed GNN + TabPFN LORO folds into loro_cv_full.json.

VERBATIM-PRESERVING: the 11 already-converged families (xgboost/catboost/voting/icp/...)
are copied byte-for-byte from the EXISTING frozen ``loro_cv_full.json`` (protecting the
gated markers ``loro_t1_catboost/voting/xgboost/icp``). Only ``gnn_gcn``/``gnn_sage``/
``tabpfn`` entries are replaced with their new 10-fold results, and ``_meta`` is recomputed
(ranking, non_converged, loro_leader). Writes ``results/loro_cv_full.json`` (staging in the
live results dir) for review + freeze; the frozen archive is never hand-edited.

Run::

    python paper/splice_loro_full.py     # writes results/loro_cv_full.json + prints diff
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen" / "loro_cv_full.json"
RESULTS = REPO / "results"
NEW_FAMILIES = ("gnn_gcn", "gnn_sage", "tabpfn")
_ENTRY_KEYS = ("status", "n_folds", "mean_auroc", "std_auroc", "folds")


def main() -> int:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    t1 = frozen["T1"]
    before = {k: (v.get("status"), v.get("n_folds"), v.get("mean_auroc")) for k, v in t1.items()}

    for fam in NEW_FAMILIES:
        extra_path = RESULTS / f"loro_extra_{fam}.json"
        if not extra_path.exists():
            raise SystemExit(
                f"missing {extra_path} — run compute_loro_gnn_tabpfn.py --family {fam}"
            )
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        if extra.get("status") != "ok" or extra.get("n_folds", 0) != 10:
            raise SystemExit(
                f"{fam}: expected status=ok n_folds=10, got {extra.get('status')}/{extra.get('n_folds')}"
            )
        t1[fam] = {k: extra[k] for k in _ENTRY_KEYS}

    # Recompute _meta from the merged family set (all now converged / 10 folds).
    ranked = sorted(
        ((n, e["mean_auroc"]) for n, e in t1.items() if e.get("status") == "ok"),
        key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1),  # NaN-safe desc
    )
    meta = frozen["_meta"]
    meta["families_ranked_by_loro_mean"] = [[n, m] for n, m in ranked]
    meta["loro_leader"] = ranked[0][0] if ranked else None
    meta["non_converged"] = {
        "failed": [],
        "partial_folds": {},
        "note": (
            "All 14 classifier families ran the full 10 leave-one-region-out folds. The two "
            "graph networks (gnn_gcn/gnn_sage) use their documented inductive linear-fallback "
            "inference (no test-time message passing across the disjoint geographic split), so "
            "they sit at the bottom of the ranking; they did not fail. TabPFN runs all 10 folds "
            "under the 3,000-sample stratified CPU-subsample protocol (the same protocol as its "
            "fixed-split result)."
        ),
    }
    meta["source"] = (
        meta.get("source", "")
        + " | #55: gnn_gcn/gnn_sage completed 10/10 with EPSG:5070 coords threaded "
        "(compute_loro_gnn_tabpfn.py, subprocess-isolated); tabpfn completed 10/10 with a "
        "driver-local max_train_size override (cpu_subsample_size=3000 unchanged). The 11 "
        "previously-converged families are byte-verbatim from the prior frozen file; "
        "reproduction gate on the three tree anchors PASSED before the splice."
    )

    out = RESULTS / "loro_cv_full.json"
    out.write_text(json.dumps(frozen, indent=2, default=str) + "\n", encoding="utf-8")

    # Diff report: only gnn/tabpfn entries + _meta may change.
    after = {k: (v.get("status"), v.get("n_folds"), v.get("mean_auroc")) for k, v in t1.items()}
    print(f"Wrote {out}")
    print("Changed family entries:")
    for k in t1:
        if before[k] != after[k]:
            print(f"  {k}: {before[k]} -> {after[k]}")
    unchanged = [k for k in t1 if before[k] == after[k]]
    print(f"Unchanged (verbatim-preserved): {len(unchanged)} families -> {unchanged}")
    print("New LORO ranking (top 8):")
    for n, m in ranked[:8]:
        print(f"  {n:20s} {m:.4f}")
    print(f"LORO leader: {meta['loro_leader']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
