"""Add the captured ICP training history to icp_diagnostics.json (additive).

Loads the frozen ``icp_diagnostics.json`` (which has only ``results``) and adds a
``history`` block from the seeded ICP re-fit (``results/icp_history.json``). Writes
``results/icp_diagnostics.json`` (staging) for freeze; the ``results`` block is preserved
verbatim so panel b's cited metrics are untouched.

Run::  python paper/splice_icp_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen" / "icp_diagnostics.json"
HIST = REPO / "results" / "icp_history.json"
OUT = REPO / "results" / "icp_diagnostics.json"


def main() -> int:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if not HIST.exists():
        raise SystemExit(f"missing {HIST} — run compute_icp_history.py first")
    hist = json.loads(HIST.read_text(encoding="utf-8"))
    merged = {
        "results": frozen["results"],  # verbatim — panel b metrics unchanged
        "history": {"T1": hist["T1"]},
        "history_meta": hist["_meta"],
    }
    OUT.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
    _meta = hist["_meta"]
    _anchor = _meta.get("test_auroc", _meta.get("val_auroc", "?"))
    print(
        f"Wrote {OUT}: results ({len(frozen['results'])} entries, verbatim) + history T1 "
        f"({len(hist['T1'])} epochs, test_auroc={_anchor})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
