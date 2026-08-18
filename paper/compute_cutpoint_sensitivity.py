#!/usr/bin/env python
"""C4 (de novo M8a): cutpoint sensitivity of the equity burden ratios.

The paper's high/low burden contrast uses an asymmetric 80th-vs-50th-percentile split
(discarding the middle 30%); the equity module's own default is a symmetric median
split. This pre-specified compute re-derives every
demographic burden ratio under three cutpoint schemes — (80, 50) as published,
(50, 50) symmetric median, (75, 25) interquartile — from the same reproduced test-set
labels and demographics the frozen equity analysis used.

Reproduction gate: the (80, 50) people-of-color ratio must match the frozen
``equity_analysis.json`` within GATE_TOL, proving the reconstructed inputs align,
before any sensitivity value is written. Output goes directly into the frozen archive
(re-run ``freeze_results.py --rehash-only`` afterwards, then ``stamp_derivations.py``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "cutpoint_sensitivity.json"
GATE_TOL = 0.05

sys.path.insert(0, str(REPO / "paper"))
from compute_areal_apportionment import GROUP_COLS, _prep_test_systems  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#: (label, high_percentile, low_percentile)
SCHEMES = (
    ("published_80_50", 80.0, 50.0),
    ("median_50_50", 50.0, 50.0),
    ("interquartile_75_25", 75.0, 25.0),
)


def main() -> int:
    import numpy as np

    from aquacontam.analysis.equity import analyze_equity

    logger.info("Reconstructing the frozen equity test set (labels + demographics)...")
    _wq, _idx, y_test, demo = _prep_test_systems()
    dummy = np.zeros(len(y_test), dtype=int)

    # Reproduction gate: published cutpoints must reproduce the frozen POC ratio.
    frozen_eq = json.loads((FROZEN / "equity_analysis.json").read_text(encoding="utf-8"))
    frozen_poc = next(e for e in frozen_eq if e.get("group") == "pct_people_of_color")
    frozen_ratio = float(frozen_poc["burden_ratio"])
    rep = analyze_equity(y_test, dummy, None, demo, "pct_people_of_color")
    got = float(rep.burden_ratio)
    if abs(got - frozen_ratio) > GATE_TOL:
        logger.error(
            "Reproduction gate FAILED: (80,50) POC ratio %.4f vs frozen %.4f (tol %.2f) — "
            "inputs do not align; nothing written",
            got,
            frozen_ratio,
            GATE_TOL,
        )
        return 1
    logger.info("Reproduction gate OK: POC %.4f vs frozen %.4f", got, frozen_ratio)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for label, hi, lo in SCHEMES:
        per_group: dict[str, dict[str, float]] = {}
        for col in GROUP_COLS:
            if col not in demo.columns:
                continue
            r = analyze_equity(
                y_test, dummy, None, demo, col, high_percentile=hi, low_percentile=lo
            )
            per_group[col] = {
                "burden_ratio": float(r.burden_ratio),
                "n_high": int(r.n_high),
                "n_low": int(r.n_low),
            }
        results[label] = per_group
        logger.info(
            "%s: POC ratio %.3f (n_high=%d, n_low=%d)",
            label,
            per_group["pct_people_of_color"]["burden_ratio"],
            per_group["pct_people_of_color"]["n_high"],
            per_group["pct_people_of_color"]["n_low"],
        )

    poc = {s: results[s]["pct_people_of_color"]["burden_ratio"] for s, _h, _l in SCHEMES}
    payload = {
        "_meta": {
            "source": (
                "C4 (M8a) cutpoint sensitivity: burden ratios re-derived from the reproduced "
                "frozen-equity test set (labels + EJScreen demographics; predictions not "
                "required for burden ratios) under three cutpoint schemes. Reproduction gate: "
                f"(80,50) POC ratio matches equity_analysis.json within {GATE_TOL}."
            ),
            "reproduction_gate": {"poc_ratio": got, "frozen": frozen_ratio},
            "n_test_systems": len(y_test),
        },
        "schemes": {s: {"high_percentile": h, "low_percentile": lo} for s, h, lo in SCHEMES},
        "results": results,
        "poc_summary": {
            **poc,
            "direction_stable": bool(all(v > 1.0 for v in poc.values())),
            "min_ratio": min(poc.values()),
            "max_ratio": max(poc.values()),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)
    logger.info(
        "POC ratio across schemes: %s (direction stable above 1.0: %s)",
        {k: round(v, 3) for k, v in poc.items()},
        payload["poc_summary"]["direction_stable"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
