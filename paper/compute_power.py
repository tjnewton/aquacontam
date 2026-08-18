#!/usr/bin/env python
"""Regenerate power_analysis.json from the frozen benchmark, not stale defaults.

The supplementary statistical-power table was computed from hardcoded AUROCs (0.873/0.857)
because the loader read a ``delong_results.json`` that never existed and silently fell back.
The frozen benchmark's top two T1 models are 0.8636 (XGBoost) and 0.8564 (TabPFN) — an effect
of ~0.0072, not 0.016. ``power_analysis._load_delong_params`` has been fixed to read the pair
from the committed ``results.json`` and to fail loud if it is absent; this script regenerates
the frozen ``power_analysis.json`` accordingly (and the power table regenerates from it under
gate clause G4).

Reproduction gate: the regenerated delong effect size equals the frozen results.json T1
top-two AUROC gap before anything is written — proving the fix reads the benchmark, not a
hardcoded pair.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
GATE_TOL = 1e-4

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_power")


def main() -> int:
    from aquacontam.analysis.power_analysis import summarize_power
    from aquacontam.pipeline.analysis._evaluation import _run_power_analysis

    results = json.loads((FROZEN / "results.json").read_text(encoding="utf-8"))
    t1 = sorted(
        (
            float(e["metrics"]["auroc"])
            for e in results
            if e.get("task") == "T1" and e.get("metrics", {}).get("auroc") is not None
        ),
        reverse=True,
    )
    expected_effect = t1[0] - t1[1]
    logger.info(
        "frozen T1 top-2 AUROC: %.4f, %.4f -> expected delong effect %.4f",
        t1[0],
        t1[1],
        expected_effect,
    )

    # --- reproduction gate: derived effect matches the frozen benchmark gap ---
    core = summarize_power(FROZEN)
    got = core["delong_auroc"]["effect_size"]
    if abs(got - expected_effect) > GATE_TOL:
        logger.error(
            "REPRODUCTION GATE FAILED: delong effect %.5f vs frozen T1 gap %.5f — nothing written",
            got,
            expected_effect,
        )
        return 1
    logger.info("Reproduction gate OK: delong effect %.5f reproduces the frozen T1 gap", got)

    _run_power_analysis(FROZEN)  # writes power_analysis.json with the corrected core_tests
    written = json.loads((FROZEN / "power_analysis.json").read_text(encoding="utf-8"))
    d = written["core_tests"]["delong_auroc"]
    logger.info(
        "wrote power_analysis.json: delong effect %.5f, power %.3f (was 0.016 / 0.423)",
        d["effect_size"],
        d["power"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
