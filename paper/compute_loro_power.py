#!/usr/bin/env python
"""Re-derive the SI S19 LORO power from the REAL fold AUROCs.

The frozen ``power_analysis.json`` ``loro_tests`` block is placeholder-derived: every entry
carries ``mean_auroc = 0.0``, ``std_auroc = 0.01``, ``power_vs_chance = 1.0`` because the
generator (``pipeline/analysis/_evaluation.py``) read a nested ``summary`` key that
``loro_cv.json`` (a flat structure) does not contain, so all values fell back to defaults.
The SI S19 sentence ("all task/model combinations have power > 0.99") was therefore
placeholder-derived (and ungated).

The root cause is fixed in ``_evaluation.py`` (``.get("summary", model_data)``). This COMPUTE
re-derives the ``loro_tests`` block from the REAL committed ``loro_cv.json`` mean AUROCs
using the same power routine and the same generator arithmetic, SPLICING it into
``power_analysis.json`` while preserving ``core_tests`` and ``ablation_tests`` byte-verbatim
(they are already correct real values). Reproduction gate: the frozen ``loro_cv.json`` T1
xgboost mean AUROC must equal the committed anchor before anything is written, and
``core_tests`` / ``ablation_tests`` are re-emitted unchanged. Then ``freeze_results.py
--rehash-only`` and ``stamp_derivations.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
POWER_PATH = FROZEN / "power_analysis.json"
LORO_PATH = FROZEN / "loro_cv.json"
# Committed anchor: loro_cv.json T1 xgboost mean AUROC (frozen). Gate rejects a wrong input.
ANCHOR_KEY = ("T1", "xgboost")
ANCHOR_VALUE = 0.7824892795063257
ANCHOR_TOL = 1e-9

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from aquacontam.analysis.power_analysis import compute_auroc_comparison_power

    loro = json.loads(LORO_PATH.read_text(encoding="utf-8"))
    power = json.loads(POWER_PATH.read_text(encoding="utf-8"))

    # ---- reproduction gate ---------------------------------------------------
    t, m = ANCHOR_KEY
    anchor = loro.get(t, {}).get(m, {})
    # loro_cv.json is flat: mean_auroc lives directly on the model dict (this IS the bug the
    # generator missed). Mirror the fixed generator's read: summary = model.get("summary", model).
    summary = anchor.get("summary", anchor)
    obs_anchor = float(summary.get("mean_auroc", -1.0))
    if abs(obs_anchor - ANCHOR_VALUE) > ANCHOR_TOL:
        logger.error(
            "GATE FAILED (HALT): loro_cv %s/%s mean_auroc %.10f != anchor %.10f",
            t, m, obs_anchor, ANCHOR_VALUE,
        )
        return 2
    logger.info("Gate OK: loro_cv %s/%s mean_auroc %.10f == anchor", t, m, obs_anchor)

    # ---- re-derive loro_tests from real fold AUROCs (fixed generator arithmetic) --
    loro_power: list[dict[str, object]] = []
    for task_name, task_data in loro.items():
        for model_name, model_data in task_data.items():
            s = model_data.get("summary", model_data)
            mean_auroc = float(s.get("mean_auroc", 0.0))
            std_auroc = float(s.get("std_auroc", 0.01))
            n_folds = int(s.get("n_folds", 10))
            if std_auroc > 0 and n_folds > 1:
                pwr = compute_auroc_comparison_power(
                    mean_auroc, 0.5, n_folds * 200, n_folds * 200
                )
                loro_power.append(
                    {
                        "task": task_name,
                        "model": model_name,
                        "mean_auroc": mean_auroc,
                        "std_auroc": std_auroc,
                        "n_folds": n_folds,
                        "power_vs_chance": pwr["power"],
                    }
                )
    if not loro_power:
        logger.error("no loro_tests re-derived; aborting")
        return 1

    min_power = min(float(e["power_vs_chance"]) for e in loro_power)
    logger.info(
        "re-derived %d loro_tests entries; min power_vs_chance = %.6f", len(loro_power), min_power
    )
    for e in loro_power:
        logger.info(
            "  %s/%s mean_auroc=%.4f std=%.4f power=%.6f",
            e["task"], e["model"], e["mean_auroc"], e["std_auroc"], e["power_vs_chance"],
        )

    # ---- splice: preserve core_tests + ablation_tests verbatim ----------------
    core_before = json.dumps(power["core_tests"], sort_keys=True)
    abl_before = json.dumps(power["ablation_tests"], sort_keys=True)
    power["loro_tests"] = loro_power
    if json.dumps(power["core_tests"], sort_keys=True) != core_before:
        logger.error("core_tests mutated; aborting")
        return 1
    if json.dumps(power["ablation_tests"], sort_keys=True) != abl_before:
        logger.error("ablation_tests mutated; aborting")
        return 1

    POWER_PATH.write_text(json.dumps(power, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s (loro_tests re-derived; core/ablation verbatim)", POWER_PATH)
    logger.info("MIN_LORO_POWER=%.6f", min_power)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
