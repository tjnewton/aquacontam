#!/usr/bin/env python
"""Standalone conformal rerun using existing results.json.

Reruns conformal prediction analysis (val calibration, test evaluation)
on existing results without re-importing the full reproduce.py pipeline.

Usage::

    python scripts/rerun_conformal.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _run_conformal_analysis(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run conformal prediction analysis at multiple alpha levels.

    Uses validation set for calibration and test set for evaluation,
    following proper split conformal inference.
    """
    from aquacontam.calibration.conformal import ConformalClassifier

    conformal_results: list[dict[str, Any]] = []

    for r in results:
        task = r.get("task", "")
        meta = r.get("metadata", {})
        if task == "T2":
            continue

        y_true = meta.get("y_true")
        y_prob = meta.get("y_prob")
        split_labels = meta.get("split_labels")
        if y_true is None or y_prob is None or split_labels is None:
            continue

        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_prob_arr = np.asarray(y_prob, dtype=float)
            split_arr = np.asarray(split_labels)

            val_mask = split_arr == "val"
            test_mask = split_arr == "test"

            if val_mask.sum() < 10 or test_mask.sum() < 10:
                logger.info(
                    "Insufficient val/test samples for conformal: %s/%s (val=%d, test=%d)",
                    task,
                    r.get("model", ""),
                    val_mask.sum(),
                    test_mask.sum(),
                )
                continue

            cal_y = y_true_arr[val_mask]
            cal_probs = y_prob_arr[val_mask]
            test_y = y_true_arr[test_mask]
            test_probs = y_prob_arr[test_mask]

            class _ProbaWrapper:
                """Wrap precomputed probabilities as a model-like object.

                Uses the input X directly as P(class=1) probabilities,
                so the same wrapper works for both calibration and test.
                """

                def predict_proba(self, X: Any) -> np.ndarray:
                    p = np.asarray(X, dtype=float)
                    return np.column_stack([1.0 - p, p])

            for alpha in (0.05, 0.10, 0.20):
                cc = ConformalClassifier(_ProbaWrapper(), alpha=alpha)
                cc.calibrate(cal_probs, cal_y)

                stats = cc.coverage_and_set_size(test_probs, test_y)

                conformal_results.append(
                    {
                        "task": task,
                        "model": r.get("model", ""),
                        "alpha": alpha,
                        "target_coverage": 1.0 - alpha,
                        "coverage": stats["coverage"],
                        "avg_set_size": stats["avg_set_size"],
                        "singleton_frac": stats["singleton_frac"],
                        "both_classes_frac": stats["both_classes_frac"],
                    }
                )
        except Exception:
            logger.warning("Conformal failed for %s/%s", task, r.get("model", ""), exc_info=True)

    if conformal_results:
        out_path = output_dir / "conformal_results.json"
        out_path.write_text(json.dumps(conformal_results, indent=2, default=str))
        logger.info("Conformal results saved to %s (%d entries)", out_path, len(conformal_results))


def main() -> None:
    results_dir = Path("results")
    results_file = results_dir / "results.json"

    if not results_file.exists():
        logger.error("results/results.json not found. Run --train-only first.")
        raise SystemExit(1)

    results = json.loads(results_file.read_text())
    logger.info("Loaded %d results from %s", len(results), results_file)

    _run_conformal_analysis(results, results_dir)

    out_path = results_dir / "conformal_results.json"
    if out_path.exists():
        conformal = json.loads(out_path.read_text())
        logger.info("Conformal results written: %d entries", len(conformal))
    else:
        logger.warning("No conformal results file was written")


if __name__ == "__main__":
    main()
