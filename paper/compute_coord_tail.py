#!/usr/bin/env python
"""C3 (de novo M6b): coordinate-perturbation robustness extended to the empirical error tail.

The environmental features use ZCTA-centroid coordinates with heavy-tailed geocoding error
(median ~6 km, p95 ~270 km). The published perturbation check only goes to 20 km, so it
does not probe the tail where the "monitoring dominates environment" measurement-precision
asymmetry would bite. This pre-specified COMPUTE re-runs the same
``run_coordinate_sensitivity`` machinery (same model config, seeds, analyte) but extends
the magnitudes to the empirical tail (50, 100, 270 km).

Reproduction gate: the 0 km (unperturbed) AUROC must match the frozen T1 headline within
GATE_TOL, proving the harness reproduces the published model, before the tail magnitudes
are written. Output goes directly into the frozen archive (then ``freeze_results.py
--rehash-only`` and ``stamp_derivations.py``).

Heavy: feature perturbation + retrain at each (magnitude, seed). Run DETACHED.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "coordinate_tail_sensitivity.json"
GATE_TOL = 0.02
BASE_SEED = 42
N_SEEDS = 5
MAGNITUDES_KM = (0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 270.0)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam._config import load_experiment_config
    from aquacontam.analysis.coordinate_sensitivity import (
        coordinate_sensitivity_summary,
        run_coordinate_sensitivity,
    )
    from aquacontam.models.xgboost import XGBoostClassifier

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    from aquacontam.pipeline.features import extract_features

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    exp_cfg = load_experiment_config()
    model_cfg = exp_cfg.get("models", {}).get("xgboost_classifier", {})

    results = run_coordinate_sensitivity(
        XGBoostClassifier,
        model_cfg,
        wq_df,
        feature_dfs=feature_dfs,
        magnitudes_km=MAGNITUDES_KM,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
    )
    summary = coordinate_sensitivity_summary(results)
    rows = summary.to_dict(orient="records")

    def _auroc_of(r: dict) -> float:
        for k in ("auroc_mean", "mean_auroc"):
            if k in r and r[k] is not None:
                return float(r[k])
        return float("nan")

    def _auroc_at(mag: float) -> float | None:
        for r in rows:
            if abs(float(r.get("magnitude_km", -1)) - mag) < 1e-9:
                return _auroc_of(r)
        return None

    base = _auroc_at(0.0)
    frozen_headline = None
    fr = json.loads((FROZEN / "coordinate_sensitivity.json").read_text(encoding="utf-8"))
    for r in fr.get("summary", []):
        if abs(float(r.get("magnitude_km", -1)) - 0.0) < 1e-9:
            frozen_headline = _auroc_of(r)
            break
    if base is None or frozen_headline is None:
        logger.error("Could not locate 0 km AUROC (this run=%s, frozen=%s)", base, frozen_headline)
        return 1
    if abs(base - frozen_headline) > GATE_TOL:
        logger.error("Gate FAILED: 0km AUROC %.4f vs frozen %.4f", base, frozen_headline)
        return 1
    logger.info("Gate OK: 0km AUROC %.4f vs frozen %.4f", base, frozen_headline)

    tail = _auroc_at(270.0)
    payload = {
        "_meta": {
            "source": (
                "C3 (M6b) coordinate-perturbation robustness extended to the empirical "
                "geocoding-error tail (50/100/270 km added to the published <=20 km). Same "
                "run_coordinate_sensitivity harness, XGBoost config, seeds. Reproduction gate "
                f"ties 0 km AUROC to coordinate_sensitivity.json within {GATE_TOL}."
            ),
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "magnitudes_km": list(MAGNITUDES_KM),
        },
        "summary": rows,
        "auroc_0km": base,
        "auroc_270km": tail,
        "auroc_drop_0_to_270km": (base - tail) if (base is not None and tail is not None) else None,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("0km %.4f -> 270km %s (drop %s) -> wrote %s", base, tail, payload["auroc_drop_0_to_270km"], OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
