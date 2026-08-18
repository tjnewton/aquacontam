#!/usr/bin/env python
"""Seed stability of the T1 headline AUROC on its OWN harness.

R5 (M4) shows the robustness analyses (multi_seed_stability, val_reuse_bias, tuning) run on
a streamlined harness at ~0.79-0.83, whereas the 0.864 headline comes from the benchmark
harness — so the headline's own seed stability was never established, and at least three T1
values circulate (0.864 / 0.809 / 0.790) without reconciliation. This COMPUTE runs the
**actual headline harness** (``train_and_evaluate``'s benchmark T1 task, ambient sources
excluded, geographic holdout, YAML xgboost_classifier config) across five seeds and builds a
reconciliation table mapping every circulating value to its (harness, feature set, split).

Reproduction gate: seed 42 on this harness reproduces the frozen headline 0.8636 before the
other seeds run — proving this is the harness that produced results.json, not a proxy.

Interpretation map (pre-registered, signed off): 5-seed mean in [0.85, 0.87] ⇒ the headline
stays 0.864 (seed 42) with "5-seed mean X +/- SD" appended (the pre-agreed fallback);
mean < 0.85 ⇒ STOP and escalate with the numbers (do not silently reword the flagship).
"""

from __future__ import annotations

import os as _os
import sys as _sys


def _early_gpu_env_setup() -> None:
    gpu_id: str | None = None
    for i, tok in enumerate(_sys.argv):
        if tok == "--gpu-id" and i + 1 < len(_sys.argv):
            gpu_id = _sys.argv[i + 1]
            break
        if tok.startswith("--gpu-id="):
            gpu_id = tok.split("=", 1)[1]
            break
    if gpu_id is not None:
        _os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        _os.environ["AQUACONTAM_GPU_ID"] = "0"


_early_gpu_env_setup()

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "headline_harness_stability.json"
GATE_TOL = 2e-3
SEEDS = [42, 123, 456, 789, 2024]
BAND = (0.85, 0.87)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_headline_stability")


def main() -> int:
    import numpy as np
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.benchmark.registry import get_task
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.models import get_model_instances
    from aquacontam.pipeline.training import _exclude_ambient_sources, build_task_kwargs

    _sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    downloaded = _discover_downloaded(data_path)
    wq_df = _exclude_ambient_sources(pd.read_parquet(data_path / "interim" / "merged_wq.parquet"))
    feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)
    _, task_fn = get_task("T1")

    def headline_auroc(seed: int) -> float:
        model = get_model_instances(
            ["xgboost"], "classification", seed, task_name="T1", use_tuned_params=False
        )[0]
        kwargs = build_task_kwargs("T1", model, wq_df, feature_dfs, downloaded)
        return float(task_fn(**kwargs).metrics["auroc"])

    # --- reproduction gate on seed 42 ---
    frozen_headline = next(
        float(e["metrics"]["auroc"])
        for e in json.loads((FROZEN / "results.json").read_text(encoding="utf-8"))
        if e.get("task") == "T1" and e.get("model") == "xgboost_classifier"
    )
    a42 = headline_auroc(42)
    logger.info("seed 42 AUROC=%.4f (frozen headline %.4f)", a42, frozen_headline)
    if abs(a42 - frozen_headline) > GATE_TOL:
        logger.error(
            "REPRODUCTION GATE FAILED: seed-42 %.4f vs frozen headline %.4f — nothing written",
            a42, frozen_headline,
        )
        return 1
    logger.info("Reproduction gate OK: this IS the headline harness")

    per_seed = {42: a42}
    for s in SEEDS:
        if s == 42:
            continue
        per_seed[s] = headline_auroc(s)
        logger.info("seed %d AUROC=%.4f", s, per_seed[s])

    vals = np.array(list(per_seed.values()))
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    in_band = BAND[0] <= mean <= BAND[1]
    if not in_band:
        logger.error(
            "ESCALATE: 5-seed mean %.4f outside band %s — do not reword the flagship", mean, BAND
        )

    # reconciliation of the circulating T1 values
    infs_seed = json.loads((FROZEN / "seed_inflation_check.json").read_text(encoding="utf-8"))
    split = json.loads((FROZEN / "split_comparison.json").read_text(encoding="utf-8"))
    loro = json.loads((FROZEN / "loro_cv.json").read_text(encoding="utf-8"))
    reconciliation = [
        {"value": round(frozen_headline, 4), "source": "results.json T1 xgboost_classifier",
         "harness": "benchmark task_fn", "features": "full", "split": "geographic holdout R8/9/10",
         "note": "the headline; this compute's seed-42 reproduces it"},
        {"value": round(mean, 4), "source": "this compute (headline harness)",
         "harness": "benchmark task_fn", "features": "full", "split": "geographic holdout R8/9/10",
         "note": f"5-seed mean +/- {sd:.4f} on the headline harness"},
        {"value": round(infs_seed["T1"]["xgboost"]["mean_auroc_multi_seed"], 4),
         "source": "multi_seed_stability.json T1 xgboost",
         "harness": "streamlined _robustness", "features": "full", "split": "fixed R8/9/10",
         "note": "the ~0.81 value — a different (streamlined) harness"},
        {"value": round(
            next(
                r["metrics"]["auroc"]
                for r in split["matrix"]["results"]
                if "xgboost" in r.get("model_name", "").lower()
                and r.get("feature_set") == "full"
                and r.get("split_strategy") == "geographic"
            ), 4),
         "source": "split_comparison.json xgboost full-feature geographic",
         "harness": "split-comparison", "features": "full", "split": "geographic",
         "note": "the ~0.790 value"},
        {"value": round(loro["T1"]["xgboost"]["mean_auroc"], 4),
         "source": "loro_cv.json T1 xgboost", "harness": "LORO", "features": "full",
         "split": "leave-one-region-out (10 folds)", "note": "out-of-region mean 0.7825"},
    ]

    payload = {
        "_meta": {
            "source": (
                "C11 (R5 M4) seed stability of the T1 headline on its own benchmark harness "
                "(train_and_evaluate T1 task; ambient sources excluded; geographic holdout; YAML "
                f"xgboost_classifier config). Reproduction gate: seed-42 reproduces results.json "
                f"headline {frozen_headline:.4f} within {GATE_TOL}. Interpretation band {BAND}."
            ),
            "seeds": SEEDS,
            "reproduction_gate": "PASSED",
            "in_band": bool(in_band),
        },
        "per_seed_auroc": per_seed,
        "seed_42_auroc": a42,
        "mean_auroc": mean,
        "std_auroc": sd,
        "reconciliation": reconciliation,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "headline 0.864 (seed 42) | 5-seed mean %.4f +/- %.4f | in_band=%s -> wrote %s",
        mean, sd, in_band, OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    _sys.exit(main())
