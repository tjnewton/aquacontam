#!/usr/bin/env python
"""C14 (R5 M7): top-decile lift on the regulatorily-actionable MCL-exceedance target.

The manuscript's "3.0x top-decile lift" prioritization claim is computed on the DETECTION
target, but the regulatorily-actionable target is MCL exceedance, which is materially harder
(frozen AUROC 0.719 vs 0.807). R5 (M7) asks for the exceedance top-decile lift in the main
text. This COMPUTE mirrors ``_run_mcl_exceedance_analysis`` exactly (same features, same
geographic split, XGBoost default config, seed 42), reproduces the frozen exceedance AUROC as
its gate, then reports the test-set top-decile lift for BOTH targets.

Reproduction gate: the MCL-exceedance test AUROC reproduces ``mcl_exceedance_analysis.json``
(0.7186) before any lift is written.

Output: ``results/paper_frozen/mcl_lift.json``. Interpretation map (pre-registered): report the
exceedance lift in the main text; if it is < 1.0 (no prioritization signal) ⇒ escalate, since
that would contradict the deployment framing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "mcl_lift.json"
GATE_TOL = 2e-3
SEED = 42
_TRAIN, _VAL, _TEST = {1, 3, 4, 5, 6}, {2, 7}, {8, 9, 10}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_mcl_lift")


def _lift(y_true, y_prob) -> float:
    import numpy as np

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if len(y) < 10 or y.sum() == 0:
        return float("nan")
    k = max(1, round(len(y) * 0.10))
    return float(y[np.argsort(-p)[:k]].mean() / y.mean())


def main() -> int:
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.analysis.mcl_threshold import compute_mcl_exceedance
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.models.xgboost import XGBoostClassifier
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.models import load_model_configs

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    sys_det = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS", target="detected"))
    X_det, y_det, _, regions_det = assemble_with_split_imputation(
        sys_det, feature_dfs, wq_df=wq_df
    )

    mcl_indexed = compute_mcl_exceedance(wq_df).set_index("pwsid")["mcl_exceedance"]
    common = X_det.index.intersection(mcl_indexed.index)

    cfg = dict(load_model_configs().get("xgboost_classifier", {}))
    cfg["random_state"] = SEED
    frozen = json.loads((FROZEN / "mcl_exceedance_analysis.json").read_text(encoding="utf-8"))

    out: dict[str, dict] = {}
    for label, X, y, regions in [
        ("detection", X_det, y_det, regions_det),
        ("mcl_exceedance", X_det.loc[common], mcl_indexed.loc[common], regions_det.loc[common]),
    ]:
        tr, va, te = regions.isin(_TRAIN), regions.isin(_VAL), regions.isin(_TEST)
        model = XGBoostClassifier(config=cfg.copy())
        model.fit(X.loc[tr], y.loc[tr], X_val=X.loc[va], y_val=y.loc[va])
        probs = model.predict_proba(X.loc[te])
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y.loc[te], probs))
        out[label] = {
            "auroc": auroc,
            "top_decile_lift": _lift(y.loc[te], probs),
            "n_test": int(te.sum()),
            "positive_rate_test": float(y.loc[te].mean()),
        }
        logger.info("%s: AUROC=%.4f lift=%.2f", label, auroc, out[label]["top_decile_lift"])

    # --- reproduction gate ---
    frozen_auroc = float(frozen["mcl_exceedance"]["metrics"]["auroc"])
    got = out["mcl_exceedance"]["auroc"]
    if abs(got - frozen_auroc) > GATE_TOL:
        logger.error("REPRODUCTION GATE FAILED: MCL AUROC %.4f vs frozen %.4f", got, frozen_auroc)
        return 1
    logger.info("Reproduction gate OK: MCL AUROC %.4f reproduces frozen %.4f", got, frozen_auroc)

    if out["mcl_exceedance"]["top_decile_lift"] < 1.0:
        logger.error("ESCALATE: MCL exceedance lift < 1.0 (no prioritization signal)")

    payload = {
        "_meta": {
            "source": (
                "C14 (R5 M7) top-decile lift for the MCL-exceedance vs detection target, folds "
                "mirrored from _run_mcl_exceedance_analysis (same features/split/XGBoost config, "
                f"seed {SEED}); reproduction gate ties MCL AUROC to mcl_exceedance_analysis.json "
                f"within {GATE_TOL}. No new pipeline run."
            ),
            "seed": SEED,
        },
        **out,
        "auroc_gap": out["detection"]["auroc"] - out["mcl_exceedance"]["auroc"],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "detection lift %.2f (AUROC %.3f) | MCL-exceedance lift %.2f (AUROC %.3f) -> wrote %s",
        out["detection"]["top_decile_lift"],
        out["detection"]["auroc"],
        out["mcl_exceedance"]["top_decile_lift"],
        out["mcl_exceedance"]["auroc"],
        OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
