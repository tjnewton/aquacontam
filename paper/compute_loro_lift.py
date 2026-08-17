#!/usr/bin/env python
"""C7 (de novo m11): out-of-region top-decile lift for the policy claim.

The paper's 3.0x top-decile lift is computed in-region while the deployment scenario is
out-of-region. This pre-specified COMPUTE re-runs the provenance-free T1 LORO folds
(mirroring ``_run_loro_cv`` exactly: same assembly, splitter, model config, seed),
collects the out-of-fold predictions every fold, and derives the out-of-region lift two
ways:

- pooled OOF: every system scored by the model that held out its region, one national
  top decile;
- per-region: the top decile WITHIN each held-out region (the deployment-realistic
  prioritization), system-count-weighted mean.

Reproduction gate: every fold's AUROC must match the frozen
``loro_cv_provenance_free.json`` fold within GATE_TOL, proving the mirrored folds are
the published ones, before any lift is written. Output goes directly into the frozen
archive (then ``freeze_results.py --rehash-only`` and ``stamp_derivations.py``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "loro_lift.json"
GATE_TOL = 2e-3
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _lift(y_true, y_prob) -> float:
    import numpy as np

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if len(y) < 10 or y.sum() == 0:
        return float("nan")
    k = max(1, int(round(len(y) * 0.10)))
    top = np.argsort(-p)[:k]
    return float(y[top].mean() / y.mean())


def main() -> int:
    import numpy as np
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.features.assembly import (
        PROVENANCE_FREE_EXCLUDE,
        aggregate_to_system_level,
        drop_leakage_columns,
        resolve_excluded_columns,
    )
    from aquacontam.models.xgboost import XGBoostClassifier
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.models import load_model_configs
    from aquacontam.preprocessing.splits import leave_one_region_out

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    drop_cols = resolve_excluded_columns(X.columns, list(PROVENANCE_FREE_EXCLUDE))
    X = X.drop(columns=drop_cols)
    logger.info("PF assembly: %d systems, %d features (dropped %d)", *X.shape, len(drop_cols))

    frozen_folds = {
        int(f["test_region"]): f
        for f in json.loads((FROZEN / "loro_cv_provenance_free.json").read_text(encoding="utf-8"))[
            "T1"
        ]["xgboost"]["folds"]
    }

    cfg = dict(load_model_configs().get("xgboost_classifier", {}))
    cfg["random_state"] = SEED

    loro_df = pd.DataFrame({"epa_region": regions}, index=X.index)
    oof_y: list[np.ndarray] = []
    oof_p: list[np.ndarray] = []
    per_region: list[dict[str, float | int]] = []
    gate_fails: list[str] = []

    for train_df, val_df, test_df, test_region in leave_one_region_out(loro_df, seed=SEED):
        train_idx = train_df.index.intersection(X.index)
        val_idx = val_df.index.intersection(X.index)
        test_idx = test_df.index.intersection(X.index)
        if len(train_idx) < 10 or len(test_idx) < 10:
            continue
        model = XGBoostClassifier(config=cfg.copy())
        model.fit(X.loc[train_idx], y.loc[train_idx], X_val=X.loc[val_idx], y_val=y.loc[val_idx])
        probs = model.predict_proba(X.loc[test_idx])
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y.loc[test_idx], probs))
        frozen = frozen_folds.get(int(test_region))
        if frozen is None:
            gate_fails.append(f"region {test_region}: no frozen fold")
        elif abs(auroc - float(frozen["auroc"])) > GATE_TOL:
            gate_fails.append(
                f"region {test_region}: AUROC {auroc:.4f} vs frozen {frozen['auroc']:.4f}"
            )
        oof_y.append(np.asarray(y.loc[test_idx], dtype=float))
        oof_p.append(np.asarray(probs, dtype=float))
        per_region.append(
            {
                "test_region": int(test_region),
                "n_test": len(test_idx),
                "auroc": auroc,
                "lift_top_decile": _lift(y.loc[test_idx], probs),
            }
        )
        logger.info(
            "fold %d: AUROC %.4f (frozen %.4f), lift %.2f",
            test_region,
            auroc,
            float(frozen["auroc"]) if frozen else float("nan"),
            per_region[-1]["lift_top_decile"],
        )

    if gate_fails:
        logger.error(
            "Reproduction gate FAILED (%d): %s — nothing written", len(gate_fails), gate_fails[:3]
        )
        return 1
    logger.info(
        "Reproduction gate OK: all %d folds match frozen within %.0e", len(per_region), GATE_TOL
    )

    y_all = np.concatenate(oof_y)
    p_all = np.concatenate(oof_p)
    pooled = _lift(y_all, p_all)
    finite = [r for r in per_region if np.isfinite(r["lift_top_decile"])]
    weights = np.array([r["n_test"] for r in finite], dtype=float)
    weighted = float(np.average([r["lift_top_decile"] for r in finite], weights=weights))
    payload = {
        "_meta": {
            "source": (
                "C7 (m11) out-of-region top-decile lift from provenance-free T1 LORO OOF "
                "predictions (folds mirrored from _run_loro_cv; per-fold AUROC reproduction "
                f"gate vs loro_cv_provenance_free.json within {GATE_TOL})."
            ),
            "n_systems_oof": len(y_all),
            "seed": SEED,
        },
        "pooled_oof_lift_top_decile": pooled,
        "weighted_mean_per_region_lift": weighted,
        "per_region": per_region,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "pooled OOF lift %.2f | weighted per-region lift %.2f -> wrote %s",
        pooled,
        weighted,
        OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
