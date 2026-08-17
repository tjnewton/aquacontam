#!/usr/bin/env python
"""C6 (de novo m1): data-source recoverability from the PROVENANCE-FREE feature set.

The frozen recoverability probe (`icp_recoverability.json`) measures how well a probe
recovers data-source identity from RAW inputs (~0.79) versus the ICP representation
(~0.64). But the deployment claim concerns the provenance-free FEATURE SET, and one-hot
`dummy_na=True` encoding can leave missingness (hence source) losslessly recoverable from
the retained value dummies (m1). This pre-specified COMPUTE trains the SAME held-out probe
on the provenance-free feature matrix and reports the source-recoverability there.

Reproduction gate: the probe re-trained on the RAW feature set must match the frozen
`icp_recoverability.json` raw-feature AUROC (logistic) within GATE_TOL, proving the probe
and population match the published ones, before the PF-set number is written. Output goes
directly into the frozen archive (then `freeze_results.py --rehash-only` and
`stamp_derivations.py`).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "pf_recoverability.json"
GATE_TOL = 0.02
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.features.assembly import (
        PROVENANCE_FREE_EXCLUDE,
        aggregate_to_system_level,
        assemble_feature_matrix,
        derive_system_epa_regions,
        drop_leakage_columns,
        resolve_excluded_columns,
    )
    from aquacontam.pipeline.features import extract_features

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    # Mirror the frozen probe's assembly exactly (assemble_feature_matrix + region derive).
    sys_targets = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS", target="detected"))
    X, y, _ = assemble_feature_matrix(sys_targets, *feature_dfs)
    regions = derive_system_epa_regions(wq_df, X.index)
    if "source" not in wq_df.columns:
        logger.error("no source column")
        return 1
    source = wq_df.groupby("pwsid")["source"].first().reindex(X.index).fillna("unknown")

    train_mask = regions.isin((1, 3, 4, 5, 6)).to_numpy()
    test_mask = regions.isin((8, 9, 10)).to_numpy()
    src_codes, src_uniques = pd.factorize(source)
    s_train, s_test = src_codes[train_mask], src_codes[test_mask]
    common = [c for c in np.unique(s_train) if (s_train == c).sum() >= 2 and (s_test == c).sum() >= 1]
    if len(common) < 2:
        logger.error("insufficient source classes")
        return 1

    def _probe_auroc(rep_train, rep_test, clf) -> float:
        sc = StandardScaler().fit(rep_train)
        clf.fit(sc.transform(rep_train), s_train)
        proba = clf.predict_proba(sc.transform(rep_test))
        trained = list(clf.classes_)
        keep = np.array([c in trained for c in s_test])
        y_keep, p_keep = s_test[keep], proba[keep]
        per_class = []
        for j, cls in enumerate(trained):
            pos = (y_keep == cls).astype(int)
            if 0 < pos.sum() < len(pos):
                try:
                    per_class.append(float(roc_auc_score(pos, p_keep[:, j])))
                except ValueError:
                    pass
        return float(np.mean(per_class)) if per_class else float("nan")

    drop_cols = resolve_excluded_columns(X.columns, list(PROVENANCE_FREE_EXCLUDE))
    X_pf = X.drop(columns=drop_cols)
    logger.info("raw features %d; PF features %d (dropped %d)", X.shape[1], X_pf.shape[1], len(drop_cols))

    raw_tr, raw_te = X.loc[train_mask].to_numpy(), X.loc[test_mask].to_numpy()
    pf_tr, pf_te = X_pf.loc[train_mask].to_numpy(), X_pf.loc[test_mask].to_numpy()

    probes: dict[str, dict[str, float]] = {}
    for name, factory in (
        ("logistic", lambda: LogisticRegression(max_iter=1000)),
        ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=SEED)),
    ):
        probes[name] = {
            "auroc_from_raw_features": _probe_auroc(raw_tr, raw_te, factory()),
            "auroc_from_pf_features": _probe_auroc(pf_tr, pf_te, factory()),
        }
        logger.info(
            "%s: raw %.4f, PF %.4f",
            name,
            probes[name]["auroc_from_raw_features"],
            probes[name]["auroc_from_pf_features"],
        )

    # Reproduction gate against the frozen raw-feature logistic AUROC.
    frozen = json.loads((FROZEN / "icp_recoverability.json").read_text(encoding="utf-8"))
    frozen_raw = float(frozen["probes"]["logistic"]["auroc_from_raw_features"])
    got_raw = probes["logistic"]["auroc_from_raw_features"]
    if abs(got_raw - frozen_raw) > GATE_TOL:
        logger.error("Gate FAILED: raw logistic AUROC %.4f vs frozen %.4f", got_raw, frozen_raw)
        return 1
    logger.info("Gate OK: raw logistic AUROC %.4f vs frozen %.4f", got_raw, frozen_raw)

    payload = {
        "_meta": {
            "source": (
                "C6 (m1) data-source recoverability from the provenance-free FEATURE SET "
                "(the deployment set), vs raw inputs. Same held-out probe and assembly as "
                "icp_recoverability.json; reproduction gate ties the raw-feature logistic "
                f"AUROC to the frozen value within {GATE_TOL}."
            ),
            "n_source_classes": len(common),
            "source_classes": [str(src_uniques[c]) for c in common],
            "n_pf_features": int(X_pf.shape[1]),
            "n_dropped": len(drop_cols),
            "seed": SEED,
        },
        "chance_auroc": 0.5,
        "probes": probes,
        "interpretation": (
            "PF-set AUROC well above 0.5 but below the raw-feature AUROC means dropping "
            "the explicit provenance features reduces but does not eliminate "
            "source-recoverability (residual dummy_na re-encoding), so the provenance-free "
            "signal is an upper bound on the environmental signal — consistent with m1."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
