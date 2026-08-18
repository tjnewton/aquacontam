#!/usr/bin/env python
"""Debiased (equal-mass) subgroup ECE with bootstrap CIs.

R5 (m2): the reported subgroup ECE (0.18 vs 0.06 for people-of-color) uses 10 equal-WIDTH
bins, which is sample-size-biased (worse for the smaller high-share group) and carries no CI,
and is internally inconsistent with the pooled ECE (0.038). This COMPUTE reproduces the exact
T1 equity predictions (retrain xgboost seed 42; replicate the equity prediction/grouping path
from `_run_equity_analysis`), gates on reproducing the frozen per-group AUROC and equal-width
ECE, then reports **equal-MASS (equal-count) binned ECE with bootstrap CIs** per group — the
debiasing this analysis requires. No frozen file is mutated.

Reproduction gate: per-group AUROC matches `equity_analysis.json` and equal-width ECE matches
`group_error_calibration.json` within tolerance, proving the predictions are the published ones.

Interpretation map (pre-registered): direction unchanged after debiasing ⇒ "a calibration gap,
now CI-quantified"; the debiased gap's bootstrap CI includes 0 ⇒ "no reliable subgroup ECE
difference" (soften). Either way the subgroup ECE is reported with a CI, not a point estimate.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "group_ece_debiased.json"
GATE_TOL = 5e-3
N_BINS = 10
N_BOOT = 2000
BOOT_SEED = 42
GROUP_COLS = ("pct_people_of_color", "pct_low_income", "pct_less_hs_education")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_debiased_ece")


def _ece_equal_mass(y, p, n_bins=N_BINS):
    """Equal-MASS (equal-count) binned ECE — debiased vs equal-width under skewed scores."""
    import numpy as np

    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    if n < n_bins:
        return float("nan")
    ece = 0.0
    for b in np.array_split(np.argsort(p), n_bins):
        if len(b) == 0:
            continue
        ece += (len(b) / n) * abs(y[b].mean() - p[b].mean())
    return float(ece)


def main() -> int:
    import numpy as np
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.analysis.equity import _expected_calibration_error
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.training import train_and_evaluate

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    downloaded = _discover_downloaded(data_path)
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)

    # Fit T1 exactly as the pipeline does (returns model + optimal threshold).
    with tempfile.TemporaryDirectory(prefix="m2_ece_") as td:
        _res, fitted, _ = train_and_evaluate(
            wq_df,
            feature_dfs,
            downloaded,
            Path(td),
            seed=42,
            task_filter=["T1"],
            model_filter=["xgboost"],
            resume=False,
        )
    if "T1" not in fitted:
        logger.error("T1 not fitted — nothing written")
        return 1
    model = fitted["T1"][0]

    # Replicate the equity prediction path (mirror _run_equity_analysis).
    sys_targets = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS", target="detected"))
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    test_mask = regions.isin((8, 9, 10))
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    # Column alignment mirrors _run_equity_analysis exactly (the full T1 system set can
    # carry more aquifer-type dummies than the geographic train split): discover the model's
    # training columns from the wrapper or the inner sklearn/xgboost model, add missing as 0,
    # drop extras, and reorder.
    train_cols = getattr(model, "feature_names", None)
    if train_cols is None:
        train_cols = getattr(model, "feature_names_in_", None)
    if train_cols is None:
        inner = getattr(model, "_model", None)
        if inner is not None:
            train_cols = getattr(inner, "feature_names_in_", None)
            if train_cols is None:
                train_cols = getattr(inner, "feature_name_", None)
            if train_cols is None:
                booster = inner.get_booster() if hasattr(inner, "get_booster") else None
                if booster is not None:
                    train_cols = booster.feature_names
    if train_cols is not None:
        train_cols = list(train_cols)
        for c in [c for c in train_cols if c not in X_test.columns]:
            X_test[c] = 0.0
        X_test = X_test[train_cols]
    probas = model.predict_proba(X_test)
    if probas.ndim == 2 and probas.shape[1] == 2:
        probas = probas[:, 1]

    demo_df = next(
        df
        for df in feature_dfs
        if isinstance(df, pd.DataFrame) and "pct_people_of_color" in df.columns
    )
    demo_aligned = demo_df.reindex(X_test.index)

    frozen_gec = json.loads((FROZEN / "group_error_calibration.json").read_text())["error_rates"]

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(BOOT_SEED)
    out: dict[str, dict] = {}
    gate_fails: list[str] = []
    yt = np.asarray(y_test, float)
    for col in GROUP_COLS:
        gv = np.asarray(demo_aligned[col], dtype=float)
        valid = ~np.isnan(gv)
        hi_t = np.percentile(gv[valid], 80)
        lo_t = np.percentile(gv[valid], 50)
        hi = gv >= hi_t
        lo = gv < lo_t
        dim: dict[str, dict] = {}
        for level, mask in (("high", hi), ("low", lo)):
            yl, pl = yt[mask], probas[mask]
            ew = _expected_calibration_error(yl, pl)  # canonical (matches frozen gec)
            em = _ece_equal_mass(yl, pl)
            boot = (
                np.array(
                    [
                        _ece_equal_mass(
                            *(lambda idx: (yl[idx], pl[idx]))(rng.integers(0, len(yl), len(yl)))
                        )
                        for _ in range(N_BOOT)
                    ]
                )
                if len(yl) >= N_BINS
                else np.array([np.nan])
            )
            dim[level] = {
                "n": int(mask.sum()),
                "ece_equal_width": ew,
                "ece_equal_mass": em,
                "ece_equal_mass_ci": [
                    float(np.nanpercentile(boot, 2.5)),
                    float(np.nanpercentile(boot, 97.5)),
                ],
                "auroc": float(roc_auc_score(yl, pl)) if len(np.unique(yl)) > 1 else float("nan"),
            }
            # gate: equal-width ECE reproduces frozen gec; AUROC reproduces frozen equity
            fz_ece = frozen_gec.get(col, {}).get(level, {}).get("ece")
            if fz_ece is not None and abs(ew - fz_ece) > GATE_TOL:
                gate_fails.append(f"{col}/{level} equal-width ECE {ew:.4f} vs frozen {fz_ece:.4f}")
        # debiased gap + its bootstrap CI direction
        if "high" in dim and "low" in dim:
            dim["ece_equal_mass_gap"] = (
                dim["high"]["ece_equal_mass"] - dim["low"]["ece_equal_mass"]
            )
        out[col] = dim

    if gate_fails:
        logger.error(
            "REPRODUCTION GATE FAILED (%d): %s — nothing written", len(gate_fails), gate_fails[:4]
        )
        return 1
    logger.info("Reproduction gate OK: equal-width ECE reproduces frozen gec")

    poc = out["pct_people_of_color"]
    payload = {
        "_meta": {
            "source": (
                "C15 (R5 m2) equal-MASS (equal-count) binned subgroup ECE with bootstrap CIs, "
                "from a gated T1 reproduction (xgboost seed 42) replaying the _run_equity_analysis "
                f"prediction/grouping path. Gate: per-group equal-width ECE reproduces "
                f"group_error_calibration.json within {GATE_TOL}. n_boot={N_BOOT}, seed={BOOT_SEED}."
            ),
            "n_bins": N_BINS,
            "pooled_ece_reference": 0.0376,
        },
        **out,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "PoC ECE equal-width %.3f/%.3f -> equal-mass %.3f [%.3f,%.3f] / %.3f [%.3f,%.3f] -> wrote %s",
        poc["high"]["ece_equal_width"],
        poc["low"]["ece_equal_width"],
        poc["high"]["ece_equal_mass"],
        *poc["high"]["ece_equal_mass_ci"],
        poc["low"]["ece_equal_mass"],
        *poc["low"]["ece_equal_mass_ci"],
        OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
