#!/usr/bin/env python
"""C9 (de novo m10): per-DEMOGRAPHIC-group conformal coverage.

The frozen group conformal analysis stratifies coverage by EPA region, but the equity
analysis flags PROTECTED DEMOGRAPHIC groups; the reviewer asks for per-group coverage on
those groups (or an explicit statement that no per-group guarantee is provided). This
pre-specified COMPUTE mirrors ``_run_group_conformal_analysis`` exactly (same assembly,
same XGBoost config+seed, same train/val/test, same GroupConformalClassifier), but
calibrates and evaluates conformal coverage stratified by a median split of each flagged
demographic (people-of-color, low-income) instead of by region. The median threshold is
computed on the calibration (val) set and applied to test — no test leakage.

Reproduction gate: the overall (marginal, alpha=0.05) coverage must match the frozen
``group_conformal_results.json`` overall coverage within GATE_TOL, proving the retrained
model and calibration reproduce the published pipeline, before per-demographic coverage is
written. Output goes directly into the frozen archive (then ``freeze_results.py
--rehash-only`` and ``stamp_derivations.py``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "group_conformal_demographic.json"
GATE_TOL = 0.01
SEED = 42
DEMOGRAPHICS = ("pct_people_of_color", "pct_low_income")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    import numpy as np

    import aquacontam.benchmark  # noqa: F401
    from aquacontam._config import load_experiment_config
    from aquacontam.calibration.conformal import GroupConformalClassifier
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.models.xgboost import XGBoostClassifier
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    wq_df = __import__("pandas").read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    sys_targets = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS", target="detected"))
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    regions = regions.fillna(0).astype(int)

    exp_cfg = load_experiment_config()
    split_cfg = exp_cfg.get("split", {})
    train_regs = set(split_cfg.get("train_regions", [1, 3, 4, 5, 6]))
    val_regs = set(split_cfg.get("val_regions", [2, 7]))
    test_regs = set(split_cfg.get("test_regions", [8, 9, 10]))
    train_mask = regions.isin(train_regs)
    val_mask = regions.isin(val_regs)
    test_mask = regions.isin(test_regs)

    model = XGBoostClassifier(config=exp_cfg.get("models", {}).get("xgboost_classifier", {}))
    model.fit(
        X.loc[train_mask],
        y.loc[train_mask],
        eval_set=[(X.loc[val_mask].to_numpy(), y.loc[val_mask].to_numpy())],
        verbose=False,
    )

    # Reproduction gate: reproduce the frozen overall (alpha=0.05) coverage using the
    # region-stratified calibration the frozen file used.
    frozen = json.loads((FROZEN / "group_conformal_results.json").read_text(encoding="utf-8"))
    frozen_overall = float(next(e for e in frozen if abs(e["alpha"] - 0.05) < 1e-9)["coverage"])
    gcc_region = GroupConformalClassifier(model, alpha=0.05)
    gcc_region.calibrate(X.loc[val_mask], y.loc[val_mask], regions.loc[val_mask].to_numpy())
    got_overall = float(
        gcc_region.coverage_and_set_size(
            X.loc[test_mask], y.loc[test_mask], regions.loc[test_mask].to_numpy()
        )["coverage"]
    )
    if abs(got_overall - frozen_overall) > GATE_TOL:
        logger.error("Gate FAILED: overall coverage %.4f vs frozen %.4f", got_overall, frozen_overall)
        return 1
    logger.info("Gate OK: overall coverage %.4f vs frozen %.4f", got_overall, frozen_overall)

    results: dict[str, dict] = {}
    for demo in DEMOGRAPHICS:
        if demo not in X.columns:
            logger.warning("%s not in features; skipping", demo)
            continue
        # Median split defined on the CALIBRATION (val) set, applied to test (no leakage).
        thr = float(np.nanmedian(X.loc[val_mask, demo]))

        def _grp(mask, _demo=demo, _thr=thr):
            return (X.loc[mask, _demo].to_numpy() >= _thr).astype(int)  # 1=high, 0=low

        per_alpha: dict[str, dict] = {}
        for alpha in (0.05, 0.10, 0.20):
            gcc = GroupConformalClassifier(model, alpha=alpha)
            gcc.calibrate(X.loc[val_mask], y.loc[val_mask], _grp(val_mask))
            stats = gcc.coverage_and_set_size(X.loc[test_mask], y.loc[test_mask], _grp(test_mask))
            pg = {str(k): v for k, v in stats.get("per_group", {}).items()}
            per_alpha[f"{alpha:.2f}"] = {
                "target_coverage": 1.0 - alpha,
                "overall_coverage": stats["coverage"],
                "high_group_coverage": pg.get("1", {}).get("coverage"),
                "low_group_coverage": pg.get("0", {}).get("coverage"),
                "high_n": pg.get("1", {}).get("n"),
                "low_n": pg.get("0", {}).get("n"),
            }
        results[demo] = {"median_threshold_val": thr, "by_alpha": per_alpha}
        a05 = per_alpha["0.05"]
        logger.info(
            "%s @a=0.05: high %.3f (n=%s), low %.3f (n=%s)",
            demo,
            a05["high_group_coverage"] or float("nan"),
            a05["high_n"],
            a05["low_group_coverage"] or float("nan"),
            a05["low_n"],
        )

    payload = {
        "_meta": {
            "source": (
                "C9 (m10) per-demographic-group conformal coverage. Mirrors "
                "_run_group_conformal_analysis (same assembly, XGBoost config+seed, splits, "
                "GroupConformalClassifier) but stratifies by a val-set median split of each "
                "flagged demographic instead of by EPA region. Reproduction gate ties the "
                f"overall alpha=0.05 coverage to group_conformal_results.json within {GATE_TOL}."
            ),
            "seed": SEED,
            "note": (
                "Per-group conformal guarantees are approximate: calibration is on the val "
                "regions (2,7), so per-demographic coverage on the test regions is a "
                "transfer estimate, not a within-group exchangeable guarantee."
            ),
        },
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
