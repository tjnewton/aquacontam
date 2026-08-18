#!/usr/bin/env python
"""Per-group burden-ratio bootstrap CIs + the excluded unknown group.

R5 (m10): the equity figure shows significance stars but no CIs/error bars, and the
"unknown demographics" group (~18% of test systems, AUROC 0.984 — a provenance artifact
consistent with the thesis) is silently dropped from the high/low comparison. This COMPUTE
provides the missing CIs — a system-level bootstrap of each demographic dimension's burden
ratio (labels only; no model fit) — and surfaces the excluded unknown group's size and AUROC
(the latter already frozen in equity_analysis.json). The figure gains error bars in Phase 3
and a one-sentence note on the excluded group.

Reproduction gate: each group's point burden ratio reproduces `equity_analysis.json` before
its CI is written.

Output: `results/paper_frozen/group_burden_ci.json`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "group_burden_ci.json"
GATE_TOL = 5e-3
N_BOOT = 5000
BOOT_SEED = 42
GROUP_COLS = ("pct_people_of_color", "pct_low_income", "pct_less_hs_education")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_burden_ci")


def main() -> int:
    import numpy as np
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    downloaded = _discover_downloaded(data_path)
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)

    sys_targets = drop_leakage_columns(aggregate_to_system_level(wq_df, "PFOS", target="detected"))
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    test_mask = regions.isin((8, 9, 10))
    y_test = np.asarray(y.loc[test_mask], dtype=float)
    demo_df = next(
        df
        for df in feature_dfs
        if isinstance(df, pd.DataFrame) and "pct_people_of_color" in df.columns
    )
    demo_aligned = demo_df.reindex(X.loc[test_mask].index)

    frozen = {e["group"]: e for e in json.loads((FROZEN / "equity_analysis.json").read_text())}
    rng = np.random.default_rng(BOOT_SEED)

    def burden(yv: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> float:
        rl = yv[lo].mean() if lo.any() else 0.0
        rh = yv[hi].mean() if hi.any() else 0.0
        return float(rh / rl) if rl > 0 else float("nan")

    out: dict[str, dict] = {}
    gate_fails: list[str] = []
    for col in GROUP_COLS:
        gv = np.asarray(demo_aligned[col], dtype=float)
        valid = ~np.isnan(gv)
        hi_t = np.percentile(gv[valid], 80)
        lo_t = np.percentile(gv[valid], 50)
        hi, lo = gv >= hi_t, gv < lo_t
        point = burden(y_test, hi, lo)

        idx = np.arange(len(y_test))
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rng.choice(idx, size=len(idx), replace=True)
            boots[b] = burden(y_test[s], hi[s], lo[s])
        ci = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]

        fz = frozen.get(col, {}).get("burden_ratio")
        if fz is not None and abs(point - float(fz)) > GATE_TOL:
            gate_fails.append(f"{col} burden {point:.4f} vs frozen {float(fz):.4f}")
        out[col] = {
            "burden_ratio": point,
            "burden_ratio_ci": ci,
            "n_high": int(hi.sum()),
            "n_low": int(lo.sum()),
            "excludes_one": bool(ci[0] > 1.0),
        }

    if gate_fails:
        logger.error("REPRODUCTION GATE FAILED: %s — nothing written", gate_fails)
        return 1
    logger.info("Reproduction gate OK: burden ratios reproduce frozen equity_analysis")

    # The excluded unknown-demographics group (AUROC already frozen in equity_analysis.json).
    poc = frozen.get("pct_people_of_color", {})
    unknown = poc.get("group_metrics", {}).get("unknown", {})
    n_unknown = int(unknown.get("n_samples", 0))
    n_test = len(y_test)
    payload = {
        "_meta": {
            "source": (
                "C19 (R5 m10) system-level bootstrap CIs for each demographic dimension's burden "
                "ratio (labels only, no model fit), + the excluded unknown-demographics group. "
                f"Reproduction gate ties each point burden ratio to equity_analysis.json within "
                f"{GATE_TOL}. n_boot={N_BOOT}, seed={BOOT_SEED}."
            ),
        },
        "burden_cis": out,
        "unknown_group": {
            "auroc": unknown.get("auroc"),
            "n_samples": n_unknown,
            "n_test": n_test,
            "fraction_of_test": (n_unknown / n_test) if n_test else None,
            "note": (
                "high AUROC on unknown-demographics systems is a provenance artifact "
                "consistent with the thesis; excluded from the high/low burden comparison"
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    poc_o = out["pct_people_of_color"]
    logger.info(
        "PoC burden %.3f CI [%.3f, %.3f] | unknown group AUROC %.3f (n=%d, %.1f%% of test) -> %s",
        poc_o["burden_ratio"],
        *poc_o["burden_ratio_ci"],
        unknown.get("auroc") or float("nan"),
        n_unknown,
        100 * n_unknown / n_test if n_test else 0,
        OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
