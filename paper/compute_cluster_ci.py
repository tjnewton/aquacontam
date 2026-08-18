#!/usr/bin/env python
"""Cluster-calibrated CIs for the two most-cited out-of-region numbers.

The transportable-signal AUROC (0.691) and the national detection-burden ratio (1.48) carry
"region-block-bootstrap" CIs built from only G = 10 EPA-region blocks. R5 (M5) shows that
with G = 10 the block bootstrap adds essentially nothing (frozen width ratio block/naive =
1.002) and is known to be anti-conservative. This COMPUTE quantifies the under-coverage by
computing, from the same 10 per-region values, a t(G-1)-calibrated interval (fatter tails for
9 df) and a cluster bootstrap (resample the 10 regions), and reports both widths against the
frozen block-bootstrap width. No retrain.

Reproduction gate: the weighted per-region means must reproduce the frozen point estimates —
LORO AUROC 0.6920 (inference_strengthening m4a) and burden 1.4823 (m4c) — proving the
per-region inputs are the published ones before any calibrated interval is written.

Interpretation map (pre-registered): the t(9)/cluster interval is wider than the block
bootstrap ⇒ the manuscript reports 0.691 with a G=10 anti-conservatism caveat and reads the
interval as indicative; if the wider interval's AUROC lower bound still exceeds 0.5 the
"signal survives out of region" claim stands; if it drops to <= 0.5 ⇒ escalate.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "cluster_ci_calibration.json"
GATE_TOL = 2e-3
T_CRIT_9 = 2.262157162740992  # t_{0.975, 9}
N_BOOT = 10000
BOOT_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_cluster_ci")


def _calibrated_intervals(values: list[float], weights: list[float]) -> dict:
    """Weighted point estimate + t(G-1) and cluster-bootstrap 95% CIs (G = len(values))."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    g = len(v)
    wmean = float(np.average(v, weights=w))
    smean = float(v.mean())
    sd = float(v.std(ddof=1))
    se_cluster = sd / math.sqrt(g)
    t_lo, t_hi = smean - T_CRIT_9 * se_cluster, smean + T_CRIT_9 * se_cluster

    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, g, size=g)  # resample G region-clusters with replacement
        boot[b] = np.average(v[idx], weights=w[idx])
    b_lo, b_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "n_clusters": g,
        "weighted_mean": wmean,
        "simple_mean": smean,
        "cluster_sd": sd,
        "t_gminus1_ci": [t_lo, t_hi],
        "t_gminus1_width": t_hi - t_lo,
        "cluster_bootstrap_ci": [b_lo, b_hi],
        "cluster_bootstrap_width": b_hi - b_lo,
    }


def main() -> int:
    pf = json.loads((FROZEN / "loro_cv_provenance_free.json").read_text(encoding="utf-8"))["T1"][
        "xgboost"
    ]["folds"]
    auroc_vals = [f["auroc"] for f in pf]
    auroc_wts = [f["n_test"] for f in pf]

    le = json.loads((FROZEN / "loro_equity_analysis.json").read_text(encoding="utf-8"))[
        "per_region"
    ]
    poc = [r for r in le if r.get("group") == "pct_people_of_color" and r.get("burden_ratio")]
    burden_vals = [r["burden_ratio"] for r in poc]
    burden_wts = [r.get("n_systems", r.get("n_with_demographics", 1)) for r in poc]

    infs = json.loads((FROZEN / "inference_strengthening.json").read_text(encoding="utf-8"))
    sbb = json.loads((FROZEN / "spatial_block_bootstrap.json").read_text(encoding="utf-8"))

    auroc = _calibrated_intervals(auroc_vals, auroc_wts)
    burden = _calibrated_intervals(burden_vals, burden_wts)

    # frozen block-bootstrap references
    auroc["block_bootstrap_width"] = infs["m4a_provenance_free_loro_ci"]["block_bootstrap"][
        "width"
    ]
    auroc["naive_normal_width"] = infs["m4a_provenance_free_loro_ci"]["naive_fold_normal"]["width"]
    burden["block_bootstrap_width"] = infs["m4c_burden_ratio_ci"]["block_bootstrap"]["width"]

    # --- reproduction gate ---
    frozen_auroc = infs["m4a_provenance_free_loro_ci"]["weighted_mean_auroc"]
    frozen_burden = infs["m4c_burden_ratio_ci"]["weighted_mean_burden_ratio"]
    fails = []
    if abs(auroc["weighted_mean"] - frozen_auroc) > GATE_TOL:
        fails.append(f"AUROC wmean {auroc['weighted_mean']:.4f} vs frozen {frozen_auroc:.4f}")
    if abs(burden["weighted_mean"] - frozen_burden) > GATE_TOL:
        fails.append(f"burden wmean {burden['weighted_mean']:.4f} vs frozen {frozen_burden:.4f}")
    if fails:
        logger.error("REPRODUCTION GATE FAILED: %s — nothing written", fails)
        return 1
    logger.info(
        "Reproduction gate OK: AUROC %.4f, burden %.4f reproduce frozen",
        auroc["weighted_mean"],
        burden["weighted_mean"],
    )

    # interpretation-map check on the AUROC survival claim
    auroc_lb = min(auroc["t_gminus1_ci"][0], auroc["cluster_bootstrap_ci"][0])
    survives = auroc_lb > 0.5
    if not survives:
        logger.error(
            "ESCALATE: widest AUROC lower bound %.4f <= 0.5 (survival claim at risk)", auroc_lb
        )

    payload = {
        "_meta": {
            "source": (
                "C12 (R5 M5) t(G-1)- and cluster-bootstrap-calibrated CIs for the LORO "
                "provenance-reduced AUROC and the national PoC burden ratio, from the G=10 "
                "per-region values in loro_cv_provenance_free.json / loro_equity_analysis.json. "
                f"Reproduction gate ties both weighted means to inference_strengthening within {GATE_TOL}. "
                "The block bootstrap resamples only G=10 blocks and is anti-conservative; the "
                "t(9) interval widens it honestly. No retrain."
            ),
            "t_crit_9": T_CRIT_9,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "auroc_survives_out_of_region": bool(survives),
            "widest_auroc_lower_bound": auroc_lb,
        },
        "loro_auroc": auroc,
        "poc_burden": burden,
        "t4_block_over_naive_ratio": sbb["T4_xgboost"]["width_ratio_block_over_naive"],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "AUROC: block width %.4f -> t(9) %.4f (x%.2f); survives=%s (lb %.3f)",
        auroc["block_bootstrap_width"],
        auroc["t_gminus1_width"],
        auroc["t_gminus1_width"] / auroc["block_bootstrap_width"],
        survives,
        auroc_lb,
    )
    logger.info(
        "burden: block width %.4f -> t(9) %.4f; t(9) CI [%.3f, %.3f]",
        burden["block_bootstrap_width"],
        burden["t_gminus1_width"],
        burden["t_gminus1_ci"][0],
        burden["t_gminus1_ci"][1],
    )
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
