"""Compute the inference-discipline statistics requested in peer review (M3/M4).

Post-hoc statistics on the frozen per-region artifacts (no model re-training):

- **M4a** - region-block-bootstrap CI for the *provenance-free* out-of-region
  (LORO) mean AUROC, the central deflated estimate (~0.698). Mirrors the
  with-provenance CI already in ``spatial_block_bootstrap.json``.
- **M4c** - region-block-bootstrap CI for the national, system-count-weighted
  people-of-colour detection burden ratio (~1.50).
- **M4b** - per-region AUC-vs-chance test (Hanley-McNeil one-sample) for the
  weakest and strongest provenance-free LORO folds; per-fold scores are not
  stored, so class counts are recovered from the fold's confusion-matrix
  summary (balanced accuracy + recall + accuracy + n_test).
- **M3b** - two-independent-AUC test of the high- vs low-burden per-group AUROC
  gap (0.756 vs 0.842) for the people-of-colour dimension in the test region.

Writes ``<results>/inference_strengthening.json``. Run against the frozen
snapshot for the published numbers::

    python paper/compute_inference_strengthening.py --results results/paper_frozen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquacontam.analysis.strengthening import (
    auc_difference,
    auc_vs_chance,
    region_block_bootstrap,
)

# Provenance-free LORO folds to test against chance (3 weakest, 2 strongest).
_M4B_REGIONS = [2, 6, 4, 5, 10]


def _recover_class_counts(fold: dict) -> tuple[int, int]:
    """Recover (n_pos, n_neg) from a fold's confusion-matrix summary.

    Solves the 2x2 confusion matrix from recall, specificity (= 2*balanced
    accuracy - recall), overall accuracy and the test-set size::

        n_pos = n * (accuracy - specificity) / (recall - specificity)
    """
    m = fold["detection_only_ablation"]["all_sources"]
    n = int(fold["n_test"])
    recall = float(m["recall"])
    bal_acc = float(m["balanced_accuracy"])
    acc = float(m["accuracy"])
    spec = 2.0 * bal_acc - recall
    denom = recall - spec
    if abs(denom) < 1e-9:
        raise ValueError("cannot recover class counts (recall == specificity)")
    n_pos = round(n * (acc - spec) / denom)
    n_pos = max(1, min(n - 1, n_pos))
    return n_pos, n - n_pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/paper_frozen", help="Frozen results dir.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    R = Path(args.results)

    loro_pf = json.loads((R / "loro_cv_provenance_free.json").read_text())
    loro_eq = json.loads((R / "loro_equity_analysis.json").read_text())
    equity = json.loads((R / "equity_analysis.json").read_text())

    out: dict[str, object] = {}

    # --- M4a: provenance-free LORO block-bootstrap CI -----------------------
    pf_folds = loro_pf["T1"]["xgboost"]["folds"]
    out["m4a_provenance_free_loro_ci"] = region_block_bootstrap(pf_folds, seed=args.seed)

    # --- M4c: national POC burden-ratio block-bootstrap CI ------------------
    per_region = loro_eq["per_region"]
    br_blocks = []
    for entry in per_region:
        if entry.get("group") != "pct_people_of_color":
            continue
        br = entry.get("burden_ratio")
        nsys = entry.get("n_systems") or (entry.get("n_high", 0) + entry.get("n_low", 0))
        if br is not None and br == br and nsys:  # finite check (br == br excludes NaN)
            br_blocks.append({"auroc": float(br), "n_test": int(nsys)})
    m4c = region_block_bootstrap(br_blocks, seed=args.seed)
    # Re-label the AUROC-named fields for clarity (the helper is generic).
    out["m4c_burden_ratio_ci"] = {
        "n_regions": m4c.get("n_folds"),
        "weighted_mean_burden_ratio": m4c.get("weighted_mean_auroc"),
        "block_bootstrap": m4c.get("block_bootstrap"),
        "n_boot": m4c.get("n_boot"),
    }

    # --- M4b: per-region AUC-vs-chance --------------------------------------
    folds_by_region = {f["test_region"]: f for f in pf_folds}
    m4b = []
    for region in _M4B_REGIONS:
        fold = folds_by_region[region]
        n_pos, n_neg = _recover_class_counts(fold)
        res = auc_vs_chance(fold["auroc"], n_pos, n_neg, two_sided=True)
        res["test_region"] = region
        m4b.append(res)
    out["m4b_per_region_auc_vs_chance"] = m4b

    # --- M3b: high- vs low-burden AUROC gap (people of colour) --------------
    poc = next(e for e in equity if e["group"] == "pct_people_of_color")
    hi, lo = poc["group_metrics"]["high"], poc["group_metrics"]["low"]
    hi_pos = round(hi["n_samples"] * hi["positive_rate"])
    lo_pos = round(lo["n_samples"] * lo["positive_rate"])
    out["m3b_group_auroc_gap"] = auc_difference(
        hi["auroc"],
        hi_pos,
        int(hi["n_samples"]) - hi_pos,
        lo["auroc"],
        lo_pos,
        int(lo["n_samples"]) - lo_pos,
    )

    dest = R / "inference_strengthening.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
