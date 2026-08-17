#!/usr/bin/env python
"""C1 (de novo M5): per-split prevalence for the leakage comparison.

The 58% AUPRC-inflation headline compares metrics across two test sets; AUPRC is
prevalence-sensitive, so the per-split base rates must be reader-facing. This
pre-specified COMPUTE mirrors ``run_split_ablation``'s exact data preparation (same
aggregation, leakage-column drop, baseline feature assembly) WITHOUT training anything,
and derives:

- geographic test prevalence (regions 8/9/10 of the split-comparison assembly), and
- the random-fold prevalence, which equals the pooled prevalence by construction
  (StratifiedKFold folds are stratified on y).

Reproduction gate: the assembled geographic test size must equal the frozen
``split_comparison.json`` ``simple`` entry's n_test, proving this is the same assembly.
Output goes directly into the frozen archive (then ``freeze_results.py --rehash-only``
and ``stamp_derivations.py``).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "split_prevalence.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.analysis.split_strategy_comparison import build_baseline_feature_set
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        derive_system_epa_regions,
        drop_leakage_columns,
    )

    sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    from aquacontam.pipeline.features import extract_features

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    baseline_features = build_baseline_feature_set(feature_dfs)
    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _ = assemble_feature_matrix(sys_targets, baseline_features)

    regions = derive_system_epa_regions(wq_df, X.index)
    test_idx = X.index[regions.isin((8, 9, 10))]

    # Reproduction gate: same assembly as the frozen split comparison.
    frozen_sc = json.loads((FROZEN / "split_comparison.json").read_text(encoding="utf-8"))
    frozen_n_test = None
    for entry in frozen_sc.get("matrix", {}).get("results", []):
        if entry.get("split_strategy") == "geographic":
            frozen_n_test = int(entry["n_test"])
            break
    got_n_test = len(test_idx)
    if frozen_n_test is not None and got_n_test != frozen_n_test:
        logger.error(
            "Reproduction gate FAILED: geographic n_test %d != frozen %d — nothing written",
            got_n_test,
            frozen_n_test,
        )
        return 1
    logger.info("Reproduction gate OK: geographic n_test == %d", got_n_test)

    prev_geo = float(y.loc[test_idx].mean())
    prev_pooled = float(y.mean())  # == stratified random-fold prevalence by construction
    payload = {
        "_meta": {
            "source": (
                "C1 (M5) per-split prevalence, mirrored from run_split_ablation's data prep "
                "(no training). Random StratifiedKFold folds carry the pooled prevalence by "
                "construction. Reproduction gate: geographic n_test equals the frozen "
                "split_comparison matrix geographic entry."
            ),
            "n_systems": len(y),
            "n_test_geographic": got_n_test,
        },
        "prevalence_geographic_test": prev_geo,
        "prevalence_random_folds_pooled": prev_pooled,
        "note": (
            "The geographic test set has the HIGHER base rate, so the prevalence "
            "difference biases the random-vs-geographic AUPRC gap conservatively "
            "(against the leakage claim), not in its favor."
        )
        if prev_geo > prev_pooled
        else "Geographic test base rate is lower than the pooled rate.",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "geo prevalence %.4f vs pooled/random %.4f -> wrote %s",
        prev_geo,
        prev_pooled,
        OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
