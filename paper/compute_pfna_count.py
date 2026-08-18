#!/usr/bin/env python
"""Quantify the PFNA-only exceeders omitted from the MCL-exceedance union.

The MCL-exceedance target (``mcl_exceedance_analysis.json``) unions five analytes
(PFOS/PFOA/PFHxS/HFPO-DA individual MCLs + PFBS via its Hazard-Index HBWC) and, for
source-coverage parity, silently OMITS the PFNA individual MCL (0.010 ug/L). R5 (m9) asks
that this be stated and quantified: a system that exceeds only PFNA is currently mislabelled
non-exceedant. This COMPUTE counts them by re-running the exact ``compute_mcl_exceedance``
labeller for {PFNA: 0.010} and for the default union, then counting PFNA-exceed AND
NOT-union-exceed systems. No retrain.

Reproduction gate: the default-union exceedance count reproduces ``mcl_exceedance_analysis.json``
mcl_summary (n_exceeding) before the PFNA delta is written.

Output: ``results/paper_frozen/pfna_exceedance_count.json`` → one main-text-or-SI sentence.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "pfna_exceedance_count.json"
PFNA_MCL = 0.010  # ug/L (10 ppt individual MCL, omitted from the union)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_pfna_count")


def main() -> int:
    import pandas as pd

    from aquacontam.analysis.mcl_threshold import (
        DEFAULT_MCL_THRESHOLDS,
        compute_mcl_exceedance,
        mcl_summary_statistics,
    )

    wq_df = pd.read_parquet(REPO / "data" / "interim" / "merged_wq.parquet")

    union = compute_mcl_exceedance(wq_df).set_index("pwsid")["mcl_exceedance"]
    stats = mcl_summary_statistics(compute_mcl_exceedance(wq_df))

    # --- reproduction gate: default union reproduces the frozen mcl_summary ---
    frozen = json.loads((FROZEN / "mcl_exceedance_analysis.json").read_text(encoding="utf-8"))
    frozen_n_exc = int(frozen["mcl_summary"]["n_exceeding"])
    if int(union.sum()) != frozen_n_exc:
        logger.error(
            "REPRODUCTION GATE FAILED: union exceeders %d vs frozen %d — nothing written",
            int(union.sum()),
            frozen_n_exc,
        )
        return 1
    logger.info(
        "Reproduction gate OK: %d union exceeders reproduce frozen mcl_summary", frozen_n_exc
    )

    pfna = compute_mcl_exceedance(wq_df, {"PFNA": PFNA_MCL}).set_index("pwsid")["mcl_exceedance"]
    common = union.index.intersection(pfna.index)
    u, p = union.loc[common], pfna.loc[common]
    pfna_only = int(((p == 1) & (u == 0)).sum())
    pfna_total = int((p == 1).sum())
    n_pfna_systems = int(pfna.index.nunique())

    payload = {
        "_meta": {
            "source": (
                "C17 (R5 m9) PFNA-only MCL exceeders — systems exceeding the PFNA individual "
                "MCL (0.010 ug/L) but no analyte in the default union "
                f"({', '.join(DEFAULT_MCL_THRESHOLDS)}). Reproduction gate ties the default-union "
                "exceedance count to mcl_exceedance_analysis.json mcl_summary. No retrain."
            ),
        },
        "pfna_mcl_ug_per_l": PFNA_MCL,
        "n_systems_with_pfna_data": n_pfna_systems,
        "n_pfna_exceeders": pfna_total,
        "n_pfna_only_exceeders": pfna_only,
        "union_exceeders": frozen_n_exc,
        "exceedance_rate": stats.get("exceedance_rate"),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "PFNA exceeders=%d (of %d systems w/ PFNA), PFNA-ONLY=%d; union=%d -> wrote %s",
        pfna_total,
        n_pfna_systems,
        pfna_only,
        frozen_n_exc,
        OUT_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
