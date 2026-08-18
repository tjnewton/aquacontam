#!/usr/bin/env python
"""Build the slim national T1-PFOS risk surface for the frozen archive.

F-pred provenance: the retrain gate was not exercised; instead
the slim surface is derived from the existing pipeline national inference
parquets (``results/predictions/``, generated 2026-06-29 by
``generate_webapp_predictions``). This script slims the T1/PFOS slice to the
four columns the supplementary risk map and webapp need
(grid_id/latitude/longitude/risk_score), keeping the frozen artifact far
under the archive's per-file size guard.

Reproduction gate: when ``results/predictions_slim.json`` already exists, the
regenerated payload must match it exactly (same rows, same rounding) or this
script refuses to write — so the committed frozen copy is provably the
deterministic image of the parquet inputs.

Usage::

    PYTHONUTF8=1 python paper/build_predictions_slim.py
    python paper/freeze_results.py --only predictions_slim.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SOURCE = Path("results/predictions/task=T1/analyte=PFOS/predictions.parquet")
OUT = Path("results/predictions_slim.json")

META = {
    "task": "T1",
    "analyte": "PFOS",
    "source": "results/predictions (pipeline xgboost national inference; 2026-06-29)",
    "n": None,  # filled below
    "note": "slim national PFAS-detection risk surface for the supplementary risk map + webapp",
}


def build() -> dict:
    """Slim the T1/PFOS prediction parquet to the 4-column JSON payload."""
    df = pd.read_parquet(SOURCE, columns=["grid_id", "latitude", "longitude", "risk_score"])
    df = df.sort_values("grid_id").reset_index(drop=True)
    # pandas (numpy) rounding, not Python round(): the original artifact was
    # rounded via DataFrame.round and the reproduction gate is byte-exact.
    df = df.round({"latitude": 5, "longitude": 5, "risk_score": 4})
    df["grid_id"] = df["grid_id"].astype(str)
    rows = df.to_dict(orient="records")
    meta = dict(META)
    meta["n"] = len(rows)
    return {"_meta": meta, "rows": rows}


def main() -> None:
    payload = build()
    if OUT.exists():
        existing = json.loads(OUT.read_text())
        if existing != payload:
            raise SystemExit(
                "REPRODUCTION GATE FAILED: regenerated slim payload differs from "
                f"existing {OUT} — inputs drifted; do not overwrite. Investigate."
            )
        print(f"reproduction gate OK: regenerated payload == existing {OUT}")
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, n={payload['_meta']['n']})")


if __name__ == "__main__":
    main()
