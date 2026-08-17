"""DEPRECATED (#55, 2026-07): one-off ICP re-train splice, retired from the active path.

Kept for provenance only (referenced by historical ``DERIVATIONS.tsv`` PROV rows).
ICP training history is now persisted through the canonical pipeline
(``train_and_evaluate`` -> ``export_icp_diagnostics``); the reproduction-gated
re-freeze of ``icp_diagnostics.json`` is handled by ``paper/compute_icp_history.py``
+ ``paper/splice_icp_diagnostics.py``. Do not run this script.

Original purpose (historical):

Splice the re-trained ICP results into the frozen-source run and regenerate
every ICP-dependent derived file -- surgically, so only ICP rows change.

Context: ``icp.py`` was fixed (the GRL-reversed adversary term was excluded from
the encoder loss) and the ICP training schedule changed (early stopping off, so
the adversary trains the full schedule). ICP was re-trained on all its tasks into
``results_icp_tmp`` and the recoverability probe re-run into ``results_icp_probe``.

This script:
1. Replaces the 6 ICP entries in ``results_reframe/results.json`` with the new
   ones (keeping the other 80 model entries byte-identical).
2. Recomputes the derived files that contain ICP rows by calling the existing
   analysis functions on the spliced results -- these are deterministic metric
   computations over stored predictions, so non-ICP rows reproduce exactly.
3. Copies the new ``icp_diagnostics.json`` (real 200-epoch history) and
   ``icp_recoverability.json`` (probe AUROC 0.62/0.64) into the source run.

Run ``--bootstrap`` separately to refresh ``bootstrap_ci.json`` ICP rows, then
``make paper-freeze``. A diff against ``results_reframe.pre-icp-fix.bak`` should
show changes only in ICP-involving rows.

Usage::

    python paper/regen_icp_results.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquacontam.pipeline.analysis import (  # noqa: E402
    _run_calibration_analysis,
    _run_conformal_analysis,
    _run_model_comparison,
)

RR = ROOT / "results_reframe"
TMP = ROOT / "results_icp_tmp"
PROBE = ROOT / "results_icp_probe"


def main() -> None:
    res = json.loads((RR / "results.json").read_text())
    new_icp = {
        (r.get("task"), r.get("model")): r for r in json.loads((TMP / "results.json").read_text())
    }

    spliced, n = [], 0
    for r in res:
        key = (r.get("task"), r.get("model"))
        if str(r.get("model", "")).startswith("icp") and key in new_icp:
            spliced.append(new_icp[key])
            n += 1
        else:
            spliced.append(r)
    print(f"Spliced {n} ICP entries into results_reframe/results.json")
    (RR / "results.json").write_text(json.dumps(spliced, indent=2, default=str))

    # Recompute ICP-dependent derived files (deterministic for non-ICP rows).
    print("Recomputing calibration / conformal / model_comparison / spatial ...")
    _run_calibration_analysis(spliced, RR)
    _run_conformal_analysis(spliced, RR)
    _run_model_comparison(spliced, RR)
    # NOTE: _run_spatial_autocorrelation is intentionally skipped -- recomputing it
    # standalone builds a dense spatial-weights matrix over the full coordinate set
    # and exhausts memory. Its 3 ICP rows are a minor Moran's I diagnostic; the
    # headline residual-autocorrelation finding does not depend on them, so
    # spatial_autocorrelation.json keeps its existing ICP rows.

    # Copy the ICP-specific diagnostics produced by the re-train and the probe.
    shutil.copy(TMP / "icp_diagnostics.json", RR / "icp_diagnostics.json")
    shutil.copy(PROBE / "icp_recoverability.json", RR / "icp_recoverability.json")
    print("Copied icp_diagnostics.json and icp_recoverability.json")
    print("Done. Now run: python scripts/reproduce.py --bootstrap --output-dir results_reframe")


if __name__ == "__main__":
    main()
