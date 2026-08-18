"""Capture the ICP T1 training history for ED Fig. 4a via the CANONICAL path.

The ICP records per-epoch task/adversary/lambda history during ``fit()``. Since
the durable fix, ``train_and_evaluate`` retains the fitted ICP per task
(``icp_models``) and ``export_icp_diagnostics`` persists its history — so this
driver runs the ICP on T1 through **exactly that canonical pipeline path** (not a
bespoke re-fit), capturing the per-epoch history AND the geographic-**test**
metric in one run, then reproduction-gates that metric against the frozen
``results.json`` ICP T1 AUROC before writing. This makes panel a the training
trace of a run that provably reproduces the panel-b metric, at the same
(seeded, GPU, best-effort-deterministic) level as the rest of the pipeline.

Reproduction gate: the run's T1 ICP test AUROC must reproduce frozen
``results.json`` ICP T1 within ``GATE_TOL``; otherwise STOP and write nothing
(the caller falls back to a metadata-only correction of the existing frozen
history; ratified maintainer decision).

Run::

    PYTHONUTF8=1 python paper/compute_icp_history.py --gpu-id 2
    python paper/splice_icp_diagnostics.py
    python paper/freeze_results.py --only icp_diagnostics.json
"""

from __future__ import annotations

import os as _os
import sys as _sys


def _early_gpu_env_setup() -> None:
    gpu_id: str | None = None
    for i, tok in enumerate(_sys.argv):
        if tok == "--gpu-id" and i + 1 < len(_sys.argv):
            gpu_id = _sys.argv[i + 1]
            break
        if tok.startswith("--gpu-id="):
            gpu_id = tok.split("=", 1)[1]
            break
    _os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if gpu_id is not None:
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        _os.environ["AQUACONTAM_GPU_ID"] = "0"


_early_gpu_env_setup()

import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT = REPO / "results" / "icp_history.json"
SEED = 42
GATE_TOL = 0.02

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("icp_history")


def main() -> int:
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam._reproducibility import set_seed
    from aquacontam.models._torch_utils import get_device
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.training import train_and_evaluate

    _sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    set_seed(SEED)

    # Frozen anchor: ICP T1 test AUROC that panel b reports.
    frozen_results = json.loads((FROZEN / "results.json").read_text(encoding="utf-8"))
    frozen_auroc = next(
        float(e["metrics"]["auroc"])
        for e in frozen_results
        if e.get("task") == "T1" and e.get("model") == "icp_classifier"
    )

    data_path = REPO / "data"
    downloaded = _discover_downloaded(data_path)
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)

    # Canonical path, scoped to T1 + ICP. train_and_evaluate excludes ambient
    # sources and applies the geographic split internally (same as --all).
    with tempfile.TemporaryDirectory(prefix="icp_hist_") as td:
        results, _fitted, icp_models = train_and_evaluate(
            wq_df,
            feature_dfs,
            downloaded,
            Path(td),
            seed=SEED,
            task_filter=["T1"],
            model_filter=["icp"],
            resume=False,
        )

    icp_row = next(
        (r for r in results if r.get("task") == "T1" and r.get("model") == "icp_classifier"),
        None,
    )
    if icp_row is None or "T1" not in icp_models:
        logger.error("ICP T1 run produced no result / no captured model")
        return 1
    test_auroc = float(icp_row["metrics"]["auroc"])
    logger.info("T1 ICP test AUROC=%.4f (frozen %.4f)", test_auroc, frozen_auroc)

    # --- reproduction gate on the reader-facing (panel b) metric ---
    if abs(test_auroc - frozen_auroc) > GATE_TOL:
        logger.error(
            "REPRODUCTION GATE FAILED: test AUROC %.4f vs frozen %.4f exceeds tol %.3f — "
            "nothing written; use the metadata-only fallback.",
            test_auroc,
            frozen_auroc,
            GATE_TOL,
        )
        return 2

    model = icp_models["T1"]
    history = model.get_training_history()
    if not history:
        logger.error("captured ICP model has empty training history")
        return 1
    n_epochs = len(history)
    warmup_end = next((h["epoch"] for h in history if h.get("lambda_adv", 0) > 0), None)
    adv_r2 = model.get_adversary_r2() if hasattr(model, "get_adversary_r2") else None
    device = str(get_device())

    payload = {
        "_meta": {
            "seed": SEED,
            "device": device,
            "test_auroc": round(test_auroc, 4),
            "frozen_test_auroc": round(frozen_auroc, 4),
            "gate_tol": GATE_TOL,
            "adversary_r2": round(float(adv_r2), 4) if adv_r2 is not None else None,
            "n_epochs": n_epochs,
            "warmup_end": warmup_end,
            "note": (
                "ICP T1 training trace captured via the canonical train_and_evaluate path "
                "(seed 42, device as stamped). The same run reproduces the frozen panel-b "
                f"test AUROC ({frozen_auroc:.3f}) within {GATE_TOL}, so panel a is the training "
                "dynamics of the panel-b model (not a separate re-fit)."
            ),
        },
        "T1": history,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(
        "Gate OK. Wrote %s (%d epochs, warmup_end=%s, device=%s)",
        OUT,
        n_epochs,
        warmup_end,
        device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
