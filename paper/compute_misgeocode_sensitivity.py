#!/usr/bin/env python
"""Wrong-ZIP mis-geocode flag counts + exclusion sensitivity.

SUPERSEDED (2026-07-12): the frozen artifact this script wrote
(``misgeocode_sensitivity.json``) is retained as the bounding-box-rule record, but the
figure caption and Supplementary S7b now cite ``misgeocode_sensitivity_v2.json`` from
``compute_misgeocode_sensitivity_v2.py`` (exact state-polygon flag). This script now
imports ``region_mismatch_mask_bbox_legacy`` (the rule it originally froze under the
name ``region_mismatch_mask``) so a re-run still reproduces the v1 artifact instead of
silently clobbering it with polygon-rule counts.

Some systems' plotted coordinates come from an out-of-state mailing/operator
ZIP centroid, placing them deep inside foreign EPA regions.
The geographic split is prefix-derived and therefore uncontaminated, but the
mis-located systems carry wrong-place proximity/land-use features. This
COMPUTE quantifies the exposure:

1. Flags every system whose PWSID-prefix EPA region disagrees with its
   coordinate-derived region (``paper/_geo_flags.region_mismatch_mask`` — the
   same rule the split map uses, so the frozen count and the figure caption
   agree by construction).
2. Re-runs the T1 and T4 xgboost benchmark tasks on the canonical headline
   harness with the flagged systems excluded end-to-end (train + eval), and
   reports the deltas against the unperturbed run.

Reproduction gate: the unperturbed seed-42 re-run must reproduce the frozen
``results.json`` T1 and T4 xgboost AUROC within ``GATE_TOL`` before anything
is written — proving harness equivalence (same protocol as
``compute_headline_stability.py`` C11).

Escalation rule (pre-registered): if excluding the flagged systems moves
either task's AUROC by more than ``MATERIAL_BAND`` the degradation is
material (the headline would depend on mis-geocoded coordinates — an S0):
STOP, write nothing, exit 2.

Usage::

    PYTHONUTF8=1 python paper/compute_misgeocode_sensitivity.py --gpu-id 2
    python paper/freeze_results.py --rehash-only
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
    if gpu_id is not None:
        _os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        _os.environ["AQUACONTAM_GPU_ID"] = "0"


_early_gpu_env_setup()

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
OUT_PATH = FROZEN / "misgeocode_sensitivity.json"
GATE_TOL = 2e-3
MATERIAL_BAND = 0.02  # |delta AUROC| beyond this = material degradation -> STOP (S0)
TASKS = ["T1", "T4"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_misgeocode_sensitivity")


def main() -> int:
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401
    from aquacontam.benchmark.registry import get_task
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.models import get_model_instances
    from aquacontam.pipeline.training import _exclude_ambient_sources, build_task_kwargs

    _sys.path.insert(0, str(REPO / "paper"))
    from _geo_flags import region_mismatch_mask_bbox_legacy as region_mismatch_mask
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    downloaded = _discover_downloaded(data_path)
    merged = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")

    # --- 1. the flag, on the same input the split map plots (full merged set) ---
    coords = merged.groupby("pwsid")[["latitude", "longitude"]].median().dropna()
    mismatch = region_mismatch_mask(coords.reset_index())
    flagged_pwsids = set(coords.index[mismatch.to_numpy()])
    n_mapped = len(coords)
    n_flagged = len(flagged_pwsids)
    logger.info("mapped systems=%d | region-mismatched flagged=%d", n_mapped, n_flagged)

    wq_df = _exclude_ambient_sources(merged)
    feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)

    frozen_results = json.loads((FROZEN / "results.json").read_text(encoding="utf-8"))

    def frozen_metrics(task: str) -> dict[str, float]:
        e = next(
            e
            for e in frozen_results
            if e.get("task") == task and e.get("model") == "xgboost_classifier"
        )
        return {"auroc": float(e["metrics"]["auroc"]), "auprc": float(e["metrics"]["auprc"])}

    def run_task(task: str, frame: pd.DataFrame) -> dict[str, float]:
        model = get_model_instances(
            ["xgboost"], "classification", 42, task_name=task, use_tuned_params=False
        )[0]
        _, task_fn = get_task(task)
        kwargs = build_task_kwargs(task, model, frame, feature_dfs, downloaded)
        m = task_fn(**kwargs).metrics
        return {"auroc": float(m["auroc"]), "auprc": float(m["auprc"])}

    # --- 2. reproduction gate: unperturbed run must reproduce frozen headline ---
    unperturbed: dict[str, dict[str, float]] = {}
    for task in TASKS:
        frozen = frozen_metrics(task)
        got = run_task(task, wq_df)
        unperturbed[task] = got
        logger.info(
            "%s unperturbed AUROC=%.4f (frozen %.4f) AUPRC=%.4f",
            task,
            got["auroc"],
            frozen["auroc"],
            got["auprc"],
        )
        if abs(got["auroc"] - frozen["auroc"]) > GATE_TOL:
            logger.error(
                "REPRODUCTION GATE FAILED (%s): %.4f vs frozen %.4f — nothing written",
                task,
                got["auroc"],
                frozen["auroc"],
            )
            return 1
    logger.info("Reproduction gate OK on both tasks: this IS the headline harness")

    # --- 3. exclusion arm ---
    wq_excl = wq_df[~wq_df["pwsid"].isin(flagged_pwsids)].copy()
    n_flagged_benchmark = int(wq_df.loc[wq_df["pwsid"].isin(flagged_pwsids), "pwsid"].nunique())
    logger.info(
        "exclusion arm: %d rows -> %d rows (%d flagged systems in the benchmark frame)",
        len(wq_df),
        len(wq_excl),
        n_flagged_benchmark,
    )
    per_task: dict[str, dict[str, float]] = {}
    material = False
    for task in TASKS:
        excl = run_task(task, wq_excl)
        delta_auroc = excl["auroc"] - unperturbed[task]["auroc"]
        delta_auprc = excl["auprc"] - unperturbed[task]["auprc"]
        per_task[task] = {
            "auroc_unperturbed": unperturbed[task]["auroc"],
            "auprc_unperturbed": unperturbed[task]["auprc"],
            "auroc_excluded": excl["auroc"],
            "auprc_excluded": excl["auprc"],
            "delta_auroc": delta_auroc,
            "delta_auprc": delta_auprc,
        }
        logger.info(
            "%s excluded-arm AUROC=%.4f (delta %+0.4f) AUPRC=%.4f (delta %+0.4f)",
            task,
            excl["auroc"],
            delta_auroc,
            excl["auprc"],
            delta_auprc,
        )
        if abs(delta_auroc) > MATERIAL_BAND:
            material = True

    if material:
        logger.error(
            "ESCALATE (S0 candidate): |delta AUROC| exceeds %.3f on at least one task — "
            "headline depends on mis-geocoded systems. NOTHING WRITTEN.",
            MATERIAL_BAND,
        )
        return 2

    payload = {
        "_meta": {
            "source": (
                "C-F (#55 item 6) wrong-ZIP mis-geocode sensitivity. Flag: PWSID-prefix EPA "
                "region != coordinate-derived region (paper/_geo_flags.region_mismatch_mask; "
                "identical rule to the ED Fig. 1 split-map exclusion). Sensitivity: T1/T4 "
                "xgboost on the canonical headline harness (train_and_evaluate task fns, "
                "ambient sources excluded, geographic holdout, seed 42) with flagged systems "
                f"removed end-to-end. Reproduction gate (tol {GATE_TOL}): unperturbed run "
                "reproduces frozen results.json T1/T4 xgboost AUROC. Material-degradation "
                f"escalation band: |delta AUROC| > {MATERIAL_BAND}."
            ),
            "reproduction_gate": "PASSED",
            "material_degradation": False,
        },
        "n_mapped_systems": n_mapped,
        "n_flagged_systems": n_flagged,
        "n_flagged_systems_in_benchmark_frame": n_flagged_benchmark,
        "per_task": per_task,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", OUT_PATH)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
