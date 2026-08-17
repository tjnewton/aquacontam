"""Per-group error rates (FPR/FNR) and calibration (ECE) for the T1 model.

Addresses referee M4: the single per-group AUROC pair (0.756 vs 0.842) does not
convey the *deployment-relevant* error asymmetry, and the fairness gap is never
reconciled with the deployment recommendation. This script quantifies, for the
T1 PFAS-detection model on the geographically held-out test set (EPA regions
8/9/10), the per-group false-negative rate (an unflagged contaminated system),
false-positive rate, and expected calibration error, stratified by demographic
burden.

Two parts, by feasibility:

1. **FPR / FNR** are derived *exactly* from the frozen ``equity_analysis.json``
   group metrics. The reported ``recall``, ``precision``, ``positive_rate`` and
   ``n_samples`` fully determine each group's confusion matrix, so no model
   re-run is needed (closed form below). This is the deployment-critical
   asymmetry and is computed with zero risk to any frozen number.

2. **ECE** needs per-system probabilities, which were slimmed out of the public
   snapshot. We reproduce them by retraining T1 (xgboost, seed 42) from the
   canonical ``merged_wq`` via the exact pipeline functions and replaying the
   equity prediction path, **gated** on reproducing the frozen per-group AUROCs
   to 2e-3. The reproduction writes only to a scratch dir; if the gate fails
   (e.g. a feature source is unavailable), ECE is reported as null and the
   limitation is owned in prose. No existing frozen file is mutated.

Output: ``results/paper_frozen/group_error_calibration.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
GROUP_COLS = (
    "pct_people_of_color",
    "pct_low_income",
    "pct_limited_english",
    "pct_less_hs_education",
)
GATE_TOL = 2e-3


def _derive_fpr_fnr(gm: dict[str, float]) -> tuple[float, float]:
    """Closed-form FPR/FNR from a group's frozen confusion-matrix summary.

    Given prevalence ``p`` (positive_rate), ``recall`` (=TPR) and ``precision``::

        FNR = 1 - recall
        FPR = recall * p * (1 - precision) / (precision * (1 - p))

    Returns ``(nan, nan)`` when the group lacks the fields or is degenerate.
    """
    p = gm.get("positive_rate")
    recall = gm.get("recall")
    precision = gm.get("precision")
    if p is None or recall is None or precision is None:
        return float("nan"), float("nan")
    if not (0.0 < p < 1.0) or precision <= 0.0:
        return float("nan"), float(1.0 - recall) if recall is not None else float("nan")
    fnr = 1.0 - recall
    fpr = (recall * p * (1.0 - precision)) / (precision * (1.0 - p))
    return float(fpr), float(fnr)


def _part1_error_rates() -> dict[str, Any]:
    """Per-group FPR/FNR derived exactly from frozen equity_analysis.json."""
    equity = json.loads((FROZEN / "equity_analysis.json").read_text())
    by_group = {e["group"]: e for e in equity}
    out: dict[str, Any] = {}
    for col in GROUP_COLS:
        entry = by_group.get(col)
        if entry is None:
            continue
        gm = entry.get("group_metrics", {})
        dim: dict[str, Any] = {}
        for level in ("high", "low"):
            g = gm.get(level)
            if not g:
                continue
            fpr, fnr = _derive_fpr_fnr(g)
            dim[level] = {
                "n": int(g.get("n_samples", 0)),
                "positive_rate": g.get("positive_rate"),
                "recall": g.get("recall"),
                "precision": g.get("precision"),
                "auroc": g.get("auroc"),
                "fpr": fpr,
                "fnr": fnr,
            }
        if "high" in dim and "low" in dim:
            dim["fnr_gap"] = dim["high"]["fnr"] - dim["low"]["fnr"]
            dim["fpr_gap"] = dim["high"]["fpr"] - dim["low"]["fpr"]
        out[col] = dim
    return out


def _discover_downloaded(data_path: Path) -> dict[str, Path]:
    """Replicate reproduce.py's existing-file discovery (no re-download)."""
    downloaded: dict[str, Path] = {}
    for sub in ("raw", "interim", "processed"):
        d = data_path / sub
        if not d.exists():
            continue
        for p in d.glob("*.parquet"):
            downloaded.setdefault(p.stem, p)
    for p in (data_path / "raw").glob("*.shp"):
        downloaded[p.stem] = p
    for p in (data_path / "raw").rglob("*.tif"):
        if not p.name.startswith("._"):
            downloaded.setdefault(p.stem, p)
    if "us_aquifers" in downloaded:
        downloaded["usgs_aquifers"] = downloaded["us_aquifers"]
    # NLCD canonical key
    for stem in list(downloaded):
        if "nlcd" in stem.lower():
            downloaded.setdefault("nlcd", downloaded[stem])
    return downloaded


def _part2_calibration() -> dict[str, Any]:
    """Reproduce T1 to obtain per-group ECE, gated on frozen-AUROC match."""
    status: dict[str, Any] = {"calibration_status": "unavailable", "gate": {}}
    try:
        import pandas as pd

        import aquacontam.benchmark  # noqa: F401  (triggers @register_task registration)
        from aquacontam.pipeline.analysis import _run_equity_analysis
        from aquacontam.pipeline.features import extract_features
        from aquacontam.pipeline.training import train_and_evaluate

        data_path = REPO / "data"
        merged = data_path / "interim" / "merged_wq.parquet"
        wq_df = pd.read_parquet(merged)
        n_sdwis = int((wq_df["source"] == "sdwis").sum()) if "source" in wq_df.columns else -1
        # Canonical SDWIS row count after the keep-non-detects decision (was 694,419
        # when non-detect 0 records were dropped; 916,899 once they are retained).
        if n_sdwis != 916899:
            status["gate"]["sdwis_rows"] = n_sdwis
            status["calibration_status"] = f"aborted: SDWIS drift ({n_sdwis} != 916899)"
            return status

        downloaded = _discover_downloaded(data_path)
        feature_dfs = extract_features(data_path, wq_df, downloaded, skip_large=False)

        with tempfile.TemporaryDirectory(prefix="m4_repro_") as td:
            scratch = Path(td)
            _results, fitted, _ = train_and_evaluate(
                wq_df,
                feature_dfs,
                downloaded,
                scratch,
                seed=42,
                task_filter=["T1"],
                model_filter=["xgboost"],
                resume=False,
            )
            if "T1" not in fitted:
                status["calibration_status"] = "aborted: T1 not fitted"
                return status
            _run_equity_analysis(wq_df, feature_dfs, fitted, scratch)
            repro_path = scratch / "equity_analysis.json"
            if not repro_path.exists():
                status["calibration_status"] = "aborted: no equity output"
                return status
            repro = {e["group"]: e for e in json.loads(repro_path.read_text())}

        # Reproduction gate: per-group AUROC must match frozen to GATE_TOL.
        frozen = {e["group"]: e for e in json.loads((FROZEN / "equity_analysis.json").read_text())}
        max_dev = 0.0
        ece: dict[str, Any] = {}
        for col in GROUP_COLS:
            if col not in repro or col not in frozen:
                continue
            rgm = repro[col].get("group_metrics", {})
            fgm = frozen[col].get("group_metrics", {})
            dim_ece: dict[str, float] = {}
            for level in ("high", "low"):
                if (
                    level in rgm
                    and level in fgm
                    and "auroc" in rgm[level]
                    and "auroc" in fgm[level]
                ):
                    max_dev = max(max_dev, abs(rgm[level]["auroc"] - fgm[level]["auroc"]))
                if level in rgm and "ece" in rgm[level]:
                    dim_ece[level] = rgm[level]["ece"]
            if dim_ece:
                ece[col] = dim_ece
        status["gate"]["max_auroc_deviation"] = max_dev
        if max_dev <= GATE_TOL:
            status["calibration_status"] = "reproduced"
            status["ece"] = ece
        else:
            status["calibration_status"] = f"gate_failed (max dev {max_dev:.4f} > {GATE_TOL})"
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Calibration reproduction failed", exc_info=True)
        status["calibration_status"] = f"error: {type(exc).__name__}: {exc}"
    return status


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    error_rates = _part1_error_rates()
    calib = _part2_calibration()

    # Merge ECE into the per-group structure when reproduced.
    ece = calib.get("ece", {})
    for col, dim in error_rates.items():
        for level in ("high", "low"):
            if level in dim and col in ece and level in ece[col]:
                dim[level]["ece"] = ece[col][level]

    out = {
        "error_rates": error_rates,
        "_meta": {
            "source": "FPR/FNR derived exactly from results/paper_frozen/equity_analysis.json "
            "group_metrics; ECE from gated T1 reproduction (seed 42, canonical merged_wq).",
            "calibration_status": calib.get("calibration_status"),
            "reproduction_gate": calib.get("gate", {}),
            "gate_tolerance": GATE_TOL,
        },
    }
    out_path = FROZEN / "group_error_calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s (calibration_status=%s)", out_path, calib.get("calibration_status"))

    poc = error_rates.get("pct_people_of_color", {})
    if "high" in poc and "low" in poc:
        logger.info(
            "POC FNR high=%.3f low=%.3f (gap %.3f); FPR high=%.3f low=%.3f",
            poc["high"]["fnr"],
            poc["low"]["fnr"],
            poc["fnr_gap"],
            poc["high"]["fpr"],
            poc["low"]["fpr"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
