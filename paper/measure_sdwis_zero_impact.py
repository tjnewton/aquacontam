"""Quantify the T4 impact of dropping vs keeping SDWIS SAMPLE_MEASURE=0 non-detects.

Root cause (see paper/revision_ledger.md): the published canonical SDWIS set is the
694,419 `SAMPLE_MEASURE > 0` rows; the original pipeline dropped 222,550 zero rows
that are EPA non-detect / below-detection 90th-percentile results (clear
non-exceedances), removing 5,572 zero-only lead systems and biasing T4's base rate.

This script measures the T4 delta in isolation, WITHOUT touching frozen results:
- ENV/ROOT-CAUSE SANITY (drop): the canonical merged_wq is already the >0 set; apply
  the M1 detection_limit=NaN correction and confirm T4 reproduces the published
  results.json (env matches software_versions.json exactly). If it can't reproduce,
  STOP — a keep-vs-drop delta would be env-confounded.
- KEEP: add the 222,550 zero rows back (as non-detect non-exceedances, concentration
  0, censored False, detection_limit NaN), with per-system coordinates copied from the
  canonical rows (mixed systems) or the raw systems file (zero-only systems), then
  re-extract features and retrain T4. Report the delta + how many added systems survive
  feature assembly.

Censoring is held at the published behavior (all non-censored) across both modes so the
delta isolates the *zero* decision. Outputs a report JSON to the scratch dir; changes
nothing under results/paper_frozen.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
DATA = REPO / "data"
ENV_TOL = 5e-3  # env matches software_versions.json exactly -> tight, with small slack


def _discover_downloaded(data_path: Path) -> dict[str, Path]:
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
    for stem in list(downloaded):
        if "nlcd" in stem.lower():
            downloaded.setdefault("nlcd", downloaded[stem])
    return downloaded


def _t4_metrics(wq_df: Any, downloaded: dict[str, Path], models: list[str], scratch: Path) -> dict:
    """Run T4 for the given models on wq_df; return {model: {auroc, auprc, n_test}}."""
    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.training import train_and_evaluate

    feature_dfs = extract_features(DATA, wq_df, downloaded, skip_large=False)
    results, _fitted, _ = train_and_evaluate(
        wq_df,
        feature_dfs,
        downloaded,
        scratch,
        seed=42,
        task_filter=["T4"],
        model_filter=models,
        resume=False,
    )
    out: dict[str, Any] = {}
    for r in results:
        if r.get("task") == "T4":
            m = r.get("metrics", {})
            md = r.get("metadata", {})
            out[r.get("model")] = {
                "auroc": m.get("auroc"),
                "auprc": m.get("auprc"),
                "n_test": md.get("n_test"),
            }
    return out


def _build_keep_extra(merged: Any, sdwis_canon: Any) -> Any:
    """Reconstruct the dropped SAMPLE_MEASURE==0 SDWIS rows in merged_wq schema."""
    import numpy as np
    import pandas as pd

    from aquacontam._constants import SDWIS_CODE_TO_ANALYTE

    raw = pd.read_csv(
        DATA / "raw" / "SDWA_LCR_SAMPLES.csv", dtype=str, low_memory=False, keep_default_na=False
    )
    raw = raw[raw["CONTAMINANT_CODE"].isin(["PB90", "CU90"])].copy()
    raw["m"] = pd.to_numeric(raw["SAMPLE_MEASURE"], errors="coerce")
    zeros = raw[raw["m"] == 0].copy()
    zeros["pwsid"] = zeros["PWSID"].astype(str)
    zeros["analyte"] = zeros["CONTAMINANT_CODE"].map(SDWIS_CODE_TO_ANALYTE)
    zeros["concentration"] = 0.0
    zeros["censored"] = False
    zeros["detection_limit"] = np.nan
    zeros["unit"] = "ug/L"
    zeros["sample_date"] = pd.NaT
    zeros["source"] = "sdwis"

    # Per-system metadata (coords + raw_*) from the canonical rows (mixed systems).
    meta_cols = [
        c
        for c in merged.columns
        if c
        not in (
            "analyte",
            "concentration",
            "censored",
            "detection_limit",
            "unit",
            "sample_date",
            "source",
            "pwsid",
        )
    ]
    sys_meta = sdwis_canon.groupby("pwsid")[meta_cols].first()
    zeros = zeros.join(sys_meta, on="pwsid")

    # Zero-only systems (absent from canonical): coords from the raw systems file.
    missing = zeros["latitude"].isna() if "latitude" in zeros.columns else None
    n_zero_only_filled = 0
    try:
        sysf = pd.read_csv(
            DATA / "raw" / "SDWA_PUB_WATER_SYSTEMS.csv", dtype=str, low_memory=False
        )
        latc = next((c for c in ("GEO_LATITUDE", "LATITUDE") if c in sysf.columns), None)
        lonc = next((c for c in ("GEO_LONGITUDE", "LONGITUDE") if c in sysf.columns), None)
        if latc and lonc and missing is not None:
            sysf = sysf.drop_duplicates("PWSID").set_index("PWSID")
            lat_map = pd.to_numeric(sysf[latc], errors="coerce")
            lon_map = pd.to_numeric(sysf[lonc], errors="coerce")
            fill_lat = zeros.loc[missing, "pwsid"].map(lat_map)
            fill_lon = zeros.loc[missing, "pwsid"].map(lon_map)
            zeros.loc[missing, "latitude"] = fill_lat.to_numpy()
            zeros.loc[missing, "longitude"] = fill_lon.to_numpy()
            n_zero_only_filled = int(fill_lat.notna().sum())
    except (OSError, ValueError, KeyError):
        pass
    logger.info(
        "Reconstructed %d zero rows (%d zero-only coords from raw systems file)",
        len(zeros),
        n_zero_only_filled,
    )
    return zeros.reindex(columns=merged.columns)


def main() -> int:
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scratch = Path(tempfile.mkdtemp(prefix="sdwis_zero_"))
    downloaded = _discover_downloaded(DATA)
    merged = pd.read_parquet(DATA / "interim" / "merged_wq.parquet")
    sdwis_mask = merged["source"] == "sdwis"
    sdwis_canon = merged[sdwis_mask].copy()

    # --- DROP mode (= published): canonical >0 set + M1 detection_limit=NaN ---
    drop_merged = merged.copy()
    drop_merged.loc[sdwis_mask, "detection_limit"] = np.nan
    logger.info("DROP merged: %d rows (%d sdwis)", len(drop_merged), int(sdwis_mask.sum()))
    drop_metrics = _t4_metrics(drop_merged, downloaded, ["xgboost"], scratch / "drop")

    frozen = {
        (r["task"], r["model"]): r["metrics"]
        for r in json.loads((FROZEN / "results.json").read_text())
    }
    pub = frozen.get(("T4", "xgboost_classifier"), {})
    dm = drop_metrics.get("xgboost_classifier", {})
    env_dev = (
        abs((dm.get("auroc") or 0) - (pub.get("auroc") or 0))
        if dm.get("auroc") is not None and pub.get("auroc") is not None
        else float("nan")
    )
    env_ok = env_dev == env_dev and env_dev <= ENV_TOL
    report: dict[str, Any] = {
        "env_sanity": {
            "drop_t4_xgboost": dm,
            "published_t4_xgboost": pub,
            "auroc_dev": env_dev,
            "tolerance": ENV_TOL,
            "reproduced": bool(env_ok),
        }
    }
    if not env_ok:
        report["status"] = "STOP: drop-mode did not reproduce published T4 (env-confounded)"
        (scratch / "report.json").write_text(json.dumps(report, indent=2, default=str))
        logger.warning(
            "ENV-SANITY FAILED (dev=%.4f > %.4f). %s", env_dev, ENV_TOL, scratch / "report.json"
        )
        print(json.dumps(report, indent=2, default=str))
        return 1

    # --- KEEP mode: add the zero rows back ---
    keep_extra = _build_keep_extra(merged, sdwis_canon)
    keep_merged = pd.concat([drop_merged, keep_extra], ignore_index=True)
    logger.info("KEEP merged: %d rows (+%d zero rows)", len(keep_merged), len(keep_extra))
    keep_metrics = _t4_metrics(keep_merged, downloaded, ["xgboost", "catboost"], scratch / "keep")

    report["keep_vs_drop"] = {
        "drop": drop_metrics,
        "keep": keep_metrics,
        "delta_xgboost_auroc": (
            (keep_metrics.get("xgboost_classifier", {}).get("auroc") or 0)
            - (drop_metrics.get("xgboost_classifier", {}).get("auroc") or 0)
        ),
        "delta_xgboost_auprc": (
            (keep_metrics.get("xgboost_classifier", {}).get("auprc") or 0)
            - (drop_metrics.get("xgboost_classifier", {}).get("auprc") or 0)
        ),
    }
    report["zero_only_systems_lead"] = 5572
    report["status"] = "OK"
    (scratch / "report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report: %s", scratch / "report.json")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
