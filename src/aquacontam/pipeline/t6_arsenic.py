"""T6 driver: public-supply → domestic arsenic transfer (USGS NGA).

Self-contained, reproducible runner for benchmark task T6. Loads measured
arsenic from the USGS National Groundwater Aggregation (CC0; see
``data/nga_arsenic.py``), assembles the project's environmental features at the
well locations, and trains each model on **public-supply** wells (geographic
split) to evaluate **zero-shot** on independent **domestic** wells.

Runs every model under both ``geographic`` (honest) and ``random`` (leaky
ablation) splits so the result captures the task's two headline findings:

* **Geographic-leakage inflation** — random-split AUROC minus geographic-split
  AUROC on the in-distribution public-supply test set.
* **Public→private transfer** — the zero-shot domestic-holdout metric, which on
  geogenic arsenic is *not* the limiting factor (it matches or exceeds the
  in-distribution geographic holdout).

Writes ``results/t6_arsenic.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.benchmark.private_wells import run_t6
from aquacontam.data.nga_arsenic import (
    download_nga_arsenic,
    load_nga_arsenic,
    nga_metadata,
)
from aquacontam.pipeline.features import _FEATURE_SPECS, extract_features
from aquacontam.pipeline.models import get_model_instances

logger = logging.getLogger(__name__)

# Tabular models suited to this task (deep/graph/foundation models are excluded:
# GNN needs a constructed graph, TabPFN caps at 10k train rows < public-supply N,
# and the deep nets are ill-suited to ~14k tabular rows). Curated set is noted in
# the paper.
T6_MODEL_FILTER: list[str] = [
    "dummy",
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
]


def _downloaded_aux(data_dir: Path) -> dict[str, Path]:
    """Locate the auxiliary feature sources for environmental feature extraction."""
    interim = data_dir / "interim"
    raw = data_dir / "raw"
    candidates: dict[str, Path] = {
        "epa_frs": interim / "epa_frs.parquet",
        "nlcd": raw / "Annual_NLCD_LndCov_2021_CU_C1V1.tif",
        "usgs_aquifers": raw / "us_aquifers.shp",
        "ejscreen": interim / "ejscreen.parquet",
        "tri_pfas": interim / "tri_pfas.parquet",
        "dod_pfas": interim / "dod_pfas.parquet",
    }
    present = {k: p for k, p in candidates.items() if p.exists()}
    missing = sorted(set(candidates) - set(present))
    if missing:
        logger.warning("NGA feature aux sources missing (features will be partial): %s", missing)
    return present


def build_nga_feature_dfs(data_dir: Path, all_wells: pd.DataFrame) -> list[pd.DataFrame]:
    """Assemble environmental + well-metadata features for NGA wells.

    Uses the project's environmental extractors (proximity, land use,
    hydrogeology, demographics, TRI, DoD) cached under ``features_nga`` — the
    ``system characteristics`` extractor is excluded (it is SDWIS-specific and
    inapplicable to wells). Adds well depth and the USGS aquifer assignment,
    the canonical geogenic-arsenic predictors, available equally to both
    populations (no leakage).
    """
    env_specs = [s for s in _FEATURE_SPECS if s.name != "system characteristics"]
    downloaded = _downloaded_aux(data_dir)
    feature_dfs = list(
        extract_features(
            data_dir,
            all_wells,
            downloaded,
            cache_subdir="features_nga",
            specs=env_specs,
        )
    )
    site_attrs = all_wells.groupby("pwsid").agg(
        well_depth_ft=("well_depth_ft", "median"),
        aquifer=("aquifer", "first"),
    )
    site_attrs["log_well_depth"] = np.log1p(site_attrs["well_depth_ft"].clip(lower=0))
    feature_dfs.append(site_attrs)

    # Sanitize categorical *values* so one-hot column names are free of special
    # characters (USGS aquifer names contain commas/parentheses that LightGBM
    # rejects as feature names). Values only — does not affect numeric features.
    for fdf in feature_dfs:
        for col in fdf.select_dtypes(include=["object", "string"]).columns:
            fdf[col] = (
                fdf[col].astype(str).str.replace(r"[^0-9A-Za-z]+", "_", regex=True).str.strip("_")
            )
    return feature_dfs


def run_t6_experiment(
    data_dir: Path,
    *,
    model_filter: list[str] | None = None,
    seed: int = 42,
    out_path: Path | None = None,
    download: bool = True,
    use_tuned_params: bool = False,
    tuned_params_path: str | None = None,
) -> dict[str, Any]:
    """Run the full T6 arsenic transfer experiment and write results JSON.

    Parameters
    ----------
    data_dir : Path
        Root data directory (expects ``raw/nga_arsenic/`` CSVs or downloads them).
    model_filter : list[str] | None
        Model keys to run. Defaults to :data:`T6_MODEL_FILTER`.
    seed : int
        Random seed for models and the random-split ablation.
    out_path : Path | None
        Output JSON path. Defaults to ``<data_dir>/../results/t6_arsenic.json``.
    download : bool
        Download the NGA CSVs if absent.
    use_tuned_params : bool
        Load each model's best config from the Optuna sweep
        (``results/optuna_tuning.json``, ``(model, "T6")``), falling back to YAML
        defaults where no tuned config exists. Default ``False`` (YAML defaults).
    tuned_params_path : str | None
        Override path to the tuned-config JSON.

    Returns
    -------
    dict
        The results structure (also written to ``out_path``).
    """
    raw_dir = data_dir / "raw"
    if download:
        try:
            download_nga_arsenic(raw_dir)
        except RuntimeError as exc:
            logger.warning("NGA download skipped/failed: %s", exc)

    pub = load_nga_arsenic(raw_dir, water_use="Public supply")
    dom = load_nga_arsenic(raw_dir, water_use="Domestic")
    if pub.empty or dom.empty:
        raise RuntimeError(
            "NGA arsenic data unavailable — cannot run T6. Place Site_Information.csv "
            "and Water_Chemistry.csv in data/raw/nga_arsenic/ (DOI 10.5066/P9JMUAPY)."
        )
    all_wells = pd.concat([pub, dom], ignore_index=True)
    feature_dfs = build_nga_feature_dfs(data_dir, all_wells)

    def _exceed_rate(df: pd.DataFrame) -> float:
        per_site = df.groupby("pwsid")["concentration"].max()
        return float((per_site >= 10.0).mean())

    results: dict[str, Any] = {
        "task": "T6",
        "description": "Private-well risk extrapolation: public-supply → domestic arsenic",
        "analyte": "arsenic",
        "target": "action_level (arsenic MCL ≥ 10 µg/L)",
        "source": nga_metadata(),
        "seed": seed,
        "use_tuned_params": use_tuned_params,
        "n_public_wells": int(pub["pwsid"].nunique()),
        "n_domestic_wells": int(dom["pwsid"].nunique()),
        "public_exceedance_rate": _exceed_rate(pub),
        "domestic_exceedance_rate": _exceed_rate(dom),
        "models": {},
    }

    filt = model_filter if model_filter is not None else T6_MODEL_FILTER
    for strategy in ("geographic", "random"):
        models = get_model_instances(
            filt,
            "classification",
            seed,
            task_name="T6",
            use_tuned_params=use_tuned_params,
            tuned_params_path=tuned_params_path,
        )
        for model in models:
            res = run_t6(
                model=model,
                data=pub,
                well_data=dom,
                feature_dfs=feature_dfs,
                split_strategy=strategy,
            )
            entry = results["models"].setdefault(model.name, {})
            entry[strategy] = {
                "public_test": res.split_metrics.get("test", {}),
                "domestic_holdout": res.split_metrics.get("holdout", {}),
            }
            logger.info(
                "T6 %s/%s: public-test AUROC=%.4f domestic AUROC=%.4f",
                model.name,
                strategy,
                res.split_metrics.get("test", {}).get("auroc", float("nan")),
                res.split_metrics.get("holdout", {}).get("auroc", float("nan")),
            )

    # Derived headline metrics per model.
    for entry in results["models"].values():
        geo, rnd = entry.get("geographic", {}), entry.get("random", {})
        geo_test = geo.get("public_test", {}).get("auroc")
        rnd_test = rnd.get("public_test", {}).get("auroc")
        geo_test_pr = geo.get("public_test", {}).get("auprc")
        rnd_test_pr = rnd.get("public_test", {}).get("auprc")
        geo_dom = geo.get("domestic_holdout", {}).get("auroc")
        if geo_test is not None and rnd_test is not None:
            entry["leakage_inflation_auroc"] = round(rnd_test - geo_test, 4)
        if geo_test_pr is not None and rnd_test_pr is not None:
            entry["leakage_inflation_auprc"] = round(rnd_test_pr - geo_test_pr, 4)
        if geo_dom is not None and geo_test is not None:
            entry["transfer_minus_indist_auroc"] = round(geo_dom - geo_test, 4)

    if out_path is None:
        out_path = data_dir.parent / "results" / "t6_arsenic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote T6 results to %s", out_path)
    return results
