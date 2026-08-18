"""Complete the T1 LORO leaderboard for the two graph nets and TabPFN.

The C18 driver (``compute_loro_full.py``) recorded ``gnn_gcn``/``gnn_sage`` as
"failed" because it never passed the ``coords`` kwarg the GNN models require, and
recorded ``tabpfn`` as 2/10 folds because 8 folds exceeded the wrapper's
``max_train_size`` guard. Both are harness gaps, not model limits. This driver:

* threads EPSG:5070 ``coords``/``coords_val`` into each GNN fold's ``fit`` (mirroring
  ``benchmark/_prep_data._add_gnn_coords``), so the graph nets run all 10 folds
  (test-time inference is the documented linear fallback, so scores land near the
  fixed-split fallback values -- the honest result);
* runs TabPFN on all 10 folds with a driver-local ``max_train_size=20000`` override
  while keeping ``cpu_subsample_size=3000`` (identical 3,000-sample effective protocol
  to the 2 folds that already ran and to the fixed split);
* runs **each family in its own process** (``--family`` invocation) so ``gnn_gcn`` and
  ``gnn_sage`` never share a CUDA context (in-process sharing corrupted the context in
  the June HPO incident);
* is **reproduction-gated** (``--gate``): re-runs the three frozen tree anchors and
  requires they reproduce ``results/paper_frozen/loro_cv.json`` before any GNN/TabPFN
  result is trusted for the splice. NEVER overwrites the 12 converged families.

Orchestration (each step a separate process)::

    python paper/compute_loro_gnn_tabpfn.py --prep          # build + cache X/y/regions/coords
    python paper/compute_loro_gnn_tabpfn.py --gate          # reproduction gate on tree anchors
    python paper/compute_loro_gnn_tabpfn.py --family gnn_gcn
    python paper/compute_loro_gnn_tabpfn.py --family gnn_sage
    python paper/compute_loro_gnn_tabpfn.py --family tabpfn

Per-family results land in ``results/loro_extra_<family>.json``; the Phase-2 splice
script consumes them + the existing frozen ``loro_cv_full.json``.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# Bind CUDA to a stable device before any torch import (env-var GPU binding).
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import json
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
PREP = REPO / "results" / "loro_prep"
SEED = 42
# Reproduction-gate tolerances (match compute_loro_full.py C18).
XGB_TOL = 2e-3
MEAN_TOL = 5e-3
GATE_FAMILIES = ["xgboost", "random_forest", "catboost"]

#: family -> (experiment.yaml config key, "module:ClassName")
SPECS: dict[str, tuple[str, str]] = {
    "xgboost": ("xgboost_classifier", "aquacontam.models.xgboost:XGBoostClassifier"),
    "random_forest": (
        "random_forest_classifier",
        "aquacontam.models.random_forest:RandomForestClassifier",
    ),
    "catboost": ("catboost_classifier", "aquacontam.models.catboost:CatBoostClassifier"),
    "gnn_gcn": ("gnn_gcn_classifier", "aquacontam.models.gnn:GNNClassifier"),
    "gnn_sage": ("gnn_sage_classifier", "aquacontam.models.gnn:GNNClassifier"),
    "tabpfn": ("tabpfn_classifier", "aquacontam.models.tabpfn:TabPFNClassifier"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("loro_gnn_tabpfn")


def _make_model(name: str, yaml_configs: dict):
    """Fresh model from the RAW yaml config (+ seed) -- mirrors compute_loro_full._make_model.

    TabPFN gets a driver-local override so the wrapper does not reject the oversized
    folds; the 3,000-sample CPU subsample is unchanged, so the effective training protocol
    matches the 2 folds that already ran and the fixed-split result.
    """
    import importlib

    key, path = SPECS[name]
    module_name, cls_name = path.split(":")
    model_cls = getattr(importlib.import_module(module_name), cls_name)
    cfg = dict(yaml_configs.get(key, {}))
    cfg["random_state"] = SEED
    if name == "tabpfn":
        cfg["max_train_size"] = 20000  # accept the folds; CPU subsample below still applies
        cfg.setdefault("cpu_subsample_size", 3000)
        cfg.setdefault("device", "cpu")
    return model_cls(config=cfg)


def _fold_metrics(y_true, preds, probs) -> dict:
    from aquacontam.benchmark.metrics import compute_classification_metrics

    return compute_classification_metrics(y_true, preds, probs, metrics=["auroc", "auprc", "f1"])


def _build_and_cache() -> None:
    """Assemble X/y/regions once + project system coords to EPSG:5070; cache to results/loro_prep."""
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.benchmark._utils import get_system_coordinates
    from aquacontam.features.assembly import aggregate_to_system_level, drop_leakage_columns
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features

    _sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )
    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    logger.info("T1 assembly: %d systems, %d features", *X.shape)

    # Project system coordinates to EPSG:5070 (mirror _add_gnn_coords), aligned to X.index.
    sys_coords = get_system_coordinates(wq_df)
    from pyproj import Transformer

    proj = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    cx, cy = proj.transform(sys_coords["longitude"].values, sys_coords["latitude"].values)
    coords = pd.DataFrame({"x": cx, "y": cy}, index=sys_coords.index)
    coords = coords.reindex(X.index)  # NaN rows = systems without coords
    n_cov = int(coords["x"].notna().sum())
    logger.info("coords coverage: %d / %d systems (%.1f%%)", n_cov, len(X), 100 * n_cov / len(X))

    PREP.mkdir(parents=True, exist_ok=True)
    X.to_parquet(PREP / "X.parquet")
    pd.DataFrame({"y": y, "epa_region": regions}, index=X.index).to_parquet(PREP / "meta.parquet")
    coords.to_parquet(PREP / "coords.parquet")
    logger.info("cached prep -> %s", PREP)


def _load_cache():
    import pandas as pd

    from aquacontam.preprocessing.splits import leave_one_region_out

    X = pd.read_parquet(PREP / "X.parquet")
    meta = pd.read_parquet(PREP / "meta.parquet")
    coords = pd.read_parquet(PREP / "coords.parquet")
    y = meta["y"]
    loro_df = pd.DataFrame({"epa_region": meta["epa_region"]}, index=X.index)
    folds = list(leave_one_region_out(loro_df, seed=SEED))
    return X, y, coords, folds


def _run_family(name: str) -> dict:
    """Run one family through all LORO folds; GNN folds thread EPSG:5070 coords."""
    import numpy as np

    from aquacontam.pipeline.models import load_model_configs

    yaml_configs = load_model_configs()
    X, y, coords, folds = _load_cache()
    is_gnn = name.startswith("gnn")
    fold_results: list[dict] = []
    for train_df, val_df, test_df, test_region in folds:
        tr = train_df.index.intersection(X.index)
        va = val_df.index.intersection(X.index)
        te = test_df.index.intersection(X.index)
        if len(tr) < 10 or len(te) < 10:
            continue
        try:
            model = _make_model(name, yaml_configs)
            fit_kw: dict = {"X_val": X.loc[va], "y_val": y.loc[va]}
            if is_gnn:
                # Restrict train/val to coord-covered systems (>=50% coverage rule); pass 5070 coords.
                tr_cov = coords.loc[tr].dropna().index
                va_cov = coords.loc[va].dropna().index
                if len(tr_cov) < 0.5 * len(tr):
                    raise ValueError(f"coord coverage <50% for fold {test_region}")
                tr, va = tr_cov, va_cov
                fit_kw = {
                    "X_val": X.loc[va],
                    "y_val": y.loc[va],
                    "coords": coords.loc[tr].to_numpy(),
                    "coords_val": coords.loc[va].to_numpy() if len(va) else None,
                }
            try:
                model.fit(X.loc[tr], y.loc[tr], **fit_kw)
            except TypeError:
                # models that don't accept X_val/coords kwargs
                model.fit(
                    X.loc[tr],
                    y.loc[tr],
                    **{k: v for k, v in fit_kw.items() if k.startswith("coords")},
                )
            preds = model.predict(X.loc[te])
            probs = model.predict_proba(X.loc[te])
            if getattr(probs, "ndim", 1) == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]
            m = _fold_metrics(y.loc[te], preds, probs)
            fold_results.append({"test_region": int(test_region), "n_test": len(te), **m})
            logger.info(
                "  %s fold %d: AUROC=%.4f", name, test_region, m.get("auroc", float("nan"))
            )
        except Exception as exc:
            logger.warning("  %s fold %d FAILED: %s", name, test_region, exc)
    if not fold_results:
        return {"status": "failed", "n_folds": 0, "mean_auroc": float("nan"), "folds": []}
    aurocs = [f["auroc"] for f in fold_results if not np.isnan(f.get("auroc", float("nan")))]
    return {
        "status": "ok",
        "n_folds": len(fold_results),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "std_auroc": float(np.std(aurocs)) if aurocs else float("nan"),
        "folds": fold_results,
    }


def _gate() -> int:
    """Reproduction gate: re-run the three tree anchors, compare to frozen loro_cv.json (read-only)."""
    from aquacontam.pipeline.models import load_model_configs

    yaml_configs = load_model_configs()
    frozen = json.loads((FROZEN / "loro_cv.json").read_text(encoding="utf-8"))["T1"]
    fails: list[str] = []
    for name in GATE_FAMILIES:
        entry = _run_family_trees(name, yaml_configs)
        fz = frozen.get(name)
        if fz is None:
            fails.append(f"{name}: absent from frozen loro_cv.json")
            continue
        if abs(entry["mean_auroc"] - float(fz["mean_auroc"])) > MEAN_TOL:
            fails.append(
                f"{name}: mean {entry['mean_auroc']:.4f} vs frozen {fz['mean_auroc']:.4f}"
            )
        if name == "xgboost":
            fzf = {int(f["test_region"]): f for f in fz["folds"]}
            for fr in entry["folds"]:
                z = fzf.get(fr["test_region"])
                if z and abs(fr["auroc"] - float(z["auroc"])) > XGB_TOL:
                    fails.append(
                        f"xgboost fold {fr['test_region']}: {fr['auroc']:.4f} vs {z['auroc']:.4f}"
                    )
    if fails:
        logger.error("REPRODUCTION GATE FAILED (%d): %s", len(fails), fails[:6])
        return 1
    logger.info("Reproduction gate OK: all %d tree anchors reproduce frozen", len(GATE_FAMILIES))
    return 0


def _run_family_trees(name: str, yaml_configs: dict) -> dict:
    """Tree-family LORO run for the gate (no coords)."""
    import numpy as np

    X, y, _coords, folds = _load_cache()
    fold_results: list[dict] = []
    for train_df, val_df, test_df, test_region in folds:
        tr = train_df.index.intersection(X.index)
        va = val_df.index.intersection(X.index)
        te = test_df.index.intersection(X.index)
        if len(tr) < 10 or len(te) < 10:
            continue
        model = _make_model(name, yaml_configs)
        try:
            model.fit(X.loc[tr], y.loc[tr], X_val=X.loc[va], y_val=y.loc[va])
        except TypeError:
            model.fit(X.loc[tr], y.loc[tr])
        preds = model.predict(X.loc[te])
        probs = model.predict_proba(X.loc[te])
        if getattr(probs, "ndim", 1) == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        m = _fold_metrics(y.loc[te], preds, probs)
        fold_results.append({"test_region": int(test_region), "n_test": len(te), **m})
    aurocs = [f["auroc"] for f in fold_results]
    return {"mean_auroc": float(np.mean(aurocs)), "folds": fold_results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--family", choices=["gnn_gcn", "gnn_sage", "tabpfn"])
    args = ap.parse_args()

    if args.prep:
        _build_and_cache()
        return 0
    if args.gate:
        return _gate()
    if args.family:
        entry = _run_family(args.family)
        out = REPO / "results" / f"loro_extra_{args.family}.json"
        out.write_text(json.dumps(entry, indent=2, default=str) + "\n", encoding="utf-8")
        logger.info(
            "Wrote %s (status=%s, n_folds=%d, mean=%.4f)",
            out,
            entry["status"],
            entry["n_folds"],
            entry.get("mean_auroc", float("nan")),
        )
        return 0
    ap.error("one of --prep / --gate / --family required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
