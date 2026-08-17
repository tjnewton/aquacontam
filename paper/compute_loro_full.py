#!/usr/bin/env python
"""C18 (R5 M3 + m5): full 20-family Leave-One-Region-Out leaderboard for T1.

The R5 referee flagged that the 20-family benchmark leaderboard rests on a single
fixed West-only geographic split, and that the fixed-split leader (XGBoost) is not
the LORO leader (CatBoost 0.7896 > XGBoost 0.7825) — yet only the three tree models
ever receive a cross-split evaluation. This maintainer-authorized COMPUTE (approved
2026-07-05, decision R5-D3) runs LORO for **every** classifier family on T1 so the
leaderboard can be re-ranked by geographic transfer, and it produces the previously
absent out-of-region ICP number (m5) because ICP is one of the families.

Design (mirrors ``paper/compute_loro_lift.py`` / ``_run_loro_cv`` exactly: same
merged_wq, same assembly, same ``leave_one_region_out`` splitter, same default
configs, seed 42):

  1. Reproduction gate — re-run xgboost / random_forest / catboost and require each
     to reproduce the committed ``loro_cv.json`` T1 values (per-fold AUROC for the
     deterministic xgboost anchor within XGB_TOL; mean AUROC for all three within
     MEAN_TOL). Nothing is trusted until the harness reproduces the published folds.
  2. Run every remaining classifier family through the same folds. Per-family and
     per-fold failures are caught and recorded (status), never fatal — a wedged deep
     model or a cap-exceeding TabPFN is a documented non-convergence, not a crash.
  3. Write a **staging** file ``results/loro_cv_full_staged.json`` (NOT the frozen
     archive) plus a completion sentinel. The interactive splice into the frozen
     ``loro_cv.json`` (3 tree entries + T4 kept byte-identical, new families added)
     and ``freeze_results.py --only`` happen after the sentinel lands and the gate
     result is inspected — so a multi-hour detached crash can never leave a
     half-written frozen file.

Launch DETACHED (PowerShell ``Start-Process -WindowStyle Hidden``) with stdout to
``results/loro_full.log`` and ``--gpu-id 2`` (RTX 6000); a plain background shell job
dies with the session.
"""

from __future__ import annotations

import os as _os
import sys as _sys


def _early_gpu_env_setup() -> None:
    """Bind CUDA to the requested GPU before any heavy/torch import (see CLAUDE.md)."""
    gpu_id: str | None = None
    argv = _sys.argv
    for i, tok in enumerate(argv):
        if tok == "--gpu-id" and i + 1 < len(argv):
            gpu_id = argv[i + 1]
            break
        if tok.startswith("--gpu-id="):
            gpu_id = tok.split("=", 1)[1]
            break
    if gpu_id is not None:
        _os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        _os.environ["AQUACONTAM_GPU_ID"] = "0"  # always 0 after the CVD remap


_early_gpu_env_setup()

import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "results" / "paper_frozen"
STAGING = REPO / "results" / "loro_cv_full_staged.json"
SENTINEL = REPO / "results" / "loro_full_complete.sentinel"
#: This driver's own status file. It deliberately does NOT write paper/PASS_STATE.json
#: (the interactive session is that file's sole writer during the multi-hour detached run,
#: so there is no concurrent read-modify-write race); the parent merges this in on resume.
STATUS = REPO / "results" / "loro_full_status.json"

XGB_TOL = 2e-3  # deterministic per-fold anchor
MEAN_TOL = 5e-3  # tree-model mean reproduction (catboost GPU may drift slightly)
SEED = 42

#: Classifier families to evaluate under LORO (the T1 leaderboard), each mapped to its
#: (experiment.yaml config key, "module:ClassName"). "dummy"/"xgboost_default" are
#: excluded (baseline / duplicate-config); regression-only families (hurdle, xgboost_aft,
#: zi_tobit) are T2, not T1. Models are constructed from the RAW yaml config (+ seed) —
#: NOT via ``get_model_instances``, whose ``_cfg`` strips ``scale_pos_weight="auto"`` and
#: so would train the tree models unbalanced and fail to reproduce the frozen loro_cv.json
#: (which was built by ``_run_loro_cv`` passing the config through verbatim).
CLASSIFIER_SPECS: dict[str, tuple[str, str]] = {
    "logistic_regression": (
        "logistic_regression",
        "aquacontam.models.logistic:LogisticRegressionClassifier",
    ),
    "xgboost": ("xgboost_classifier", "aquacontam.models.xgboost:XGBoostClassifier"),
    "random_forest": (
        "random_forest_classifier",
        "aquacontam.models.random_forest:RandomForestClassifier",
    ),
    "lightgbm": ("lightgbm_classifier", "aquacontam.models.lightgbm:LightGBMClassifier"),
    "catboost": ("catboost_classifier", "aquacontam.models.catboost:CatBoostClassifier"),
    "mlp": ("mlp_classifier", "aquacontam.models.mlp:MLPClassifier"),
    "cnn1d": ("cnn1d_classifier", "aquacontam.models.cnn1d:CNN1DClassifier"),
    "gnn_gcn": ("gnn_gcn_classifier", "aquacontam.models.gnn:GNNClassifier"),
    "gnn_sage": ("gnn_sage_classifier", "aquacontam.models.gnn:GNNClassifier"),
    "deep_tobit": ("deep_tobit_classifier", "aquacontam.models.deep_tobit:DeepTobitClassifier"),
    "tabpfn": ("tabpfn_classifier", "aquacontam.models.tabpfn:TabPFNClassifier"),
    "voting_ensemble": (
        "voting_ensemble_classifier",
        "aquacontam.models.ensemble:VotingEnsembleClassifier",
    ),
    "stacking_ensemble": (
        "stacking_ensemble_classifier",
        "aquacontam.models.ensemble:StackingEnsembleClassifier",
    ),
    "icp": ("icp_classifier", "aquacontam.models.icp:ICPClassifier"),
}
FAMILIES = list(CLASSIFIER_SPECS)
#: The reproduction-gate anchors (must reproduce committed loro_cv.json).
GATE_FAMILIES = ["xgboost", "random_forest", "catboost"]


def _make_model(name: str, yaml_configs: dict):
    """Fresh model from the RAW yaml config (+ seed) — mirrors ``_run_loro_cv`` exactly."""
    import importlib

    key, path = CLASSIFIER_SPECS[name]
    module_name, cls_name = path.split(":")
    model_cls = getattr(importlib.import_module(module_name), cls_name)
    cfg = dict(yaml_configs.get(key, {}))
    cfg["random_state"] = SEED
    return model_cls(config=cfg)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compute_loro_full")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_STATUS_STATE: dict[str, object] = {}


def _write_status(**kw: object) -> None:
    """Write this driver's own status file (sole owner — no PASS_STATE race)."""
    _STATUS_STATE.update(kw)
    _STATUS_STATE["updated"] = _now()
    STATUS.write_text(json.dumps(_STATUS_STATE, indent=2, default=str) + "\n", encoding="utf-8")


def _write_sentinel(status: str, detail: str) -> None:
    SENTINEL.write_text(f"{status} {_now()} {detail}\n", encoding="utf-8")


def _fold_metrics(y_true, preds, probs) -> dict:
    from aquacontam.benchmark.metrics import compute_classification_metrics

    return compute_classification_metrics(y_true, preds, probs, metrics=["auroc", "auprc", "f1"])


def main() -> int:
    import numpy as np
    import pandas as pd

    import aquacontam.benchmark  # noqa: F401  (register tasks)
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
    )
    from aquacontam.pipeline.assembly import assemble_with_split_imputation
    from aquacontam.pipeline.features import extract_features
    from aquacontam.pipeline.models import load_model_configs
    from aquacontam.preprocessing.splits import leave_one_region_out

    yaml_configs = load_model_configs()

    _sys.path.insert(0, str(REPO / "paper"))
    from compute_areal_apportionment import _discover_downloaded

    _write_status(status="RUNNING", started=_now(), pid=_os.getpid())
    logger.info(
        "C18 full 20-family LORO — pid=%d gpu_env CVD=%s",
        _os.getpid(),
        _os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    # --- data load (identical to compute_loro_lift.py) ---
    data_path = REPO / "data"
    wq_df = pd.read_parquet(data_path / "interim" / "merged_wq.parquet")
    feature_dfs = extract_features(
        data_path, wq_df, _discover_downloaded(data_path), skip_large=False
    )

    # --- T1 full-feature assembly (NOT provenance-free — matches loro_cv.json) ---
    sys_targets = aggregate_to_system_level(wq_df, "PFOS", target="detected")
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)
    logger.info("T1 assembly: %d systems, %d features", *X.shape)
    loro_df = pd.DataFrame({"epa_region": regions}, index=X.index)
    folds = list(leave_one_region_out(loro_df, seed=SEED))

    frozen = json.loads((FROZEN / "loro_cv.json").read_text(encoding="utf-8"))["T1"]

    def run_family(name: str) -> dict:
        """Run one family through all LORO folds; return an entry dict with status."""
        fold_results: list[dict] = []
        for train_df, val_df, test_df, test_region in folds:
            tr = train_df.index.intersection(X.index)
            va = val_df.index.intersection(X.index)
            te = test_df.index.intersection(X.index)
            if len(tr) < 10 or len(te) < 10:
                continue
            try:
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
                logger.info(
                    "  %s fold %d: AUROC=%.4f", name, test_region, m.get("auroc", float("nan"))
                )
            except Exception as exc:  # one bad fold/family must never kill the whole run
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

    # --- reproduction gate on the three tree anchors ---
    results: dict[str, dict] = {}
    gate_fails: list[str] = []
    for name in GATE_FAMILIES:
        logger.info("Reproduction-gate family: %s", name)
        entry = run_family(name)
        results[name] = entry
        fz = frozen.get(name)
        if fz is None:
            gate_fails.append(f"{name}: absent from frozen loro_cv.json")
            continue
        if abs(entry["mean_auroc"] - float(fz["mean_auroc"])) > MEAN_TOL:
            gate_fails.append(
                f"{name}: mean AUROC {entry['mean_auroc']:.4f} vs frozen {fz['mean_auroc']:.4f}"
            )
        if name == "xgboost":  # deterministic per-fold anchor
            fz_folds = {int(f["test_region"]): f for f in fz["folds"]}
            for fr in entry["folds"]:
                z = fz_folds.get(fr["test_region"])
                if z and abs(fr["auroc"] - float(z["auroc"])) > XGB_TOL:
                    gate_fails.append(
                        f"xgboost fold {fr['test_region']}: {fr['auroc']:.4f} vs {z['auroc']:.4f}"
                    )

    if gate_fails:
        logger.error("REPRODUCTION GATE FAILED (%d): %s", len(gate_fails), gate_fails[:5])
        STAGING.write_text(
            json.dumps({"_gate_failed": gate_fails, "T1": results}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        _write_status(status="GATE_FAILED", gate_fails=gate_fails[:5], finished=_now())
        _write_sentinel("GATE_FAILED", f"{len(gate_fails)} anchor mismatches")
        return 1
    logger.info("Reproduction gate OK: all %d tree anchors reproduce frozen", len(GATE_FAMILIES))

    # --- remaining families ---
    for name in FAMILIES:
        if name in results:
            continue
        logger.info("Family: %s", name)
        results[name] = run_family(name)
        _write_status(status="RUNNING", families_done=len(results))

    ranked = sorted(
        ((n, e["mean_auroc"]) for n, e in results.items() if e["status"] == "ok"),
        key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1),  # NaN-safe
    )
    payload = {
        "_meta": {
            "source": (
                "C18 (R5 M3+m5) full-family T1 LORO — folds mirrored from _run_loro_cv "
                f"(merged_wq, default configs, seed {SEED}); tree anchors reproduced frozen "
                f"loro_cv.json within mean {MEAN_TOL}/xgb-fold {XGB_TOL}. Deep-model folds are "
                "single-run (GPU non-determinism); staging file — splice into frozen after review."
            ),
            "seed": SEED,
            "reproduction_gate": "PASSED",
            "families_ranked_by_loro_mean": ranked,
            "generated": _now(),
        },
        "T1": results,
    }
    STAGING.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Wrote %s — LORO leaderboard (top): %s", STAGING, ranked[:5])
    _write_status(status="COMPLETE", finished=_now(), ranked_top=ranked[:5])
    _write_sentinel("COMPLETE", f"{len(results)} families, top={ranked[0] if ranked else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
