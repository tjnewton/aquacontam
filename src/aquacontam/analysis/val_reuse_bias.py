"""Quantify validation set reuse bias in the holdout-LORO AUROC gap.

Runs matched experiments where only validation handling varies (same test
set, same hyperparameters) to decompose the ~0.085 AUROC gap between the
geographic holdout (XGBoost T1 0.859) and LORO CV (0.774) into:

- **Val reuse effect**: Using external regions (2, 7) for early stopping
  vs no early stopping at all (same training data, same test).
- **Val identity effect**: External val regions vs internal val held out
  from training regions.
- **Early stopping benefit**: Internal val early stopping vs no ES.

AUROC is threshold-independent, so threshold optimization cannot bias it.
Hyperparameters are pre-set in experiment.yaml, not tuned on val.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _make_internal_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Hold out a random fraction of rows as internal validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (same index as *X*).
    fraction : float
        Approximate fraction to hold out (0 < fraction < 1).
    seed : int
        Random seed for reproducible selection.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_val, y_val)`` with disjoint indices.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    n_val = max(1, round(n * fraction))
    val_idx = rng.choice(X.index, size=n_val, replace=False)
    val_mask = X.index.isin(val_idx)

    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]
    X_train = X.loc[~val_mask]
    y_train = y.loc[~val_mask]
    return X_train, y_train, X_val, y_val


def _evaluate_per_region(
    y_true: pd.Series,
    y_prob: np.ndarray,
    regions: pd.Series,
    test_regions: tuple[int, ...],
) -> dict[int, float]:
    """Compute AUROC separately for each test region.

    Parameters
    ----------
    y_true : pd.Series
        True labels for the full test set.
    y_prob : np.ndarray
        Predicted probabilities for the full test set.
    regions : pd.Series
        EPA region for each test system (same index as *y_true*).
    test_regions : tuple[int, ...]
        Region numbers to evaluate.

    Returns
    -------
    dict[int, float]
        ``{region: auroc}``; NaN if region has < 2 classes or < 2 samples.
    """
    result: dict[int, float] = {}
    for r in test_regions:
        mask = regions == r
        y_r = y_true.loc[mask]
        p_r = (
            np.asarray(y_prob)[mask.to_numpy()] if isinstance(y_prob, np.ndarray) else y_prob[mask]
        )
        if len(y_r) < 2 or y_r.nunique() < 2:
            result[r] = float("nan")
        else:
            result[r] = float(roc_auc_score(y_r, p_r))
    return result


def run_val_reuse_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    regions: pd.Series,
    *,
    model_config: dict[str, Any],
    train_regions: tuple[int, ...] = (1, 3, 4, 5, 6),
    val_regions: tuple[int, ...] = (2, 7),
    test_regions: tuple[int, ...] = (8, 9, 10),
    internal_val_fraction: float = 0.15,
    seeds: list[int] | None = None,
    loro_per_region: dict[int, float] | None = None,
    holdout_auroc: float | None = None,
    loro_mean_auroc: float | None = None,
) -> dict[str, Any]:
    """Run matched experiments to quantify validation set reuse bias.

    Three conditions are compared, all with the same test set:

    A. **triple_use** — Train on *train_regions*, early-stop on
       *val_regions* (status quo).
    B. **internal_val** — Train on ~85 % of *train_regions*, early-stop
       on held-out 15 % from within *train_regions*.  *val_regions* data
       excluded entirely.
    C. **no_es** — Train on *train_regions* (no *val_regions*), no early
       stopping (all ``n_estimators`` rounds executed).

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (all regions).
    y : pd.Series
        Full target vector (same index as *X*).
    regions : pd.Series
        EPA region number per system (same index as *X*).
    model_config : dict
        XGBoost classifier config (from ``experiment.yaml``).
    train_regions, val_regions, test_regions : tuple[int, ...]
        EPA region assignments for each split.
    internal_val_fraction : float
        Fraction of training data to hold out as internal val in
        condition B (default 0.15).
    seeds : list[int] or None
        Random seeds for multi-seed stability (default 5 seeds).
    loro_per_region : dict[int, float] or None
        Pre-computed LORO AUROC per test region for comparison.
    holdout_auroc : float or None
        Canonical single-split holdout AUROC (the paper's Table 2 value for
        the same model/task). Used with *loro_mean_auroc* to derive the
        holdout-to-LORO gap; the gap is NaN when either is missing.
    loro_mean_auroc : float or None
        Mean LORO AUROC across all folds for the same model/task.

    Returns
    -------
    dict
        Structured results with per-condition AUROC, decomposition, and
        per-region holdout vs LORO comparison.
    """
    from aquacontam.models.xgboost import XGBoostClassifier

    if seeds is None:
        seeds = [42, 123, 456, 789, 2024]

    # --- Partition data by region ----------------------------------------
    train_mask = regions.isin(train_regions)
    val_mask = regions.isin(val_regions)
    test_mask = regions.isin(test_regions)

    X_train_full = X.loc[train_mask]
    y_train_full = y.loc[train_mask]
    X_val_ext = X.loc[val_mask]
    y_val_ext = y.loc[val_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    regions_test = regions.loc[test_mask]

    if X_test.empty or y_test.nunique() < 2:
        logger.warning("Test set empty or single-class; cannot run val-reuse experiment")
        return {"error": "insufficient test data"}

    # --- Helper: train one condition and return AUROC --------------------
    def _run_condition(
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_vl: pd.DataFrame | None,
        y_vl: pd.Series | None,
        cfg: dict[str, Any],
        seed: int,
    ) -> tuple[float, dict[int, float]]:
        """Train XGBoost, return (overall_auroc, per_region_auroc)."""
        c = dict(cfg)
        c["random_state"] = seed
        model = XGBoostClassifier(config=c)

        fit_kwargs: dict[str, Any] = {}
        if X_vl is not None and not X_vl.empty:
            fit_kwargs["X_val"] = X_vl
            fit_kwargs["y_val"] = y_vl

        model.fit(X_tr, y_tr, **fit_kwargs)

        probs = model.predict_proba(X_test)
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]

        overall = float(roc_auc_score(y_test, probs))
        per_region = _evaluate_per_region(y_test, probs, regions_test, test_regions)
        return overall, per_region

    # --- Condition configs -----------------------------------------------
    cfg_es = dict(model_config)  # with early_stopping_rounds

    cfg_no_es = dict(model_config)
    cfg_no_es.pop("early_stopping_rounds", None)

    # --- Run conditions across seeds -------------------------------------
    results_a: list[tuple[float, dict[int, float]]] = []
    results_b: list[tuple[float, dict[int, float]]] = []
    results_c: list[tuple[float, dict[int, float]]] = []

    for seed in seeds:
        # A: triple_use — ES on external val regions
        a = _run_condition(X_train_full, y_train_full, X_val_ext, y_val_ext, cfg_es, seed)
        results_a.append(a)

        # B: internal_val — ES on held-out subset of training data
        X_tr_sub, y_tr_sub, X_vl_int, y_vl_int = _make_internal_val_split(
            X_train_full, y_train_full, internal_val_fraction, seed
        )
        b = _run_condition(X_tr_sub, y_tr_sub, X_vl_int, y_vl_int, cfg_es, seed)
        results_b.append(b)

        # C: no_es — no early stopping, same training data as A
        c = _run_condition(X_train_full, y_train_full, None, None, cfg_no_es, seed)
        results_c.append(c)

    # --- Aggregate -------------------------------------------------------
    def _summarize(
        results: list[tuple[float, dict[int, float]]],
        label: str,
        description: str,
        n_train: int,
        n_val: int,
    ) -> dict[str, Any]:
        aurocs = [r[0] for r in results]
        per_region_all: dict[int, list[float]] = {r: [] for r in test_regions}
        for _, pr in results:
            for r in test_regions:
                v = pr.get(r, float("nan"))
                if not np.isnan(v):
                    per_region_all[r].append(v)

        per_region_summary: dict[str, dict[str, float]] = {}
        for r in test_regions:
            vals = per_region_all[r]
            per_region_summary[str(r)] = {
                "mean_auroc": float(np.mean(vals)) if vals else float("nan"),
                "std_auroc": float(np.std(vals)) if vals else float("nan"),
            }

        return {
            "label": label,
            "description": description,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": len(X_test),
            "n_seeds": len(seeds),
            "auroc_per_seed": [float(a) for a in aurocs],
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "per_region": per_region_summary,
        }

    n_internal_val = max(1, round(len(X_train_full) * internal_val_fraction))

    conditions = {
        "triple_use": _summarize(
            results_a,
            "triple_use",
            f"Train R{','.join(map(str, train_regions))}; "
            f"val R{','.join(map(str, val_regions))} for ES; "
            f"test R{','.join(map(str, test_regions))}",
            len(X_train_full),
            len(X_val_ext),
        ),
        "internal_val": _summarize(
            results_b,
            "internal_val",
            f"Train ~{100 - round(internal_val_fraction * 100)}% of "
            f"R{','.join(map(str, train_regions))}; "
            f"internal {round(internal_val_fraction * 100)}% val for ES; "
            f"R{','.join(map(str, val_regions))} excluded",
            len(X_train_full) - n_internal_val,
            n_internal_val,
        ),
        "no_es": _summarize(
            results_c,
            "no_es",
            f"Train R{','.join(map(str, train_regions))}; "
            f"no early stopping (all {model_config.get('n_estimators', 500)} rounds); "
            f"test R{','.join(map(str, test_regions))}",
            len(X_train_full),
            0,
        ),
    }

    # --- Decomposition ---------------------------------------------------
    mean_a = conditions["triple_use"]["mean_auroc"]
    mean_b = conditions["internal_val"]["mean_auroc"]
    mean_c = conditions["no_es"]["mean_auroc"]

    val_reuse_effect = mean_a - mean_c  # total effect of ES on R2,7
    val_identity_effect = mean_a - mean_b  # R2,7 vs internal val
    es_benefit = mean_b - mean_c  # ES benefit (internal val)

    # Derive the holdout-to-LORO gap from the supplied estimates rather than
    # hardcoding a manuscript value (a hardcoded 0.145 survived a results
    # regeneration and contradicted the regenerated endpoints).
    if holdout_auroc is not None and loro_mean_auroc is not None:
        holdout_loro_gap = round(float(holdout_auroc) - float(loro_mean_auroc), 4)
    else:
        holdout_loro_gap = float("nan")

    decomposition = {
        "val_reuse_effect": round(val_reuse_effect, 4),
        "val_reuse_effect_description": (
            "AUROC difference: ES on external val regions vs no ES "
            "(same training data, same test). Isolates total effect of "
            "using R2,7 for early stopping."
        ),
        "val_identity_effect": round(val_identity_effect, 4),
        "val_identity_effect_description": (
            "AUROC difference: ES on external R2,7 vs ES on internal val "
            "(~15% training holdout). Positive = external val helps more."
        ),
        "es_benefit": round(es_benefit, 4),
        "es_benefit_description": (
            "AUROC difference: ES on internal val vs no ES. "
            "Early stopping benefit independent of val source."
        ),
        "holdout_loro_gap": holdout_loro_gap,
        "holdout_auroc": holdout_auroc,
        "loro_mean_auroc": loro_mean_auroc,
        "pct_explained_by_val_reuse": (
            round(val_reuse_effect / holdout_loro_gap * 100, 1)
            if holdout_loro_gap == holdout_loro_gap and holdout_loro_gap != 0
            else float("nan")
        ),
    }

    # --- Per-region holdout vs LORO comparison ---------------------------
    per_region_comparison: dict[str, dict[str, float]] = {}
    if loro_per_region:
        # Use triple_use condition A as the holdout proxy
        for r in test_regions:
            holdout_auroc = (
                conditions["triple_use"]["per_region"]
                .get(str(r), {})
                .get("mean_auroc", float("nan"))
            )
            loro_auroc = loro_per_region.get(r, float("nan"))
            per_region_comparison[str(r)] = {
                "holdout_auroc": round(holdout_auroc, 4),
                "loro_auroc": round(loro_auroc, 4),
                "delta": round(holdout_auroc - loro_auroc, 4),
            }

    return {
        "conditions": conditions,
        "decomposition": decomposition,
        "per_region_comparison": per_region_comparison,
        "methodology_note": (
            "AUROC is threshold-independent; only early stopping (not "
            "threshold optimization) can bias AUROC via val set reuse. "
            "Hyperparameters are pre-set in experiment.yaml, not tuned "
            "on the validation set."
        ),
    }
