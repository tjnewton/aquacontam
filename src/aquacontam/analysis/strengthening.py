"""Strengthening analyses requested in peer review.

Three robustness analyses that answer specific referee objections to the
monitoring-confounding manuscript:

- :func:`region_block_bootstrap` - re-estimates the LORO headline confidence
  interval by resampling EPA regions (blocks) with replacement, answering the
  objection that i.i.d. bootstrap intervals understate uncertainty under
  residual spatial autocorrelation.
- :func:`buffered_boundary_loro` - re-runs leave-one-region-out CV after
  excluding training systems within a buffer of the held-out region, showing
  the transportable-signal headline is not driven by region-seam leakage.
- :func:`icp_recoverability_probe` - trains a fresh adversary probe on the
  frozen ICP representation to measure how much monitoring-source information
  remains recoverable, turning the "suppresses recoverable monitoring
  information" claim into a measured number.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def region_block_bootstrap(
    fold_results: list[dict[str, Any]],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Block bootstrap of the LORO mean AUROC, resampling regions.

    Treats each EPA region as an exchangeable block and resamples the held-out
    regions with replacement, recomputing the sample-size-weighted mean AUROC
    each iteration. The resulting interval accounts for between-region variance
    (the spatial-clustering scale) rather than assuming systems are i.i.d.

    Parameters
    ----------
    fold_results : list of dict
        Per-region LORO folds with ``auroc`` and ``n_test`` keys.
    n_boot : int
        Bootstrap iterations.
    seed : int
        RNG seed.
    ci : float
        Central interval mass (default 0.95).

    Returns
    -------
    dict
        Block-bootstrap mean/CI, the naive across-fold (i.i.d.-fold) interval,
        and the ratio of interval widths.
    """
    aurocs = np.array(
        [
            f["auroc"]
            for f in fold_results
            if f.get("auroc") is not None and np.isfinite(f["auroc"])
        ]
    )
    weights = np.array(
        [
            f.get("n_test", 1)
            for f in fold_results
            if f.get("auroc") is not None and np.isfinite(f["auroc"])
        ],
        dtype=float,
    )
    n = len(aurocs)
    if n < 2:
        return {"error": "insufficient_folds", "n_folds": n}

    weighted_mean = float(np.average(aurocs, weights=weights))
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot, dtype=float)
    idx_all = np.arange(n)
    for b in range(n_boot):
        draw = rng.choice(idx_all, size=n, replace=True)
        boot_means[b] = np.average(aurocs[draw], weights=weights[draw])

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    block_lo, block_hi = np.percentile(boot_means, [100 * lo_q, 100 * hi_q])

    # Naive across-fold normal interval (treats folds as i.i.d. observations).
    fold_se = float(np.std(aurocs, ddof=1) / np.sqrt(n))
    z = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else float(_normal_quantile(hi_q))
    naive_lo = float(np.mean(aurocs) - z * fold_se)
    naive_hi = float(np.mean(aurocs) + z * fold_se)

    block_width = float(block_hi - block_lo)
    naive_width = float(naive_hi - naive_lo)
    return {
        "n_folds": n,
        "weighted_mean_auroc": weighted_mean,
        "block_bootstrap": {
            "mean": float(np.mean(boot_means)),
            "ci_lower": float(block_lo),
            "ci_upper": float(block_hi),
            "width": block_width,
        },
        "naive_fold_normal": {
            "mean": float(np.mean(aurocs)),
            "ci_lower": naive_lo,
            "ci_upper": naive_hi,
            "width": naive_width,
            "se": fold_se,
        },
        "width_ratio_block_over_naive": (
            block_width / naive_width if naive_width > 0 else float("nan")
        ),
        "n_boot": n_boot,
        "ci": ci,
    }


def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF via the Acklam approximation."""
    # Coefficients for the rational approximation.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return float(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return float(
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _system_coords(wq_df: pd.DataFrame) -> pd.DataFrame:
    """Return one (latitude, longitude) row per pwsid from sample-level data."""
    sub = wq_df.dropna(subset=["latitude", "longitude"])
    out: pd.DataFrame = sub.groupby("pwsid")[["latitude", "longitude"]].first()
    return out


def buffered_boundary_loro(
    wq_df: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    buffer_km: float = 50.0,
    analyte: str = "PFOS",
    target: str = "detected",
    seed: int = 42,
    exclude_features: list[str] | None = None,
) -> dict[str, Any]:
    """LORO CV excluding training systems within ``buffer_km`` of the test region.

    For each held-out EPA region, training systems whose nearest held-out-region
    system is within ``buffer_km`` (EPSG:5070 distance) are dropped before
    training. If the transportable-signal headline were an artifact of systems
    straddling region seams, buffering would collapse it; a stable mean AUROC
    shows it is not.

    Returns a dict with per-region buffered AUROC and the buffered mean, for
    both the with-provenance and (if ``exclude_features``) provenance-free
    feature sets.
    """
    import geopandas as gpd

    from aquacontam._constants import CRS_STORAGE
    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        drop_leakage_columns,
        resolve_excluded_columns,
    )
    from aquacontam.geo.distance import clear_tree_cache, nearest_distances
    from aquacontam.models.xgboost import XGBoostClassifier
    from aquacontam.pipeline.assembly import assemble_with_split_imputation

    sys_targets = aggregate_to_system_level(wq_df, analyte, target=target)
    sys_targets = drop_leakage_columns(sys_targets)
    # Train-only imputation, aligned with the primary LORO path (the earlier
    # full-dataset assemble_feature_matrix leaked imputation statistics — m9).
    X, y, _, regions = assemble_with_split_imputation(sys_targets, feature_dfs, wq_df=wq_df)

    drop_cols: list[str] = []
    if exclude_features:
        drop_cols = resolve_excluded_columns(X.columns, exclude_features)

    coords = _system_coords(wq_df)
    coords = coords.reindex(X.index).dropna()
    gdf = gpd.GeoDataFrame(
        {"pwsid": coords.index},
        geometry=gpd.points_from_xy(coords["longitude"], coords["latitude"]),
        crs=CRS_STORAGE,
    )
    buffer_m = buffer_km * 1000.0

    # True leave-one-region-out over ALL observed regions (the earlier variant
    # evaluated only regions 8-10 while labeled LORO — m9); early-stopping
    # validation uses regions {2, 7} minus the held-out region.
    all_regions = sorted(int(r) for r in regions.dropna().unique())
    val_regions_base = (2, 7)
    fold_results: list[dict[str, Any]] = []

    for test_region in all_regions:
        fold_val_regions = tuple(r for r in val_regions_base if r != test_region) or (
            val_regions_base[0],
        )
        test_idx = X.index[regions == test_region]
        val_idx = X.index[regions.isin(fold_val_regions)]
        train_pool = X.index[~regions.isin((test_region, *fold_val_regions)) & regions.notna()]
        if len(test_idx) < 10 or len(train_pool) < 10:
            continue

        # Distance from each candidate training system to the nearest held-out
        # region system; drop those inside the buffer.
        test_gdf = gdf[gdf["pwsid"].isin(test_idx)]
        train_gdf = gdf[gdf["pwsid"].isin(train_pool)]
        clear_tree_cache()
        if len(test_gdf) and len(train_gdf):
            dist = nearest_distances(train_gdf, test_gdf)
            keep = dist[dist >= buffer_m].index
            train_idx = train_pool.intersection(keep)
        else:
            train_idx = train_pool
        n_buffered = len(train_pool) - len(train_idx)

        if len(train_idx) < 10:
            continue

        entry: dict[str, Any] = {
            "test_region": int(test_region),
            "n_test": len(test_idx),
            "n_train_full": len(train_pool),
            "n_train_buffered": len(train_idx),
            "n_excluded_by_buffer": int(n_buffered),
        }
        for label, cols_to_drop in (("with_provenance", []), ("provenance_free", drop_cols)):
            if label == "provenance_free" and not drop_cols:
                continue
            Xc = X.drop(columns=cols_to_drop) if cols_to_drop else X
            cfg = {"random_state": seed}
            model = XGBoostClassifier(config=cfg)
            try:
                model.fit(
                    Xc.loc[train_idx],
                    y.loc[train_idx],
                    X_val=Xc.loc[val_idx],
                    y_val=y.loc[val_idx],
                )
                probs = model.predict_proba(Xc.loc[test_idx])
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:, 1]
                preds = model.predict(Xc.loc[test_idx])
                m = compute_classification_metrics(
                    y.loc[test_idx], preds, probs, metrics=["auroc", "auprc"]
                )
                entry[label] = {"auroc": m["auroc"], "auprc": m["auprc"]}
            except (ValueError, RuntimeError, ArithmeticError):
                logger.warning("Buffered LORO fold %d (%s) failed", test_region, label)
                entry[label] = {"auroc": float("nan"), "auprc": float("nan")}
        fold_results.append(entry)

    def _mean(label: str) -> float:
        vals = [
            f[label]["auroc"]
            for f in fold_results
            if label in f and np.isfinite(f[label]["auroc"])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "buffer_km": buffer_km,
        "folds": fold_results,
        "mean_auroc_with_provenance": _mean("with_provenance"),
        "mean_auroc_provenance_free": _mean("provenance_free"),
        "n_test_regions": len(fold_results),
    }


def icp_recoverability_probe(
    wq_df: pd.DataFrame,
    feature_dfs: list[pd.DataFrame],
    *,
    analyte: str = "PFOS",
    target: str = "detected",
    seed: int = 42,
) -> dict[str, Any]:
    """Measure monitoring-source recoverability from the ICP representation.

    Trains the ICP encoder on the T1 training split, extracts its
    representation on the held-out test split, then trains a fresh held-out
    probe (logistic regression and a small MLP) to predict the data-source
    label from that representation. Compares against (i) a majority-class
    baseline and (ii) the same probe trained on the raw features, so the
    "suppresses recoverable monitoring information" claim becomes a measured
    multiclass AUROC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        derive_system_epa_regions,
        drop_leakage_columns,
    )

    try:
        from aquacontam.models.icp import ICPClassifier
    except (ImportError, OSError, RuntimeError):
        return {"error": "icp_unavailable"}

    sys_targets = aggregate_to_system_level(wq_df, analyte, target=target)
    sys_targets = drop_leakage_columns(sys_targets)
    X, y, _ = assemble_feature_matrix(sys_targets, *feature_dfs)
    regions = derive_system_epa_regions(wq_df, X.index)

    if "source" not in wq_df.columns:
        return {"error": "no_source_column"}
    source = wq_df.groupby("pwsid")["source"].first().reindex(X.index).fillna("unknown")

    train_mask = regions.isin((1, 3, 4, 5, 6)).to_numpy()
    val_mask = regions.isin((2, 7)).to_numpy()
    test_mask = regions.isin((8, 9, 10)).to_numpy()
    if test_mask.sum() < 30 or train_mask.sum() < 30:
        return {"error": "insufficient_data"}

    # ICP confounder targets: [log(n_samples), mean_detection_limit].
    conf_cols = [c for c in ("n_samples", "mean_detection_limit") if c in X.columns]
    conf = None
    if "n_samples" in X.columns:
        conf = np.column_stack(
            [
                np.log1p(X["n_samples"].to_numpy()),
                (
                    X["mean_detection_limit"].to_numpy()
                    if "mean_detection_limit" in X.columns
                    else np.zeros(len(X))
                ),
            ]
        )

    # early_stopping off so the gradient-reversal adversary trains the full schedule
    # (otherwise task val loss plateaus during warmup and the GRL never engages).
    icp = ICPClassifier(config={"random_state": seed, "epochs": 100, "early_stopping": False})
    fit_kw: dict[str, Any] = {
        "X_val": X.loc[val_mask],
        "y_val": y.loc[val_mask],
        "groups": regions.fillna(0).astype(int).to_numpy()[train_mask],
    }
    if conf is not None:
        fit_kw["confounder_targets"] = conf[train_mask]
        fit_kw["confounder_targets_val"] = conf[val_mask]
    try:
        icp.fit(X.loc[train_mask], y.loc[train_mask], **fit_kw)
        z_train = icp.transform(X.loc[train_mask])
        z_test = icp.transform(X.loc[test_mask])
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        return {"error": f"icp_fit_failed: {exc}"}

    src_codes, src_uniques = pd.factorize(source)
    # Keep only source classes present in both train and test with >=2 members.
    s_train = src_codes[train_mask]
    s_test = src_codes[test_mask]
    common = [
        c for c in np.unique(s_train) if (s_train == c).sum() >= 2 and (s_test == c).sum() >= 1
    ]
    if len(common) < 2:
        return {"error": "insufficient_source_classes"}

    def _probe_auroc(rep_train: np.ndarray, rep_test: np.ndarray, clf: Any) -> float:
        # Macro one-vs-rest AUROC averaged only over source classes that are
        # both modelled (seen in training) and present in the test split, so a
        # class absent from one fold does not collapse the whole score to NaN.
        sc = StandardScaler().fit(rep_train)
        clf.fit(sc.transform(rep_train), s_train)
        proba = clf.predict_proba(sc.transform(rep_test))
        trained = list(clf.classes_)
        keep = np.array([c in trained for c in s_test])
        y_keep = s_test[keep]
        p_keep = proba[keep]
        per_class: list[float] = []
        for j, cls in enumerate(trained):
            pos = (y_keep == cls).astype(int)
            if pos.sum() == 0 or pos.sum() == len(pos):
                continue
            try:
                per_class.append(float(roc_auc_score(pos, p_keep[:, j])))
            except ValueError:
                continue
        return float(np.mean(per_class)) if per_class else float("nan")

    raw_train = X.loc[train_mask].to_numpy()
    raw_test = X.loc[test_mask].to_numpy()
    results: dict[str, Any] = {}
    for probe_name, factory in (
        ("logistic", lambda: LogisticRegression(max_iter=1000)),
        ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed)),
    ):
        results[probe_name] = {
            "auroc_from_icp_representation": _probe_auroc(z_train, z_test, factory()),
            "auroc_from_raw_features": _probe_auroc(raw_train, raw_test, factory()),
        }

    # Majority-class baseline AUROC for multiclass macro-OVR is 0.5.
    n_classes = len(common)
    return {
        "n_source_classes": int(n_classes),
        "source_classes": [str(src_uniques[c]) for c in common],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "chance_auroc": 0.5,
        "conf_columns_used": conf_cols,
        "probes": results,
        "interpretation": (
            "AUROC near 0.5 from the ICP representation indicates monitoring "
            "source is not recoverable; a large gap below the raw-feature probe "
            "quantifies the suppression."
        ),
    }


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _auc_variance_hanley(auroc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil large-sample variance of a single ROC AUC.

    Parameters
    ----------
    auroc : float
        Area under the ROC curve in ``[0, 1]``.
    n_pos, n_neg : int
        Number of positive and negative cases (each must be >= 1).

    Returns
    -------
    float
        Estimated variance of the AUC.
    """
    if n_pos < 1 or n_neg < 1:
        raise ValueError("n_pos and n_neg must each be >= 1")
    a = float(auroc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    return (a * (1.0 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (
        n_pos * n_neg
    )


def auc_vs_chance(
    auroc: float, n_pos: int, n_neg: int, *, two_sided: bool = True
) -> dict[str, float]:
    """One-sample test of an AUC against chance (0.5).

    Uses the Hanley-McNeil standard error of a single AUC - appropriate for
    asking whether a held-out region's discrimination differs from chance when
    only the AUC and class counts are available (per-fold scores are not stored
    in ``loro_cv.json``). For ``auroc`` exactly 0.5 the z-statistic is 0.

    Parameters
    ----------
    auroc : float
        Observed AUC.
    n_pos, n_neg : int
        Positive / negative case counts.
    two_sided : bool
        Two-sided p-value if True (default); else one-sided (``H1: AUC > 0.5``).

    Returns
    -------
    dict
        ``{"auroc", "n_pos", "n_neg", "se", "z", "p_value", "two_sided"}``.
    """
    se = math.sqrt(_auc_variance_hanley(auroc, n_pos, n_neg))
    z = (float(auroc) - 0.5) / se if se > 0 else 0.0
    p = 2.0 * (1.0 - _normal_cdf(abs(z))) if two_sided else 1.0 - _normal_cdf(z)
    return {
        "auroc": float(auroc),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "se": float(se),
        "z": float(z),
        "p_value": float(p),
        "two_sided": bool(two_sided),
    }


def auc_difference(
    auroc1: float,
    n1_pos: int,
    n1_neg: int,
    auroc2: float,
    n2_pos: int,
    n2_neg: int,
) -> dict[str, float]:
    """Two-sample test of the difference between two INDEPENDENT AUCs.

    For AUCs estimated on disjoint case sets (e.g. high- vs low-burden
    subgroups), the groups are independent, so the variance of the difference
    is the sum of the two Hanley-McNeil variances. Returns a two-sided p-value
    for ``H0: AUC1 == AUC2``.

    Returns
    -------
    dict
        ``{"auroc1", "auroc2", "delta", "se", "z", "p_value"}``.
    """
    v1 = _auc_variance_hanley(auroc1, n1_pos, n1_neg)
    v2 = _auc_variance_hanley(auroc2, n2_pos, n2_neg)
    se = math.sqrt(v1 + v2)
    delta = float(auroc1) - float(auroc2)
    z = delta / se if se > 0 else 0.0
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return {
        "auroc1": float(auroc1),
        "auroc2": float(auroc2),
        "delta": float(delta),
        "se": float(se),
        "z": float(z),
        "p_value": float(p),
    }
