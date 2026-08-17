"""Causal deconfounding via Double Machine Learning (DML).

Implements Partially Linear DML (Robinson 1988) to estimate the causal
effect of each environmental feature on contamination, orthogonalizing
against monitoring intensity confounders (n_samples, population_served, etc.).

Produces:
- A comparison table: SHAP correlational importance vs. causal effect per feature
- A deconfounded AUROC: model performance on residualized features

This addresses the key reviewer concern: "Are you finding real contamination
patterns or measurement artifacts from monitoring intensity?"
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _impute_fold(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN using train-fold means only (prevents leakage).

    Works for both 1-D (single feature) and 2-D (matrix) arrays.
    Returns copies; inputs are not modified.
    """
    train_out = train.copy()
    test_out = test.copy()
    if train.ndim == 1:
        mean_val = np.nanmean(train)
        if np.isnan(mean_val):
            mean_val = 0.0
        train_nan = np.isnan(train_out)
        test_nan = np.isnan(test_out)
        if train_nan.any():
            train_out[train_nan] = mean_val
        if test_nan.any():
            test_out[test_nan] = mean_val
    else:
        col_means = np.nanmean(train, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for j in range(train.shape[1]):
            t_mask = np.isnan(train_out[:, j])
            s_mask = np.isnan(test_out[:, j])
            if t_mask.any():
                train_out[t_mask, j] = col_means[j]
            if s_mask.any():
                test_out[s_mask, j] = col_means[j]
    return train_out, test_out


def _make_cv_splitter(n_folds: int = 5) -> Any:
    """Create a deterministic KFold splitter for synchronized cross-fitting.

    Using the same KFold object for both Y and X residualization ensures
    identical fold assignments, preventing subtle bias from independent splits.
    """
    from sklearn.model_selection import KFold

    return KFold(n_splits=n_folds, shuffle=False)


def _residualize_target(
    y: np.ndarray,
    W: np.ndarray,
    *,
    cv: Any = None,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute target residuals Y_tilde = Y - E[Y | W] via cross-fitting.

    NaN values in ``W`` are imputed per fold using train-fold means only
    to prevent data leakage.

    Parameters
    ----------
    y : np.ndarray
        Target variable, shape (n,).
    W : np.ndarray
        Confounder matrix, shape (n, p). May contain NaN.
    cv : sklearn splitter or None
        Cross-validation splitter. If None, a KFold(n_folds) is created.
    n_folds : int
        Number of cross-fitting folds (used only if ``cv`` is None).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Residualized target, shape (n,).
    """
    from sklearn.ensemble import GradientBoostingRegressor

    if cv is None:
        cv = _make_cv_splitter(n_folds)

    y_hat = np.zeros_like(y)
    for train_idx, test_idx in cv.split(W):
        W_train, W_test = _impute_fold(W[train_idx], W[test_idx])
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=seed, subsample=0.8
        )
        model.fit(W_train, y[train_idx])
        y_hat[test_idx] = model.predict(W_test)
    return np.asarray(y - y_hat)


def _residualize_feature(
    x_j: np.ndarray,
    W: np.ndarray,
    *,
    cv: Any = None,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute feature residuals X_tilde_j = X_j - E[X_j | W] via cross-fitting.

    NaN values in both ``x_j`` and ``W`` are imputed per fold using train-fold
    means only to prevent data leakage.

    Parameters
    ----------
    x_j : np.ndarray
        Single feature column, shape (n,). May contain NaN.
    W : np.ndarray
        Confounder matrix, shape (n, p). May contain NaN.
    cv : sklearn splitter or None
        Cross-validation splitter. If None, a KFold(n_folds) is created.
    n_folds : int
        Number of cross-fitting folds (used only if ``cv`` is None).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Residualized feature, shape (n,).
    """
    from sklearn.ensemble import GradientBoostingRegressor

    if cv is None:
        cv = _make_cv_splitter(n_folds)

    x_imputed = np.empty_like(x_j)
    x_hat = np.empty_like(x_j)
    for train_idx, test_idx in cv.split(W):
        W_train, W_test = _impute_fold(W[train_idx], W[test_idx])
        xj_train, xj_test = _impute_fold(x_j[train_idx], x_j[test_idx])
        x_imputed[test_idx] = xj_test
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=seed, subsample=0.8
        )
        model.fit(W_train, xj_train)
        x_hat[test_idx] = model.predict(W_test)
    return np.asarray(x_imputed - x_hat)


def causal_feature_analysis(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    confounders: list[str],
    *,
    shap_importances: pd.Series | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate causal effects via Partially Linear DML.

    For each non-confounder feature, estimates the causal effect after
    orthogonalizing against monitoring intensity confounders.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Target variable (binary detection or continuous concentration).
    confounders : list[str]
        Column names of confounding variables to partial out.
        Must be non-empty and present in X.
    shap_importances : pd.Series | None
        If provided, SHAP importance values indexed by feature name.
        Added to the output for comparison.
    n_folds : int
        Cross-fitting folds for DML.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: feature, causal_effect, std_error, p_value, p_value_fdr,
        significant_fdr, shap_importance, ratio.  ``causal_effect`` is a
        SEMI-STANDARDIZED partial coefficient (effect on the residualized outcome
        per 1 standard deviation of the residualized feature), so its magnitude is
        comparable across features measured in different units. The ``p_value_fdr``
        column contains Benjamini-Hochberg FDR-corrected p-values across all tested
        features; FDR significance is invariant to the feature rescaling.

    Raises
    ------
    ValueError
        If confounders list is empty or contains columns not in X.
    """
    from scipy import stats

    if not confounders:
        raise ValueError("confounders list must be non-empty")

    present = [c for c in confounders if c in X.columns]
    if not present:
        raise ValueError(f"None of the specified confounders {confounders} found in X.columns")

    W = X[present].to_numpy().astype(np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Create a single CV splitter shared across all residualizations
    # to ensure identical fold assignments for Y and X_j.
    # NaN imputation is handled per fold inside _residualize_* to prevent leakage.
    cv = _make_cv_splitter(n_folds)

    # Step 1: Residualize target
    y_tilde = _residualize_target(y_arr, W, cv=cv, seed=seed)

    # Step 2: For each non-confounder feature, estimate causal effect
    treatment_features = [c for c in X.columns if c not in present]
    results: list[dict[str, Any]] = []

    for feat in treatment_features:
        x_j = X[feat].to_numpy().astype(np.float64)

        # Skip features with no variance or all NaN
        valid = ~np.isnan(x_j)
        if valid.sum() < 10 or np.nanstd(x_j) < 1e-10:
            continue

        # Residualize feature (same cv splitter as target)
        # NaN in x_j and W are imputed per fold inside _residualize_feature.
        x_tilde = _residualize_feature(x_j, W, cv=cv, seed=seed)

        # Residualization quality: R² = 1 - var(residual)/var(original)
        x_var = np.nanvar(x_j)
        xt_var = np.var(x_tilde)
        resid_r2 = 1.0 - xt_var / max(x_var, 1e-10) if x_var > 1e-10 else 0.0

        # Overlap check: if R² > 0.95, confounders almost fully predict
        # the feature, violating the positivity/overlap assumption.
        overlap_warn = bool(resid_r2 > 0.95)
        if overlap_warn:
            logger.warning(
                "Feature '%s' residualization R²=%.3f (>0.95) — "
                "overlap assumption may be violated.",
                feat,
                resid_r2,
            )

        # Neyman orthogonality diagnostic: residuals should be uncorrelated
        # with confounders. Check max |corr(x_tilde, W_j)| across confounders.
        max_resid_corr = 0.0
        for w_idx in range(W.shape[1]):
            w_col = W[:, w_idx]
            valid_both = ~np.isnan(w_col)
            if valid_both.sum() > 2:
                corr_val = abs(np.corrcoef(x_tilde[valid_both], w_col[valid_both])[0, 1])
                if not np.isnan(corr_val):
                    max_resid_corr = max(max_resid_corr, corr_val)
        if max_resid_corr > 0.1:
            logger.warning(
                "Feature '%s': max |corr(x_tilde, W)| = %.3f (>0.1) — "
                "first-stage model may be misspecified.",
                feat,
                max_resid_corr,
            )

        # Semi-standardize the residualized feature to unit SD so the estimated
        # coefficient is the effect per 1-SD change in the feature. This makes the
        # |causal_effect| magnitude (and the SHAP-vs-DML rank comparison) SCALE-
        # INVARIANT -- comparable across features measured in different units
        # (metre-scale distances vs 0-100 percentage demographics). Diagnostics
        # above (resid_r2, Neyman corr) deliberately use the unstandardized x_tilde.
        # The t-statistic, SE and p-value are unchanged by this rescaling (beta and
        # se scale together), so FDR significance is preserved exactly.
        xt_sd = np.std(x_tilde)
        if xt_sd < 1e-10:
            continue
        x_tilde = x_tilde / xt_sd

        # OLS: y_tilde = beta * x_tilde + epsilon
        denom = np.dot(x_tilde, x_tilde)
        if denom < 1e-10:
            continue
        beta = np.dot(x_tilde, y_tilde) / denom
        residuals = y_tilde - beta * x_tilde
        n = len(y_tilde)
        # HC1 heteroscedasticity-robust sandwich standard error
        resid2 = residuals**2
        meat = np.sum(resid2 * x_tilde**2)
        bread = denom
        se = np.sqrt(meat / bread**2 * n / max(n - 1, 1))
        t_stat = beta / max(se, 1e-10)
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(n - 2, 1)))

        entry: dict[str, Any] = {
            "feature": feat,
            "causal_effect": float(beta),
            "std_error": float(se),
            "p_value": float(p_value),
            "residualization_r2": float(resid_r2),
            "overlap_warning": overlap_warn,
            "max_residual_confounder_corr": float(max_resid_corr),
        }

        if shap_importances is not None and feat in shap_importances.index:
            shap_val = float(shap_importances[feat])
            entry["shap_importance"] = shap_val
            raw_ratio = abs(float(beta)) / max(abs(shap_val), 1e-10)
            entry["ratio"] = min(raw_ratio, 100.0)
        else:
            entry["shap_importance"] = np.nan
            entry["ratio"] = np.nan

        results.append(entry)

    result_df: pd.DataFrame = pd.DataFrame(results).sort_values("p_value", ignore_index=True)

    # Apply Benjamini-Hochberg FDR correction across all tested features
    if len(result_df) > 0:
        from aquacontam.analysis.equity import apply_multiple_testing_correction

        correction = apply_multiple_testing_correction(
            result_df["p_value"].tolist(), method="fdr_bh", alpha=0.05
        )
        result_df["p_value_fdr"] = correction["corrected_p_values"]
        result_df["significant_fdr"] = correction["reject"]
    else:
        result_df["p_value_fdr"] = pd.Series(dtype=float)
        result_df["significant_fdr"] = pd.Series(dtype=bool)

    return result_df


def compare_dml_shap_rankings(
    dml_results: pd.DataFrame,
    demographic_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """Compute scale-free rank-based comparison of DML vs SHAP importance.

    Replaces the raw per-feature ratio (which divides incommensurable
    quantities — a regression coefficient vs. a marginal contribution)
    with rank-based metrics that are invariant to the scale of each method.

    Parameters
    ----------
    dml_results : pd.DataFrame
        Output of ``causal_feature_analysis()``.  Must contain columns
        ``feature``, ``causal_effect``, ``shap_importance``.
    demographic_prefixes : list[str], optional
        Feature name prefixes identifying demographic/EJ features
        (from ``experiment.yaml`` ablation config).

    Returns
    -------
    dict[str, Any]
        Scale-free comparison metrics including Spearman rank correlation,
        per-feature ranks and displacements, and demographic subset summary.
    """
    from scipy import stats as sp_stats

    # Filter to features with both metrics present and positive
    mask = (
        dml_results["causal_effect"].notna()
        & dml_results["shap_importance"].notna()
        & (dml_results["causal_effect"].abs() > 1e-10)
        & (dml_results["shap_importance"].abs() > 1e-10)
    )
    df = dml_results.loc[mask].copy()
    n = len(df)

    if n < 3:
        logger.warning("Too few features (%d) for rank comparison", n)
        return {"n_features": n, "spearman_rho": np.nan, "spearman_p": np.nan}

    dml_abs = df["causal_effect"].abs().to_numpy()
    shap_abs = df["shap_importance"].abs().to_numpy()

    # Spearman rank correlation
    rho, p_rho = sp_stats.spearmanr(dml_abs, shap_abs)

    # Per-feature ranks (1 = largest)
    dml_ranks = sp_stats.rankdata(-dml_abs).astype(int)
    shap_ranks = sp_stats.rankdata(-shap_abs).astype(int)
    displacements = (shap_ranks - dml_ranks).astype(int)

    feature_ranks = []
    for i, row in enumerate(df.itertuples()):
        feature_ranks.append(
            {
                "feature": row.feature,
                "dml_rank": int(dml_ranks[i]),
                "shap_rank": int(shap_ranks[i]),
                "displacement": int(displacements[i]),
            }
        )

    result: dict[str, Any] = {
        "n_features": n,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "feature_ranks": feature_ranks,
    }

    # Demographic subset analysis
    if demographic_prefixes:
        is_demo = df["feature"].apply(lambda f: any(f.startswith(p) for p in demographic_prefixes))
        demo_idx = np.where(is_demo.to_numpy())[0]

        if len(demo_idx) > 0:
            demo_disps = displacements[demo_idx]
            demo_ranks_list = [feature_ranks[i] for i in demo_idx]
            result["demographic_ranks"] = demo_ranks_list
            result["demographic_mean_displacement"] = float(np.mean(demo_disps))
            result["demographic_n"] = len(demo_idx)

    return result


def _build_feature_to_category(
    feature_names: list[str],
    category_prefixes: dict[str, list[str]],
) -> dict[str, str]:
    """Map each feature name to its category via prefix matching.

    Parameters
    ----------
    feature_names : list[str]
        Feature names to classify.
    category_prefixes : dict[str, list[str]]
        Mapping from category name to list of prefix strings.
        First matching category wins (priority = iteration order).

    Returns
    -------
    dict[str, str]
        Feature name → category. Unmatched features get ``"other"``.
    """
    mapping: dict[str, str] = {}
    for feat in feature_names:
        matched = False
        for cat, prefixes in category_prefixes.items():
            if any(feat.startswith(p) or feat == p for p in prefixes):
                mapping[feat] = cat
                matched = True
                break
        if not matched:
            mapping[feat] = "other"
    return mapping


def category_displacement_test(
    rank_comparison: dict[str, Any],
    feature_to_category: dict[str, str],
    *,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test whether feature categories show larger rank displacements than chance.

    Permutation test: shuffle category labels, recompute per-category mean
    absolute displacement, compare to observed values.

    Parameters
    ----------
    rank_comparison : dict[str, Any]
        Output from :func:`compare_dml_shap_rankings`, must contain
        ``"feature_ranks"`` list of dicts with ``"feature"`` and
        ``"displacement"`` keys.
    feature_to_category : dict[str, str]
        Feature name → category mapping (from :func:`_build_feature_to_category`).
    n_permutations : int
        Number of permutations for the null distribution.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``category``, ``n_features``, ``mean_abs_displacement``,
        ``p_value``, ``p_value_fdr``, ``significant_fdr``.
    """
    feature_ranks = rank_comparison.get("feature_ranks", [])
    if not feature_ranks:
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                columns=[
                    "category",
                    "n_features",
                    "mean_abs_displacement",
                    "p_value",
                    "p_value_fdr",
                    "significant_fdr",
                ]
            ),
        )

    features = [r["feature"] for r in feature_ranks]
    displacements = np.array([abs(r["displacement"]) for r in feature_ranks])
    categories = np.array([feature_to_category.get(f, "other") for f in features])
    unique_cats = sorted(set(categories))

    # Observed mean |displacement| per category
    observed: dict[str, float] = {}
    cat_sizes: dict[str, int] = {}
    for cat in unique_cats:
        mask = categories == cat
        cat_sizes[cat] = int(mask.sum())
        observed[cat] = float(displacements[mask].mean())

    # Permutation test: shuffle category labels
    rng = np.random.RandomState(seed)
    exceed_counts: dict[str, int] = {cat: 0 for cat in unique_cats}

    for _ in range(n_permutations):
        perm_cats = rng.permutation(categories)
        for cat in unique_cats:
            mask = perm_cats == cat
            null_mean = float(displacements[mask].mean())
            if null_mean >= observed[cat]:
                exceed_counts[cat] += 1

    # P-values: (count + 1) / (N + 1) convention
    p_values = {cat: (exceed_counts[cat] + 1) / (n_permutations + 1) for cat in unique_cats}

    # FDR correction
    from aquacontam.analysis.equity import apply_multiple_testing_correction

    cats_ordered = sorted(unique_cats)
    raw_ps = [p_values[c] for c in cats_ordered]
    correction = apply_multiple_testing_correction(raw_ps, method="fdr_bh")
    fdr_ps = correction["corrected_p_values"]

    rows = []
    for i, cat in enumerate(cats_ordered):
        rows.append(
            {
                "category": cat,
                "n_features": cat_sizes[cat],
                "mean_abs_displacement": round(observed[cat], 2),
                "p_value": round(raw_ps[i], 4),
                "p_value_fdr": round(float(fdr_ps[i]), 4),
                "significant_fdr": bool(fdr_ps[i] < 0.05),
            }
        )

    return cast(pd.DataFrame, pd.DataFrame(rows))


def train_deconfounded_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    confounders: list[str],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train XGBoost on residualized features and report deconfounded AUROC.

    This represents the "true environmental signal" ceiling — performance
    that cannot be attributed to monitoring intensity confounders.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series | np.ndarray
        Binary target.
    confounders : list[str]
        Confounder column names to partial out.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``"deconfounded_auroc"``, ``"original_auroc"``, ``"n_features"``.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    present = [c for c in confounders if c in X.columns]
    y_arr = np.asarray(y, dtype=np.float64)

    # Original cross-validated AUROC with per-fold imputation (no leakage)
    X_np = X.to_numpy().astype(np.float64)
    cv_orig = _make_cv_splitter(n_folds)
    orig_proba = np.zeros((len(y_arr), 2))
    for train_idx, test_idx in cv_orig.split(X_np):
        X_train, X_test = _impute_fold(X_np[train_idx], X_np[test_idx])
        clf_orig = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=seed, subsample=0.8
        )
        clf_orig.fit(X_train, y_arr[train_idx])
        orig_proba[test_idx] = clf_orig.predict_proba(X_test)
    original_auroc = roc_auc_score(y_arr, orig_proba[:, 1])

    # Residualize all non-confounder features
    if not present:
        return {
            "deconfounded_auroc": original_auroc,
            "original_auroc": original_auroc,
            "n_features": X.shape[1],
        }

    W = X[present].to_numpy().astype(np.float64)
    treatment_cols = [c for c in X.columns if c not in present]
    X_resid = np.zeros((len(y_arr), len(treatment_cols)), dtype=np.float64)

    # Shared CV splitter for synchronized fold assignments.
    # NaN imputation is handled per fold inside _residualize_feature.
    cv = _make_cv_splitter(n_folds)

    for i, col in enumerate(treatment_cols):
        x_j = X[col].to_numpy().astype(np.float64)
        valid = ~np.isnan(x_j)
        if valid.sum() < 10 or np.nanstd(x_j) < 1e-10:
            X_resid[:, i] = 0.0
            continue
        X_resid[:, i] = _residualize_feature(x_j, W, cv=cv, seed=seed)

    # Cross-validated AUROC on residualized features
    cv_deconf = _make_cv_splitter(n_folds)
    deconf_proba = np.zeros((len(y_arr), 2))
    for train_idx, test_idx in cv_deconf.split(X_resid):
        X_train, X_test = _impute_fold(X_resid[train_idx], X_resid[test_idx])
        clf_deconf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=seed, subsample=0.8
        )
        clf_deconf.fit(X_train, y_arr[train_idx])
        deconf_proba[test_idx] = clf_deconf.predict_proba(X_test)
    deconfounded_auroc = roc_auc_score(y_arr, deconf_proba[:, 1])

    return {
        "deconfounded_auroc": float(deconfounded_auroc),
        "original_auroc": float(original_auroc),
        "n_features": len(treatment_cols),
    }


def compute_sensitivity_bounds(
    causal_effect: float,
    std_error: float,
    *,
    gamma_range: tuple[float, float] = (1.0, 3.0),
    n_steps: int = 21,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rosenbaum-style sensitivity bounds for unmeasured confounding.

    For each sensitivity parameter gamma, computes how large an unmeasured
    confounder would need to be to explain away the estimated causal effect.
    At gamma=1 there is no unmeasured confounding; at gamma=k an unmeasured
    confounder could multiply the odds of treatment by up to k.

    The key output is the "tipping point" Gamma* — the smallest gamma at
    which the confidence interval first includes zero, meaning the causal
    conclusion would be overturned.

    Parameters
    ----------
    causal_effect : float
        Point estimate of the causal effect (e.g. DML coefficient).
    std_error : float
        Standard error of the causal effect estimate.
    gamma_range : tuple[float, float]
        Range of sensitivity parameter values to evaluate.  Must satisfy
        ``gamma_range[0] >= 1.0``.
    n_steps : int
        Number of evenly spaced gamma values to evaluate.
    alpha : float
        Significance level for two-sided confidence intervals.

    Returns
    -------
    pd.DataFrame
        Columns: ``gamma``, ``effect_lower``, ``effect_upper``,
        ``p_value_bound``, ``significant``.  One row per gamma value.

    Notes
    -----
    Follows the Rosenbaum (2002) framework adapted for continuous outcomes:
    at sensitivity level gamma the effect is bounded by
    ``[effect - log(gamma) * se, effect + log(gamma) * se]``, and the
    worst-case p-value bound comes from the shifted z-statistic
    ``z_gamma = (|effect| - log(gamma) * se) / se``.

    References
    ----------
    Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
    """
    from scipy import stats

    if std_error <= 0:
        raise ValueError("std_error must be positive")
    if gamma_range[0] < 1.0:
        raise ValueError("gamma_range lower bound must be >= 1.0")

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_steps)

    rows: list[dict[str, float | bool]] = []
    for gamma in gammas:
        shift = np.log(gamma) * std_error
        lower = causal_effect - shift - z_crit * std_error
        upper = causal_effect + shift + z_crit * std_error

        # Worst-case z-statistic: how significant is the effect after
        # accounting for a confounder of strength gamma?
        z_gamma = (abs(causal_effect) - shift) / std_error
        p_bound = 2.0 * (1.0 - stats.norm.cdf(max(z_gamma, 0.0)))

        rows.append(
            {
                "gamma": float(gamma),
                "effect_lower": float(lower),
                "effect_upper": float(upper),
                "p_value_bound": float(p_bound),
                "significant": bool(p_bound < alpha),
            }
        )

    return cast(pd.DataFrame, pd.DataFrame(rows))


def sensitivity_analysis(
    dml_results: list[dict[str, Any]],
    *,
    features: list[str] | None = None,
    gamma_range: tuple[float, float] = (1.0, 3.0),
) -> dict[str, pd.DataFrame]:
    """Run Rosenbaum sensitivity analysis on DML causal effect estimates.

    Wraps :func:`compute_sensitivity_bounds` for each feature in the DML
    output, making it easy to assess which causal conclusions are robust
    to unmeasured confounding.

    Parameters
    ----------
    dml_results : list[dict[str, Any]]
        List of DML result dicts, each containing at minimum the keys
        ``"feature"``, ``"causal_effect"``, and ``"std_error"``.
    features : list[str] | None
        If provided, only compute sensitivity bounds for these features.
        If None, all features in ``dml_results`` are analyzed.
    gamma_range : tuple[float, float]
        Range of sensitivity parameter values passed to
        :func:`compute_sensitivity_bounds`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from feature name to a DataFrame of sensitivity bounds
        (one row per gamma value).
    """
    out: dict[str, pd.DataFrame] = {}
    for row in dml_results:
        feat = row["feature"]
        if features is not None and feat not in features:
            continue
        se = row["std_error"]
        if se <= 0:
            logger.warning("Skipping feature '%s': std_error=%.6g is non-positive.", feat, se)
            continue
        bounds = compute_sensitivity_bounds(
            causal_effect=row["causal_effect"],
            std_error=se,
            gamma_range=gamma_range,
        )
        out[feat] = bounds
    return out
