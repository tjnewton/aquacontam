"""Coordinate perturbation sensitivity analysis.

Quantifies the impact of ZIP centroid geocoding error on model performance
by perturbing system coordinates at multiple magnitudes, re-computing
spatial features, and measuring metric degradation.

This addresses the primary threat to validity: all 67,594 systems are
geocoded to ZCTA centroids (0-30 km error), and all proximity, land use,
hydrogeology, and demographics features depend on these coordinates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from aquacontam._constants import CRS_STORAGE

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result from a single perturbation run."""

    magnitude_km: float
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    n_systems: int = 0
    mean_displacement_km: float = 0.0


def perturb_coordinates(
    gdf: gpd.GeoDataFrame,
    magnitude_km: float,
    *,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Add Gaussian noise to coordinates to simulate geocoding error.

    Operates in EPSG:5070 (meters) for isotropic perturbation, then
    converts back to EPSG:4326.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Systems with point geometries. Must have a CRS set.
    magnitude_km : float
        Standard deviation of Gaussian noise in kilometers.
        Use 0 for no perturbation (returns a copy).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    gpd.GeoDataFrame
        Copy with perturbed coordinates in EPSG:4326.
    """
    if gdf.empty:
        return gdf.copy()

    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS set")

    if magnitude_km == 0.0:
        result = gdf.copy()
        if str(gdf.crs) != CRS_STORAGE:
            from aquacontam.geo.crs import to_wgs84

            result = to_wgs84(result)
        return result

    if magnitude_km < 0:
        raise ValueError(f"magnitude_km must be >= 0, got {magnitude_km}")

    from aquacontam.geo.crs import to_conus_albers, to_wgs84

    # Work in meters (EPSG:5070)
    gdf_proj = to_conus_albers(gdf)

    rng = np.random.RandomState(seed)
    x = gdf_proj.geometry.x.to_numpy()
    y = gdf_proj.geometry.y.to_numpy()

    noise_x = rng.normal(0, magnitude_km * 1000, len(x))
    noise_y = rng.normal(0, magnitude_km * 1000, len(y))

    new_x = x + noise_x
    new_y = y + noise_y

    new_geom = [Point(nx, ny) for nx, ny in zip(new_x, new_y, strict=True)]
    gdf_perturbed = gdf_proj.copy()
    gdf_perturbed = gdf_perturbed.set_geometry(new_geom)

    # Compute actual displacement for reporting
    displacements = np.sqrt(noise_x**2 + noise_y**2) / 1000.0  # km
    mean_disp = float(np.mean(displacements))
    logger.info(
        "Perturbed %d systems: target=%.1f km, actual mean=%.2f km",
        len(gdf_proj),
        magnitude_km,
        mean_disp,
    )

    # Convert back to WGS84
    result = to_wgs84(gdf_perturbed)
    return result


def feature_shortcut_perturbation(
    feature_dfs: list[pd.DataFrame],
    magnitude_km: float,
    *,
    seed: int = 42,
    spatial_prefixes: tuple[str, ...] = ("nearest_", "count_", "pct_", "aquifer_"),
) -> list[pd.DataFrame]:
    """Lightweight alternative: add scaled noise to spatial feature values.

    Instead of re-extracting features from perturbed coordinates (expensive),
    adds proportional noise to columns with spatial prefixes. This provides
    a fast approximation of coordinate sensitivity.

    Parameters
    ----------
    feature_dfs : list[pd.DataFrame]
        Feature DataFrames indexed by pwsid.
    magnitude_km : float
        Perturbation magnitude in km (controls noise scale).
    seed : int
        Random seed.
    spatial_prefixes : tuple[str, ...]
        Column name prefixes identifying spatial features.

    Returns
    -------
    list[pd.DataFrame]
        Perturbed copies of the input feature DataFrames.
    """
    if magnitude_km == 0.0:
        return [df.copy() for df in feature_dfs]

    rng = np.random.RandomState(seed)
    result = []
    # Scale factor: 1 km ~ 5% relative noise (empirical heuristic)
    noise_scale = magnitude_km * 0.05

    for df in feature_dfs:
        df_out = df.copy()
        spatial_cols = [
            c
            for c in df_out.columns
            if any(c.startswith(p) for p in spatial_prefixes)
            and pd.api.types.is_numeric_dtype(df_out[c])
        ]
        for col in spatial_cols:
            vals = df_out[col].to_numpy(dtype=float)
            noise = rng.normal(0, noise_scale, len(vals))
            # Multiplicative noise: feature * (1 + noise)
            df_out[col] = vals * (1.0 + noise)
        result.append(df_out)

    return result


def run_coordinate_sensitivity(
    model_cls: type,
    model_config: dict[str, Any],
    wq_df: pd.DataFrame,
    feature_extraction_fn: Callable[[gpd.GeoDataFrame], list[pd.DataFrame]] | None = None,
    *,
    feature_dfs: list[pd.DataFrame] | None = None,
    magnitudes_km: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 20.0),
    n_seeds: int = 5,
    base_seed: int = 42,
    analyte: str = "PFOS",
    target: str = "detected",
) -> list[PerturbationResult]:
    """Run coordinate sensitivity analysis at multiple perturbation magnitudes.

    For each (magnitude, seed): perturb coordinates, extract/perturb features,
    train model, evaluate, record metrics.

    Parameters
    ----------
    model_cls : type
        Model class (must implement BaseModel interface).
    model_config : dict
        Model configuration dict.
    wq_df : pd.DataFrame
        Water quality DataFrame.
    feature_extraction_fn : callable or None
        Callback ``fn(gdf) -> list[pd.DataFrame]`` that extracts features
        from a GeoDataFrame. If None, uses ``feature_shortcut_perturbation``
        on ``feature_dfs``.
    feature_dfs : list[pd.DataFrame] or None
        Pre-computed feature DataFrames (required if feature_extraction_fn
        is None).
    magnitudes_km : tuple[float, ...]
        Perturbation magnitudes to test.
    n_seeds : int
        Number of random seeds per magnitude.
    base_seed : int
        Base random seed.
    analyte : str
        Analyte to model.
    target : str
        Target variable type.

    Returns
    -------
    list[PerturbationResult]
        One result per (magnitude, seed) combination.
    """
    from aquacontam.benchmark.metrics import compute_classification_metrics
    from aquacontam.features.assembly import (
        aggregate_to_system_level,
        assemble_feature_matrix,
        build_system_geodataframe,
        derive_system_epa_regions,
        drop_leakage_columns,
    )
    from aquacontam.preprocessing.splits import geographic_split

    if feature_extraction_fn is None and feature_dfs is None:
        raise ValueError("Either feature_extraction_fn or feature_dfs must be provided")

    results: list[PerturbationResult] = []

    for magnitude in magnitudes_km:
        for seed_offset in range(n_seeds):
            seed = base_seed + seed_offset

            try:
                if feature_extraction_fn is not None:
                    # Full re-extraction path
                    gdf = build_system_geodataframe(wq_df)
                    gdf_perturbed = perturb_coordinates(gdf, magnitude, seed=seed)
                    try:
                        perturbed_features = feature_extraction_fn(gdf_perturbed)
                    except Exception:
                        logger.warning(
                            "Feature extraction failed at magnitude=%.1f km, seed=%d; "
                            "falling back to shortcut",
                            magnitude,
                            seed,
                        )
                        if feature_dfs is not None:
                            perturbed_features = feature_shortcut_perturbation(
                                feature_dfs, magnitude, seed=seed
                            )
                        else:
                            continue
                else:
                    # Shortcut path: perturb feature values directly
                    assert feature_dfs is not None
                    perturbed_features = feature_shortcut_perturbation(
                        feature_dfs,
                        magnitude,
                        seed=seed,
                    )

                # Aggregate and split. Regions come from the sample-level frame
                # so the coordinate fallback can rescue non-state PWSID prefixes.
                sys_agg = aggregate_to_system_level(wq_df, analyte, target=target)
                sys_agg = drop_leakage_columns(sys_agg)
                sys_agg = sys_agg.join(derive_system_epa_regions(wq_df).rename("epa_region"))
                train_sys, val_sys, test_sys = geographic_split(sys_agg.reset_index())

                train_idx = train_sys.set_index("pwsid")
                val_idx = val_sys.set_index("pwsid")
                test_idx = test_sys.set_index("pwsid")

                drop_cols = [
                    "epa_region",
                    "any_detected",
                    "detection_rate",
                    "max_concentration",
                ]
                train_idx = train_idx.drop(
                    columns=[c for c in drop_cols if c in train_idx.columns]
                )
                val_idx = val_idx.drop(columns=[c for c in drop_cols if c in val_idx.columns])
                test_idx = test_idx.drop(columns=[c for c in drop_cols if c in test_idx.columns])

                X_tr, y_tr, stats = assemble_feature_matrix(train_idx, *perturbed_features)
                X_val, y_val, _ = assemble_feature_matrix(
                    val_idx, *perturbed_features, impute_stats=stats
                )
                X_te, y_te, _ = assemble_feature_matrix(
                    test_idx, *perturbed_features, impute_stats=stats
                )

                if X_tr.empty or len(y_tr.unique()) < 2:
                    logger.warning(
                        "Insufficient data at magnitude=%.1f km, seed=%d",
                        magnitude,
                        seed,
                    )
                    continue

                model = model_cls(config={**model_config, "random_state": seed})
                fit_kwargs: dict[str, Any] = {}
                if not X_val.empty and len(y_val.unique()) >= 2:
                    fit_kwargs["X_val"] = X_val
                    fit_kwargs["y_val"] = y_val
                model.fit(X_tr, y_tr, **fit_kwargs)

                preds = model.predict(X_te)
                probs = model.predict_proba(X_te)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:, 1]

                metrics = compute_classification_metrics(
                    y_te, preds, probs, metrics=["auroc", "auprc"]
                )

                results.append(
                    PerturbationResult(
                        magnitude_km=magnitude,
                        seed=seed,
                        metrics=metrics,
                        n_systems=len(sys_agg),
                        mean_displacement_km=magnitude * np.sqrt(2 / np.pi),
                    )
                )

            except (ValueError, OSError, RuntimeError, TypeError, KeyError):
                logger.warning(
                    "Perturbation failed at magnitude=%.1f km, seed=%d",
                    magnitude,
                    seed,
                    exc_info=True,
                )
                continue

    return results


def coordinate_sensitivity_summary(
    results: list[PerturbationResult],
) -> pd.DataFrame:
    """Aggregate perturbation results: mean/std per magnitude per metric.

    Parameters
    ----------
    results : list[PerturbationResult]
        Raw results from ``run_coordinate_sensitivity()``.

    Returns
    -------
    pd.DataFrame
        One row per magnitude with columns: magnitude_km, auroc_mean,
        auroc_std, auprc_mean, auprc_std, delta_auroc, delta_auprc.
    """
    if not results:
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                columns=[
                    "magnitude_km",
                    "auroc_mean",
                    "auroc_std",
                    "auprc_mean",
                    "auprc_std",
                    "delta_auroc",
                    "delta_auprc",
                ]
            ),
        )

    rows = []
    magnitudes = sorted(set(r.magnitude_km for r in results))

    # Baseline metrics (magnitude=0)
    baseline_auroc = 0.0
    baseline_auprc = 0.0
    zero_results = [r for r in results if r.magnitude_km == 0.0]
    if zero_results:
        baseline_auroc = float(np.mean([r.metrics.get("auroc", 0) for r in zero_results]))
        baseline_auprc = float(np.mean([r.metrics.get("auprc", 0) for r in zero_results]))

    for mag in magnitudes:
        mag_results = [r for r in results if r.magnitude_km == mag]
        aurocs = [r.metrics.get("auroc", 0) for r in mag_results]
        auprcs = [r.metrics.get("auprc", 0) for r in mag_results]

        auroc_mean = float(np.mean(aurocs))
        auprc_mean = float(np.mean(auprcs))

        rows.append(
            {
                "magnitude_km": mag,
                "auroc_mean": auroc_mean,
                "auroc_std": float(np.std(aurocs)) if len(aurocs) > 1 else 0.0,
                "auprc_mean": auprc_mean,
                "auprc_std": float(np.std(auprcs)) if len(auprcs) > 1 else 0.0,
                "delta_auroc": auroc_mean - baseline_auroc,
                "delta_auprc": auprc_mean - baseline_auprc,
                "n_runs": len(mag_results),
            }
        )

    return cast(pd.DataFrame, pd.DataFrame(rows))
