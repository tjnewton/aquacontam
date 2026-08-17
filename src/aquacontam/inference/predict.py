"""Generate pre-computed grid predictions for the risk map.

Produces Parquet files with risk scores for ~227k CONUS grid points,
partitioned by task and analyte.  The webapp loads these static
predictions rather than running inference in real-time.

Typical usage::

    from aquacontam.inference.predict import generate_grid_predictions

    generate_grid_predictions(
        models={"T1": fitted_xgb},
        feature_df=features,
        grid=grid_gdf,
        output_dir=Path("data/processed/predictions"),
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from aquacontam._constants import (
    T1_DEFAULT_ANALYTES,
    T2_DEFAULT_ANALYTES,
    T4_DEFAULT_ANALYTES,
)
from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)

# Task → default analytes mapping
_TASK_ANALYTES: dict[str, tuple[str, ...]] = {
    "T1": T1_DEFAULT_ANALYTES,
    "T2": T2_DEFAULT_ANALYTES,
    "T4": T4_DEFAULT_ANALYTES,
}


def _assign_grid_epa_region(
    latitudes: pd.Series | np.ndarray,
    longitudes: pd.Series | np.ndarray,
) -> pd.Series:
    """Approximate EPA region for grid points from longitude bands.

    This is a coarse heuristic — grid points don't have PWSIDs so we
    estimate region from geographic position.  Adequate for map display
    but not for official reporting.

    Parameters
    ----------
    latitudes : array-like
        Latitude values.
    longitudes : array-like
        Longitude values.

    Returns
    -------
    pd.Series
        Integer EPA region estimates.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    regions = np.full(len(lats), 9, dtype=int)  # default Pacific

    # Rough longitude/latitude bands for EPA regions
    regions = np.where((lons >= -67) & (lats >= 41), 1, regions)  # New England
    regions = np.where((lons >= -80) & (lons < -67) & (lats >= 40), 2, regions)  # NY/NJ
    regions = np.where((lons >= -84) & (lons < -74) & (lats >= 36) & (lats < 41), 3, regions)
    regions = np.where((lons >= -91) & (lons < -75) & (lats >= 24) & (lats < 37), 4, regions)
    regions = np.where((lons >= -93) & (lons < -80) & (lats >= 37) & (lats < 50), 5, regions)
    regions = np.where((lons >= -107) & (lons < -89) & (lats >= 25) & (lats < 37), 6, regions)
    regions = np.where((lons >= -104) & (lons < -89) & (lats >= 37) & (lats < 44), 7, regions)
    regions = np.where((lons >= -117) & (lons < -96) & (lats >= 37) & (lats < 50), 8, regions)
    regions = np.where((lons >= -125) & (lons < -114) & (lats < 42), 9, regions)
    regions = np.where((lons >= -125) & (lons < -114) & (lats >= 42), 10, regions)

    return cast(pd.Series, pd.Series(regions, dtype=int))


def _predict_for_grid(
    model: BaseModel,
    features: pd.DataFrame,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model prediction on grid features.

    Parameters
    ----------
    model : BaseModel
        Fitted model.
    features : pd.DataFrame
        Feature matrix for grid points.
    task_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(risk_scores, predictions)`` — continuous scores and
        binary/continuous predictions.
    """
    if task_type == "classification":
        probs = model.predict_proba(features)
        if probs.ndim == 2 and probs.shape[1] == 2:
            risk_scores = probs[:, 1]
        elif probs.ndim == 2:
            risk_scores = probs[:, -1]
        else:
            risk_scores = probs
        predictions = (risk_scores >= 0.5).astype(int)
    else:
        preds = model.predict(features)
        # Normalise regression to 0-1 range for risk score
        pmin, pmax = float(preds.min()), float(preds.max())
        if pmax > pmin:
            risk_scores = (preds - pmin) / (pmax - pmin)
        else:
            risk_scores = np.full_like(preds, 0.5 if pmin > 0 else 0.0, dtype=float)
        predictions = preds

    return np.asarray(risk_scores, dtype=float), np.asarray(predictions, dtype=float)


def generate_grid_predictions(
    models: dict[str, BaseModel],
    feature_df: pd.DataFrame,
    grid: pd.DataFrame,
    output_dir: str | Path,
    *,
    task_analytes: dict[str, tuple[str, ...]] | None = None,
    task_types: dict[str, str] | None = None,
) -> list[Path]:
    """Generate pre-computed risk predictions for a CONUS grid.

    Parameters
    ----------
    models : dict[str, BaseModel]
        Fitted models keyed by task name (e.g. ``{"T1": xgb_model}``).
    feature_df : pd.DataFrame
        Feature matrix aligned with *grid* rows.  Must have same length
        as *grid* and contain only numeric columns.
    grid : pd.DataFrame | GeoDataFrame
        Grid points with ``grid_id``, ``latitude`` / ``longitude`` (or
        geometry from which they can be extracted).
    output_dir : str | Path
        Root directory for Hive-partitioned Parquet output.
    task_analytes : dict mapping task→analytes, optional
        Override default analytes per task.
    task_types : dict mapping task→task_type, optional
        Override task types. Defaults: T1/T4=classification, T2=regression.

    Returns
    -------
    list[Path]
        Paths to written Parquet files.
    """
    output_dir = Path(output_dir)

    if task_analytes is None:
        task_analytes = dict(_TASK_ANALYTES)
    if task_types is None:
        task_types = {"T1": "classification", "T2": "regression", "T4": "classification"}

    # Extract lat/lon from grid
    if "latitude" in grid.columns and "longitude" in grid.columns:
        latitudes = grid["latitude"]
        longitudes = grid["longitude"]
    elif hasattr(grid, "geometry") and grid.geometry is not None:
        longitudes = grid.geometry.x
        latitudes = grid.geometry.y
    else:
        raise ValueError("Grid must have latitude/longitude columns or point geometry")

    grid_ids = grid["grid_id"] if "grid_id" in grid.columns else pd.RangeIndex(len(grid))
    epa_regions = _assign_grid_epa_region(latitudes, longitudes)

    written: list[Path] = []

    for task_name, model in models.items():
        analytes = task_analytes.get(task_name, ("unknown",))
        task_type = task_types.get(task_name, "classification")

        logger.info(
            "Predicting %s (%d analytes, %d grid points)", task_name, len(analytes), len(grid)
        )

        risk_scores, predictions = _predict_for_grid(model, feature_df, task_type)

        for analyte in analytes:
            result_df = pd.DataFrame(
                {
                    "grid_id": grid_ids.to_numpy(),
                    "latitude": np.asarray(latitudes, dtype=float),
                    "longitude": np.asarray(longitudes, dtype=float),
                    "task": task_name,
                    "analyte": analyte,
                    "model_name": model.name,
                    "risk_score": risk_scores,
                    "prediction": predictions,
                    "epa_region": epa_regions.to_numpy(),
                }
            )

            part_dir = output_dir / f"task={task_name}" / f"analyte={analyte}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_path = part_dir / "predictions.parquet"
            result_df.to_parquet(out_path, index=False)
            written.append(out_path)

            logger.info("  Wrote %s (%d rows)", out_path, len(result_df))

    logger.info("Generated %d prediction files in %s", len(written), output_dir)
    return written
