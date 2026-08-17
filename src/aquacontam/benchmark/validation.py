"""Known site validation — sanity-check model predictions at documented contamination sites.

A good model should predict detection at sites with well-documented
PFAS or heavy metal contamination. This module provides a curated list
of known sites and functions to evaluate model performance against them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aquacontam.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated known contamination sites
# ---------------------------------------------------------------------------

KNOWN_CONTAMINATION_SITES: list[dict[str, Any]] = [
    # Military bases with AFFF contamination (DoD PFAS list)
    {
        "name": "Pease AFB, NH",
        "state": "NH",
        "lat": 43.078,
        "lon": -70.823,
        "contaminants": ["PFOS", "PFOA"],
        "expected_detection": True,
    },
    {
        "name": "Peterson AFB, CO",
        "state": "CO",
        "lat": 38.824,
        "lon": -104.703,
        "contaminants": ["PFOS", "PFOA"],
        "expected_detection": True,
    },
    {
        "name": "Wurtsmith AFB, MI",
        "state": "MI",
        "lat": 44.452,
        "lon": -83.379,
        "contaminants": ["PFOS", "PFOA"],
        "expected_detection": True,
    },
    {
        "name": "Naval Air Weapons Station China Lake, CA",
        "state": "CA",
        "lat": 35.686,
        "lon": -117.692,
        "contaminants": ["PFOS", "PFOA"],
        "expected_detection": True,
    },
    # Superfund / industrial sites with known PFAS
    {
        "name": "Chemours/Fayetteville Works, NC",
        "state": "NC",
        "lat": 34.844,
        "lon": -78.824,
        "contaminants": ["HFPO-DA", "PFOA", "PFOS"],
        "expected_detection": True,
    },
    {
        "name": "3M Cottage Grove, MN",
        "state": "MN",
        "lat": 44.828,
        "lon": -92.943,
        "contaminants": ["PFOS", "PFOA", "PFBS"],
        "expected_detection": True,
    },
    {
        "name": "Saint-Gobain Performance Plastics, Hoosick Falls, NY",
        "state": "NY",
        "lat": 42.905,
        "lon": -73.349,
        "contaminants": ["PFOA"],
        "expected_detection": True,
    },
    {
        "name": "Wolverine World Wide, Belmont MI",
        "state": "MI",
        "lat": 43.165,
        "lon": -85.589,
        "contaminants": ["PFOS", "PFOA"],
        "expected_detection": True,
    },
    # Known heavy metal hotspots
    {
        "name": "Flint, MI",
        "state": "MI",
        "lat": 43.013,
        "lon": -83.687,
        "contaminants": ["lead"],
        "expected_detection": True,
    },
    {
        "name": "Newark, NJ",
        "state": "NJ",
        "lat": 40.735,
        "lon": -74.172,
        "contaminants": ["lead", "copper"],
        "expected_detection": True,
    },
]
"""Curated known contamination sites for model validation."""


@dataclass
class SiteResult:
    """Result for a single known site.

    Parameters
    ----------
    site_name : str
        Human-readable site name.
    matched_pwsid : str | None
        Nearest matched water system ID.
    distance_km : float
        Distance to nearest system (km).
    predicted_detection : bool
        Whether model predicted detection.
    predicted_probability : float | None
        Predicted probability of detection (if available).
    expected_detection : bool
        Ground truth expectation.
    is_hit : bool
        True if prediction matches expectation.
    """

    site_name: str
    matched_pwsid: str | None
    distance_km: float
    predicted_detection: bool
    predicted_probability: float | None
    expected_detection: bool
    is_hit: bool


@dataclass
class ValidationReport:
    """Aggregate validation report across known sites.

    Parameters
    ----------
    site_results : list[SiteResult]
        Per-site results.
    hit_rate : float
        Fraction of known sites correctly identified.
    false_negatives : list[str]
        Names of sites where detection was expected but not predicted.
    metadata : dict[str, Any]
        Additional info (model name, n_sites, etc.).
    """

    site_results: list[SiteResult]
    hit_rate: float
    false_negatives: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _find_nearest_system(
    lat: float,
    lon: float,
    systems_lats: np.ndarray,
    systems_lons: np.ndarray,
    pwsids: np.ndarray,
) -> tuple[str, float]:
    """Find the nearest water system to a given coordinate.

    Uses haversine approximation for distance calculation.

    Returns
    -------
    tuple[str, float]
        (pwsid, distance_km)
    """
    # Haversine distance approximation
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    sys_lat_rad = np.radians(systems_lats)
    sys_lon_rad = np.radians(systems_lons)

    dlat = sys_lat_rad - lat_rad
    dlon = sys_lon_rad - lon_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(sys_lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in km

    idx = int(np.argmin(km))
    return str(pwsids[idx]), float(km[idx])


def validate_against_known_sites(
    model: BaseModel,
    X: pd.DataFrame,
    systems_gdf: pd.DataFrame,
    known_sites: list[dict[str, Any]] | None = None,
    *,
    max_distance_km: float = 50.0,
) -> ValidationReport:
    """Validate model predictions against known contamination sites.

    For each known site, finds the nearest water system in the dataset,
    generates a prediction, and checks whether the model correctly
    identifies the contamination.

    Parameters
    ----------
    model : BaseModel
        Trained model for prediction.
    X : pd.DataFrame
        Feature matrix indexed by pwsid.
    systems_gdf : pd.DataFrame
        Must have ``latitude`` and ``longitude`` columns, indexed by pwsid.
    known_sites : list[dict], optional
        Known contamination sites. Defaults to ``KNOWN_CONTAMINATION_SITES``.
    max_distance_km : float
        Maximum distance to consider a match (km).

    Returns
    -------
    ValidationReport
        Aggregated results across all known sites.
    """
    if known_sites is None:
        known_sites = KNOWN_CONTAMINATION_SITES

    if systems_gdf.empty or X.empty:
        return ValidationReport(
            site_results=[],
            hit_rate=0.0,
            false_negatives=[s["name"] for s in known_sites if s.get("expected_detection", True)],
            metadata={"error": "empty input data"},
        )

    # Extract system coordinates
    sys_lats = np.asarray(systems_gdf["latitude"])
    sys_lons = np.asarray(systems_gdf["longitude"])
    sys_pwsids = np.asarray(systems_gdf.index)

    site_results: list[SiteResult] = []
    false_negatives: list[str] = []

    for site in known_sites:
        pwsid, dist_km = _find_nearest_system(
            site["lat"], site["lon"], sys_lats, sys_lons, sys_pwsids
        )

        if dist_km > max_distance_km or pwsid not in X.index:
            # No nearby system in dataset
            result = SiteResult(
                site_name=site["name"],
                matched_pwsid=None,
                distance_km=dist_km,
                predicted_detection=False,
                predicted_probability=None,
                expected_detection=site["expected_detection"],
                is_hit=False,
            )
            if site["expected_detection"]:
                false_negatives.append(site["name"])
        else:
            # Get prediction for matched system
            x_row = X.loc[[pwsid]]
            pred = bool(model.predict(x_row)[0])

            try:
                proba = model.predict_proba(x_row)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    prob_val = float(proba[0, 1])
                else:
                    prob_val = float(proba[0])
            except NotImplementedError:
                prob_val = None

            is_hit = pred == site["expected_detection"]
            result = SiteResult(
                site_name=site["name"],
                matched_pwsid=pwsid,
                distance_km=dist_km,
                predicted_detection=pred,
                predicted_probability=prob_val,
                expected_detection=site["expected_detection"],
                is_hit=is_hit,
            )

            if not is_hit and site["expected_detection"]:
                false_negatives.append(site["name"])

        site_results.append(result)

    n_hits = sum(1 for r in site_results if r.is_hit)
    n_total = len(site_results)
    hit_rate = n_hits / n_total if n_total > 0 else 0.0

    logger.info(
        "Known site validation: %d/%d hits (%.1f%%), %d false negatives",
        n_hits,
        n_total,
        hit_rate * 100,
        len(false_negatives),
    )

    return ValidationReport(
        site_results=site_results,
        hit_rate=hit_rate,
        false_negatives=false_negatives,
        metadata={
            "model_name": model.name,
            "n_sites": n_total,
            "n_matched": sum(1 for r in site_results if r.matched_pwsid is not None),
            "max_distance_km": max_distance_km,
        },
    )
