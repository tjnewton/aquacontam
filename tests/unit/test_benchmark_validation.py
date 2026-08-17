"""Tests for benchmark validation against known contamination sites."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.benchmark.validation import (
    KNOWN_CONTAMINATION_SITES,
    ValidationReport,
    _find_nearest_system,
    validate_against_known_sites,
)
from aquacontam.models.base import BaseModel


class _AlwaysDetectModel(BaseModel):
    """Test model that always predicts detection."""

    @property
    def name(self) -> str:
        return "always_detect"

    def fit(self, X_train, y_train, **kwargs) -> None:
        pass

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.1), np.full(n, 0.9)])


class _NeverDetectModel(BaseModel):
    """Test model that never predicts detection."""

    @property
    def name(self) -> str:
        return "never_detect"

    def fit(self, X_train, y_train, **kwargs) -> None:
        pass

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])


@pytest.fixture()
def systems_near_known_sites():
    """Systems GeoDataFrame with systems near known contamination sites."""
    # Place systems near known sites
    pwsids = [f"SYS{i:05d}" for i in range(10)]
    lats = [43.08, 38.82, 44.45, 35.69, 34.84, 44.83, 42.91, 43.17, 43.01, 40.74]
    lons = [-70.82, -104.70, -83.38, -117.69, -78.82, -92.94, -73.35, -85.59, -83.69, -74.17]

    systems = pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
        },
        index=pwsids,
    )
    systems.index.name = "pwsid"

    X = pd.DataFrame(
        np.random.RandomState(42).randn(10, 3),
        columns=["f1", "f2", "f3"],
        index=pwsids,
    )
    X.index.name = "pwsid"

    return systems, X


class TestFindNearestSystem:
    """Tests for _find_nearest_system."""

    def test_exact_match(self) -> None:
        lats = np.array([40.0, 41.0, 42.0])
        lons = np.array([-74.0, -73.0, -72.0])
        pwsids = np.array(["A", "B", "C"])

        pwsid, dist = _find_nearest_system(40.0, -74.0, lats, lons, pwsids)
        assert pwsid == "A"
        assert dist == pytest.approx(0.0, abs=0.1)

    def test_nearest_found(self) -> None:
        lats = np.array([40.0, 41.0, 42.0])
        lons = np.array([-74.0, -73.0, -72.0])
        pwsids = np.array(["A", "B", "C"])

        pwsid, dist = _find_nearest_system(40.1, -73.9, lats, lons, pwsids)
        assert pwsid == "A"
        assert dist > 0


class TestValidateAgainstKnownSites:
    """Tests for validate_against_known_sites."""

    def test_always_detect_model(self, systems_near_known_sites) -> None:
        systems, X = systems_near_known_sites
        model = _AlwaysDetectModel()

        report = validate_against_known_sites(model, X, systems)
        assert isinstance(report, ValidationReport)
        assert report.hit_rate > 0.0
        assert report.metadata["model_name"] == "always_detect"

    def test_never_detect_model_false_negatives(self, systems_near_known_sites) -> None:
        systems, X = systems_near_known_sites
        model = _NeverDetectModel()

        report = validate_against_known_sites(model, X, systems)
        assert len(report.false_negatives) > 0

    def test_empty_systems(self) -> None:
        model = _AlwaysDetectModel()
        systems = pd.DataFrame(
            {"latitude": pd.Series(dtype=float), "longitude": pd.Series(dtype=float)},
        )
        systems.index.name = "pwsid"
        X = pd.DataFrame()

        report = validate_against_known_sites(model, X, systems)
        assert report.hit_rate == 0.0
        assert len(report.false_negatives) == len(KNOWN_CONTAMINATION_SITES)

    def test_max_distance_filtering(self, systems_near_known_sites) -> None:
        systems, X = systems_near_known_sites
        model = _AlwaysDetectModel()

        # Very small radius → no matches
        report = validate_against_known_sites(model, X, systems, max_distance_km=0.001)
        assert all(r.matched_pwsid is None for r in report.site_results)

    def test_custom_known_sites(self, systems_near_known_sites) -> None:
        systems, X = systems_near_known_sites
        model = _AlwaysDetectModel()

        custom_sites = [
            {
                "name": "Test Site",
                "state": "NH",
                "lat": 43.08,
                "lon": -70.82,
                "contaminants": ["PFOS"],
                "expected_detection": True,
            }
        ]
        report = validate_against_known_sites(model, X, systems, known_sites=custom_sites)
        assert len(report.site_results) == 1
        assert report.site_results[0].is_hit is True

    def test_site_result_has_probability(self, systems_near_known_sites) -> None:
        systems, X = systems_near_known_sites
        model = _AlwaysDetectModel()

        custom_sites = [
            {
                "name": "Test",
                "state": "NH",
                "lat": 43.08,
                "lon": -70.82,
                "contaminants": ["PFOS"],
                "expected_detection": True,
            }
        ]
        report = validate_against_known_sites(model, X, systems, known_sites=custom_sites)
        result = report.site_results[0]
        assert result.predicted_probability is not None
        assert result.predicted_probability == pytest.approx(0.9)

    def test_empty_systems_filters_false_negatives_by_expected_detection(self) -> None:
        """Empty input only lists sites with expected_detection=True as false negatives."""
        model = _AlwaysDetectModel()
        systems = pd.DataFrame(
            {"latitude": pd.Series(dtype=float), "longitude": pd.Series(dtype=float)},
        )
        systems.index.name = "pwsid"
        X = pd.DataFrame()

        custom_sites = [
            {
                "name": "Expected Site",
                "state": "CA",
                "lat": 35.0,
                "lon": -118.0,
                "contaminants": ["PFOS"],
                "expected_detection": True,
            },
            {
                "name": "Not Expected Site",
                "state": "CA",
                "lat": 36.0,
                "lon": -119.0,
                "contaminants": ["PFOS"],
                "expected_detection": False,
            },
        ]
        report = validate_against_known_sites(model, X, systems, known_sites=custom_sites)
        assert report.false_negatives == ["Expected Site"]
        assert "Not Expected Site" not in report.false_negatives


class TestKnownContaminationSites:
    """Tests for the curated site list."""

    def test_sites_not_empty(self) -> None:
        assert len(KNOWN_CONTAMINATION_SITES) > 0

    def test_sites_have_required_fields(self) -> None:
        required = {"name", "state", "lat", "lon", "contaminants", "expected_detection"}
        for site in KNOWN_CONTAMINATION_SITES:
            assert required.issubset(site.keys()), f"Missing fields in {site['name']}"

    def test_sites_have_valid_coordinates(self) -> None:
        for site in KNOWN_CONTAMINATION_SITES:
            assert 24.0 < site["lat"] < 50.0, f"Bad lat for {site['name']}"
            assert -125.0 < site["lon"] < -66.0, f"Bad lon for {site['name']}"
