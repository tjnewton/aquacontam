"""Tests for coordinate-based EPA region lookup."""

from __future__ import annotations

import pytest

from aquacontam.preprocessing._region_lookup import latlon_to_epa_region


class TestLatLonToEpaRegion:
    """Tests for latlon_to_epa_region."""

    @pytest.mark.parametrize(
        "lat, lon, expected_region",
        [
            (42.36, -71.06, 1),  # Boston, MA → Region 1
            (40.71, -74.01, 2),  # New York City → Region 2
            (38.91, -77.04, 3),  # Washington DC → Region 3
            (33.75, -84.39, 4),  # Atlanta, GA → Region 4
            (41.88, -87.63, 5),  # Chicago, IL → Region 5
            (30.27, -97.74, 6),  # Austin, TX → Region 6
            (39.10, -94.58, 7),  # Kansas City, MO → Region 7
            (39.74, -104.99, 8),  # Denver, CO → Region 8
            (34.05, -118.24, 9),  # Los Angeles, CA → Region 9
            (47.61, -122.33, 10),  # Seattle, WA → Region 10
        ],
    )
    def test_known_cities(self, lat: float, lon: float, expected_region: int) -> None:
        assert latlon_to_epa_region(lat, lon) == expected_region

    def test_outside_conus_returns_none(self) -> None:
        # Middle of Atlantic Ocean
        assert latlon_to_epa_region(35.0, -40.0) is None

    def test_returns_int_or_none(self) -> None:
        result = latlon_to_epa_region(40.0, -90.0)
        assert isinstance(result, int) or result is None
