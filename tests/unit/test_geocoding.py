"""Tests for ZIP code centroid geocoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquacontam.geo.geocoding import (
    _haversine_km,
    compute_coord_source_stats,
    geocode_by_zipcode,
)


@pytest.fixture
def zcta_centroids() -> pd.DataFrame:
    """Fake ZCTA centroid lookup table."""
    return pd.DataFrame(
        {
            "latitude": [40.7128, 34.0522, 41.8781, 29.7604],
            "longitude": [-74.0060, -118.2437, -87.6298, -95.3698],
        },
        index=pd.Index(["10001", "90210", "60601", "77001"], name="zipcode"),
    )


def test_geocode_fills_missing_coords(zcta_centroids: pd.DataFrame) -> None:
    """Rows with NaN lat/lon should get ZIP centroid coordinates."""
    df = pd.DataFrame(
        {
            "pwsid": ["NY0100001", "CA0100001", "IL0100001"],
            "latitude": [np.nan, np.nan, np.nan],
            "longitude": [np.nan, np.nan, np.nan],
            "raw_ZipCode": ["10001", "90210", "60601"],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    assert result["latitude"].notna().all()
    assert result["longitude"].notna().all()
    assert (result["coord_source"] == "zip_centroid").all()
    assert result.loc[0, "latitude"] == pytest.approx(40.7128)
    assert result.loc[0, "longitude"] == pytest.approx(-74.0060)


def test_geocode_preserves_existing_coords(zcta_centroids: pd.DataFrame) -> None:
    """Rows with existing coordinates should not be overwritten."""
    df = pd.DataFrame(
        {
            "pwsid": ["CA0100001"],
            "latitude": [34.0],
            "longitude": [-118.0],
            "raw_ZipCode": ["90210"],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    assert result.loc[0, "latitude"] == 34.0
    assert result.loc[0, "longitude"] == -118.0
    assert result.loc[0, "coord_source"] == "original"


def test_geocode_handles_unmatched_zips(zcta_centroids: pd.DataFrame) -> None:
    """Unmatched ZIP codes should leave coordinates as NaN."""
    df = pd.DataFrame(
        {
            "pwsid": ["XX0100001"],
            "latitude": [np.nan],
            "longitude": [np.nan],
            "raw_ZipCode": ["99999"],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    assert pd.isna(result.loc[0, "latitude"])
    assert pd.isna(result.loc[0, "longitude"])


def test_geocode_handles_missing_zip_col(zcta_centroids: pd.DataFrame) -> None:
    """If ZIP column doesn't exist, return df unchanged."""
    df = pd.DataFrame(
        {
            "pwsid": ["NY0100001"],
            "latitude": [np.nan],
            "longitude": [np.nan],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    assert "coord_source" not in result.columns
    assert pd.isna(result.loc[0, "latitude"])


def test_geocode_handles_short_zip_codes(zcta_centroids: pd.DataFrame) -> None:
    """Short ZIP codes should be zero-padded to 5 digits."""
    df = pd.DataFrame(
        {
            "pwsid": ["NY0100001"],
            "latitude": [np.nan],
            "longitude": [np.nan],
            "raw_ZipCode": ["1"],  # Should pad to "00001" (no match)
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    assert pd.isna(result.loc[0, "latitude"])


def test_geocode_handles_nan_zip(zcta_centroids: pd.DataFrame) -> None:
    """NaN ZIP codes should not cause errors."""
    df = pd.DataFrame(
        {
            "pwsid": ["NY0100001"],
            "latitude": [np.nan],
            "longitude": [np.nan],
            "raw_ZipCode": [np.nan],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    assert pd.isna(result.loc[0, "latitude"])


def test_geocode_handles_zip_plus4(zcta_centroids: pd.DataFrame) -> None:
    """ZIP+4 codes (e.g. '75801-2733') should be stripped to 5 digits."""
    df = pd.DataFrame(
        {
            "pwsid": ["TX0100001", "TX0100002", "MO0100001"],
            "latitude": [np.nan, np.nan, np.nan],
            "longitude": [np.nan, np.nan, np.nan],
            "raw_ZipCode": ["77001-2733", "10001-0001", "90210-1234"],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    # All three should be geocoded after stripping ZIP+4 suffix
    assert result["latitude"].notna().all()
    assert result["longitude"].notna().all()
    assert (result["coord_source"] == "zip_centroid").all()
    # Verify correct centroid lookup
    assert result.loc[0, "latitude"] == pytest.approx(29.7604)  # 77001
    assert result.loc[1, "latitude"] == pytest.approx(40.7128)  # 10001
    assert result.loc[2, "latitude"] == pytest.approx(34.0522)  # 90210


def test_geocode_mixed_coords_and_zips(zcta_centroids: pd.DataFrame) -> None:
    """Mix of existing coords, geocodable ZIPs, and unmatched ZIPs."""
    df = pd.DataFrame(
        {
            "pwsid": ["A", "B", "C"],
            "latitude": [40.0, np.nan, np.nan],
            "longitude": [-74.0, np.nan, np.nan],
            "raw_ZipCode": ["10001", "90210", "99999"],
        }
    )
    result = geocode_by_zipcode(df, zcta_centroids, zip_col="raw_ZipCode")

    # A: existing coords preserved
    assert result.loc[0, "latitude"] == 40.0
    assert result.loc[0, "coord_source"] == "original"
    # B: geocoded from ZIP
    assert result.loc[1, "latitude"] == pytest.approx(34.0522)
    assert result.loc[1, "coord_source"] == "zip_centroid"
    # C: unmatched ZIP, still NaN
    assert pd.isna(result.loc[2, "latitude"])


class TestHaversineKm:
    """Tests for _haversine_km vectorized distance."""

    def test_known_distance(self) -> None:
        """NYC to LA is approximately 3,944 km."""
        lat1 = pd.Series([40.7128])
        lon1 = pd.Series([-74.0060])
        lat2 = pd.Series([34.0522])
        lon2 = pd.Series([-118.2437])
        d = _haversine_km(lat1, lon1, lat2, lon2)
        assert d.iloc[0] == pytest.approx(3944, abs=50)

    def test_zero_distance(self) -> None:
        lat = pd.Series([40.0])
        lon = pd.Series([-74.0])
        d = _haversine_km(lat, lon, lat, lon)
        assert d.iloc[0] == pytest.approx(0.0, abs=0.001)

    def test_vectorized(self) -> None:
        lat1 = pd.Series([40.7128, 41.8781])
        lon1 = pd.Series([-74.0060, -87.6298])
        lat2 = pd.Series([34.0522, 29.7604])
        lon2 = pd.Series([-118.2437, -95.3698])
        d = _haversine_km(lat1, lon1, lat2, lon2)
        assert len(d) == 2
        assert d.iloc[0] > 3000  # NYC-LA
        assert d.iloc[1] > 1000  # Chicago-Houston


class TestComputeCoordSourceStats:
    """Tests for compute_coord_source_stats."""

    @pytest.fixture
    def centroids_with_intpt(self) -> pd.DataFrame:
        """ZCTA centroids with INTPTLAT/INTPTLONG columns."""
        return pd.DataFrame(
            {
                "INTPTLAT": [40.7500, 34.0000],
                "INTPTLONG": [-73.9900, -118.2000],
            },
            index=pd.Index(["10001", "90210"], name="zipcode"),
        )

    def test_counts(self, centroids_with_intpt: pd.DataFrame) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "B", "C", "D"],
                "coord_source": ["original", "original", "zip_centroid", "zip_centroid"],
                "latitude": [40.71, 34.05, 40.75, 34.00],
                "longitude": [-74.01, -118.24, -73.99, -118.20],
                "raw_ZipCode": ["10001", "90210", "10001", "90210"],
            }
        )
        result = compute_coord_source_stats(df, centroids_with_intpt)
        assert result["system_counts"]["original"] == 2
        assert result["system_counts"]["zip_centroid"] == 2
        assert result["row_counts"]["original"] == 2
        assert result["row_counts"]["zip_centroid"] == 2

    def test_imprecision_computed(self, centroids_with_intpt: pd.DataFrame) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "B"],
                "coord_source": ["original", "original"],
                "latitude": [40.7128, 34.0522],
                "longitude": [-74.0060, -118.2437],
                "raw_ZipCode": ["10001", "90210"],
            }
        )
        result = compute_coord_source_stats(df, centroids_with_intpt)
        imp = result["centroid_imprecision_km"]
        assert imp["n_systems"] == 2
        assert imp["median"] > 0
        assert imp["mean"] > 0

    def test_no_original_coords(self, centroids_with_intpt: pd.DataFrame) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A"],
                "coord_source": ["zip_centroid"],
                "latitude": [40.75],
                "longitude": [-73.99],
                "raw_ZipCode": ["10001"],
            }
        )
        result = compute_coord_source_stats(df, centroids_with_intpt)
        assert result["system_counts"]["original"] == 0
        assert result["centroid_imprecision_km"] == {}

    def test_no_coord_source_column(self, centroids_with_intpt: pd.DataFrame) -> None:
        df = pd.DataFrame({"pwsid": ["A"], "latitude": [40.0], "longitude": [-74.0]})
        result = compute_coord_source_stats(df, centroids_with_intpt)
        assert result["system_counts"] == {}

    def test_missing_coord_source_counted(self, centroids_with_intpt: pd.DataFrame) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["A", "B"],
                "coord_source": ["original", ""],
                "latitude": [40.71, np.nan],
                "longitude": [-74.01, np.nan],
                "raw_ZipCode": ["10001", "90210"],
            }
        )
        result = compute_coord_source_stats(df, centroids_with_intpt)
        assert result["system_counts"]["missing"] == 1
