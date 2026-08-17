"""Tests for data.epa_frs — EPA Facility Registry Service loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest

from aquacontam.data.epa_frs import filter_by_type, load_frs

# Mock config matching configs/data.yaml epa_frs section
_MOCK_CONFIG = {
    "epa_frs": {
        "url": "https://example.com/frs.zip",
        "expected_files": ["NATIONAL_SINGLE.CSV"],
        "format": {"encoding": "latin-1"},
        "column_map": {
            "REGISTRY_ID": "registry_id",
            "PRIMARY_NAME": "facility_name",
            "LATITUDE83": "latitude",
            "LONGITUDE83": "longitude",
            "SIC_CODES": "sic_codes",
            "NAICS_CODES": "naics_codes",
            "STATE_CODE": "state_code",
            "INTEREST_TYPES": "interest_types",
        },
    }
}


@pytest.fixture()
def frs_csv(tmp_path: Path) -> Path:
    """Create a minimal FRS CSV with 4 facility types."""
    csv_path = tmp_path / "NATIONAL_SINGLE.CSV"
    df = pd.DataFrame(
        {
            "REGISTRY_ID": ["1001", "1002", "1003", "1004"],
            "PRIMARY_NAME": [
                "Chemical Plant",
                "Wastewater Facility",
                "Regional Airport",
                "City Landfill",
            ],
            "LATITUDE83": ["30.27", "30.28", "30.29", "30.30"],
            "LONGITUDE83": ["-97.74", "-97.73", "-97.72", "-97.71"],
            "SIC_CODES": ["2819", "4952", "4512", "4953"],
            "NAICS_CODES": ["", "221320", "481111", "562212"],
            "STATE_CODE": ["TX", "TX", "TX", "TX"],
            "INTEREST_TYPES": ["TRIS", "NPDES", "AIRPORT", "RCRAINFO"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


class TestLoadFrs:
    def test_loads_all_facilities(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            gdf = load_frs(frs_csv.parent)
        assert len(gdf) == 4
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs is not None

    def test_has_expected_columns(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            gdf = load_frs(frs_csv.parent)
        for col in ("facility_name", "latitude", "longitude", "sic_codes", "registry_id"):
            assert col in gdf.columns

    def test_filter_by_wwtp(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            gdf = load_frs(frs_csv.parent, facility_types=["wwtp"])
        assert len(gdf) >= 1
        assert "Wastewater Facility" in gdf["facility_name"].to_numpy()

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with (
            patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG),
            pytest.raises(FileNotFoundError, match="FRS data file not found"),
        ):
            load_frs(tmp_path)

    def test_drops_invalid_coords(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "NATIONAL_SINGLE.CSV"
        df = pd.DataFrame(
            {
                "REGISTRY_ID": ["1001", "1002"],
                "PRIMARY_NAME": ["Good", "Bad"],
                "LATITUDE83": ["30.27", ""],
                "LONGITUDE83": ["-97.74", ""],
                "SIC_CODES": ["2819", "2819"],
                "NAICS_CODES": ["", ""],
                "STATE_CODE": ["TX", "TX"],
                "INTEREST_TYPES": ["TRIS", "TRIS"],
            }
        )
        df.to_csv(csv_path, index=False)
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            gdf = load_frs(tmp_path)
        assert len(gdf) == 1


class TestFilterByType:
    def test_filter_matches_sic_code(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            all_fac = load_frs(frs_csv.parent)
        wwtp = filter_by_type(all_fac, ["wwtp"])
        assert len(wwtp) >= 1

    def test_filter_matches_naics_code(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            all_fac = load_frs(frs_csv.parent)
        landfill = filter_by_type(all_fac, ["landfill"])
        assert len(landfill) >= 1

    def test_filter_matches_interest_type(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            all_fac = load_frs(frs_csv.parent)
        industrial = filter_by_type(all_fac, ["industrial"])
        # Chemical Plant has SIC 2819 (industrial) and TRIS interest type
        assert len(industrial) >= 1

    def test_filter_empty_gdf(self) -> None:
        empty = gpd.GeoDataFrame()
        result = filter_by_type(empty, ["wwtp"])
        assert len(result) == 0

    def test_filter_multiple_types(self, frs_csv: Path) -> None:
        with patch("aquacontam.data.epa_frs.load_data_config", return_value=_MOCK_CONFIG):
            all_fac = load_frs(frs_csv.parent)
        combined = filter_by_type(all_fac, ["wwtp", "airport"])
        assert len(combined) >= 2
