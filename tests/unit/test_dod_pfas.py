"""Tests for DoD PFAS site data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam.data.dod_pfas import load_dod_pfas


@pytest.fixture()
def sample_dod_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample DoD PFAS sites CSV file."""
    data = pd.DataFrame(
        {
            "site_name": ["FORT BRAGG", "WRIGHT-PATTERSON AFB", "NASA LANGLEY", "CAMP LEJEUNE"],
            "agency": ["Army", "Air Force", "NASA", "Navy"],
            "state": ["NC", "OH", "VA", "NC"],
            "latitude": [35.14, 39.82, 37.08, 34.62],
            "longitude": [-79.00, -84.05, -76.37, -77.36],
            "pfas_presence": [
                "Known Detection",
                "Known Detection",
                "Known Detection",
                "Known Detection",
            ],
            "property_type": ["Active", "Active", "Active", "Active"],
            "cleanup_status": [
                "PA/SI Completed - RI Underway",
                "PA/SI Completed",
                "PA/SI Completed",
                "RI Completed",
            ],
            "is_dod": [True, True, False, True],
        }
    )
    path = tmp_data_dirs["raw"] / "dod_pfas_sites.csv"
    data.to_csv(path, index=False)
    return path


class TestLoadDodPfas:
    """Tests for load_dod_pfas."""

    def test_loads_csv(self, sample_dod_csv: Path, tmp_data_dirs: dict[str, Path]) -> None:
        df = load_dod_pfas(tmp_data_dirs["raw"], dod_only=False)
        assert len(df) == 4
        assert "site_name" in df.columns
        assert "agency" in df.columns
        assert "latitude" in df.columns

    def test_dod_only_filter(self, sample_dod_csv: Path, tmp_data_dirs: dict[str, Path]) -> None:
        """dod_only=True should exclude NASA."""
        df = load_dod_pfas(tmp_data_dirs["raw"], dod_only=True)
        assert len(df) == 3
        assert "NASA" not in df["agency"].to_numpy()

    def test_all_agencies(self, sample_dod_csv: Path, tmp_data_dirs: dict[str, Path]) -> None:
        """dod_only=False should include all agencies."""
        df = load_dod_pfas(tmp_data_dirs["raw"], dod_only=False)
        assert "NASA" in df["agency"].to_numpy()

    def test_conus_filter(self, tmp_data_dirs: dict[str, Path]) -> None:
        """Records outside CONUS should be dropped."""
        data = pd.DataFrame(
            {
                "site_name": ["IN_CONUS", "ALASKA_SITE"],
                "agency": ["Army", "Air Force"],
                "state": ["OH", "AK"],
                "latitude": [41.0, 61.0],
                "longitude": [-83.0, -150.0],
                "pfas_presence": ["Known Detection", "Known Detection"],
                "property_type": ["Active", "Active"],
                "cleanup_status": ["PA/SI Completed", "PA/SI Completed"],
                "is_dod": [True, True],
            }
        )
        path = tmp_data_dirs["raw"] / "dod_pfas_sites.csv"
        data.to_csv(path, index=False)

        df = load_dod_pfas(tmp_data_dirs["raw"])
        assert len(df) == 1
        assert df.iloc[0]["site_name"] == "IN_CONUS"

    def test_file_not_found(self, tmp_data_dirs: dict[str, Path]) -> None:
        with pytest.raises(FileNotFoundError, match="Run download_dod_pfas"):
            load_dod_pfas(tmp_data_dirs["raw"])
