"""Tests for TRI PFAS data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam.data.tri import load_tri_pfas


@pytest.fixture()
def sample_tri_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample TRI PFAS CSV file."""
    data = pd.DataFrame(
        {
            "frs_id": ["110002464182", "110000437885", "110000437885"],
            "facility_name": ["ACME CHEMICALS", "BASE AIR FORCE", "BASE AIR FORCE"],
            "city": ["TOLEDO", "DAYTON", "DAYTON"],
            "state": ["OH", "OH", "OH"],
            "latitude": [41.65, 39.78, 39.78],
            "longitude": [-83.54, -84.19, -84.19],
            "chemical": [
                "Perfluorooctanoic acid",
                "Perfluorooctane sulfonic acid",
                "Hexafluoropropylene oxide dimer acid",
            ],
            "cas_number": ["335-67-1", "1763-23-1", "13252-13-6"],
            "year": [2023, 2023, 2023],
            "onsite_release_lb": [10.5, 5.2, 3.1],
            "water_release_lb": [2.0, 1.1, 0.5],
            "potw_transfer_lb": [0.0, 0.0, 0.0],
        }
    )
    path = tmp_data_dirs["raw"] / "tri_pfas_facilities.csv"
    data.to_csv(path, index=False)
    return path


class TestLoadTriPfas:
    """Tests for load_tri_pfas."""

    def test_loads_csv(self, sample_tri_csv: Path, tmp_data_dirs: dict[str, Path]) -> None:
        df = load_tri_pfas(tmp_data_dirs["raw"])
        assert len(df) == 3
        assert "frs_id" in df.columns
        assert "facility_name" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns

    def test_conus_filter(self, tmp_data_dirs: dict[str, Path]) -> None:
        """Records outside CONUS should be dropped."""
        data = pd.DataFrame(
            {
                "frs_id": ["1", "2"],
                "facility_name": ["IN_CONUS", "OUTSIDE"],
                "latitude": [41.0, 60.0],
                "longitude": [-83.0, -150.0],
                "chemical": ["PFOA", "PFOS"],
                "year": [2023, 2023],
                "onsite_release_lb": [10.0, 5.0],
                "water_release_lb": [1.0, 0.5],
                "potw_transfer_lb": [0.0, 0.0],
            }
        )
        path = tmp_data_dirs["raw"] / "tri_pfas_facilities.csv"
        data.to_csv(path, index=False)

        df = load_tri_pfas(tmp_data_dirs["raw"])
        assert len(df) == 1
        assert df.iloc[0]["facility_name"] == "IN_CONUS"

    def test_file_not_found(self, tmp_data_dirs: dict[str, Path]) -> None:
        with pytest.raises(FileNotFoundError, match="Run download_tri_pfas"):
            load_tri_pfas(tmp_data_dirs["raw"])

    def test_release_quantities(
        self, sample_tri_csv: Path, tmp_data_dirs: dict[str, Path]
    ) -> None:
        df = load_tri_pfas(tmp_data_dirs["raw"])
        assert df["onsite_release_lb"].sum() > 0
        assert df["water_release_lb"].sum() > 0
