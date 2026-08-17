"""Tests for NJ Private Well Testing Act data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _write_mock_pwta_csv(path: Path, n_rows: int = 20) -> None:
    """Create a mock NJ PWTA CSV file for testing."""
    rows = []
    for i in range(n_rows):
        detected = i % 3 == 0
        rows.append(
            {
                "SAMPLE_ID": f"PWTA{i:06d}",
                "LATITUDE": 40.0 + i * 0.01,
                "LONGITUDE": -74.5 + i * 0.01,
                "ANALYTE_NAME": "PFOS" if i % 2 == 0 else "PFOA",
                "RESULT_VALUE": str(5.0 + i) if detected else "0.0",
                "DETECTION_LIMIT": "2.0",
                "SAMPLE_DATE": "2023-06-15",
                "RESULT_QUALIFIER": "" if detected else "U",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestDownloadNjPrivateWells:
    """Tests for download_nj_private_wells()."""

    def test_download_raises_when_unavailable(self, tmp_path: Path) -> None:
        """Download should raise RuntimeError when source is marked unavailable."""
        from aquacontam.data.nj_private_wells import download_nj_private_wells

        with pytest.raises(RuntimeError, match="NJ private wells download skipped"):
            download_nj_private_wells(tmp_path)


class TestLoadNjPrivateWells:
    """Tests for load_nj_private_wells()."""

    def test_loads_with_valid_data(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        _write_mock_pwta_csv(csv_path)
        gdf = load_nj_private_wells(tmp_path)
        assert len(gdf) > 0

    def test_has_crs(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        _write_mock_pwta_csv(csv_path)
        gdf = load_nj_private_wells(tmp_path)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_has_required_columns(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        _write_mock_pwta_csv(csv_path)
        gdf = load_nj_private_wells(tmp_path)
        for col in ["sample_id", "analyte", "concentration", "censored"]:
            assert col in gdf.columns, f"Missing column: {col}"

    def test_censored_column_derived(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        _write_mock_pwta_csv(csv_path)
        gdf = load_nj_private_wells(tmp_path)
        assert gdf["censored"].dtype == bool
        # Some should be censored (result_qualifier="U"), some not
        assert gdf["censored"].any()
        assert not gdf["censored"].all()

    def test_filter_by_analyte(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        _write_mock_pwta_csv(csv_path)
        gdf = load_nj_private_wells(tmp_path, analytes=["PFOS"])
        assert (gdf["analyte"] == "PFOS").all()

    def test_file_not_found(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        with pytest.raises(FileNotFoundError):
            load_nj_private_wells(tmp_path)

    def test_handles_missing_coordinates(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        rows = [
            {
                "SAMPLE_ID": "PWTA000001",
                "LATITUDE": "",
                "LONGITUDE": "-74.5",
                "ANALYTE_NAME": "PFOS",
                "RESULT_VALUE": "5.0",
                "DETECTION_LIMIT": "2.0",
                "SAMPLE_DATE": "2023-06-15",
                "RESULT_QUALIFIER": "",
            },
            {
                "SAMPLE_ID": "PWTA000002",
                "LATITUDE": "40.0",
                "LONGITUDE": "-74.5",
                "ANALYTE_NAME": "PFOS",
                "RESULT_VALUE": "3.0",
                "DETECTION_LIMIT": "2.0",
                "SAMPLE_DATE": "2023-06-15",
                "RESULT_QUALIFIER": "",
            },
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        gdf = load_nj_private_wells(tmp_path)
        # Only the row with valid coordinates should remain
        assert len(gdf) == 1

    def test_concentration_is_numeric(self, tmp_path: Path) -> None:
        from aquacontam.data.nj_private_wells import load_nj_private_wells

        csv_path = tmp_path / "PWTA_Data.csv"
        _write_mock_pwta_csv(csv_path)
        gdf = load_nj_private_wells(tmp_path)
        assert pd.api.types.is_numeric_dtype(gdf["concentration"])
