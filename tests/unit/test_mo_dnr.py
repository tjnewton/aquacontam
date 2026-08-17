"""Tests for Missouri DNR data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data.mo_dnr import MoDnrSource


@pytest.fixture()
def mo_source(tmp_data_dirs: dict[str, Path]) -> MoDnrSource:
    return MoDnrSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_mo_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample MO DNR CSV file."""
    data = pd.DataFrame(
        {
            "PWSID": ["MO1010001", "MO1010001", "MO2020002"],
            "ANALYTE": ["PFOS", "PFOA", "PFBS"],
            "CONCENTRAT": [10.0, 0.0, 25.0],
            "CONCEN_UOM": ["NG/L", "NG/L", "NG/L"],
            "LESS_THAN_IND": ["", "<", ""],
            "DETECT_LMT": [2.0, 4.0, 2.0],
            "DET_LI_UOM": ["NG/L", "NG/L", "NG/L"],
            "COLLECT_DT": ["2023-06-15", "2023-06-15", "2023-07-20"],
            "LATITUDE": [38.63, 38.63, 37.21],
            "LONGITUDE": [-90.24, -90.24, -93.29],
        }
    )
    path = tmp_data_dirs["raw"] / "mo_dnr_pfas.csv"
    data.to_csv(path, index=False)
    return path


class TestMoDnrSource:
    """Tests for MoDnrSource."""

    def test_name_property(self, mo_source: MoDnrSource) -> None:
        assert mo_source.name == "mo_dnr"

    def test_config_loaded(self, mo_source: MoDnrSource) -> None:
        assert "arcgis_base_url" in mo_source._config
        assert "expected_files" in mo_source._config

    def test_parse_column_mapping(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        df = mo_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert "concentration" in df.columns
        assert "unit" in df.columns
        assert "censored" in df.columns
        assert "detection_limit" in df.columns
        assert "sample_date" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert len(df) == 3

    def test_ngl_to_ugl_conversion(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        """Concentrations should be converted from NG/L to ug/L."""
        df = mo_source.parse()
        pfos_row = df[df["analyte"] == "PFOS"].iloc[0]
        # 10.0 NG/L → 0.010 ug/L
        assert abs(pfos_row["concentration"] - 10.0 * PPT_TO_UGL) < 1e-9

    def test_censoring_less_than_ind(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        """LESS_THAN_IND '<' should mark row as censored."""
        df = mo_source.parse()
        pfoa_row = df[df["analyte"] == "PFOA"].iloc[0]
        assert bool(pfoa_row["censored"]) is True
        assert pfoa_row["concentration"] == 0.0

    def test_detected_not_censored(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        df = mo_source.parse()
        pfos_row = df[df["analyte"] == "PFOS"].iloc[0]
        assert bool(pfos_row["censored"]) is False
        assert pfos_row["concentration"] > 0

    def test_detection_limit_converted(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        """Detection limits should also be converted to ug/L."""
        df = mo_source.parse()
        pfoa_row = df[df["analyte"] == "PFOA"].iloc[0]
        # DL 4.0 NG/L → 0.004 ug/L
        assert abs(pfoa_row["detection_limit"] - 4.0 * PPT_TO_UGL) < 1e-9

    def test_coordinates_preserved(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        df = mo_source.parse()
        assert df["latitude"].notna().all()
        assert df["longitude"].notna().all()
        first = df.iloc[0]
        assert abs(first["latitude"] - 37.21) < 0.01 or abs(first["latitude"] - 38.63) < 0.01

    def test_sample_date_parsed(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        df = mo_source.parse()
        assert df["sample_date"].notna().all()
        assert pd.api.types.is_datetime64_any_dtype(df["sample_date"])

    def test_pwsid_normalization(
        self, mo_source: MoDnrSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Short PWSIDs should be zero-padded."""
        data = pd.DataFrame(
            {
                "PWSID": ["123", "MO45"],
                "ANALYTE": ["PFOS", "PFOA"],
                "CONCENTRAT": [10.0, 5.0],
                "CONCEN_UOM": ["NG/L", "NG/L"],
                "LESS_THAN_IND": ["", ""],
                "DETECT_LMT": [2.0, 2.0],
                "DET_LI_UOM": ["NG/L", "NG/L"],
                "COLLECT_DT": ["2023-06-15", "2023-06-15"],
                "LATITUDE": [38.63, 37.21],
                "LONGITUDE": [-90.24, -93.29],
            }
        )
        path = tmp_data_dirs["raw"] / "mo_dnr_pfas.csv"
        data.to_csv(path, index=False)

        df = mo_source.parse()
        pwsids = df["pwsid"].unique().tolist()
        assert "MO0000123" in pwsids
        assert "MO0000045" in pwsids
        for p in pwsids:
            assert " " not in p

    def test_validate_warns_unexpected_analyte(
        self, mo_source: MoDnrSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["MO1010001"],
                "analyte": ["UNKNOWN_PFAS"],
                "concentration": [1.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.5],
                "sample_date": pd.to_datetime(["2023-06-15"]),
                "latitude": [38.63],
                "longitude": [-90.24],
            }
        )
        with caplog.at_level("WARNING"):
            mo_source.validate(df)
        assert "Unexpected MO DNR analytes" in caplog.text

    def test_download_skip_existing(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        """Download should skip if file already exists."""
        result = mo_source.download(force=False)
        assert len(result) == 1
        assert result[0].exists()

    def test_to_parquet(self, mo_source: MoDnrSource, sample_mo_csv: Path) -> None:
        df = mo_source.parse()
        path = mo_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)

    def test_censoring_y_flag(
        self, mo_source: MoDnrSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """LESS_THAN_IND 'Y' should also be treated as censored."""
        data = pd.DataFrame(
            {
                "PWSID": ["MO1010001"],
                "ANALYTE": ["PFOS"],
                "CONCENTRAT": [5.0],
                "CONCEN_UOM": ["NG/L"],
                "LESS_THAN_IND": ["Y"],
                "DETECT_LMT": [10.0],
                "DET_LI_UOM": ["NG/L"],
                "COLLECT_DT": ["2023-06-15"],
                "LATITUDE": [38.63],
                "LONGITUDE": [-90.24],
            }
        )
        path = tmp_data_dirs["raw"] / "mo_dnr_pfas.csv"
        data.to_csv(path, index=False)

        df = mo_source.parse()
        assert bool(df.iloc[0]["censored"]) is True
        assert df.iloc[0]["concentration"] == 0.0
