"""Tests for Ohio EPA PFAS data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data.oh_epa import _ANALYTE_MAP, OhEpaSource


@pytest.fixture()
def oh_source(tmp_data_dirs: dict[str, Path]) -> OhEpaSource:
    return OhEpaSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_oh_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample OH EPA CSV file."""
    data = pd.DataFrame(
        {
            "pwsid": ["OH7762812", "OH7762812", "OH0300015", "OH0300015"],
            "sys_name": ["Manchester Umc", "Manchester Umc", "Akron WTP", "Akron WTP"],
            "samp_date": ["2020-02-13", "2020-02-13", "2021-06-01", "2021-06-01"],
            "analyte": [
                "PERFLUOROCTANOIC ACID (PFOA)",
                "PERFLUOROCTANE SULFONIC ACID (PFOS)",
                "PERFLUOROCTANOIC ACID (PFOA)",
                "PERFLUOROHEXANE SULFONIC ACID (PFHxS)",
            ],
            "lessthandetect": ["N", "N", "Y", "N"],
            "sample_result": ["08.50", "30.00", "02.00", "05.00"],
            "unit": ["NG/L", "NG/L", "NG/L", "NG/L"],
            "_latitude": [41.08, 41.08, 41.05, 41.05],
            "_longitude": [-81.52, -81.52, -81.51, -81.51],
        }
    )
    path = tmp_data_dirs["raw"] / "oh_epa_pfas.csv"
    data.to_csv(path, index=False)
    return path


class TestOhEpaSource:
    """Tests for OhEpaSource."""

    def test_name_property(self, oh_source: OhEpaSource) -> None:
        assert oh_source.name == "oh_epa"

    def test_config_loaded(self, oh_source: OhEpaSource) -> None:
        assert "arcgis_base_url" in oh_source._config
        assert "expected_files" in oh_source._config

    def test_parse_column_mapping(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        df = oh_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert "concentration" in df.columns
        assert "unit" in df.columns
        assert len(df) == 4

    def test_analyte_standardization(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        """Long analyte names should be mapped to standard abbreviations."""
        df = oh_source.parse()
        analytes = set(df["analyte"].unique())
        assert "PFOA" in analytes
        assert "PFOS" in analytes
        assert "PFHxS" in analytes

    def test_ngl_to_ugl_conversion(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        """Concentrations should be converted from ng/L to ug/L."""
        df = oh_source.parse()
        pfoa_det = df[(df["analyte"] == "PFOA") & (~df["censored"])]
        assert len(pfoa_det) > 0
        # 8.50 ng/L → 0.0085 ug/L
        row = pfoa_det.iloc[0]
        assert abs(row["concentration"] - 8.50 * PPT_TO_UGL) < 1e-9

    def test_lessthandetect_censoring(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        """lessthandetect='Y' rows should be censored."""
        df = oh_source.parse()
        censored = df[df["censored"]]
        assert len(censored) == 1
        assert censored.iloc[0]["concentration"] == 0.0

    def test_detected_not_censored(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        """lessthandetect='N' rows should not be censored."""
        df = oh_source.parse()
        detected = df[~df["censored"]]
        assert len(detected) == 3

    def test_coordinates_preserved(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        df = oh_source.parse()
        assert df["latitude"].notna().all()
        assert df["longitude"].notna().all()

    def test_sample_date_parsed(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        df = oh_source.parse()
        assert df["sample_date"].notna().all()

    def test_pwsid_preserved(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        """OH PWSIDs should be preserved as-is."""
        df = oh_source.parse()
        pwsids = set(df["pwsid"].unique())
        assert "OH7762812" in pwsids
        assert "OH0300015" in pwsids

    def test_validate_warns_unexpected_analyte(
        self, oh_source: OhEpaSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["OH0000001"],
                "analyte": ["UNKNOWN"],
                "concentration": [1.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.5],
                "sample_date": pd.to_datetime(["2023-01-15"]),
                "latitude": [41.0],
                "longitude": [-81.5],
            }
        )
        with caplog.at_level("WARNING"):
            oh_source.validate(df)
        assert "Unexpected OH EPA analytes" in caplog.text

    def test_to_parquet(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        df = oh_source.parse()
        path = oh_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)

    def test_download_skip_existing(self, oh_source: OhEpaSource, sample_oh_csv: Path) -> None:
        """Download should skip if file already exists."""
        result = oh_source.download(force=False)
        assert len(result) == 1
        assert result[0].exists()


class TestAnalyteMap:
    """Tests for the analyte name mapping."""

    def test_all_standard_names_present(self) -> None:
        expected = {"PFOA", "PFOS", "PFHxS", "PFBS", "PFNA", "HFPO-DA"}
        assert expected == set(_ANALYTE_MAP.values())
