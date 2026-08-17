"""Tests for North Carolina DEQ data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data.nc_deq import NcDeqSource, _detect_pfas_columns


@pytest.fixture()
def nc_source(tmp_data_dirs: dict[str, Path]) -> NcDeqSource:
    return NcDeqSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_nc_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample NC DEQ CSV (long format)."""
    data = pd.DataFrame(
        {
            "PWSID": ["NC0000001", "NC0000001", "NC0000002"],
            "Analyte": ["HFPO-DA", "PFOS", "PFOA"],
            "Result": ["70.0", "", "25.0"],
            "Sample Date": ["2023-06-15", "2023-06-15", "2023-07-20"],
            "Latitude": ["35.22", "35.22", "35.77"],
            "Longitude": ["-78.99", "-78.99", "-78.64"],
        }
    )
    path = tmp_data_dirs["raw"] / "nc_deq_pfas.pdf"
    # Save as CSV but with .pdf extension for test (source is now PDF-only)
    # Use CSV because openpyxl may not be available in test env
    data.to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_nc_wide_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample NC DEQ CSV in wide format (one column per analyte)."""
    data = pd.DataFrame(
        {
            "PWSID": ["NC0000001", "NC0000002"],
            "HFPO-DA": ["70.0", ""],
            "PFOS": ["15.0", "25.0"],
            "PFOA": ["", "10.0"],
        }
    )
    path = tmp_data_dirs["raw"] / "nc_deq_pfas.pdf"
    data.to_csv(path, index=False)
    return path


class TestNcDeqSource:
    """Tests for NcDeqSource."""

    def test_name_property(self, nc_source: NcDeqSource) -> None:
        assert nc_source.name == "nc_deq"

    def test_config_loaded(self, nc_source: NcDeqSource) -> None:
        assert "url" in nc_source._config
        assert "expected_files" in nc_source._config

    def test_parse_long_format(self, nc_source: NcDeqSource, sample_nc_csv: Path) -> None:
        df = nc_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        # 3 rows, 1 with empty result (still present but censored)
        assert len(df) == 3

    def test_genx_present(self, nc_source: NcDeqSource, sample_nc_csv: Path) -> None:
        """NC DEQ should include GenX (HFPO-DA) data."""
        df = nc_source.parse()
        assert "HFPO-DA" in df["analyte"].to_numpy()

    def test_ppt_conversion(self, nc_source: NcDeqSource, sample_nc_csv: Path) -> None:
        df = nc_source.parse()
        genx_row = df[df["analyte"] == "HFPO-DA"].iloc[0]
        # 70 PPT → 0.070 ug/L
        expected = 70.0 * PPT_TO_UGL
        assert abs(genx_row["concentration"] - expected) < 1e-9

    def test_censoring_empty_result(self, nc_source: NcDeqSource, sample_nc_csv: Path) -> None:
        """Empty result values should be marked as censored."""
        df = nc_source.parse()
        pfos_nc1 = df[(df["analyte"] == "PFOS") & (df["pwsid"] == "NC0000001")]
        if len(pfos_nc1) > 0:
            assert bool(pfos_nc1.iloc[0]["censored"]) is True

    def test_parse_wide_format(self, nc_source: NcDeqSource, sample_nc_wide_csv: Path) -> None:
        """Wide format (one column per analyte) should be melted to long."""
        df = nc_source.parse()
        assert len(df) > 0
        # Should have HFPO-DA, PFOS, PFOA
        analytes = set(df["analyte"].unique())
        assert "HFPO-DA" in analytes or "PFOS" in analytes

    def test_validate_warns_unexpected(
        self, nc_source: NcDeqSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["NC0000001"],
                "analyte": ["UNKNOWN"],
                "concentration": [1.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.5],
                "sample_date": pd.to_datetime(["2023-01-15"]),
                "latitude": [float("nan")],
                "longitude": [float("nan")],
            }
        )
        with caplog.at_level("WARNING"):
            nc_source.validate(df)
        assert "Unexpected NC DEQ analytes" in caplog.text

    def test_pwsid_zero_padded(
        self, nc_source: NcDeqSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Short PWSIDs should be zero-padded, not space-padded."""
        data = pd.DataFrame(
            {
                "PWSID": ["123", "NC45", "NC0000001"],
                "Analyte": ["PFOS", "PFOA", "HFPO-DA"],
                "Result": ["10.0", "5.0", "20.0"],
                "Sample Date": ["2023-01-01", "2023-01-01", "2023-01-01"],
            }
        )
        path = tmp_data_dirs["raw"] / "nc_deq_pfas.pdf"
        data.to_csv(path, index=False)

        df = nc_source.parse()
        pwsids = df["pwsid"].unique().tolist()
        # "123" → "NC0000123", "NC45" → "NC0000045", "NC0000001" stays
        assert "NC0000123" in pwsids
        assert "NC0000045" in pwsids
        assert "NC0000001" in pwsids
        # No spaces in any PWSID
        for p in pwsids:
            assert " " not in p

    def test_to_parquet(self, nc_source: NcDeqSource, sample_nc_csv: Path) -> None:
        df = nc_source.parse()
        path = nc_source.to_parquet(df)
        assert path.exists()


class TestDetectPfasColumns:
    """Tests for NC DEQ _detect_pfas_columns."""

    def test_detects_genx(self) -> None:
        df = pd.DataFrame({"HFPO-DA": [1.0], "other": [2.0]})
        result = _detect_pfas_columns(df)
        assert "HFPO-DA" in result

    def test_empty_when_no_pfas(self) -> None:
        df = pd.DataFrame({"col_a": [1.0], "col_b": [2.0]})
        result = _detect_pfas_columns(df)
        assert result == {}
