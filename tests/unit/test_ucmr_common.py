"""Tests for shared UCMR parsing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aquacontam.data._ucmr_common import parse_ucmr_txt

# Minimal UCMR-like config for testing
_TEST_CONFIG: dict = {
    "format": {
        "delimiter": "\t",
        "encoding": "utf-8",
        "dtype_overrides": {"PWSID": "str"},
    },
    "column_map": {
        "PWSID": "pwsid",
        "Contaminant": "analyte",
        "AnalyticalResultValue": "concentration",
        "MRL": "detection_limit",
        "CollectionDate": "sample_date",
    },
    "derived_columns": {
        "censored": {
            "source_column": "AnalyticalResultsSign",
            "condition": "<",
        },
        "unit": "ug/L",
    },
    "extra_columns": ["PWSName", "Region"],
}

_SAMPLE_TSV = (
    "PWSID\tPWSName\tRegion\tContaminant\tAnalyticalResultValue\t"
    "AnalyticalResultsSign\tMRL\tCollectionDate\n"
    "CA0101001\tAcme Water\t9\tPFOS\t12.5\t=\t2.0\t01/15/2023\n"
    "TX0200002\tLone Star\t6\tPFOA\t\t<\t4.0\t02/20/2023\n"
    "MI0300003\tGreat Lakes\t5\tPFHxS\t8.3\t=\t2.0\t03/10/2023\n"
)


def _write_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "UCMR_Test.txt"
    p.write_text(_SAMPLE_TSV)
    return p


class TestParseUcmrTxt:
    """Tests for parse_ucmr_txt."""

    def test_parses_tsv(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        assert len(df) == 3

    def test_column_mapping(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert "concentration" in df.columns
        assert "PWSID" not in df.columns

    def test_censored_derivation(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        # Sorted by pwsid: CA0101001 (=), MI0300003 (=), TX0200002 (<)
        assert df["censored"].tolist() == [False, False, True]

    def test_date_parsing(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        assert pd.api.types.is_datetime64_any_dtype(df["sample_date"])

    def test_raw_prefix_on_extra_columns(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        assert "raw_PWSName" in df.columns
        assert "raw_Region" in df.columns
        assert "PWSName" not in df.columns

    def test_lat_lon_nan_placeholders(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        assert df["latitude"].isna().all()
        assert df["longitude"].isna().all()

    def test_unit_set(self, tmp_path: Path) -> None:
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        assert (df["unit"] == "ug/L").all()

    def test_censored_nan_concentration_filled(self, tmp_path: Path) -> None:
        """Censored rows with NaN concentration should be filled with 0."""
        path = _write_tsv(tmp_path)
        df = parse_ucmr_txt(path, _TEST_CONFIG)
        censored_row = df[df["censored"]]
        assert (censored_row["concentration"] == 0.0).all()
