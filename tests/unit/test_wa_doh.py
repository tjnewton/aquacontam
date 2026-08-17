"""Tests for Washington State DOH PFAS data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import PPT_TO_UGL
from aquacontam.data.wa_doh import WaDohSource, _clean_excel_id, _extract_analyte


@pytest.fixture()
def wa_source(tmp_data_dirs: dict[str, Path]) -> WaDohSource:
    return WaDohSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_wa_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample WA DOH PFAS CSV file."""
    data = pd.DataFrame(
        {
            "Water System ID": ['="60850"', '="60850"', '="66116"', '="66116"'],
            "Water System Name": [
                "North Offut Lake",
                "North Offut Lake",
                "Paradise Park",
                "Paradise Park",
            ],
            "Water Source": ['="01"', '="01"', '="03"', '="03"'],
            "County": ["Thurston", "Thurston", "Whatcom", "Whatcom"],
            "Testing Date": ["2021-10-11", "2021-10-11", "2021-12-14", "2021-12-14"],
            "Sample Number": ['="20663"', '="20663"', '="60777"', '="60777"'],
            "PFAS Measure": [
                "(PFBS) Perfluorobutanesulfonic acid",
                "(PFOS) Perfluorooctanesulfonic acid",
                "(PFBS) Perfluorobutanesulfonic acid",
                "(PFOA) Perfluorooctanoic acid",
            ],
            "Result": [
                "None Detected",
                "None Detected",
                "Detected but lower than state action level (SAL)",
                "Detected but lower than state action level (SAL)",
            ],
            "PFAS Level": ["None Detected", "None Detected", "2 ng/L", "3 ng/L"],
            "State Action Level (SAL)": ["345 ng/L", "70 ng/L", "345 ng/L", "10 ng/L"],
            "Date updated": ["2026-02-25", "2026-02-25", "2026-02-25", "2026-02-25"],
        }
    )
    path = tmp_data_dirs["raw"] / "wa_doh_pfas.csv"
    data.to_csv(path, index=False)
    return path


class TestWaDohSource:
    """Tests for WaDohSource."""

    def test_name_property(self, wa_source: WaDohSource) -> None:
        assert wa_source.name == "wa_doh"

    def test_config_loaded(self, wa_source: WaDohSource) -> None:
        assert "url" in wa_source._config
        assert "expected_files" in wa_source._config

    def test_parse_column_mapping(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        df = wa_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert "concentration" in df.columns
        assert "unit" in df.columns
        assert len(df) == 4

    def test_analyte_extraction(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """Analyte abbreviations should be extracted from parenthesized names."""
        df = wa_source.parse()
        analytes = set(df["analyte"].unique())
        assert "PFBS" in analytes
        assert "PFOS" in analytes
        assert "PFOA" in analytes

    def test_ngl_conversion(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """Detected values should be converted from ng/L to ug/L."""
        df = wa_source.parse()
        detected = df[~df["censored"]]
        assert len(detected) == 2
        # Paradise Park PFBS: 2 ng/L → 0.002 ug/L
        pfbs = detected[detected["analyte"] == "PFBS"].iloc[0]
        assert abs(pfbs["concentration"] - 2.0 * PPT_TO_UGL) < 1e-9

    def test_non_detect_censored(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """None Detected results should be censored."""
        df = wa_source.parse()
        censored = df[df["censored"]]
        assert len(censored) == 2
        assert (censored["concentration"] == 0.0).all()

    def test_excel_id_cleaning(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """Excel-escaped Water System IDs should be cleaned."""
        df = wa_source.parse()
        pwsids = set(df["pwsid"].unique())
        # "60850" → "WA0060850"
        assert "WA0060850" in pwsids
        assert "WA0066116" in pwsids

    def test_sample_date_parsed(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        df = wa_source.parse()
        assert df["sample_date"].notna().all()

    def test_no_coordinates(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """WA DOH CSV has no lat/lon columns."""
        df = wa_source.parse()
        assert df["latitude"].isna().all()
        assert df["longitude"].isna().all()

    def test_validate_no_coords(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """Validation should skip coord check."""
        df = wa_source.parse()
        result = wa_source.validate(df)
        assert len(result) > 0

    def test_to_parquet(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        df = wa_source.parse()
        path = wa_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)

    def test_download_skip_existing(self, wa_source: WaDohSource, sample_wa_csv: Path) -> None:
        """Download should skip if file already exists."""
        result = wa_source.download(force=False)
        assert len(result) == 1
        assert result[0].exists()


class TestHelpers:
    """Tests for helper functions."""

    def test_clean_excel_id(self) -> None:
        assert _clean_excel_id('="60850"') == "60850"
        assert _clean_excel_id("12345") == "12345"
        assert _clean_excel_id('  ="99"  ') == "99"

    def test_extract_analyte(self) -> None:
        assert _extract_analyte("(PFBS) Perfluorobutanesulfonic acid") == "PFBS"
        assert _extract_analyte("(PFOS) Perfluorooctanesulfonic acid") == "PFOS"
        assert _extract_analyte("(HFPO-DA) Hexafluoropropylene oxide") == "HFPO-DA"
        assert _extract_analyte("Unknown compound") == "Unknown compound"
