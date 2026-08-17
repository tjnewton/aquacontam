"""Tests for New Jersey DEP data source (waterviewer.nj.gov API)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aquacontam._constants import NJ_DEP_ANALYTE_CODES, NJ_DEP_ANALYTES, PPT_TO_UGL
from aquacontam.data.nj_dep import NjDepSource, _normalize_nj_analyte


@pytest.fixture()
def nj_source(tmp_data_dirs: dict[str, Path]) -> NjDepSource:
    return NjDepSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_inventory_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample NJ DEP inventory CSV (waterviewer format)."""
    df = pd.DataFrame(
        {
            "TINWSYS_IS_NUMBER": [1, 2, 3],
            "TINWSYS_ST_CODE": ["NJ", "NJ", "NJ"],
            "NUMBER0": ["NJ0101001   ", "NJ0200002   ", "NJ0300003   "],
            "NAME": ["Smalltown Water Co.", "Bigcity MUA", "Coastal WD"],
            "ACTIVITY_STATUS_CD": ["A", "A", "I"],
            "D_PRIN_CNTY_SVD_NM": ["ATLANTIC", "BERGEN", "CAPE MAY"],
            "D_POPULATION_COUNT": ["1000", "50000", "500"],
        }
    )
    csv_path = tmp_data_dirs["raw"] / "nj_dep_inventory.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def sample_facilities_csv(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample NJ DEP facilities CSV with coordinates."""
    df = pd.DataFrame(
        {
            "_PWSID": ["NJ0101001   ", "NJ0200002   "],
            "TINWSF_IS_NUMBER": [10, 20],
            "ST_ASGN_IDENT_CD": ["001", "002"],
            "TYPE_CODE": ["WL", "TP"],
            "WATER_TYPE_CODE": ["GW", "SW"],
            "LATITUDE_MEASURE": [40.73, 40.50],
            "LONGITUDE_MEASURE": [-74.17, -74.41],
        }
    )
    csv_path = tmp_data_dirs["raw"] / "nj_dep_facilities.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def sample_manual_export(tmp_data_dirs: dict[str, Path]) -> Path:
    """Create a sample manual export CSV matching waterviewer grid columns."""
    df = pd.DataFrame(
        {
            "Water System ID": ["NJ0101001", "NJ0101001", "NJ0200002"],
            "Water System Name": ["Smalltown Water Co.", "Smalltown Water Co.", "Bigcity MUA"],
            "Analyte Code": [2806, 2805, 2804],
            "Analyte Name": ["PFOA", "PFOS", "PFNA"],
            "Concentration": [14.0, 0.0, 13.0],
            "Detection Value": [2.0, 2.0, 2.0],
            "Collection Date": ["2023-01-15", "2023-01-15", "2023-02-20"],
        }
    )
    csv_path = tmp_data_dirs["raw"] / "nj_dep_pfas_export.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestNjDepSource:
    """Tests for NjDepSource."""

    def test_name_property(self, nj_source: NjDepSource) -> None:
        assert nj_source.name == "nj_dep"

    def test_config_loaded(self, nj_source: NjDepSource) -> None:
        assert "base_url" in nj_source._config
        assert "expected_files" in nj_source._config
        assert "inventory_endpoint" in nj_source._config

    def test_parse_inventory_only(
        self,
        nj_source: NjDepSource,
        sample_inventory_csv: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Parse should return inventory records with coords when no manual export."""
        df = nj_source.parse()
        assert "pwsid" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        # 3 systems in inventory
        assert len(df) == 3

    def test_inventory_coords_merged(
        self,
        nj_source: NjDepSource,
        sample_inventory_csv: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Facility coordinates should be merged into inventory records."""
        df = nj_source.parse()
        sys1 = df[df["pwsid"] == "NJ0101001"].iloc[0]
        assert abs(sys1["latitude"] - 40.73) < 0.01
        assert abs(sys1["longitude"] - (-74.17)) < 0.01

    def test_inventory_missing_coords(
        self,
        nj_source: NjDepSource,
        sample_inventory_csv: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Systems without facility data should have NaN coordinates."""
        df = nj_source.parse()
        sys3 = df[df["pwsid"] == "NJ0300003"].iloc[0]
        assert pd.isna(sys3["latitude"])
        assert pd.isna(sys3["longitude"])

    def test_parse_manual_export(
        self,
        nj_source: NjDepSource,
        sample_manual_export: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Parse should prefer manual export file when present."""
        df = nj_source.parse()
        assert "pwsid" in df.columns
        assert "analyte" in df.columns
        assert len(df) == 3  # 3 sample rows

    def test_manual_export_analytes(
        self,
        nj_source: NjDepSource,
        sample_manual_export: Path,
        sample_facilities_csv: Path,
    ) -> None:
        df = nj_source.parse()
        analytes = set(df["analyte"].unique())
        assert analytes == {"PFOA", "PFOS", "PFNA"}

    def test_manual_export_ppt_conversion(
        self,
        nj_source: NjDepSource,
        sample_manual_export: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Concentrations should be converted from PPT to ug/L."""
        df = nj_source.parse()
        pfoa = df[df["analyte"] == "PFOA"].iloc[0]
        # 14 PPT -> 0.014 ug/L
        assert abs(pfoa["concentration"] - 14.0 * PPT_TO_UGL) < 1e-9

    def test_manual_export_censoring(
        self,
        nj_source: NjDepSource,
        sample_manual_export: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Zero concentration should be flagged as censored."""
        df = nj_source.parse()
        pfos = df[df["analyte"] == "PFOS"].iloc[0]
        assert pfos["censored"] is True or bool(pfos["censored"])
        assert pfos["concentration"] == 0.0

    def test_manual_export_coords_from_facilities(
        self,
        nj_source: NjDepSource,
        sample_manual_export: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Coordinates should be merged from facility file."""
        df = nj_source.parse()
        sys1 = df[df["pwsid"] == "NJ0101001"].iloc[0]
        assert abs(sys1["latitude"] - 40.73) < 0.01

    def test_validate_warns_unexpected_analyte(
        self, nj_source: NjDepSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["NJ0101001"],
                "analyte": ["UNKNOWN_CHEM"],
                "concentration": [1.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [0.5],
                "sample_date": pd.to_datetime(["2023-01-15"]),
                "latitude": [40.73],
                "longitude": [-74.17],
            }
        )
        with caplog.at_level("WARNING"):
            nj_source.validate(df)
        assert "Unexpected NJ DEP analytes" in caplog.text

    def test_pwsid_zero_padded(
        self, nj_source: NjDepSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Short PWSIDs in manual export should be zero-padded."""
        df = pd.DataFrame(
            {
                "Water System ID": ["123", "NJ45", "NJ0101001"],
                "Analyte Code": [2806, 2805, 2804],
                "Analyte Name": ["PFOA", "PFOS", "PFNA"],
                "Concentration": [10.0, 5.0, 8.0],
                "Detection Value": [2.0, 2.0, 2.0],
                "Collection Date": ["2023-01-15", "2023-02-20", "2023-03-10"],
            }
        )
        csv_path = tmp_data_dirs["raw"] / "nj_dep_pfas_export.csv"
        df.to_csv(csv_path, index=False)
        # Also create empty facilities file
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "nj_dep_facilities.csv", index=False)

        result = nj_source.parse()
        pwsids = result["pwsid"].unique().tolist()
        assert "NJ0000123" in pwsids
        assert "NJ0000045" in pwsids
        assert "NJ0101001" in pwsids
        for p in pwsids:
            assert " " not in p

    def test_to_parquet(
        self,
        nj_source: NjDepSource,
        sample_manual_export: Path,
        sample_facilities_csv: Path,
    ) -> None:
        df = nj_source.parse()
        path = nj_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)

    def test_download_skip_existing(
        self,
        nj_source: NjDepSource,
        sample_inventory_csv: Path,
        sample_facilities_csv: Path,
    ) -> None:
        """Download should skip if files already exist."""
        result = nj_source.download(force=False)
        assert len(result) == 2
        assert all(p.exists() for p in result)

    def test_parse_empty_inventory(
        self, nj_source: NjDepSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Empty inventory should return empty DataFrame with correct schema."""
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "nj_dep_inventory.csv", index=False)
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "nj_dep_facilities.csv", index=False)
        df = nj_source.parse()
        assert len(df) == 0
        assert "pwsid" in df.columns
        assert "analyte" in df.columns

    def test_summary_codes_excluded(
        self, nj_source: NjDepSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Summary analyte codes (PFAS RULE, TOTAL PFOA AND PFOS) should be excluded."""
        df = pd.DataFrame(
            {
                "Water System ID": ["NJ0101001", "NJ0101001", "NJ0101001"],
                "Analyte Code": [2806, 2830, 2840],
                "Analyte Name": ["PFOA", "TOTAL PFOA AND PFOS", "HAZARD INDEX PFAS"],
                "Concentration": [14.0, 28.0, 0.5],
                "Detection Value": [2.0, 2.0, 0.0],
                "Collection Date": ["2023-01-15", "2023-01-15", "2023-01-15"],
            }
        )
        csv_path = tmp_data_dirs["raw"] / "nj_dep_pfas_export.csv"
        df.to_csv(csv_path, index=False)
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "nj_dep_facilities.csv", index=False)

        result = nj_source.parse()
        assert len(result) == 1
        assert result.iloc[0]["analyte"] == "PFOA"


class TestNormalizeNjAnalyte:
    """Tests for _normalize_nj_analyte helper."""

    def test_short_code_passthrough(self) -> None:
        assert _normalize_nj_analyte("PFOA") == "PFOA"
        assert _normalize_nj_analyte("PFOS") == "PFOS"
        assert _normalize_nj_analyte("HFPO-DA") == "HFPO-DA"

    def test_long_name_normalization(self) -> None:
        assert _normalize_nj_analyte("Perfluorooctanoic acid") == "PFOA"
        assert _normalize_nj_analyte("PERFLUOROOCTANE SULFONIC ACID") == "PFOS"

    def test_parenthesized_abbreviation(self) -> None:
        assert _normalize_nj_analyte("Perfluorooctanoic acid (PFOA)") == "PFOA"

    def test_unknown_passthrough(self) -> None:
        assert _normalize_nj_analyte("UNKNOWN_CHEM") == "UNKNOWN_CHEM"


class TestNjDepAnalyteCodes:
    """Tests for NJ DEP analyte code constants."""

    def test_all_analytes_have_codes(self) -> None:
        """Every NJ_DEP_ANALYTES entry should appear in NJ_DEP_ANALYTE_CODES values."""
        code_values = set(NJ_DEP_ANALYTE_CODES.values())
        for analyte in NJ_DEP_ANALYTES:
            assert analyte in code_values, f"{analyte} not in NJ_DEP_ANALYTE_CODES"

    def test_code_count(self) -> None:
        """Should have 28 analyte codes (30 codes including 2 summary/composite)."""
        # 25 individual PFAS + 3 summary codes (2800, 2830, 2840) = 28 total
        assert len(NJ_DEP_ANALYTE_CODES) == 28

    def test_known_codes(self) -> None:
        assert NJ_DEP_ANALYTE_CODES[2806] == "PFOA"
        assert NJ_DEP_ANALYTE_CODES[2805] == "PFOS"
        assert NJ_DEP_ANALYTE_CODES[2804] == "PFNA"
        assert NJ_DEP_ANALYTE_CODES[2816] == "HFPO-DA"


class TestScraperHelpers:
    """Tests for scrape_nj_dep.py helper functions (no Playwright required)."""

    def test_build_api_url(self) -> None:
        """API URL should include OData params and date range."""
        from scripts.scrape_nj_dep import _build_api_url

        url = _build_api_url("2024-01-01", "2026-01-01", offset=5000, page_size=100)
        assert "$skip=5000" in url
        assert "$top=100" in url
        assert "BeginDate=2024-01-01" in url
        assert "AnalyteType=OC" in url

    def test_fetch_page_in_browser_full_page(self) -> None:
        """Full page should return records, count, and has_more=True."""
        from scripts.scrape_nj_dep import _fetch_page_in_browser

        mock_page = MagicMock()
        mock_page.evaluate.return_value = {
            "@odata.count": 500,
            "value": [{"id": i} for i in range(100)],
        }

        url = "/sdwis/SamplesSearchResults?$skip=0&$top=100&$count=true"
        records, count, has_more = _fetch_page_in_browser(mock_page, "xsrf123", url)

        assert len(records) == 100
        assert count == 500
        assert has_more is True

    def test_fetch_page_in_browser_last_page(self) -> None:
        """Partial batch should set has_more=False."""
        from scripts.scrape_nj_dep import _fetch_page_in_browser

        mock_page = MagicMock()
        mock_page.evaluate.return_value = {
            "@odata.count": 52,
            "value": [{"id": 1}, {"id": 2}],
        }

        url = "/sdwis/SamplesSearchResults?$skip=50&$top=100&$count=true"
        records, _count, has_more = _fetch_page_in_browser(mock_page, "xsrf123", url)

        assert len(records) == 2
        assert has_more is False

    def test_fetch_page_in_browser_empty(self) -> None:
        """Empty response should return no records and has_more=False."""
        from scripts.scrape_nj_dep import _fetch_page_in_browser

        mock_page = MagicMock()
        mock_page.evaluate.return_value = {"@odata.count": 0, "value": []}

        url = "/sdwis/SamplesSearchResults?$skip=0&$top=100&$count=true"
        records, _count, has_more = _fetch_page_in_browser(mock_page, "xsrf123", url)

        assert len(records) == 0
        assert has_more is False

    def test_fetch_page_in_browser_retry_on_503(self) -> None:
        """Should retry on HTTP 503 error response."""
        from scripts.scrape_nj_dep import _fetch_page_in_browser

        mock_page = MagicMock()
        mock_page.evaluate.side_effect = [
            {"error": 503, "statusText": "Service Unavailable", "value": []},
            {"@odata.count": 1, "value": [{"id": 1}]},
        ]

        url = "/sdwis/SamplesSearchResults?$skip=0&$top=100&$count=true"
        records, _count, _has_more = _fetch_page_in_browser(
            mock_page, "xsrf123", url, max_retries=1
        )

        assert len(records) == 1
        assert mock_page.evaluate.call_count == 2

    def test_fetch_page_in_browser_http_error(self) -> None:
        """Non-retryable HTTP error should raise RuntimeError."""
        from scripts.scrape_nj_dep import _fetch_page_in_browser

        mock_page = MagicMock()
        mock_page.evaluate.return_value = {
            "error": 403,
            "statusText": "Forbidden",
            "value": [],
        }

        url = "/sdwis/SamplesSearchResults?$skip=0&$top=100&$count=true"
        with pytest.raises(RuntimeError, match="HTTP 403"):
            _fetch_page_in_browser(mock_page, "xsrf123", url)

    def test_checkpoint_save_load(self, tmp_path: Path) -> None:
        """Checkpoint should round-trip through JSON."""
        from scripts.scrape_nj_dep import load_checkpoint, save_checkpoint

        ckpt_path = tmp_path / ".nj_dep_scrape_checkpoint.json"

        # No checkpoint yet
        assert load_checkpoint(ckpt_path) == {}

        # Save and reload
        save_checkpoint(ckpt_path, offset=5000, records_so_far=4500, date_range_idx=1)
        loaded = load_checkpoint(ckpt_path)

        assert loaded["offset"] == 5000
        assert loaded["records_so_far"] == 4500
        assert loaded["date_range_idx"] == 1
        assert "timestamp" in loaded

    def test_checkpoint_load_corrupt(self, tmp_path: Path) -> None:
        """Corrupt checkpoint file should return empty dict."""
        from scripts.scrape_nj_dep import load_checkpoint

        ckpt_path = tmp_path / ".nj_dep_scrape_checkpoint.json"
        ckpt_path.write_text("not valid json {{{")

        assert load_checkpoint(ckpt_path) == {}

    def test_records_to_csv(self, tmp_path: Path) -> None:
        """Records should be written to CSV with correct columns."""
        from scripts.scrape_nj_dep import records_to_csv

        records = [
            {
                "PWS_ID": "NJ0101001   ",
                "ANALYTE_CODE": 2806,
                "ANALYTE_NAME": "PERFLUOROCTANOIC ACID (PFOA)",
                "RESULT": "14.0",
                "LESS_THAN_IND": "N",
                "NON_DETECT": None,
                "COLLECTION_DATE": "2023-01-15T00:00:00-05:00",
                "UOM": "UG/L     ",
            },
            {
                "PWS_ID": "NJ0200002   ",
                "ANALYTE_CODE": 2805,
                "ANALYTE_NAME": "PERFLUOROCTANE SULFONIC ACID (PFOS)",
                "RESULT": "<2",
                "LESS_THAN_IND": "Y",
                "NON_DETECT": "Y",
                "COLLECTION_DATE": "2023-02-20T00:00:00-05:00",
                "UOM": "UG/L     ",
            },
        ]

        out_path = tmp_path / "test_output.csv"
        records_to_csv(records, out_path)

        assert out_path.exists()
        df = pd.read_csv(out_path, dtype=str)
        assert len(df) == 2
        assert "PWS_ID" in df.columns
        assert "ANALYTE_CODE" in df.columns

    def test_output_csv_compatible_with_parse(
        self, nj_source: NjDepSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """CSV from scraper should be parseable by NjDepSource._parse_manual_export."""
        from scripts.scrape_nj_dep import records_to_csv

        # Simulate actual API response records
        records = [
            {
                "PWS_ID": "NJ0101001   ",
                "ANALYTE_CODE": "2806",
                "ANALYTE_NAME": "PERFLUOROCTANOIC ACID (PFOA)",
                "RESULT": "14.0",
                "LESS_THAN_IND": "N",
                "NON_DETECT": None,
                "COLLECTION_DATE": "2023-01-15T00:00:00-05:00",
                "UOM": "UG/L     ",
            },
        ]
        out_path = tmp_data_dirs["raw"] / "nj_dep_pfas_export.csv"
        records_to_csv(records, out_path)

        # Create empty facilities file
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "nj_dep_facilities.csv", index=False)

        # Parse should succeed
        df = nj_source.parse()
        assert len(df) == 1
        assert df.iloc[0]["analyte"] == "PFOA"
        assert df.iloc[0]["pwsid"] == "NJ0101001"
