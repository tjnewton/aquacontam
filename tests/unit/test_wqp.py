"""Tests for Water Quality Portal data source."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aquacontam.data.wqp import (
    _CHARACTERISTIC_TO_ANALYTE,
    _WQP_PFAS_CHARACTERISTICS,
    WqpSource,
    _site_id_to_pwsid,
)


@pytest.fixture()
def wqp_source(tmp_data_dirs: dict[str, Path]) -> WqpSource:
    return WqpSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_wqp_csvs(tmp_data_dirs: dict[str, Path]) -> tuple[Path, Path]:
    """Create sample WQP results and station CSV files."""
    results = pd.DataFrame(
        {
            "MonitoringLocationIdentifier": [
                "USGS-12345678",
                "USGS-12345678",
                "21TXSWQ-99001",
            ],
            "CharacteristicName": [
                "Perfluorooctane sulfonic acid",
                "Perfluorooctanoic acid",
                "Perfluorobutanesulfonic acid",
            ],
            "ResultMeasureValue": ["0.015", "", "0.008"],
            "ResultMeasure/MeasureUnitCode": ["ug/l", "ug/l", "ug/l"],
            "ResultDetectionConditionText": ["", "Not Detected", ""],
            "DetectionQuantitationLimitMeasure/MeasureValue": ["0.004", "0.004", "0.002"],
            "DetectionQuantitationLimitMeasure/MeasureUnitCode": ["ug/l", "ug/l", "ug/l"],
            "ActivityStartDate": ["2023-03-15", "2023-03-15", "2023-04-20"],
        }
    )
    stations = pd.DataFrame(
        {
            "MonitoringLocationIdentifier": ["USGS-12345678", "21TXSWQ-99001"],
            "LatitudeMeasure": ["30.2672", "32.7767"],
            "LongitudeMeasure": ["-97.7431", "-96.7970"],
        }
    )
    results_path = tmp_data_dirs["raw"] / "wqp_pfas_results.csv"
    stations_path = tmp_data_dirs["raw"] / "wqp_pfas_stations.csv"
    results.to_csv(results_path, index=False)
    stations.to_csv(stations_path, index=False)
    return results_path, stations_path


class TestWqpSource:
    """Tests for WqpSource."""

    def test_name_property(self, wqp_source: WqpSource) -> None:
        assert wqp_source.name == "wqp"

    def test_config_loaded(self, wqp_source: WqpSource) -> None:
        assert "url" in wqp_source._config
        assert "station_url" in wqp_source._config
        assert "expected_files" in wqp_source._config

    def test_parse_column_mapping(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        df = wqp_source.parse()
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

    def test_analyte_mapping(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        """WQP characteristic names should be mapped to standard abbreviations."""
        df = wqp_source.parse()
        analytes = set(df["analyte"].unique())
        assert "PFOS" in analytes
        assert "PFOA" in analytes
        assert "PFBS" in analytes

    def test_censoring_not_detected(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        """'Not Detected' should mark row as censored."""
        df = wqp_source.parse()
        pfoa_rows = df[df["analyte"] == "PFOA"]
        assert len(pfoa_rows) == 1
        assert bool(pfoa_rows.iloc[0]["censored"]) is True
        assert pfoa_rows.iloc[0]["concentration"] == 0.0

    def test_detected_not_censored(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        df = wqp_source.parse()
        pfos_rows = df[df["analyte"] == "PFOS"]
        assert len(pfos_rows) == 1
        assert bool(pfos_rows.iloc[0]["censored"]) is False
        assert pfos_rows.iloc[0]["concentration"] > 0

    def test_detection_limit_parsed(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        df = wqp_source.parse()
        assert df["detection_limit"].notna().all()
        pfos_row = df[df["analyte"] == "PFOS"].iloc[0]
        assert abs(pfos_row["detection_limit"] - 0.004) < 1e-9

    def test_coordinates_from_stations(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        """Coordinates should be looked up from station file."""
        df = wqp_source.parse()
        assert df["latitude"].notna().all()
        assert df["longitude"].notna().all()

    def test_sample_date_parsed(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        df = wqp_source.parse()
        assert df["sample_date"].notna().all()
        assert pd.api.types.is_datetime64_any_dtype(df["sample_date"])

    def test_synthetic_pwsid_prefix(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        """Non-standard site IDs should get synthetic WQP_ prefix."""
        df = wqp_source.parse()
        # All our test site IDs are non-standard
        assert all(p.startswith("WQP_") for p in df["pwsid"])

    def test_validate_keeps_synthetic(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        """Validation should keep WQP_ synthetic IDs (PWSID check skipped)."""
        df = wqp_source.parse()
        validated = wqp_source.validate(df)
        # WQP skips PWSID validation, so synthetic IDs are preserved
        assert len(validated) > 0
        assert all(validated["pwsid"].str.startswith("WQP_"))

    def test_empty_results(self, wqp_source: WqpSource, tmp_data_dirs: dict[str, Path]) -> None:
        """Empty result file should return empty DataFrame."""
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "wqp_pfas_results.csv", index=False)
        pd.DataFrame().to_csv(tmp_data_dirs["raw"] / "wqp_pfas_stations.csv", index=False)
        df = wqp_source.parse()
        assert len(df) == 0

    def test_download_skip_existing(
        self, wqp_source: WqpSource, sample_wqp_csvs: tuple[Path, Path]
    ) -> None:
        """Download should skip if files already exist."""
        result = wqp_source.download(force=False)
        assert len(result) == 2
        assert all(r.exists() for r in result)


class TestSiteIdToPwsid:
    """Tests for _site_id_to_pwsid helper."""

    def test_empty_site_id(self) -> None:
        assert _site_id_to_pwsid("") == "WQP_0000000"

    def test_valid_pwsid_passthrough(self) -> None:
        """A site ID that looks like a PWSID should be extracted."""
        result = _site_id_to_pwsid("TX1234567")
        assert result == "TX1234567"

    def test_synthetic_id_for_non_standard(self) -> None:
        """Non-standard IDs should get WQP_ prefix."""
        result = _site_id_to_pwsid("USGS-12345678")
        assert result.startswith("WQP_")

    def test_deterministic(self) -> None:
        """Same input should always produce same output."""
        a = _site_id_to_pwsid("USGS-99999999")
        b = _site_id_to_pwsid("USGS-99999999")
        assert a == b

    def test_different_inputs_differ(self) -> None:
        """Different inputs should generally produce different IDs."""
        a = _site_id_to_pwsid("USGS-11111111")
        b = _site_id_to_pwsid("USGS-22222222")
        assert a != b

    def test_deterministic_known_value(self) -> None:
        """Hash-based IDs must be stable across Python runs (no PYTHONHASHSEED dependence)."""
        # Pre-computed via hashlib.md5("USGS-12345678".encode()).hexdigest()
        result = _site_id_to_pwsid("USGS-12345678")
        assert result.startswith("WQP_")
        # Verify the exact value is deterministic (md5-based, not Python hash())
        import hashlib

        expected_num = int(hashlib.md5(b"USGS-12345678").hexdigest(), 16) % 9999999
        assert result == f"WQP_{expected_num:07d}"


class TestCharacteristicConsistency:
    """Verify internal consistency of WQP characteristic name mappings."""

    def test_characteristics_match_analyte_keys(self) -> None:
        """_WQP_PFAS_CHARACTERISTICS and _CHARACTERISTIC_TO_ANALYTE must have the same names."""
        char_set = set(_WQP_PFAS_CHARACTERISTICS)
        analyte_keys = set(_CHARACTERISTIC_TO_ANALYTE.keys())
        assert char_set == analyte_keys, (
            f"Mismatch — in characteristics only: {char_set - analyte_keys}, "
            f"in analyte map only: {analyte_keys - char_set}"
        )

    def test_no_duplicate_characteristics(self) -> None:
        """Each characteristic name should appear exactly once."""
        assert len(_WQP_PFAS_CHARACTERISTICS) == len(set(_WQP_PFAS_CHARACTERISTICS))

    def test_no_duplicate_analyte_abbreviations(self) -> None:
        """Each analyte abbreviation should map from exactly one characteristic."""
        values = list(_CHARACTERISTIC_TO_ANALYTE.values())
        assert len(values) == len(set(values))


def _make_wqp_zip(df: pd.DataFrame) -> bytes:
    """Create a minimal WQP-style ZIP response containing a single CSV."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        csv_bytes = df.to_csv(index=False).encode()
        zf.writestr("result.csv", csv_bytes)
    return buf.getvalue()


class TestWqpMultiState:
    """Tests for multi-state WQP download robustness (mocked, no network)."""

    def test_failed_state_continues(
        self, wqp_source: WqpSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """If one state fails, the other state's data should still be saved."""
        import requests as req

        wqp_source._config["states"] = ["US:48", "US:29"]
        wqp_source._config["request_delay_seconds"] = 0

        # Remove any pre-existing files
        for f in wqp_source._config["expected_files"]:
            dest = tmp_data_dirs["raw"] / f
            if dest.exists():
                dest.unlink()

        good_results = pd.DataFrame(
            {
                "MonitoringLocationIdentifier": ["USGS-001"],
                "CharacteristicName": ["Perfluorooctanoic acid"],
                "ResultMeasureValue": ["0.01"],
            }
        )
        good_stations = pd.DataFrame(
            {
                "MonitoringLocationIdentifier": ["USGS-001"],
                "LatitudeMeasure": ["30.0"],
                "LongitudeMeasure": ["-97.0"],
            }
        )

        def mock_get(url: str, **kwargs: object) -> MagicMock:
            # Inspect the statecode param to decide pass/fail
            params = kwargs.get("params", [])
            # params is a list of tuples (key, value)
            state = next((v for k, v in params if k == "statecode"), "")
            if state == "US:48":
                raise req.ConnectionError("mocked connection error")
            # US:29 → succeed
            resp = MagicMock()
            resp.status_code = 200
            if "Result" in url:
                resp.content = _make_wqp_zip(good_results)
            else:
                resp.content = _make_wqp_zip(good_stations)
            resp.raise_for_status = MagicMock()
            return resp

        with patch("aquacontam.data.wqp.requests.get", side_effect=mock_get):
            files = wqp_source.download(force=True)

        assert len(files) == 2
        # Results file should contain data from the successful state
        results_path = files[0]
        assert results_path.exists()
        df = pd.read_csv(results_path)
        assert len(df) >= 1

    def test_rate_limit_delay(self, wqp_source: WqpSource, tmp_data_dirs: dict[str, Path]) -> None:
        """time.sleep should be called between state queries."""
        wqp_source._config["states"] = ["US:48", "US:29"]
        wqp_source._config["request_delay_seconds"] = 2

        for f in wqp_source._config["expected_files"]:
            dest = tmp_data_dirs["raw"] / f
            if dest.exists():
                dest.unlink()

        empty_zip = _make_wqp_zip(pd.DataFrame({"col": []}))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = empty_zip
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("aquacontam.data.wqp.requests.get", return_value=mock_resp),
            patch("aquacontam.data.wqp.time.sleep") as mock_sleep,
        ):
            wqp_source.download(force=True)

        # Sleep should be called between the two states (not after the last)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2)

    def test_empty_state_response(
        self, wqp_source: WqpSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        """Empty ZIP response should be handled gracefully."""
        wqp_source._config["states"] = ["US:02"]
        wqp_source._config["request_delay_seconds"] = 0

        for f in wqp_source._config["expected_files"]:
            dest = tmp_data_dirs["raw"] / f
            if dest.exists():
                dest.unlink()

        # ZIP with empty CSV (no data rows)
        empty_df = pd.DataFrame(columns=["MonitoringLocationIdentifier"])
        empty_zip = _make_wqp_zip(empty_df)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = empty_zip
        mock_resp.raise_for_status = MagicMock()

        with patch("aquacontam.data.wqp.requests.get", return_value=mock_resp):
            files = wqp_source.download(force=True)

        assert len(files) == 2
        assert all(f.exists() for f in files)

    def test_all_conus_states_in_config(self) -> None:
        """Config should list all 49 CONUS codes (48 states + DC)."""
        from aquacontam._config import load_data_config

        cfg = load_data_config(source_name="wqp")
        states = cfg.get("states", [])
        assert len(states) >= 49, f"Expected >= 49 states, got {len(states)}"

        # Verify some known codes are present
        codes = set(states)
        assert "US:48" in codes  # Texas
        assert "US:06" in codes  # California
        assert "US:11" in codes  # DC
        assert "US:36" in codes  # New York
