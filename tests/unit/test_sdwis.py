"""Tests for SDWIS heavy metal data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import (
    HEAVY_METAL_ANALYTES,
    MGL_TO_UGL,
    SDWIS_CODE_TO_ANALYTE,
    SDWIS_CONTAMINANT_CODES,
)
from aquacontam.data.sdwis import SDWISSource


@pytest.fixture()
def sdwis_source(tmp_data_dirs: dict[str, Path]) -> SDWISSource:
    return SDWISSource(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


@pytest.fixture()
def sample_sdwis_files(tmp_data_dirs: dict[str, Path]) -> tuple[Path, Path]:
    """Create sample SDWIS LCR and system CSV files."""
    # LCR samples
    lcr = pd.DataFrame(
        {
            "PWSID": ["CA0101001", "CA0101001", "TX0200002", "TX0200002", "OH0300003"],
            "CONTAMINANT_CODE": ["PB90", "CU90", "PB90", "CU90", "PB90"],
            "SAMPLE_MEASURE": ["0.015", "1.3", "0.010", "0.005", "0.020"],
            "RESULT_SIGN_CODE": ["", "", "<", "", ""],
            "SAMPLE_COLLECTION_DATE": [
                "2023-01-15",
                "2023-01-15",
                "2023-02-20",
                "2023-03-10",
                "2023-04-01",
            ],
            "UNIT_OF_MEASURE": ["MG/L", "MG/L", "MG/L", "MG/L", "MG/L"],
        }
    )
    lcr_path = tmp_data_dirs["raw"] / "SDWA_LCR_SAMPLES.csv"
    lcr.to_csv(lcr_path, index=False)

    # System metadata
    systems = pd.DataFrame(
        {
            "PWSID": ["CA0101001", "TX0200002", "OH0300003"],
            "PWS_NAME": ["Test System CA", "Test System TX", "Test System OH"],
            "EPA_REGION": ["9", "6", "5"],
            "GW_SW_CODE": ["GW", "SW", "GW"],
            "POPULATION_SERVED_COUNT": ["10000", "50000", "5000"],
            "STATE_CODE": ["CA", "TX", "OH"],
            "GEO_LATITUDE": ["34.05", "30.27", "41.50"],
            "GEO_LONGITUDE": ["-118.24", "-97.74", "-81.69"],
        }
    )
    sys_path = tmp_data_dirs["raw"] / "SDWA_PUB_WATER_SYSTEMS.csv"
    systems.to_csv(sys_path, index=False)

    return lcr_path, sys_path


class TestSDWISSource:
    """Tests for SDWISSource."""

    def test_name_property(self, sdwis_source: SDWISSource) -> None:
        assert sdwis_source.name == "sdwis"

    def test_config_loaded(self, sdwis_source: SDWISSource) -> None:
        assert "url" in sdwis_source._config
        assert "expected_files" in sdwis_source._config
        assert len(sdwis_source._config["expected_files"]) == 2

    def test_parse_filters_target_codes(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        df = sdwis_source.parse()
        assert len(df) == 5  # All 5 rows have valid contaminant codes
        assert set(df["analyte"].unique()).issubset(set(HEAVY_METAL_ANALYTES))

    def test_contaminant_code_mapping(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        df = sdwis_source.parse()
        lead_rows = df[df["analyte"] == "lead"]
        assert len(lead_rows) == 3  # CA, TX, OH have lead samples

    def test_censored_from_sign_code(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        """RESULT_SIGN_CODE == '<' should mark as censored."""
        df = sdwis_source.parse()
        # TX lead sample has sign code "<"
        tx_lead = df[(df["pwsid"] == "TX0200002") & (df["analyte"] == "lead")]
        if len(tx_lead) > 0:
            assert bool(tx_lead.iloc[0]["censored"]) is True
            assert tx_lead.iloc[0]["concentration"] == 0.0

    def test_mgl_to_ugl_conversion(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        """mg/L values should be converted to ug/L."""
        df = sdwis_source.parse()
        # CA lead: 0.015 mg/L → 15 ug/L
        ca_lead = df[(df["pwsid"] == "CA0101001") & (df["analyte"] == "lead")]
        assert len(ca_lead) == 1
        assert abs(ca_lead.iloc[0]["concentration"] - 0.015 * MGL_TO_UGL) < 0.1

    def test_system_metadata_joined(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        """System metadata should be joined onto samples."""
        df = sdwis_source.parse()
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        # CA system should have coordinates
        ca_rows = df[df["pwsid"] == "CA0101001"]
        assert ca_rows["latitude"].notna().all()

    def test_coord_quality_flag(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        """SDWIS coordinates should be flagged as low quality."""
        df = sdwis_source.parse()
        assert "raw_coord_quality" in df.columns
        assert (df["raw_coord_quality"] == "low").all()

    def test_validate_checks_analytes(
        self, sdwis_source: SDWISSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["unknown_metal"],
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
            sdwis_source.validate(df)
        assert "Unexpected analytes in sdwis data" in caplog.text

    def test_validate_warns_duplicates(
        self, sdwis_source: SDWISSource, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001", "CA0101001"],
                "analyte": ["lead", "lead"],
                "concentration": [15.0, 20.0],
                "unit": ["ug/L", "ug/L"],
                "censored": [False, False],
                "detection_limit": [1.0, 1.0],
                "sample_date": pd.to_datetime(["2023-01-15", "2023-01-15"]),
                "latitude": [float("nan"), float("nan")],
                "longitude": [float("nan"), float("nan")],
            }
        )
        with caplog.at_level("WARNING"):
            sdwis_source.validate(df)
        assert "duplicate samples in sdwis" in caplog.text

    def test_to_parquet(
        self, sdwis_source: SDWISSource, sample_sdwis_files: tuple[Path, Path]
    ) -> None:
        df = sdwis_source.parse()
        path = sdwis_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == len(df)


class TestZeroHandling:
    """zero_handling parameter: keep (default) vs drop SAMPLE_MEASURE<=0.

    EPA LCR reports non-detect / below-detection 90th-percentile results as a
    literal 0. ``keep`` (default) retains them as non-detect non-exceedances;
    ``drop`` excludes them (reproduces the originally-frozen canonical parse).
    """

    @staticmethod
    def _write(tmp_data_dirs: dict[str, Path]) -> None:
        lcr = pd.DataFrame(
            {
                "PWSID": ["CA0101001", "CA0101001", "TX0200002"],
                "CONTAMINANT_CODE": ["PB90", "PB90", "PB90"],
                "SAMPLE_MEASURE": ["0.015", "0", "0.020"],  # middle row is a non-detect 0
                "RESULT_SIGN_CODE": ["", "", ""],
                "SAMPLE_COLLECTION_DATE": ["2023-01-15", "2023-01-16", "2023-02-20"],
                "UNIT_OF_MEASURE": ["MG/L", "MG/L", "MG/L"],
            }
        )
        lcr.to_csv(tmp_data_dirs["raw"] / "SDWA_LCR_SAMPLES.csv", index=False)
        systems = pd.DataFrame(
            {
                "PWSID": ["CA0101001", "TX0200002"],
                "PWS_NAME": ["Test System CA", "Test System TX"],
                "EPA_REGION": ["9", "6"],
                "GW_SW_CODE": ["GW", "SW"],
                "POPULATION_SERVED_COUNT": ["10000", "50000"],
                "STATE_CODE": ["CA", "TX"],
                "GEO_LATITUDE": ["34.05", "30.27"],
                "GEO_LONGITUDE": ["-118.24", "-97.74"],
            }
        )
        systems.to_csv(tmp_data_dirs["raw"] / "SDWA_PUB_WATER_SYSTEMS.csv", index=False)

    def test_default_keeps_zeros(
        self, sdwis_source: SDWISSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        self._write(tmp_data_dirs)
        df = sdwis_source.parse()  # default == "keep" -> no behavior change
        assert len(df) == 3
        assert int((df["concentration"] == 0.0).sum()) == 1

    def test_drop_removes_zeros(
        self, sdwis_source: SDWISSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        self._write(tmp_data_dirs)
        df = sdwis_source.parse(zero_handling="drop")
        assert len(df) == 2
        assert int((df["concentration"] == 0.0).sum()) == 0

    def test_invalid_zero_handling_raises(
        self, sdwis_source: SDWISSource, tmp_data_dirs: dict[str, Path]
    ) -> None:
        self._write(tmp_data_dirs)
        with pytest.raises(ValueError, match="zero_handling"):
            sdwis_source.parse(zero_handling="bogus")


class TestConstants:
    """Verify SDWIS constant consistency."""

    def test_code_to_analyte_round_trip(self) -> None:
        for analyte, code in SDWIS_CONTAMINANT_CODES.items():
            assert SDWIS_CODE_TO_ANALYTE[code] == analyte

    def test_heavy_metals_count(self) -> None:
        assert len(HEAVY_METAL_ANALYTES) == 2
