"""Tests for UCMR5 data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import UCMR5_ANALYTES
from aquacontam.data.ucmr5 import UCMR5Source


@pytest.fixture()
def ucmr5_source(tmp_data_dirs: dict[str, Path]) -> UCMR5Source:
    return UCMR5Source(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


class TestUCMR5Source:
    """Tests for the UCMR5Source data loader."""

    def test_name_property(self, ucmr5_source: UCMR5Source) -> None:
        assert ucmr5_source.name == "ucmr5"

    def test_config_loaded(self, ucmr5_source: UCMR5Source) -> None:
        assert "url" in ucmr5_source._config
        assert "column_map" in ucmr5_source._config
        assert "expected_files" in ucmr5_source._config

    def test_validate_passes_valid_df(self, ucmr5_source: UCMR5Source) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001", "TX0200002"],
                "analyte": ["PFOS", "PFOA"],
                "concentration": [12.5, 0.0],
                "unit": ["ug/L", "ug/L"],
                "censored": [False, True],
                "detection_limit": [2.0, 4.0],
                "sample_date": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "latitude": [float("nan"), float("nan")],
                "longitude": [float("nan"), float("nan")],
            }
        )
        result = ucmr5_source.validate(df)
        assert len(result) == 2

    def test_censored_flag_respected(self, ucmr5_source: UCMR5Source) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["PFOS"],
                "concentration": [0.0],
                "unit": ["ug/L"],
                "censored": [True],
                "detection_limit": [2.0],
                "sample_date": pd.to_datetime(["2023-01-15"]),
                "latitude": [float("nan")],
                "longitude": [float("nan")],
            }
        )
        result = ucmr5_source.validate(df)
        assert bool(result["censored"].iloc[0]) is True

    def test_unexpected_analyte_warning(
        self, ucmr5_source: UCMR5Source, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["UNKNOWN_CHEMICAL"],
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
            ucmr5_source.validate(df)
        assert "Unexpected analytes" in caplog.text

    def test_ucmr5_analytes_count(self) -> None:
        """UCMR5 should have 30 analytes (29 PFAS + lithium)."""
        assert len(UCMR5_ANALYTES) == 30

    def test_to_parquet(self, ucmr5_source: UCMR5Source) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["PFOS"],
                "concentration": [12.5],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [2.0],
                "sample_date": pd.to_datetime(["2023-01-15"]),
                "latitude": [float("nan")],
                "longitude": [float("nan")],
            }
        )
        path = ucmr5_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == 1
