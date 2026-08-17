"""Tests for UCMR3 data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aquacontam._constants import UCMR3_ANALYTES
from aquacontam.data.ucmr3 import UCMR3Source


@pytest.fixture()
def ucmr3_source(tmp_data_dirs: dict[str, Path]) -> UCMR3Source:
    return UCMR3Source(
        raw_dir=tmp_data_dirs["raw"],
        interim_dir=tmp_data_dirs["interim"],
        processed_dir=tmp_data_dirs["processed"],
    )


class TestUCMR3Source:
    """Tests for the UCMR3Source data loader."""

    def test_name_property(self, ucmr3_source: UCMR3Source) -> None:
        assert ucmr3_source.name == "ucmr3"

    def test_config_loaded(self, ucmr3_source: UCMR3Source) -> None:
        assert "url" in ucmr3_source._config
        assert "column_map" in ucmr3_source._config

    def test_ucmr3_analytes_count(self) -> None:
        """UCMR3 should have exactly 6 PFAS compounds."""
        assert len(UCMR3_ANALYTES) == 6

    def test_validate_passes_valid_df(self, ucmr3_source: UCMR3Source) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001", "TX0200002"],
                "analyte": ["PFOS", "PFOA"],
                "concentration": [5.0, 0.0],
                "unit": ["ug/L", "ug/L"],
                "censored": [False, True],
                "detection_limit": [1.0, 2.0],
                "sample_date": pd.to_datetime(["2014-06-01", "2014-07-15"]),
                "latitude": [float("nan"), float("nan")],
                "longitude": [float("nan"), float("nan")],
            }
        )
        result = ucmr3_source.validate(df)
        assert len(result) == 2

    def test_to_parquet(self, ucmr3_source: UCMR3Source) -> None:
        df = pd.DataFrame(
            {
                "pwsid": ["CA0101001"],
                "analyte": ["PFOS"],
                "concentration": [5.0],
                "unit": ["ug/L"],
                "censored": [False],
                "detection_limit": [1.0],
                "sample_date": pd.to_datetime(["2014-06-01"]),
                "latitude": [float("nan")],
                "longitude": [float("nan")],
            }
        )
        path = ucmr3_source.to_parquet(df)
        assert path.exists()
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == 1
